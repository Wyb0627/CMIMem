"""Memory-R1 trajectory-level reward: apply flat actions -> answer QA with the
frozen answerer -> mean score over the QA pairs. Scoring mode is selectable via
``config["memory_r1_em_mode"]`` (env ``MEMORY_R1_EM_MODE``): ``em`` (strict
normalized Exact Match, default), ``subem`` (substring EM, denser signal), or
``judge`` (LLM-as-judge, same ``evaluate_answer`` pipeline used by the other
training modes). A single scalar per rollout.
"""
from typing import Any, Dict, List

from training.memory_r1.flat_memory import apply_memory_r1_actions, score_answer_em
from training.memory_r1.prompts import render_answer_prompt
from training.memory_r1.session_loader import load_flat_memory_before


def _answer_question(question: str, memories: List[str], config: Dict) -> str:
    """Answer via the frozen external model (qwen3.7-max) using the short-answer
    prompt. Mocked in tests. Never raises: returns '' on failure so the QA
    scores 0 rather than crashing the batch."""
    try:
        from training.reward_server.reward_function import _get_inproc_answer_client
        client = _get_inproc_answer_client(config)
        prompt = render_answer_prompt(memories, question)
        resp = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        return resp if isinstance(resp, str) else str(resp)
    except Exception as e:
        print(f"    [memory_r1] answer failed: {str(e)[:80]}")
        return ""


def _load_before_and_build_after(parsed_ops: List[Dict[str, Any]],
                                 ground_truth: Dict[str, Any],
                                 config: Dict[str, Any]) -> List[str]:
    """Load the prior flat bank and apply the manager's ops. Mocked in tests."""
    state_path = ground_truth.get("state_before_path", "") or ""
    before = load_flat_memory_before(state_path) if state_path else []
    return apply_memory_r1_actions(before, parsed_ops)


def _retrieve_per_qa(after: List[str], qa_list: List[Dict], top_k: int,
                     config: Dict) -> List[List[str]]:
    """Return, per QA, the top-k retrieved memory texts from the after-bank.

    Uses the shared flat-memory builder + the Nebula embedding service. On any
    failure, falls back to feeding a bounded slice of the after-bank so
    answering still proceeds. Mocked in tests.
    """
    if not after:
        return [[] for _ in qa_list]
    try:
        from training.reward_server.cmi_reward import get_nebula_embedding_func
        from training.memory_r1.memory_build import build_flat_memory_system
        embedding_func = get_nebula_embedding_func()
        memory = build_flat_memory_system(after, None, embedding_func)
        out = []
        for qa in qa_list:
            res = memory.search(qa.get("question", ""), user_id="memory_r1", top_k=top_k)
            hits = res.get("results", []) if isinstance(res, dict) else []
            out.append([h["memory"] for h in hits])
        return out
    except Exception as e:
        print(f"    [memory_r1] retrieval failed ({str(e)[:60]}); feeding bank head")
        capped = after[: top_k * 4]
        return [capped for _ in qa_list]


def compute_task_reward_memory_r1(
    parsed_ops: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    session_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply Memory-R1 flat actions, answer each QA with the frozen answerer,
    reward = mean normalized EM over the QA pairs (single trajectory scalar)."""
    after = _load_before_and_build_after(parsed_ops, ground_truth, config)

    qa_per = int(config.get("memory_r1_qa_per_session", 5))
    qa_list = list(ground_truth.get("qa_questions", []))[:qa_per]
    if not qa_list:
        return {"task_reward": 0.0, "dominant_agent": "memory_r1", "n_qa": 0,
                "n_mem_after": len(after)}

    top_k = int(config.get("memory_r1_top_k", 8))
    em_mode = str(config.get("memory_r1_em_mode", "em"))
    hits_memories = _retrieve_per_qa(after, qa_list, top_k, config)

    correct = 0.0
    if em_mode == "judge":
        # LLM-as-judge scoring, identical to _compute_task_reward_inproc in
        # training/reward_server/reward_function.py. Lazy imports to avoid
        # circular dependency when the reward module is imported without the
        # judge infra available.
        from eval.llm_judge import evaluate_answer
        from training.reward_server.reward_function import _get_inproc_judge_client

        judge_client = _get_inproc_judge_client(config)
        judge_model = config.get("inproc_judge_model")
        for qa, mems in zip(qa_list, hits_memories):
            pred = _answer_question(qa.get("question", ""), mems, config)
            try:
                score = evaluate_answer(
                    qa.get("question", ""),
                    qa.get("answer", ""),
                    pred,
                    llm_client=judge_client,
                    judge_model=judge_model,
                )
            except Exception as e:
                print(f"    [memory_r1] judge failed: {str(e)[:80]}")
                score = 0
            correct += int(score or 0)
    else:
        for qa, mems in zip(qa_list, hits_memories):
            pred = _answer_question(qa.get("question", ""), mems, config)
            correct += score_answer_em(pred, qa.get("answer", ""), mode=em_mode)

    return {
        "task_reward": correct / len(qa_list),
        "dominant_agent": "memory_r1",
        "n_qa": len(qa_list),
        "n_mem_after": len(after),
        "em_mode": em_mode,
    }
