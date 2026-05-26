"""TrajectoryRewardManager: Aggregates per-turn dense CMI reward into reward_tensor.

Reads pre-computed per-turn rewards from AgentLoopOutput.extra_fields
(stored in DataProto.non_tensor_batch by verl's agent_loop._postprocess)
and writes them to the correct positions in the reward tensor.

Optionally computes a **final QA reward** at the end of each trajectory:
the model's final memory state is used to answer dataset QA questions via
retrieve → generate → LLM-judge, and the accuracy score is added to the
last turn's reward position. Controlled by environment variables:
  MEMBUILDER_REWARD_MODE=cmi_plus_qa   (enable final QA, blended with CMI)
  MEMBUILDER_REWARD_MODE=qa            (pure final QA reward, no CMI mixing;
                                        per-turn CMI is also skipped in
                                        agent_loop.py for efficiency)
  CMI_WEIGHT=0.5                        (alpha: traj_reward = α*CMI + (1-α)*QA;
                                         ignored when mode=qa)

Reward placement:
- Each turn's CMI reward → written at turn_boundaries[t] (inclusive end token)
- Final QA bonus (optional) → added at the last turn's end token

GRPO group normalization is handled by verl's advantage estimator
(norm_adv_by_std_in_grpo: true), not by this manager.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from verl.protocol import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _to_list(value: Any) -> list:
    """Normalize numpy object arrays / lists / tuples to plain python list."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def _to_dict(value: Any) -> dict:
    """Normalize numpy object / JSON string / dict to plain python dict."""
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(value, dict):
        return value
    return {}

# ---------------------------------------------------------------------------
# Final QA reward computation (runs in reward phase, not rollout phase)
# ---------------------------------------------------------------------------

# Lazily cached clients — created once per worker process.
_qa_clients_cache: Dict[str, Any] = {}

def _get_qa_config() -> Dict[str, Any]:
    """Read QA reward configuration from environment variables.

    Returns dict with 'enabled' set when MEMBUILDER_REWARD_MODE is either:
      - 'cmi_plus_qa': QA blended with CMI via cmi_weight (alpha)
      - 'qa'         : pure QA reward, no CMI blending (CMI is also skipped
                       per-turn in agent_loop.py)
    The 'mode' field is exposed so the reward manager can choose between
    blending and pure-replacement at the last turn position.
    """
    mode = os.environ.get("MEMBUILDER_REWARD_MODE", "").strip().lower()
    return {
        "enabled": mode in ("cmi_plus_qa", "qa"),
        "mode": mode,
        "cmi_weight": float(os.environ.get("CMI_WEIGHT", "0.5")),
        "api_key": os.environ.get("LLM_API_KEY", ""),
        "base_url": os.environ.get("LLM_BASE_URL", os.environ.get("OPENAI_BASE_URL", "")),
        "answer_model": os.environ.get("MEMBUILDER_INPROC_ANSWER_MODEL", "deepseek-v4-pro"),
        "judge_model": os.environ.get("MEMBUILDER_INPROC_JUDGE_MODEL", "deepseek-v4-flash"),
        "embedding_model": os.environ.get("MEMBUILDER_INPROC_EMBEDDING_MODEL", "text-embedding-v4"),
        "max_qa": int(os.environ.get("MEMBUILDER_INPROC_QA_PER_SESSION", "5")),
    }


def _get_llm_client(config: Dict[str, Any], role: str):
    """Get or create a cached LLM client for answer/judge."""
    if role in _qa_clients_cache:
        return _qa_clients_cache[role]

    try:
        from llm_client import create_llm_client
    except ImportError:
        from llm_client_internal import create_llm_client

    model = config["answer_model"] if role == "answer" else config["judge_model"]
    client = create_llm_client(provider="openai", model=model)
    _qa_clients_cache[role] = client
    return client


def _compute_final_qa_reward(
    final_memory: Dict[str, Any],
    questions: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> float:
    """Compute QA accuracy using final memory state.

    Builds a lightweight MemorySystem, loads the final memory snapshot,
    then for each QA question: retrieve → generate answer → LLM judge.

    Returns accuracy in [0, 1].
    """
    if not questions:
        return 0.0

    from training.reward_server.cmi_reward import get_cluster_embedding_func
    from memory_system import MemorySystem
    from eval.llm_judge import evaluate_answer

    answer_client = _get_llm_client(config, "answer")
    judge_client = _get_llm_client(config, "judge")

    embedding_func = get_cluster_embedding_func()
    memory = MemorySystem(llm_client=answer_client, embedding_func=embedding_func)

    # Populate memory from the snapshot (plain text → vector store)
    core_text = final_memory.get("core_memory", "")
    if core_text:
        memory.core_memory.human = core_text

    dummy_user_id = "qa_eval"
    searchable_memories: List[str] = []
    for mem_type in ("episodic", "semantic", "procedural"):
        prefix = f"[{mem_type.upper()}]"
        for item in final_memory.get(mem_type, []):
            text = str(item).strip()
            if text:
                tagged = text if text.startswith(prefix) else f"{prefix} {text}"
                searchable_memories.append(tagged)

    if searchable_memories:
        metadatas = [{"user_id": dummy_user_id, "type": "memory"} for _ in searchable_memories]
        memory.vector_store.add(searchable_memories, metadatas)

    # Evaluate QA
    qa_list = list(questions)[: config["max_qa"]]
    correct = 0
    judge_model = config["judge_model"]

    try:
        from config import QA_ANSWERING_TOP_K
    except Exception:
        QA_ANSWERING_TOP_K = 10

    for qa in qa_list:
        question = qa.get("question", "") if isinstance(qa, dict) else str(qa)
        gold = qa.get("answer", "") if isinstance(qa, dict) else ""
        if not question or not gold:
            continue

        try:
            generated = memory.generate_answer(question, user_id=dummy_user_id)
        except Exception:
            generated = ""

        try:
            score = evaluate_answer(
                question, gold, generated,
                llm_client=judge_client, judge_model=judge_model,
            )
            correct += int(score or 0)
        except Exception:
            pass

    return correct / len(qa_list) if qa_list else 0.0


@register("trajectory")
class TrajectoryRewardManager(AbstractRewardManager):
    """Aggregate per-turn dense CMI rewards into reward_tensor.

    When MEMBUILDER_REWARD_MODE=cmi_plus_qa, additionally computes final
    QA reward from the trajectory's end-state memory and blends it:
        last_turn_reward = α * cmi_sum + (1-α) * qa_accuracy
    """

    def __init__(
        self,
        tokenizer: Any = None,
        num_examine: int = 0,
        compute_score: Optional[RawRewardFn] = None,
        reward_fn_key: str = "data_source",
        *,
        turn_reward_weight: float = 1.0,
        format_penalty: float = -0.5,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.turn_reward_weight = float(turn_reward_weight)
        self.format_penalty = float(format_penalty)

    def __call__(
        self, data: DataProto, return_dict: bool = False, **kwargs: Any
    ):
        responses: torch.Tensor = data.batch["responses"]
        batch_size, resp_len = responses.shape
        reward_tensor = torch.zeros((batch_size, resp_len), dtype=torch.float32)

        non_tensor = data.non_tensor_batch
        turn_rewards_batch = non_tensor.get("turn_rewards")
        turn_boundaries_batch = non_tensor.get("turn_boundaries")
        format_valid_batch = non_tensor.get("format_valid_per_turn")
        n_action_batch = non_tensor.get("n_action_per_turn")
        final_memory_batch = non_tensor.get("final_memory_snapshot")
        questions_batch = non_tensor.get("questions")

        qa_config = _get_qa_config()
        qa_enabled = qa_config["enabled"]
        qa_mode = qa_config.get("mode", "")
        cmi_weight = qa_config["cmi_weight"]

        if qa_enabled:
            if qa_mode == "qa":
                logger.info(
                    "[TrajectoryRM] Pure QA reward enabled (no CMI blending; "
                    "per-turn CMI assumed skipped in agent_loop)"
                )
            else:
                logger.info(
                    f"[TrajectoryRM] Final QA reward enabled: α={cmi_weight:.2f} "
                    f"(reward = α*CMI + (1-α)*QA)"
                )

        reward_extra_info: Dict[str, list] = defaultdict(list)

        for i in range(batch_size):
            turn_rewards = _to_list(
                turn_rewards_batch[i] if turn_rewards_batch is not None else None
            )
            turn_boundaries = _to_list(
                turn_boundaries_batch[i] if turn_boundaries_batch is not None else None
            )
            format_valid = _to_list(
                format_valid_batch[i] if format_valid_batch is not None else None
            )
            n_actions = _to_list(
                n_action_batch[i] if n_action_batch is not None else None
            )

            # ---- Per-turn CMI rewards ----
            traj_reward_sum = 0.0
            seen_positions: set = set()
            num_turns = min(len(turn_boundaries), len(turn_rewards))

            for t in range(num_turns):
                end_idx = int(turn_boundaries[t])
                if end_idx < 0 or end_idx >= resp_len:
                    continue
                if end_idx in seen_positions:
                    continue
                seen_positions.add(end_idx)
                contrib = self.turn_reward_weight * float(turn_rewards[t])
                reward_tensor[i, end_idx] += contrib
                traj_reward_sum += contrib

            # ---- Final QA reward (optional) ----
            qa_accuracy = 0.0
            if qa_enabled and num_turns > 0:
                final_memory = _to_dict(
                    final_memory_batch[i] if final_memory_batch is not None else None
                )
                questions = _to_list(
                    questions_batch[i] if questions_batch is not None else None
                )

                if final_memory and questions:
                    try:
                        qa_accuracy = _compute_final_qa_reward(
                            final_memory, questions, qa_config,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"[TrajectoryRM] QA reward failed for sample {i}: {exc}"
                        )
                        qa_accuracy = 0.0

                    # Blend: replace the trajectory reward at the last turn
                    # position with the weighted combination.
                    last_turn_idx = turn_boundaries[-1] if turn_boundaries else -1
                    if 0 <= last_turn_idx < resp_len:
                        if qa_mode == "qa":
                            # Pure QA mode: per-turn CMI was skipped (=0), so just add
                            # QA accuracy at the final position. Format penalties (if
                            # any) on prior turns are preserved untouched.
                            reward_tensor[i, last_turn_idx] += qa_accuracy
                        else:
                            # cmi_plus_qa: replace last-turn CMI with α*CMI + (1-α)*QA
                            # Original CMI contribution at this position
                            original_cmi_at_last = reward_tensor[i, last_turn_idx].item()
                            # Blended reward: α * total_cmi_mean + (1-α) * qa_accuracy
                            cmi_normalized = (
                                traj_reward_sum / num_turns if num_turns > 0 else 0.0
                            )
                            blended = cmi_weight * cmi_normalized + (1.0 - cmi_weight) * qa_accuracy
                            # Replace last turn: remove old CMI, add blended
                            reward_tensor[i, last_turn_idx] += blended - original_cmi_at_last

            # Per-sample metrics
            num_turns_total = len(turn_rewards)
            n_format_valid = sum(1 for v in format_valid if bool(v))
            reward_extra_info["traj_reward_sum"].append(float(traj_reward_sum))
            reward_extra_info["turn_reward_mean"].append(
                float(np.mean(turn_rewards)) if turn_rewards else 0.0
            )
            reward_extra_info["turn_reward_abs_mean"].append(
                float(np.mean(np.abs(turn_rewards))) if turn_rewards else 0.0
            )
            reward_extra_info["format_valid_ratio"].append(
                float(n_format_valid / num_turns_total) if num_turns_total > 0 else 0.0
            )
            reward_extra_info["n_action_per_turn_mean"].append(
                float(np.mean(n_actions)) if n_actions else 0.0
            )
            reward_extra_info["num_turns_executed"].append(float(num_turns_total))
            reward_extra_info["qa_accuracy"].append(float(qa_accuracy))

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor

__all__ = ["TrajectoryRewardManager"]
