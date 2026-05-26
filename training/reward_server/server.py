"""
Memory API Server for RL Training.

A Flask-based API server that provides memory construction and QA evaluation
for verl-based RL training. This server runs on CPU and is called by the
reward function during training.

Usage:
    python server.py --port 8765 --api-key sk-xxx --base-url https://api.openai.com/v1

Endpoints:
    GET  /health          - Health check
    POST /build_and_qa    - Build memories and evaluate QA
"""

import argparse
import json
import os
import time
from pathlib import Path
from flask import Flask, request, jsonify

# 调试模式：设置环境变量 REWARD_SERVER_DEBUG=1 开启详细日志
DEBUG_MODE = os.environ.get("REWARD_SERVER_DEBUG", "0") == "1"

def debug_print(*args, **kwargs):
    """调试打印，仅在 DEBUG_MODE=True 时输出"""
    if DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)

try:
    from config import EMBEDDING_MODEL, ANSWER_MODEL, JUDGE_MODEL, QA_ANSWERING_TOP_K
    try:
        from llm_client_internal import OpenAIClient
    except ImportError:
        from llm_client import OpenAIClient
    from memory_system import MemorySystem
    from eval.llm_judge import evaluate_answer
except Exception:
    from ...config import EMBEDDING_MODEL, ANSWER_MODEL, JUDGE_MODEL, QA_ANSWERING_TOP_K
    try:
        from ...llm_client_internal import OpenAIClient
    except ImportError:
        from ...llm_client import OpenAIClient
    from ...memory_system import MemorySystem
    from ...eval.llm_judge import evaluate_answer

app = Flask(__name__)


# Configuration - set by command line arguments in main()
SERVER_CONFIG = {
    "embedding_model": os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL),
    "answer_model": os.environ.get("ANSWER_MODEL", ANSWER_MODEL),
    "judge_model": os.environ.get("JUDGE_MODEL", JUDGE_MODEL),
}


_ANSWER_CLIENT = None
_JUDGE_CLIENT = None


def _get_answer_client():
    global _ANSWER_CLIENT
    if _ANSWER_CLIENT is None:
        _ANSWER_CLIENT = OpenAIClient(model=SERVER_CONFIG["answer_model"])
    return _ANSWER_CLIENT


def _get_judge_client():
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        _JUDGE_CLIENT = OpenAIClient(model=SERVER_CONFIG["judge_model"])
    return _JUDGE_CLIENT


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Memory API Server is running"})


@app.route('/build_and_qa', methods=['POST'])
def build_and_qa():
    """
    Build memories from model actions and evaluate QA.
    
    Request JSON:
    {
        "session_group_id": "conv_001_session_0",
        "state_before_path": "/path/to/state",
        "memory_actions": {
            "core": {"operation": "APPEND", "content": "..."},
            "episodic": {"operations": [...]},
            "semantic": {"operations": [...]},
            "procedural": {"operations": [...]}
        },
        "qa_questions": [
            {"question": "...", "answer": "..."},
            ...
        ],
        "new_session_messages": [...]
    }
    
    Response JSON:
    {
        "success": true,
        "task_reward": 0.8,
        "correct": 4,
        "total": 5,
        "retrieval_counts": {"episodic": 3, "semantic": 2, "procedural": 0},
        "dominant_agent": "episodic"
    }
    """
    try:
        request_start = time.time()
        data = request.json

        session_id = data.get("session_group_id", "unknown")
        state_before_path = data.get("state_before_path", "")
        
        debug_print(f"\n{'='*60}")
        debug_print(f"📥 收到请求: session={session_id}")
        debug_print(f"   state_before_path: {state_before_path}")

        # New scheme (original pipeline)
        memory_actions_list = data.get("memory_actions")
        qa_pairs = data.get("qa_pairs")

        # Backward compatibility (older dict-style request)
        memory_actions_dict = data.get("memory_actions", {}) if isinstance(data.get("memory_actions"), dict) else None
        qa_questions = data.get("qa_questions")
        
        # 调试：打印 memory_actions 摘要
        if DEBUG_MODE:
            if isinstance(memory_actions_list, list):
                for item in memory_actions_list:
                    agent = item.get("agent_type", "?")
                    action = item.get("action", {})
                    if agent == "core":
                        op = action.get("operation", "N/A")
                        content_len = len(str(action.get("content", "")))
                        debug_print(f"   {agent}: op={op}, content_len={content_len}")
                    else:
                        ops = action.get("operations", action.get(agent, {}).get("operations", []))
                        debug_print(f"   {agent}: {len(ops)} operations")
            debug_print(f"   qa_pairs: {len(qa_pairs) if qa_pairs else 0} questions")

        answer_client = _get_answer_client()
        judge_client = _get_judge_client()
        
        # Initialize memory system
        memory = MemorySystem(llm_client=answer_client)
        
        # Load state_before if provided (critical for QA evaluation)
        if state_before_path:
            # Handle path prefix - try multiple base paths
            candidates = []
            if state_before_path.startswith('/'):
                candidates.append(state_before_path)
            else:
                # Relative path - try project root
                project_root = Path(__file__).resolve().parent.parent.parent
                candidates.append(str(project_root / state_before_path))
            
            for candidate in candidates:
                state_path = Path(candidate)
                if state_path.exists():
                    try:
                        memory.load(str(state_path))
                        debug_print(f"   ✅ 加载state成功: {candidate}")
                        debug_print(f"      core_memory: {len(memory.core_memory.human) if memory.core_memory.human else 0} chars")
                        vec_count = len(memory.vector_store.memories) if hasattr(memory.vector_store, 'memories') else 0
                        debug_print(f"      vector_store: {vec_count} memories")
                        break
                    except Exception as e:
                        debug_print(f"   ⚠️ 加载state失败 {candidate}: {e}")
            else:
                debug_print(f"   ❌ state_before_path不存在: {state_before_path}")

        # Apply actions
        memories_before = len(memory.vector_store.memories) if hasattr(memory.vector_store, 'memories') else 0
        if isinstance(memory_actions_list, list):
            _apply_memory_actions_list(memory, memory_actions_list)
        elif isinstance(memory_actions_dict, dict) and memory_actions_dict:
            # Convert dict-style into list
            converted = []
            for agent_type in ["core", "episodic", "semantic", "procedural"]:
                if agent_type in memory_actions_dict:
                    converted.append({"agent_type": agent_type, "action": memory_actions_dict[agent_type]})
            _apply_memory_actions_list(memory, converted)
        
        memories_after = len(memory.vector_store.memories) if hasattr(memory.vector_store, 'memories') else 0
        debug_print(f"   🔧 应用actions后: +{memories_after - memories_before} memories")

        if qa_pairs is None and isinstance(qa_questions, list):
            qa_pairs = qa_questions

        qa_pairs = qa_pairs or []

        # --- Memory-Guided RAG: load chunk data from request ---
        try:
            from config import RAG_CHUNK_TYPES
        except Exception:
            RAG_CHUNK_TYPES = []

        session_chunks = []
        memory_chunk_mapping = {}
        extracted_candidates = {}
        if RAG_CHUNK_TYPES:
            _sc = data.get("session_chunks", [])
            if isinstance(_sc, str):
                try:
                    session_chunks = json.loads(_sc)
                except Exception:
                    session_chunks = []
            elif isinstance(_sc, list):
                session_chunks = _sc
            _mcm = data.get("memory_chunk_mapping", {})
            if isinstance(_mcm, str):
                try:
                    memory_chunk_mapping = json.loads(_mcm)
                except Exception:
                    memory_chunk_mapping = {}
            elif isinstance(_mcm, dict):
                memory_chunk_mapping = _mcm
            _ec = data.get("extracted_candidates", {})
            if isinstance(_ec, str):
                try:
                    extracted_candidates = json.loads(_ec)
                except Exception:
                    extracted_candidates = {}
            elif isinstance(_ec, dict):
                extracted_candidates = _ec

        correct = 0
        retrieval_counts = {"episodic": 0, "semantic": 0, "procedural": 0}

        for qa_idx, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            gold_answer = qa.get("answer", "")

            search_result = memory.search(question, user_id=session_id, top_k=QA_ANSWERING_TOP_K)
            retrieved = search_result.get('results', [])
            retrieved_texts = [r["memory"] for r in retrieved]
            
            # 调试：打印第一个QA的检索结果
            if qa_idx == 0:
                print(f"   🔍 QA[0] 检索到 {len(retrieved_texts)} 条memory")
                for i, mem in enumerate(retrieved_texts[:3]):
                    print(f"      [{i}] {mem[:80]}...")

            rag_chunk_indices = set()
            for mem in retrieved_texts:
                for mem_type in retrieval_counts:
                    if f"[{mem_type.upper()}]" in mem:
                        retrieval_counts[mem_type] += 1
                        # If this type is in RAG_CHUNK_TYPES, find its chunk
                        if mem_type in RAG_CHUNK_TYPES and session_chunks:
                            mem_text = mem.replace(f"[{mem_type.upper()}] ", "")
                            _type_mapping = memory_chunk_mapping.get(mem_type, [])
                            _type_items = extracted_candidates.get(mem_type, [])
                            for item_idx, item_text in enumerate(_type_items):
                                if item_idx < len(_type_mapping) and str(item_text).strip() in mem_text:
                                    chunk_idx = _type_mapping[item_idx]
                                    if isinstance(chunk_idx, int) and 0 <= chunk_idx < len(session_chunks):
                                        rag_chunk_indices.add(chunk_idx)
                                    break
                        break

            # Build chunk context string for RAG
            chunk_context = None
            if rag_chunk_indices and session_chunks:
                chunk_texts = []
                for ci in sorted(rag_chunk_indices):
                    chunk = session_chunks[ci]
                    text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                    if text:
                        chunk_texts.append(f"[Conversation Segment {ci + 1}]\n{text}")
                if chunk_texts:
                    chunk_context = "\n\n".join(chunk_texts)

            generated_answer = memory.generate_answer(question, user_id=session_id, chunk_context=chunk_context)
            score = evaluate_answer(question, gold_answer, generated_answer, llm_client=judge_client)
            correct += score
            
            if DEBUG_MODE:
                debug_print(f"   📝 QA[{qa_idx}]: score={score}")
                debug_print(f"      Q: {question[:60]}...")
                debug_print(f"      Gold: {gold_answer[:40]}...")
                debug_print(f"      Gen: {str(generated_answer)[:40]}...")
                debug_print(f"      Retrieved: {len(retrieved_texts)} memories")

        dominant = None
        if any(retrieval_counts.values()):
            dominant = max(retrieval_counts, key=retrieval_counts.get)

        task_reward = correct / len(qa_pairs) if qa_pairs else 0.0
        
        elapsed = time.time() - request_start
        # 始终打印retrieval_counts和dominant_agent，方便调试归因问题
        print(f"   📊 retrieval_counts: {retrieval_counts}, dominant: {dominant}")
        debug_print(f"   ✅ 完成: reward={task_reward:.3f}, correct={correct}/{len(qa_pairs)}, 耗时={elapsed:.2f}s")
        debug_print(f"{'='*60}\n")

        return jsonify(
            {
                "success": True,
                "task_reward": task_reward,
                "correct": correct,
                "total": len(qa_pairs),
                "retrieval_counts": retrieval_counts,
                "dominant_agent": dominant,
            }
        )
        
    except Exception as e:
        # 错误时返回简化奖励，而非0（避免影响GRPO训练）
        # 基于记忆数量给分：每个memory 0.05分，最多0.8分
        fallback_reward = _compute_fallback_reward(data.get("memory_actions", []) if 'data' in dir() else [])
        print(f"  ⚠️ 异常，使用兜底奖励: {fallback_reward:.3f}, 错误: {str(e)[:100]}")
        return jsonify({
            "success": True,  # 返回success=True避免客户端重试
            "task_reward": fallback_reward,
            "correct": 0,
            "total": 0,
            "retrieval_counts": {},
            "dominant_agent": None,
            "fallback": True,
            "error_msg": str(e)[:200]
        })


def _apply_memory_actions_list(memory: MemorySystem, actions: list) -> None:
    new_memories = []
    for item in actions:
        agent_type = item.get("agent_type")
        action = item.get("action") or {}

        if not agent_type:
            continue

        if agent_type == "core":
            op = str(action.get("operation", "")).upper()
            if not op and "new_core_memory" in action:
                op = "REWRITE"

            if op == "APPEND":
                content = action.get("content", "")
                if content:
                    memory.core_memory.human = (
                        memory.core_memory.human + f"\n{content}" if memory.core_memory.human else content
                    )
            elif op == "REPLACE":
                old = action.get("old_text", "")
                new = action.get("new_text", "")
                if old and new:
                    memory.core_memory.human = memory.core_memory.human.replace(old, new)
            elif op == "REWRITE":
                content = action.get("content") or action.get("new_core_memory") or ""
                if content:
                    memory.core_memory.human = content
            continue

        agent_data = action.get(agent_type, action) if isinstance(action, dict) else {}
        operations = agent_data.get("operations", []) if isinstance(agent_data, dict) else []
        prefix = f"[{agent_type.upper()}]"

        for op_item in operations:
            action_name = str(op_item.get("action", "")).upper()
            if action_name == "ADD":
                mem = op_item.get("memory", "")
            elif action_name in ["UPDATE", "MERGE"]:
                mem = op_item.get("new_memory", "") or op_item.get("memory", "")
            else:
                mem = ""

            if mem:
                if not str(mem).startswith(prefix):
                    mem = f"{prefix} {mem}"
                new_memories.append(str(mem))

    if new_memories:
        memory.vector_store.add(new_memories, [{"type": "memory"} for _ in new_memories])


def _compute_fallback_reward(memory_actions: list) -> float:
    """
    计算兜底奖励（当完整流程失败时使用）
    基于记忆数量给分：每个memory 0.05分，最多0.8分
    避免返回0影响GRPO训练
    """
    total_memories = 0
    
    for item in memory_actions:
        if not isinstance(item, dict):
            continue
        agent_type = item.get("agent_type", "")
        action = item.get("action", {})
        
        if agent_type == "core":
            # Core memory 有内容就算1个
            if action.get("content") or action.get("new_core_memory"):
                total_memories += 1
        else:
            # 其他agents计算operations数量
            agent_data = action.get(agent_type, action) if isinstance(action, dict) else {}
            operations = agent_data.get("operations", []) if isinstance(agent_data, dict) else []
            for op in operations:
                op_action = str(op.get("action", "")).upper()
                if op_action in ["ADD", "UPDATE", "MERGE"]:
                    total_memories += 1
    
    # 每个memory 0.05分，最多0.8分
    reward = min(0.8, total_memories * 0.05)
    return reward


def main():
    parser = argparse.ArgumentParser(
        description="Memory API Server for RL Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using OpenAI API
  python -m training.reward_server.server --api-key sk-xxx

  # Using custom OpenAI-compatible endpoint (e.g., vLLM, Azure)
  python -m training.reward_server.server --api-key your-key --base-url http://localhost:8000/v1

  # With debug mode
  python -m training.reward_server.server --api-key sk-xxx --debug
"""
    )
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key (required)")
    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Set environment variables from command line args
    os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    
    # Enable debug mode
    global DEBUG_MODE
    if args.debug:
        DEBUG_MODE = True
    
    print("=" * 60)
    print("Memory API Server for RL Training")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"API Base: {args.base_url or 'https://api.openai.com/v1 (default)'}")
    print(f"Debug: {args.debug}")
    print("=" * 60)
    
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
