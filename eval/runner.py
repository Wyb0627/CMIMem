#!/usr/bin/env python3
"""
Unified evaluation runner for MemBuilder.

Usage:
    # LOCOMO - 单个对话
    python -m eval.runner --dataset locomo --conv-id conv-26 --questions 10
    
    # LOCOMO - 全部对话
    python -m eval.runner --dataset locomo --mode full
    
    # 使用 vLLM
    python -m eval.runner --dataset locomo --provider vllm --base-url http://localhost:8000/v1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_system import MemorySystem

# Client module: internal version adds extra providers; public version has openai/vllm only.
try:
    from llm_client_internal import create_llm_client, AVAILABLE_PROVIDERS, DEFAULT_PROVIDER
except ImportError:
    from llm_client import create_llm_client, AVAILABLE_PROVIDERS, DEFAULT_PROVIDER

from config import (
    ANSWER_MODEL, JUDGE_MODEL, EMBEDDING_MODEL,
    QA_ANSWERING_TOP_K, OPENAI_API_KEY, OPENAI_BASE_URL,
    OPENAI_EMBEDDINGS_BASE_URL, OPENAI_EMBEDDINGS_API_KEY
)

from eval.llm_judge import evaluate_answer, LLMJudge
from eval.metrics import compute_accuracy, print_accuracy_stats
from eval.datasets import load_dataset, get_current_time


def _make_embedding_func():
    """Create a separate embedding function if a dedicated embedding endpoint is configured.
    
    Only activates when OPENAI_EMBEDDINGS_BASE_URL is explicitly set in env/config.
    Otherwise returns None — the llm_client handles embedding internally
    (internal client has auto-fallback for embeddings; open-source uses same endpoint).
    
    Returns:
        Embedding function, or None.
    """
    if not OPENAI_EMBEDDINGS_BASE_URL:
        return None
    
    from openai import OpenAI
    emb_client = OpenAI(
        api_key=OPENAI_EMBEDDINGS_API_KEY or "EMPTY",
        base_url=OPENAI_EMBEDDINGS_BASE_URL
    )

    batch_size = int(os.environ.get("OPENAI_EMBEDDINGS_BATCH_SIZE", "64"))

    def _embedding_func(texts):
        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            response = emb_client.embeddings.create(input=chunk, model=EMBEDDING_MODEL)
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings
    
    return _embedding_func


def _create_memory_system(llm_client):
    """Create MemorySystem with proper embedding configuration.
    
    If a separate embedding endpoint is configured (via config.py),
    it will be used instead of the llm_client's default embeddings.
    This is essential when the chat provider (e.g., vLLM) cannot serve embeddings.
    """
    embedding_func = _make_embedding_func()
    return MemorySystem(llm_client=llm_client, embedding_func=embedding_func)

def _build_accuracy_summary(results: List[Dict]) -> Dict[str, Any]:
    """Build JSON-serializable overall and per-question-type accuracy summary."""
    stats = compute_accuracy(results)
    by_type = {}

    for question_type, accuracy in sorted(stats.get("per_type", {}).items()):
        counts = stats.get("by_type_counts", {}).get(question_type, {})
        by_type[question_type] = {
            "total": counts.get("total", 0),
            "correct": counts.get("correct", 0),
            "accuracy": accuracy,
            "accuracy_percent": round(accuracy * 100, 2),
        }

    return {
        "total": stats.get("total", 0),
        "correct": stats.get("correct", 0),
        "accuracy": stats.get("overall", 0.0),
        "accuracy_percent": round(stats.get("overall", 0.0) * 100, 2),
        "by_question_type": by_type,
    }

def _log_longmemeval_metrics_to_mltracker(
    accuracy_summary: Dict[str, Any],
    project_name: str,
    experiment_name: str,
    step: int,
) -> None:
    """Optionally log LongMemEval official QA metrics to the experiment tracker."""
    import ml_tracker

    ml_tracker.init(project=project_name, name=experiment_name)
    metrics = {
        "eval/longmemeval/qa_accuracy": accuracy_summary["accuracy"],
        "eval/longmemeval/qa_total": accuracy_summary["total"],
        "eval/longmemeval/qa_correct": accuracy_summary["correct"],
    }

    for question_type, type_stats in accuracy_summary["by_question_type"].items():
        safe_question_type = str(question_type).replace("/", "_")
        metrics[f"eval/longmemeval/{safe_question_type}/qa_accuracy"] = type_stats["accuracy"]
        metrics[f"eval/longmemeval/{safe_question_type}/qa_total"] = type_stats["total"]
        metrics[f"eval/longmemeval/{safe_question_type}/qa_correct"] = type_stats["correct"]

    ml_tracker.log(data=metrics, step=step)
    ml_tracker.finish()

# ---------------------------------------------------------------------------
# Memory-Guided RAG helpers for eval
# ---------------------------------------------------------------------------

def _build_eval_chunk_store(sample: Dict, chunk_turn_size: int = 4) -> Dict[int, List[Dict]]:
    """Build per-session chunks from all haystack sessions in a sample.

    Returns dict of {session_idx: [{"chunk_index": int, "text": str}, ...]}.
    """
    sessions = sample.get("haystack_sessions", [])
    session_chunks = {}
    for sess_idx, session in enumerate(sessions):
        messages = session.get("messages", session) if isinstance(session, dict) else session
        if not isinstance(messages, list):
            continue
        chunks = []
        for start in range(0, len(messages), chunk_turn_size):
            end = min(start + chunk_turn_size, len(messages))
            lines = []
            for msg in messages[start:end]:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "").strip()
                if content:
                    lines.append(f"[{role}]: {content}")
            chunks.append({
                "chunk_index": len(chunks),
                "text": "\n".join(lines),
            })
        session_chunks[sess_idx] = chunks
    return session_chunks


def _build_memory_to_session_mapping(
    memory: MemorySystem,
) -> Dict[str, int]:
    """Build a mapping from memory text → session_idx for eval.

    Reads all memories from vector store and extracts session_idx from
    metadata if available, or returns empty dict.
    """
    mapping = {}
    if not hasattr(memory, 'vector_store'):
        return mapping
    vs = memory.vector_store
    memories_list = getattr(vs, 'memories', [])
    metadata_list = getattr(vs, 'metadata', [])
    for i, text in enumerate(memories_list):
        if i < len(metadata_list):
            meta = metadata_list[i]
            sess_idx = meta.get("session_idx") if isinstance(meta, dict) else None
            if sess_idx is not None and text:
                mapping[text] = sess_idx
    return mapping


def _match_memory_to_chunk(
    memory_text: str,
    chunks: List[Dict],
) -> int:
    """Match a single memory item to its best source chunk (same as train).

    Uses Jaccard token overlap. Returns best chunk_index.
    """
    import re

    if not memory_text or not chunks:
        return 0

    mem_tokens = set(re.findall(r"\w+", memory_text.lower()))
    if not mem_tokens:
        return 0

    best_idx = 0
    best_score = -1.0

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_tokens = set(re.findall(r"\w+", chunk_text.lower()))
        if not chunk_tokens:
            continue
        intersection = mem_tokens & chunk_tokens
        union = mem_tokens | chunk_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        if jaccard > best_score:
            best_score = jaccard
            best_idx = chunk["chunk_index"]

    return best_idx


def _build_chunk_context_for_eval(
    retrieved: List[Dict],
    session_chunks: Dict[int, List[Dict]],
    memory_session_map: Dict[str, int],
    rag_chunk_types: List[str],
    max_chunks: int = 3,
) -> Optional[str]:
    """Build chunk_context for eval, matching train's approach.

    For each retrieved memory of a configured RAG type:
    1. Look up which session it came from (via memory_session_map)
    2. Get that session's chunks
    3. Find the best matching chunk (via token overlap, same as train)
    4. Collect unique chunks and format as context

    Returns formatted chunk_context string, or None if no matches.
    """
    if not rag_chunk_types or not session_chunks or not retrieved:
        return None

    matched_chunks = []  # (session_idx, chunk_index, text)
    seen = set()

    for r in retrieved:
        mem_text = r.get("memory", "") if isinstance(r, dict) else str(r)
        # Check if this memory type is in RAG_CHUNK_TYPES
        matched_type = None
        for t in rag_chunk_types:
            if f"[{t.upper()}]" in mem_text:
                matched_type = t
                break
        if not matched_type:
            continue

        # Strip type prefix for matching
        clean_text = mem_text.replace(f"[{matched_type.upper()}] ", "")

        # Look up session index
        sess_idx = memory_session_map.get(mem_text)
        if sess_idx is None:
            # Fallback: try without prefix
            sess_idx = memory_session_map.get(clean_text)
        if sess_idx is None:
            continue

        # Get session chunks and find best match
        chunks = session_chunks.get(sess_idx, [])
        if not chunks:
            continue

        chunk_idx = _match_memory_to_chunk(clean_text, chunks)
        key = (sess_idx, chunk_idx)
        if key not in seen:
            seen.add(key)
            matched_chunks.append((sess_idx, chunk_idx, chunks[chunk_idx].get("text", "")))

    if not matched_chunks:
        return None

    # Limit and format
    matched_chunks = matched_chunks[:max_chunks]
    chunk_texts = []
    for sess_idx, chunk_idx, text in matched_chunks:
        if text:
            chunk_texts.append(f"[Session {sess_idx + 1}, Segment {chunk_idx + 1}]\n{text}")
    return "\n\n".join(chunk_texts) if chunk_texts else None


def _get_chunk_context(
    memory: MemorySystem,
    sample: Dict,
    retrieved: List[Dict],
) -> Optional[str]:
    """One-stop helper: build chunk_context for eval from memory + sample + retrieved.

    Returns None if RAG is disabled or no chunks matched.
    """
    from config import RAG_CHUNK_TYPES, CHUNK_TURN_SIZE
    if not RAG_CHUNK_TYPES:
        return None
    session_chunks = _build_eval_chunk_store(sample, CHUNK_TURN_SIZE)
    if not session_chunks:
        return None
    memory_session_map = _build_memory_to_session_mapping(memory)
    return _build_chunk_context_for_eval(
        retrieved, session_chunks, memory_session_map, RAG_CHUNK_TYPES
    )


def process_longmemeval_sample(sample_args):
    """Process a single LongMemEval sample (for parallel processing)."""
    (sample, idx, total, provider, model_name, judge_model_name, judge_provider,
     kwargs, judge_kwargs, max_sessions, top_k, mode, verbose,
     db_path_root, vector_store, *extra) = sample_args
    session_concurrency = extra[0] if extra else 8
    # Answer model config: (answer_provider, answer_model, answer_kwargs)
    answer_provider_name = extra[1] if len(extra) > 1 else None
    answer_model_name = extra[2] if len(extra) > 2 else None
    answer_kwargs = extra[3] if len(extra) > 3 else {}
    
    sample_id = sample['question_id']
    user_id = f'test_{sample_id}'
    
    # Compute per-sample persistence path (mirrors serial branch)
    if db_path_root:
        sample_db_path = os.path.join(db_path_root, sample_id)
    else:
        sample_db_path = get_db_path(
            dataset_name='longmemeval',
            model_name=model_name,
            vector_store=vector_store,
            sample_id=sample_id
        )
    
    # Create new LLM client for this process
    sample_llm_client = create_llm_client(provider=provider, model=model_name, **kwargs)
    sample_judge_client = create_llm_client(provider=judge_provider, model=judge_model_name, **judge_kwargs)
    sample_judge = LLMJudge(sample_judge_client)
    
    # Create answer client (separate from memory construction client)
    if answer_model_name and answer_provider_name:
        sample_answer_client = create_llm_client(
            provider=answer_provider_name, model=answer_model_name, **answer_kwargs
        )
    else:
        sample_answer_client = sample_llm_client
    
    # Limit sessions if requested
    if max_sessions:
        sample = sample.copy()
        sample['haystack_sessions'] = sample['haystack_sessions'][:max_sessions]
        if 'haystack_session_datetimes' in sample:
            sample['haystack_session_datetimes'] = sample['haystack_session_datetimes'][:max_sessions]
    
    # Initialize MemorySystem
    memory = _create_memory_system(sample_llm_client)
    
    result = {
        'question_id': sample_id,
        'question': sample['question'],
        'expected_answer': sample['answer'],
        'question_type': sample.get('question_type', 'unknown'),
        'index': idx
    }
    
    try:
        # Build memories
        if mode in ['full', 'build']:
            build_result = build_memories(
                memory, sample, user_id,
                max_sessions=max_sessions,
                verbose=False,  # Quiet for parallel
                session_concurrency=session_concurrency
            )
            result['memories_built'] = build_result['total_memories']
            # Persist to disk so a later --mode answer can reload
            memory.save(sample_db_path)
        
        # Answer-only mode: load previously persisted memories
        if mode == 'answer':
            if not memory.load(sample_db_path):
                result['error'] = f'Memory DB not found at {sample_db_path}; run --mode build first'
                result['llm_score'] = 0
                result['is_correct'] = False
                print(f"[{idx}/{total}] ⚠️  SKIP {sample_id}: 记忆数据库不存在")
                return result
        
        # Answer question (use answer_client for QA generation)
        if mode in ['full', 'answer']:
            current_time = get_current_time(sample)
            
            # Switch to answer model for QA generation
            if sample_answer_client is not sample_llm_client:
                memory.llm_client = sample_answer_client
            
            search_start = time.time()
            search_result = memory.search(sample['question'], user_id=user_id, limit=top_k)
            search_time = time.time() - search_start
            memories = search_result.get('results', []) if isinstance(search_result, dict) else []
            
            # Memory-Guided RAG: build chunk_context if configured
            chunk_context = _get_chunk_context(memory, sample, memories)
            
            gen_start = time.time()
            answer = memory.generate_answer(
                sample['question'], memories, user_id,
                current_time=current_time, chunk_context=chunk_context
            )
            gen_time = time.time() - gen_start
            
            judge_start = time.time()
            llm_score = sample_judge.evaluate(sample['question'], sample['answer'], answer)
            judge_time = time.time() - judge_start
            
            is_correct = (llm_score == 1)
            result.update({
                'generated_answer': answer,
                'is_correct': is_correct,
                'llm_score': llm_score,
                'memories_found': len(memories),
                'search_time': search_time,
                'gen_time': gen_time,
                'judge_time': judge_time,
            })
            
            status = '✅' if is_correct else '❌'
            print(f"[{idx}/{total}] {status} {sample_id}: {result.get('memories_built', 0)}条记忆, {search_time+gen_time:.1f}s")
        
    except Exception as e:
        result['error'] = str(e)
        result['llm_score'] = 0
        result['is_correct'] = False
        print(f"[{idx}/{total}] ❌ ERROR {sample_id}: {str(e)[:50]}")
    
    return result


def process_longmemeval_sample_agentic(sample_args):
    """Process a single LongMemEval sample using agentic (unified DSL) mode.

    In agentic mode, the model receives a unified prompt per session containing
    the full memory state + session text, and outputs compact DSL lines for all
    4 memory types at once. This matches how the model is trained in agentic RL.
    """
    from training.agentic.prompts import (
        build_turn_messages,
        format_memory_list,
        format_session_text,
    )
    from training.agentic.agent_loop import (
        LightMemoryState,
        parse_unified_response,
    )

    (sample, idx, total, provider, model_name, judge_model_name, judge_provider,
     kwargs, judge_kwargs, max_sessions, top_k, mode, verbose,
     db_path_root, vector_store, *extra) = sample_args
    session_concurrency = extra[0] if extra else 8
    answer_provider_name = extra[1] if len(extra) > 1 else None
    answer_model_name = extra[2] if len(extra) > 2 else None
    answer_kwargs = extra[3] if len(extra) > 3 else {}

    sample_id = sample['question_id']
    user_id = f'test_{sample_id}'

    # Create LLM clients
    sample_llm_client = create_llm_client(provider=provider, model=model_name, **kwargs)
    sample_judge_client = create_llm_client(provider=judge_provider, model=judge_model_name, **judge_kwargs)
    sample_judge = LLMJudge(sample_judge_client)

    # Limit sessions if requested
    sessions = sample.get('haystack_sessions', [])
    timestamps = sample.get('haystack_session_datetimes', [])
    if max_sessions:
        sessions = sessions[:max_sessions]
        timestamps = timestamps[:max_sessions]

    result = {
        'question_id': sample_id,
        'question': sample['question'],
        'expected_answer': sample['answer'],
        'question_type': sample.get('question_type', 'unknown'),
        'index': idx,
    }

    try:
        # ---- Build memories using agentic unified prompt ----
        if mode in ['full', 'build']:
            memory_state = LightMemoryState()
            build_start = time.time()

            for turn_idx, session in enumerate(sessions):
                messages = session.get('messages', session) if isinstance(session, dict) else session
                timestamp = timestamps[turn_idx] if turn_idx < len(timestamps) else ""

                session_text = format_session_text(messages)
                relevant = memory_state.get_relevant(session_text, top_k)

                turn_messages = build_turn_messages(
                    core_memory=memory_state.core_memory,
                    episodic_memories=format_memory_list(relevant["episodic"]),
                    semantic_memories=format_memory_list(relevant["semantic"]),
                    procedural_memories=format_memory_list(relevant["procedural"]),
                    session_text=session_text,
                    session_index=turn_idx,
                    session_date=timestamp,
                    top_k=top_k,
                )

                response = sample_llm_client.chat_completion(
                    messages=turn_messages,
                    temperature=0.6,
                    response_format=None,
                )

                parsed, _ = parse_unified_response(response)
                memory_state.apply_actions(parsed)

            build_time = time.time() - build_start
            all_memories = memory_state.get_all_memories()
            result['memories_built'] = len(all_memories)
            result['build_time'] = build_time

            # Store into FAISS for embedding-based retrieval at answer time
            embedding_func = _make_embedding_func()
            if embedding_func is None:
                embedding_func = lambda texts: sample_llm_client.get_embeddings(texts)

            memory_system = MemorySystem(llm_client=sample_llm_client, embedding_func=embedding_func)
            memory_system.core_memory.human = memory_state.core_memory

            non_core_memories = (
                [f"[EPISODIC] {m}" for m in memory_state.episodic]
                + [f"[SEMANTIC] {m}" for m in memory_state.semantic]
                + [f"[PROCEDURAL] {m}" for m in memory_state.procedural]
            )
            if non_core_memories:
                metadatas = [{"user_id": user_id, "type": "memory"} for _ in non_core_memories]
                memory_system.vector_store.add(non_core_memories, metadatas)

            # Persist
            if db_path_root:
                sample_db_path = os.path.join(db_path_root, sample_id)
            else:
                sample_db_path = get_db_path(
                    dataset_name='longmemeval',
                    model_name=model_name,
                    vector_store=vector_store,
                    sample_id=sample_id,
                )
            memory_system.save(sample_db_path)

        # ---- Answer-only mode: load from disk ----
        if mode == 'answer':
            embedding_func = _make_embedding_func()
            if embedding_func is None:
                embedding_func = lambda texts: sample_llm_client.get_embeddings(texts)
            memory_system = MemorySystem(llm_client=sample_llm_client, embedding_func=embedding_func)
            if db_path_root:
                sample_db_path = os.path.join(db_path_root, sample_id)
            else:
                sample_db_path = get_db_path(
                    dataset_name='longmemeval',
                    model_name=model_name,
                    vector_store=vector_store,
                    sample_id=sample_id,
                )
            if not memory_system.load(sample_db_path):
                result['error'] = f'Memory DB not found at {sample_db_path}; run --mode build first'
                result['llm_score'] = 0
                result['is_correct'] = False
                print(f"[{idx}/{total}] ⚠️  SKIP {sample_id}: 记忆数据库不存在")
                return result

        # ---- Answer question ----
        if mode in ['full', 'answer']:
            current_time = get_current_time(sample)

            search_start = time.time()
            search_result = memory_system.search(sample['question'], user_id=user_id, limit=top_k)
            search_time = time.time() - search_start
            memories = search_result.get('results', []) if isinstance(search_result, dict) else []

            # Memory-Guided RAG: build chunk_context if configured
            chunk_context = _get_chunk_context(memory_system, sample, memories)

            gen_start = time.time()
            answer = memory_system.generate_answer(
                sample['question'], memories, user_id,
                current_time=current_time, chunk_context=chunk_context,
            )
            gen_time = time.time() - gen_start

            judge_start = time.time()
            llm_score = sample_judge.evaluate(sample['question'], sample['answer'], answer)
            judge_time = time.time() - judge_start

            is_correct = (llm_score == 1)
            result.update({
                'generated_answer': answer,
                'is_correct': is_correct,
                'llm_score': llm_score,
                'memories_found': len(memories),
                'search_time': search_time,
                'gen_time': gen_time,
                'judge_time': judge_time,
            })

            status = '✅' if is_correct else '❌'
            print(f"[{idx}/{total}] {status} {sample_id}: {result.get('memories_built', '?')}条记忆, {search_time+gen_time:.1f}s")

    except Exception as e:
        result['error'] = str(e)
        result['llm_score'] = 0
        result['is_correct'] = False
        print(f"[{idx}/{total}] ❌ ERROR {sample_id}: {str(e)[:80]}")

    return result


def get_db_path(dataset_name: str, model_name: str, vector_store: str = "faiss", 
                conversation_id: str = None, sample_id: str = None, 
                character_id: str = None) -> str:
    """
    生成统一的数据库路径（参考老代码 test_memory_system.py）
    
    结构: {vector_store}_data/{dataset}/{model}/{id}/
    例如: faiss_data/locomo/gpt-4o-mini/conv-26/
    
    Args:
        dataset_name: 数据集名称 (locomo, longmemeval, perltqa)
        model_name: 模型名称 (gpt-4o-mini, claude-sonnet-4-5-20250929等)
        vector_store: 向量数据库类型 (faiss, qdrant)
        conversation_id: 对话ID (用于locomo，如conv-26)
        sample_id: 样本ID (用于longmemeval)
        character_id: 人物ID (用于perltqa)
    
    Returns:
        数据库路径字符串
    """
    # 清理模型名称中的特殊字符
    clean_model = model_name.replace('/', '_').replace('.', '_')
    
    # 基础路径
    base_dir = f"{vector_store}_data"
    
    # 构建路径: base_dir/dataset/model/id
    if conversation_id:
        # LOCOMO: faiss_data/locomo/gpt-4o-mini/conv-26
        db_path = f"./{base_dir}/{dataset_name}/{clean_model}/{conversation_id}"
    elif sample_id:
        # LongMemEval: faiss_data/longmemeval/gpt-4o-mini/sample_xxx
        db_path = f"./{base_dir}/{dataset_name}/{clean_model}/{sample_id}"
    elif character_id:
        # PerLTQA: faiss_data/perltqa/gpt-4o-mini/char_xxx
        db_path = f"./{base_dir}/{dataset_name}/{clean_model}/{character_id}"
    else:
        # 默认（单对话测试）
        db_path = f"./{base_dir}/{dataset_name}/{clean_model}/default"
    
    return db_path


def _process_single_session(args_tuple):
    """Process a single session for build_memories (used by ThreadPoolExecutor)."""
    memory, session, i, total, timestamp, user_id = args_tuple
    metadata = {'timestamp': timestamp} if timestamp else None
    
    try:
        session_start = time.time()
        messages = session.get('messages', session) if isinstance(session, dict) else session
        result = memory.add(messages, user_id=user_id, metadata=metadata, session_idx=i)
        session_time = time.time() - session_start
        
        if result and 'results' in result:
            count = len(result['results'])
            return {'session_id': i, 'memories_count': count, 'timestamp': timestamp, 'session_time': session_time}
        return {'session_id': i, 'memories_count': 0, 'timestamp': timestamp, 'session_time': session_time}
    except Exception as e:
        return {'session_id': i, 'error': str(e)}


def build_memories(
    memory: MemorySystem, 
    sample: Dict, 
    user_id: str,
    max_sessions: int = None, 
    verbose: bool = True,
    session_concurrency: int = 8
) -> Dict[str, Any]:
    """Build memories from conversation sessions.
    
    Args:
        memory: MemorySystem instance
        sample: Sample dict containing haystack_sessions
        user_id: User identifier
        max_sessions: Limit number of sessions to process
        verbose: Print progress details
        session_concurrency: Number of sessions to process concurrently.
            Higher values improve throughput by saturating the vLLM server
            with more concurrent requests. Trade-off: slightly reduced
            deduplication quality since concurrent sessions cannot see
            each other's newly written memories.
    """
    sessions = sample['haystack_sessions']
    timestamps = sample.get('haystack_session_datetimes', [])
    
    if max_sessions:
        sessions = sessions[:max_sessions]
        timestamps = timestamps[:max_sessions]
    
    print(f"\n📦 构建记忆 ({len(sessions)} 个会话, 并发={session_concurrency})")
    print("=" * 60)
    
    build_start = time.time()
    total_memories = 0
    session_results = []
    
    if session_concurrency <= 1:
        # Sequential fallback
        for i, session in enumerate(sessions):
            if verbose:
                print(f"处理会话 {i+1}/{len(sessions)}...", end=" ", flush=True)
            
            timestamp = timestamps[i] if i < len(timestamps) else None
            metadata = {'timestamp': timestamp} if timestamp else None
            
            try:
                session_start = time.time()
                messages = session.get('messages', session) if isinstance(session, dict) else session
                result = memory.add(messages, user_id=user_id, metadata=metadata, session_idx=i)
                session_time = time.time() - session_start
                
                if result and 'results' in result:
                    count = len(result['results'])
                    total_memories += count
                    session_results.append({
                        'session_id': i,
                        'memories_count': count,
                        'timestamp': timestamp,
                        'session_time': session_time
                    })
                    if verbose:
                        print(f"✅ {count} 条记忆 ({session_time:.1f}s)")
                else:
                    if verbose:
                        print(f"⚠️ 无结果")
            except Exception as e:
                print(f"❌ 错误: {str(e)[:50]}")
                session_results.append({'session_id': i, 'error': str(e)})
    else:
        # Concurrent session processing: saturate vLLM with more parallel requests
        task_args = [
            (memory, session, i, len(sessions),
             timestamps[i] if i < len(timestamps) else None, user_id)
            for i, session in enumerate(sessions)
        ]
        
        with ThreadPoolExecutor(max_workers=session_concurrency) as executor:
            futures = {executor.submit(_process_single_session, arg): arg[2] for arg in task_args}
            
            for future in as_completed(futures):
                session_idx = futures[future]
                result = future.result()
                session_results.append(result)
                
                if 'error' in result:
                    print(f"❌ 会话 {session_idx+1}: {result['error']}")
                else:
                    count = result.get('memories_count', 0)
                    total_memories += count
                    if verbose:
                        print(f"✅ 会话 {session_idx+1}/{len(sessions)}: {count} 条记忆 ({result.get('session_time', 0):.1f}s)")
        
        # Sort results by session_id for consistent ordering
        session_results.sort(key=lambda x: x.get('session_id', 0))
    
    build_time = time.time() - build_start
    
    print(f"\n✅ 记忆构建完成:")
    print(f"   总记忆数: {total_memories}")
    print(f"   处理会话: {len(sessions)}")
    print(f"   总耗时: {build_time:.1f}秒")
    
    return {
        'total_memories': total_memories,
        'sessions_processed': len(sessions),
        'build_time': build_time,
        'session_results': session_results,
        'user_id': user_id
    }


def _answer_single_question(
    memory: MemorySystem,
    q: Dict,
    user_id: str,
    judge: LLMJudge,
    index: int,
    total: int,
    top_k: int = 10,
    current_time: str = None,
    rag_context: Optional[tuple] = None,
) -> Dict:
    """Answer a single question (for parallel processing).

    Args:
        rag_context: Optional (session_chunks, memory_session_map) tuple for RAG.
    """
    result = {
        'question_id': q['question_id'],
        'question': q['question'],
        'expected_answer': q['answer'],
        'question_type': q.get('question_type', 'unknown'),
        'index': index
    }
    
    try:
        # Search memories
        search_start = time.time()
        search_result = memory.search(q['question'], user_id=user_id, limit=top_k)
        search_time = time.time() - search_start
        
        memories = search_result.get('results', []) if isinstance(search_result, dict) else []
        
        # Memory-Guided RAG: build chunk_context if configured
        chunk_context = None
        if rag_context:
            from config import RAG_CHUNK_TYPES
            if RAG_CHUNK_TYPES:
                session_chunks, memory_session_map = rag_context
                chunk_context = _build_chunk_context_for_eval(
                    memories, session_chunks, memory_session_map, RAG_CHUNK_TYPES
                )
        
        # Generate answer
        gen_start = time.time()
        answer = memory.generate_answer(
            q['question'], memories, user_id,
            current_time=current_time, chunk_context=chunk_context
        )
        gen_time = time.time() - gen_start
        
        # Judge
        judge_start = time.time()
        llm_score = judge.evaluate(q['question'], q['answer'], answer)
        judge_time = time.time() - judge_start
        
        is_correct = (llm_score == 1)
        
        result.update({
            'generated_answer': answer,
            'is_correct': is_correct,
            'llm_score': llm_score,
            'memories_found': len(memories),
            'search_time': search_time,
            'gen_time': gen_time,
            'judge_time': judge_time,
        })
        
        status = '✅' if is_correct else '❌'
        print(f"[{index}/{total}] {status} {q['question_id']}: {search_time+gen_time:.1f}s")
        
    except Exception as e:
        result['error'] = str(e)
        result['llm_score'] = 0
        result['is_correct'] = False
        print(f"[{index}/{total}] ❌ ERROR {q['question_id']}: {str(e)[:50]}")
    
    return result


def answer_questions(
    memory: MemorySystem,
    questions: List[Dict],
    user_id: str,
    judge: LLMJudge,
    limit: int = None,
    top_k: int = 10,
    current_time: str = None,
    verbose: bool = True,
    concurrency: int = 1,
    rag_context: Optional[tuple] = None,
) -> List[Dict]:
    """Answer questions using the memory system with optional concurrency.
    
    Args:
        concurrency: Number of concurrent workers (1 = sequential)
        rag_context: Optional (session_chunks, memory_session_map) tuple for RAG.
    """
    if limit:
        questions = questions[:limit]
    
    total_questions = len(questions)
    
    if concurrency > 1 and total_questions > 1:
        # Parallel processing
        print(f"\n📝 回答问题 ({total_questions} 个问题, {concurrency} 并发)")
        print("=" * 60)
        
        total_start = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    _answer_single_question, memory, q, user_id, judge,
                    i, total_questions, top_k, current_time, rag_context
                ): i for i, q in enumerate(questions, 1)
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    print(f"[{idx}/{total_questions}] ❌ 并发错误: {str(e)[:50]}")
        
        # Sort by original index
        results.sort(key=lambda x: x.get('index', 0))
        
        total_time = time.time() - total_start
        print_accuracy_stats(results, "📊 答题总结")
        print(f"   总耗时: {total_time:.1f}秒")
        
        return results
    
    # Sequential processing
    print(f"\n📝 回答问题 ({total_questions} 个问题)")
    print("=" * 60)
    
    results = []
    total_start = time.time()
    
    for i, q in enumerate(questions, 1):
        if verbose:
            print(f"\n[{i}/{total_questions}] {q['question_id']}")
            print(f"  问题: {q['question']}")
            print(f"  标准答案: {q['answer']}")
            print(f"  类型: {q.get('question_type', 'unknown')}")
        
        result = {
            'question_id': q['question_id'],
            'question': q['question'],
            'expected_answer': q['answer'],
            'question_type': q.get('question_type', 'unknown')
        }
        
        try:
            # Search memories first (matching old code pattern)
            search_start = time.time()
            search_result = memory.search(q['question'], user_id=user_id, limit=top_k)
            search_time = time.time() - search_start
            
            # Extract memories list from result
            memories = search_result.get('results', []) if isinstance(search_result, dict) else []
            
            if verbose:
                print(f"  检索到 {len(memories)} 条相关记忆 ({search_time:.1f}s)")
            
            # Memory-Guided RAG: build chunk_context if configured
            chunk_context = None
            if rag_context:
                from config import RAG_CHUNK_TYPES
                if RAG_CHUNK_TYPES:
                    session_chunks, memory_session_map = rag_context
                    chunk_context = _build_chunk_context_for_eval(
                        memories, session_chunks, memory_session_map, RAG_CHUNK_TYPES
                    )

            # Generate answer - pass memories to match old code signature
            gen_start = time.time()
            answer = memory.generate_answer(
                q['question'], memories, user_id,
                current_time=current_time, chunk_context=chunk_context
            )
            gen_time = time.time() - gen_start
            
            if verbose:
                print(f"  💡 生成答案: {answer[:200]}..." if len(answer) > 200 else f"  💡 生成答案: {answer}")
            
            # Judge
            judge_start = time.time()
            llm_score = judge.evaluate(q['question'], q['answer'], answer)
            judge_time = time.time() - judge_start
            
            is_correct = (llm_score == 1)
            
            result.update({
                'generated_answer': answer,
                'is_correct': is_correct,
                'llm_score': llm_score,
                'memories_found': len(memories),
                'search_time': search_time,
                'gen_time': gen_time,
                'judge_time': judge_time,
            })
            
            status = '✅' if is_correct else '❌'
            if verbose:
                print(f"  ⏱️  耗时: 搜索{search_time:.1f}秒 + 生成{gen_time:.1f}秒")
                print(f"  🎯 LLM Judge: {status} {'CORRECT' if is_correct else 'WRONG'} ({judge_time:.1f}秒)")
            else:
                print(f"[{i}/{total_questions}] {status} {q['question_id']}: {search_time+gen_time:.1f}s")
            
        except Exception as e:
            result['error'] = str(e)
            result['llm_score'] = 0
            result['is_correct'] = False
            print(f"[{i}/{total_questions}] ❌ ERROR {q['question_id']}: {str(e)[:50]}")
        
        results.append(result)
    
    total_time = time.time() - total_start
    
    print_accuracy_stats(results, "📊 答题总结")
    print(f"   总耗时: {total_time:.1f}秒")
    
    return results


def run_evaluation(args) -> int:
    """Main evaluation loop."""
    
    # ========== API 配置 ==========
    provider = args.provider
    model = args.model or ANSWER_MODEL
    judge_model = args.judge_model or JUDGE_MODEL
    
    # Resolve API config from args / env / config.py (used for openai provider and judge)
    _orig_api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    _orig_base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
    
    # Provider-specific setup (only openai needs env var config here)
    if provider == 'openai':
        if not _orig_api_key:
            print("Error: OPENAI_API_KEY not configured.")
            print("Set via: --api-key, OPENAI_API_KEY env var, or config.py")
            return 1
        
        os.environ["OPENAI_API_KEY"] = _orig_api_key
        if _orig_base_url:
            os.environ["OPENAI_BASE_URL"] = _orig_base_url
    
    print("=" * 60)
    print(f"🧪 MemBuilder Evaluation")
    print(f"   数据集: {args.dataset}")
    print(f"   模式: {args.mode}")
    print(f"   模型: {model}")
    print(f"   Judge: {judge_model}")
    print(f"   Provider: {provider}")
    print(f"   Client: {AVAILABLE_PROVIDERS}")
    if OPENAI_EMBEDDINGS_BASE_URL:
        print(f"   Embedding: {EMBEDDING_MODEL} @ {OPENAI_EMBEDDINGS_BASE_URL}")
    elif len(AVAILABLE_PROVIDERS) > 2:
        print(f"   Embedding: {EMBEDDING_MODEL} (auto-fallback)")
    else:
        if provider == 'vllm':
            print(f"   Embedding: {EMBEDDING_MODEL} ⚠️  WARNING: vLLM may not support embeddings!")
            print(f"              Set OPENAI_EMBEDDINGS_BASE_URL env var for a separate embedding endpoint.")
        else:
            print(f"   Embedding: {EMBEDDING_MODEL} (same as chat endpoint)")
    print("=" * 60)
    
    # ========== 处理 --split 参数 ==========
    subset_file = getattr(args, 'subset_file', None)
    split_sample_ids = None
    if args.split and args.dataset == 'longmemeval':
        splits_file = Path(__file__).parent.parent / "data" / "longmemeval" / "splits" / "longmemeval_splits.json"
        if splits_file.exists():
            with open(splits_file, "r", encoding="utf-8") as f:
                splits_data = json.load(f)
            if args.split in splits_data:
                split_sample_ids = splits_data[args.split]
                print(f"   Using {args.split} split: {len(split_sample_ids)} samples")
            else:
                print(f"Error: Split '{args.split}' not found in {splits_file}")
                return 1
        else:
            print(f"Error: Splits file not found: {splits_file}")
            return 1
    
    # ========== 加载数据 ==========
    print("\n📚 加载数据...")
    # --questions limits question count, but for PerLTQA we need a
    # dedicated user/character cap to avoid accidentally truncating users.
    dataset_num_samples = args.questions
    if args.dataset in ('perltqa',):
        dataset_num_samples = args.num_users

    data = load_dataset(
        dataset_name=args.dataset,
        conv_id=args.conv_id,
        sample_id=args.sample_id,
        character_id=getattr(args, 'character_id', None),
        num_samples=dataset_num_samples,
        subset_file=subset_file,
        sample_ids=split_sample_ids
    )
    print(f"   加载 {len(data)} 条数据")
    
    # ========== 初始化客户端 ==========
    # 使用 create_llm_client 工厂函数 (internal or open-source, auto-detected)
    kwargs = {}
    if provider == 'vllm':
        kwargs['base_url'] = args.base_url or 'http://localhost:8000/v1'
        kwargs['api_key'] = args.api_key or 'EMPTY'
    elif provider == 'openai':
        if args.base_url:
            kwargs['base_url'] = args.base_url
        if args.api_key:
            kwargs['api_key'] = args.api_key
    # Other providers (e.g. internal) handle their own auth via create_llm_client
    
    llm_client = create_llm_client(provider=provider, model=model, **kwargs)
    
    # Judge uses a separate provider when the main provider can't run judge models
    judge_provider = args.judge_provider
    if judge_provider is None:
        if provider == 'vllm':
            # vLLM only serves the loaded model; judge needs a full API
            judge_provider = DEFAULT_PROVIDER
        else:
            judge_provider = provider
    
    if judge_provider not in AVAILABLE_PROVIDERS:
        print(f"Error: judge-provider '{judge_provider}' not available. Supported: {AVAILABLE_PROVIDERS}")
        return 1
    
    # Build judge kwargs: when judge uses a different provider OR a different
    # model than main, pass independent API config so the judge doesn't
    # accidentally point at the vLLM server (which only serves the fine-tuned model).
    if judge_provider == provider and judge_model == model:
        judge_kwargs = kwargs
    else:
        judge_kwargs = {}
        if judge_provider == 'openai':
            judge_api_key = args.judge_api_key or os.environ.get("DASHSCOPE_API_KEY") or _orig_api_key
            judge_base_url = args.judge_base_url or os.environ.get("DASHSCOPE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            if judge_api_key:
                judge_kwargs['api_key'] = judge_api_key
            if judge_base_url:
                judge_kwargs['base_url'] = judge_base_url
            if not judge_api_key:
                print("⚠️  Warning: No API key found for judge.")
                print("   Set DASHSCOPE_API_KEY / OPENAI_API_KEY env var or --judge-api-key before running.")
    
    judge_client = create_llm_client(provider=judge_provider, model=judge_model, **judge_kwargs)
    judge = LLMJudge(judge_client)
    
    print(f"   Judge Provider: {judge_provider}")
    
    # ========== Answer Model (独立于 memory construction model) ==========
    # When --answer-model is specified, QA answer generation uses a separate
    # (typically more capable) model instead of the memory construction model.
    answer_model = args.answer_model
    answer_provider = args.answer_provider
    answer_kwargs = {}
    
    if answer_model:
        # Resolve answer provider
        if answer_provider is None:
            if provider == 'vllm':
                answer_provider = DEFAULT_PROVIDER
            else:
                answer_provider = provider
        
        if answer_provider not in AVAILABLE_PROVIDERS:
            print(f"Error: answer-provider '{answer_provider}' not available. Supported: {AVAILABLE_PROVIDERS}")
            return 1
        
        # Build answer kwargs
        if answer_provider == 'openai':
            answer_base_url = args.answer_base_url or os.environ.get("DASHSCOPE_BASE_URL") or _orig_base_url
            answer_api_key = args.answer_api_key or os.environ.get("DASHSCOPE_API_KEY") or _orig_api_key
            if answer_base_url:
                answer_kwargs['base_url'] = answer_base_url
            if answer_api_key:
                answer_kwargs['api_key'] = answer_api_key
        
        answer_client = create_llm_client(provider=answer_provider, model=answer_model, **answer_kwargs)
        print(f"   Answer Model: {answer_model} ({answer_provider})")
        if answer_kwargs.get('base_url'):
            print(f"   Answer URL: {answer_kwargs['base_url']}")
    else:
        # No separate answer model: use same client as memory construction
        answer_model = model
        answer_provider = provider
        answer_kwargs = kwargs
        answer_client = llm_client
    
    # ========== 按对话分组处理 (LOCOMO) ==========
    if args.dataset == 'locomo':
        all_conv_ids = sorted(set(d['conversation_id'] for d in data))
        
        if len(all_conv_ids) > 1:
            print(f"\n🔄 检测到 {len(all_conv_ids)} 个对话，将逐个处理")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(args.output_dir) / f"locomo_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for i, conv_id in enumerate(all_conv_ids, 1):
            print(f"\n{'='*60}")
            print(f"🔹 [{i}/{len(all_conv_ids)}] 处理对话: {conv_id}")
            print(f"{'='*60}")
            
            conv_questions = [d for d in data if d['conversation_id'] == conv_id]
            conv_sample = conv_questions[0]
            user_id = f'test_{conv_id}'
            
            print(f"   问题数: {len(conv_questions)}")
            print(f"   会话数: {conv_sample.get('num_sessions', 'unknown')}")
            
            # 生成数据库路径（如果用户指定了db_path，使用它；否则自动生成）
            if args.db_path:
                conv_db_path = os.path.join(args.db_path, conv_id)
            else:
                conv_db_path = get_db_path(
                    dataset_name='locomo',
                    model_name=model,
                    vector_store=args.vector_store,
                    conversation_id=conv_id
                )
            print(f"   📁 记忆路径: {conv_db_path}")
            
            # Limit sessions if requested
            if args.sessions:
                conv_sample = conv_sample.copy()
                conv_sample['haystack_sessions'] = conv_sample['haystack_sessions'][:args.sessions]
                conv_sample['haystack_session_datetimes'] = conv_sample['haystack_session_datetimes'][:args.sessions]
            
            # 初始化 MemorySystem
            memory = _create_memory_system(llm_client)
            
            # 构建记忆
            if args.mode in ['full', 'build']:
                build_result = build_memories(
                    memory, conv_sample, user_id,
                    max_sessions=args.sessions,
                    verbose=args.verbose,
                    session_concurrency=getattr(args, 'session_concurrency', 8)
                )
                print(f"   ✅ 构建完成: {build_result['total_memories']} 条记忆")
                
                # 保存记忆到磁盘
                memory.save(conv_db_path)
            
            # 仅回答模式：从磁盘加载记忆
            elif args.mode == 'answer':
                if not memory.load(conv_db_path):
                    print(f"   ⚠️  跳过: 记忆数据库不存在，请先运行 --mode build")
                    continue
            
            # 回答问题
            if args.mode in ['full', 'answer']:
                current_time = get_current_time(conv_sample)
                if current_time:
                    print(f"📅 数据集时间线: {current_time}")
                
                # Switch to answer model for QA generation
                if answer_client is not llm_client:
                    memory.llm_client = answer_client
                
                # Memory-Guided RAG: build rag_context if configured
                rag_context = None
                from config import RAG_CHUNK_TYPES, CHUNK_TURN_SIZE
                if RAG_CHUNK_TYPES:
                    session_chunks = _build_eval_chunk_store(conv_sample, CHUNK_TURN_SIZE)
                    memory_session_map = _build_memory_to_session_mapping(memory)
                    if session_chunks:
                        rag_context = (session_chunks, memory_session_map)
                        print(f"   📎 RAG chunk context enabled: types={RAG_CHUNK_TYPES}")
                
                results = answer_questions(
                    memory, conv_questions, user_id, judge,
                    limit=args.questions,
                    top_k=args.top_k,
                    current_time=current_time,
                    verbose=args.verbose,
                    concurrency=args.concurrency,
                    rag_context=rag_context
                )
                
                # 保存单个对话结果
                valid_results = [r for r in results if 'llm_score' in r]
                conv_result = {
                    'conversation_id': conv_id,
                    'questions': len(valid_results),
                    'correct': sum(r['llm_score'] for r in valid_results),
                    'accuracy': round(sum(r['llm_score'] for r in valid_results) / len(valid_results) * 100, 2) if valid_results else 0,
                    'results': results
                }
                
                with open(result_dir / f'{conv_id}_results.json', 'w', encoding='utf-8') as f:
                    json.dump(conv_result, f, indent=2, ensure_ascii=False)
                
                all_results.append(conv_result)
        
        # 汇总报告
        if all_results:
            total_q = sum(r['questions'] for r in all_results)
            total_c = sum(r['correct'] for r in all_results)
            acc = total_c / total_q * 100 if total_q > 0 else 0
            
            summary = {
                'timestamp': timestamp,
                'dataset': args.dataset,
                'model': model,
                'judge_model': judge_model,
                'total_conversations': len(all_results),
                'total_questions': total_q,
                'total_correct': total_c,
                'overall_accuracy': f"{acc:.2f}%",
                'by_conversation': {r['conversation_id']: f"{r['accuracy']:.2f}%" for r in all_results}
            }
            
            with open(result_dir / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            print(f"📊 总体结果")
            print(f"{'='*60}")
            print(f"   准确率: {acc:.2f}% ({total_c}/{total_q})")
            for r in all_results:
                print(f"   {r['conversation_id']}: {r['accuracy']:.2f}%")
            print(f"\n💾 结果保存到: {result_dir}/")
    
    # ========== LongMemEval: 每个样本独立处理 ==========
    elif args.dataset == 'longmemeval':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(args.output_dir) / f"longmemeval_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        # Prepare arguments for each sample
        session_concurrency = getattr(args, 'session_concurrency', 8)
        sample_args_list = [
            (sample, i, len(data), provider, model, judge_model, judge_provider, kwargs, judge_kwargs,
             args.sessions, args.top_k, args.mode, args.verbose,
             args.db_path, args.vector_store,
             session_concurrency, answer_provider, answer_model, answer_kwargs)
            for i, sample in enumerate(data, 1)
        ]
        
        # Select processing function based on eval-mode
        eval_mode = getattr(args, 'eval_mode', 'batch')
        if eval_mode == 'agentic':
            sample_processor = process_longmemeval_sample_agentic
            print(f"   Eval Mode: agentic (unified DSL prompt)")
        else:
            sample_processor = process_longmemeval_sample
            print(f"   Eval Mode: batch (4 independent agents)")

        if args.parallel and len(data) > 1:
            # Parallel processing (multiprocessing)
            print(f"\n🚀 LongMemEval 并行处理 ({args.workers} 进程, {len(data)} 样本)")
            print("=" * 60)
            
            from multiprocessing import Pool
            
            with Pool(processes=args.workers) as pool:
                all_results = list(tqdm(
                    pool.imap(sample_processor, sample_args_list),
                    total=len(sample_args_list),
                    desc="Eval Progress",
                    unit="sample",
                ))
            
            # Sort by original index
            all_results.sort(key=lambda x: x.get('index', 0))
        elif eval_mode == 'agentic':
            # Sequential agentic processing (use unified sample_processor)
            for sample_arg in tqdm(sample_args_list, desc="Eval Progress", unit="sample"):
                result = sample_processor(sample_arg)
                all_results.append(result)
        else:
            # Sequential batch processing (original logic)
            pbar = tqdm(enumerate(data, 1), total=len(data), desc="Eval Progress", unit="sample")
            for i, sample in pbar:
                sample_id = sample['question_id']
                user_id = f'test_{sample_id}'
                pbar.set_postfix_str(sample_id)
                
                print(f"\n{'='*60}")
                print(f"🔹 [{i}/{len(data)}] 样本: {sample_id}")
                print(f"{'='*60}")
                print(f"   会话数: {len(sample.get('haystack_sessions', []))}")
                
                # 生成数据库路径
                if args.db_path:
                    sample_db_path = os.path.join(args.db_path, sample_id)
                else:
                    sample_db_path = get_db_path(
                        dataset_name='longmemeval',
                        model_name=model,
                        vector_store=args.vector_store,
                        sample_id=sample_id
                    )
                print(f"   📁 记忆路径: {sample_db_path}")
                
                # Limit sessions if requested
                if args.sessions:
                    sample = sample.copy()
                    sample['haystack_sessions'] = sample['haystack_sessions'][:args.sessions]
                    if 'haystack_session_datetimes' in sample:
                        sample['haystack_session_datetimes'] = sample['haystack_session_datetimes'][:args.sessions]
                
                # 初始化 MemorySystem
                memory = _create_memory_system(llm_client)
                
                # 构建记忆
                if args.mode in ['full', 'build']:
                    build_result = build_memories(
                        memory, sample, user_id,
                        max_sessions=args.sessions,
                        verbose=args.verbose,
                        session_concurrency=getattr(args, 'session_concurrency', 8)
                    )
                    print(f"   ✅ 构建完成: {build_result['total_memories']} 条记忆")
                    
                    # 保存记忆到磁盘
                    memory.save(sample_db_path)
                
                # 仅回答模式：从磁盘加载记忆
                elif args.mode == 'answer':
                    if not memory.load(sample_db_path):
                        print(f"   ⚠️  跳过: 记忆数据库不存在，请先运行 --mode build")
                        continue
                
                # 回答问题 (LongMemEval 每个样本只有一个问题)
                if args.mode in ['full', 'answer']:
                    current_time = get_current_time(sample)
                    
                    # Switch to answer model for QA generation
                    if answer_client is not llm_client:
                        memory.llm_client = answer_client
                    
                    # 构造问题格式
                    question = {
                        'question_id': sample['question_id'],
                        'question': sample['question'],
                        'answer': sample['answer'],
                        'question_type': sample.get('question_type', 'unknown')
                    }
                    
                    results = answer_questions(
                        memory, [question], user_id, judge,
                        top_k=args.top_k,
                        current_time=current_time,
                        verbose=args.verbose
                    )
                    
                    if results:
                        all_results.extend(results)
        
        # 汇总报告
        if all_results:
            accuracy_summary = _build_accuracy_summary(all_results)
            summary = {
                'timestamp': timestamp,
                'dataset': args.dataset,
                'split': args.split,
                'mode': args.mode,
                'model': model,
                'judge_model': judge_model,
                'total_samples': accuracy_summary['total'],
                'total_correct': accuracy_summary['correct'],
                'overall_accuracy': f"{accuracy_summary['accuracy_percent']:.2f}%",
                'accuracy': accuracy_summary,
            }
            
            with open(result_dir / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            with open(result_dir / 'results.json', 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print_accuracy_stats(all_results, "📊 LongMemEval 总体结果")

            if args.track_metrics:
                tracker_project = args.tracker_project or os.environ.get("ML_TRACKER_PROJECT") or "membuilder_eval"
                tracker_experiment = args.tracker_experiment or f"longmemeval_{args.split or 'all'}_{timestamp}"
                _log_longmemeval_metrics_to_mltracker(
                    accuracy_summary=accuracy_summary,
                    project_name=tracker_project,
                    experiment_name=tracker_experiment,
                    step=args.tracker_step,
                )
                print(f"📈 LongMemEval 指标已上报到 ml_tracker: project={tracker_project}, name={tracker_experiment}")

            print(f"\n💾 结果保存到: {result_dir}/")
    
    # ========== PerLTQA: 按人物处理 ==========
    elif args.dataset == 'perltqa':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(args.output_dir) / f"perltqa_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for i, character in enumerate(data, 1):
            char_id = character['character_id']
            char_name = character.get('character_name', char_id)
            user_id = f'test_{char_id}'
            questions = character.get('questions', [])
            
            print(f"\n{'='*60}")
            print(f"🔹 [{i}/{len(data)}] {char_name} ({char_id})")
            print(f"{'='*60}")
            print(f"   问题数: {len(questions)}")
            print(f"   会话数: {character.get('num_sessions', 'unknown')}")
            
            # 生成数据库路径
            if args.db_path:
                char_db_path = os.path.join(args.db_path, char_id)
            else:
                char_db_path = get_db_path(
                    dataset_name='perltqa',
                    model_name=model,
                    vector_store=args.vector_store,
                    character_id=char_id
                )
            print(f"   📁 记忆路径: {char_db_path}")
            
            # Limit sessions if requested
            if args.sessions:
                character = character.copy()
                character['haystack_sessions'] = character['haystack_sessions'][:args.sessions]
                if 'haystack_session_datetimes' in character:
                    character['haystack_session_datetimes'] = character['haystack_session_datetimes'][:args.sessions]
            
            # 初始化 MemorySystem
            memory = _create_memory_system(llm_client)
            
            # 构建记忆
            if args.mode in ['full', 'build']:
                build_result = build_memories(
                    memory, character, user_id,
                    max_sessions=args.sessions,
                    verbose=args.verbose,
                    session_concurrency=getattr(args, 'session_concurrency', 8)
                )
                print(f"   ✅ 构建完成: {build_result['total_memories']} 条记忆")
                
                # 保存记忆到磁盘
                memory.save(char_db_path)
            
            # 仅回答模式：从磁盘加载记忆
            elif args.mode == 'answer':
                if not memory.load(char_db_path):
                    print(f"   ⚠️  跳过: 记忆数据库不存在，请先运行 --mode build")
                    continue
            
            # 回答问题
            if args.mode in ['full', 'answer']:
                current_time = get_current_time(character)
                
                # Switch to answer model for QA generation
                if answer_client is not llm_client:
                    memory.llm_client = answer_client
                
                # Memory-Guided RAG: build rag_context if configured
                rag_context = None
                from config import RAG_CHUNK_TYPES, CHUNK_TURN_SIZE
                if RAG_CHUNK_TYPES:
                    session_chunks = _build_eval_chunk_store(character, CHUNK_TURN_SIZE)
                    memory_session_map = _build_memory_to_session_mapping(memory)
                    if session_chunks:
                        rag_context = (session_chunks, memory_session_map)
                        print(f"   📎 RAG chunk context enabled: types={RAG_CHUNK_TYPES}")
                
                results = answer_questions(
                    memory, questions, user_id, judge,
                    limit=args.questions,
                    top_k=args.top_k,
                    current_time=current_time,
                    verbose=args.verbose,
                    concurrency=args.concurrency,
                    rag_context=rag_context
                )
                
                # 保存单个人物结果
                valid_results = [r for r in results if 'llm_score' in r]
                char_result = {
                    'character_id': char_id,
                    'character_name': char_name,
                    'questions': len(valid_results),
                    'correct': sum(r['llm_score'] for r in valid_results),
                    'accuracy': round(sum(r['llm_score'] for r in valid_results) / len(valid_results) * 100, 2) if valid_results else 0,
                    'results': results
                }
                
                with open(result_dir / f'{char_id}_results.json', 'w', encoding='utf-8') as f:
                    json.dump(char_result, f, indent=2, ensure_ascii=False)
                
                all_results.append(char_result)
        
        # 汇总报告
        if all_results:
            total_q = sum(r['questions'] for r in all_results)
            total_c = sum(r['correct'] for r in all_results)
            acc = total_c / total_q * 100 if total_q > 0 else 0
            
            summary = {
                'timestamp': timestamp,
                'dataset': args.dataset,
                'model': model,
                'judge_model': judge_model,
                'total_characters': len(all_results),
                'total_questions': total_q,
                'total_correct': total_c,
                'overall_accuracy': f"{acc:.2f}%",
                'by_character': {r['character_id']: f"{r['accuracy']:.2f}%" for r in all_results}
            }
            
            with open(result_dir / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            print(f"📊 PerLTQA 总体结果")
            print(f"{'='*60}")
            print(f"   准确率: {acc:.2f}% ({total_c}/{total_q})")
            for r in all_results:
                print(f"   {r['character_name']}: {r['accuracy']:.2f}%")
            print(f"\n💾 结果保存到: {result_dir}/")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='MemBuilder Unified Evaluation')
    
    # Dataset options
    parser.add_argument('--dataset', choices=['locomo', 'longmemeval', 'perltqa'], 
                       default='locomo', help='Dataset to use')
    parser.add_argument('--conv-id', default=None, help='Conversation ID (for locomo)')
    parser.add_argument('--sample-id', default=None, help='Sample ID (for longmemeval)')
    parser.add_argument('--character-id', default=None, help='Character ID (for perltqa)')
    parser.add_argument('--subset-file', default=None, help='Path to subset config file')
    parser.add_argument('--split', choices=['sft', 'rl', 'test'], default=None,
                       help='Use predefined split from data/longmemeval/splits/longmemeval_splits.json')
    
    # Mode options
    parser.add_argument('--mode', choices=['build', 'answer', 'full'], 
                       default='full', help='Test mode')
    parser.add_argument('--eval-mode', choices=['batch', 'agentic'], default='batch',
                       help='Eval rollout mode: batch (4 independent agents) or agentic (unified DSL prompt)')

    
    # Limit options
    parser.add_argument('--sessions', type=int, default=None, help='Limit sessions to build')
    parser.add_argument('--questions', type=int, default=None, help='Limit questions to test (per user for perltqa)')
    parser.add_argument('--num-users', type=int, default=None, help='Limit users/characters to evaluate (perltqa)')
    
    # Model configuration
    parser.add_argument('--model', default=None, help='LLM model for memory agents')
    parser.add_argument('--judge-model', default=None, help='LLM model for answer judge')
    parser.add_argument('--answer-model', default=None,
                       help='LLM model for QA answer generation (default: same as --model). '
                            'Use this to separate memory construction model from answer model.')
    parser.add_argument('--answer-provider', default=None,
                       help='Provider for answer model (default: same as --provider)')
    parser.add_argument('--answer-base-url', default=None,
                       help='Base URL for answer model provider')
    parser.add_argument('--answer-api-key', default=None,
                       help='API key for answer model provider')
    
    # Provider options
    _provider_choices = AVAILABLE_PROVIDERS
    parser.add_argument('--provider', default='openai', 
                       choices=_provider_choices,
                       help='LLM provider (default: openai)')
    parser.add_argument('--judge-provider', default=None,
                       choices=_provider_choices,
                       help='Judge provider (default: same as --provider; auto-falls back to API when --provider=vllm)')
    parser.add_argument('--judge-base-url', default=None,
                       help='Base URL for judge model provider (default: uses DASHSCOPE_BASE_URL or same as --base-url)')
    parser.add_argument('--judge-api-key', default=None,
                       help='API key for judge model provider (default: uses DASHSCOPE_API_KEY or same as --api-key)')
    parser.add_argument('--base-url', default=None,
                       help='API base URL (default: http://localhost:8000/v1 for vllm)')
    parser.add_argument('--api-key', default=None,
                       help='API key (default: EMPTY for vllm; or set OPENAI_API_KEY env var)')
    
    # Retrieval options
    parser.add_argument('--top-k', type=int, default=QA_ANSWERING_TOP_K, help='Top-K memories')
    
    # Persistence options
    parser.add_argument('--db-path', default=None,
                       help='Custom database path for memory persistence (auto-generated if not specified)')
    parser.add_argument('--vector-store', default='faiss', choices=['faiss'],
                       help='Vector store type (default: faiss)')
    
    # Concurrency options
    parser.add_argument('--concurrency', type=int, default=1,
                       help='Concurrent workers for answering questions (LOCOMO)')
    parser.add_argument('--session-concurrency', type=int, default=8,
                       help='Concurrent sessions within build_memories (higher = faster but less dedup)')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel sample processing (LongMemEval)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers for LongMemEval sample processing')
    
    # Output options
    parser.add_argument('--output-dir', default='logs', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    # Optional metric tracking for offline/standalone evaluation.
    # Disabled by default so original MemBuilder reproduction behavior is unchanged.
    parser.add_argument('--track-metrics', action='store_true',
                       help='Log LongMemEval QA accuracy metrics to ml_tracker')
    parser.add_argument('--tracker-project', default=None,
                       help='ml_tracker project name for --track-metrics')
    parser.add_argument('--tracker-experiment', default=None,
                       help='ml_tracker experiment/run name for --track-metrics')
    parser.add_argument('--tracker-step', type=int, default=0,
                       help='ml_tracker step for --track-metrics')
    
    args = parser.parse_args()
    return run_evaluation(args)


if __name__ == "__main__":
    exit(main())
