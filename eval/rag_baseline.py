#!/usr/bin/env python3
"""
Embedding-based RAG Baseline for MemBuilder evaluation.

This baseline skips the memory construction phase entirely. For each question,
it directly chunks the corresponding sessions, builds a FAISS index over the
chunks, retrieves the top-k most similar chunks via embedding similarity, and
feeds them as context to an answer model for QA.

Usage:
    # LongMemEval test split (default)
    python -m eval.rag_baseline --dataset longmemeval --split test

    # With custom parameters
    python -m eval.rag_baseline --dataset longmemeval --split test \
        --top-k 5 --chunk-turn-size 1 --parallel --workers 8

    # LoCoMo
    python -m eval.rag_baseline --dataset locomo --top-k 10

    # PerLTQA
    python -m eval.rag_baseline --dataset perltqa --top-k 5
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.datasets import load_dataset, get_current_time
from eval.llm_judge import LLMJudge
from eval.metrics import compute_accuracy, print_accuracy_stats
from config import (
    EMBEDDING_MODEL, QA_ANSWERING_TOP_K, RAG_BASELINE_TOP_K,
    OPENAI_EMBEDDINGS_BASE_URL, OPENAI_EMBEDDINGS_API_KEY,
)

# Try internal client first, fallback to public
try:
    from llm_client_internal import create_llm_client, AVAILABLE_PROVIDERS
except ImportError:
    from llm_client import create_llm_client, AVAILABLE_PROVIDERS


# =============================================================================
# Chunking
# =============================================================================

def chunk_all_sessions(
    sessions: List,
    chunk_turn_size: int = 1,
) -> List[Dict[str, Any]]:
    """Split all sessions into fixed-turn-size chunks.

    Args:
        sessions: List of session dicts (with 'messages' key) or lists of messages.
        chunk_turn_size: Number of turns (messages) per chunk.

    Returns:
        List of chunk dicts: {
            "session_idx": int,
            "chunk_index": int,
            "turn_start": int,
            "turn_end": int,
            "text": str,
        }
    """
    all_chunks = []
    global_chunk_idx = 0

    for sess_idx, session in enumerate(sessions):
        messages = session.get("messages", session) if isinstance(session, dict) else session
        if not isinstance(messages, list):
            continue

        for start in range(0, len(messages), chunk_turn_size):
            end = min(start + chunk_turn_size, len(messages))
            lines = []
            for msg in messages[start:end]:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "").strip()
                if content:
                    lines.append(f"[{role}]: {content}")

            text = "\n".join(lines)
            if text.strip():
                all_chunks.append({
                    "session_idx": sess_idx,
                    "chunk_index": global_chunk_idx,
                    "turn_start": start,
                    "turn_end": end,
                    "text": text,
                })
                global_chunk_idx += 1

    return all_chunks


# =============================================================================
# FAISS Index
# =============================================================================

def build_chunk_index(
    chunks: List[Dict[str, Any]],
    embedding_func,
) -> Tuple[Any, np.ndarray]:
    """Build a FAISS index from chunk embeddings.

    Args:
        chunks: List of chunk dicts with 'text' field.
        embedding_func: Function that takes List[str] and returns List[List[float]].

    Returns:
        (faiss_index, vectors_array)
    """
    import faiss

    texts = [c["text"] for c in chunks]
    if not texts:
        return None, np.array([])

    embeddings = embedding_func(texts)
    vectors = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    return index, vectors


def search_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    index,
    embedding_func,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve top-k most similar chunks for a question.

    Args:
        question: Query string.
        chunks: All chunks (for retrieving text by index).
        index: FAISS index.
        embedding_func: Embedding function.
        top_k: Number of results.

    Returns:
        List of chunk dicts with added 'score' field.
    """
    import faiss

    if index is None or not chunks:
        return []

    query_embedding = embedding_func([question])
    query_vector = np.array(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_vector)

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(chunks):
            result = dict(chunks[idx])
            result["score"] = float(score)
            results.append(result)

    return results


# =============================================================================
# Answer Generation
# =============================================================================

RAG_ANSWER_PROMPT = """You are answering a question based on retrieved conversation excerpts.

Retrieved Context (ranked by relevance):
{context}

{time_context}
Question: {question}

Instructions:
1. Carefully read the retrieved conversation excerpts to find relevant information.
2. Consider synonyms and related concepts.
3. If the excerpts mention specific dates/times, use those to answer time-related questions.
4. If excerpts contain contradictory information, prioritize the most recent one.

**For factual questions (What/When/Where/Who):**
- Answer based on direct information in the excerpts.
- If the specific fact is not mentioned, respond: "Not answerable"

**For inference/reasoning questions (Would/Could/Likely):**
- You CAN make reasonable inferences based on related information in the excerpts.

**When to say "Not answerable":**
- If the question asks about something NOT mentioned in ANY of the excerpts.
- If the excerpts are about a DIFFERENT person/entity than what the question asks about.

Provide a concise, direct answer based on the available information, or state "Not answerable" if the specific information is not present."""


def format_chunks_as_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string for the answer model."""
    if not retrieved_chunks:
        return "[No relevant context found]"

    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        sess_idx = chunk.get("session_idx", 0)
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")
        parts.append(f"[Excerpt {i} | Session {sess_idx + 1} | Relevance: {score:.3f}]\n{text}")

    return "\n\n".join(parts)


def generate_rag_answer(
    answer_client,
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    current_time: Optional[str] = None,
) -> str:
    """Generate an answer using retrieved chunks as context.

    Args:
        answer_client: LLM client for answer generation.
        question: The question to answer.
        retrieved_chunks: Retrieved chunk dicts.
        current_time: Current time in dataset timeline.

    Returns:
        Generated answer string.
    """
    context = format_chunks_as_context(retrieved_chunks)

    time_context = ""
    if current_time:
        time_context = (
            f"\n**The current date/time is {current_time}. Use this as the reference "
            f"point when answering questions about relative time (e.g., 'last year', "
            f"'yesterday', 'recently').**\n"
        )

    prompt = RAG_ANSWER_PROMPT.format(
        context=context,
        question=question,
        time_context=time_context,
    )

    return answer_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )


# =============================================================================
# Embedding Function Factory
# =============================================================================

def make_embedding_func(answer_client=None):
    """Create an embedding function based on environment configuration.

    Priority:
    1. Dedicated OPENAI_EMBEDDINGS_BASE_URL (remote embedding service)
    2. EMBEDDING_PROVIDER=local (local sentence-transformers)
    3. Fallback to answer_client.get_embeddings()

    Returns:
        Callable that takes List[str] and returns List[List[float]].
    """
    # Option 1: Dedicated remote embedding endpoint
    if OPENAI_EMBEDDINGS_BASE_URL:
        from openai import OpenAI
        emb_client = OpenAI(
            api_key=OPENAI_EMBEDDINGS_API_KEY or "EMPTY",
            base_url=OPENAI_EMBEDDINGS_BASE_URL,
        )
        batch_size = int(os.environ.get("OPENAI_EMBEDDINGS_BATCH_SIZE", "64"))
        model = os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)

        def _remote_embedding_func(texts):
            all_embeddings = []
            for start in range(0, len(texts), batch_size):
                chunk = texts[start:start + batch_size]
                response = emb_client.embeddings.create(input=chunk, model=model)
                all_embeddings.extend(item.embedding for item in response.data)
            return all_embeddings

        return _remote_embedding_func

    # Option 2: Local embedding via LLMClient
    if answer_client is not None:
        def _client_embedding_func(texts):
            return answer_client.get_embeddings(texts)
        return _client_embedding_func

    raise RuntimeError(
        "No embedding backend configured. Set OPENAI_EMBEDDINGS_BASE_URL for remote, "
        "or EMBEDDING_PROVIDER=local with LOCAL_EMBEDDING_MODEL for local embeddings."
    )


# =============================================================================
# Sample Processing
# =============================================================================

def process_single_sample(
    sample: Dict[str, Any],
    embedding_func,
    answer_client,
    judge: LLMJudge,
    top_k: int,
    chunk_turn_size: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Process a single evaluation sample through the RAG baseline pipeline.

    Returns a result dict with question_id, score, generated_answer, etc.
    """
    # Determine question and answer fields
    question_id = sample.get("question_id", sample.get("id", "unknown"))
    question = sample.get("question", "")
    gold_answer = sample.get("answer", "")
    question_type = sample.get("question_type", sample.get("type", "unknown"))
    current_time = get_current_time(sample)

    # Get sessions
    sessions = sample.get("haystack_sessions", [])

    result = {
        "question_id": question_id,
        "question": question,
        "expected_answer": gold_answer,
        "question_type": question_type,
    }

    try:
        # 1. Chunk all sessions
        chunks = chunk_all_sessions(sessions, chunk_turn_size)
        result["total_chunks"] = len(chunks)

        if not chunks:
            result["generated_answer"] = "Not answerable"
            result["llm_score"] = 0
            result["is_correct"] = False
            result["error"] = "No chunks generated from sessions"
            return result

        # 2. Build FAISS index
        index, _ = build_chunk_index(chunks, embedding_func)

        # 3. Retrieve top-k chunks
        retrieved = search_chunks(question, chunks, index, embedding_func, top_k)
        result["chunks_retrieved"] = len(retrieved)

        # 4. Generate answer
        gen_start = time.time()
        answer = generate_rag_answer(answer_client, question, retrieved, current_time)
        gen_time = time.time() - gen_start
        result["generated_answer"] = answer
        result["gen_time"] = gen_time

        # 5. Judge
        judge_start = time.time()
        llm_score = judge.evaluate(question, gold_answer, answer)
        judge_time = time.time() - judge_start

        result["llm_score"] = llm_score
        result["is_correct"] = (llm_score == 1)
        result["judge_time"] = judge_time

        if verbose:
            status = "✅" if result["is_correct"] else "❌"
            print(f"  {status} {question_id}: chunks={len(chunks)}, "
                  f"retrieved={len(retrieved)}, gen={gen_time:.1f}s")

    except Exception as e:
        result["error"] = str(e)[:200]
        result["llm_score"] = 0
        result["is_correct"] = False
        if verbose:
            print(f"  ❌ ERROR {question_id}: {str(e)[:80]}")

    return result


def _process_sample_wrapper(args):
    """Wrapper for multiprocessing (unpacks args tuple)."""
    (sample, idx, total, embedding_func, answer_client, judge,
     top_k, chunk_turn_size, verbose) = args

    result = process_single_sample(
        sample, embedding_func, answer_client, judge,
        top_k, chunk_turn_size, verbose,
    )
    result["index"] = idx

    is_correct = result.get("is_correct", False)
    question_id = result.get("question_id", "?")
    status = "✅" if is_correct else "❌"
    print(f"[{idx}/{total}] {status} {question_id}", flush=True)

    return result


# =============================================================================
# Main Runner
# =============================================================================

def run_rag_baseline(args) -> int:
    """Main RAG baseline evaluation loop."""
    print("=" * 60)
    print("🔍 RAG Baseline Evaluation")
    print("=" * 60)
    print(f"   Dataset: {args.dataset}")
    print(f"   Top-K: {args.top_k}")
    print(f"   Chunk Turn Size: {args.chunk_turn_size}")
    print(f"   Answer Model: {args.answer_model}")
    print(f"   Judge Model: {args.judge_model}")
    print("")

    # ── Load dataset ──
    load_kwargs = {}
    if args.dataset == "longmemeval" and args.split:
        # Load split sample IDs
        splits_path = Path(__file__).parent.parent / "data" / "longmemeval" / "splits" / "longmemeval_splits.json"
        if splits_path.exists():
            with open(splits_path) as f:
                splits = json.load(f)
            sample_ids = splits.get(args.split, [])
            if sample_ids:
                load_kwargs["sample_ids"] = sample_ids
                print(f"   Split: {args.split} ({len(sample_ids)} samples)")

    data = load_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        sample_id=args.sample_id,
        conv_id=args.conv_id,
        character_id=args.character_id,
        num_samples=args.limit,
        **load_kwargs,
    )

    if not data:
        print("ERROR: No data loaded.")
        return 1

    # ── Adapt data format ──
    # For LoCoMo, data is already per-question with haystack_sessions.
    # For LongMemEval, each sample has its own haystack_sessions + one question.
    # For PerLTQA, each character has haystack_sessions + multiple questions.
    # We flatten everything to a list of (question, sessions) pairs.
    samples = _flatten_samples(data, args.dataset)
    print(f"   Total samples to evaluate: {len(samples)}")
    print("")

    # ── Create clients ──
    # Answer model client
    answer_kwargs = {}
    if args.answer_base_url:
        answer_kwargs["base_url"] = args.answer_base_url
    if args.answer_api_key:
        answer_kwargs["api_key"] = args.answer_api_key
    elif os.environ.get("DASHSCOPE_API_KEY"):
        answer_kwargs["api_key"] = os.environ["DASHSCOPE_API_KEY"]
    if not answer_kwargs.get("base_url") and os.environ.get("DASHSCOPE_BASE_URL"):
        answer_kwargs["base_url"] = os.environ["DASHSCOPE_BASE_URL"]

    answer_client = create_llm_client(
        model=args.answer_model,
        **answer_kwargs,
    )

    # Judge client
    judge_kwargs = {}
    if args.judge_base_url:
        judge_kwargs["base_url"] = args.judge_base_url
    elif os.environ.get("DASHSCOPE_BASE_URL"):
        judge_kwargs["base_url"] = os.environ["DASHSCOPE_BASE_URL"]
    if args.judge_api_key:
        judge_kwargs["api_key"] = args.judge_api_key
    elif os.environ.get("DASHSCOPE_API_KEY"):
        judge_kwargs["api_key"] = os.environ["DASHSCOPE_API_KEY"]

    judge_client = create_llm_client(
        model=args.judge_model,
        **judge_kwargs,
    )
    judge = LLMJudge(judge_client)

    # Embedding function
    embedding_func = make_embedding_func(answer_client)

    # ── Process samples ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.output_dir) / f"rag_baseline_{args.dataset}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(samples)

    if args.parallel and total > 1:
        print(f"🚀 Parallel processing ({args.workers} workers, {total} samples)")
        print("=" * 60)

        # For parallel, we use ThreadPoolExecutor (embedding is IO-bound)
        sample_args = [
            (sample, i + 1, total, embedding_func, answer_client, judge,
             args.top_k, args.chunk_turn_size, args.verbose)
            for i, sample in enumerate(samples)
        ]

        correct_count = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process_sample_wrapper, arg): arg[1]
                for arg in sample_args
            }
            pbar = tqdm(
                as_completed(futures),
                total=total,
                desc="RAG Baseline",
                unit="sample",
            )
            for future in pbar:
                try:
                    result = future.result()
                    all_results.append(result)
                    if result.get("is_correct", False):
                        correct_count += 1
                    done = len(all_results)
                    acc = correct_count / done * 100 if done > 0 else 0
                    pbar.set_postfix_str(f"acc={acc:.1f}% ({correct_count}/{done})")
                except Exception as e:
                    idx = futures[future]
                    print(f"[{idx}/{total}] ❌ FATAL: {str(e)[:80]}")
            pbar.close()

        # Sort by original index
        all_results.sort(key=lambda x: x.get("index", 0))
    else:
        print(f"📝 Sequential processing ({total} samples)")
        print("=" * 60)

        correct_count = 0
        pbar = tqdm(enumerate(samples, 1), total=total, desc="RAG Baseline", unit="sample")
        for i, sample in pbar:
            result = process_single_sample(
                sample, embedding_func, answer_client, judge,
                args.top_k, args.chunk_turn_size, args.verbose,
            )
            result["index"] = i

            if result.get("is_correct", False):
                correct_count += 1
            acc = correct_count / i * 100
            pbar.set_postfix_str(f"acc={acc:.1f}% ({correct_count}/{i})")

            all_results.append(result)
        pbar.close()

    # ── Summary ──
    if all_results:
        accuracy_summary = _build_accuracy_summary(all_results)

        summary = {
            "timestamp": timestamp,
            "method": "rag_baseline",
            "dataset": args.dataset,
            "split": args.split,
            "top_k": args.top_k,
            "chunk_turn_size": args.chunk_turn_size,
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "total_samples": accuracy_summary["total"],
            "total_correct": accuracy_summary["correct"],
            "overall_accuracy": f"{accuracy_summary['accuracy_percent']:.2f}%",
            "accuracy": accuracy_summary,
        }

        with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(result_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print_accuracy_stats(all_results, "📊 RAG Baseline 总体结果")

        # Print per-type breakdown
        print(f"\n💾 结果保存到: {result_dir}/")

    return 0


# =============================================================================
# Helpers
# =============================================================================

def _flatten_samples(data: List[Dict], dataset_name: str) -> List[Dict]:
    """Flatten dataset into uniform per-question samples.

    Each returned sample has: question_id, question, answer, question_type,
    haystack_sessions, (optionally) haystack_session_datetimes.
    """
    if dataset_name == "longmemeval":
        # LongMemEval: each item IS already a per-question sample
        return data

    elif dataset_name == "locomo":
        # LoCoMo: load_locomo already flattens to per-question format
        # Each sample has: question, answer, question_type, haystack_sessions, etc.
        # Ensure question_id exists
        for i, sample in enumerate(data):
            if "question_id" not in sample:
                sample["question_id"] = f"{sample.get('conversation_id', 'conv')}_{i}"
        return data

    elif dataset_name == "perltqa":
        # PerLTQA: each character has multiple questions + shared sessions
        flat = []
        for char in data:
            char_id = char.get("character_id", "unknown")
            sessions = char.get("haystack_sessions", [])
            session_datetimes = char.get("haystack_session_datetimes", [])
            questions = char.get("questions", [])

            for q_idx, q in enumerate(questions):
                sample = {
                    "question_id": q.get("question_id", f"{char_id}_q{q_idx}"),
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "question_type": q.get("question_type", q.get("type", "unknown")),
                    "haystack_sessions": sessions,
                    "haystack_session_datetimes": session_datetimes,
                    "character_id": char_id,
                }
                # Copy question_date if present
                if "question_date" in q:
                    sample["question_date"] = q["question_date"]
                elif "question_date" in char:
                    sample["question_date"] = char["question_date"]
                flat.append(sample)
        return flat

    else:
        return data


def _build_accuracy_summary(results: List[Dict]) -> Dict[str, Any]:
    """Build JSON-serializable accuracy summary (same as eval/runner.py)."""
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


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG Baseline: Embedding-based retrieval without memory construction"
    )

    # Dataset options
    parser.add_argument("--dataset", choices=["locomo", "longmemeval", "perltqa"],
                        default="longmemeval", help="Dataset to evaluate")
    parser.add_argument("--data-path", default=None, help="Custom dataset path")
    parser.add_argument("--split", choices=["sft", "rl", "test"], default=None,
                        help="LongMemEval split (default: test)")
    parser.add_argument("--sample-id", default=None, help="Single sample ID (longmemeval)")
    parser.add_argument("--conv-id", default=None, help="Conversation ID (locomo)")
    parser.add_argument("--character-id", default=None, help="Character ID (perltqa)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")

    # RAG parameters
    parser.add_argument("--top-k", type=int, default=RAG_BASELINE_TOP_K,
                        help=f"Number of chunks to retrieve (default: {RAG_BASELINE_TOP_K})")
    parser.add_argument("--chunk-turn-size", type=int, default=1,
                        help="Number of turns per chunk (default: 1)")

    # Model configuration
    parser.add_argument("--answer-model", default=None,
                        help="Answer model (default: deepseek-v4-pro)")
    parser.add_argument("--answer-base-url", default=None,
                        help="Base URL for answer model")
    parser.add_argument("--answer-api-key", default=None,
                        help="API key for answer model")
    parser.add_argument("--judge-model", default=None,
                        help="Judge model (default: deepseek-v4-flash)")
    parser.add_argument("--judge-base-url", default=None,
                        help="Base URL for judge model")
    parser.add_argument("--judge-api-key", default=None,
                        help="API key for judge model")

    # Concurrency
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel sample processing")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers (default: 16)")

    # Output
    parser.add_argument("--output-dir", default="logs",
                        help="Output directory (default: logs)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output per sample")

    args = parser.parse_args()

    # Apply defaults for answer/judge models
    if args.answer_model is None:
        args.answer_model = os.environ.get("MEMBUILDER_INPROC_ANSWER_MODEL", "deepseek-v4-pro")
    if args.judge_model is None:
        args.judge_model = os.environ.get("MEMBUILDER_INPROC_JUDGE_MODEL", "deepseek-v4-flash")

    return run_rag_baseline(args)


if __name__ == "__main__":
    sys.exit(main())
