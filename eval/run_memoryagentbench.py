#!/usr/bin/env python3
"""
MemoryAgentBench evaluation runner (standalone).

Evaluates a checkpoint / API model on the MemoryAgentBench dataset with two
selectable pipelines, and reports BOTH:
  1. LLM-judge binary accuracy (same judge as locomo/longmemeval, for cross-
     dataset comparison)
  2. MemoryAgentBench official metrics (substring / exact_match / f1 / ROUGE /
     recsys recall), routed by dataset `source`.

Pipelines (--pipeline):
  - rag    : chunk context -> FAISS top-k retrieval -> answer -> judge.
             Pure API, no vLLM needed. Reuses eval/rag_baseline.py helpers.
  - memory : build structured memory from pseudo sessions -> search ->
             generate answer -> judge. Needs a chat backend (e.g. local vLLM
             serving the checkpoint). Reuses eval/runner.py build_memories +
             MemorySystem. Memory is built ONCE per context (shared by all
             questions of that context) to avoid expensive re-building.

This module does NOT modify any existing file. It only imports and reuses them.

Usage:
    # RAG pipeline (no vLLM), Conflict_Resolution split, 2 samples
    python -m eval.run_memoryagentbench \
        --pipeline rag --split Conflict_Resolution --limit 2 \
        --data-dir /path/to/MemoryAgentBench \
        --output-dir /tmp/mab_out

    # Memory pipeline against a local vLLM checkpoint
    python -m eval.run_memoryagentbench \
        --pipeline memory --split Accurate_Retrieval \
        --provider vllm --base-url http://localhost:8000/v1 --model <ckpt> \
        --data-dir /path/to/MemoryAgentBench
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.memoryagentbench_data import (
    load_memoryagentbench,
    compute_official_metrics,
    primary_metric_name,
    SPLITS,
)
from eval.answering_prompt import (
    build_mab_answering_prompt,
    is_mab_protocol_source,
)
from eval.llm_judge import LLMJudge
from eval.metrics import compute_accuracy, print_accuracy_stats


def _memories_to_strings(memories) -> List[str]:
    """Normalize retrieved memories into a flat list of strings.

    ``MemorySystem.search`` returns a list of dicts (each with a
    ``memory`` / ``text`` key). The TTL prompt builder takes ``List[str]``
    so we coerce here.
    """
    out: List[str] = []
    for m in memories or []:
        if isinstance(m, dict):
            text = m.get("memory") or m.get("text") or ""
        else:
            text = str(m)
        text = (text or "").strip()
        if text:
            out.append(text)
    return out

# Client module: internal version adds extra providers; public has openai/vllm.
try:
    from llm_client_internal import create_llm_client, AVAILABLE_PROVIDERS, DEFAULT_PROVIDER
except ImportError:
    from llm_client import create_llm_client, AVAILABLE_PROVIDERS, DEFAULT_PROVIDER


# =============================================================================
# Client / judge construction
# =============================================================================

def _build_answer_client(args):
    """Build the client used for answer generation (generate_answer).

    When an explicit --answer-model is provided and differs from the
    memory-construction model (--model), the answer client always uses the
    external API (e.g. DashScope) regardless of whether the memory pipeline
    runs on a local vLLM checkpoint.  This ensures a strong model like
    qwen3.7-max is used for QA even when memory construction uses a smaller
    fine-tuned checkpoint.

    Falls back to the local vLLM checkpoint only when no separate answer
    model is configured (i.e. answer_model == model in vLLM mode).
    """
    # If a separate answer model is specified (different from the build model),
    # always route to external API.
    use_external_answer = (
        args.answer_model
        and args.answer_model != args.model
    )

    if args.pipeline == "memory" and args.provider == "vllm" and not use_external_answer:
        kwargs = {}
        kwargs["base_url"] = args.base_url or "http://localhost:8000/v1"
        kwargs["api_key"] = args.api_key or "EMPTY"
        return create_llm_client(provider="vllm", model=args.model, **kwargs)

    # External API answer model (rag pipeline, openai provider, or explicit answer-model)
    kwargs = {}
    if args.answer_base_url:
        kwargs["base_url"] = args.answer_base_url
    elif os.environ.get("DASHSCOPE_BASE_URL"):
        kwargs["base_url"] = os.environ["DASHSCOPE_BASE_URL"]
    if args.answer_api_key:
        kwargs["api_key"] = args.answer_api_key
    elif os.environ.get("DASHSCOPE_API_KEY"):
        kwargs["api_key"] = os.environ["DASHSCOPE_API_KEY"]
    return create_llm_client(model=args.answer_model, **kwargs)


def _build_judge(args) -> LLMJudge:
    kwargs = {}
    if args.judge_base_url:
        kwargs["base_url"] = args.judge_base_url
    elif os.environ.get("DASHSCOPE_BASE_URL"):
        kwargs["base_url"] = os.environ["DASHSCOPE_BASE_URL"]
    if args.judge_api_key:
        kwargs["api_key"] = args.judge_api_key
    elif os.environ.get("DASHSCOPE_API_KEY"):
        kwargs["api_key"] = os.environ["DASHSCOPE_API_KEY"]
    judge_client = create_llm_client(model=args.judge_model, **kwargs)
    return LLMJudge(judge_client)


# =============================================================================
# RAG pipeline (per-question; reuses eval/rag_baseline helpers)
# =============================================================================

def _eval_rag(samples, args, judge: LLMJudge, data_dir: str) -> List[Dict]:
    from eval.rag_baseline import (
        chunk_all_sessions, build_chunk_index, search_chunks,
        generate_rag_answer, make_embedding_func,
    )

    answer_client = _build_answer_client(args)
    embedding_func = make_embedding_func(answer_client)

    def _process_one(sample, idx, total):
        result = _base_result(sample, idx)
        try:
            sessions = sample["haystack_sessions"]
            chunks = chunk_all_sessions(sessions, args.chunk_turn_size)
            if not chunks:
                result["generated_answer"] = "Not answerable"
                result["llm_score"] = 0
                result["is_correct"] = False
                result["error"] = "no chunks"
                return _finalize(result, sample, data_dir)

            index, _ = build_chunk_index(chunks, embedding_func)
            retrieved = search_chunks(sample["question"], chunks, index,
                                      embedding_func, args.top_k)
            answer = generate_rag_answer(answer_client, sample["question"],
                                         retrieved, current_time=None)
            result["generated_answer"] = answer
            result["chunks_retrieved"] = len(retrieved)
            result["llm_score"] = judge.evaluate(
                sample["question"], sample["answer"], answer)
            result["is_correct"] = (result["llm_score"] == 1)
        except Exception as e:
            result["error"] = str(e)[:200]
            result["llm_score"] = 0
            result["is_correct"] = False
        return _finalize(result, sample, data_dir)

    return _run_samples(samples, args, _process_one)

# =============================================================================
# Memory pipeline — MemBuilder (per-context; uses memory_system.MemorySystem)
# =============================================================================

def _eval_memory(samples, args, judge: LLMJudge, data_dir: str) -> List[Dict]:
    """Evaluate MemBuilder memory system on MemoryAgentBench.

    Memory is built once per context and then shared by all questions from that
    context.

    Two optional context-augmentation features:
      1. Memory-Guided RAG (chunk_context): when RAG_CHUNK_TYPES env var is set
         (e.g. "episodic|procedural"), for each retrieved memory of a matching
         type, look up the original session chunk and attach it as supplementary
         context — same mechanism as eval/runner.py for longmemeval/locomo.
      2. Extra RAG (--extra-rag-topk N): perform an independent embedding search
         over the raw session chunks and append the top-N results to the prompt.
         Default 0 = off. This is "方案 A" fusion.
    """
    from eval.datasets import get_current_time
    from eval.runner import (
        build_memories, _make_embedding_func,
        _build_eval_chunk_store, _build_memory_to_session_mapping,
        _build_chunk_context_for_eval,
    )
    from memory_system import MemorySystem

    build_client = create_llm_client(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url or "http://localhost:8000/v1",
        api_key=args.api_key or "EMPTY",
    )
    answer_client = _build_answer_client(args)
    top_k = int(args.top_k)
    session_concurrency = int(getattr(args, "session_concurrency", 1) or 1)
    embedding_func = _make_embedding_func()

    # Memory-Guided RAG config
    from config import RAG_CHUNK_TYPES, CHUNK_TURN_SIZE
    use_chunk_context = bool(RAG_CHUNK_TYPES)

    # Extra standalone RAG config
    extra_rag_topk = int(getattr(args, "extra_rag_topk", 0) or 0)

    print(f"\n   [membuilder] build_model  : {args.model}")
    print(f"   [membuilder] answer_model : {args.answer_model or args.model}")
    print(f"   [membuilder] top_k        : {top_k}")
    print(f"   [membuilder] session_conc : {session_concurrency}")
    if use_chunk_context:
        print(f"   [membuilder] chunk_context: enabled (types={RAG_CHUNK_TYPES})")
    if extra_rag_topk > 0:
        print(f"   [membuilder] extra_rag    : top-{extra_rag_topk}")

    by_ctx: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        by_ctx[sample["context_id"]].append(sample)

    all_results: List[Dict] = []
    total = len(samples)
    done = 0
    correct_count = 0

    for ctx_id, ctx_samples in tqdm(
        list(by_ctx.items()), desc="MemBuilder MAB (ctx)", unit="ctx"
    ):
        user_id = ctx_id.replace("/", "_")
        memory = MemorySystem(build_client, embedding_func=embedding_func)
        build_info = {}

        try:
            build_info = build_memories(
                memory=memory,
                sample=ctx_samples[0],
                user_id=user_id,
                max_sessions=getattr(args, "max_sessions", None),
                verbose=False,
                session_concurrency=session_concurrency,
            )
        except Exception as exc:
            build_info = {"error": str(exc)[:200]}
            print(
                f"[membuilder] build failed for {ctx_id}: {str(exc)[:120]}",
                flush=True,
            )

        memory.llm_client = answer_client

        # Pre-build chunk index for this context (shared by all questions)
        session_chunks = None
        memory_session_map = None
        rag_chunk_index = None
        rag_chunks_flat = None

        if use_chunk_context or extra_rag_topk > 0:
            session_chunks = _build_eval_chunk_store(ctx_samples[0], CHUNK_TURN_SIZE)

        if use_chunk_context and session_chunks:
            memory_session_map = _build_memory_to_session_mapping(memory)

        if extra_rag_topk > 0 and session_chunks:
            from eval.rag_baseline import build_chunk_index, make_embedding_func
            rag_embedding_func = embedding_func or make_embedding_func(answer_client)
            rag_chunks_flat = []
            for sess_idx in sorted(session_chunks.keys()):
                for chunk in session_chunks[sess_idx]:
                    rag_chunks_flat.append({
                        "text": chunk["text"],
                        "session_idx": sess_idx,
                        "chunk_index": chunk["chunk_index"],
                    })
            if rag_chunks_flat:
                rag_chunk_index, _ = build_chunk_index(rag_chunks_flat, rag_embedding_func)

        for sample in ctx_samples:
            done += 1
            result = _base_result(sample, done)
            try:
                if build_info.get("error"):
                    raise RuntimeError(build_info["error"])

                search_start = time.time()
                search_result = memory.search(
                    sample["question"], user_id=user_id, limit=top_k
                )
                search_time = time.time() - search_start
                memories = (
                    search_result.get("results", [])
                    if isinstance(search_result, dict)
                    else []
                )

                # --- Build chunk_context (Memory-Guided RAG) ---
                chunk_context = None
                if use_chunk_context and session_chunks and memory_session_map:
                    chunk_context = _build_chunk_context_for_eval(
                        memories, session_chunks, memory_session_map,
                        RAG_CHUNK_TYPES,
                    )

                # --- Build extra RAG context (standalone supplement) ---
                extra_rag_context = None
                if extra_rag_topk > 0 and rag_chunk_index is not None and rag_chunks_flat:
                    from eval.rag_baseline import search_chunks
                    rag_results = search_chunks(
                        sample["question"], rag_chunks_flat, rag_chunk_index,
                        rag_embedding_func, extra_rag_topk,
                    )
                    if rag_results:
                        rag_texts = []
                        for i, chunk in enumerate(rag_results, 1):
                            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                            rag_texts.append(f"[Excerpt {i}]\n{text}")
                        extra_rag_context = "\n\n".join(rag_texts)

                # --- Combine supplementary context ---
                combined_context = None
                parts = []
                if chunk_context:
                    parts.append(chunk_context)
                if extra_rag_context:
                    parts.append(extra_rag_context)
                if parts:
                    combined_context = "\n\n".join(parts)

                has_chunk = bool(chunk_context or extra_rag_context)

                gen_start = time.time()
                if is_mab_protocol_source(sample.get("source", "")):
                    proto_prompt, proto_max_tokens = build_mab_answering_prompt(
                        source=sample["source"],
                        memories=_memories_to_strings(memories),
                        question=sample["question"],
                    )
                    if combined_context:
                        proto_prompt = proto_prompt.rstrip() + f"\n\n[Supporting Original Context]\n{combined_context}\n"
                    answer = answer_client.chat_completion(
                        messages=[{"role": "user", "content": proto_prompt}],
                        temperature=0.0,
                        max_tokens=proto_max_tokens,
                    )
                else:
                    answer = memory.generate_answer(
                        question=sample["question"],
                        memories=memories,
                        user_id=user_id,
                        current_time=get_current_time(sample),
                        chunk_context=combined_context,
                    )
                gen_time = time.time() - gen_start

                judge_start = time.time()
                llm_score = judge.evaluate(
                    sample["question"], sample["answer"], answer
                )
                judge_time = time.time() - judge_start

                result.update({
                    "generated_answer": answer,
                    "is_correct": llm_score == 1,
                    "llm_score": llm_score,
                    "memories_found": len(memories),
                    "has_chunk_context": has_chunk,
                    "search_time": search_time,
                    "gen_time": gen_time,
                    "judge_time": judge_time,
                    "total_memories": build_info.get("total_memories"),
                    "sessions_processed": build_info.get("sessions_processed"),
                    "build_time": build_info.get("build_time"),
                })
            except Exception as exc:
                result["error"] = str(exc)[:200]
                result["llm_score"] = 0
                result["is_correct"] = False
                if build_info.get("error"):
                    result["build_error"] = build_info["error"]

            result = _finalize(result, sample, data_dir)
            all_results.append(result)
            if result.get("is_correct"):
                correct_count += 1
            status = "✅" if result.get("is_correct") else "❌"
            print(f"[{done}/{total}] {status} {result['question_id']} "
                  f"(acc={correct_count/done*100:.1f}%)", flush=True)

    all_results.sort(key=lambda item: item.get("index", 0))
    return all_results
# Shared helpers
# =============================================================================

def _base_result(sample: Dict, idx: int) -> Dict:
    return {
        "index": idx,
        "question_id": sample["question_id"],
        "question": sample["question"],
        "expected_answer": sample["answer"],
        "expected_answer_list": sample.get("answer_list", [sample["answer"]]),
        "question_type": sample["source"],
        "source": sample["source"],
        "split": sample["split"],
        "context_id": sample["context_id"],
    }


def _finalize(result: Dict, sample: Dict, data_dir: str) -> Dict:
    """Attach official metrics to a result that already has generated_answer."""
    pred = result.get("generated_answer", "") or ""
    try:
        result["official_metrics"] = compute_official_metrics(
            pred, sample.get("answer_list", [sample["answer"]]),
            sample["source"], data_dir=data_dir,
        )
    except Exception as e:
        result["official_metrics"] = {"_error": str(e)[:120]}
    return result


def _run_samples(samples, args, process_fn) -> List[Dict]:
    """Run process_fn over samples (parallel or sequential). Used by RAG."""
    total = len(samples)
    all_results: List[Dict] = []

    if args.parallel and total > 1:
        print(f"🚀 Parallel ({args.workers} workers, {total} samples)")
        correct = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_fn, s, i + 1, total): i + 1
                for i, s in enumerate(samples)
            }
            pbar = tqdm(as_completed(futures), total=total, desc="MemAgentBench")
            for future in pbar:
                try:
                    r = future.result()
                    all_results.append(r)
                    if r.get("is_correct"):
                        correct += 1
                    n = len(all_results)
                    pbar.set_postfix_str(f"acc={correct/n*100:.1f}% ({correct}/{n})")
                except Exception as e:
                    idx = futures[future]
                    print(f"[{idx}/{total}] ❌ FATAL: {str(e)[:80]}")
            pbar.close()
        all_results.sort(key=lambda x: x.get("index", 0))
    else:
        print(f"📝 Sequential ({total} samples)")
        correct = 0
        for i, s in enumerate(samples, 1):
            r = process_fn(s, i, total)
            all_results.append(r)
            if r.get("is_correct"):
                correct += 1
            status = "✅" if r.get("is_correct") else "❌"
            print(f"[{i}/{total}] {status} {s['question_id']} "
                  f"(acc={correct/i*100:.1f}%)", flush=True)

    return all_results


# =============================================================================
# Summary
# =============================================================================

def _aggregate_official(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Average official metrics per source, and overall."""
    by_source: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    overall: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        src = r.get("source", "unknown")
        om = r.get("official_metrics", {}) or {}
        for k, v in om.items():
            if isinstance(v, (int, float)):
                by_source[src][k].append(float(v))
                overall[k].append(float(v))
    out = {"overall": {k: sum(v) / len(v) for k, v in overall.items() if v}}
    for src, md in by_source.items():
        out[src] = {k: sum(v) / len(v) for k, v in md.items() if v}
    return out


def _build_summary(results: List[Dict], args) -> Dict[str, Any]:
    judge_stats = compute_accuracy(results)
    official = _aggregate_official(results)

    # primary official metric per source
    primary = {}
    for src, md in official.items():
        if src == "overall":
            continue
        pm = primary_metric_name(src)
        if pm in md:
            primary[src] = {pm: md[pm]}

    llm_judge_per_source = {
        t: round(acc * 100, 2)
        for t, acc in judge_stats.get("per_type", {}).items()
    }

    table_columns = _compute_table_columns(official, llm_judge_per_source)

    system = getattr(args, "system", "membuilder")

    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": "memoryagentbench",
        "system": system,
        "split": args.split,
        "pipeline": args.pipeline,
        "top_k": args.top_k,
        "chunk_char_size": args.chunk_char_size,
        "memory_model": args.model,
        "answer_model": args.answer_model or args.model,
        "judge_model": args.judge_model,
        "total_samples": judge_stats.get("total", 0),
        "llm_judge_accuracy": {
            "overall_percent": round(judge_stats.get("overall", 0.0) * 100, 2),
            "correct": judge_stats.get("correct", 0),
            "total": judge_stats.get("total", 0),
            "per_source": llm_judge_per_source,
        },
        "official_metrics": official,
        "official_primary_per_source": primary,
        "table_columns": table_columns,
    }


def _compute_table_columns(
    official: Dict[str, Dict[str, float]],
    llm_judge_per_source: Dict[str, float],
) -> Dict[str, Any]:
    """Compute values corresponding to the MemoryAgentBench paper table columns.

    Mapping (verified against the original MAB benchmark README):
      AR:  SH-QA, MH-QA        -> official substring_exact_match * 100
           LME(S*), EventQA     -> llm_judge per_source (open-ended, needs LLM judge)
      TTL: MCC                  -> 5 ICL tasks official substring_exact_match avg * 100
           Recom.               -> official recsys_recall@5 * 100 (Recall@5)
      LRU: Summ.                -> official rougeL_f1 * 100
           DetQA                -> official substring_exact_match * 100
      SF:  FC-SH, FC-MH        -> official factconsolidation_*_262k substring_exact_match * 100
    """

    def _get_official(source: str, metric: str, scale: float = 100.0) -> Optional[float]:
        val = official.get(source, {}).get(metric)
        return round(val * scale, 1) if val is not None else None

    def _get_judge(source: str) -> Optional[float]:
        val = llm_judge_per_source.get(source)
        return round(val, 1) if val is not None else None

    def _avg(values: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None]
        return round(sum(valid) / len(valid), 1) if valid else None

    # --- AR (Accurate Retrieval) ---
    sh_qa = _get_official("ruler_qa1_197K", "substring_exact_match")
    mh_qa = _get_official("ruler_qa2_421K", "substring_exact_match")
    lme = _get_judge("longmemeval_s*")
    eventqa = _get_judge("eventqa_full")
    ar_avg = _avg([sh_qa, mh_qa, lme, eventqa])

    # --- TTL (Test-Time Learning) ---
    icl_tasks = [
        "icl_banking77_5900shot_balance",
        "icl_clinic150_7050shot_balance",
        "icl_nlu_8296shot_balance",
        "icl_trec_coarse_6600shot_balance",
        "icl_trec_fine_6400shot_balance",
    ]
    icl_vals = [_get_official(t, "substring_exact_match") for t in icl_tasks]
    mcc = _avg(icl_vals)
    recom = _get_official("recsys_redial_full", "recsys_recall@5")
    ttl_avg = _avg([mcc, recom])

    # --- LRU (Long-Range Understanding) ---
    summ = _get_official("infbench_sum_eng_shots2", "rougeL_f1")
    det_qa = _get_official("detective_qa", "substring_exact_match")
    lru_avg = _avg([summ, det_qa])

    # --- SF (Selective Forgetting / Conflict Resolution) ---
    # NOTE: factconsolidation_mh_* is excluded because its ground-truth is a
    # synthesized counterfactual chain (e.g. Steve Sax -> association football
    # -> Italy) that penalizes any answer model with strong factual priors;
    # SF.Avg falls back to FC-SH.
    fc_sh = _get_official("factconsolidation_sh_262k", "substring_exact_match")
    fc_mh = None
    sf_avg = _avg([fc_sh])

    # --- Overall ---
    overall = _avg([ar_avg, ttl_avg, lru_avg, sf_avg])

    return {
        "AR": {"SH-QA": sh_qa, "MH-QA": mh_qa, "LME(S*)": lme, "EventQA": eventqa, "Avg": ar_avg},
        "TTL": {"MCC": mcc, "Recom": recom, "Avg": ttl_avg},
        "LRU": {"Summ": summ, "DetQA": det_qa, "Avg": lru_avg},
        "SF": {"FC-SH": fc_sh, "FC-MH": fc_mh, "Avg": sf_avg},
        "Overall": overall,
    }


def _print_table_columns(table_cols: Dict[str, Any]):
    """Print table column values matching the MemoryAgentBench paper table format."""
    print("\n" + "=" * 70)
    print("📋 MemoryAgentBench Table Columns (paper format)")
    print("=" * 70)

    for group_name in ["AR", "TTL", "LRU", "SF"]:
        group = table_cols.get(group_name, {})
        parts = " | ".join(
            f"{col}={val}" if val is not None else f"{col}=N/A"
            for col, val in group.items()
        )
        print(f"  {group_name:4s}: {parts}")

    overall = table_cols.get("Overall")
    print(f"  {'Overall':4s}: {overall}" if overall is not None else "  Overall: N/A")

    # Print as a single LaTeX row for easy copy-paste
    ar = table_cols.get("AR", {})
    ttl = table_cols.get("TTL", {})
    lru = table_cols.get("LRU", {})
    sf = table_cols.get("SF", {})

    def _fmt(v):
        return f"{v}" if v is not None else "—"

    latex_vals = " & ".join([
        _fmt(ar.get("SH-QA")), _fmt(ar.get("MH-QA")), _fmt(ar.get("LME(S*)")),
        _fmt(ar.get("EventQA")), _fmt(ar.get("Avg")),
        _fmt(ttl.get("MCC")), _fmt(ttl.get("Recom")), _fmt(ttl.get("Avg")),
        _fmt(lru.get("Summ")), _fmt(lru.get("DetQA")), _fmt(lru.get("Avg")),
        _fmt(sf.get("FC-SH")), _fmt(sf.get("FC-MH")), _fmt(sf.get("Avg")),
        _fmt(overall),
    ])
    print(f"\n  LaTeX row: & {latex_vals} \\\\")
    print("=" * 70)


def _print_official(official: Dict[str, Dict[str, float]]):
    print("\n📊 官方指标 (official metrics):")
    for src in sorted(official.keys()):
        if src == "overall":
            continue
        md = official[src]
        parts = ", ".join(f"{k}={v:.4f}" for k, v in sorted(md.items()))
        print(f"   [{src}] {parts}")
    if "overall" in official:
        parts = ", ".join(f"{k}={v:.4f}" for k, v in sorted(official["overall"].items()))
        print(f"   [overall] {parts}")


# =============================================================================
# Main
# =============================================================================

def run(args) -> int:
    system = getattr(args, "system", "membuilder")
    print("=" * 60)
    print("🧪 MemoryAgentBench Evaluation")
    print(f"   data_dir : {args.data_dir}")
    print(f"   system   : {system}")
    print(f"   split    : {args.split}")
    print(f"   pipeline : {args.pipeline}")
    print(f"   top_k    : {args.top_k}")
    print(f"   chunk    : {args.chunk_char_size} tokens/pseudo-session")
    print(f"   judge    : {args.judge_model}")
    print("=" * 60)

    # Resolve splits (accept "+" or "|" as separator for multi-split)
    raw_split = args.split
    if raw_split == "all":
        splits = list(SPLITS)
    elif "+" in raw_split or "|" in raw_split:
        sep = "+" if "+" in raw_split else "|"
        splits = [s.strip() for s in raw_split.split(sep) if s.strip()]
        unknown = [s for s in splits if s not in SPLITS]
        if unknown:
            print(f"ERROR: unknown split(s) in --split: {unknown}. "
                  f"Supported: {SPLITS} or 'all'.")
            return 1
    else:
        if raw_split not in SPLITS:
            print(f"ERROR: unknown split: {raw_split}. "
                  f"Supported: {SPLITS} or 'all' or '+'-separated list.")
            return 1
        splits = [raw_split]

    # Load samples across splits
    samples: List[Dict] = []
    for sp in splits:
        part = load_memoryagentbench(
            args.data_dir, sp, limit=None, chunk_char_size=args.chunk_char_size)
        samples.extend(part)

    # Drop sources excluded from headline metrics (factconsolidation_mh_*).
    # These have synthesized counterfactual chains that systematically penalize
    # any answer model with strong factual priors, and are not counted in
    # table_columns. Skip them to save inference cost.
    if not getattr(args, "keep_excluded_sources", False):
        before_excl = len(samples)
        samples = [
            s for s in samples
            if not str(s.get("source", "")).startswith("factconsolidation_mh_")
        ]
        dropped = before_excl - len(samples)
        if dropped:
            print(f"[exclude-sources] Dropped {dropped} samples from "
                  f"factconsolidation_mh_* "
                  f"(use --keep-excluded-sources to retain)")

    # Filter by splits-file (context-level RL/test partition)
    if getattr(args, "splits_file", None):
        with open(args.splits_file, "r", encoding="utf-8") as f:
            splits_partition = json.load(f)
        eval_key = getattr(args, "eval_split_key", "test")
        allowed_contexts = set(splits_partition.get(eval_key, []))
        before_count = len(samples)
        samples = [s for s in samples if s["context_id"] in allowed_contexts]
        print(f"[splits-file] Filtered by '{eval_key}': {before_count} -> {len(samples)} samples "
              f"({len(allowed_contexts)} allowed contexts)")

    if args.limit:
        samples = samples[:args.limit]
    print(f"\n📚 Loaded {len(samples)} per-question samples from {splits}")

    # Multi-worker data sharding (context-level to keep same context on one worker)
    worker_rank = getattr(args, "worker_rank", None)
    worker_count = getattr(args, "worker_count", None)
    if worker_rank is not None and worker_count is not None and worker_count > 1:
        all_contexts = list(dict.fromkeys(s["context_id"] for s in samples))
        shard_contexts = set(all_contexts[worker_rank::worker_count])
        full_count = len(samples)
        samples = [s for s in samples if s["context_id"] in shard_contexts]
        print(f"   🔀 Worker {worker_rank}/{worker_count}: processing "
              f"{len(samples)}/{full_count} samples "
              f"({len(shard_contexts)}/{len(all_contexts)} contexts)")

    if not samples:
        print("ERROR: no samples loaded.")
        return 1

    judge = _build_judge(args)

    if args.pipeline == "rag":
        results = _eval_rag(samples, args, judge, args.data_dir)
    elif args.pipeline == "memory":
        results = _eval_memory(samples, args, judge, args.data_dir)
    else:
        raise ValueError(f"unknown pipeline: {args.pipeline}")

    # Summary & output
    is_sharded = (worker_rank is not None and worker_count is not None and worker_count > 1)

    summary = _build_summary(results, args)
    official = summary["official_metrics"]

    timestamp = summary["timestamp"]
    out_dir = Path(args.output_dir) / f"memoryagentbench_{args.pipeline}_{args.split}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_sharded:
        shard_file = out_dir / f"results_rank{worker_rank}.json"
        with open(shard_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print_accuracy_stats(results,
                             f"📊 Worker {worker_rank}/{worker_count} 局部结果")
        print(f"\n💾 Shard 结果保存到: {shard_file}")
    else:
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print_accuracy_stats(results, "📊 LLM-Judge 准确率")
        _print_official(official)
        _print_table_columns(summary["table_columns"])
        print(f"\n💾 结果保存到: {out_dir}/")
    return 0


def main():
    parser = argparse.ArgumentParser(description="MemoryAgentBench evaluation runner")

    # Dataset
    parser.add_argument("--data-dir",
                        default="./data/MemoryAgentBench",
                        help="MemoryAgentBench dataset root (contains data/)")
    parser.add_argument("--split", default="Accurate_Retrieval",
                        help=("Which split(s) to evaluate. Accepts a single "
                              "split name, 'all', or a '+'-separated list "
                              "(e.g. 'Test_Time_Learning+Long_Range_Understanding'). "
                              "'|' also accepted but unsafe in shell pipelines."))
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap total per-question samples (debug)")
    parser.add_argument("--chunk-char-size", type=int, default=4096,
                        dest="chunk_char_size",
                        help="Token chunk size for context -> pseudo session")

    # Pipeline
    parser.add_argument("--pipeline", choices=["rag", "memory"], default="rag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-turn-size", type=int, default=1,
                        dest="chunk_turn_size",
                        help="(rag) turns per chunk when chunking pseudo sessions")
    parser.add_argument("--extra-rag-topk", type=int, default=0,
                        dest="extra_rag_topk",
                        help="(memory) extra standalone RAG supplement: retrieve top-N "
                             "raw chunks by embedding similarity and append to prompt. "
                             "Default 0 = off. Set to 5 or 10 to enable fusion.")

    # memory pipeline: checkpoint backend
    parser.add_argument("--provider", default="vllm",
                        help="(memory) chat provider for the checkpoint")
    parser.add_argument("--model", default=None,
                        help="(memory) checkpoint model name/path served by vLLM")
    parser.add_argument("--base-url", default=None, dest="base_url")
    parser.add_argument("--api-key", default=None, dest="api_key")
    parser.add_argument("--max-sessions", type=int, default=None, dest="max_sessions",
                        help="(memory) cap sessions per context (debug / cost)")
    parser.add_argument("--session-concurrency", type=int, default=8,
                        dest="session_concurrency")

    # answer / judge API models
    parser.add_argument("--answer-model", default=None, dest="answer_model")
    parser.add_argument("--answer-base-url", default=None, dest="answer_base_url")
    parser.add_argument("--answer-api-key", default=None, dest="answer_api_key")
    parser.add_argument("--judge-model", default=None, dest="judge_model")
    parser.add_argument("--judge-base-url", default=None, dest="judge_base_url")
    parser.add_argument("--judge-api-key", default=None, dest="judge_api_key")

    # Splits-based filtering
    parser.add_argument("--splits-file", default=None, dest="splits_file",
                        help="Path to memoryagentbench_splits.json (keys: rl, test). "
                             "When provided, only contexts in the chosen key are evaluated.")
    parser.add_argument("--eval-split-key", default="test", dest="eval_split_key",
                        choices=["rl", "test"],
                        help="Which key in --splits-file to use (default: test)")
    parser.add_argument("--keep-excluded-sources", action="store_true",
                        dest="keep_excluded_sources",
                        help="Keep factconsolidation_mh_* sources "
                             "(excluded by default because their ground-truth "
                             "uses synthesized counterfactual chains that "
                             "penalize answer models with strong factual "
                             "priors and are not counted in table_columns).")

    # Multi-worker data sharding
    parser.add_argument("--worker-rank", type=int, default=None, dest="worker_rank",
                        help="Current worker rank (0-indexed). Data is sharded across workers.")
    parser.add_argument("--worker-count", type=int, default=None, dest="worker_count",
                        help="Total number of workers. Used with --worker-rank for data sharding.")

    # Concurrency / output
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", default="logs", dest="output_dir")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Defaults for answer/judge models
    if args.answer_model is None:
        args.answer_model = os.environ.get("MEMBUILDER_INPROC_ANSWER_MODEL", "qwen3.7-max")
    if args.judge_model is None:
        args.judge_model = os.environ.get("MEMBUILDER_INPROC_JUDGE_MODEL", "qwen3.7-max")

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
