"""
MemoryAgentBench dataset loader + official metrics (standalone module).

This module is fully self-contained and does NOT modify any existing eval code
(eval/datasets.py, eval/metrics.py). It adapts the MemoryAgentBench parquet
dataset to the per-question sample format used by the existing eval framework,
and provides the official MemoryAgentBench metrics (substring / exact_match /
f1 / ROUGE / recsys recall) routed by the dataset `source`.

Dataset structure (4 parquet splits under <data_dir>/data/):
  - context  (str)          : a single long document
  - questions (array[str])  : multiple questions for the context
  - answers  (array[array]) : multiple reference answers per question
  - metadata (dict)         : qa_pair_ids, source, haystack_sessions(optional), ...

Most samples only have a long-document `context` (no haystack_sessions). We
chunk the context into fixed-token "pseudo sessions" so that the existing
memory-building / RAG pipeline (which expects haystack_sessions) can consume it.

The official metric functions below are ported (self-contained copies) from
MemoryAgentBench utils/eval_other_utils.py to avoid depending on the external
repo path at runtime.
"""

import json
import os
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Split -> parquet file mapping
# =============================================================================

SPLITS = [
    "Accurate_Retrieval",
    "Test_Time_Learning",
    "Long_Range_Understanding",
    "Conflict_Resolution",
]

_PARQUET_TEMPLATE = "{split}-00000-of-00001.parquet"


# =============================================================================
# Context chunking (token-based, sentence-preserving) -> pseudo sessions
# =============================================================================

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences on .!? boundaries followed by whitespace.

    Lightweight stdlib-only replacement for nltk.sent_tokenize. Good enough
    for English documents and avoids the nltk + punkt-data dependency.
    """
    sentences = _SENTENCE_BOUNDARY_RE.split(text.strip())
    return [s for s in sentences if s]


def _chunk_text_by_tokens(text: str, chunk_size: int = 4096,
                          model_name: str = "gpt-4o-mini") -> List[str]:
    """Split a long text into chunks of ~chunk_size tokens, preserving sentence
    boundaries. Mirrors MemoryAgentBench chunk_text_into_sentences, but uses a
    regex sentence splitter instead of nltk to stay lightweight.

    Raises if tiktoken is unavailable (no silent char-based fallback).
    """
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    sentences = _split_into_sentences(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for sentence in sentences:
        n = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))
        if cur_tok + n > chunk_size and cur:
            chunks.append(" ".join(cur))
            cur = [sentence]
            cur_tok = n
        else:
            cur.append(sentence)
            cur_tok += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _context_to_pseudo_sessions(context: str, chunk_char_size: int) -> List[List[Dict[str, str]]]:
    """Convert a long context into a list of pseudo sessions.

    Each chunk becomes one pseudo session containing a single user message:
        [{"role": "user", "content": chunk_text}]

    `chunk_char_size` here is interpreted as the token chunk size passed to the
    token-based splitter (kept the name aligned with the CLI flag for clarity).
    """
    chunks = _chunk_text_by_tokens(context, chunk_size=chunk_char_size)
    sessions = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk:
            sessions.append([{"role": "user", "content": chunk}])
    return sessions


def _normalize_haystack_sessions(raw_sessions: Any) -> List[List[Dict[str, str]]]:
    """Normalize an existing haystack_sessions structure into a list of sessions,
    each a list of {"role","content"} messages.

    MemoryAgentBench haystack_sessions is a list of sessions, each session a list
    of turns with keys content/role/has_answer.
    """
    sessions: List[List[Dict[str, str]]] = []
    for session in raw_sessions:
        msgs = []
        for turn in session:
            if isinstance(turn, dict):
                role = turn.get("role", "user") or "user"
                content = turn.get("content", "") or ""
            else:
                role, content = "user", str(turn)
            if content:
                msgs.append({"role": role, "content": content})
        if msgs:
            sessions.append(msgs)
    return sessions


# =============================================================================
# Loader
# =============================================================================

def load_memoryagentbench(
    data_dir: str,
    split: str,
    limit: Optional[int] = None,
    chunk_char_size: int = 4096,
) -> List[Dict[str, Any]]:
    """Load a MemoryAgentBench split and flatten to per-question samples.

    Args:
        data_dir: Root dir of the MemoryAgentBench dataset (contains data/).
        split: One of SPLITS.
        limit: Optional cap on number of per-question samples returned.
        chunk_char_size: Token chunk size for context -> pseudo session.

    Returns:
        List of per-question sample dicts with keys:
          question_id, question, answer, answer_list, question_type, source,
          split, haystack_sessions, haystack_session_datetimes, context_id
    """
    import pandas as pd

    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}. Supported: {SPLITS}")

    parquet_path = Path(data_dir) / "data" / _PARQUET_TEMPLATE.format(split=split)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    samples: List[Dict[str, Any]] = []

    for ctx_idx in range(len(df)):
        row = df.iloc[ctx_idx]
        context = row["context"]
        questions = list(row["questions"])
        answers = list(row["answers"])
        md = row["metadata"] if row["metadata"] is not None else {}

        source = md.get("source", split)
        qa_pair_ids = md.get("qa_pair_ids")
        qa_pair_ids = list(qa_pair_ids) if qa_pair_ids is not None else None
        raw_haystack = md.get("haystack_sessions")

        # Build sessions once per context (shared by all questions of this row)
        if raw_haystack is not None:
            haystack_sessions = _normalize_haystack_sessions(raw_haystack)
        else:
            haystack_sessions = _context_to_pseudo_sessions(context, chunk_char_size)

        context_id = f"{split}__{source}__ctx{ctx_idx}"

        n = min(len(questions), len(answers))
        for qi in range(n):
            q = questions[qi]
            ans_raw = answers[qi]
            # answers[qi] may be an ndarray / list of reference answers
            if hasattr(ans_raw, "tolist"):
                answer_list = [str(a) for a in ans_raw.tolist()]
            elif isinstance(ans_raw, (list, tuple)):
                answer_list = [str(a) for a in ans_raw]
            else:
                answer_list = [str(ans_raw)]
            answer_list = [a for a in answer_list if a is not None]
            main_answer = answer_list[0] if answer_list else ""

            if qa_pair_ids is not None and qi < len(qa_pair_ids):
                qid = str(qa_pair_ids[qi])
            else:
                qid = f"{source}_q{qi}"

            samples.append({
                "question_id": qid,
                "question": str(q),
                "answer": main_answer,
                "answer_list": answer_list,
                "question_type": source,
                "source": source,
                "split": split,
                "context_id": context_id,
                "haystack_sessions": haystack_sessions,
                "haystack_session_datetimes": [],
            })

    if limit:
        samples = samples[:limit]

    return samples


# =============================================================================
# Official metrics (self-contained port from MemoryAgentBench eval_other_utils)
# =============================================================================

def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _f1_score(prediction: str, ground_truth: str):
    np_, ng_ = _normalize_answer(prediction), _normalize_answer(ground_truth)
    ZERO = (0.0, 0.0, 0.0)
    special = {"yes", "no", "noanswer"}
    if (np_ in special or ng_ in special) and np_ != ng_:
        return ZERO
    p_tokens = np_.split()
    g_tokens = ng_.split()
    common = Counter(p_tokens) & Counter(g_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return ZERO
    precision = num_common / len(p_tokens)
    recall = num_common / len(g_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def _drqa_exact_match(prediction: str, ground_truth: str) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _substring_exact_match(prediction: str, ground_truth: str) -> bool:
    return _normalize_answer(ground_truth) in _normalize_answer(prediction)


def _max_over_ground_truths(metric_fn, prediction: str, ground_truths) -> float:
    if isinstance(ground_truths, str):
        gt_list = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        gt_list = [gt for sub in ground_truths for gt in sub]
    else:
        gt_list = list(ground_truths)
    if not gt_list:
        return 0.0
    return max(float(metric_fn(prediction, gt)) for gt in gt_list)


_rouge_scorer_instance = None


def _get_rouge_scorer():
    global _rouge_scorer_instance
    if _rouge_scorer_instance is None:
        from rouge_score import rouge_scorer
        _rouge_scorer_instance = rouge_scorer.RougeScorer(
            ["rougeL", "rougeLsum"], use_stemmer=True
        )
    return _rouge_scorer_instance


def _rouge_metrics(prediction: str, answer_list: List[str]) -> Dict[str, float]:
    scorer = _get_rouge_scorer()
    scores = [scorer.score(target=a, prediction=prediction) for a in answer_list]
    out = {}
    for rouge_type in scorer.rouge_types:
        out[rouge_type + "_f1"] = max(s[rouge_type].fmeasure for s in scores)
        out[rouge_type + "_recall"] = max(s[rouge_type].recall for s in scores)
    return out


# ── recsys helpers (ported) ──────────────────────────────────────────────────

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_parentheses(text: str) -> str:
    return re.sub(r"\([^()]*\)", "", text)


def _clean_text_elements(text: str) -> str:
    text = re.sub(r"\([^()]*\)", "", text)
    text = re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_movie_name(text: str) -> str:
    filename = text.split("/")[-1]
    cleaned = filename.replace("_", " ").replace("-", " ").replace(">", " ")
    return _normalize_whitespace(_clean_parentheses(cleaned))


def _find_nearest_movie(target_name: str, candidates: List[str]) -> Dict[str, Any]:
    from editdistance import eval as edit_distance
    import numpy as np
    uniq = list(set(candidates))
    distances = [edit_distance(target_name.lower(), c.lower()) for c in uniq]
    idx = int(np.argmin(distances))
    return {"movie_name": target_name, "nearest_movie": uniq[idx]}


def _extract_recommendation_list(text: str, movie_candidates: List[str]):
    try:
        _, rec_text = text.split("1.", maxsplit=1)
    except Exception:
        rec_text = text.replace(",", "\n")
    raw = [_clean_text_elements(item.strip()) for item in rec_text.split("\n")]
    raw = [r for r in raw if r]
    return [_find_nearest_movie(item, movie_candidates) for item in raw]


_RECSYS_ENTITY_MAP = None


def _load_recsys_id_to_name(data_dir: str) -> Optional[Dict[int, str]]:
    """Load entity2id.json and build id->movie_name. Returns None if unavailable."""
    global _RECSYS_ENTITY_MAP
    if _RECSYS_ENTITY_MAP is not None:
        return _RECSYS_ENTITY_MAP
    candidates = [
        os.path.join(data_dir, "entity2id.json"),
        "./processed_data/Recsys_Redial/entity2id.json",
    ]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        return None
    try:
        name_to_id = json.load(open(path))
        _RECSYS_ENTITY_MAP = {
            int(eid): _extract_movie_name(name) for name, eid in name_to_id.items()
        }
        return _RECSYS_ENTITY_MAP
    except Exception:
        return None


def _recsys_metrics(prediction: str, answer_list: List[str], data_dir: str) -> Dict[str, float]:
    id_to_name = _load_recsys_id_to_name(data_dir)
    if not id_to_name:
        return {"recsys_recall@1": 0.0, "recsys_recall@5": 0.0,
                "recsys_recall@10": 0.0, "recsys_unavailable": 1.0}
    movie_candidates = list(id_to_name.values())
    predicted = _extract_recommendation_list(prediction, movie_candidates)
    predicted_movies = [item["nearest_movie"] for item in predicted]

    try:
        gt_ids = [int(str(a).strip()) for a in answer_list]
        gt_movies = [id_to_name[mid] for mid in gt_ids if mid in id_to_name]
    except Exception:
        gt_movies = []
    if not gt_movies:
        return {"recsys_recall@1": 0.0, "recsys_recall@5": 0.0, "recsys_recall@10": 0.0}

    def recall_at(k):
        topk = predicted_movies[:k]
        return sum(m in topk for m in gt_movies) / len(gt_movies)

    return {
        "recsys_recall@1": recall_at(1),
        "recsys_recall@5": recall_at(5),
        "recsys_recall@10": recall_at(10),
    }


# ── source routing ───────────────────────────────────────────────────────────

def compute_official_metrics(
    prediction: str,
    answer_list: List[str],
    source: str,
    data_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Compute MemoryAgentBench official metrics routed by `source`.

    Args:
        prediction: model generated answer text.
        answer_list: list of reference answers (multiple gold per question).
        source: dataset source string (e.g. ruler_qa1_197K, infbench_sum_eng_shots2,
                recsys_redial_full, factconsolidation_mh_6k).
        data_dir: dataset root (needed for recsys entity2id.json).

    Returns:
        dict of metric_name -> float value.
    """
    pred = prediction or ""
    src = (source or "").lower()
    if not answer_list:
        answer_list = [""]

    # recsys (Test-Time Learning recommendation)
    if "recsys" in src:
        return _recsys_metrics(pred, answer_list, data_dir or ".")

    metrics: Dict[str, float] = {}

    # ROUGE-heavy tasks: summarization / eventqa
    if "infbench_sum" in src or "eventqa" in src or "sum" in src:
        metrics.update(_rouge_metrics(pred, answer_list))
        # eventqa also reports element recall
        if "eventqa" in src:
            denom = len(answer_list) if answer_list else 1
            recall_score = sum(a.lower() in pred.lower() for a in answer_list) / denom
            metrics["eventqa_recall"] = float(int(recall_score == 1))
        return metrics

    # ruler niah: element recall
    if "ruler_niah" in src:
        denom = len(answer_list) if answer_list else 1
        recall_score = sum(a.lower() in pred.lower() for a in answer_list) / denom
        return {"ruler_recall": recall_score}

    # Default: exact_match + f1 + substring + rougeL
    # (covers ruler_qa, longmemeval, factconsolidation, and unknowns)
    metrics["exact_match"] = _max_over_ground_truths(
        lambda p, g: 1.0 if _drqa_exact_match(p, g) else 0.0, pred, answer_list)
    metrics["substring_exact_match"] = _max_over_ground_truths(
        lambda p, g: 1.0 if _substring_exact_match(p, g) else 0.0, pred, answer_list)
    metrics["f1"] = _max_over_ground_truths(
        lambda p, g: _f1_score(p, g)[0], pred, answer_list)
    try:
        rouge = _rouge_metrics(pred, answer_list)
        metrics["rougeL_f1"] = rouge.get("rougeL_f1", 0.0)
    except Exception:
        pass
    return metrics


# Convenience: which metric is the "primary" headline metric for a source.
def primary_metric_name(source: str) -> str:
    src = (source or "").lower()
    if "recsys" in src:
        return "recsys_recall@5"
    if "infbench_sum" in src or "sum" in src:
        return "rougeL_f1"
    if "eventqa" in src:
        return "eventqa_recall"
    if "ruler_niah" in src:
        return "ruler_recall"
    return "substring_exact_match"
