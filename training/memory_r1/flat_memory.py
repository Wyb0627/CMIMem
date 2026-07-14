"""Flat single-memory state, Memory-R1 response parsing, and EM scoring."""
import json
import re
from typing import Any, Dict, List

_VALID_EVENTS = {"ADD", "UPDATE", "DELETE", "NONE", "NOOP"}


def _strip_think_and_fences(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return ""


def parse_memory_r1_response(raw: str) -> List[Dict[str, Any]]:
    """Parse a Memory-R1 manager response into a list of operation dicts.

    Returns [] on any malformed / empty input (so the reward penalizes it).
    """
    if not raw:
        return []
    cleaned = _strip_think_and_fences(raw)
    blob = _extract_first_json_object(cleaned)
    if not blob:
        return []
    try:
        obj = json.loads(blob)
    except Exception:
        return []
    mem = obj.get("memory") if isinstance(obj, dict) else None
    if not isinstance(mem, list):
        return []
    ops: List[Dict[str, Any]] = []
    for item in mem:
        if not isinstance(item, dict):
            continue
        event = str(item.get("event", "")).upper().strip()
        if event not in _VALID_EVENTS:
            continue
        ops.append(
            {
                "id": str(item.get("id", "")),
                "text": str(item.get("text", "")),
                "event": "NONE" if event == "NOOP" else event,
                "old_memory": str(item.get("old_memory", "")),
            }
        )
    return ops


def extract_answer_span(text: str) -> str:
    """Extract the short answer after '**Answer:**'. Falls back to last line."""
    if not text:
        return ""
    m = re.search(r"\*\*Answer:\*\*\s*(.+?)(?:\n|$)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


class FlatMemoryState:
    """Single flat memory bank keyed by string id (Memory-R1 style).

    Internal store: list of {id, text}. UPDATE/DELETE match by id first, then
    fall back to exact old_memory text match. UPDATE with no matching target
    degrades to ADD (paper-tolerant). NONE/NOOP is a no-op.
    """

    def __init__(self, initial_texts: List[str] = None):
        self._items: List[Dict[str, str]] = []
        self._next_id = 0
        for t in initial_texts or []:
            if t and str(t).strip():
                self._items.append({"id": str(self._next_id), "text": str(t).strip()})
                self._next_id += 1

    def _find_index(self, op: Dict[str, Any]) -> int:
        op_id = str(op.get("id", ""))
        for i, it in enumerate(self._items):
            if op_id and it["id"] == op_id:
                return i
        old = str(op.get("old_memory", "")).strip()
        if old:
            for i, it in enumerate(self._items):
                if it["text"].strip() == old:
                    return i
        return -1

    def _add(self, text: str) -> None:
        self._items.append({"id": str(self._next_id), "text": text})
        self._next_id += 1

    def apply(self, ops: List[Dict[str, Any]]) -> None:
        for op in ops:
            event = str(op.get("event", "NONE")).upper()
            text = str(op.get("text", "")).strip()
            if event == "ADD":
                if text:
                    self._add(text)
            elif event == "UPDATE":
                idx = self._find_index(op)
                if idx >= 0 and text:
                    self._items[idx]["text"] = text
                elif text:
                    self._add(text)
            elif event == "DELETE":
                idx = self._find_index(op)
                if idx >= 0:
                    self._items.pop(idx)
            # NONE / NOOP: no-op

    def to_list(self) -> List[str]:
        return [it["text"] for it in self._items]


def apply_memory_r1_actions(initial_texts: List[str], ops: List[Dict[str, Any]]) -> List[str]:
    st = FlatMemoryState(initial_texts)
    st.apply(ops)
    return st.to_list()


def score_answer_em(prediction: str, gold: str, mode: str = "em") -> float:
    """Normalized Exact Match (mode='em') or substring EM (mode='subem').

    Reuses verl.utils.reward_score.search_r1_like_qa_em for SQuAD-style
    normalization. Extracts the **Answer:** span from the prediction first.
    """
    from verl.utils.reward_score.search_r1_like_qa_em import em_check, subem_check

    pred = extract_answer_span(prediction)
    if mode == "subem":
        return float(subem_check(pred, [gold]))
    return float(em_check(pred, [gold]))
