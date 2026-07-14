"""Load raw sessions from agent_calls.jsonl and the prior flat memory bank
from a state snapshot. Shared by data prep and the reward.
"""
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

_TYPE_PREFIXES = ("[CORE]", "[EPISODIC]", "[SEMANTIC]", "[PROCEDURAL]")


def conv_dir_from_state_path(state_before_path: str) -> str:
    """Derive the conversation dir (which holds agent_calls.jsonl)."""
    if "/states/" in state_before_path:
        return state_before_path.rsplit("/states/", 1)[0]
    return os.path.dirname(os.path.dirname(state_before_path))


def _resolve(path: str) -> str:
    """Resolve a possibly-relative path against the repo root (as reward does)."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cand = os.path.join(repo_root, path)
    return cand if os.path.exists(cand) else path


@lru_cache(maxsize=256)
def _load_agent_calls(conv_dir: str) -> Tuple[Tuple[int, str], ...]:
    """Return ((session_index, new_session_json), ...) for the 'core' rows.

    Cached per conversation dir. Session text is identical across the 4 agents,
    so only the core row is needed.
    """
    path = _resolve(os.path.join(conv_dir, "agent_calls.jsonl"))
    out = []
    if not os.path.exists(path):
        return tuple()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if str(rec.get("agent_type", "")).lower() != "core":
                continue
            new_session = (rec.get("input") or {}).get("new_session")
            out.append((int(rec.get("session_index", -1)), json.dumps(new_session)))
    return tuple(out)


def load_session_messages(state_before_path: str, session_index: int) -> Optional[Dict]:
    """Return the new_session dict {timestamp, messages:[...]} for this session."""
    conv_dir = conv_dir_from_state_path(state_before_path)
    for idx, ns_json in _load_agent_calls(conv_dir):
        if idx == session_index:
            try:
                return json.loads(ns_json)
            except Exception:
                return None
    return None


def format_session_text(messages: List[Dict]) -> str:
    """Render session turns as '[role] content' lines."""
    parts = []
    for m in messages or []:
        role = str(m.get("role", "user"))
        content = str(m.get("content", ""))
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def _strip_prefix(text: str) -> str:
    s = str(text)
    for pref in _TYPE_PREFIXES:
        if s.startswith(pref):
            return s[len(pref):].strip()
    return s.strip()


def load_flat_memory_before(state_before_path: str) -> List[str]:
    """Load the prior flat memory list from the on-disk state snapshot.

    Reads core_memory.json (human string) + vector_store/payload.json (memories
    list), strips the [TYPE] prefixes so the Memory-R1 single-bank view is
    prefix-free. Returns [] if the snapshot is missing/empty (e.g. state_0).
    """
    resolved = _resolve(state_before_path)
    if not resolved or not os.path.exists(resolved):
        return []
    texts: List[str] = []
    core_path = os.path.join(resolved, "core_memory.json")
    if os.path.exists(core_path):
        try:
            with open(core_path, "r", encoding="utf-8") as f:
                human = json.load(f).get("human", "") or ""
            if human.strip():
                texts.append(human.strip())
        except Exception:
            pass
    payload_path = os.path.join(resolved, "vector_store", "payload.json")
    if os.path.exists(payload_path):
        try:
            with open(payload_path, "r", encoding="utf-8") as f:
                for m in json.load(f).get("memories", []) or []:
                    s = _strip_prefix(m)
                    if s:
                        texts.append(s)
        except Exception:
            pass
    return texts
