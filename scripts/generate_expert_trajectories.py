#!/usr/bin/env python3
"""
Generate Expert Memory Building Trajectories for SFT Training.

This script generates expert memory construction trajectories using a strong
expert model (e.g., Claude 4.5 Sonnet) for supervised fine-tuning.

Output structure:
expert_trajectories/{dataset}/{sample_id}/
├── states/          # Memory state snapshots before each session
├── agent_calls.jsonl # Call records for 4 agents (Core, Episodic, Semantic, Procedural)
└── metadata.json

Usage:
    # Single conversation (LoCoMo)
    python generate_expert_trajectories.py \
        --dataset locomo \
        --conv-id conv-26 \
        --expert-model claude-sonnet-4-5

    # Using predefined split (LongMemEval)
    python generate_expert_trajectories.py \
        --dataset longmemeval \
        --split sft \
        --expert-model claude-sonnet-4-5 \
        --parallel --workers 4
"""

import argparse
import json
import sys
import threading
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_system import MemorySystem
from config import SFT_EXPERT_MODEL

# Client module: internal version adds extra providers; public version has openai/vllm only.
try:
    from llm_client_internal import create_llm_client as _factory_create_client, AVAILABLE_PROVIDERS
except ImportError:
    from llm_client import create_llm_client as _factory_create_client, AVAILABLE_PROVIDERS

# Provider factory: creates the appropriate LLM client
_LLM_CLIENT_PROVIDER = "openai"  # default; overridden via --provider CLI arg

# =============================================================================
# Logging verbosity gate.
# =============================================================================
# With WORKERS=16 the legacy per-session/per-agent prints flood the terminal
# (~hundreds of lines per minute). In quiet mode (default) we suppress them
# and rely on the ProgressTracker for at-a-glance status. Errors and the
# LLMClient backoff warnings always remain visible.
#
#   LOG_VERBOSE=0 (default)  -> progress + errors only
#   LOG_VERBOSE=1            -> verbose (every session start, etc.)
def _verbose() -> bool:
    return os.environ.get("LOG_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}

def _vprint(msg: str) -> None:
    """Print only when LOG_VERBOSE=1."""
    if _verbose():
        print(msg, flush=True)


def _create_llm_client(model: str):
    """Create LLM client based on the global provider setting."""
    return _factory_create_client(provider=_LLM_CLIENT_PROVIDER, model=model)


class _ProgressTracker:
    """Thread-safe progress reporter for batch trajectory generation.

    Prints a single-line, in-place updating status while workers churn through
    conversations. On every conv completion (or skip/failure) we render:

        [12/50  ✅10 ⏭ 1 ❌1]  avg 6.4min/conv  ETA 1h 23m  last: ✅ <conv_id>

    The header (printed once via ``start()``) explains the icons. The status
    line uses ``\\r`` so it overwrites itself; permanent messages (errors)
    print on a new line via ``log()`` which preserves the in-place line.
    """

    _ICONS = {"completed": "✅", "skipped": "⏭", "failed": "❌"}

    def __init__(self, total: int, label: str = "trajectories"):
        self.total = max(0, int(total))
        self.label = label
        self.lock = threading.Lock()
        self.start_ts = time.time()
        self.counts = {"completed": 0, "skipped": 0, "failed": 0}
        self.done = 0
        self._last_line_len = 0
        self._tty = sys.stdout.isatty()

    def start(self) -> None:
        if self.total == 0:
            return
        print(
            f"[progress] generating {self.total} {self.label} "
            f"(✅ done | ⏭ skipped | ❌ failed)",
            flush=True,
        )

    def _format_eta(self) -> str:
        if self.done == 0:
            return "?"
        elapsed = time.time() - self.start_ts
        rate = elapsed / self.done  # seconds per conv
        remaining = max(0, self.total - self.done) * rate
        if remaining < 90:
            return f"{remaining:.0f}s"
        if remaining < 3600:
            return f"{remaining / 60:.1f}m"
        h, m = divmod(int(remaining // 60), 60)
        return f"{h}h {m}m"

    def _format_avg(self) -> str:
        completed = self.counts["completed"]
        if completed == 0:
            return "?"
        # Use only "completed" convs for per-conv timing -- skipped/failed
        # finish near-instantly and would skew the estimate downward.
        avg = (time.time() - self.start_ts) / completed
        if avg < 60:
            return f"{avg:.0f}s/conv"
        return f"{avg / 60:.1f}m/conv"

    def update(self, status: str, conv_id: str) -> None:
        with self.lock:
            if status not in self.counts:
                status = "failed"
            self.counts[status] += 1
            self.done += 1
            icon = self._ICONS[status]
            line = (
                f"[{self.done}/{self.total}  "
                f"✅{self.counts['completed']} "
                f"⏭{self.counts['skipped']} "
                f"❌{self.counts['failed']}]  "
                f"avg {self._format_avg()}  "
                f"ETA {self._format_eta()}  "
                f"last: {icon} {conv_id}"
            )
            self._render(line, transient=self._tty)

    def log(self, msg: str) -> None:
        """Print a permanent line (e.g. error) without breaking the in-place
        progress line."""
        with self.lock:
            if self._tty and self._last_line_len:
                # Clear the current transient line, write the permanent
                # message, then re-render the latest progress.
                sys.stdout.write("\r" + " " * self._last_line_len + "\r")
                sys.stdout.write(msg + "\n")
                sys.stdout.flush()
                # No memory of last status line content here -- caller will
                # re-emit on next update().
                self._last_line_len = 0
            else:
                print(msg, flush=True)

    def _render(self, line: str, transient: bool) -> None:
        if transient:
            # \r-overwrite previous line; pad with spaces if shorter.
            pad = " " * max(0, self._last_line_len - len(line))
            sys.stdout.write("\r" + line + pad)
            sys.stdout.flush()
            self._last_line_len = len(line)
        else:
            print(line, flush=True)
            self._last_line_len = 0

    def finish(self) -> None:
        with self.lock:
            if self._tty and self._last_line_len:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._last_line_len = 0
            elapsed = time.time() - self.start_ts
            mins = elapsed / 60
            print(
                f"[progress] done in {mins:.1f}m: "
                f"✅{self.counts['completed']} ⏭{self.counts['skipped']} "
                f"❌{self.counts['failed']}",
                flush=True,
            )

class _RecordingClientWrapper:
    """Thin wrapper that records LLM chat_completion calls for trajectory logging,
    while delegating all execution to the underlying client.

    Thread-safety: ``calls`` and ``clear()`` are per-thread (backed by
    ``threading.local``). This lets the 4 expert agents (core / episodic /
    semantic / procedural) run concurrently against the same wrapper while
    each one observes only its own recorded calls -- which is exactly what
    the per-agent ``recording.calls[0]["prompt"]`` / ``recording.calls[-1]
    ["response"]`` lookups in ``_call_*_agent`` need.
    """

    def __init__(self, real_client):
        self._real = real_client
        self._tls = threading.local()  # per-thread {calls: list}

    def _bucket(self) -> List[Dict[str, Any]]:
        bucket = getattr(self._tls, "calls", None)
        if bucket is None:
            bucket = []
            self._tls.calls = bucket
        return bucket

    @property
    def calls(self) -> List[Dict[str, Any]]:
        return self._bucket()

    def chat_completion(self, **kwargs):
        prompt = ""
        msgs = kwargs.get("messages", [])
        if msgs:
            prompt = msgs[0].get("content", "")
        start = time.time()
        result = self._real.chat_completion(**kwargs)
        elapsed_ms = (time.time() - start) * 1000
        self._bucket().append({"prompt": prompt, "response": result, "time_ms": elapsed_ms})
        return result

    def get_embeddings(self, *args, **kwargs):
        return self._real.get_embeddings(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate unknown attributes to the real client."""
        return getattr(self._real, name)

    def clear(self):
        self._bucket().clear()


class ExpertTrajectoryGenerator:
    """Generate expert memory building trajectories."""
    
    def __init__(
        self, 
        dataset: str, 
        expert_model: str, 
        output_dir: str = "./expert_trajectories"
    ):
        self.dataset = dataset
        self.expert_model = expert_model
        self.output_dir = Path(output_dir) / dataset
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Init banner is verbose-only -- in quiet mode the progress tracker
        # is the source of truth.
        _vprint(
            f"[init] dataset={dataset}  model={expert_model}  out={self.output_dir}"
        )
    
    # Canonical agent ordering: a session is "fully completed" only when all
    # 4 agent_types appear in agent_calls.jsonl for that session_index.
    _REQUIRED_AGENT_TYPES = ("core", "episodic", "semantic", "procedural")

    def _load_existing_progress(
        self,
        conv_dir: Path,
        states_dir: Path,
        total_sessions: int,
    ) -> Dict[str, Any]:
        """Inspect on-disk artifacts to determine where to resume.

        Returns a dict with:
            resume_idx          int -- next session_index to process
            completed_calls     list -- records to keep in agent_calls.jsonl
            truncated_jsonl     bool -- True if we rewrote jsonl to drop a
                                        partially-written session at the tail
            load_state_dir      Optional[Path] -- state_{resume_idx} dir to
                                        restore before the loop (None if
                                        resume_idx == 0, i.e. fresh start)
        """
        jsonl_path = conv_dir / "agent_calls.jsonl"
        if not jsonl_path.exists():
            return {
                "resume_idx": 0,
                "completed_calls": [],
                "truncated_jsonl": False,
                "load_state_dir": None,
            }

        # Bucket existing records by session_index. Tolerate trailing partial
        # lines from a hard kill (skip non-parseable lines).
        by_session: Dict[int, List[Dict[str, Any]]] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sidx = rec.get("session_index")
                if isinstance(sidx, int):
                    by_session.setdefault(sidx, []).append(rec)

        # Walk session indices contiguously from 0; stop at the first gap or
        # the first session that doesn't have all 4 agent_types. That session
        # (and everything after it) needs to be redone.
        required = set(self._REQUIRED_AGENT_TYPES)
        resume_idx = 0
        completed_calls: List[Dict[str, Any]] = []
        for sidx in range(total_sessions):
            recs = by_session.get(sidx, [])
            agent_types = {r.get("agent_type") for r in recs}
            if required.issubset(agent_types):
                # Keep canonical order; drop accidental duplicates if any.
                seen = set()
                ordered: List[Dict[str, Any]] = []
                for at in self._REQUIRED_AGENT_TYPES:
                    for r in recs:
                        if r.get("agent_type") == at and at not in seen:
                            ordered.append(r)
                            seen.add(at)
                            break
                completed_calls.extend(ordered)
                resume_idx = sidx + 1
            else:
                break

        # Truncate jsonl iff its tail had partial/garbage data we are dropping.
        # We always rewrite from completed_calls so the file is canonical even
        # if the prior writer left e.g. 2 records for an aborted session.
        original_record_count = sum(len(v) for v in by_session.values())
        truncated = original_record_count != len(completed_calls)
        if truncated:
            tmp_path = jsonl_path.with_suffix(".jsonl.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                for call in completed_calls:
                    f.write(json.dumps(call, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, jsonl_path)

        # Determine which state dir to restore.
        load_state_dir: Optional[Path] = None
        if resume_idx > 0:
            candidate = states_dir / f"state_{resume_idx}"
            if not (candidate / "core_memory.json").exists():
                # State checkpoint missing -- can't trust resume. Fall back to
                # full restart and warn loudly.
                print(
                    f"  ⚠️ resume checkpoint missing: {candidate}. "
                    f"Restarting conv from scratch."
                )
                # Wipe the canonicalized jsonl so the conv starts cleanly.
                jsonl_path.unlink(missing_ok=True)
                return {
                    "resume_idx": 0,
                    "completed_calls": [],
                    "truncated_jsonl": True,
                    "load_state_dir": None,
                }
            load_state_dir = candidate

        return {
            "resume_idx": resume_idx,
            "completed_calls": completed_calls,
            "truncated_jsonl": truncated,
            "load_state_dir": load_state_dir,
        }

    @staticmethod
    def _append_jsonl(jsonl_path: Path, records: List[Dict[str, Any]]) -> None:
        """Atomically append records to agent_calls.jsonl (flush+fsync)."""
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def generate_conversation_trajectory(
        self,
        conv_id: str,
        sessions: List[Dict],
        skip_existing: bool = False
    ) -> Dict:
        """Generate complete trajectory for a single conversation.

        Resume-aware: if ``agent_calls.jsonl`` already covers the first N
        sessions (with all 4 agent_types each) and ``states/state_N/`` exists,
        the loop restores that checkpoint and continues from session N. This
        means hard kills / API outages mid-conv no longer wipe progress -- the
        next run picks up at the boundary of the last fully-completed session.
        """
        conv_dir = self.output_dir / conv_id
        states_dir = conv_dir / "states"
        jsonl_path = conv_dir / "agent_calls.jsonl"
        total_sessions = len(sessions)

        conv_dir.mkdir(parents=True, exist_ok=True)
        states_dir.mkdir(exist_ok=True)

        # Reconstruct progress from disk before doing anything else.
        progress = self._load_existing_progress(conv_dir, states_dir, total_sessions)
        resume_idx = progress["resume_idx"]
        agent_calls: List[Dict[str, Any]] = list(progress["completed_calls"])

        # Already fully done? Honor skip_existing.
        if skip_existing and resume_idx >= total_sessions and total_sessions > 0:
            _vprint(f"[skip] already complete: {conv_id}")
            return {"conv_id": conv_id, "status": "skipped"}

        if resume_idx > 0:
            _vprint(
                f"[resume] {conv_id}: from session {resume_idx + 1}/"
                f"{total_sessions} ({resume_idx} done)"
                + (" [jsonl truncated]" if progress["truncated_jsonl"] else "")
            )

        # Initialize memory system with recording wrapper around the real client
        real_client = _create_llm_client(model=self.expert_model)
        recording_client = _RecordingClientWrapper(real_client)
        memory = MemorySystem(llm_client=recording_client)

        # Restore vector_store + core_memory if resuming.
        if progress["load_state_dir"] is not None:
            memory.load_state(str(progress["load_state_dir"]))

        # Initialize chunk_store: maps session_key → chunks list.
        # Restore from checkpoint if resuming.
        try:
            from config import CHUNK_TURN_SIZE as _chunk_turn_size
        except Exception:
            _chunk_turn_size = 4
        chunk_store: Dict[str, List[Dict[str, Any]]] = {}
        if progress["load_state_dir"] is not None:
            _cs_path = Path(progress["load_state_dir"]) / "chunk_store.json"
            if _cs_path.exists():
                with open(_cs_path) as _f:
                    chunk_store = json.load(_f)

        max_session_retries = 3

        for session_idx in range(resume_idx, total_sessions):
            session = sessions[session_idx]
            messages = session.get("messages", session) if isinstance(session, dict) else session
            timestamp = session.get("timestamp") if isinstance(session, dict) else None

            _vprint(f"  [{conv_id}] session {session_idx + 1}/{total_sessions}")

            state_before_path = states_dir / f"state_{session_idx}"
            # Always (re)save the before-snapshot: when resuming from session
            # N, state_{N} on disk is the post-state of session N-1, which is
            # exactly what we want to write here. Re-saving is idempotent and
            # cheap, and keeps the invariant "state_{i}/ exists iff session i
            # has been entered".
            self._save_state(memory, state_before_path, chunk_store=chunk_store)

            last_error = None
            for attempt in range(1, max_session_retries + 1):
                try:
                    session_calls = self._process_session(
                        memory=memory,
                        messages=messages,
                        session_index=session_idx,
                        conv_id=conv_id,
                        user_id=conv_id,
                        state_before_path=str(state_before_path),
                        timestamp=timestamp,
                    )
                    # Persist this session's records BEFORE moving on, so a
                    # crash on the next session doesn't lose what we just did.
                    self._append_jsonl(jsonl_path, session_calls)
                    agent_calls.extend(session_calls)
                    # Accumulate chunk info for this session
                    session_key = f"{conv_id}_sess{session_idx}"
                    chunk_store[session_key] = self._chunk_session_messages(
                        messages, _chunk_turn_size
                    )
                    break  # success
                except Exception as e:
                    last_error = e
                    if attempt < max_session_retries:
                        wait_sec = attempt * 5
                        print(f"  ⚠️ Session {session_idx + 1} failed (attempt {attempt}/{max_session_retries}): {str(e)[:80]}")
                        print(f"     Restoring state and retrying in {wait_sec}s...")
                        time.sleep(wait_sec)
                        # Restore memory state from checkpoint before retry
                        memory.load_state(str(state_before_path))
                    else:
                        print(f"  ❌ Session {session_idx + 1} failed after {max_session_retries} attempts: {str(e)[:100]}")
                        raise last_error

        # Save final state
        final_state_path = states_dir / f"state_{total_sessions}"
        self._save_state(memory, final_state_path, chunk_store=chunk_store)

        # Save metadata (jsonl is already on disk via incremental appends)
        metadata = {
            "conv_id": conv_id,
            "num_sessions": total_sessions,
            "expert_model": self.expert_model,
            "total_agent_calls": len(agent_calls)
        }
        with open(conv_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "conv_id": conv_id,
            "status": "completed",
            "num_sessions": total_sessions,
            "agent_calls": len(agent_calls)
        }

    def _save_state(self, memory: MemorySystem, state_path: Path,
                    chunk_store: Optional[Dict[str, Any]] = None):
        """Save memory state (and optional chunk store) to disk."""
        memory.save_state(str(state_path))
        if chunk_store is not None:
            chunk_store_path = Path(state_path) / "chunk_store.json"
            with open(chunk_store_path, "w") as f:
                json.dump(chunk_store, f, ensure_ascii=False)

    @staticmethod
    def _chunk_session_messages(
        messages: List[Dict[str, Any]],
        chunk_turn_size: int = 4,
    ) -> List[Dict[str, Any]]:
        """Split session messages into fixed-turn-size chunks.

        Returns list of chunk dicts with chunk_index, turn_start, turn_end, text.
        """
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
                "turn_start": start,
                "turn_end": end,
                "text": "\n".join(lines),
            })
        return chunks

    def _process_session(
        self,
        memory: MemorySystem,
        messages: List[Dict[str, Any]],
        session_index: int,
        conv_id: str,
        user_id: str,
        state_before_path: str,
        timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Run the 4 expert agents concurrently. They are independent at this
        # stage:
        #   - all agents see the same vector_store snapshot (no writes happen
        #     until the batch-insert below)
        #   - core_agent mutates self.core_memory.human, but only one thread
        #     touches it
        #   - episodic / semantic / procedural only READ the vector store via
        #     _get_existing_memories (FAISS reads are thread-safe)
        # The recording wrapper isolates per-thread chat_completion logs via
        # threading.local, so each _call_*_agent still sees only its own
        # prompt/response.
        agent_order = ["core", "episodic", "semantic", "procedural"]

        def _run(agent_type: str):
            if agent_type == "core":
                return self._call_core_agent(
                    memory_system=memory,
                    messages=messages,
                    session_index=session_index,
                    conv_id=conv_id,
                    user_id=user_id,
                    state_before_path=state_before_path,
                    timestamp=timestamp,
                ), []  # core produces no searchable memories
            call_record, memories = self._call_memory_agent(
                memory_system=memory,
                agent_type=agent_type,
                messages=messages,
                session_index=session_index,
                conv_id=conv_id,
                user_id=user_id,
                state_before_path=state_before_path,
                timestamp=timestamp,
            )
            return call_record, memories

        results: Dict[str, Any] = {}
        first_error: Optional[BaseException] = None
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_run, t): t for t in agent_order}
            for future in as_completed(futures):
                agent_type = futures[future]
                try:
                    results[agent_type] = future.result()
                except BaseException as e:  # noqa: BLE001 -- rethrown below
                    if first_error is None:
                        first_error = e
        if first_error is not None:
            # Surface to outer per-session retry loop, which restores state.
            raise first_error

        # Preserve the canonical agent order in the trajectory log.
        calls = [results[t][0] for t in agent_order]
        all_searchable_memories: List[str] = []
        for t in ["episodic", "semantic", "procedural"]:
            all_searchable_memories.extend(results[t][1])

        # Batch-insert all searchable memories AFTER all agents complete
        # (matches MemorySystem.add() behavior where agents run in parallel
        # and all see the same vector store state)
        if all_searchable_memories:
            memory.vector_store.add(
                all_searchable_memories,
                [{"user_id": user_id, "type": "memory"} for _ in all_searchable_memories],
            )

        return calls

    def _call_core_agent(
        self,
        memory_system: MemorySystem,
        messages: List[Dict[str, Any]],
        session_index: int,
        conv_id: str,
        user_id: str,
        state_before_path: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Capture state before for recording
        current_core = memory_system.core_memory.human or "[Empty]"
        core_usage = memory_system.core_memory.get_usage()

        # Clear recorder, then delegate ALL logic to MemorySystem._core_agent
        # (handles APPEND/REPLACE/REWRITE, content validation, compression)
        recording = memory_system.llm_client
        recording.clear()

        call_start = time.time()
        operation, new_core_memory = memory_system._core_agent(messages, user_id)
        call_time_ms = (time.time() - call_start) * 1000
        if str(operation).upper() == "ERROR":
            # Let session-level retry handle transient parsing/request failures.
            raise RuntimeError(f"Core agent returned ERROR at session {session_index}")

        # Retrieve the prompt from the recorded wrapper (first call = agent decision)
        full_prompt = recording.calls[0]["prompt"] if recording.calls else ""

        return {
            "conversation_id": conv_id,
            "session_index": session_index,
            "agent_type": "core",
            "call_id": f"{conv_id}_session-{session_index}_core",
            "state_before_path": state_before_path,
            "input": {
                "system_prompt": memory_system.prompts["core"],
                "retrieved_memories": {
                    "core_memory": current_core,
                    "usage_percent": core_usage,
                },
                "new_session": {
                    "timestamp": timestamp,
                    "messages": messages,
                },
                "full_prompt": full_prompt,
            },
            "expert_output": {
                "operation": operation,
                "new_core_memory": new_core_memory,
            },
            "metadata": {
                "expert_model": self.expert_model,
                "temperature": 0.0,
                "call_time_ms": call_time_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }

    def _call_memory_agent(
        self,
        memory_system: MemorySystem,
        agent_type: str,
        messages: List[Dict[str, Any]],
        session_index: int,
        conv_id: str,
        user_id: str,
        state_before_path: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Capture existing memories for recording
        existing = memory_system._get_existing_memories(agent_type, user_id, messages)

        # Clear recorder, then delegate ALL logic to MemorySystem's agent method
        # (handles JSON retry, operation parsing, memory extraction with prefixes)
        recording = memory_system.llm_client
        recording.clear()

        call_start = time.time()
        metadata_dict = {"timestamp": timestamp} if timestamp else None

        if agent_type == "episodic":
            memories, stats = memory_system._episodic_agent(messages, user_id, metadata_dict)
        elif agent_type == "semantic":
            memories, stats = memory_system._semantic_agent(messages, user_id)
        else:  # procedural
            memories, stats = memory_system._procedural_agent(messages, user_id)

        call_time_ms = (time.time() - call_start) * 1000

        # Retrieve prompt and raw response from the recording wrapper
        full_prompt = recording.calls[0]["prompt"] if recording.calls else ""

        # Parse raw response for the operations list (for call record)
        operations = []
        if recording.calls:
            last_response = recording.calls[-1]["response"]
            try:
                clean = memory_system._clean_json_response(last_response)
                operations = json.loads(clean).get("operations", [])
            except Exception:
                pass

        # Return call record AND memories list (caller batches vector store insertion)
        call_record = {
            "conversation_id": conv_id,
            "session_index": session_index,
            "agent_type": agent_type,
            "call_id": f"{conv_id}_session-{session_index}_{agent_type}",
            "state_before_path": state_before_path,
            "input": {
                "system_prompt": memory_system.prompts[agent_type],
                "retrieved_memories": {
                    agent_type: [
                        {"id": f"{agent_type}_{i}", "content": mem, "score": 0.0}
                        for i, mem in enumerate(existing[:50])
                    ]
                },
                "new_session": {
                    "timestamp": timestamp,
                    "messages": messages,
                },
                "full_prompt": full_prompt,
            },
            "expert_output": {
                "operations": operations,
                "memories_added": memories,
            },
            "metadata": {
                "expert_model": self.expert_model,
                "temperature": 0.0,
                "call_time_ms": call_time_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }
        return call_record, memories
    
    def generate_batch(
        self,
        conversations: List[Dict],
        parallel: bool = False,
        workers: int = 4,
        skip_existing: bool = True
    ):
        """Generate trajectories for multiple conversations."""
        tracker = _ProgressTracker(total=len(conversations), label="trajectories")
        tracker.start()

        results: List[Dict[str, Any]] = []

        def _record(conv_id: str, result_or_exc):
            if isinstance(result_or_exc, BaseException):
                # Permanent error line above the progress bar.
                tracker.log(f"❌ {conv_id}: {str(result_or_exc)[:140]}")
                results.append({"conv_id": conv_id, "status": "failed", "error": str(result_or_exc)})
                tracker.update("failed", conv_id)
            else:
                results.append(result_or_exc)
                status = result_or_exc.get("status", "completed")
                tracker.update(status, conv_id)

        if parallel:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self.generate_conversation_trajectory,
                        conv["id"],
                        conv["sessions"],
                        skip_existing
                    ): conv["id"]
                    for conv in conversations
                }
                for future in as_completed(futures):
                    conv_id = futures[future]
                    try:
                        _record(conv_id, future.result())
                    except Exception as e:
                        _record(conv_id, e)
        else:
            for conv in conversations:
                try:
                    _record(conv["id"], self.generate_conversation_trajectory(
                        conv["id"], conv["sessions"], skip_existing
                    ))
                except Exception as e:
                    _record(conv["id"], e)

        tracker.finish()
        return results


def main():
    parser = argparse.ArgumentParser(description="Generate expert trajectories")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset file (JSONL/JSON)")
    parser.add_argument("--conv-id", type=str, help="Single conversation ID")
    parser.add_argument("--subset-file", type=str, help="JSON file with conversation IDs")
    parser.add_argument("--split", type=str, choices=["sft", "rl", "test"], 
                       help="Use predefined split from data/longmemeval/splits/longmemeval_splits.json")
    parser.add_argument("--expert-model", type=str, default=SFT_EXPERT_MODEL, 
                       help="Expert model for generation")
    parser.add_argument("--output-dir", type=str, default="./expert_trajectories",
                       help="Output directory")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing trajectories")
    parser.add_argument("--provider", type=str, default="openai", choices=AVAILABLE_PROVIDERS,
                       help="LLM provider (default: openai)")
    
    args = parser.parse_args()
    
    # Set global provider
    global _LLM_CLIENT_PROVIDER
    _LLM_CLIENT_PROVIDER = args.provider
    print(f"LLM provider: {args.provider}")
    
    # Handle --split parameter: convert to subset_file path
    if args.split and args.dataset == "longmemeval":
        splits_file = Path(__file__).parent.parent / "data" / "longmemeval" / "splits" / "longmemeval_splits.json"
        if splits_file.exists():
            with open(splits_file, "r", encoding="utf-8") as f:
                splits_data = json.load(f)
            if args.split in splits_data:
                # Create a temporary list of question IDs for this split
                args._split_question_ids = splits_data[args.split]
                print(f"Using {args.split} split: {len(args._split_question_ids)} dialogues")
            else:
                raise ValueError(f"Split '{args.split}' not found in {splits_file}")
        else:
            raise FileNotFoundError(f"Splits file not found: {splits_file}")
    else:
        args._split_question_ids = None
    
    generator = ExpertTrajectoryGenerator(
        dataset=args.dataset,
        expert_model=args.expert_model,
        output_dir=args.output_dir
    )
    
    def load_jsonl(path: Path) -> List[Dict[str, Any]]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def resolve_locomo_path(dataset_path: Optional[str]) -> Path:
        if dataset_path:
            return Path(dataset_path)

        candidates = [
            Path(__file__).parent.parent / "data" / "locomo_mc10.json",
            Path(__file__).parent.parent.parent / "locomo" / "locomo_mc10" / "data" / "locomo_mc10.json",
            Path(__file__).parent.parent.parent / "locomo" / "locomo_mc10" / "data" / "locomo_mc10.jsonl",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError("LoCoMo dataset file not found. Provide --dataset-path.")

    def locomo_conv_sessions(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        sessions = sample.get("haystack_sessions", [])
        dates = sample.get("haystack_session_datetimes") or sample.get("haystack_dates") or []
        out = []
        for idx, sess in enumerate(sessions):
            ts = dates[idx] if idx < len(dates) else None
            msgs = []
            for turn in sess:
                if isinstance(turn, dict):
                    role = turn.get("speaker") or turn.get("role") or "Unknown"
                    content = turn.get("message") or turn.get("text") or turn.get("content") or ""
                    if content:
                        msgs.append({"role": str(role), "content": str(content)})
            out.append({"messages": msgs, "timestamp": ts})
        return out

    def load_locomo_conversations(
        dataset_path: Optional[str],
        conv_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        path = resolve_locomo_path(dataset_path)
        data = load_jsonl(path)
        grouped = {}
        for item in data:
            qid = item.get("question_id")
            if not qid or "_q" not in qid:
                continue
            cid = qid.split("_q")[0]
            if conv_ids and cid not in set(conv_ids):
                continue
            if cid not in grouped:
                grouped[cid] = {
                    "id": cid,
                    "sessions": locomo_conv_sessions(item),
                }
        if conv_ids:
            ordered = []
            for cid in conv_ids:
                if cid in grouped:
                    ordered.append(grouped[cid])
            return ordered
        return list(grouped.values())

    def parse_subset_ids(obj: Any) -> List[str]:
        if isinstance(obj, list):
            if not obj:
                return []
            if isinstance(obj[0], str):
                return obj
            return []
        if isinstance(obj, dict):
            if "conv_ids" in obj and isinstance(obj["conv_ids"], list):
                return [str(x) for x in obj["conv_ids"]]
            if "question_ids" in obj and isinstance(obj["question_ids"], list):
                convs = sorted({str(q).split("_q")[0] for q in obj["question_ids"] if "_q" in str(q)})
                return convs
        return []

    if args.dataset == "locomo":
        if args.conv_id:
            conversations = load_locomo_conversations(args.dataset_path, conv_ids=[args.conv_id])
            if not conversations:
                raise ValueError(f"Conversation not found: {args.conv_id}")
            generator.generate_conversation_trajectory(
                conv_id=conversations[0]["id"],
                sessions=conversations[0]["sessions"],
                skip_existing=args.skip_existing,
            )
        elif args.subset_file:
            with open(args.subset_file, "r", encoding="utf-8") as f:
                subset_obj = json.load(f)
            conv_ids = parse_subset_ids(subset_obj)
            if not conv_ids and isinstance(subset_obj, list) and subset_obj and isinstance(subset_obj[0], dict):
                conversations = subset_obj
            else:
                conversations = load_locomo_conversations(args.dataset_path, conv_ids=conv_ids or None)
            generator.generate_batch(
                conversations,
                parallel=args.parallel,
                workers=args.workers,
                skip_existing=args.skip_existing,
            )
        else:
            parser.print_help()
    elif args.dataset == "longmemeval":
        # LongMemEval dataset support
        def resolve_longmemeval_path(dataset_path: Optional[str]) -> Path:
            if dataset_path:
                return Path(dataset_path)
            candidates = [
                Path(__file__).parent.parent / "data" / "longmemeval" / "longmemeval_s_cleaned.json",
                Path(__file__).parent.parent.parent / "longmemeval" / "longmemeval_s.json",
            ]
            for p in candidates:
                if p.exists():
                    return p
            raise FileNotFoundError("LongMemEval dataset not found. Provide --dataset-path.")

        def longmemeval_conv_sessions(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
            sessions = sample.get("haystack_sessions", [])
            dates = sample.get("haystack_dates", [])
            out = []
            for idx, sess in enumerate(sessions):
                ts = dates[idx] if idx < len(dates) else None
                msgs = []
                for turn in sess:
                    if isinstance(turn, dict):
                        role = turn.get("role", "user")
                        content = turn.get("content", "")
                        if content:
                            msgs.append({"role": str(role), "content": str(content)})
                out.append({"messages": msgs, "timestamp": ts})
            return out

        def load_longmemeval_conversations(
            dataset_path: Optional[str],
            question_ids: Optional[List[str]] = None,
        ) -> List[Dict[str, Any]]:
            path = resolve_longmemeval_path(dataset_path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Group by question_id (each sample is a unique conversation)
            conversations = []
            seen_ids = set()
            for item in data:
                qid = item.get("question_id", "")
                if not qid or qid in seen_ids:
                    continue
                if question_ids and qid not in set(question_ids):
                    continue
                seen_ids.add(qid)
                conversations.append({
                    "id": qid,
                    "sessions": longmemeval_conv_sessions(item),
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                })
            return conversations

        if args.conv_id:
            conversations = load_longmemeval_conversations(args.dataset_path, question_ids=[args.conv_id])
            if not conversations:
                raise ValueError(f"Conversation not found: {args.conv_id}")
            generator.generate_conversation_trajectory(
                conv_id=conversations[0]["id"],
                sessions=conversations[0]["sessions"],
                skip_existing=args.skip_existing,
            )
        elif args._split_question_ids:
            # Use --split parameter
            conversations = load_longmemeval_conversations(args.dataset_path, question_ids=args._split_question_ids)
            generator.generate_batch(
                conversations,
                parallel=args.parallel,
                workers=args.workers,
                skip_existing=args.skip_existing,
            )
        elif args.subset_file:
            with open(args.subset_file, "r", encoding="utf-8") as f:
                subset_obj = json.load(f)
            question_ids = subset_obj if isinstance(subset_obj, list) else subset_obj.get("question_ids", [])
            conversations = load_longmemeval_conversations(args.dataset_path, question_ids=question_ids or None)
            generator.generate_batch(
                conversations,
                parallel=args.parallel,
                workers=args.workers,
                skip_existing=args.skip_existing,
            )
        else:
            # Default: load first conversation
            conversations = load_longmemeval_conversations(args.dataset_path)
            if conversations:
                print(f"No --conv-id specified. Using first conversation: {conversations[0]['id']}")
                generator.generate_conversation_trajectory(
                    conv_id=conversations[0]["id"],
                    sessions=conversations[0]["sessions"],
                    skip_existing=args.skip_existing,
                )
            else:
                parser.print_help()
    else:
        if args.subset_file:
            with open(args.subset_file, "r", encoding="utf-8") as f:
                subset_data = json.load(f)

            generator.generate_batch(
                subset_data,
                parallel=args.parallel,
                workers=args.workers,
                skip_existing=args.skip_existing,
            )
        else:
            parser.print_help()



if __name__ == "__main__":
    main()
