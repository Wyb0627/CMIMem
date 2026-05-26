"""MemoryBuilderAgentLoop: Multi-turn agentic RL for memory construction.

Each trajectory = one conversation with N sessions.
Each turn = one session → model generates unified memory actions → execute → CMI reward.

Key design choices:
- Unified output: model outputs all 4 agent types in one JSON per turn
- Sequential execution: actions applied in deterministic order (core → episodic → semantic → procedural)
- Per-turn CMI reward: computed in-loop via embedding-based CMI from cmi_reward.py
- Lightweight memory state: no FAISS/embedding during rollout; memory stored as plain text lists
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Lightweight in-memory state (no FAISS, no embedding calls during rollout)
# ---------------------------------------------------------------------------

class LightMemoryState:
    """Minimal memory state for agentic rollout.

    Stores memories as plain text lists. Retrieval uses simple substring
    matching (not embedding) to keep rollout fast. The full embedding-based
    CMI is computed only at reward time.
    """

    def __init__(self, state_before: Optional[Dict[str, Any]] = None):
        state = state_before or {}
        self.core_memory: str = str(state.get("core_memory", "") or "")
        self.episodic: List[str] = list(state.get("episodic", []) or [])
        self.semantic: List[str] = list(state.get("semantic", []) or [])
        self.procedural: List[str] = list(state.get("procedural", []) or [])

    def get_all_memories(self) -> List[str]:
        """Return all non-core memories as a flat list with type prefixes."""
        result = []
        if self.core_memory:
            result.append(f"[CORE] {self.core_memory}")
        for m in self.episodic:
            result.append(f"[EPISODIC] {m}")
        for m in self.semantic:
            result.append(f"[SEMANTIC] {m}")
        for m in self.procedural:
            result.append(f"[PROCEDURAL] {m}")
        return result

    def get_relevant(self, session_text: str, top_k: int = 10) -> Dict[str, List[str]]:
        """Fast keyword-overlap retrieval for prompt construction.

        Uses word overlap ratio instead of embeddings to avoid API calls
        during rollout. This is intentionally approximate — the model sees
        a rough context, and CMI reward is computed with proper embeddings.
        """
        session_words = set(session_text.lower().split())

        def score(memory_text: str) -> float:
            mem_words = set(memory_text.lower().split())
            if not mem_words:
                return 0.0
            return len(session_words & mem_words) / len(mem_words)

        def top_k_items(items: List[str], k: int) -> List[str]:
            if len(items) <= k:
                return list(items)
            scored = sorted(items, key=score, reverse=True)
            return scored[:k]

        return {
            "episodic": top_k_items(self.episodic, top_k),
            "semantic": top_k_items(self.semantic, top_k),
            "procedural": top_k_items(self.procedural, top_k),
        }

    def snapshot_as_list(self) -> List[str]:
        """Snapshot for CMI computation (same format as cmi_reward.py expects)."""
        return self.get_all_memories()

    def apply_actions(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply parsed actions to memory state in deterministic order.

        Returns the ordered list of action steps (same format as
        cmi_reward._extract_cmi_action_steps) for CMI computation.
        """
        action_steps: List[Dict[str, Any]] = []

        # 1. Core
        core_data = parsed.get("core", {})
        if isinstance(core_data, dict):
            operation = str(core_data.get("operation", "")).upper()
            if operation == "APPEND":
                content = str(core_data.get("content", "")).strip()
                if content:
                    self.core_memory = (self.core_memory + "\n" + content).strip()
                    action_steps.append({
                        "agent_type": "core",
                        "memory_type": "core",
                        "action": "APPEND",
                        "memory": f"[CORE] {content}",
                        "old_memory": "",
                        "old_memories": [],
                    })
            elif operation == "REPLACE":
                old_text = str(core_data.get("old_text", "")).strip()
                new_text = str(core_data.get("new_text", "")).strip()
                if new_text and old_text:
                    self.core_memory = self.core_memory.replace(old_text, new_text)
                    action_steps.append({
                        "agent_type": "core",
                        "memory_type": "core",
                        "action": "REPLACE",
                        "memory": f"[CORE] {new_text}",
                        "old_memory": old_text,
                        "old_memories": [],
                    })
            elif operation == "REWRITE":
                # REWRITE replaces the entire core memory with new content
                new_text = str(
                    core_data.get("new_text", "") or core_data.get("content", "")
                ).strip()
                if new_text:
                    old_core = self.core_memory
                    self.core_memory = new_text
                    action_steps.append({
                        "agent_type": "core",
                        "memory_type": "core",
                        "action": "REWRITE",
                        "memory": f"[CORE] {new_text}",
                        "old_memory": old_core,
                        "old_memories": [],
                    })

        # 2-4. Episodic, Semantic, Procedural
        for agent_type in ("episodic", "semantic", "procedural"):
            agent_data = parsed.get(agent_type, {})
            if not isinstance(agent_data, dict):
                continue
            operations = agent_data.get("operations", [])
            if not isinstance(operations, list):
                continue
            mem_list = getattr(self, agent_type)

            for op in operations:
                if not isinstance(op, dict):
                    continue
                action = str(op.get("action", "")).upper()
                if action == "SKIP":
                    continue

                if action == "ADD":
                    memory = str(op.get("memory", "")).strip()
                    if memory:
                        mem_list.append(memory)
                        action_steps.append({
                            "agent_type": agent_type,
                            "memory_type": agent_type,
                            "action": "ADD",
                            "memory": f"[{agent_type.upper()}] {memory}",
                            "old_memory": "",
                            "old_memories": [],
                        })

                elif action == "UPDATE":
                    old_memory = str(op.get("old_memory", "")).strip()
                    new_memory = str(op.get("new_memory", op.get("memory", ""))).strip()
                    if new_memory:
                        # Remove old if exact match found
                        if old_memory and old_memory in mem_list:
                            mem_list.remove(old_memory)
                        mem_list.append(new_memory)
                        action_steps.append({
                            "agent_type": agent_type,
                            "memory_type": agent_type,
                            "action": "UPDATE",
                            "memory": f"[{agent_type.upper()}] {new_memory}",
                            "old_memory": old_memory,
                            "old_memories": [],
                        })

                elif action == "MERGE" and agent_type == "episodic":
                    old_memories = op.get("old_memories", [])
                    new_memory = str(op.get("new_memory", op.get("memory", ""))).strip()
                    if new_memory and isinstance(old_memories, list):
                        for old in old_memories:
                            old_str = str(old).strip()
                            if old_str in mem_list:
                                mem_list.remove(old_str)
                        mem_list.append(new_memory)
                        old_strs = [str(m).strip() for m in old_memories if str(m).strip()]
                        action_steps.append({
                            "agent_type": agent_type,
                            "memory_type": agent_type,
                            "action": "MERGE",
                            "memory": f"[{agent_type.upper()}] {new_memory}",
                            "old_memory": "",
                            "old_memories": old_strs,
                        })

        return action_steps




# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_unified_response(response_text: str) -> Tuple[Dict[str, Any], bool]:
    """Parse model output into a unified actions dict.

    Supports two formats:
    1. DSL (preferred, compact): one operation per line, TYPE:ACTION|fields
    2. JSON (legacy fallback): full JSON object

    Returns (parsed_dict, format_valid) where parsed_dict is compatible
    with LightMemoryState.apply_actions().
    """
    clean = response_text.strip()
    # Remove think tags
    clean = re.sub(r"", "", clean, flags=re.DOTALL | re.IGNORECASE).strip()
    # Remove markdown fences
    clean = re.sub(r"```(?:json)?|```", "", clean, flags=re.IGNORECASE).strip()

    # Try DSL format first (lines like CORE:APPEND|content)
    parsed, valid = _parse_dsl_response(clean)
    if valid:
        return parsed, True

    # Fallback to JSON parsing
    return _parse_json_response(clean)


def _parse_dsl_response(clean: str) -> Tuple[Dict[str, Any], bool]:
    """Parse compact DSL format into the same dict structure as JSON.

    DSL line format: TYPE:ACTION|field1|field2
    Examples:
        CORE:SKIP
        CORE:APPEND|User prefers morning meetings
        CORE:REPLACE|old profile text|new profile text
        EPISODIC:ADD|2023-05-20: Discussed travel plans
        SEMANTIC:UPDATE|old fact|updated fact
        PROCEDURAL:SKIP
    """
    valid_types = {"CORE", "EPISODIC", "SEMANTIC", "PROCEDURAL"}
    valid_actions = {"SKIP", "APPEND", "REPLACE", "ADD", "UPDATE", "MERGE", "REWRITE"}

    parsed: Dict[str, Any] = {
        "core": {"operation": "SKIP"},
        "episodic": {"operations": []},
        "semantic": {"operations": []},
        "procedural": {"operations": []},
    }

    has_valid_line = False

    for raw_line in clean.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Split TYPE:ACTION|fields
        colon_pos = line.find(":")
        if colon_pos == -1:
            continue

        mem_type = line[:colon_pos].strip().upper()
        if mem_type not in valid_types:
            continue

        rest = line[colon_pos + 1:]
        parts = rest.split("|")
        action = parts[0].strip().upper()
        if action not in valid_actions:
            continue

        has_valid_line = True

        if mem_type == "CORE":
            if action == "APPEND" and len(parts) >= 2:
                parsed["core"] = {"operation": "APPEND", "content": parts[1].strip()}
            elif action == "REPLACE" and len(parts) >= 3:
                parsed["core"] = {
                    "operation": "REPLACE",
                    "old_text": parts[1].strip(),
                    "new_text": parts[2].strip(),
                }
            elif action in ("REWRITE",) and len(parts) >= 2:
                parsed["core"] = {"operation": "REWRITE", "content": parts[1].strip()}
            else:
                parsed["core"] = {"operation": "SKIP"}
        else:
            # episodic / semantic / procedural
            ops_list = parsed[mem_type.lower()]["operations"]
            if action == "ADD" and len(parts) >= 2:
                ops_list.append({"action": "ADD", "memory": parts[1].strip()})
            elif action == "UPDATE" and len(parts) >= 3:
                ops_list.append({
                    "action": "UPDATE",
                    "old_memory": parts[1].strip(),
                    "new_memory": parts[2].strip(),
                })
            elif action == "MERGE" and len(parts) >= 3:
                # MERGE|old1;;old2|new merged text
                old_memories = [m.strip() for m in parts[1].split(";;") if m.strip()]
                ops_list.append({
                    "action": "MERGE",
                    "old_memories": old_memories,
                    "new_memory": parts[2].strip(),
                })
            # SKIP lines are valid but produce no operations

    return parsed, has_valid_line


def _parse_json_response(clean: str) -> Tuple[Dict[str, Any], bool]:
    """Parse legacy JSON format (fallback)."""
    start = clean.find("{")
    if start == -1:
        return {}, False

    depth = 0
    in_string = False
    escape = False
    end = -1
    for i, ch in enumerate(clean[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return {}, False

    try:
        parsed = json.loads(clean[start:end])
    except json.JSONDecodeError:
        return {}, False

    if not isinstance(parsed, dict):
        return {}, False

    has_content = False
    for key in ("core", "episodic", "semantic", "procedural"):
        section = parsed.get(key, {})
        if not isinstance(section, dict):
            continue
        if key == "core":
            op = str(section.get("operation", "")).upper()
            if op in ("APPEND", "REPLACE", "REWRITE"):
                has_content = True
        else:
            ops = section.get("operations", [])
            if isinstance(ops, list) and ops:
                has_content = True

    return parsed, has_content


# ---------------------------------------------------------------------------
# CMI reward computation (lightweight, reuses cmi_reward.py internals)
# ---------------------------------------------------------------------------

# Module-level cache for CMI dependencies.  Imported once and reused across
# all turns / trajectories in the same worker process.
_cmi_deps = None

def _get_cmi_deps():
    """Lazy-load and cache CMI imports, config, and factory objects."""
    global _cmi_deps
    if _cmi_deps is not None:
        return _cmi_deps

    from training.reward_server.cmi_config import load_cmi_config
    from training.reward_server.cmi_reward import (
        _aggregate_action_cmi,
        _apply_cmi_action_step,
        _build_local_transition,
        _estimate_action_local_cmi,
        _get_estimator,
        _get_shaper,
    )

    cmi_config = load_cmi_config()
    cmi_method = cmi_config.get("cmi_method", "embedding")
    shaping = cmi_config.get("cmi_shaping", "gaussian")

    _cmi_deps = {
        "load_cmi_config": load_cmi_config,
        "aggregate": _aggregate_action_cmi,
        "apply_step": _apply_cmi_action_step,
        "build_transition": _build_local_transition,
        "estimate_local": _estimate_action_local_cmi,
        "get_estimator": _get_estimator,
        "shaper": _get_shaper(shaping, cmi_config),
        "config": cmi_config,
        "method": cmi_method,
    }
    return _cmi_deps


def compute_turn_cmi(
    memories_before: List[str],
    action_steps: List[Dict[str, Any]],
    session_text: str,
) -> float:
    """Compute CMI reward for one turn.

    Supports two modes based on cmi_config["cmi_method"]:
    - "logprob": Session-level CMI via LLM log-probability difference.
      One LLM call per turn (efficient for external API models).
    - Other (embedding/kernel_mi): Per-action local CMI via embeddings/KDE.

    Falls back to a simple heuristic if the CMI module is not available.
    """
    if not action_steps:
        return 0.0

    try:
        deps = _get_cmi_deps()
        cmi_config = deps["config"]
        shaper = deps["shaper"]

        # --- Logprob path: session-level CMI (one LLM call per turn) ---
        if deps["method"] == "logprob":
            new_memories = [step["memory"] for step in action_steps]
            estimator = deps["get_estimator"]("logprob", cmi_config)
            raw_cmi = estimator.estimate(
                new_memories=new_memories,
                existing_memories=memories_before,
                context=session_text,
                config=cmi_config,
            )
            shaped = shaper.shape(raw_cmi) if raw_cmi != 0.0 else 0.0
            return float(max(-1.0, min(1.0, shaped)))

        # --- Embedding/KDE path: per-action local CMI ---
        current_memories = list(memories_before)
        action_rewards: List[float] = []

        for action_step in action_steps:
            local_before, local_after = deps["build_transition"](
                current_memories=current_memories,
                action_step=action_step,
                context=session_text,
                config=cmi_config,
            )
            raw_cmi = deps["estimate_local"](
                action_step=action_step,
                local_before=local_before,
                local_after=local_after,
                context=session_text,
                config=cmi_config,
            )
            shaped = shaper.shape(raw_cmi) if raw_cmi != 0.0 else 0.0
            shaped = max(-1.0, min(1.0, shaped))
            action_rewards.append(shaped)
            current_memories = deps["apply_step"](current_memories, action_step)

        return float(deps["aggregate"](action_rewards))

    except Exception as exc:
        raise RuntimeError(
            f"[AgentLoop] CMI computation failed: {exc}. "
            f"Check embedding service actor and CMI config."
        ) from exc


def _heuristic_turn_reward(action_steps: List[Dict[str, Any]]) -> float:
    """Fallback reward: small positive for valid actions, 0 otherwise."""
    if not action_steps:
        return 0.0
    meaningful = sum(1 for step in action_steps if step.get("action", "").upper() != "SKIP")
    return min(0.3, 0.05 * meaningful)


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

@register("memory_builder_agent")
class MemoryBuilderAgentLoop(AgentLoopBase):
    """Multi-turn agentic agent loop for MemBuilder.

    Each turn processes one session:
    1. Build prompt with current memory state + session context
    2. Generate model response
    3. Parse unified JSON output
    4. Execute actions → update memory state
    5. Compute turn-level CMI reward

    The loop runs for N turns (N = number of sessions in the trajectory).
    """

    _class_initialized = False

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.info("Initializing MemoryBuilderAgentLoop class-level state")

        cls.tokenizer = tokenizer
        cls.processor = processor

        # Read agentic-specific config
        agent_extra = {}
        try:
            agent_extra = config.actor_rollout_ref.get("agent_extra", {})
            if hasattr(agent_extra, "__iter__"):
                agent_extra = dict(agent_extra)
            else:
                agent_extra = {}
        except Exception:
            pass

        cls.top_k = int(agent_extra.get("top_k", 10))
        cls.format_penalty = float(agent_extra.get("format_penalty", -0.5))
        cls.max_turns = int(agent_extra.get("max_turns", 64))

        cls.apply_chat_template_kwargs = {}
        try:
            cls.apply_chat_template_kwargs = dict(
                config.data.get("apply_chat_template_kwargs", {})
            )
        except Exception:
            pass

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        # Per-turn generation budget: defaults to response_length (no extra limit per turn,
        # only the total budget constrains output). Can be overridden in agent_extra.
        cls.per_turn_max_tokens = int(agent_extra.get("per_turn_max_tokens", 0)) or cls.response_length

        # Reward mode (cached at class level): controls whether per-turn CMI is computed.
        # When set to "qa", compute_turn_cmi is skipped entirely (no embedding/logprob calls),
        # and only the final QA reward (added by TrajectoryRewardManager) drives the gradient.
        cls._reward_mode = os.environ.get("MEMBUILDER_REWARD_MODE", "").strip().lower()
        if cls._reward_mode == "qa":
            logger.info("[AgentLoop] MEMBUILDER_REWARD_MODE=qa → skipping per-turn CMI computation")

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> AgentLoopOutput:
        # Import prompts lazily to avoid circular imports at registration time
        from training.agentic.prompts import (
            build_turn_messages,
            format_memory_list,
            format_session_text,
        )

        # ---- Extract trajectory data from dataset ----
        session_messages_list: List[List[Dict[str, str]]] = list(
            kwargs.get("session_messages_list") or []
        )
        session_timestamps: List[str] = list(
            kwargs.get("session_timestamps") or []
        )
        state_before_list: List[Dict[str, Any]] = list(
            kwargs.get("state_before_list") or []
        )
        questions: List[Dict[str, Any]] = list(kwargs.get("questions") or [])

        metrics: Dict[str, Any] = {}
        request_id = uuid4().hex

        # Initialize memory from first session's state_before (or empty)
        initial_state = state_before_list[0] if state_before_list else {}
        memory = LightMemoryState(initial_state)

        # ---- Trajectory accumulators ----
        # Plan A: response sequence contains ONLY generation tokens (mask=1).
        # No observation tokens are appended — each turn is generated with an
        # independent full prompt (memory + session) by vLLM, but only the
        # model's JSON output is kept in the training sequence.  This makes
        # response_length budget scale with turn count × gen_tokens (~500/turn)
        # instead of turn count × (obs + gen) (~3500/turn).
        prompt_ids: List[int] = []
        response_ids_accum: List[int] = []
        response_mask: List[int] = []

        turn_rewards: List[float] = []
        turn_boundaries: List[int] = []
        format_valid_per_turn: List[bool] = []
        n_action_per_turn: List[int] = []

        num_sessions = min(len(session_messages_list), self.max_turns)

        for turn_idx in range(num_sessions):
            session_msgs = session_messages_list[turn_idx]
            timestamp = (
                session_timestamps[turn_idx]
                if turn_idx < len(session_timestamps)
                else ""
            )

            # ---- (a) Build turn prompt ----
            session_text = format_session_text(session_msgs)
            relevant = memory.get_relevant(session_text, self.top_k)

            turn_messages = build_turn_messages(
                core_memory=memory.core_memory,
                episodic_memories=format_memory_list(relevant["episodic"]),
                semantic_memories=format_memory_list(relevant["semantic"]),
                procedural_memories=format_memory_list(relevant["procedural"]),
                session_text=session_text,
                session_index=turn_idx,
                session_date=timestamp,
                top_k=self.top_k,
            )

            # ---- (b) Tokenize turn prompt ----
            # Full prompt (with system) for generation — each turn is independent
            full_turn_ids = await self.loop.run_in_executor(
                None,
                lambda m=turn_messages: self.tokenizer.apply_chat_template(
                    m,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

            if turn_idx == 0:
                # Turn 0's full prompt becomes the training prompt_ids
                prompt_ids = list(full_turn_ids)

            # ---- (c) Length cap check before generation ----
            # (Plan A: no observation tokens in response sequence —
            #  only gen tokens are appended, so budget is used efficiently)
            if len(response_mask) >= self.response_length:
                logger.warning(
                    f"[AgentLoop] response_length cap before gen @turn {turn_idx + 1}/{num_sessions}"
                )
                break

            # ---- (d) Generate — use only the current turn's full prompt ----
            # Cap per-turn generation to avoid wasting tokens on verbose output.
            # Remaining budget = response_length - tokens already generated.
            remaining_budget = self.response_length - len(response_mask)
            turn_max_tokens = min(self.per_turn_max_tokens, remaining_budget)
            turn_sampling_params = dict(sampling_params)
            turn_sampling_params["max_tokens"] = turn_max_tokens

            with simple_timer("generate_sequences", metrics):
                gen_out = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=full_turn_ids,
                    sampling_params=turn_sampling_params,
                    image_data=None,
                )
            turn_response_ids = list(gen_out.token_ids)

            # Warn if generation was truncated by max_tokens (no EOS emitted)
            if len(turn_response_ids) >= turn_max_tokens:
                logger.warning(
                    f"[AgentLoop] Turn {turn_idx + 1}/{num_sessions} truncated at "
                    f"{len(turn_response_ids)} tokens (budget={turn_max_tokens}, "
                    f"remaining={remaining_budget}). JSON may be incomplete."
                )

            response_ids_accum.extend(turn_response_ids)
            response_mask.extend([1] * len(turn_response_ids))
            turn_end_inclusive = len(response_mask) - 1
            turn_boundaries.append(turn_end_inclusive)

            # ---- (e) Decode + parse ----
            response_text = await self.loop.run_in_executor(
                None,
                lambda ids=turn_response_ids: self.tokenizer.decode(
                    ids, skip_special_tokens=True
                ),
            )
            parsed, fmt_valid = parse_unified_response(response_text)
            format_valid_per_turn.append(fmt_valid)

            # ---- (f) Execute actions + compute CMI ----
            memories_before = memory.snapshot_as_list()
            action_steps = memory.apply_actions(parsed)
            n_action_per_turn.append(len(action_steps))

            # Run CMI in thread pool to avoid blocking the async event loop.
            # CMI involves synchronous GPU embedding RPC + KDE math.
            # Skip entirely in pure-QA mode (MEMBUILDER_REWARD_MODE=qa): no embedding/
            # logprob calls, only the final QA reward (computed in reward_manager) matters.
            if self._reward_mode == "qa":
                turn_reward = 0.0
            else:
                turn_reward = await self.loop.run_in_executor(
                    None,
                    compute_turn_cmi,
                    memories_before,
                    action_steps,
                    session_text,
                )
            if not fmt_valid:
                turn_reward += self.format_penalty
            turn_rewards.append(float(turn_reward))

            # ---- (g) Length cap after generation ----
            if len(response_mask) >= self.response_length:
                logger.warning(
                    f"[AgentLoop] response_length cap after gen @turn {turn_idx + 1}/{num_sessions}"
                )
                break

        # ---- Finalize ----
        # Left-truncate prompt to max_prompt_length (keep rightmost tokens)
        if len(prompt_ids) > self.prompt_length:
            prompt_ids = prompt_ids[-self.prompt_length:]

        # Truncate response to max length
        response_ids_accum = response_ids_accum[: self.response_length]
        response_mask = response_mask[: self.response_length]
        turn_boundaries = [min(b, len(response_mask) - 1) for b in turn_boundaries]

        # Capture final memory state for downstream QA reward computation.
        # This adds no API calls — LightMemoryState is plain-text only.
        final_memory_snapshot = {
            "core_memory": memory.core_memory,
            "episodic": list(memory.episodic),
            "semantic": list(memory.semantic),
            "procedural": list(memory.procedural),
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids_accum,
            response_mask=response_mask,
            num_turns=len(turn_rewards),
            metrics=AgentLoopMetrics(
                generate_sequences=float(metrics.get("generate_sequences", 0.0)),
            ),
            extra_fields={
                "turn_rewards": turn_rewards,
                "turn_boundaries": turn_boundaries,
                "format_valid_per_turn": format_valid_per_turn,
                "n_action_per_turn": n_action_per_turn,
                "final_memory_snapshot": final_memory_snapshot,
                "questions": questions,
            },
        )


__all__ = ["MemoryBuilderAgentLoop"]
