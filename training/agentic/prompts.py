"""Unified prompt templates for agentic RL memory builder.

In agentic mode, each turn corresponds to one session. The model outputs
compact DSL lines (one per memory operation) to minimize token usage.

This avoids the 4-agent split of the original MemBuilder and enables
sequential action execution with per-action CMI reward.
"""

# ---------------------------------------------------------------------------
# System prompt (shared across all turns)
# ---------------------------------------------------------------------------

AGENTIC_SYSTEM_PROMPT = """\
You are a Memory Manager. After each conversation session, output memory \
operations in compact DSL format (one operation per line).

Memory types: CORE, EPISODIC, SEMANTIC, PROCEDURAL.

Format — each line is TYPE:ACTION|field1|field2:
  CORE:SKIP
  CORE:APPEND|new info to add
  CORE:REPLACE|old text|new text
  EPISODIC:SKIP
  EPISODIC:ADD|YYYY-MM-DD: event summary
  EPISODIC:UPDATE|old memory text|new memory text
  SEMANTIC:SKIP
  SEMANTIC:ADD|Topic - concise fact
  SEMANTIC:UPDATE|old memory text|new memory text
  PROCEDURAL:SKIP
  PROCEDURAL:ADD|How to X: 1. step 2. step

Rules:
- Output ONLY DSL lines. No explanations, no markdown, no JSON.
- Every response MUST include exactly one line for CORE (SKIP/APPEND/REPLACE).
- For EPISODIC, SEMANTIC, PROCEDURAL: output one or more ADD/UPDATE lines, \
or a single SKIP line if nothing relevant.
- Be concise: 1 sentence per memory entry. No redundancy.
- Use | as field separator. Do not use | inside field values."""

# ---------------------------------------------------------------------------
# Turn-level user prompt template
# ---------------------------------------------------------------------------

AGENTIC_TURN_TEMPLATE = """\
[Memory] Core: {core_memory}
[Memory] Episodic: {episodic_memories}
[Memory] Semantic: {semantic_memories}
[Memory] Procedural: {procedural_memories}
[Session #{session_index}, {session_date}]
{session_text}
Output memory operations:"""


def build_turn_messages(
    core_memory: str,
    episodic_memories: str,
    semantic_memories: str,
    procedural_memories: str,
    session_text: str,
    session_index: int,
    session_date: str,
    top_k: int = 10,
) -> list[dict[str, str]]:
    """Build chat messages for one turn of the agentic agent loop.

    Returns a list of two dicts (system + user) suitable for
    tokenizer.apply_chat_template().
    """
    user_content = AGENTIC_TURN_TEMPLATE.format(
        core_memory=core_memory or "[empty]",
        episodic_memories=episodic_memories or "[none]",
        semantic_memories=semantic_memories or "[none]",
        procedural_memories=procedural_memories or "[none]",
        session_text=session_text,
        session_index=session_index,
        session_date=session_date or "unknown",
        top_k=top_k,
    )
    return [
        {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def format_session_text(messages: list[dict[str, str]]) -> str:
    """Format a session's messages into readable text."""
    lines: list[str] = []
    for msg in messages or []:
        role = (msg.get("role") or "user").strip()
        content = msg.get("content") or msg.get("text") or ""
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def format_memory_list(memories: list[str], max_items: int = 0) -> str:
    """Format a list of memory strings as a bullet list."""
    if not memories:
        return "[none]"
    items = memories[:max_items] if max_items > 0 else memories
    return "\n".join(f"- {m}" for m in items)
