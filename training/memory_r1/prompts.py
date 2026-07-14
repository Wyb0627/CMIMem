"""Memory-R1 prompts (paper Figures 9-11) + jinja2 render helpers.

The manager prompt is a single-flat-memory manager (ADD/UPDATE/DELETE/NONE).
The answer prompt is short-answer constrained so normalized Exact Match works.
Both prompts are shared by data prep, the training reward, and eval, so the
policy always sees the exact string it was trained on.
"""
from typing import Dict, List

from jinja2 import Template

# Memory Manager prompt — paper Figures 9-10 (single flat bank).
MEMORY_R1_MANAGER_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.
Based on the above four operations, the memory will change.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element
- DELETE: Delete an existing memory element
- NONE: Make no change (if the fact is already present or irrelevant)

1. Add: If the retrieved facts contain new information not present in the memory, add it by generating a new id in the id field.
2. Update: If the retrieved facts contain information already present but the information is different, update it. Keep the same id and preserve old_memory. If the retrieved fact conveys the same thing as the memory, keep the version with more detail; do NOT update if they convey the same information.
3. Delete: If the retrieved facts contradict the memory, delete it. Return the same id — do not generate a new id.
4. None: If the retrieved facts are already present, make no change.

Current memory:
{{existing_memory}}

New facts (from the latest dialogue turn):
{{new_facts}}

Return ONLY a JSON object of this exact form (no prose, no markdown fences):
{
  "memory": [
    {"id": "<id>", "text": "<updated or new text>", "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "<only for UPDATE>"}
  ]
}
"""

# Answer Agent prompt — paper Figure 11 (short-answer constrained for EM).
MEMORY_R1_ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# INSTRUCTIONS:
1. Carefully analyze all provided memories.
2. Pay special attention to timestamps to determine the answer.
3. If the memories contain contradictory information, prioritize the most recent memory.
4. If there is a question about time references (like "last year", "two months ago"), calculate the actual date based on the memory timestamp.
5. Always convert relative time references to specific dates, months, or years.
6. Focus only on the content of the memories. Do not confuse character names.
7. The answer should be less than 5-6 words.
8. IMPORTANT: Output the final answer after **Answer:**

Memories:
{{memories}}

Question: {{question}}
"""


def render_manager_prompt(existing_memory: List[str], session: Dict) -> str:
    """Render the manager prompt from a list of existing memory texts and a
    session dict {timestamp, messages:[{role,content}]}."""
    from training.memory_r1.session_loader import format_session_text

    existing = "\n".join(
        f'- {{"id": "{i}", "text": "{t}"}}' for i, t in enumerate(existing_memory)
    ) or "[Empty]"
    msgs = (session or {}).get("messages", [])
    ts = (session or {}).get("timestamp", "")
    new_facts = (f"[Session timestamp: {ts}]\n" if ts else "") + format_session_text(msgs)
    return Template(MEMORY_R1_MANAGER_PROMPT).render(
        existing_memory=existing, new_facts=(new_facts or "[Empty]")
    )


def render_answer_prompt(memories: List[str], question: str) -> str:
    """Render the short-answer prompt from a list of memory texts + question."""
    mem_block = "\n".join(f"- {m}" for m in memories) or "[Empty]"
    return Template(MEMORY_R1_ANSWER_PROMPT).render(memories=mem_block, question=question)
