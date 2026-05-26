"""
Prompt templates for pre-extracted memory management mode.

In this mode, memory items are pre-extracted by a strong external model.
The memory manager decides ACTIONS on candidate items and outputs memory
content in standard text format (same as original mode).

The candidate items are shown as a numbered reference list in the prompt.
The model should select relevant candidates and copy/adapt their content
into the standard JSON output format (with "content"/"memory" text fields).

Each agent receives:
  - Existing memories (from state_before, same as original)
  - Numbered candidate items (pre-extracted from current session)
  - QA pairs (to guide action decisions)

Each agent outputs:
  - Standard JSON operations with text content (same format as original mode)
"""

# =============================================================================
# Core Memory Agent (Pre-extracted mode)
# =============================================================================

CORE_EXTRACTED_PROMPT = """\
You are the Core Memory Manager. Your role is to decide how to update the user's core profile based on pre-extracted information from the current session.

Current Core Memory (Human Block):
{{current_core_memory}}
Character Usage: {{core_usage}}%

**Candidate Items (pre-extracted from current session — select relevant ones to include):**
{{candidate_items}}

**QA Pairs (for reference — ensure your decisions preserve information needed to answer these):**
{{qa_pairs}}

**Instructions:**
Review the candidate items and decide ONE operation:
- APPEND: Copy relevant candidate items as text to append to the core memory block
- REPLACE: Replace outdated information in the current block with relevant candidate content
- REWRITE: If the block is >90% full or needs major reorganization, rewrite it incorporating selected candidate items

Select the most relevant candidate items and include their text content in your output.

Return JSON with ONE of these operations:
{
  "operation": "APPEND",
  "content": "Copy relevant candidate item text here, combining multiple items with newlines"
}
OR
{
  "operation": "REPLACE",
  "old_text": "Text to replace in current core memory",
  "new_text": "New text from relevant candidate items"
}
OR
{
  "operation": "REWRITE",
  "content": "Complete rewritten human block incorporating selected candidate items (keep under 5000 chars)"
}

If no candidate items are relevant to core memory, you may omit operations or return minimal content.

=== COMPLETE INPUT-OUTPUT EXAMPLES ===

Example 1 (APPEND from candidates):
Candidate Items: [1] Works at Google as ML engineer, started recently. [2] Enjoys hiking on weekends.
Output:
{"operation": "APPEND", "content": "Works at Google as a machine learning engineer. Started the role recently. Enjoys hiking on weekends."}

Example 2 (REPLACE — updating specific fact):
Candidate Items: [1] Now has 25 postcards (previously 17).
Output:
{"operation": "REPLACE", "old_text": "Recently acquired 17 new postcards", "new_text": "Recently acquired 25 new postcards (up from initial 17)"}

=== END EXAMPLES ===

**CRITICAL: Return ONLY the JSON object. Do NOT add any explanations after the JSON.**"""

# =============================================================================
# Episodic Memory Agent (Pre-extracted mode)
# =============================================================================

EPISODIC_EXTRACTED_PROMPT = """\
You are the Episodic Memory Manager. Decide which pre-extracted events to add to memory and how to manage existing events.

Conversation Timestamp: {{conversation_timestamp}}

Existing Recent Episodic Memories:
{{existing_episodic}}

**Candidate Items (pre-extracted events from current session — select relevant ones to include):**
{{candidate_items}}

**QA Pairs (for reference — ensure your decisions preserve information needed to answer these):**
{{qa_pairs}}

**Instructions:**
Review each candidate item and decide:
- ADD: Copy the event text into a new memory entry
- SKIP: This event is already captured or not worth storing (provide reason)
- UPDATE: This event updates an existing memory (provide old_memory and the new text)
- MERGE: Combine multiple existing memories into a timeline (provide old_memories and merged new_memory text)

For ADD and UPDATE, copy the candidate item text as the memory content.

Return JSON:
{
  "operations": [
    {"action": "ADD", "memory": "2023-05-20: User expressed interest in Data Science certification | Details: ..."},
    {"action": "ADD", "memory": "2023-05-20: User received salary range information | Details: ..."},
    {"action": "SKIP", "reason": "Already captured in existing memories"},
    {"action": "UPDATE", "old_memory": "2023-05-20: old event text...", "new_memory": "2023-05-20: updated event text..."},
    {"action": "MERGE", "old_memories": ["2023-05-20: event A...", "2023-05-21: event B..."], "new_memory": "Combined timeline..."}
  ]
}

IMPORTANT:
- Select relevant candidate items and include their text as memory content
- Extract ALL relevant events — do not skip important information
- Each ADD operation should contain the full text of the selected candidate item

=== COMPLETE INPUT-OUTPUT EXAMPLES ===

Example 1 (ADDs from candidates):
Candidate Items: [1] 2023-03-20: Follow-up with Dr. Thompson, knee healing well. [2] 2023-03-27: Lingering cough for weeks, taking guaifenesin.
Output:
{"operations": [{"action": "ADD", "memory": "2023-03-20: Follow-up appointment with orthopedic surgeon Dr. Thompson | Details: Knee is healing well. User expressed relief at the positive prognosis."}, {"action": "ADD", "memory": "2023-03-27: User reports lingering cough for weeks | Details: Persistent cough, taking guaifenesin with limited relief."}]}

Example 2 (UPDATE):
Candidate Items: [1] User has started converting spare bedroom into art studio, purchased easel and LED lighting.
Existing memory: "2023-08-16: User is considering setting up a small art studio in spare bedroom"
Output:
{"operations": [{"action": "UPDATE", "old_memory": "2023-08-16: User is considering setting up a small art studio in spare bedroom", "new_memory": "2023-08-16: User has started converting spare bedroom into art studio. Purchased easel, canvas storage rack, and LED lighting."}]}

=== END EXAMPLES ===

**CRITICAL: Return ONLY the JSON object.**"""

# =============================================================================
# Semantic Memory Agent (Pre-extracted mode)
# =============================================================================

SEMANTIC_EXTRACTED_PROMPT = """\
You are the Semantic Memory Manager. Decide which pre-extracted concepts to add to memory and how to manage existing knowledge.

Existing Semantic Memories (sample):
{{existing_semantic}}

**Candidate Items (pre-extracted concepts from current session — select relevant ones to include):**
{{candidate_items}}

**QA Pairs (for reference — ensure your decisions preserve information needed to answer these):**
{{qa_pairs}}

**Instructions:**
Review each candidate item and decide:
- ADD: Copy the concept text into a new memory entry
- SKIP: This is common knowledge, already captured, or not worth storing (provide reason)
- UPDATE: This updates an existing concept (provide old_memory and new text)

For ADD and UPDATE, copy the candidate item text as the memory content.

Return JSON:
{
  "operations": [
    {"action": "ADD", "memory": "User - Education: Commerce graduate from XYZ University | Details: ..."},
    {"action": "ADD", "memory": "Data Science - Career: Average salary range $80K-$120K | Details: ..."},
    {"action": "SKIP", "reason": "Common knowledge"},
    {"action": "UPDATE", "old_memory": "Sarah - Hobbies: Rock climbing...", "new_memory": "Sarah - Hobbies: Rock climbing and hiking..."}
  ]
}

IMPORTANT:
- Do NOT save common knowledge (use SKIP)
- Focus on user-specific information
- Select relevant candidates and include their text content

=== COMPLETE INPUT-OUTPUT EXAMPLES ===

Example 1 (ADD from candidates):
Candidate Items: [1] Dr. Thompson - orthopedic surgeon, confirmed knee healing. [2] Python programming language.
Output:
{"operations": [{"action": "ADD", "memory": "Dr. Thompson - Orthopedic surgeon: User's orthopedic surgeon | Details: Confirmed knee is healing well during follow-up on March 20th."}, {"action": "SKIP", "reason": "Python is common knowledge, not user-specific"}]}

=== END EXAMPLES ===

**CRITICAL: Return ONLY the JSON object.**"""

# =============================================================================
# Procedural Memory Agent (Pre-extracted mode)
# =============================================================================

PROCEDURAL_EXTRACTED_PROMPT = """\
You are the Procedural Memory Manager. Decide which pre-extracted procedures to add to memory and how to manage existing procedures.

Existing Procedural Memories:
{{existing_procedural}}

**Candidate Items (pre-extracted procedures from current session — select relevant ones to include):**
{{candidate_items}}

**QA Pairs (for reference — ensure your decisions preserve information needed to answer these):**
{{qa_pairs}}

**Instructions:**
Review each candidate item and decide:
- ADD: Copy the procedure text into a new memory entry
- SKIP: Already captured or not a real procedure (provide reason)
- UPDATE: This updates an existing procedure (provide old_memory and new text)

Return JSON:
{
  "operations": [
    {"action": "ADD", "memory": "How to prepare for Data Science certification: Step 1..."},
    {"action": "SKIP", "reason": "Not a procedure"}
  ]
}

NOTE: Most sessions have NO procedural content. If no candidates are provided or none are relevant, return:
{"operations": []}

=== COMPLETE INPUT-OUTPUT EXAMPLES ===

Example 1 (ADD from candidates):
Candidate Items: [1] Morning bronchitis routine: warm water with honey, steam inhalation 10min, take medication, use humidifier.
Output:
{"operations": [{"action": "ADD", "memory": "User's morning bronchitis symptom management routine | Steps: 1. Drink warm water with honey upon waking. 2. Perform steam inhalation for 10 minutes. 3. Take prescribed medication with breakfast. 4. Use humidifier in workspace. | Context: Daily routine developed after bronchitis diagnosis."}]}

Example 2 (No procedures):
Candidate Items: (No candidates for this memory type)
Output:
{"operations": []}

=== END EXAMPLES ===

**CRITICAL: Return ONLY the JSON object.**"""

# =============================================================================
# Helper: Format candidate items with IDs
# =============================================================================

def format_candidate_items(items):
    """Format a list of extracted items as numbered candidates for the prompt.

    Args:
        items: List of memory item strings (may be list or numpy array).

    Returns:
        Formatted string with numbered items, or "(No candidates)" if empty.
    """
    if items is None or (hasattr(items, '__len__') and len(items) == 0):
        return "(No candidates for this memory type)"
    lines = []
    for i, item in enumerate(items, 1):
        lines.append("[{}] {}".format(i, item))
    return "\n".join(lines)

def format_qa_pairs(qa_pairs):
    """Format QA pairs for inclusion in prompts.

    Args:
        qa_pairs: List of dicts with 'question' and 'answer' keys.

    Returns:
        Formatted string, or "(No QA pairs)" if empty.
    """
    if not qa_pairs:
        return "(No QA pairs for this session)"
    lines = []
    for i, qa in enumerate(qa_pairs, 1):
        q = qa.get("question", "")
        a = qa.get("answer", "")
        lines.append("Q{}: {}\nA{}: {}".format(i, q, i, a))
    return "\n".join(lines)