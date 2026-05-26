"""
Type-specific QA generation prompts with few-shot examples from LongMemEval.

Each question type has its own prompt template with:
- Clear type definition and constraints
- 1-2 real examples from LongMemEval (including source conversation turns)
- Applicability check (model can indicate NOT_APPLICABLE)
"""

# Common preamble shared across all type-specific prompts
_COMMON_PREAMBLE = """You are a QA pair generator for evaluating a memory system's effectiveness.
Given the user's conversation session and existing memories, generate exactly ONE question-answer pair.

**CONTEXT PROVIDED:**
- Core Memory (user profile): {core_memory}
- Episodic Memories (past events): {episodic_memories}
- Semantic Memories (facts/knowledge): {semantic_memories}
- Procedural Memories (how-to): {procedural_memories}
- Current Session ({session_timestamp}):
{current_session}

**RULES:**
1. The answer MUST be unambiguously derivable from the provided context.
2. Do NOT ask questions whose answers are subjective or uncertain.
3. Questions should be phrased as if asked by the user to their assistant.
4. Keep answers concise and factual (1-2 sentences max).
5. If this session does NOT contain sufficient information for this question type, return {{"not_applicable": true}}.
"""

# ==============================================================================
# Type 1: single-session-user
# ==============================================================================
PROMPT_SINGLE_SESSION_USER = _COMMON_PREAMBLE + """
**QUESTION TYPE: single-session-user**
Generate a question about factual information the USER explicitly stated or did in the current session.
Focus on user-provided facts: their experiences, plans, actions, personal details they mentioned.
Do NOT ask about information provided by the assistant.

**FEW-SHOT EXAMPLES:**

--- Example 1 ---
Session excerpt:
[user]: I'm trying to organize my life a bit better, can you recommend some task management apps that can help me prioritize my work and personal tasks? I've been using a planner, but I think I need something more digital.
[assistant]: Making the leap from a planner to a digital task management system! There are many excellent task management apps out there...
[user]: Thanks! By the way, I graduated with a degree in Business Administration and now I'm working in project management, so I really need something that helps with both.

Generated QA:
{{"question": "What degree did I graduate with?", "answer": "Business Administration"}}

--- Example 2 ---
Session excerpt:
[user]: I've been commuting to work recently and it takes me about 45 minutes each way. I'm thinking of listening to audiobooks during the commute.
[assistant]: That's a great idea! 45 minutes each way gives you plenty of time...

Generated QA:
{{"question": "How long is my daily commute to work?", "answer": "45 minutes each way"}}

**OUTPUT FORMAT:**
Return ONLY a JSON object:
{{"question": "...", "answer": "...", "type": "single-session-user", "source": "current_session"}}

Or if not applicable:
{{"not_applicable": true}}
"""

# ==============================================================================
# Type 2: single-session-assistant
# ==============================================================================
PROMPT_SINGLE_SESSION_ASSISTANT = _COMMON_PREAMBLE + """
**QUESTION TYPE: single-session-assistant**
Generate a question about specific information, recommendations, or content the ASSISTANT provided in the current session.
Focus on assistant-generated content: recommendations, explanations, answers to user queries, structured data created.
The user should be asking to recall what the assistant told them.

**FEW-SHOT EXAMPLES:**

--- Example 1 ---
Session excerpt:
[user]: can u create a shift rotation sheet for GM social media agents
[assistant]: Sure, I can help you create a shift rotation sheet for GM social media agents. Here's an example template:
| Agent | Monday | ... | Sunday |
| Admon | Day Shift (8am-4pm) | ... | Day Shift (8am-4pm) |
| Beth | Swing (4pm-12am) | ... | Off |
[user]: 7 agents

Generated QA:
{{"question": "I'm checking our previous chat about the shift rotation sheet for GM social media agents. Can you remind me what was the rotation for Admon on a Sunday?", "answer": "Admon was assigned to the 8 am - 4 pm (Day Shift) on Sundays."}}

--- Example 2 ---
Session excerpt:
[user]: I'm planning to visit Bandung and want some restaurant recommendations.
[assistant]: Bandung has some wonderful dining options! For a unique experience, I recommend Miss Bee Providore - it's a charming European-style cafe known for their pastries and brunch menu. You might also enjoy...

Generated QA:
{{"question": "Can you remind me of the name of that restaurant you recommended when I was planning my Bandung trip?", "answer": "Miss Bee Providore"}}

**OUTPUT FORMAT:**
Return ONLY a JSON object:
{{"question": "...", "answer": "...", "type": "single-session-assistant", "source": "current_session"}}

Or if not applicable:
{{"not_applicable": true}}
"""

# ==============================================================================
# Type 3: single-session-preference
# ==============================================================================
PROMPT_SINGLE_SESSION_PREFERENCE = _COMMON_PREAMBLE + """
**QUESTION TYPE: single-session-preference**
Generate a question that tests whether the system captured the user's PREFERENCES, OPINIONS, or STYLE from the current session.
The answer should describe what the user would prefer (detailed), not just a yes/no.
Focus on: tool preferences, style choices, brand loyalty, workflow preferences, taste.

**FEW-SHOT EXAMPLES:**

--- Example 1 ---
Session excerpt:
[user]: I'm trying to learn more about some advanced settings for video editing with Adobe Premiere Pro, which I enjoy to use. Can you give me some tips on where to start?
[assistant]: Adobe Premiere Pro is a powerful video editing software with a vast array of advanced settings. Here are some tips to get you started: 1. Master the Timeline Panel...
[user]: I want to focus specifically on color grading and audio mixing in Premiere Pro.

Generated QA:
{{"question": "Can you recommend some resources where I can learn more about video editing?", "answer": "The user would prefer responses that suggest resources specifically tailored to Adobe Premiere Pro, especially those that delve into its advanced settings. They might not prefer general video editing resources or resources related to other video editing software."}}

--- Example 2 ---
Session excerpt:
[user]: I'm looking to upgrade my camera setup. I currently use a Sony A7III and I really love the Sony ecosystem.
[assistant]: Great choice! The Sony A7III is excellent. For upgrades within the Sony ecosystem, you might consider...

Generated QA:
{{"question": "Can you suggest some accessories that would complement my current photography setup?", "answer": "The user would prefer suggestions of Sony-compatible accessories or high-quality photography gear that integrates well with the Sony ecosystem."}}

**OUTPUT FORMAT:**
Return ONLY a JSON object:
{{"question": "...", "answer": "...", "type": "single-session-preference", "source": "current_session"}}

Or if not applicable:
{{"not_applicable": true}}
"""

# ==============================================================================
# Type 4: temporal-reasoning
# ==============================================================================
PROMPT_TEMPORAL_REASONING = _COMMON_PREAMBLE + """
**QUESTION TYPE: temporal-reasoning**
Generate a question that requires TIME-BASED reasoning: calculating durations, comparing dates, ordering events chronologically, or determining when something happened.
The question MUST involve temporal anchors (dates, "X days ago", "before/after Y", "how long between").
The answer should involve temporal calculation or comparison.

**FEW-SHOT EXAMPLES:**

--- Example 1 ---
Session excerpt:
[user]: I'm looking for some information on modern art movements. I just got back from a guided tour at the Museum of Modern Art focused on 20th-century modern art movements, and it really sparked my interest.
[assistant]: What a great experience! The Museum of Modern Art (MoMA) is an iconic institution...
[Later in conversation, session date: March 15]
[user]: I'm planning to visit the Metropolitan Museum of Art next week for the 'Ancient Civilizations' exhibit.
[Session date of MoMA visit was March 8]

Generated QA:
{{"question": "How many days passed between my visit to the Museum of Modern Art (MoMA) and the 'Ancient Civilizations' exhibit at the Metropolitan Museum of Art?", "answer": "7 days."}}

--- Example 2 ---
Session excerpt:
[user]: I helped my friend prepare the nursery last weekend. Before that, I was at my cousin's baby shower picking out stuff. And tomorrow I have the prenatal yoga class.
[Session timestamp provides date context]

Generated QA:
{{"question": "Which three events happened in the order from first to last: the day I helped my friend prepare the nursery, my cousin's baby shower, and the prenatal yoga class?", "answer": "First the baby shower, then preparing the nursery, then the prenatal yoga class."}}

**IMPORTANT:** The temporal information must be derivable from session timestamps, explicit dates, or relative time references in the conversation. Do NOT invent dates.

**OUTPUT FORMAT:**
Return ONLY a JSON object:
{{"question": "...", "answer": "...", "type": "temporal-reasoning", "source": "current_session"}}

Or if not applicable (e.g., no temporal information in the session):
{{"not_applicable": true}}
"""

# ==============================================================================
# Type 5: multi-session
# ==============================================================================
PROMPT_MULTI_SESSION = _COMMON_PREAMBLE + """
**QUESTION TYPE: multi-session**
Generate a question that REQUIRES combining information from MULTIPLE sessions (current + historical memories).
The answer must aggregate, count, compare, or synthesize facts from different time periods.
This type tests whether the memory system preserves cross-session knowledge.

**FEW-SHOT EXAMPLES:**

--- Example 1 ---
Existing memories:
- "User mentioned needing to pick up dry cleaning for navy blue blazer" (Session 3)
- "User bought a red dress for a wedding and needs to return the wrong-size shoes" (Session 7)
Current session:
[user]: I need help organizing my closet. Also, I still need to pick up my dry cleaning for the navy blue blazer I wore to a meeting.

Generated QA:
{{"question": "How many items of clothing do I need to pick up or return from a store?", "answer": "3 (dry cleaning blazer, return wrong-size shoes, and one more from earlier sessions)"}}

--- Example 2 ---
Existing memories:
- "User led a marketing campaign project at work" (Session 2)
Current session:
[user]: I'm now leading a new product launch project. It's my second major project as a team lead.

Generated QA:
{{"question": "How many projects have I led or am currently leading?", "answer": "2"}}

**IMPORTANT:** If no historical memories are provided (episodic/semantic are empty), you may still generate a multi-session style question if the current session references past events, mentions counts/totals of things done over time, or aggregates information from multiple topics discussed within the session. The key is testing information SYNTHESIS rather than single-fact recall.

**OUTPUT FORMAT:**
Return ONLY a JSON object:
{{"question": "...", "answer": "...", "type": "multi-session", "source": "cross_session"}}

Or if not applicable (e.g., no relevant historical memories):
{{"not_applicable": true}}
"""

# ==============================================================================
# Type 6: knowledge-update
# ==============================================================================
PROMPT_KNOWLEDGE_UPDATE = _COMMON_PREAMBLE + """
**QUESTION TYPE: knowledge-update**
Generate a question that tests whether the system correctly tracks CHANGES or UPDATES to previously known facts.
The answer should reflect the MOST RECENT / UPDATED value, not the old one.
This tests: job changes, moved cities, new personal bests, updated preferences, corrected information.

**FEW-SHOT EXAMPLES:**

--- Example 1 ---
Existing memories:
- "User's personal best in 5K run was 27 minutes" (Session 4)
Current session:
[user]: I'm trying to plan out my fitness schedule. I just ran a charity 5K in 25 minutes and 50 seconds - new personal best!
[assistant]: Congratulations on the new PR! That's a great improvement...

Generated QA:
{{"question": "What was my personal best time in the charity 5K run?", "answer": "25 minutes and 50 seconds (or 25:50)"}}

--- Example 2 ---
Existing memories:
- "User has tried 3 Korean restaurants in their city" (Session 6)
Current session:
[user]: I just tried another Korean restaurant last night - that makes four I've been to in this city now.

Generated QA:
{{"question": "How many Korean restaurants have I tried in my city?", "answer": "four"}}

**IMPORTANT:** This type tests whether the system tracks the LATEST state of evolving information. Valid scenarios include:
1. Something explicitly CHANGED from a previously known state (ideal case)
2. The session mentions a CURRENT value for something that naturally evolves over time (counts, statuses, progress, rankings, scores)
3. The user states a NEW fact that supersedes or adds to prior knowledge (new hobby, recent achievement, latest plan)

Only return not_applicable if the session contains NO evolving/updatable facts whatsoever (e.g., purely abstract discussion with no personal information).

**OUTPUT FORMAT:**
Return ONLY a JSON object:
{{"question": "...", "answer": "...", "type": "knowledge-update", "source": "cross_session"}}

Or if not applicable (e.g., no knowledge updates in this session):
{{"not_applicable": true}}
"""

# ==============================================================================
# Registry: type -> prompt template
# ==============================================================================
QA_TYPE_PROMPTS = {
    "single-session-user": PROMPT_SINGLE_SESSION_USER,
    "single-session-assistant": PROMPT_SINGLE_SESSION_ASSISTANT,
    "single-session-preference": PROMPT_SINGLE_SESSION_PREFERENCE,
    "temporal-reasoning": PROMPT_TEMPORAL_REASONING,
    "multi-session": PROMPT_MULTI_SESSION,
    "knowledge-update": PROMPT_KNOWLEDGE_UPDATE,
}

ALL_QA_TYPES = list(QA_TYPE_PROMPTS.keys())
