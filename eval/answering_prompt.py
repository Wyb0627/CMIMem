"""Unified QA prompt templates for MemoryAgentBench evaluation."""

from typing import List, Tuple


UNIFIED_ANSWERING_PROMPT = """You are a helpful assistant with access to the user's memories.
Answer the user's question using ONLY the information in the retrieved memories below.

Rules:
- Treat the retrieved memories as the sole source of truth. Do NOT use your own world knowledge or training-time facts to answer or to "correct" the memories.
- If the memories assert a fact that contradicts what you would otherwise believe, follow the memories.
- Do NOT add facts, names, dates, or relations that are not present in the memories.
- If the memories do not contain enough information to answer, say "I don't have enough information."
- Be concise.

## Retrieved Memories
{memories}

## Question
{question}

## Answer"""


# MemoryAgentBench source-specific templates, mirroring the original MAB
# protocol in utils/templates.py. The benchmark inherits scoring rules from
# heterogeneous source datasets, several of which require the model to emit
# the answer in a *very* constrained format that ``UNIFIED_ANSWERING_PROMPT``
# breaks:
#
#   * ICL (Multi-hop Completion Challenge): SEM expects ``label: N`` only.
#   * Recsys-Redial: Recall@5 parser expects a numbered ``1.movie\n2.movie``
#     list of 20 recommendations.
#   * InfBench summarization (Summ): ROUGE-L F1 needs a long, free-form plot
#     summary; the unified prompt's "if memories not enough, say I don't have
#     enough information" line short-circuits the model into a 32-character
#     refusal on every sample.
#   * DetectiveQA (DetQA): multi-choice answer; same refusal short-circuit
#     drops ~40% of samples to "I don't have enough information.".
#
# We feed our retrieved memories in place of the raw long context so the
# memory system is still in the loop, but we drop the refusal language and
# preserve the original task framing for these four source families.

_MAB_ICL_PROMPT = """Use the provided mapping from the context to numerical label to assign a numerical label to the question. The relevant labelled examples have been retrieved from your memory and are listed below.

[Memorized Examples]
{memories}

Only output "label: {{label}}" and nothing else.

Question: {question}

label:"""


_MAB_RECSYS_PROMPT = """Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. The relevant dialogues have been retrieved from your memory and are listed below.

[Memorized Dialogues]
{memories}

Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences.

For Example:

[Conversation]

The recommendations are:
1.movie1
2.movie2
...

Here is the conversation: {question}

The recommendations are:
"""


_MAB_INFBENCH_SUM_PROMPT = """You are given memorized excerpts from a book and you are tasked to summarize it. The relevant excerpts have been retrieved from your memory and are listed below.

[Memorized Excerpts]
{memories}

{question}

Now summarize the book."""


_MAB_DETECTIVE_QA_PROMPT = """Based on the memorized excerpts from the book below, answer the question. You are required to answer the question based on the strict output format.

[Memorized Excerpts]
{memories}

{question}

Strict rules:
- This is a multiple-choice question with options A, B, C, D. You MUST choose exactly one of these options. Do NOT answer "None of the above" or invent a new option.
- Reproduce the chosen option text VERBATIM as it appears in the question above (preserve the option letter, the exact spelling, romanization, punctuation, and spacing — even if a name looks misspelled or unusual, do NOT normalize or "correct" it from your prior knowledge).
- Use ONLY the memorized excerpts as evidence. Do not add facts that are not in the excerpts.
"""


def _format_memories_block(memories: List[str]) -> str:
    if not memories:
        return "(No memories retrieved)"
    return "\n".join(f"- {m}" for m in memories)


def format_answering_prompt(memories: List[str], question: str) -> str:
    """将检索到的记忆和问题格式化为统一的问答 prompt。

    Args:
        memories: 检索到的记忆文本列表
        question: 用户问题

    Returns:
        格式化后的完整 prompt 字符串
    """
    return UNIFIED_ANSWERING_PROMPT.format(
        memories=_format_memories_block(memories),
        question=question,
    )


def is_mab_protocol_source(source: str) -> bool:
    """Return True when the MAB ``source`` needs the original task-specific protocol.

    Covers four source families that have idiosyncratic output-format / scoring
    requirements that ``UNIFIED_ANSWERING_PROMPT`` breaks:

      - ICL (TTL/MCC): ``icl_*``
      - Recsys-Redial (TTL/Recom): ``recsys_redial*``
      - InfBench summarization (LRU/Summ): ``infbench_sum*``
      - DetectiveQA (LRU/DetQA): ``detective_qa*``
    """
    src = (source or "").lower()
    return (
        src.startswith("icl_")
        or "recsys_redial" in src
        or "infbench_sum" in src
        or "detective_qa" in src
    )


def is_mab_ttl_source(source: str) -> bool:
    """Return True when the MemoryAgentBench source belongs to the TTL split.

    Kept for backwards compatibility with callers that only need to know about
    the TTL families (ICL + Recsys-Redial). For the broader set of sources
    that need the original MAB protocol (which now also includes the LRU
    families InfBench Sum and DetectiveQA), use ``is_mab_protocol_source``.
    """
    src = (source or "").lower()
    return src.startswith("icl_") or "recsys_redial" in src


def build_mab_answering_prompt(
    source: str,
    memories: List[str],
    question: str,
) -> Tuple[str, "int | None"]:
    """Build the answering prompt + optional ``max_tokens`` budget for one MAB sample.

    For four MAB source families (ICL, Recsys-Redial, InfBench summarization,
    DetectiveQA) we use the original MemoryAgentBench prompt protocol so the
    model emits the format the SEM / Recall@5 / ROUGE-L graders expect, and
    so it does not short-circuit into "I don't have enough information." on
    long-form tasks. For all other sources we fall back to
    ``UNIFIED_ANSWERING_PROMPT`` (the existing behavior) and return ``None``
    for ``max_tokens`` so the caller keeps whatever budget it was using
    before.

    Args:
        source: MAB ``source`` field (e.g. ``icl_banking77_5900shot_balance``).
        memories: Retrieved memory strings (already filtered by ``user_id``).
        question: The query text from the parquet ``questions`` field.

    Returns:
        ``(prompt, max_tokens)``. ``max_tokens`` mirrors the
        ``generation_max_length`` from the original MAB data configs (20 for
        ICL, 300 for Recsys, 1200 for InfBench Sum, 2000 for DetectiveQA);
        for sources outside the four protocol families it is ``None``,
        meaning the caller should preserve its existing default.
    """
    src = (source or "").lower()
    memories_block = _format_memories_block(memories)

    if src.startswith("icl_"):
        return (
            _MAB_ICL_PROMPT.format(memories=memories_block, question=question),
            20,
        )
    if "recsys_redial" in src:
        return (
            _MAB_RECSYS_PROMPT.format(memories=memories_block, question=question),
            300,
        )
    if "infbench_sum" in src:
        return (
            _MAB_INFBENCH_SUM_PROMPT.format(memories=memories_block, question=question),
            1200,
        )
    if "detective_qa" in src:
        return (
            _MAB_DETECTIVE_QA_PROMPT.format(memories=memories_block, question=question),
            2000,
        )
    return (
        UNIFIED_ANSWERING_PROMPT.format(memories=memories_block, question=question),
        None,
    )
