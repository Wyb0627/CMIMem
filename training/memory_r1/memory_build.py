"""Build a searchable MemorySystem from a flat list of memory texts.

Shared by the training reward and the eval build step so retrieval semantics
are identical in both. Memories are stored WITHOUT type prefixes (single flat
bank). Embeds all texts once via embedding_func.
"""
from typing import Callable, List


def build_flat_memory_system(
    texts: List[str],
    llm_client,
    embedding_func: Callable[[List[str]], List[List[float]]],
):
    """Return a MemorySystem whose vector_store holds `texts` (flat, no prefix).

    `llm_client` is used later by generate_answer; may be None when only search
    is needed (e.g. the reward answers via a separate client).
    """
    from memory_system import MemorySystem

    memory = MemorySystem(llm_client=llm_client, embedding_func=embedding_func)
    clean = [str(t).strip() for t in (texts or []) if str(t).strip()]
    if clean:
        memory.vector_store.add(clean, [{"type": "memory"} for _ in clean])
    return memory
