"""Configuration and constants for MemBuilder."""

import os

# =============================================================================
# API Configuration (from environment or defaults)
# =============================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

# Separate embedding endpoint (for when chat and embedding use different providers,
# e.g., vLLM for chat + external API for embeddings)
OPENAI_EMBEDDINGS_BASE_URL = os.environ.get("OPENAI_EMBEDDINGS_BASE_URL", "")
OPENAI_EMBEDDINGS_API_KEY = os.environ.get("OPENAI_EMBEDDINGS_API_KEY", "")

# =============================================================================
# Model Configuration (Paper Section 4.1 - Implementation Details)
# =============================================================================

# Base model for memory construction
BASE_MODEL = "Qwen3-4B-Instruct-2507"

# Expert model for SFT trajectory collection (env override for non-Claude providers)
SFT_EXPERT_MODEL = os.environ.get("SFT_EXPERT_MODEL", "claude-sonnet-4-5")

# Expert model for synthetic QA generation (env override for non-Claude providers)
QA_GENERATION_MODEL = os.environ.get("QA_GENERATION_MODEL", "deepseek-v4-pro")

# Answer model for RL reward and QA generation
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "deepseek-v4-pro")

# LLM Judge model for evaluation (deepseek-v4-pro)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v4-flash")

# Embedding model for retrieval (env override for non-OpenAI providers)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")

# =============================================================================
# Memory Architecture Configuration (Paper Section 3.2)
# =============================================================================

# Core Memory limits
CORE_MEMORY_CHAR_LIMIT = 5000

# Memory type prefixes for vector database
MEMORY_PREFIXES = {
    "episodic": "[EPISODIC]",
    "semantic": "[SEMANTIC]",
    "procedural": "[PROCEDURAL]"
}

# Agent action spaces (as described in paper Section 3.2)
AGENT_ACTIONS = {
    "core": ["APPEND", "REPLACE", "REWRITE"],
    "episodic": ["ADD", "UPDATE", "MERGE"],
    "semantic": ["ADD", "UPDATE", "SKIP"],
    "procedural": ["ADD", "UPDATE"]
}

# =============================================================================
# Retrieval Settings (Paper Appendix - Embedding and Retrieval)
# =============================================================================

# Memory construction: retrieve top-K for agent context
MEMORY_CONSTRUCTION_TOP_K = int(os.environ.get("MEMORY_CONSTRUCTION_TOP_K", "20"))

# Embedding query truncation: limit query text to last N characters.
# 8192 tokens ≈ 32000 chars for English (4 chars/token avg).
# Only the most recent messages are kept (recency = relevance for retrieval).
EMBEDDING_QUERY_MAX_CHARS = 32000

# QA answering: retrieve top-10 for answer generation
QA_ANSWERING_TOP_K = 10

# QA generation: retrieve top-20 for synthetic question context
QA_GENERATION_TOP_K = 20

# RAG baselines: retrieve top-5 chunks
RAG_BASELINE_TOP_K = 5

# Default (for backward compatibility)
DEFAULT_TOP_K = QA_ANSWERING_TOP_K
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL

# =============================================================================
# ADRPO hyperparameters (Paper Appendix - RL Training)
# =============================================================================
ADRPO_CONFIG = {
    "learning_rate": 1e-6,
    "batch_size": 128,
    "num_rollouts": 8,
    "clip_epsilon": 0.2,
    "alpha": 4,  # Contribution-aware gradient weight
    "lambda_length": 0.8,  # Length penalty coefficient
    "theta_min": 150,  # Core memory penalty-free threshold
    "theta_max": 400,  # Core memory full-penalty threshold
    "gamma_l": 0.5,  # Other memory tolerance range lower
    "gamma_u": 1.3,  # Other memory tolerance range upper
}

# =============================================================================
# CMI-Mem Configuration (Our Method - Conditional Mutual Information Reward)
# =============================================================================
# CMI computation method: "logprob" | "embedding" | "entropy"
CMI_METHOD = os.environ.get("CMI_METHOD", "logprob")

# Reward shaping strategy: "threshold" | "gaussian"
CMI_SHAPING = os.environ.get("CMI_SHAPING", "gaussian")

# Weight for CMI in combined mode (cmi_plus_qa): reward = alpha*CMI + (1-alpha)*QA
CMI_WEIGHT = float(os.environ.get("CMI_WEIGHT", "0.5"))

# =============================================================================
# Memory-Guided RAG Configuration
# =============================================================================
# Which memory types should have original conversation chunks appended as
# supplementary context when the answer model generates answers.
# Comma-separated list of memory types: "episodic", "semantic", "procedural", "core".
# Empty string = disabled (pure memory mode, backward compatible).
# Example: RAG_CHUNK_TYPES="episodic|semantic"
# Uses "|" as delimiter (not comma) to avoid conflicts with cluster ENV_VARS parsing.
RAG_CHUNK_TYPES = [t.strip() for t in os.environ.get("RAG_CHUNK_TYPES", "").split("|") if t.strip()]

# Number of turns (messages) per chunk when splitting session conversations.
CHUNK_TURN_SIZE = int(os.environ.get("CHUNK_TURN_SIZE", "4"))
