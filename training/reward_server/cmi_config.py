"""
CMI (Conditional Mutual Information) Reward Configuration.

All hyperparameters for the CMI-Mem reward computation.
Supports JSON file hot-reload and environment variable overrides.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# =============================================================================
# Default Configuration
# =============================================================================

CMI_DEFAULT_CONFIG: Dict[str, Any] = {
    # ── CMI Computation Method ──────────────────────────────────────────────
    # "logprob"   : log P difference method (requires logprobs API support)
    # "embedding" : embedding cosine similarity method (fastest, no logprobs needed)
    # "entropy"   : perplexity-based entropy reduction method
    "cmi_method": "logprob",

    # ── Reward Shaping Strategy ─────────────────────────────────────────────
    # "threshold" : fixed interval [low, high], in-range gets positive reward
    # "gaussian"  : bell curve centered at optimal CMI value
    "cmi_shaping": "gaussian",

    # ── Threshold Shaping Parameters ────────────────────────────────────────
    "cmi_threshold_low": 0.1,       # Lower bound of ideal CMI range
    "cmi_threshold_high": 0.8,      # Upper bound of ideal CMI range
    "cmi_threshold_in_reward": 1.0,   # Max reward when CMI is in range
    "cmi_threshold_out_reward": -0.3, # Min reward when CMI is far out of range

    # ── Gaussian Shaping Parameters ─────────────────────────────────────────
    # Embedding method: CMI = relevance × (1 - redundancy) ∈ [0, 1]
    #   Good memories (relevant + novel): ~0.3-0.5
    #   Bad memories (irrelevant or redundant): ~0.0-0.1
    # mu=0.35 rewards moderately good memories; sigma=0.15 gives wide acceptance
    # scale=1.0 ensures peak reward = 1.0 (no clipping distortion)
    "cmi_gaussian_mu": 0.35,        # Optimal CMI center for embedding method
    "cmi_gaussian_sigma": 0.15,     # Width (tolerance)
    "cmi_gaussian_scale": 1.0,      # Peak reward value (must be ≤1.0 to avoid clip)

    # ── Empty Action Penalty ────────────────────────────────────────────────
    # Penalty applied when model generates no valid memory operations (empty
    # actions). Under GRPO advantage normalization, reward=0 is not a sufficient
    # deterrent — a negative reward ensures empty/trivial responses always
    # produce negative advantage, preventing response length collapse.
    "cmi_empty_action_penalty": -0.5,

    # ── Reward Composition ──────────────────────────────────────────────────
    # "cmi_only"    : pure CMI reward (no QA evaluation)
    # "cmi_plus_qa" : weighted combination: alpha * cmi + (1-alpha) * qa
    "reward_composition": "cmi_only",
    "cmi_weight": 0.5,  # alpha in the combination formula

    # ── Logprob Method Parameters ───────────────────────────────────────────
    "logprob_model": "",            # Empty = use CMI_LOGPROB_MODEL env or fallback to qwen3.6-27b
    "logprob_eval_max_tokens": 200, # Max tokens for profile generation (logprob measurement)
    "logprob_batch_size": 8,        # Batch size for logprob requests

    # ── Embedding Method Parameters ─────────────────────────────────────────
    "embedding_redundancy_threshold": 0.85,  # Cosine sim above this = redundant
    "embedding_novelty_floor": 0.3,          # Cosine sim with context below this = irrelevant
    "embedding_aggregation": "mean",         # "mean" | "max" | "min" for multi-item CMI

    # ── Entropy Method Parameters ───────────────────────────────────────────
    "entropy_context_window": 512,   # Max tokens for perplexity evaluation context
    "entropy_normalize": True,       # Normalize CMI to [0, 1] range
    "entropy_model": "",             # Empty = use inproc_answer_model

    # ── Kernel MI Method Parameters ─────────────────────────────────────────
    "kernel_mi_bandwidth": "auto",          # RBF bandwidth: "auto" (Silverman's rule) or float
    "kernel_mi_top_k": 20,                  # Top-k items to select per memory type
    "kernel_mi_type_weights": {             # Per-type CMI aggregation weights
        "core": 0.2,
        "episodic": 0.3,
        "semantic": 0.3,
        "procedural": 0.2,
    },
    "kernel_mi_normalize": True,            # Normalize MI to [0, 1] via sigmoid
    "kernel_mi_embedding_model": "",        # Dedicated model path; empty = use global
    "kernel_mi_embedding_batch_size": 64,   # Batch size for embedding calls
    "kernel_mi_min_samples": 3,             # Min samples for meaningful MI estimate

    # ── Residual Projection Method Parameters ──────────────────────────────
    "residual_projection_lambda": 1e-4,         # Tikhonov regularization strength
    "residual_projection_recent_k": 15,         # Recent existing memories to include
    "residual_projection_top_similar_k": 5,     # Top-k similar non-recent memories
    "residual_projection_aggregation": "mean",  # "mean" | "max" | "min" for multi-item

    # ── Action-Conditioned Local CMI Parameters ─────────────────────────────
    "local_cmi_top_k": 8,                    # Top-k local memories by action query
    "local_cmi_recent_k": 3,                 # Recent episodic memories to include
    "local_cmi_query_context_weight": 0.5,   # Weight for current session embedding
    "local_cmi_query_new_weight": 0.5,       # Weight for new/merged memory embedding
    "local_cmi_query_old_weight": 0.3,       # Weight for old memory embedding(s)
    "local_cmi_min_samples": 3,              # Min samples before using KDE MI
    "local_cmi_skip_positive_reward": 0.1,   # Small reward for justified semantic SKIP
    "local_cmi_skip_negative_reward": -0.1,  # Penalty for likely unjustified SKIP
    "local_cmi_skip_similarity_threshold": 0.75,
    "local_cmi_zero_delta_normalization": "zero",

    # ── Merge Summary via External Model ───────────────────────────────────
    # When enabled, MERGE actions delegate summary generation to an external
    # LLM (e.g. deepseek-v4-pro) instead of using the policy model's
    # own output.  This treats summary as a deterministic "tool call" — the
    # policy decides *what* to merge, and the external model decides *how* to
    # summarize.  The summary text is then scored by CMI reward as usual.
    "merge_summary_external_enabled": False,
    "merge_summary_model": "",           # Empty = use MERGE_SUMMARY_MODEL env or "deepseek-v4-pro"
    "merge_summary_base_url": "",        # Empty = use LLM_BASE_URL env
    "merge_summary_api_key": "",         # Empty = use LLM_API_KEY env
    "merge_summary_max_tokens": 256,     # Max tokens for the summary response
    "merge_summary_temperature": 0.3,    # Low temperature for factual summary

    # ── Intra-Session Duplicate Penalty ─────────────────────────────────────
    # Penalizes repeated ADD operations within the same session's action_steps.
    # This prevents the model from gaming CMI by adding the same memory
    # multiple times.  The penalty is multiplicative:
    #   final_reward *= (1 - duplicate_penalty_weight * duplicate_ratio)
    # where duplicate_ratio = (total_duplicates / total_adds).
    "duplicate_penalty_enabled": True,
    "duplicate_penalty_weight": 0.3,

    # ── Cross-Session Duplicate Penalty ──────────────────────────────────────
    # Penalizes ADD operations that repeat the same memory across different
    # sessions within the same conversation (batch-level check).
    # Unlike intra-session dedup (above), this catches "universal memory"
    # hacking where the model ADD's the same content in every session because
    # each session's state_before is different (so per-session redundancy
    # detection misses it).
    # Applied in compute_score_batch() after all sessions are processed.
    "cross_session_dedup_enabled": True,
    "cross_session_dedup_weight": 0.5,  # Penalty strength (0=off, 1=full)

    # ── Core Memory Specificity Check ───────────────────────────────────────
    # Detects and penalizes generic/templated core memory content that lacks
    # user-specific information (e.g. "lifelong learner", "values-driven").
    # Applies a multiplicative penalty to the core agent's CMI contribution:
    #   core_reward *= (1 - core_specificity_penalty * generic_ratio)
    "core_specificity_enabled": True,
    "core_specificity_penalty": 0.3,
    # Minimum number of generic pattern matches to trigger penalty
    "core_specificity_min_matches": 2,

    # ── General ─────────────────────────────────────────────────────────────
    "cmi_fallback_method": "embedding",  # Fallback when primary method fails
    "cmi_debug": False,                  # Enable verbose CMI computation logging
}

# =============================================================================
# Configuration Loading
# =============================================================================

_CMI_CONFIG: Optional[Dict[str, Any]] = None
_CMI_CONFIG_MTIME: Optional[float] = None


def load_cmi_config(force_reload: bool = False) -> Dict[str, Any]:
    """Load CMI configuration with hot-reload support.

    Priority (highest to lowest):
    1. Environment variables (CMI_METHOD, CMI_SHAPING, CMI_WEIGHT, etc.)
    2. JSON config file (cmi_reward_config.json)
    3. Default values (CMI_DEFAULT_CONFIG)

    Args:
        force_reload: If True, bypass mtime cache and reload from disk.

    Returns:
        Dict with all CMI configuration parameters.
    """
    global _CMI_CONFIG, _CMI_CONFIG_MTIME

    # Locate config file
    config_path = Path(__file__).parent / "cmi_reward_config.json"

    try:
        mtime = config_path.stat().st_mtime if config_path.exists() else None
    except Exception:
        mtime = None

    # Use cached config if file hasn't changed
    if (
        not force_reload
        and _CMI_CONFIG is not None
        and _CMI_CONFIG_MTIME is not None
        and mtime is not None
        and mtime <= _CMI_CONFIG_MTIME
    ):
        return _CMI_CONFIG

    # Start from defaults
    cfg = CMI_DEFAULT_CONFIG.copy()

    # Layer 2: JSON config file
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
            if isinstance(file_cfg, dict):
                cfg.update(file_cfg)
        except Exception as e:
            print(f"[CMI_CONFIG] Warning: Failed to load {config_path}: {e}")

    # Layer 1: Environment variable overrides
    _env_overrides = {
        "CMI_METHOD": ("cmi_method", str),
        "CMI_SHAPING": ("cmi_shaping", str),
        "CMI_WEIGHT": ("cmi_weight", float),
        "CMI_THRESHOLD_LOW": ("cmi_threshold_low", float),
        "CMI_THRESHOLD_HIGH": ("cmi_threshold_high", float),
        "CMI_GAUSSIAN_MU": ("cmi_gaussian_mu", float),
        "CMI_GAUSSIAN_SIGMA": ("cmi_gaussian_sigma", float),
        "CMI_GAUSSIAN_SCALE": ("cmi_gaussian_scale", float),
        "CMI_REWARD_COMPOSITION": ("reward_composition", str),
        "CMI_DEBUG": ("cmi_debug", lambda x: x.lower() in ("1", "true", "yes")),
        "CMI_KERNEL_MI_BANDWIDTH": ("kernel_mi_bandwidth", str),
        "CMI_KERNEL_MI_TOP_K": ("kernel_mi_top_k", int),
        "CMI_KERNEL_MI_NORMALIZE": ("kernel_mi_normalize", lambda x: x.lower() in ("1", "true", "yes")),
        "CMI_KERNEL_MI_EMBEDDING_MODEL": ("kernel_mi_embedding_model", str),
        "CMI_KERNEL_MI_EMBEDDING_BATCH_SIZE": ("kernel_mi_embedding_batch_size", int),
        "CMI_KERNEL_MI_MIN_SAMPLES": ("kernel_mi_min_samples", int),
        "CMI_LOGPROB_MODEL": ("logprob_model", str),
        "MERGE_SUMMARY_EXTERNAL_ENABLED": ("merge_summary_external_enabled", lambda x: x.lower() in ("1", "true", "yes")),
        "MERGE_SUMMARY_MODEL": ("merge_summary_model", str),
        "MERGE_SUMMARY_BASE_URL": ("merge_summary_base_url", str),
        "MERGE_SUMMARY_API_KEY": ("merge_summary_api_key", str),
        "MERGE_SUMMARY_MAX_TOKENS": ("merge_summary_max_tokens", int),
        "MERGE_SUMMARY_TEMPERATURE": ("merge_summary_temperature", float),
        "DUPLICATE_PENALTY_ENABLED": ("duplicate_penalty_enabled", lambda x: x.lower() in ("1", "true", "yes")),
        "DUPLICATE_PENALTY_WEIGHT": ("duplicate_penalty_weight", float),
        "CROSS_SESSION_DEDUP_ENABLED": ("cross_session_dedup_enabled", lambda x: x.lower() in ("1", "true", "yes")),
        "CROSS_SESSION_DEDUP_WEIGHT": ("cross_session_dedup_weight", float),
        "CORE_SPECIFICITY_ENABLED": ("core_specificity_enabled", lambda x: x.lower() in ("1", "true", "yes")),
        "CORE_SPECIFICITY_PENALTY": ("core_specificity_penalty", float),
        "CORE_SPECIFICITY_MIN_MATCHES": ("core_specificity_min_matches", int),
        "LOCAL_CMI_TOP_K": ("local_cmi_top_k", int),
        "LOCAL_CMI_RECENT_K": ("local_cmi_recent_k", int),
        "LOCAL_CMI_MIN_SAMPLES": ("local_cmi_min_samples", int),
        "LOCAL_CMI_SKIP_SIMILARITY_THRESHOLD": ("local_cmi_skip_similarity_threshold", float),
        "CMI_EMBEDDING_RECENT_K": ("embedding_recent_k", int),
        "CMI_EMBEDDING_TOP_SIMILAR_K": ("embedding_top_similar_k", int),
        "CMI_RESIDUAL_PROJECTION_LAMBDA": ("residual_projection_lambda", float),
    }

    for env_key, (cfg_key, cast_fn) in _env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                cfg[cfg_key] = cast_fn(val.strip())
            except (ValueError, TypeError):
                pass

    # Validate
    cfg = _validate_cmi_config(cfg)

    # Cache
    _CMI_CONFIG = cfg
    _CMI_CONFIG_MTIME = mtime

    # Print on first load
    if _CMI_CONFIG_MTIME is None or force_reload:
        print(f"[CMI_CONFIG] Loaded: method={cfg['cmi_method']}, "
              f"shaping={cfg['cmi_shaping']}, "
              f"composition={cfg['reward_composition']}, "
              f"weight={cfg['cmi_weight']}")

    return cfg


def _validate_cmi_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize CMI configuration values."""
    # Method validation
    valid_methods = ("logprob", "embedding", "kernel_mi", "entropy")
    if cfg.get("cmi_method") not in valid_methods:
        print(f"[CMI_CONFIG] Invalid cmi_method='{cfg.get('cmi_method')}', "
              f"falling back to 'logprob'")
        cfg["cmi_method"] = "logprob"

    # Shaping validation
    valid_shapings = ("threshold", "gaussian")
    if cfg.get("cmi_shaping") not in valid_shapings:
        print(f"[CMI_CONFIG] Invalid cmi_shaping='{cfg.get('cmi_shaping')}', "
              f"falling back to 'gaussian'")
        cfg["cmi_shaping"] = "gaussian"

    # Composition validation
    valid_compositions = ("cmi_only", "cmi_plus_qa")
    if cfg.get("reward_composition") not in valid_compositions:
        print(f"[CMI_CONFIG] Invalid reward_composition='{cfg.get('reward_composition')}', "
              f"falling back to 'cmi_only'")
        cfg["reward_composition"] = "cmi_only"

    # Numeric range validation
    cfg["cmi_weight"] = max(0.0, min(1.0, float(cfg.get("cmi_weight", 0.5))))
    cfg["cmi_gaussian_sigma"] = max(0.01, float(cfg.get("cmi_gaussian_sigma", 0.2)))
    cfg["cmi_threshold_low"] = max(0.0, float(cfg.get("cmi_threshold_low", 0.1)))
    cfg["cmi_threshold_high"] = max(
        cfg["cmi_threshold_low"] + 0.01,
        float(cfg.get("cmi_threshold_high", 0.8))
    )
    cfg["local_cmi_top_k"] = max(1, int(cfg.get("local_cmi_top_k", 8)))
    cfg["local_cmi_recent_k"] = max(0, int(cfg.get("local_cmi_recent_k", 3)))
    cfg["local_cmi_min_samples"] = max(2, int(cfg.get("local_cmi_min_samples", 3)))
    cfg["local_cmi_query_context_weight"] = max(
        0.0, float(cfg.get("local_cmi_query_context_weight", 0.5))
    )
    cfg["local_cmi_query_new_weight"] = max(
        0.0, float(cfg.get("local_cmi_query_new_weight", 0.5))
    )
    cfg["local_cmi_query_old_weight"] = max(
        0.0, float(cfg.get("local_cmi_query_old_weight", 0.3))
    )
    cfg["local_cmi_skip_similarity_threshold"] = max(
        0.0,
        min(1.0, float(cfg.get("local_cmi_skip_similarity_threshold", 0.75))),
    )
    cfg["local_cmi_skip_positive_reward"] = float(
        cfg.get("local_cmi_skip_positive_reward", 0.1)
    )
    cfg["local_cmi_skip_negative_reward"] = float(
        cfg.get("local_cmi_skip_negative_reward", -0.1)
    )
    if cfg.get("local_cmi_zero_delta_normalization") != "zero":
        cfg["local_cmi_zero_delta_normalization"] = "zero"

    return cfg
