"""
CMI (Conditional Mutual Information) Reward Computation.

Implements three CMI estimation methods and two reward shaping strategies
for training Memory Managers without relying on QA evaluation or expert
trajectories.

Design:
    CMI measures the information gain from memory_before to memory_after
    conditioned on the current session context. This is a session-level
    scalar reward shared by all 4 agent samples (same backbone LLM).

Methods:
    1. LogprobCMIEstimator  - LLM logprob difference
    2. EmbeddingCMIEstimator - Embedding cosine similarity
    3. EntropyCMIEstimator  - Perplexity-based entropy reduction

Shaping:
    1. ThresholdRewardShaper - Fixed interval reward
    2. GaussianRewardShaper  - Bell curve reward

Usage:
    from training.reward_server.cmi_reward import compute_cmi_reward
    result = compute_cmi_reward(parsed_responses, memory_state_before,
                                session_messages, config)
    # result = {"task_reward": 0.65, "dominant_agent": None, "cmi_raw": 0.42}
"""

import math
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

try:
    from .cmi_config import load_cmi_config
except ImportError:
    from training.reward_server.cmi_config import load_cmi_config


# =============================================================================
# CMI Estimator Base Class
# =============================================================================

class BaseCMIEstimator(ABC):
    """Abstract base class for CMI estimation methods."""

    @abstractmethod
    def estimate(
        self,
        new_memories: List[str],
        existing_memories: List[str],
        context: str,
        config: Dict[str, Any],
    ) -> float:
        """Estimate CMI for a set of new memory items.

        Args:
            new_memories: Newly generated memory items (from all 4 agents combined).
            existing_memories: Pre-existing memory bank items (from state_before).
            context: Current session conversation text.
            config: Full CMI configuration dict.

        Returns:
            float: Raw CMI estimate (unbounded; will be shaped later).
        """
        pass


# =============================================================================
# Method 1: Logprob Difference
# =============================================================================

class LogprobCMIEstimator(BaseCMIEstimator):
    """Estimate CMI via user-profile generation confidence difference.

    Uses an OpenAI-compatible API with ``logprobs=True`` to obtain
    real token-level log-probabilities.  Requires a Qwen model that supports
    the ``logprobs`` parameter (e.g. ``qwen3.6-27b``).  DeepSeek models on
    Some providers do **not** support ``logprobs`` — configure the model via the
    ``CMI_LOGPROB_MODEL`` env var or ``logprob_model`` config key.

    Algorithm:
        1. Ask the model to generate a user profile conditioned on
           (memory_before, context) — measure avg logprob of output tokens.
        2. Ask the model to generate a user profile conditioned on
           (memory_after, context) — measure avg logprob of output tokens.

    CMI ≈ mean_logprob(profile_generation | memory_after, context)
          - mean_logprob(profile_generation | memory_before, context)

    Intuition: When new memories are valuable, the model can generate a user
    profile more confidently (higher logprob) because it has more factual
    grounding. When memories are irrelevant/redundant, confidence doesn't
    change or drops.

    Normalized to [0, 1] via sigmoid:
        High (close to 1) = new memories significantly help (valuable)
        Low  (close to 0) = new memories don't help or hurt
        0.5 = neutral (no information gain)
    """

    def estimate(
        self,
        new_memories: List[str],
        existing_memories: List[str],
        context: str,
        config: Dict[str, Any],
    ) -> float:
        if not new_memories:
            return 0.0

        openai_client, model = _get_logprob_openai_client(config)

        existing_text = "\n".join(existing_memories[-20:]) if existing_memories else "[Empty]"
        all_memories_text = "\n".join(existing_memories[-20:] + new_memories)
        context_truncated = context[:2000]

        eval_max_tokens = int(config.get("logprob_eval_max_tokens", 200))
        debug = bool(config.get("cmi_debug", False))

        # ── Step 1: Measure logprob of profile generation with memory_before ───
        logprob_before = self._measure_profile_logprob(
            openai_client, model, existing_text, context_truncated,
            eval_max_tokens, debug, label="before",
        )

        # ── Step 2: Measure logprob of profile generation with memory_after ────
        logprob_after = self._measure_profile_logprob(
            openai_client, model, all_memories_text, context_truncated,
            eval_max_tokens, debug, label="after",
        )

        # CMI = logprob_after - logprob_before
        # Positive → new memories help generate user profile more confidently → valuable
        # Negative → new memories don't help or hurt → low value
        raw_cmi = logprob_after - logprob_before

        # Normalize to [0, 1] via sigmoid
        # Scale factor 3.0 ensures reasonable spread: raw_cmi in [-1, 1] maps to [0.05, 0.95]
        normalized = 1.0 / (1.0 + math.exp(-raw_cmi * 3.0))

        if debug:
            print(f"[CMI/logprob] RESULT: before={logprob_before:.4f}, "
                  f"after={logprob_after:.4f}, raw_cmi={raw_cmi:.4f}, "
                  f"normalized={normalized:.4f}, model={model}")

        return normalized

    @staticmethod
    def _measure_profile_logprob(
        openai_client: "OpenAI",
        model: str,
        memory_text: str,
        context: str,
        max_tokens: int,
        debug: bool,
        label: str = "",
    ) -> float:
        """Measure how confidently the model generates a user profile.

        We provide the memory state and context, then ask the model to
        independently generate a user profile. The avg logprob of the generated
        tokens measures how much information the memory state provides for
        understanding this user.

        Key insight: with richer/more-relevant memories, the model can generate
        a profile more confidently (higher logprob), because it has more factual
        grounding. With sparse/irrelevant memories, the model is less certain
        about what to say (lower logprob).
        """
        from llm_client import _retry_with_backoff

        system_prompt = (
            f"You are a user profiling assistant. Based on the memory state and "
            f"conversation below, write a concise but comprehensive profile of "
            f"this user. Include their preferences, personality traits, key facts, "
            f"and any important context about their life.\n\n"
            f"Memory state:\n{memory_text}\n\n"
            f"Recent conversation:\n{context}"
        )

        response = _retry_with_backoff(
            lambda: openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Write a brief user profile based on the above information."},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                logprobs=True,
                extra_body={"enable_thinking": False},
            ),
            op_name=f"logprob_cmi_{label}",
        )

        choice = response.choices[0]
        if choice.logprobs is None or choice.logprobs.content is None:
            raise RuntimeError(
                f"[CMI/logprob] Model {model} did not return logprobs. "
                f"Ensure the model supports the logprobs parameter "
                f"(Ensure the provider supports logprobs for the chosen model)."
            )

        token_logprobs = [tok.logprob for tok in choice.logprobs.content]
        if not token_logprobs:
            raise RuntimeError(
                f"[CMI/logprob] Model {model} returned empty logprobs content."
            )

        avg_logprob = sum(token_logprobs) / len(token_logprobs)

        if debug:
            print(f"[CMI/logprob]   [{label}] tokens={len(token_logprobs)}, "
                  f"avg_logprob={avg_logprob:.4f}")

        return avg_logprob


# =============================================================================
# Method 2: Embedding Distance
# =============================================================================

class EmbeddingCMIEstimator(BaseCMIEstimator):
    """Estimate CMI via embedding cosine similarity.

    CMI ≈ relevance × (1 - redundancy)

    Where:
        relevance  = cos_sim(new_items, context)
        redundancy = max cos_sim(new_item, any existing_item)

    Result in [0, 1]:
        High (close to 1) = relevant AND novel (good memory)
        Low  (close to 0) = irrelevant OR redundant (bad memory)
    """

    def estimate(
        self,
        new_memories: List[str],
        existing_memories: List[str],
        context: str,
        config: Dict[str, Any],
    ) -> float:
        if not new_memories:
            return 0.0

        import numpy as np

        # Single-pass embedding: embed ALL texts at once.
        # Layout: [new_memories | all_existing | context]
        # _get_embeddings returns L2-normalized np.ndarray, so
        # dot product == cosine similarity (no extra normalization needed).
        try:
            all_texts = new_memories + existing_memories + [context]
            all_embs = np.asarray(
                self._get_embeddings(all_texts, config),
                dtype=np.float32,
            )
            if all_embs.ndim != 2 or all_embs.shape[0] != len(all_texts):
                raise ValueError(
                    f"Expected embeddings shape ({len(all_texts)}, d), got {all_embs.shape}"
                )
        except Exception as e:
            if config.get("cmi_debug"):
                print(f"[CMI/embedding] Embedding failed: {e}. Returning 0.5.")
            return 0.5  # Neutral fallback

        n_new = len(new_memories)
        n_all_existing = len(existing_memories)

        new_embs = all_embs[:n_new]                         # (n_new, d)
        all_existing_embs = all_embs[n_new:n_new + n_all_existing]  # (n_existing, d)
        context_emb = all_embs[-1:]                         # (1, d)

        # Smart selection of existing memories for redundancy checking.
        # Combines most-recent-K (temporal) + top-K-similar (semantic)
        # to prevent "periodic redundancy hack".
        configured_recent_k = int(config.get("embedding_recent_k", 15))
        configured_top_similar_k = int(config.get("embedding_top_similar_k", 5))

        n_recent = min(configured_recent_k, n_all_existing)
        recent_indices = list(range(n_all_existing - n_recent, n_all_existing))

        top_similar_indices: list = []
        if n_all_existing > n_recent:
            n_non_recent = n_all_existing - n_recent
            n_top_similar = min(configured_top_similar_k, n_non_recent)

            if n_top_similar > 0:
                non_recent_embs = all_existing_embs[:n_non_recent]  # (n_non_recent, d)
                # Matrix multiply: (n_new, d) @ (d, n_non_recent) → (n_new, n_non_recent)
                sim_matrix = new_embs @ non_recent_embs.T
                # For each non-recent memory, take max similarity across all new items
                max_sims = sim_matrix.max(axis=0)  # (n_non_recent,)
                top_similar_indices = np.argsort(max_sims)[-n_top_similar:].tolist()

        # Deduplicate and collect selected embeddings
        selected_set = dict.fromkeys(top_similar_indices + recent_indices)
        selected_indices = list(selected_set)

        if selected_indices:
            selected_embs = all_existing_embs[selected_indices]  # (n_selected, d)
        else:
            selected_embs = np.empty((0, all_embs.shape[1]), dtype=np.float32)

        aggregation = config.get("embedding_aggregation", "mean")

        # Vectorized CMI computation:
        # relevance  = cos_sim(new_item, context)        ∈ [0, 1]
        # redundancy = max cos_sim(new_item, existing)   ∈ [0, 1]
        # cmi = relevance * (1 - redundancy)
        #   High = relevant AND novel (good memory)
        #   Low  = irrelevant OR redundant (bad memory)
        relevance = (new_embs @ context_emb.T).ravel()  # (n_new,)
        relevance = np.clip(relevance, 0.0, 1.0)

        if selected_embs.shape[0] > 0:
            redundancy_matrix = new_embs @ selected_embs.T  # (n_new, n_selected)
            redundancy = redundancy_matrix.max(axis=1)      # (n_new,)
            redundancy = np.clip(redundancy, 0.0, 1.0)
        else:
            redundancy = np.zeros(n_new, dtype=np.float32)

        # Exact/containment duplicates are treated as fully redundant.
        exact_duplicate = np.asarray(
            [
                _has_exact_or_containment_match(new_memory, existing_memories)
                for new_memory in new_memories
            ],
            dtype=np.float32,
        )
        redundancy = np.maximum(redundancy, exact_duplicate)

        cmi_values = relevance * (1.0 - redundancy)  # (n_new,)

        if len(cmi_values) == 0:
            return 0.0
        if aggregation == "max":
            return float(cmi_values.max())
        if aggregation == "min":
            return float(cmi_values.min())
        return float(cmi_values.mean())

    def _get_embeddings(self, texts: List[str], config: Dict[str, Any]) -> "np.ndarray":
        """Instance wrapper kept for tests and backward-compatible patching."""
        return _get_embeddings(texts, config)

# =============================================================================
# Method 3: Kernel MI (RBF Kernel Density Estimation)
# =============================================================================

class KernelMICMIEstimator(BaseCMIEstimator):
    """Estimate CMI via RBF kernel density estimation in embedding space.

    Uses Gaussian (RBF) kernels to estimate mutual information between
    session context and memory state, computing the difference between
    MI(context; memory_after) and MI(context; memory_before).

    Memory items are grouped by type (core/episodic/semantic/procedural)
    and top-k relevant items are selected per type. Per-type MI values
    are aggregated with configurable weights.

    Algorithm:
        1. Encode all texts with a large embedding model (e.g. Qwen3-Embedding-8B)
        2. For each memory type, select top-k items by cosine similarity to context
        3. Estimate MI(X; Y) using KDE with RBF kernel:
           MI ≈ (1/N) Σ_i log( p̂(x_i, y_i) / (p̂(x_i) · p̂(y_i)) )
           where p̂ is the kernel density estimate
        4. CMI = Σ_t w_t · [MI(context; memory_after_t) - MI(context; memory_before_t)]

    Bandwidth selection:
        "auto" uses Silverman's rule: h = σ · (4/(d+2))^(1/(d+4)) · N^(-1/(d+4))
        where σ is the median pairwise distance, d is embedding dimension, N is sample count.
    """

    def estimate(
        self,
        new_memories: List[str],
        existing_memories: List[str],
        context: str,
        config: Dict[str, Any],
    ) -> float:
        """Compute kernel MI-based CMI estimate.

        This method overrides BaseCMIEstimator.estimate but internally
        uses a richer structure: it reconstructs memory_before and memory_after
        from existing_memories + new_memories, groups by type, and computes
        per-type MI differences.

        Args:
            new_memories: New memory items produced by the 4 agents this session.
            existing_memories: All memories from state_before (with type prefixes).
            context: Formatted session messages.
            config: CMI config dict (merged with cmi_config defaults).

        Returns:
            float: CMI estimate in [0, 1] (if normalized) or raw value.
        """
        if not new_memories and not existing_memories:
            return 0.0

        top_k = int(config.get("kernel_mi_top_k", 20))
        type_weights = config.get("kernel_mi_type_weights", {
            "core": 0.2, "episodic": 0.3, "semantic": 0.3, "procedural": 0.2,
        })
        min_samples = int(config.get("kernel_mi_min_samples", 3))
        normalize = bool(config.get("kernel_mi_normalize", True))
        debug = bool(config.get("cmi_debug", False))

        # --- Step 1: Group memories by type ---
        memory_types = ["core", "episodic", "semantic", "procedural"]
        prefixes = {"core": "[CORE]", "episodic": "[EPISODIC]",
                     "semantic": "[SEMANTIC]", "procedural": "[PROCEDURAL]"}

        before_by_type: Dict[str, List[str]] = {t: [] for t in memory_types}
        for mem in existing_memories:
            for t, prefix in prefixes.items():
                if mem.startswith(prefix):
                    before_by_type[t].append(mem)
                    break

        after_by_type: Dict[str, List[str]] = {t: list(items) for t, items in before_by_type.items()}
        for mem in new_memories:
            for t, prefix in prefixes.items():
                if mem.startswith(prefix):
                    after_by_type[t].append(mem)
                    break

        # --- Step 2: Collect all texts for batch embedding ---
        # We embed: context + all before items + all after items (deduplicated by position)
        all_texts = [context]
        text_index = {"context": 0}

        before_indices: Dict[str, List[int]] = {t: [] for t in memory_types}
        after_indices: Dict[str, List[int]] = {t: [] for t in memory_types}

        for t in memory_types:
            for mem in before_by_type[t]:
                idx = len(all_texts)
                all_texts.append(mem)
                before_indices[t].append(idx)
            for mem in after_by_type[t]:
                # Only add genuinely new items (avoid re-embedding existing ones)
                if mem not in before_by_type[t]:
                    idx = len(all_texts)
                    all_texts.append(mem)
                    after_indices[t].append(idx)
                # Items already in before_by_type share indices via before_indices

        # Compute after_indices including shared items from before
        after_full_indices: Dict[str, List[int]] = {t: [] for t in memory_types}
        for t in memory_types:
            after_full_indices[t] = list(before_indices[t]) + list(after_indices[t])

        if len(all_texts) < 2:
            return 0.0

        # --- Step 3: Get embeddings ---
        batch_size = int(config.get("kernel_mi_embedding_batch_size", 64))
        embeddings = _get_embeddings_for_kernel_mi(all_texts, config, batch_size)

        import numpy as np
        emb_matrix = np.asarray(embeddings, dtype=np.float64)
        context_emb = emb_matrix[0:1]  # shape (1, d)

        # --- Step 4: Top-k selection per type ---
        def select_top_k(indices: List[int], k: int) -> np.ndarray:
            """Select top-k embeddings by cosine similarity to context."""
            if not indices:
                return np.empty((0, emb_matrix.shape[1]))
            type_embs = emb_matrix[indices]  # (n, d)
            # Cosine similarity with context
            ctx = context_emb[0]
            norms = np.linalg.norm(type_embs, axis=1) * np.linalg.norm(ctx)
            norms = np.maximum(norms, 1e-10)
            sims = type_embs @ ctx / norms
            # Top-k
            if len(sims) <= k:
                return type_embs
            top_idx = np.argsort(sims)[-k:]
            return type_embs[top_idx]

        # --- Step 5: Compute per-type MI difference ---
        bandwidth_cfg = config.get("kernel_mi_bandwidth", "auto")
        weighted_cmi = 0.0
        total_weight = 0.0

        for t in memory_types:
            w = float(type_weights.get(t, 0.25))
            before_embs = select_top_k(before_indices[t], top_k)
            after_embs = select_top_k(after_full_indices[t], top_k)

            # Need minimum samples for meaningful MI estimate
            if len(before_embs) < min_samples and len(after_embs) < min_samples:
                if debug:
                    print(f"[CMI/kernel_mi] {t}: skipped (before={len(before_embs)}, "
                          f"after={len(after_embs)}, min={min_samples})")
                continue

            ctx_repeated_before = np.repeat(context_emb, max(len(before_embs), 1), axis=0)
            ctx_repeated_after = np.repeat(context_emb, max(len(after_embs), 1), axis=0)

            mi_before = self._estimate_mi_kde(
                ctx_repeated_before[:len(before_embs)], before_embs, bandwidth_cfg
            ) if len(before_embs) >= min_samples else 0.0

            mi_after = self._estimate_mi_kde(
                ctx_repeated_after[:len(after_embs)], after_embs, bandwidth_cfg
            ) if len(after_embs) >= min_samples else 0.0

            type_cmi = mi_after - mi_before
            weighted_cmi += w * type_cmi
            total_weight += w

            if debug:
                print(f"[CMI/kernel_mi] {t}: MI_before={mi_before:.4f}, "
                      f"MI_after={mi_after:.4f}, delta={type_cmi:.4f}, w={w}")

        if total_weight > 0:
            weighted_cmi /= total_weight

        # --- Step 6: Normalize ---
        if normalize:
            # Sigmoid normalization to [0, 1]
            weighted_cmi = 1.0 / (1.0 + math.exp(-weighted_cmi * 5.0))

        if debug:
            print(f"[CMI/kernel_mi] final_cmi={weighted_cmi:.4f}, "
                  f"normalize={normalize}, n_texts={len(all_texts)}")

        return weighted_cmi

    @staticmethod
    def _estimate_mi_kde(
        X: "np.ndarray",
        Y: "np.ndarray",
        bandwidth: str = "auto",
    ) -> float:
        """Estimate MI(X; Y) using KDE with RBF kernel.

        Uses the KSG-like estimator:
            MI ≈ (1/N) Σ_i log( p̂(x_i, y_i) / (p̂(x_i) · p̂(y_i)) )

        where p̂ is estimated via Gaussian kernel density estimation.

        Args:
            X: (N, d_x) array of context embeddings.
            Y: (N, d_y) array of memory embeddings.
            bandwidth: "auto" for Silverman's rule, or a float value.

        Returns:
            float: Estimated mutual information (in nats). Non-negative by theory
                   but can be slightly negative due to finite-sample estimation.
        """
        import numpy as np

        N = min(len(X), len(Y))
        if N < 2:
            return 0.0

        X = X[:N]
        Y = Y[:N]

        # Compute bandwidth
        h = KernelMICMIEstimator._compute_bandwidth(X, Y, bandwidth)
        if h <= 0:
            return 0.0

        h_sq = h * h

        # Compute pairwise squared distances for X, Y, and joint (X,Y)
        # Using vectorized operations for efficiency
        # For marginals
        X_sq_dist = KernelMICMIEstimator._pairwise_sq_dist(X)  # (N, N)
        Y_sq_dist = KernelMICMIEstimator._pairwise_sq_dist(Y)  # (N, N)
        # Joint distance = X distance + Y distance (product kernel)
        joint_sq_dist = X_sq_dist + Y_sq_dist

        # KDE log-densities (leave-one-out to avoid self-density bias)
        # log p̂(x_i) = log( (1/(N-1)) Σ_{j≠i} K_h(x_i, x_j) )
        log_p_x = KernelMICMIEstimator._log_kde_loo(X_sq_dist, h_sq, N)
        log_p_y = KernelMICMIEstimator._log_kde_loo(Y_sq_dist, h_sq, N)
        log_p_xy = KernelMICMIEstimator._log_kde_loo(joint_sq_dist, h_sq, N)

        # MI ≈ (1/N) Σ_i [log p̂(x_i, y_i) - log p̂(x_i) - log p̂(y_i)]
        mi = float(np.mean(log_p_xy - log_p_x - log_p_y))

        # MI should be non-negative; clamp small negative values from estimation noise
        return max(0.0, mi)

    @staticmethod
    def _compute_bandwidth(
        X: "np.ndarray", Y: "np.ndarray", bandwidth: str = "auto"
    ) -> float:
        """Compute RBF kernel bandwidth.

        Silverman's rule adapted for high-dimensional data:
            h = median_dist · (4 / (d + 2))^(1 / (d + 4)) · N^(-1 / (d + 4))
        where median_dist is the median pairwise distance in the joint space.
        """
        import numpy as np

        if bandwidth != "auto":
            try:
                return float(bandwidth)
            except (ValueError, TypeError):
                pass  # Fall through to auto

        N = len(X)
        XY = np.concatenate([X, Y], axis=1)  # (N, d_x + d_y)
        d = XY.shape[1]

        # Median pairwise distance (subsample if N is large for speed)
        if N > 200:
            rng = np.random.default_rng(42)
            idx = rng.choice(N, 200, replace=False)
            sub = XY[idx]
        else:
            sub = XY

        sq_dist = KernelMICMIEstimator._pairwise_sq_dist(sub)
        # Extract upper triangle (exclude diagonal)
        triu_idx = np.triu_indices(len(sub), k=1)
        dists = np.sqrt(sq_dist[triu_idx])
        median_dist = float(np.median(dists)) if len(dists) > 0 else 1.0
        median_dist = max(median_dist, 1e-6)

        # Silverman's rule
        silverman = (4.0 / (d + 2)) ** (1.0 / (d + 4)) * (N ** (-1.0 / (d + 4)))
        h = median_dist * silverman

        return max(h, 1e-8)

    @staticmethod
    def _pairwise_sq_dist(X: "np.ndarray") -> "np.ndarray":
        """Compute pairwise squared Euclidean distances. Returns (N, N) matrix."""
        import numpy as np
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
        sq_norms = np.sum(X * X, axis=1)  # (N,)
        return sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)

    @staticmethod
    def _log_kde_loo(
        sq_dist: "np.ndarray", h_sq: float, N: int
    ) -> "np.ndarray":
        """Compute leave-one-out log KDE density for each sample.

        log p̂(x_i) = log( (1/(N-1)) Σ_{j≠i} exp(-||x_i - x_j||^2 / (2h^2)) )

        Uses log-sum-exp for numerical stability.

        Args:
            sq_dist: (N, N) pairwise squared distance matrix.
            h_sq: Squared bandwidth.
            N: Number of samples.

        Returns:
            (N,) array of log-density values.
        """
        import numpy as np

        # Exponent: -||x_i - x_j||^2 / (2 h^2)
        log_kernels = -sq_dist / (2.0 * h_sq)

        # Mask out self (diagonal) by setting to -inf
        np.fill_diagonal(log_kernels, -np.inf)

        # log-sum-exp over j for each i, then subtract log(N-1)
        max_log = np.max(log_kernels, axis=1, keepdims=True)
        # Avoid -inf max (happens when N=1, but we guard N>=2 above)
        max_log = np.where(np.isfinite(max_log), max_log, 0.0)

        log_sum = max_log.squeeze() + np.log(
            np.sum(np.exp(log_kernels - max_log), axis=1)
        )
        log_density = log_sum - math.log(max(N - 1, 1))

        return log_density


# ---------------------------------------------------------------------------
# Per-text embedding cache to avoid re-encoding identical texts across
# actions within the same turn.  Bounded by max size to prevent unbounded
# memory growth across trajectories.
# ---------------------------------------------------------------------------
_EMBEDDING_CACHE: Dict[str, "np.ndarray"] = {}
_EMBEDDING_CACHE_MAX_SIZE: int = 4096  # ~4K unique texts; each vector is ~10KB
_DEDICATED_ACTOR_FAILED: bool = False  # Skip Ray actor path after first failure
_EMBEDDER_LOCK: threading.Lock = threading.Lock()  # Protects _KERNEL_MI_EMBEDDERS loading
_CACHE_LOCK: threading.Lock = threading.Lock()  # Protects _EMBEDDING_CACHE reads/writes


def _truncate_text_for_embedding(text: str, max_words: int) -> str:
    """Tail-truncate a single text to max_words (whitespace-split) to prevent embedding OOM.

    Keeps the **last** max_words words (most recent content) because for
    session context strings the latest dialogue turns are most relevant to
    memory operations.

    For English text, whitespace-split words approximate BPE tokens reasonably
    well (~1.3 words per token), so 8192 words ≈ 6300 tokens — safely within
    Qwen3-Embedding's 32K token context window.

    Long texts (e.g. full session context > 4K tokens) cause excessive GPU
    memory usage during SentenceTransformer forward pass (attention is O(n²)).
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[-max_words:])


def _get_embeddings_for_kernel_mi(
    texts: List[str],
    config: Dict[str, Any],
    batch_size: int = 64,
) -> "np.ndarray":
    """Get embeddings for kernel MI computation, with per-text caching.

    Supports an optional dedicated embedding model for CMI kernel MI
    (configured via kernel_mi_embedding_model or CMI_KERNEL_MI_EMBEDDING_MODEL
    env var). Falls back to the global embedding infrastructure if not set.

    Resolution order for dedicated model:
      1. Ray EmbeddingServiceActor (shared across workers, avoids N copies)
      2. In-process SentenceTransformer via _KERNEL_MI_EMBEDDERS cache
      3. Global _get_embeddings (LLMClient, local or remote)

    Caching: Individual text→vector results are cached process-wide so that
    repeated embeddings of the same context/memory strings across actions
    within a turn (and across turns) are served from cache.

    Each input text is tail-truncated to ``embedding_max_words`` (default
    8192 whitespace-split words, ~6300 tokens) to prevent OOM on the embedding
    service GPU when full session context strings exceed the model's context
    window. Tail-truncation preserves the most recent dialogue content which
    is most relevant to memory ops.

    Args:
        texts: Texts to embed.
        config: CMI config dict.
        batch_size: Batch size for embedding calls.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim).
    """
    import numpy as np

    # Tail-truncate long texts to prevent embedding service OOM.
    # Qwen3-Embedding max input is 32K tokens. For English text,
    # ~1.3 whitespace words ≈ 1 BPE token, so 8192 words ≈ 6300 tokens —
    # safely within the model's context window.
    max_words = int(config.get("embedding_max_words", 8192))
    texts = [_truncate_text_for_embedding(t, max_words) for t in texts]

    # --- Check cache for hits (under lock for thread safety) ---
    cached_indices: List[int] = []
    uncached_indices: List[int] = []
    uncached_texts: List[str] = []
    with _CACHE_LOCK:
        for idx, text in enumerate(texts):
            if text in _EMBEDDING_CACHE:
                cached_indices.append(idx)
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

    # --- Encode only uncached texts ---
    new_vectors = None
    if uncached_texts:
        global _DEDICATED_ACTOR_FAILED
        dedicated_model = (
            config.get("kernel_mi_embedding_model", "")
            or os.environ.get("CMI_KERNEL_MI_EMBEDDING_MODEL", "")
        ).strip()

        if dedicated_model and not _DEDICATED_ACTOR_FAILED:
            try:
                new_vectors = _get_embeddings_via_ray_actor(
                    uncached_texts, dedicated_model, batch_size
                )
            except Exception as dedicated_err:
                _DEDICATED_ACTOR_FAILED = True
                print(
                    f"[CMI/kernel_mi] Ray embedding actor failed "
                    f"({type(dedicated_err).__name__}: {str(dedicated_err)[:100]}), "
                    f"falling back to in-process embedder.",
                    flush=True,
                )
                new_vectors = _get_embeddings_inprocess_dedicated(
                    uncached_texts, dedicated_model, batch_size
                )
        elif dedicated_model:
            new_vectors = _get_embeddings_inprocess_dedicated(
                uncached_texts, dedicated_model, batch_size
            )
        else:
            new_vectors = _get_embeddings(uncached_texts, config)

        new_vectors = np.asarray(new_vectors, dtype=np.float32)

        # Store in cache (under lock)
        with _CACHE_LOCK:
            if len(_EMBEDDING_CACHE) + len(uncached_texts) > _EMBEDDING_CACHE_MAX_SIZE:
                _EMBEDDING_CACHE.clear()
            for i, text in enumerate(uncached_texts):
                _EMBEDDING_CACHE[text] = new_vectors[i]

    # --- Assemble result in original order ---
    if not cached_indices:
        return new_vectors

    # Assemble from cache. If a race condition caused cache eviction between
    # the initial lookup and now, collect the missing texts and re-embed them
    # rather than raising a KeyError.
    with _CACHE_LOCK:
        # Determine embedding dimension from whatever is available
        dim = None
        if new_vectors is not None:
            dim = new_vectors.shape[1]
        else:
            for idx in cached_indices:
                if texts[idx] in _EMBEDDING_CACHE:
                    dim = _EMBEDDING_CACHE[texts[idx]].shape[0]
                    break
        if dim is None:
            # Extremely unlikely: all cached entries were evicted and no new
            # vectors exist. Re-embed everything outside the lock.
            pass
        else:
            result = np.empty((len(texts), dim), dtype=np.float32)
            cache_miss_indices: List[int] = []

            for idx in cached_indices:
                if texts[idx] in _EMBEDDING_CACHE:
                    result[idx] = _EMBEDDING_CACHE[texts[idx]]
                else:
                    cache_miss_indices.append(idx)

            for i, idx in enumerate(uncached_indices):
                if uncached_texts[i] in _EMBEDDING_CACHE:
                    result[idx] = _EMBEDDING_CACHE[uncached_texts[i]]
                elif new_vectors is not None and i < len(new_vectors):
                    result[idx] = new_vectors[i]
                else:
                    cache_miss_indices.append(idx)

            if not cache_miss_indices:
                return result

    # Re-embed any texts that were evicted from cache (race condition recovery)
    if dim is None:
        # Full re-embed fallback
        cache_miss_indices = list(range(len(texts)))
        cache_miss_texts = texts
    else:
        cache_miss_texts = [texts[idx] for idx in cache_miss_indices]

    recomputed = np.asarray(
        _get_embeddings_for_kernel_mi(cache_miss_texts, config, batch_size),
        dtype=np.float32,
    )

    if dim is None:
        return recomputed

    for i, idx in enumerate(cache_miss_indices):
        result[idx] = recomputed[i]

    return result


def _get_embeddings_via_ray_actor(
    texts: List[str],
    model_ref: str,
    batch_size: int = 64,
) -> "np.ndarray":
    """Get embeddings via shared Ray EmbeddingServiceActor.

    Raises on timeout (30s) to avoid deadlock when GPUs are fully occupied
    by vLLM workers and the actor cannot be scheduled.
    """
    import numpy as np
    import ray

    actor = _get_remote_embedding_actor(model_ref)
    vectors = ray.get(actor.encode.remote(texts, batch_size), timeout=30)
    return np.asarray(vectors, dtype=np.float32)


def _get_embeddings_inprocess_dedicated(
    texts: List[str],
    model_ref: str,
    batch_size: int = 64,
) -> "np.ndarray":
    """Get embeddings via in-process SentenceTransformer (cached in _KERNEL_MI_EMBEDDERS).

    This is the fallback when the Ray actor cannot be scheduled. The model is
    loaded once and cached process-wide. A threading lock prevents concurrent
    threads from racing to load the same model simultaneously.
    """
    import numpy as np

    if model_ref not in _KERNEL_MI_EMBEDDERS:
        with _EMBEDDER_LOCK:
            # Double-check after acquiring lock (another thread may have loaded it)
            if model_ref not in _KERNEL_MI_EMBEDDERS:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise ImportError(
                        "KernelMICMIEstimator with a dedicated embedding model "
                        "requires `sentence-transformers`. Install via: "
                        "pip install sentence-transformers"
                    ) from e

                from llm_client import _resolve_local_model_path

                device = (
                    os.environ.get("CMI_KERNEL_MI_EMBEDDING_DEVICE", "").strip()
                    or "cpu"
                )
                local_path = _resolve_local_model_path(model_ref, auto_download=True)
                print(
                    f"[CMI/kernel_mi] Loading in-process embedder: {model_ref} "
                    f"on {device} (path={local_path})",
                    flush=True,
                )
                try:
                    model = SentenceTransformer(
                        local_path, device=device, trust_remote_code=True
                    )
                except Exception as load_err:
                    print(
                        f"[CMI/kernel_mi] FAILED to load in-process embedder: "
                        f"{type(load_err).__name__}: {load_err}",
                        flush=True,
                    )
                    raise

                _KERNEL_MI_EMBEDDERS[model_ref] = model
                try:
                    dim = model.get_sentence_embedding_dimension()
                except Exception:
                    dim = "?"
                print(
                    f"[CMI/kernel_mi] In-process embedder loaded. dim={dim}",
                    flush=True,
                )

    embedder = _KERNEL_MI_EMBEDDERS[model_ref]
    vectors = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


# Cached reference to the remote embedding actor
_REMOTE_EMBEDDING_ACTOR = None

def _get_remote_embedding_actor(model_ref: str):
    """Get the shared embedding actor. Raises on failure (no fallback)."""
    global _REMOTE_EMBEDDING_ACTOR
    if _REMOTE_EMBEDDING_ACTOR is not None:
        return _REMOTE_EMBEDDING_ACTOR

    from training.agentic.embedding_service import get_embedding_actor
    actor = get_embedding_actor(model_path=model_ref)
    _REMOTE_EMBEDDING_ACTOR = actor
    print(f"[CMI/kernel_mi] Using shared embedding actor for {model_ref}", flush=True)
    return actor

# Module-level cache for dedicated kernel MI embedding models
_KERNEL_MI_EMBEDDERS: Dict[str, Any] = {}

# =============================================================================
# Method 4: Residual Projection (Partial Correlation CMI)
# =============================================================================

class ResidualProjectionCMIEstimator(BaseCMIEstimator):
    """Estimate CMI via Tikhonov-regularized orthogonal residual projection.

    Computes I(C; m_new | M) by projecting context and new-memory embeddings
    onto the null space of existing memories, then measuring their residual
    cosine similarity (partial correlation under Gaussian assumption).

    For each new memory item:
        r_C   = e_C   - M^T (M M^T + lambda I)^{-1} M e_C
        r_new = e_new - M^T (M M^T + lambda I)^{-1} M e_new
        CMI_i = cos(r_C, r_new)

    Final CMI = mean(CMI_i) over all new memory items, clamped to [0, 1].

    Theoretical justification:
        Under jointly Gaussian embeddings, the partial correlation
        rho_{C, m_new | M} equals cos(r_C, r_new), and the conditional MI
        satisfies I(C; m_new | M) = -0.5 * log(1 - rho^2). Since reward
        shaping is applied downstream, using rho directly preserves the
        correct ordering of actions by conditional information content.

    Degeneracy handling:
        When existing_memories is empty, no projection is applied and the
        method degrades to cos(e_C, e_new), i.e. raw relevance.
    """

    def estimate(
        self,
        new_memories: List[str],
        existing_memories: List[str],
        context: str,
        config: Dict[str, Any],
    ) -> float:
        if not new_memories:
            return 0.0

        import numpy as np

        # ── Step 1: Embed all texts in a single batch ──
        try:
            all_texts = new_memories + existing_memories + [context]
            all_embs = np.asarray(
                _get_embeddings(all_texts, config),
                dtype=np.float32,
            )
            if all_embs.ndim != 2 or all_embs.shape[0] != len(all_texts):
                raise ValueError(
                    f"Expected embeddings shape ({len(all_texts)}, d), "
                    f"got {all_embs.shape}"
                )
        except Exception as e:
            if config.get("cmi_debug"):
                print(f"[CMI/residual_projection] Embedding failed: {e}. "
                      f"Returning 0.5.")
            return 0.5  # Neutral fallback

        n_new = len(new_memories)
        n_existing = len(existing_memories)

        new_embs = all_embs[:n_new]                          # (n_new, d)
        existing_embs = all_embs[n_new:n_new + n_existing]   # (n_existing, d)
        context_emb = all_embs[-1]                           # (d,)

        # ── Step 2: Select existing memories (recent_k + top_similar_k) ──
        configured_recent_k = int(
            config.get("residual_projection_recent_k", 15)
        )
        configured_top_similar_k = int(
            config.get("residual_projection_top_similar_k", 5)
        )

        if n_existing == 0:
            # Degenerate case: no existing memories → no projection needed.
            # CMI degrades to raw relevance: cos(context, new_memory).
            relevance = new_embs @ context_emb  # (n_new,)
            relevance = np.clip(relevance, 0.0, 1.0)
            return float(relevance.mean())

        n_recent = min(configured_recent_k, n_existing)
        recent_indices = list(range(n_existing - n_recent, n_existing))

        top_similar_indices: list = []
        if n_existing > n_recent:
            n_non_recent = n_existing - n_recent
            n_top_similar = min(configured_top_similar_k, n_non_recent)
            if n_top_similar > 0:
                non_recent_embs = existing_embs[:n_non_recent]
                sim_matrix = new_embs @ non_recent_embs.T  # (n_new, n_non_recent)
                max_sims = sim_matrix.max(axis=0)  # (n_non_recent,)
                top_similar_indices = np.argsort(max_sims)[-n_top_similar:].tolist()

        # Deduplicate and collect selected embeddings
        selected_set = dict.fromkeys(top_similar_indices + recent_indices)
        selected_indices = list(selected_set)

        if not selected_indices:
            # Should not happen given n_existing > 0, but guard anyway
            relevance = new_embs @ context_emb
            relevance = np.clip(relevance, 0.0, 1.0)
            return float(relevance.mean())

        M = existing_embs[selected_indices]  # (N, d) where N = len(selected)

        # ── Step 3: Compute projection via Tikhonov regularization ──
        # P = M^T (M M^T + lambda I)^{-1} M
        # Residual: r = e - P @ e = e - M^T @ solve(M M^T + lambda I, M @ e)
        lam = float(config.get("residual_projection_lambda", 1e-4))
        N = M.shape[0]
        d = M.shape[1]

        # Gram matrix: G = M M^T + lambda I, shape (N, N)
        G = M @ M.T + lam * np.eye(N, dtype=np.float32)

        # Precompute G^{-1} M for all projections: shape (N, d)
        # solve(G, M) gives G^{-1} M, then projection = M^T @ G^{-1} @ M @ e
        # More efficient: for each vector e, compute M @ e (N,), solve G @ alpha = M @ e,
        # then residual = e - M^T @ alpha
        try:
            G_inv_M = np.linalg.solve(G, M)  # (N, d)
        except np.linalg.LinAlgError:
            # If solve fails (shouldn't with regularization), use pseudo-inverse
            if config.get("cmi_debug"):
                print("[CMI/residual_projection] linalg.solve failed, "
                      "using lstsq fallback.")
            G_inv_M, _, _, _ = np.linalg.lstsq(G, M, rcond=None)

        # Projection matrix P = M^T @ G^{-1} @ M = M^T @ G_inv_M, shape (d, d)
        # But computing full (d, d) matrix is expensive for large d.
        # Instead, compute residuals directly:
        #   r = e - M^T @ (G^{-1} @ (M @ e))
        #     = e - (G_inv_M)^T @ (M @ e)
        # Since G_inv_M = G^{-1} M, we have:
        #   M^T @ G^{-1} @ M @ e = (G_inv_M)^T @ M @ e
        # Wait, let me re-derive:
        #   P @ e = M^T @ G^{-1} @ M @ e
        #   G^{-1} @ M has shape (N, d), so (G^{-1} @ M) @ e has shape (N,)
        #   Then M^T @ that = (d, N) @ (N,) = (d,)
        # But G_inv_M = G^{-1} @ M with shape (N, d)
        # So P @ e = M^T @ (G_inv_M @ ... ) -- no, G_inv_M @ e is wrong shape.
        #
        # Correct approach:
        #   alpha = G^{-1} @ (M @ e)  -- shape (N,)
        #   projection = M^T @ alpha  -- shape (d,)
        #   residual = e - projection
        #
        # We can precompute L = G^{-1} (or use Cholesky) for multiple vectors.
        # For efficiency with multiple vectors, precompute:
        #   P_matrix = M^T @ G^{-1} @ M  -- but this is (d, d), too large
        #
        # Better: batch approach
        #   For vectors stacked as E (k, d):
        #   ME = M @ E^T  -- (N, k)
        #   alpha = solve(G, ME)  -- (N, k)
        #   projections = M^T @ alpha  -- (d, k)
        #   residuals = E^T - projections  -- (d, k)

        # Stack all vectors to project: context + all new memories
        vectors_to_project = np.vstack([context_emb[None, :], new_embs])  # (1+n_new, d)

        # M @ vectors^T: (N, 1+n_new)
        M_dot_V = M @ vectors_to_project.T

        # solve(G, M_dot_V): (N, 1+n_new)
        try:
            alpha = np.linalg.solve(G, M_dot_V)
        except np.linalg.LinAlgError:
            alpha, _, _, _ = np.linalg.lstsq(G, M_dot_V, rcond=None)

        # Projections: M^T @ alpha -> (d, 1+n_new)
        projections = M.T @ alpha  # (d, 1+n_new)

        # Residuals: vectors^T - projections -> (d, 1+n_new)
        residuals = vectors_to_project.T - projections  # (d, 1+n_new)

        # Extract context residual and new memory residuals
        r_context = residuals[:, 0]     # (d,)
        r_new = residuals[:, 1:]        # (d, n_new)

        # ── Step 4: Compute residual cosine similarities ──
        r_context_norm = np.linalg.norm(r_context)
        if r_context_norm < 1e-10:
            # Context is fully explained by existing memories
            if config.get("cmi_debug"):
                print("[CMI/residual_projection] Context residual near zero. "
                      "Returning 0.0.")
            return 0.0

        r_context_normalized = r_context / r_context_norm

        r_new_norms = np.linalg.norm(r_new, axis=0)  # (n_new,)
        # Avoid division by zero for individual new memories
        valid_mask = r_new_norms > 1e-10

        if not valid_mask.any():
            # All new memories are fully redundant
            return 0.0

        cmi_values = np.zeros(n_new, dtype=np.float32)
        # cos(r_context, r_new_i) = r_context_normalized . (r_new_i / ||r_new_i||)
        for i in range(n_new):
            if valid_mask[i]:
                cmi_values[i] = float(
                    np.dot(r_context_normalized, r_new[:, i] / r_new_norms[i])
                )

        # Clamp to [0, 1]: negative partial correlation means anti-correlated
        # (memory is misleading), treat as zero information gain.
        cmi_values = np.clip(cmi_values, 0.0, 1.0)

        # ── Step 5: Aggregate ──
        aggregation = config.get("residual_projection_aggregation", "mean")
        if aggregation == "max":
            result = float(cmi_values.max())
        elif aggregation == "min":
            result = float(cmi_values[valid_mask].min()) if valid_mask.any() else 0.0
        else:
            result = float(cmi_values.mean())

        if config.get("cmi_debug"):
            print(f"[CMI/residual_projection] n_new={n_new}, "
                  f"n_selected={len(selected_indices)}, lambda={lam}, "
                  f"cmi_values={cmi_values.tolist()}, result={result:.4f}")

        return result

# =============================================================================
# Method 5: Entropy Reduction
# =============================================================================

class EntropyCMIEstimator(BaseCMIEstimator):
    """Estimate CMI via perplexity-based entropy reduction.

    CMI ≈ log(PPL_before) - log(PPL_after)

    Where:
        PPL_before = perplexity of context given memory_before
        PPL_after  = perplexity of context given memory_after

    Positive CMI = memory update reduced uncertainty (good).
    Negative CMI = memory update increased uncertainty (bad/conflicting).

    Normalized to [0, 1] via sigmoid.
    """

    def estimate(
        self,
        new_memories: List[str],
        existing_memories: List[str],
        context: str,
        config: Dict[str, Any],
    ) -> float:
        if not new_memories:
            return 0.0

        try:
            client = _get_cmi_llm_client(config)
        except Exception as e:
            if config.get("cmi_debug"):
                print(f"[CMI/entropy] Failed to get LLM client: {e}. Falling back.")
            fallback = EmbeddingCMIEstimator()
            return fallback.estimate(new_memories, existing_memories, context, config)

        context_window = int(config.get("entropy_context_window", 512))
        truncated_context = context[:context_window * 4]  # ~4 chars/token

        # Build memory strings
        existing_text = "\n".join(existing_memories[-20:]) if existing_memories else "[Empty]"
        all_text = "\n".join(existing_memories[-20:] + new_memories)

        try:
            ppl_before = self._estimate_perplexity(
                client, existing_text, truncated_context, config
            )
            ppl_after = self._estimate_perplexity(
                client, all_text, truncated_context, config
            )

            # CMI = entropy reduction = log(PPL_before) - log(PPL_after)
            if ppl_before <= 0 or ppl_after <= 0:
                return 0.5

            raw_cmi = math.log(ppl_before) - math.log(ppl_after)

            # Normalize to [0, 1] via sigmoid
            if config.get("entropy_normalize", True):
                normalized = 1.0 / (1.0 + math.exp(-raw_cmi * 3.0))
                return normalized
            else:
                return raw_cmi

        except Exception as e:
            if config.get("cmi_debug"):
                print(f"[CMI/entropy] Error: {e}. Falling back to embedding.")
            fallback = EmbeddingCMIEstimator()
            return fallback.estimate(new_memories, existing_memories, context, config)

    def _estimate_perplexity(
        self, client: Any, memory_text: str, context: str, config: Dict
    ) -> float:
        """Estimate perplexity of context given memory as prefix.

        Uses a simple proxy: ask LLM to predict/continue the context,
        measure confidence via response length/coherence.
        """
        prompt = (
            f"Memory state:\n{memory_text[:2000]}\n\n"
            f"Given the above memory, predict what the user will discuss next. "
            f"Be specific and confident.\n\n"
            f"Actual conversation:\n{context[:500]}"
        )

        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
                logprobs=True,
            )

            # If logprobs available, use average as perplexity proxy
            if hasattr(response, "logprobs") and response.logprobs:
                avg_logprob = sum(response.logprobs) / max(len(response.logprobs), 1)
                return math.exp(-avg_logprob)

            # Fallback: use response length as proxy
            # Shorter, more confident responses → lower perplexity
            resp_text = str(response) if not isinstance(response, str) else response
            # Heuristic: map response length to perplexity-like value
            return max(1.0, len(resp_text) / 50.0)

        except Exception:
            return 10.0  # Default moderate perplexity


# =============================================================================
# Reward Shaping Strategies
# =============================================================================

class ThresholdRewardShaper:
    """Fixed threshold interval reward shaping.

    CMI in [low, high] → positive reward (linear interpolation to in_reward).
    CMI outside → linear decay toward out_reward.
    """

    def __init__(self, config: Dict[str, Any]):
        self.low = float(config.get("cmi_threshold_low", 0.1))
        self.high = float(config.get("cmi_threshold_high", 0.8))
        self.in_reward = float(config.get("cmi_threshold_in_reward", 1.0))
        self.out_reward = float(config.get("cmi_threshold_out_reward", -0.3))

    def shape(self, cmi: float) -> float:
        """Map raw CMI value to shaped reward."""
        if self.low <= cmi <= self.high:
            # In-range: linearly interpolate from out_reward to in_reward
            # Peak at the center of the range
            center = (self.low + self.high) / 2.0
            half_width = (self.high - self.low) / 2.0
            if half_width <= 0:
                return self.in_reward
            distance_from_center = abs(cmi - center) / half_width
            return self.in_reward - distance_from_center * (self.in_reward - self.out_reward) * 0.3
        elif cmi < self.low:
            # Below range: linear decay
            if self.low <= 0:
                return self.out_reward
            ratio = max(0.0, cmi / self.low)
            return self.out_reward + ratio * (self.in_reward * 0.5 - self.out_reward)
        else:
            # Above range: linear decay (redundancy penalty)
            excess = cmi - self.high
            max_excess = 1.0 - self.high
            if max_excess <= 0:
                return self.out_reward
            ratio = min(1.0, excess / max_excess)
            return self.in_reward * 0.5 - ratio * (self.in_reward * 0.5 - self.out_reward)


class NoShaper:
    """Identity shaper: return raw CMI as reward (no shaping). Used for ablation."""

    def shape(self, cmi: float) -> float:
        return cmi


class GaussianRewardShaper:
    """Gaussian (bell curve) reward shaping.

    reward = scale × exp(-((cmi - mu)² / (2σ²)))

    Centered at mu (optimal CMI), σ controls tolerance width.
    """

    def __init__(self, config: Dict[str, Any]):
        self.mu = float(config.get("cmi_gaussian_mu", 0.4))
        self.sigma = float(config.get("cmi_gaussian_sigma", 0.2))
        self.scale = float(config.get("cmi_gaussian_scale", 1.0))

    def shape(self, cmi: float) -> float:
        """Map raw CMI value to Gaussian-shaped reward."""
        exponent = -((cmi - self.mu) ** 2) / (2.0 * self.sigma ** 2)
        return self.scale * math.exp(exponent)


# =============================================================================
# Fast Embedding CMI (single embed call for overall + per-agent)
# =============================================================================

def _compute_cmi_embedding_fast(
    all_new_memories: List[str],
    per_agent_memories: Dict[str, List[str]],
    existing_memories: List[str],
    context: str,
    config: Dict[str, Any],
) -> tuple:
    """Compute overall and per-agent embedding CMI with a single embed call.

    Instead of calling EmbeddingCMIEstimator.estimate 5 times (1 overall +
    4 per-agent), we embed all unique texts once and reuse the embeddings.

    Returns:
        (raw_cmi: float, per_agent_cmi: Dict[str, float])
    """
    import numpy as np

    agent_types = ["core", "episodic", "semantic", "procedural"]

    # Build the unique text list with index tracking.
    # Layout: [all_new_memories | existing_memories | context]
    # We track which indices belong to each agent for per-agent slicing.
    n_new = len(all_new_memories)
    n_existing = len(existing_memories)

    all_texts = all_new_memories + existing_memories + [context]

    # Map agent_type → indices into all_new_memories.
    # all_new_memories is built by _extract_new_memories which iterates
    # agent_types in order (core, episodic, semantic, procedural) and appends
    # each agent's memories contiguously. Mirror that layout for indexing.
    agent_indices: Dict[str, List[int]] = {}
    offset = 0
    for agent_type in agent_types:
        agent_count = len(per_agent_memories.get(agent_type, []))
        agent_indices[agent_type] = list(range(offset, offset + agent_count))
        offset += agent_count

    try:
        all_embs = _get_embeddings(all_texts, config)  # (N, d) L2-normalized
    except Exception as e:
        if config.get("cmi_debug"):
            print(f"[CMI/embedding-fast] Embedding failed: {e}. Returning 0.")
        return 0.0, {a: 0.0 for a in agent_types}

    new_embs = all_embs[:n_new]                        # (n_new, d)
    exist_embs = all_embs[n_new:n_new + n_existing]    # (n_existing, d)
    context_emb = all_embs[-1:]                         # (1, d)

    # --- Select top-5 similar + recent-15 existing memories (same logic) ---
    n_recent = min(15, n_existing)
    recent_indices = list(range(n_existing - n_recent, n_existing))
    top_similar_indices: list = []

    if n_existing > n_recent:
        n_non_recent = n_existing - n_recent
        n_top_similar = min(5, n_non_recent)
        if n_top_similar > 0 and n_new > 0:
            non_recent_embs = exist_embs[:n_non_recent]
            sim_matrix = new_embs @ non_recent_embs.T  # (n_new, n_non_recent)
            max_sims = sim_matrix.max(axis=0)
            top_similar_indices = np.argsort(max_sims)[-n_top_similar:].tolist()

    selected_set = dict.fromkeys(top_similar_indices + recent_indices)
    selected_idx = list(selected_set)

    if selected_idx:
        selected_embs = exist_embs[selected_idx]  # (n_selected, d)
    else:
        selected_embs = np.empty((0, all_embs.shape[1]), dtype=np.float32)

    def _cmi_for_indices(indices: List[int]) -> float:
        """Compute mean CMI for a subset of new_embs by index."""
        if not indices:
            return 0.0
        sub_embs = new_embs[indices]  # (k, d)
        relevance = (sub_embs @ context_emb.T).ravel()
        relevance = np.clip(relevance, 0.0, 1.0)
        if selected_embs.shape[0] > 0:
            redundancy = (sub_embs @ selected_embs.T).max(axis=1)
            redundancy = np.clip(redundancy, 0.0, 1.0)
        else:
            redundancy = np.zeros(len(indices), dtype=np.float32)
        cmi_vals = relevance * (1.0 - redundancy)

        aggregation = config.get("embedding_aggregation", "mean")
        if aggregation == "max":
            return float(cmi_vals.max())
        elif aggregation == "min":
            return float(cmi_vals.min())
        return float(cmi_vals.mean())

    # Overall CMI (all new memories)
    overall_indices = list(range(n_new))
    raw_cmi = _cmi_for_indices(overall_indices)

    # Per-agent CMI
    per_agent_cmi: Dict[str, float] = {}
    for agent_type in agent_types:
        per_agent_cmi[agent_type] = _cmi_for_indices(agent_indices.get(agent_type, []))

    return raw_cmi, per_agent_cmi


# =============================================================================
# Unified Entry Point
# =============================================================================

def compute_cmi_reward(
    parsed_responses: Dict[str, Any],
    memory_state_before: Dict[str, Any],
    session_messages: List[Dict[str, str]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute CMI reward with method-based dispatch.

    When ``cmi_method`` is ``"logprob"``, uses session-level LLM logprob
    difference (two API calls per session).  For all other methods
    (``"embedding"``, ``"kernel_mi"``, ``"entropy"``), uses the existing
    action-conditioned local CMI path with per-action scoring.

    The return dict always contains the same keys for downstream compatibility.
    """
    cmi_config = load_cmi_config()
    cmi_method = cmi_config.get("cmi_method", "logprob")

    existing_memories = list(memory_state_before.get("memories", []) or [])
    core_memory = memory_state_before.get("core_memory", "")
    if core_memory:
        existing_memories = [f"[CORE] {core_memory}"] + existing_memories

    context = _format_session_context(session_messages)
    action_steps = _extract_cmi_action_steps(parsed_responses)

    # Penalty for empty responses (no valid memory operations).
    # Returning 0 is not a sufficient signal under GRPO advantage normalization —
    # the model can exploit this by generating minimal JSON with no operations.
    # A negative reward ensures empty/trivial responses always have negative advantage.
    empty_action_penalty = float(cmi_config.get("cmi_empty_action_penalty", -0.5))

    empty_result = {
        "task_reward": empty_action_penalty,
        "dominant_agent": None,
        "cmi_raw": 0.0,
        "cmi_method": cmi_method,
        "per_agent_cmi": {},
        "action_cmi": [],
    }

    if not action_steps:
        if cmi_config.get("cmi_debug"):
            print(f"[CMI] No valid CMI actions extracted. Returning penalty={empty_action_penalty}.")
        return empty_result

    # ── Logprob path: session-level CMI via LLM logprobs ────────────────
    if cmi_method == "logprob":
        result = _compute_cmi_reward_logprob(
            action_steps, existing_memories, context, cmi_config,
        )
    # ── Embedding path: session-level CMI via relevance × novelty ──────
    elif cmi_method == "embedding":
        result = _compute_cmi_reward_embedding(
            action_steps, existing_memories, context, cmi_config,
        )
    # ── Deprecated methods: raise explicitly ───────────────────────────
    else:
        raise NotImplementedError(
            f"CMI method '{cmi_method}' is no longer supported. "
            f"Use 'logprob' or 'embedding' instead."
        )

    # ── Core memory factual consistency penalty ────────────────────────
    # Penalize hallucinated core memory (e.g. wrong user name).
    consistency_mult = _check_core_memory_consistency(
        parsed_responses, session_messages, core_memory,
    )
    if consistency_mult < 1.0:
        result["task_reward"] = result["task_reward"] * consistency_mult
        if cmi_config.get("cmi_debug"):
            print(f"[CMI] Core memory inconsistency penalty applied: "
                  f"×{consistency_mult:.2f}, final_reward={result['task_reward']:.4f}")

    return result

def _compute_cmi_reward_embedding(
    action_steps: List[Dict[str, Any]],
    existing_memories: List[str],
    context: str,
    cmi_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Session-level CMI via embedding relevance × novelty.

    Extracts all new memories from the action steps, then calls
    ``EmbeddingCMIEstimator.estimate`` to compute a session-level CMI scalar
    using the direct formula: CMI ≈ relevance × (1 - redundancy).

    This bypasses the residual-projection approach (which suffered from
    high-dimensional orthogonal complement dilution) and instead uses a
    simple, interpretable geometric signal with naturally high variance.
    """
    new_memories = [step["memory"] for step in action_steps if step.get("memory")]
    if not new_memories:
        return {
            "task_reward": 0.0,
            "dominant_agent": None,
            "cmi_raw": 0.0,
            "cmi_method": "embedding",
            "per_agent_cmi": {},
            "action_cmi": [],
        }

    estimator = EmbeddingCMIEstimator()
    raw_cmi = estimator.estimate(
        new_memories=new_memories,
        existing_memories=existing_memories,
        context=context,
        config=cmi_config,
    )
    shaping = cmi_config.get("cmi_shaping", "gaussian")
    shaper = _get_shaper(shaping, cmi_config)
    shaped_reward = shaper.shape(raw_cmi) if raw_cmi != 0.0 else 0.0
    shaped_reward = max(-1.0, min(1.0, shaped_reward))

    # ── Anti-hacking: duplicate penalty + core specificity ─────────────
    if not cmi_config.get("extracted_mode", False):
        dup_penalty = _compute_duplicate_penalty(action_steps, cmi_config)
        core_penalty = _compute_core_specificity_penalty(action_steps, cmi_config)
        combined_penalty = min(1.0, dup_penalty + core_penalty)
        if combined_penalty > 0:
            shaped_reward *= (1.0 - combined_penalty)
            shaped_reward = max(-1.0, min(1.0, shaped_reward))
    else:
        dup_penalty = 0.0
        core_penalty = 0.0

    if cmi_config.get("cmi_debug"):
        print(f"[CMI] method=embedding, granularity=session-level, "
              f"raw={raw_cmi:.4f}, shaping={shaping}, reward={shaped_reward:.4f}, "
              f"new_memories={len(new_memories)}, existing={len(existing_memories)}, "
              f"dup_penalty={dup_penalty:.3f}, core_penalty={core_penalty:.3f}")

    return {
        "task_reward": shaped_reward,
        "dominant_agent": None,
        "cmi_raw": raw_cmi,
        "cmi_method": "embedding",
        "per_agent_cmi": {},
        "action_cmi": [],
    }


def _compute_cmi_reward_logprob(
    action_steps: List[Dict[str, Any]],
    existing_memories: List[str],
    context: str,
    cmi_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Session-level CMI via LLM log-probability difference.

    Extracts all new memories from the action steps, then calls
    ``LogprobCMIEstimator.estimate`` once to compute a single session-level
    CMI scalar.  This is then shaped and returned.

    ``action_cmi`` and ``per_agent_cmi`` are empty because the logprob method
    operates at session granularity, not per-action.
    """
    new_memories = [step["memory"] for step in action_steps if step.get("memory")]
    if not new_memories:
        return {
            "task_reward": 0.0,
            "dominant_agent": None,
            "cmi_raw": 0.0,
            "cmi_method": "logprob",
            "per_agent_cmi": {},
            "action_cmi": [],
        }

    estimator = LogprobCMIEstimator()
    raw_cmi = estimator.estimate(
        new_memories=new_memories,
        existing_memories=existing_memories,
        context=context,
        config=cmi_config,
    )
    shaping = cmi_config.get("cmi_shaping", "gaussian")
    shaper = _get_shaper(shaping, cmi_config)
    shaped_reward = shaper.shape(raw_cmi) if raw_cmi != 0.0 else 0.0
    shaped_reward = max(-1.0, min(1.0, shaped_reward))

    # ── Anti-hacking: duplicate penalty + core specificity ─────────────
    # Skip in extracted mode: candidates are pre-extracted per-session,
    # so intra-session duplication and generic core content are not possible.
    if not cmi_config.get("extracted_mode", False):
        dup_penalty = _compute_duplicate_penalty(action_steps, cmi_config)
        core_penalty = _compute_core_specificity_penalty(action_steps, cmi_config)
        combined_penalty = min(1.0, dup_penalty + core_penalty)
        if combined_penalty > 0:
            shaped_reward *= (1.0 - combined_penalty)
            shaped_reward = max(-1.0, min(1.0, shaped_reward))
    else:
        dup_penalty = 0.0
        core_penalty = 0.0

    if cmi_config.get("cmi_debug"):
        print(f"[CMI] method=logprob, granularity=session-level, "
              f"raw={raw_cmi:.4f}, shaping={shaping}, reward={shaped_reward:.4f}, "
              f"new_memories={len(new_memories)}, existing={len(existing_memories)}, "
              f"dup_penalty={dup_penalty:.3f}, core_penalty={core_penalty:.3f}")

    return {
        "task_reward": shaped_reward,
        "dominant_agent": None,
        "cmi_raw": raw_cmi,
        "cmi_method": "logprob",
        "per_agent_cmi": {},
        "action_cmi": [],
    }

def _compute_cmi_reward_local(
    action_steps: List[Dict[str, Any]],
    existing_memories: List[str],
    context: str,
    cmi_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Action-conditioned local CMI reward (embedding/kernel_mi/entropy).

    Scores each valid memory operation on an action-specific local memory
    neighborhood.  Operations are applied sequentially to a temporary memory
    state, so later actions see the effects of earlier actions.
    """
    cmi_method = cmi_config.get("cmi_method", "embedding")

    current_memories = list(existing_memories)
    action_cmi: List[Dict[str, Any]] = []
    per_agent_action_scores: Dict[str, List[float]] = {
        "core": [], "episodic": [], "semantic": [], "procedural": [],
    }
    shaping = cmi_config.get("cmi_shaping", "gaussian")
    shaper = _get_shaper(shaping, cmi_config)

    for action_index, action_step in enumerate(action_steps):
        local_before, local_after = _build_local_transition(
            current_memories=current_memories,
            action_step=action_step,
            context=context,
            config=cmi_config,
        )
        raw_action_cmi = _estimate_action_local_cmi(
            action_step=action_step,
            local_before=local_before,
            local_after=local_after,
            context=context,
            config=cmi_config,
        )
        if raw_action_cmi == 0.0:
            shaped_action_reward = 0.0
        else:
            shaped_action_reward = shaper.shape(raw_action_cmi)
            shaped_action_reward = max(-1.0, min(1.0, shaped_action_reward))

        current_memories = _apply_cmi_action_step(current_memories, action_step)

        agent_type = str(action_step["agent_type"])
        per_agent_action_scores.setdefault(agent_type, []).append(raw_action_cmi)
        # Determine which estimator was used for this action.
        action_str = str(action_step.get("action", "")).upper()
        if action_str == "SKIP":
            estimator_used = "skip_rule"
        else:
            estimator_used = "residual_projection"

        action_cmi.append({
            "index": action_index,
            "agent_type": agent_type,
            "memory_type": action_step.get("memory_type", agent_type),
            "action": action_step["action"],
            "cmi_raw": raw_action_cmi,
            "reward": shaped_action_reward,
            "memory": action_step.get("memory", ""),
            "old_memory": action_step.get("old_memory", ""),
            "old_memories": action_step.get("old_memories", []),
            "local_before_size": len(local_before),
            "local_after_size": len(local_after),
            "estimator": estimator_used,
        })

    action_raw_scores = [item["cmi_raw"] for item in action_cmi]
    action_rewards = [item["reward"] for item in action_cmi]
    raw_cmi = _aggregate_action_cmi(action_raw_scores)
    shaped_reward = _aggregate_action_cmi(action_rewards)

    per_agent_cmi = {
        agent_type: _aggregate_action_cmi(scores)
        for agent_type, scores in per_agent_action_scores.items()
    }

    # ── Anti-hacking: duplicate penalty + core specificity ─────────────
    # Skip in extracted mode (same reasoning as logprob path above).
    if not cmi_config.get("extracted_mode", False):
        dup_penalty = _compute_duplicate_penalty(action_steps, cmi_config)
        core_penalty = _compute_core_specificity_penalty(action_steps, cmi_config)
        combined_penalty = min(1.0, dup_penalty + core_penalty)
        if combined_penalty > 0:
            shaped_reward *= (1.0 - combined_penalty)
            shaped_reward = max(-1.0, min(1.0, shaped_reward))
    else:
        dup_penalty = 0.0
        core_penalty = 0.0

    if cmi_config.get("cmi_debug"):
        print(f"[CMI] method={cmi_method}, granularity=action-local, "
              f"raw={raw_cmi:.4f}, shaping={shaping}, reward={shaped_reward:.4f}, "
              f"actions={len(action_steps)}, existing={len(existing_memories)}, "
              f"per_agent={per_agent_cmi}, "
              f"dup_penalty={dup_penalty:.3f}, core_penalty={core_penalty:.3f}")

    return {
        "task_reward": max(-1.0, min(1.0, shaped_reward)),
        "dominant_agent": None,
        "cmi_raw": raw_cmi,
        "cmi_method": cmi_method,
        "per_agent_cmi": per_agent_cmi,
        "action_cmi": action_cmi,
    }

# =============================================================================
# Helper Functions
# =============================================================================

def _aggregate_action_cmi(action_scores: List[float]) -> float:
    """Aggregate action-level CMI values into one scalar."""
    if not action_scores:
        return 0.0
    return float(sum(action_scores) / len(action_scores))


def _check_core_memory_consistency(
    parsed_responses: Dict[str, Any],
    session_messages: List[Dict[str, str]],
    existing_core_memory: str,
) -> float:
    """Check if core memory is factually consistent with conversation.

    Detects hallucinated user names in core memory updates by comparing
    names mentioned in the conversation (and existing core memory) against
    the newly generated core memory content.

    Returns:
        float: Penalty multiplier in [0.5, 1.0].
            - 1.0 = no issues detected (no penalty)
            - 0.5 = clear factual inconsistency (e.g., wrong name)
    """
    import re

    # Extract names from conversation (user messages only)
    user_msgs = " ".join(
        m.get("content", "") for m in session_messages
        if m.get("role") == "user"
    )
    name_patterns = [
        r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+)",
        r"(?:this is)\s+([A-Z][a-z]+)\s+(?:speaking|here)",
    ]
    conversation_names = set()
    for pat in name_patterns:
        for match in re.finditer(pat, user_msgs, re.IGNORECASE):
            name = match.group(1).strip()
            if len(name) >= 2:  # Skip single-char noise
                conversation_names.add(name)

    # Also check existing core memory for established name
    if existing_core_memory:
        first_part = existing_core_memory.split(",")[0].strip().split()
        if first_part and first_part[0][0].isupper() and len(first_part[0]) >= 2:
            conversation_names.add(first_part[0])

    if not conversation_names:
        return 1.0  # Can't verify, no penalty

    # Extract new core memory content from parsed responses
    core_parsed = parsed_responses.get("core", {})
    if not core_parsed:
        return 1.0  # No core update, no penalty

    # Handle various parsed formats
    if isinstance(core_parsed, dict):
        new_core_content = (
            core_parsed.get("content", "")
            or core_parsed.get("new_content", "")
            or core_parsed.get("memory_content", "")
        )
        operation = core_parsed.get("operation", "").upper()
    else:
        new_core_content = str(core_parsed) if core_parsed else ""
        operation = ""

    if not new_core_content or operation == "SKIP":
        return 1.0  # No core update, no penalty

    # Check if the first entity in new core matches known names
    first_part = new_core_content.split(",")[0].strip().split()
    if first_part and first_part[0][0].isupper() and len(first_part[0]) >= 2:
        new_name = first_part[0]
        # Skip common non-name tokens
        skip_words = {"name", "role", "unknown", "user", "the", "core", "memory",
                      "updated", "none", "null", "skip", "add", "replace"}
        if new_name.lower() not in skip_words:
            if new_name not in conversation_names:
                # Hallucinated name detected
                return 0.5

    return 1.0


def _compute_duplicate_penalty(
    action_steps: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> float:
    """Compute intra-session duplicate penalty for ADD operations.

    Counts how many ADD memories are duplicates (by first 100 chars) within
    the same session's action_steps. Returns a penalty multiplier in [0, 1]
    that should be applied as: reward *= (1 - penalty).

    A penalty of 0 means no duplicates; 1 means all ADDs are duplicates.
    """
    if not config.get("duplicate_penalty_enabled", True):
        return 0.0

    weight = float(config.get("duplicate_penalty_weight", 0.3))
    add_memories = [
        step["memory"][:100]
        for step in action_steps
        if step.get("action", "").upper() == "ADD" and step.get("memory")
    ]
    if len(add_memories) <= 1:
        return 0.0

    from collections import Counter
    counter = Counter(add_memories)
    duplicate_count = sum(count - 1 for count in counter.values() if count > 1)
    duplicate_ratio = duplicate_count / len(add_memories)

    return min(1.0, weight * duplicate_ratio)


# Known generic patterns that indicate non-specific core memory content.
# These are phrases frequently produced by mode-collapsed models that
# apply to virtually any user and carry no distinguishing information.
_CORE_GENERIC_PATTERNS = [
    "lifelong learner",
    "evidence-based",
    "cultural humility",
    "values-driven",
    "ethical technology",
    "continuous learning",
    "growth mindset",
    "open-minded",
    "critical thinking",
    "emotional intelligence",
    "holistic approach",
    "data-driven",
    "community-oriented",
    "sustainability-focused",
    "inclusive perspective",
    "mindful practice",
    "reflective practitioner",
    "interdisciplinary",
    "solution-oriented",
    "empathetic communicator",
]


def _compute_core_specificity_penalty(
    action_steps: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> float:
    """Compute penalty for generic/templated core memory content.

    Scans core agent APPEND/REPLACE memory text for known generic patterns.
    Returns a penalty multiplier in [0, 1] to apply to the final reward.

    A penalty of 0 means core memory is specific; up to 1 for fully generic.
    """
    if not config.get("core_specificity_enabled", True):
        return 0.0

    penalty_weight = float(config.get("core_specificity_penalty", 0.3))
    min_matches = int(config.get("core_specificity_min_matches", 2))

    core_texts = [
        step["memory"].lower()
        for step in action_steps
        if step.get("agent_type") == "core"
        and step.get("action", "").upper() in ("APPEND", "REPLACE")
        and step.get("memory")
    ]
    if not core_texts:
        return 0.0

    combined = " ".join(core_texts)
    match_count = sum(1 for pattern in _CORE_GENERIC_PATTERNS if pattern in combined)

    if match_count < min_matches:
        return 0.0

    # Scale: 2 matches → base penalty, 5+ matches → full penalty
    generic_ratio = min(1.0, (match_count - min_matches + 1) / max(len(_CORE_GENERIC_PATTERNS) * 0.3, 1))
    return min(1.0, penalty_weight * generic_ratio)


# =============================================================================
# External Model Summary for MERGE Operations
# =============================================================================

_MERGE_SUMMARY_CLIENT = None

def _get_merge_summary_client(config: Dict[str, Any]):
    """Lazy-init OpenAI-compatible client for MERGE summary generation."""
    global _MERGE_SUMMARY_CLIENT
    if _MERGE_SUMMARY_CLIENT is not None:
        return _MERGE_SUMMARY_CLIENT

    from openai import OpenAI

    api_key = (
        config.get("merge_summary_api_key")
        or os.environ.get("MERGE_SUMMARY_API_KEY")
        or os.environ.get("LLM_API_KEY", "")
    )
    base_url = (
        config.get("merge_summary_base_url")
        or os.environ.get("MERGE_SUMMARY_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL", "")
    )
    _MERGE_SUMMARY_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _MERGE_SUMMARY_CLIENT


def _summarize_memories_via_external_model(
    old_memories: List[str],
    agent_type: str,
    config: Dict[str, Any],
) -> Optional[str]:
    """Summarize a list of memories into one using an external LLM.

    Treats MERGE as a deterministic tool call: the policy model decides
    *what* to merge, and the external model decides *how* to summarize.
    The resulting summary text is then scored by CMI reward as usual.

    Args:
        old_memories: List of memory strings to merge/summarize.
        agent_type: Memory type (episodic/semantic/procedural).
        config: CMI config dict with merge_summary_* parameters.

    Returns:
        Summarized memory string, or None if the call fails.
    """
    if not old_memories:
        return None

    model = (
        config.get("merge_summary_model")
        or os.environ.get("MERGE_SUMMARY_MODEL")
        or "deepseek-v4-pro"
    )
    max_tokens = int(config.get("merge_summary_max_tokens", 256))
    temperature = float(config.get("merge_summary_temperature", 0.3))

    memories_text = "\n".join(f"- {mem}" for mem in old_memories)
    prompt = (
        f"You are a memory management assistant. Merge the following {agent_type} "
        f"memories into a single, concise summary that preserves all key facts, "
        f"timestamps, and details. Do NOT add any information not present in the "
        f"original memories. Output only the merged memory text, nothing else.\n\n"
        f"Memories to merge:\n{memories_text}"
    )

    try:
        client = _get_merge_summary_client(config)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        summary = response.choices[0].message.content.strip()
        if config.get("cmi_debug"):
            print(f"[CMI/merge_summary] External model={model}, "
                  f"input_count={len(old_memories)}, "
                  f"output_len={len(summary)}")
        return summary if summary else None
    except Exception as exc:
        print(f"[CMI/merge_summary] External summary failed "
              f"({type(exc).__name__}: {str(exc)[:100]}), "
              f"falling back to policy output.", flush=True)
        return None


def _extract_cmi_action_steps(parsed_responses: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract CMI-scored operations in deterministic execution order."""
    action_steps: List[Dict[str, Any]] = []

    core_parsed = parsed_responses.get("core")
    if core_parsed:
        core_data = core_parsed.get("core", core_parsed) if isinstance(core_parsed, dict) else {}
        operation = str(core_data.get("operation", "")).upper()
        if operation == "APPEND":
            content = str(core_data.get("content", "")).strip()
            if content:
                action_steps.append({
                    "agent_type": "core",
                    "memory_type": "core",
                    "action": "APPEND",
                    "memory": _ensure_memory_prefix(content, "core"),
                    "old_memory": "",
                    "old_memories": [],
                    "reward_mode": "add",
                })
        elif operation in ("REPLACE", "REWRITE"):
            # REWRITE is functionally equivalent to REPLACE for CMI scoring:
            # both produce a new core memory text replacing the old one.
            new_text = str(
                core_data.get("new_text", "") or core_data.get("content", "")
            ).strip()
            old_text = str(core_data.get("old_text", "")).strip()
            if new_text:
                action_steps.append({
                    "agent_type": "core",
                    "memory_type": "core",
                    "action": "REPLACE",
                    "memory": _ensure_memory_prefix(new_text, "core"),
                    "old_memory": old_text,
                    "old_memories": [],
                    "reward_mode": "replace",
                })

    for agent_type in ["episodic", "semantic", "procedural"]:
        parsed = parsed_responses.get(agent_type)
        if not parsed:
            continue

        agent_data = parsed.get(agent_type, parsed) if isinstance(parsed, dict) else {}
        operations = agent_data.get("operations", []) if isinstance(agent_data, dict) else []

        for operation_data in operations:
            if not isinstance(operation_data, dict):
                continue

            action = str(operation_data.get("action", "")).upper()
            old_memory = str(operation_data.get("old_memory", "")).strip()
            old_memories = operation_data.get("old_memories", [])
            if not isinstance(old_memories, list):
                old_memories = []

            if action == "SKIP":
                # SKIP actions are not scored — they do not participate in
                # reward aggregation, matching original MemBuilder behavior.
                continue

            if action == "ADD":
                memory = operation_data.get("memory", "")
                reward_mode = "add"
            elif action == "UPDATE":
                memory = operation_data.get("new_memory", "") or operation_data.get("memory", "")
                reward_mode = "episodic_update" if agent_type == "episodic" else "replace"
            elif action == "MERGE" and agent_type == "episodic":
                cmi_cfg = load_cmi_config()
                policy_memory = operation_data.get("new_memory", "") or operation_data.get("memory", "")
                merge_old_memories = operation_data.get("old_memories", [])
                if not isinstance(merge_old_memories, list):
                    merge_old_memories = []
                merge_old_memories = [str(m).strip() for m in merge_old_memories if str(m).strip()]

                if cmi_cfg.get("merge_summary_external_enabled") and merge_old_memories:
                    external_summary = _summarize_memories_via_external_model(
                        merge_old_memories, agent_type, cmi_cfg,
                    )
                    memory = external_summary if external_summary else policy_memory
                else:
                    memory = policy_memory
                reward_mode = "merge"
            else:
                continue

            memory = str(memory).strip()
            if not memory:
                continue

            action_steps.append({
                "agent_type": agent_type,
                "memory_type": agent_type,
                "action": action,
                "memory": _ensure_memory_prefix(memory, agent_type),
                "old_memory": old_memory,
                "old_memories": [str(item).strip() for item in old_memories if str(item).strip()],
                "reward_mode": reward_mode,
            })

    return action_steps

def _apply_cmi_action_step(
    current_memories: List[str],
    action_step: Dict[str, Any],
) -> List[str]:
    """Apply one parsed memory operation to the temporary CMI memory state."""
    updated_memories = list(current_memories)
    action = str(action_step.get("action", "")).upper()
    memory_type = str(action_step.get("memory_type", action_step.get("agent_type", "")))
    new_memory = str(action_step.get("memory", ""))

    if action == "SKIP":
        return updated_memories

    if action in ("REPLACE", "UPDATE") and memory_type != "episodic":
        updated_memories = _remove_matching_memory(
            updated_memories, str(action_step.get("old_memory", ""))
        )

    # Episodic UPDATE/MERGE keep historical old memories by design.

    if new_memory:
        updated_memories.append(new_memory)

    return updated_memories

def _build_local_transition(
    current_memories: List[str],
    action_step: Dict[str, Any],
    context: str,
    config: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Build action-conditioned local before/after memory sets."""
    local_before = _select_local_memories_for_action(
        current_memories=current_memories,
        action_step=action_step,
        context=context,
        config=config,
    )
    action = str(action_step.get("action", "")).upper()
    memory_type = str(action_step.get("memory_type", action_step.get("agent_type", "")))
    new_memory = str(action_step.get("memory", ""))

    if action == "SKIP":
        return local_before, list(local_before)

    local_after = list(local_before)
    if action in ("REPLACE", "UPDATE") and memory_type != "episodic":
        local_after = _remove_matching_memory(
            local_after, str(action_step.get("old_memory", ""))
        )

    # Episodic UPDATE/MERGE add summaries/new events but retain old history.
    if new_memory:
        local_after.append(new_memory)

    return _deduplicate_memories(local_before), _deduplicate_memories(local_after)

def _select_local_memories_for_action(
    current_memories: List[str],
    action_step: Dict[str, Any],
    context: str,
    config: Dict[str, Any],
) -> List[str]:
    """Select action-conditioned same-type local memories."""
    memory_type = str(action_step.get("memory_type", action_step.get("agent_type", "")))
    type_memories = _filter_memories_by_type(current_memories, memory_type)
    if not type_memories:
        return []

    query = _compute_action_query(action_step, context, config)
    top_k = int(config.get("local_cmi_top_k", 8))
    selected = _select_top_k_similar(type_memories, query, top_k, config)

    if memory_type == "episodic":
        recent_k = int(config.get("local_cmi_recent_k", 3))
        selected.extend(_select_recent_memories(type_memories, recent_k))

    old_memory = str(action_step.get("old_memory", "")).strip()
    if old_memory:
        selected.append(_find_matching_memory(type_memories, old_memory) or old_memory)

    old_memories = action_step.get("old_memories", [])
    if isinstance(old_memories, list):
        for old in old_memories:
            old_text = str(old).strip()
            if old_text:
                selected.append(_find_matching_memory(type_memories, old_text) or old_text)

    return _deduplicate_memories(selected)

def _compute_action_query(
    action_step: Dict[str, Any],
    context: str,
    config: Dict[str, Any],
) -> str:
    """Create the text query used for local retrieval.

    Embedding weights are approximated by repeated text segments to keep the
    implementation compatible with the existing text embedding interface.
    """
    parts: List[str] = []
    context_weight = float(config.get("local_cmi_query_context_weight", 0.5))
    new_weight = float(config.get("local_cmi_query_new_weight", 0.5))
    old_weight = float(config.get("local_cmi_query_old_weight", 0.3))

    if context_weight > 0 and context:
        parts.append(context)
    if new_weight > 0 and action_step.get("memory"):
        parts.append(str(action_step.get("memory", "")))
    if old_weight > 0 and action_step.get("old_memory"):
        parts.append(str(action_step.get("old_memory", "")))
    old_memories = action_step.get("old_memories", [])
    if old_weight > 0 and isinstance(old_memories, list):
        parts.extend(str(item) for item in old_memories if str(item).strip())

    return "\n".join(parts) if parts else context

def _select_top_k_similar(
    memories: List[str],
    query: str,
    k: int,
    config: Dict[str, Any],
) -> List[str]:
    """Select top-k memories by cosine similarity to the action query."""
    if not memories or k <= 0:
        return []
    import numpy as np
    embeddings = _get_embeddings_for_kernel_mi([query] + memories, config)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    query_emb = embeddings[0]
    memory_embs = embeddings[1:]
    sims = memory_embs @ query_emb
    ranked = sims.argsort()[-min(k, len(memories)):][::-1]
    return [memories[int(index)] for index in ranked]

def _select_recent_memories(memories: List[str], k: int) -> List[str]:
    """Select the most recent memories by list order."""
    if k <= 0:
        return []
    return list(memories[-min(k, len(memories)):])

def _estimate_action_local_cmi(
    action_step: Dict[str, Any],
    local_before: List[str],
    local_after: List[str],
    context: str,
    config: Dict[str, Any],
) -> float:
    """Estimate raw local CMI for one action."""
    if str(action_step.get("action", "")).upper() == "SKIP":
        return _estimate_skip_reward(local_before, context, config)

    if _deduplicate_memories(local_before) == _deduplicate_memories(local_after):
        return 0.0

    # All non-SKIP operations use residual-projection conditional MI.
    #
    # This estimates I(C; m_new | M) by projecting context and new-memory
    # embeddings onto the orthogonal complement of the existing memory
    # subspace and measuring their residual cosine similarity.
    #
    # For REPLACE/UPDATE operations, the function computes the net gain:
    #   I_res(C; m_new | M) - I_res(C; m_old | M)
    #
    # Using a single unified estimator across all operation types ensures
    # consistent scale and semantics for the reward signal, which is
    # critical for stable GRPO training.
    return _estimate_residual_projection_cmi(
        action_step, local_before, context, config,
    )

def _estimate_memory_set_mi(
    memories: List[str],
    context: str,
    bandwidth: Any,
    config: Dict[str, Any],
) -> float:
    """Estimate MI(context; local memory set) with the existing RBF KDE helper."""
    if len(memories) < 2:
        return 0.0
    embeddings = _get_embeddings_for_kernel_mi(
        [context] + memories,
        config,
        int(config.get("kernel_mi_embedding_batch_size", 64)),
    )
    import numpy as np
    emb_matrix = np.asarray(embeddings, dtype=np.float64)
    context_emb = emb_matrix[0:1]
    memory_embs = emb_matrix[1:]
    repeated_context = np.repeat(context_emb, len(memory_embs), axis=0)
    return KernelMICMIEstimator._estimate_mi_kde(
        repeated_context,
        memory_embs,
        str(bandwidth),
    )

def _estimate_residual_projection_cmi(
    action_step: Dict[str, Any],
    local_before: List[str],
    context: str,
    config: Dict[str, Any],
) -> float:
    """Estimate conditional MI via orthogonal residual projection.

    Computes I(C; m_new | M) by projecting context and new-memory embeddings
    onto the orthogonal complement of the subspace spanned by existing local
    memories.  The cosine similarity of the residual vectors approximates
    the conditional mutual information under a Gaussian embedding assumption.

    This method avoids the numerical instability of the KDE MI difference
    approach when adding a single memory to a large local set.

    For REPLACE/UPDATE actions with an old_memory, returns the difference
    between the residual CMI of the new memory and the old memory, measuring
    the net information gain of the replacement.

    Args:
        action_step: Parsed action dict with memory, old_memory, reward_mode.
        local_before: Local memory neighborhood before the action.
        context: Dialogue context string.
        config: CMI configuration dict.

    Returns:
        float: Estimated conditional MI (non-negative for ADD, signed for REPLACE).
    """
    import numpy as np

    new_memory = str(action_step.get("memory", "")).strip()
    if not new_memory:
        return 0.0

    # Collect texts: [context, new_memory, *local_before_memories]
    texts = [context, new_memory] + local_before
    embeddings = np.asarray(
        _get_embeddings_for_kernel_mi(texts, config), dtype=np.float64
    )
    context_emb = embeddings[0]      # (d,)
    new_mem_emb = embeddings[1]      # (d,)
    mem_embs = embeddings[2:]        # (N, d)

    # --- Degenerate case: no existing memories to condition on ---
    # Falls back to simple cosine similarity (unconditional MI proxy).
    if len(mem_embs) == 0:
        sim = float(np.dot(context_emb, new_mem_emb) / (
            max(np.linalg.norm(context_emb), 1e-10)
            * max(np.linalg.norm(new_mem_emb), 1e-10)
        ))
        return max(0.0, sim)

    # --- Orthogonal projection onto the column space of M ---
    # M: (N, d) matrix of existing memory embeddings
    # Projection matrix: P = M^T (M M^T + lambda I)^{-1} M
    # Residual: r = v - M^T (M M^T + lambda I)^{-1} M v
    #
    # We use Tikhonov regularization (lambda) for numerical stability
    # when M M^T is ill-conditioned (common with correlated memories).
    M = mem_embs  # (N, d)
    N = M.shape[0]
    reg_lambda = float(config.get("residual_projection_lambda", 1e-4))

    # Compute M M^T: (N, N)
    MMt = M @ M.T
    # Regularize: (M M^T + lambda I)
    MMt_reg = MMt + reg_lambda * np.eye(N, dtype=np.float64)

    # Solve for projection coefficients:
    # alpha_c = (M M^T + lambda I)^{-1} M v_context
    # alpha_n = (M M^T + lambda I)^{-1} M v_new
    Mv_context = M @ context_emb   # (N,)
    Mv_new = M @ new_mem_emb       # (N,)

    try:
        alpha_c = np.linalg.solve(MMt_reg, Mv_context)  # (N,)
        alpha_n = np.linalg.solve(MMt_reg, Mv_new)      # (N,)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if solve fails
        MMt_inv = np.linalg.pinv(MMt_reg)
        alpha_c = MMt_inv @ Mv_context
        alpha_n = MMt_inv @ Mv_new

    # Compute residuals: r = v - M^T alpha
    r_context = context_emb - M.T @ alpha_c  # (d,)
    r_new = new_mem_emb - M.T @ alpha_n      # (d,)

    # Cosine similarity of residuals = conditional MI proxy
    r_context_norm = max(np.linalg.norm(r_context), 1e-10)
    r_new_norm = max(np.linalg.norm(r_new), 1e-10)
    cmi = float(np.dot(r_context, r_new) / (r_context_norm * r_new_norm))

    # --- For REPLACE-style actions: compute net gain over old memory ---
    old_memory = str(action_step.get("old_memory", "")).strip()
    reward_mode = str(action_step.get("reward_mode", ""))
    if old_memory and reward_mode in ("replace", "episodic_update"):
        old_emb = np.asarray(
            _get_embeddings_for_kernel_mi([old_memory], config), dtype=np.float64
        )[0]
        Mv_old = M @ old_emb
        try:
            alpha_o = np.linalg.solve(MMt_reg, Mv_old)
        except np.linalg.LinAlgError:
            alpha_o = np.linalg.pinv(MMt_reg) @ Mv_old
        r_old = old_emb - M.T @ alpha_o
        r_old_norm = max(np.linalg.norm(r_old), 1e-10)
        old_cmi = float(np.dot(r_context, r_old) / (r_context_norm * r_old_norm))
        return cmi - old_cmi

    # For ADD/MERGE: clamp to non-negative (adding info should not be penalized)
    return max(0.0, cmi)


def _estimate_local_fallback(
    action_step: Dict[str, Any],
    local_before: List[str],
    context: str,
    config: Dict[str, Any],
) -> float:
    """Small-sample relevance-novelty fallback for local CMI."""
    new_memory = str(action_step.get("memory", "")).strip()
    if not new_memory:
        return 0.0

    import numpy as np
    texts = [context, new_memory] + local_before
    embeddings = np.asarray(
        _get_embeddings_for_kernel_mi(texts, config), dtype=np.float32
    )
    context_emb = embeddings[0]
    new_emb = embeddings[1]
    old_embs = embeddings[2:]
    relevance = max(0.0, min(1.0, float(new_emb @ context_emb)))
    redundancy = 0.0
    if len(old_embs) > 0:
        redundancy = max(0.0, min(1.0, float((old_embs @ new_emb).max())))
    new_score = relevance * (1.0 - redundancy)

    old_memory = str(action_step.get("old_memory", "")).strip()
    if old_memory and str(action_step.get("reward_mode")) == "replace":
        old_embeddings = np.asarray(
            _get_embeddings_for_kernel_mi([context, old_memory], config),
            dtype=np.float32,
        )
        old_score = max(0.0, min(1.0, float(old_embeddings[0] @ old_embeddings[1])))
        return new_score - old_score

    return new_score

def _estimate_skip_reward(
    local_before: List[str],
    context: str,
    config: Dict[str, Any],
) -> float:
    """Rule-based reward for semantic SKIP."""
    if not local_before:
        return float(config.get("local_cmi_skip_negative_reward", -0.1))

    import numpy as np
    embeddings = np.asarray(
        _get_embeddings_for_kernel_mi([context] + local_before, config),
        dtype=np.float32,
    )
    context_emb = embeddings[0]
    memory_embs = embeddings[1:]
    max_sim = float((memory_embs @ context_emb).max()) if len(memory_embs) else 0.0

    threshold = float(config.get("local_cmi_skip_similarity_threshold", 0.75))
    if max_sim >= threshold:
        return float(config.get("local_cmi_skip_positive_reward", 0.1))
    return float(config.get("local_cmi_skip_negative_reward", -0.1))

def _extract_new_memories(parsed_responses: Dict[str, Any]) -> List[str]:
    """Backward-compatible extraction of newly generated memories."""
    return [
        step["memory"]
        for step in _extract_cmi_action_steps(parsed_responses)
        if step.get("memory")
    ]

def _format_session_context(session_messages: List[Dict[str, str]]) -> str:
    """Format session messages into a compact context string."""
    lines = []
    for message in session_messages or []:
        role = str(message.get("role", "unknown")).upper()
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)

def _filter_memories_by_type(memories: List[str], memory_type: str) -> List[str]:
    """Return memories with the requested bracketed type prefix."""
    prefix = f"[{memory_type.upper()}]"
    return [str(memory) for memory in memories if str(memory).startswith(prefix)]

def _ensure_memory_prefix(memory: str, memory_type: str) -> str:
    """Ensure a memory string starts with its bracketed type prefix."""
    prefix = f"[{memory_type.upper()}]"
    memory_text = str(memory).strip()
    if memory_text.startswith(prefix):
        return memory_text
    return f"{prefix} {memory_text}"

def _find_matching_memory(memories: List[str], old_memory: str) -> Optional[str]:
    """Find the first memory matching or containing old_memory."""
    if not old_memory:
        return None
    old_memory_text = str(old_memory).strip()
    for memory in memories:
        memory_text = str(memory)
        if memory_text == old_memory_text or old_memory_text.endswith(memory_text):
            return memory_text
        if old_memory_text in memory_text or memory_text in old_memory_text:
            return memory_text
    return None

def _has_exact_or_containment_match(memory: str, candidate_memories: List[str]) -> bool:
    """Return whether memory is duplicated by an existing memory.

    Exact matches always count. Containment matches only count when the shorter
    string is long enough to avoid accidental short-token substring matches.
    """
    memory_text = str(memory).strip()
    if not memory_text:
        return False

    min_containment_chars = 20
    for candidate_memory in candidate_memories:
        candidate_text = str(candidate_memory).strip()
        if not candidate_text:
            continue
        if memory_text == candidate_text:
            return True

        shorter_text, longer_text = sorted(
            [memory_text, candidate_text],
            key=len,
        )
        if len(shorter_text) >= min_containment_chars and shorter_text in longer_text:
            return True

    return False

def _remove_matching_memory(memories: List[str], old_memory: str) -> List[str]:
    """Remove the first memory matching or containing old_memory."""
    match = _find_matching_memory(memories, old_memory)
    if match is None:
        return memories
    removed = False
    result: List[str] = []
    for memory in memories:
        if not removed and str(memory) == match:
            removed = True
            continue
        result.append(memory)
    return result

def _deduplicate_memories(memories: List[str]) -> List[str]:
    """Deduplicate memories while preserving order."""
    seen = set()
    result: List[str] = []
    for memory in memories:
        memory_text = str(memory).strip()
        if not memory_text or memory_text in seen:
            continue
        seen.add(memory_text)
        result.append(memory_text)
    return result

def _get_estimator(method: str, config: Dict[str, Any]) -> BaseCMIEstimator:
    """Factory: get CMI estimator by method name."""
    if method == "logprob":
        return LogprobCMIEstimator()
    elif method == "embedding":
        return EmbeddingCMIEstimator()
    elif method == "kernel_mi":
        return KernelMICMIEstimator()
    elif method == "residual_projection":
        return ResidualProjectionCMIEstimator()
    elif method == "entropy":
        return EntropyCMIEstimator()
    else:
        print(f"[CMI] Unknown method '{method}', defaulting to embedding.")
        return EmbeddingCMIEstimator()


def _get_shaper(shaping: str, config: Dict[str, Any]):
    """Factory: get reward shaper by strategy name."""
    if shaping == "threshold":
        return ThresholdRewardShaper(config)
    elif shaping == "gaussian":
        return GaussianRewardShaper(config)
    elif shaping == "none":
        return NoShaper()
    else:
        print(f"[CMI] Unknown shaping '{shaping}', defaulting to gaussian.")
        return GaussianRewardShaper(config)


# =============================================================================
# LLM / Embedding Client Utilities
# =============================================================================

_CMI_LLM_CLIENT = None
_CMI_EMBEDDING_CLIENT = None
_CMI_EMBEDDING_CLIENT_LOCK: threading.Lock = threading.Lock()
_LOGPROB_OPENAI_CLIENT: Optional[Tuple[Any, str]] = None

def _get_logprob_openai_client(config: Dict[str, Any]) -> Tuple[Any, str]:
    """Get or create a raw OpenAI SDK client for logprob CMI.

    Returns the raw ``openai.OpenAI`` instance (not ``LLMClient``) so that
    callers can access ``response.choices[0].logprobs`` directly.

    Model resolution order:
      1. ``config["logprob_model"]`` (from cmi_reward_config.json or env CMI_LOGPROB_MODEL)
      2. ``CMI_LOGPROB_MODEL`` env var
      3. Default: ``qwen3.6-27b``

    Returns:
        (openai_client, model_name)
    """
    global _LOGPROB_OPENAI_CLIENT
    if _LOGPROB_OPENAI_CLIENT is not None:
        return _LOGPROB_OPENAI_CLIENT

    from openai import OpenAI

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL",
        os.environ.get("OPENAI_BASE_URL", ""),
    )
    model = (
        (config.get("logprob_model") or "").strip()
        or os.environ.get("CMI_LOGPROB_MODEL", "").strip()
        or "qwen3.6-27b"
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    _LOGPROB_OPENAI_CLIENT = (client, model)

    print(f"[CMI/logprob] OpenAI client initialized: model={model}, "
          f"base_url={base_url}", flush=True)

    return _LOGPROB_OPENAI_CLIENT


def _get_cmi_llm_client(config: Dict[str, Any]):
    """Get or create LLM client for CMI computation (entropy method)."""
    global _CMI_LLM_CLIENT
    if _CMI_LLM_CLIENT is not None:
        return _CMI_LLM_CLIENT

    # Use inproc config from main reward config (shared LLM credentials)
    from llm_client import LLMClient

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL",
        os.environ.get("OPENAI_BASE_URL", ""),
    )
    model = (
        config.get("entropy_model")
        or os.environ.get("MEMBUILDER_INPROC_ANSWER_MODEL", "deepseek-v4-pro")
    )

    _CMI_LLM_CLIENT = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    return _CMI_LLM_CLIENT

def _discover_embedding_service_url() -> str:
    """Auto-discover embedding service URL from shared discovery files.

    Checks the following locations in order:
      1. ${DATA_DIR}/services/embedding_service_address.txt
      2. ${DATA_DIR}/services/embedding_service_address.json

    The .txt file contains a single-line URL (e.g. http://IP:PORT/v1).
    The .json file contains a JSON object with a "base_url" or "url" field.

    Returns:
        The discovered base URL string.

    Raises:
        FileNotFoundError: If neither discovery file exists.
        ValueError: If discovery file exists but contains no valid URL.
    """
    import json as _json

    discovery_dir = os.environ.get("SERVICE_DISCOVERY_DIR", "./data/services")
    txt_path = os.path.join(discovery_dir, "embedding_service_address.txt")
    json_path = os.path.join(discovery_dir, "embedding_service_address.json")

    # Try .txt file first (simple one-liner)
    if os.path.isfile(txt_path):
        url = open(txt_path).read().strip()
        if url:
            print(f"[CMI] Auto-discovered embedding service from {txt_path}: {url}",
                  flush=True)
            return url
        raise ValueError(
            f"Embedding service discovery file exists but is empty: {txt_path}"
        )

    # Try .json file
    if os.path.isfile(json_path):
        data = _json.loads(open(json_path).read())
        url = (
            data.get("base_url", "")
            or data.get("url", "")
            or data.get("address", "")
        ).strip()
        if url:
            print(f"[CMI] Auto-discovered embedding service from {json_path}: {url}",
                  flush=True)
            return url
        raise ValueError(
            f"Embedding service discovery file exists but contains no valid URL: "
            f"{json_path} → {data}"
        )

    raise FileNotFoundError(
        f"Embedding service discovery files not found. Checked:\n"
        f"  - {txt_path}\n"
        f"  - {json_path}\n"
        f"Is the embedding service running? Deploy it via:\n"
        f"  bash scripts/submit_embedding_service.sh"
    )


def get_cluster_embedding_func():
    """Create an embedding function backed by the cluster embedding service.

    Returns a ``Callable[[List[str]], List[List[float]]]`` that resolves the
    cluster embedding service URL via discovery files (no fallback — raises on
    failure) and calls it through ``LLMClient._get_embeddings_remote``.

    Thread-safe: the underlying LLMClient is created once under lock.

    This is the **single source of truth** for all MemorySystem / QA-reward
    embedding calls — both batch and agentic modes.
    """
    global _CMI_EMBEDDING_CLIENT

    if _CMI_EMBEDDING_CLIENT is None:
        with _CMI_EMBEDDING_CLIENT_LOCK:
            if _CMI_EMBEDDING_CLIENT is None:
                _init_embedding_client()

    client = _CMI_EMBEDDING_CLIENT

    def _embedding_func(texts: List[str]) -> List[List[float]]:
        return client.get_embeddings(texts)

    return _embedding_func


def _init_embedding_client():
    """Initialize the shared embedding client (called under lock)."""
    global _CMI_EMBEDDING_CLIENT
    from llm_client import LLMClient

    api_key = os.environ.get(
        "OPENAI_EMBEDDINGS_API_KEY",
        os.environ.get("LLM_API_KEY", "EMPTY"),
    )
    base_url = os.environ.get("OPENAI_EMBEDDINGS_BASE_URL", "").strip()

    if not base_url:
        base_url = _discover_embedding_service_url()

    embedding_model = os.environ.get("MEMBUILDER_INPROC_EMBEDDING_MODEL", "text-embedding-v4")

    _CMI_EMBEDDING_CLIENT = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=embedding_model,
        embedding_model=embedding_model,
    )

    provider = os.environ.get("EMBEDDING_PROVIDER", "").strip().lower()
    local_model = os.environ.get("LOCAL_EMBEDDING_MODEL", "").strip()
    print(f"[CMI] Embedding client initialized: provider={provider or 'remote'}, "
          f"base_url={base_url}, model={local_model or embedding_model}",
          flush=True)


def _get_embeddings(texts: List[str], config: Dict[str, Any]) -> "np.ndarray":
    """Get embeddings as a numpy array for efficient vectorized computation.

    Uses the shared cluster embedding client (initialized via _init_embedding_client).
    Thread-safe: initialization is protected by _CMI_EMBEDDING_CLIENT_LOCK.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), L2-normalized.
    """
    import numpy as np

    global _CMI_EMBEDDING_CLIENT

    if _CMI_EMBEDDING_CLIENT is None:
        with _CMI_EMBEDDING_CLIENT_LOCK:
            if _CMI_EMBEDDING_CLIENT is None:
                _init_embedding_client()

    raw = _CMI_EMBEDDING_CLIENT.get_embeddings(texts)
    emb = np.asarray(raw, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    emb = emb / norms
    return emb

def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors (numpy-aware).

    When inputs are L2-normalized numpy arrays (as returned by _get_embeddings),
    this reduces to a simple dot product.
    """
    import numpy as np
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape[0] == 0 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))
