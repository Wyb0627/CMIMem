"""
Unit tests for KernelMICMIEstimator.

Tests the mathematical correctness of:
1. RBF kernel MI estimation (_estimate_mi_kde)
2. Silverman's rule bandwidth selection (_compute_bandwidth)
3. Pairwise squared distance computation (_pairwise_sq_dist)
4. Leave-one-out KDE log-density (_log_kde_loo)
5. Full estimate() method with synthetic embeddings (mocked _get_embeddings)
6. Min-samples fallback behavior
7. Per-type grouping and top-k selection

Usage:
    cd /path/to/MemBuilder
    python -m training.reward_server.test_kernel_mi
"""

import math
import sys
import os
import unittest
from unittest.mock import patch

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from training.reward_server.cmi_reward import KernelMICMIEstimator


class TestPairwiseSqDist(unittest.TestCase):
    """Test _pairwise_sq_dist static method."""

    def test_identity(self):
        """Diagonal should be zero."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        D = KernelMICMIEstimator._pairwise_sq_dist(X)
        for i in range(len(X)):
            self.assertAlmostEqual(D[i, i], 0.0, places=10)

    def test_symmetry(self):
        """Distance matrix should be symmetric."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 5))
        D = KernelMICMIEstimator._pairwise_sq_dist(X)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_known_distances(self):
        """Test known pairwise distances."""
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        D = KernelMICMIEstimator._pairwise_sq_dist(X)
        # ||[0,0] - [3,4]||^2 = 9 + 16 = 25
        self.assertAlmostEqual(D[0, 1], 25.0, places=6)
        self.assertAlmostEqual(D[1, 0], 25.0, places=6)


class TestComputeBandwidth(unittest.TestCase):
    """Test _compute_bandwidth static method."""

    def test_auto_returns_positive(self):
        """Auto bandwidth should always be positive."""
        rng = np.random.default_rng(123)
        X = rng.standard_normal((20, 4))
        Y = rng.standard_normal((20, 4))
        h = KernelMICMIEstimator._compute_bandwidth(X, Y, "auto")
        self.assertGreater(h, 0.0)

    def test_manual_bandwidth(self):
        """Manual bandwidth should be returned as-is."""
        X = np.ones((5, 2))
        Y = np.ones((5, 2))
        h = KernelMICMIEstimator._compute_bandwidth(X, Y, "0.5")
        self.assertAlmostEqual(h, 0.5)

    def test_invalid_manual_falls_back_to_auto(self):
        """Invalid manual bandwidth should fall back to auto."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((10, 3))
        Y = rng.standard_normal((10, 3))
        h = KernelMICMIEstimator._compute_bandwidth(X, Y, "not_a_number")
        self.assertGreater(h, 0.0)

    def test_silverman_scales_with_dimension(self):
        """Higher dimension should generally produce different bandwidth."""
        rng = np.random.default_rng(42)
        X_low = rng.standard_normal((50, 2))
        Y_low = rng.standard_normal((50, 2))
        h_low = KernelMICMIEstimator._compute_bandwidth(X_low, Y_low, "auto")

        X_high = rng.standard_normal((50, 100))
        Y_high = rng.standard_normal((50, 100))
        h_high = KernelMICMIEstimator._compute_bandwidth(X_high, Y_high, "auto")

        # Both should be positive; we don't assert ordering because median
        # distance in high-dim space can be larger
        self.assertGreater(h_low, 0.0)
        self.assertGreater(h_high, 0.0)


class TestLogKdeLoo(unittest.TestCase):
    """Test _log_kde_loo static method."""

    def test_output_shape(self):
        """Should return N-length array."""
        N = 10
        sq_dist = np.random.default_rng(42).standard_normal((N, N)) ** 2
        log_dens = KernelMICMIEstimator._log_kde_loo(sq_dist, h_sq=1.0, N=N)
        self.assertEqual(len(log_dens), N)

    def test_closer_points_higher_density(self):
        """Points closer together should have higher log density."""
        # Two clusters: tight (0,0) and loose (10, 10)
        X = np.array([[0.0, 0.0], [0.01, 0.01], [0.02, 0.0],
                       [10.0, 10.0], [10.5, 10.5], [11.0, 11.0]])
        sq_dist = KernelMICMIEstimator._pairwise_sq_dist(X)
        log_dens = KernelMICMIEstimator._log_kde_loo(sq_dist, h_sq=0.1, N=6)
        # Tight cluster (indices 0-2) should have higher density than loose (3-5)
        avg_tight = np.mean(log_dens[:3])
        avg_loose = np.mean(log_dens[3:])
        self.assertGreater(avg_tight, avg_loose)


class TestEstimateMiKde(unittest.TestCase):
    """Test _estimate_mi_kde static method."""

    def test_independent_variables_low_mi(self):
        """Independent X and Y should yield near-zero MI."""
        rng = np.random.default_rng(42)
        N = 100
        X = rng.standard_normal((N, 4))
        Y = rng.standard_normal((N, 4))  # Independent of X
        mi = KernelMICMIEstimator._estimate_mi_kde(X, Y, "auto")
        # MI of independent variables should be close to 0
        # (finite sample may give small positive value)
        self.assertGreaterEqual(mi, 0.0)
        self.assertLess(mi, 0.5, "MI of independent variables should be small")

    def test_dependent_variables_higher_mi(self):
        """Dependent X and Y should yield higher MI than independent."""
        rng = np.random.default_rng(42)
        N = 100
        X = rng.standard_normal((N, 4))
        Y_dep = X + rng.standard_normal((N, 4)) * 0.1  # Y ≈ X (highly dependent)
        Y_ind = rng.standard_normal((N, 4))  # Y independent of X

        mi_dep = KernelMICMIEstimator._estimate_mi_kde(X, Y_dep, "auto")
        mi_ind = KernelMICMIEstimator._estimate_mi_kde(X, Y_ind, "auto")

        self.assertGreater(mi_dep, mi_ind,
                           f"Dependent MI ({mi_dep:.4f}) should exceed "
                           f"independent MI ({mi_ind:.4f})")

    def test_too_few_samples_returns_zero(self):
        """Fewer than 2 samples should return 0."""
        X = np.array([[1.0, 2.0]])
        Y = np.array([[3.0, 4.0]])
        mi = KernelMICMIEstimator._estimate_mi_kde(X, Y, "auto")
        self.assertEqual(mi, 0.0)

    def test_non_negative(self):
        """MI should always be non-negative (clamped)."""
        rng = np.random.default_rng(77)
        for _ in range(10):
            N = rng.integers(5, 50)
            d = rng.integers(2, 10)
            X = rng.standard_normal((N, d))
            Y = rng.standard_normal((N, d))
            mi = KernelMICMIEstimator._estimate_mi_kde(X, Y, "auto")
            self.assertGreaterEqual(mi, 0.0)


class TestEstimateMethod(unittest.TestCase):
    """Test the full estimate() method with mocked embeddings."""

    def _make_config(self, **overrides):
        base = {
            "kernel_mi_top_k": 20,
            "kernel_mi_type_weights": {
                "core": 0.25, "episodic": 0.25,
                "semantic": 0.25, "procedural": 0.25,
            },
            "kernel_mi_min_samples": 3,
            "kernel_mi_normalize": True,
            "kernel_mi_embedding_batch_size": 64,
            "kernel_mi_bandwidth": "auto",
            "cmi_debug": False,
        }
        base.update(overrides)
        return base

    def _mock_embeddings(self, texts, config, batch_size=64):
        """Generate deterministic embeddings based on text content."""
        rng = np.random.default_rng(hash(str(texts)) % (2**31))
        dim = 8  # Low-dim for speed
        return rng.standard_normal((len(texts), dim)).tolist()

    def test_empty_memories_returns_zero(self):
        """No memories at all should return 0.0."""
        estimator = KernelMICMIEstimator()
        config = self._make_config()
        result = estimator.estimate([], [], "Hello", config)
        self.assertEqual(result, 0.0)

    @patch("training.reward_server.cmi_reward._get_embeddings_for_kernel_mi")
    def test_with_typed_memories(self, mock_embed):
        """Should handle typed memory items and produce a valid CMI."""
        mock_embed.side_effect = self._mock_embeddings

        existing = [
            "[CORE] User is Alice",
            "[EPISODIC] Met Bob on Monday",
            "[EPISODIC] Had lunch at cafe",
            "[EPISODIC] Discussed project X",
            "[SEMANTIC] Project X is about NLP",
            "[SEMANTIC] Alice works at CompanyY",
            "[SEMANTIC] NLP uses transformers",
            "[PROCEDURAL] To deploy: run make build",
            "[PROCEDURAL] To test: run pytest",
            "[PROCEDURAL] To lint: run flake8",
        ]
        new = [
            "[EPISODIC] Met Charlie on Tuesday",
            "[SEMANTIC] Charlie is an expert in RL",
            "[CORE] User prefers Python",
        ]

        estimator = KernelMICMIEstimator()
        config = self._make_config()
        result = estimator.estimate(new, existing, "Tell me about the project", config)

        # Should be in [0, 1] when normalized
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    @patch("training.reward_server.cmi_reward._get_embeddings_for_kernel_mi")
    def test_min_samples_fallback(self, mock_embed):
        """Types with fewer than min_samples should be skipped."""
        mock_embed.side_effect = self._mock_embeddings

        # Only 1 episodic item (below min_samples=3)
        existing = ["[EPISODIC] One event"]
        new = ["[EPISODIC] Another event"]

        estimator = KernelMICMIEstimator()
        config = self._make_config(kernel_mi_min_samples=5)

        result = estimator.estimate(new, existing, "context", config)
        # With all types below min_samples, weighted_cmi stays 0
        # After sigmoid normalization: sigmoid(0) = 0.5
        self.assertAlmostEqual(result, 0.5, places=2)

    @patch("training.reward_server.cmi_reward._get_embeddings_for_kernel_mi")
    def test_unnormalized_output(self, mock_embed):
        """Without normalization, output can be any real number."""
        mock_embed.side_effect = self._mock_embeddings

        existing = [f"[EPISODIC] Event {i}" for i in range(10)]
        new = [f"[EPISODIC] New event {i}" for i in range(5)]

        estimator = KernelMICMIEstimator()
        config = self._make_config(kernel_mi_normalize=False)
        result = estimator.estimate(new, existing, "What happened?", config)

        # Should be a finite number
        self.assertTrue(math.isfinite(result))


class TestFactoryRegistration(unittest.TestCase):
    """Test that kernel_mi is properly registered in _get_estimator."""

    def test_factory_returns_kernel_mi(self):
        from training.reward_server.cmi_reward import _get_estimator
        estimator = _get_estimator("kernel_mi", {})
        self.assertIsInstance(estimator, KernelMICMIEstimator)


if __name__ == "__main__":
    print("=" * 60)
    print("KernelMICMIEstimator Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
