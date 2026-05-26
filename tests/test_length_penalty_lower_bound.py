"""
Unit tests for the response length lower bound penalty in CMI mode.

Verifies that _compute_agent_length_penalty correctly penalizes
too-short responses when no expert data is available (CMI training).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch


class TestLengthPenaltyLowerBound(unittest.TestCase):
    """Test the lower bound penalty logic in _compute_agent_length_penalty."""

    def setUp(self):
        """Set up test config matching production defaults."""
        self.config = {
            "enable_length_penalty": True,
            "length_penalty_weight": 0.3,
            "core_length_start_penalty": 150,
            "core_length_max_penalty": 400,
            "other_length_upper_tolerance": 1.3,
            "other_length_lower_tolerance": 0.5,
            "other_length_upper_max_ratio": 2.0,
            "other_length_lower_min_ratio": 0.2,
            "other_length_min_diff_tokens": 200,
            "min_content_tokens_threshold": 100,
            "other_length_no_expert_budget": 4000,
            "other_length_no_expert_max": 6000,
            "other_length_no_expert_min_budget": 150,
            "other_length_no_expert_min_floor": 30,
            "reward_mode": "cmi_plus_qa",
        }

    def _call_penalty(self, response_tokens, agent_type="episodic"):
        """Helper to call _compute_agent_length_penalty with mocked token count."""
        from training.reward_server.reward_function import _compute_agent_length_penalty

        # Create a response string of appropriate length
        # _count_content_tokens counts actual content tokens from the response
        # We'll mock _count_content_tokens to return our desired value
        response = "x " * response_tokens  # Approximate tokens
        ground_truth = {}  # No expert data
        
        with patch('training.reward_server.reward_function._count_content_tokens',
                   return_value=response_tokens):
            return _compute_agent_length_penalty(
                response, ground_truth, agent_type, self.config
            )

    def test_below_min_content_threshold(self):
        """Tokens below min_content_tokens_threshold (100) are caught by the
        min_content check BEFORE reaching the CMI lower bound logic.
        Penalty = 1.0 - (model_tokens / min_content_tokens_threshold)."""
        # tokens=0: 1.0 (from min_content check: model_tokens <= 0 → 1.0)
        penalty = self._call_penalty(0)
        self.assertEqual(penalty, 1.0)

        # tokens=10: 1.0 - (10/100) = 0.9
        penalty = self._call_penalty(10)
        self.assertAlmostEqual(penalty, 0.9, places=2)

        # tokens=50: 1.0 - (50/100) = 0.5
        penalty = self._call_penalty(50)
        self.assertAlmostEqual(penalty, 0.5, places=2)

        # tokens=99: 1.0 - (99/100) = 0.01
        penalty = self._call_penalty(99)
        self.assertAlmostEqual(penalty, 0.01, places=2)

    def test_between_threshold_and_min_budget_cmi_lower_bound(self):
        """Tokens between min_content_threshold (100) and min_budget (150)
        are handled by the CMI lower bound logic.
        Penalty = (min_budget - model_tokens) / (min_budget - min_floor)."""
        # tokens=100: (150-100)/(150-30) = 50/120 ≈ 0.417
        penalty = self._call_penalty(100)
        self.assertAlmostEqual(penalty, 50.0 / 120.0, places=2)

        # tokens=120: (150-120)/(150-30) = 30/120 = 0.25
        penalty = self._call_penalty(120)
        self.assertAlmostEqual(penalty, 0.25, places=2)

        # tokens=140: (150-140)/(150-30) = 10/120 ≈ 0.083
        penalty = self._call_penalty(140)
        self.assertAlmostEqual(penalty, 10.0 / 120.0, places=2)

    def test_at_min_budget_no_penalty(self):
        """Tokens at min_budget (150) should have zero penalty."""
        penalty = self._call_penalty(150)
        self.assertAlmostEqual(penalty, 0.0, places=5)

    def test_between_min_budget_and_upper_budget_no_penalty(self):
        """Tokens between min_budget (150) and upper_budget (4000) should have no penalty."""
        penalty = self._call_penalty(200)
        self.assertEqual(penalty, 0.0)

        penalty = self._call_penalty(500)
        self.assertEqual(penalty, 0.0)

        penalty = self._call_penalty(2000)
        self.assertEqual(penalty, 0.0)

        penalty = self._call_penalty(4000)
        self.assertEqual(penalty, 0.0)

    def test_above_upper_budget_increasing_penalty(self):
        """Tokens above upper_budget (4000) should have increasing penalty."""
        # At 5000: (5000-4000)/(6000-4000) = 0.5
        penalty = self._call_penalty(5000)
        self.assertAlmostEqual(penalty, 0.5, places=2)

        # At 6000: full penalty
        penalty = self._call_penalty(6000)
        self.assertEqual(penalty, 1.0)

        # Above 6000: still 1.0 (capped)
        penalty = self._call_penalty(8000)
        self.assertEqual(penalty, 1.0)

    def test_min_content_threshold_override(self):
        """Tokens below min_content_tokens_threshold (100) should get penalty regardless."""
        from training.reward_server.reward_function import _compute_agent_length_penalty

        # With model_tokens < min_content_tokens_threshold (100)
        # The min_content check runs BEFORE the expert/no-expert branch
        with patch('training.reward_server.reward_function._count_content_tokens',
                   return_value=5):
            penalty = _compute_agent_length_penalty(
                "short", {}, "episodic", self.config
            )
        # Should be penalized (either from min_content check or from lower bound)
        self.assertGreater(penalty, 0.5)

    def test_core_agent_not_affected(self):
        """Core agent uses its own penalty logic, not the CMI fallback."""
        from training.reward_server.reward_function import _compute_agent_length_penalty

        # Core agent with low tokens should use core penalty logic (theta_min/theta_max)
        with patch('training.reward_server.reward_function._count_content_tokens',
                   return_value=50):
            penalty = _compute_agent_length_penalty(
                "short core", {}, "core", self.config
            )
        # Core penalty: 50 < theta_min (150), so penalty = 0.0
        self.assertEqual(penalty, 0.0)

    def test_procedural_agent_also_penalized(self):
        """Procedural agent should also get lower bound penalty in CMI mode."""
        # tokens=50 < min_content_threshold (100), caught by min_content check
        # penalty = 1.0 - (50/100) = 0.5
        penalty = self._call_penalty(50, agent_type="procedural")
        self.assertAlmostEqual(penalty, 0.5, places=2)

        # tokens=120, in CMI lower bound range: (150-120)/(150-30) = 0.25
        penalty = self._call_penalty(120, agent_type="procedural")
        self.assertAlmostEqual(penalty, 0.25, places=2)


if __name__ == "__main__":
    unittest.main()
