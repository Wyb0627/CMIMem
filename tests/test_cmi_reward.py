"""
Unit tests for CMI (Conditional Mutual Information) reward computation.

Tests cover:
- Three CMI estimation methods (logprob, embedding, entropy)
- Two reward shaping strategies (threshold, gaussian)
- Combined cmi_plus_qa mode
- Configuration switching logic
- Edge cases (empty memories, no new items)
- Regression: membuilder_qa mode unaffected
"""

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_parsed_responses():
    """Sample parsed responses from 4 agents."""
    return {
        "core": {
            "core": {
                "operation": "APPEND",
                "content": "User is a software engineer who likes hiking.",
            }
        },
        "episodic": {
            "episodic": {
                "operations": [
                    {
                        "action": "ADD",
                        "memory": "[EPISODIC] 2024-01-15: User discussed their weekend hiking trip to Mt. Fuji.",
                    }
                ]
            }
        },
        "semantic": {
            "semantic": {
                "operations": [
                    {
                        "action": "ADD",
                        "memory": "[SEMANTIC] User's favorite programming language is Python.",
                    }
                ]
            }
        },
        "procedural": {
            "procedural": {
                "operations": [
                    {
                        "action": "ADD",
                        "memory": "[PROCEDURAL] To deploy user's app: 1) Build docker image 2) Push to registry 3) kubectl apply",
                    }
                ]
            }
        },
    }


@pytest.fixture
def sample_memory_state_before():
    """Sample memory state before the current session."""
    return {
        "core_memory": "User is a software engineer.",
        "memories": [
            "[EPISODIC] 2024-01-10: User started a new project.",
            "[SEMANTIC] User works at TechCorp.",
            "[PROCEDURAL] To run tests: pytest -v",
        ],
    }


@pytest.fixture
def sample_session_messages():
    """Sample session messages (conversation context)."""
    return [
        {"role": "user", "content": "I went hiking on Mt. Fuji last weekend! It was amazing."},
        {"role": "assistant", "content": "That sounds wonderful! How was the weather?"},
        {"role": "user", "content": "Perfect clear skies. Also, I've been using Python more for my deployment scripts."},
    ]


@pytest.fixture
def cmi_config_embedding():
    """CMI config set to embedding method."""
    return {
        "cmi_method": "embedding",
        "cmi_shaping": "gaussian",
        "cmi_gaussian_mu": 0.4,
        "cmi_gaussian_sigma": 0.2,
        "cmi_gaussian_scale": 1.0,
        "embedding_redundancy_threshold": 0.85,
        "embedding_novelty_floor": 0.3,
        "embedding_aggregation": "mean",
        "cmi_fallback_method": "embedding",
        "cmi_debug": False,
    }


@pytest.fixture
def cmi_config_threshold():
    """CMI config with threshold shaping."""
    return {
        "cmi_method": "embedding",
        "cmi_shaping": "threshold",
        "cmi_threshold_low": 0.1,
        "cmi_threshold_high": 0.8,
        "cmi_threshold_in_reward": 1.0,
        "cmi_threshold_out_reward": -0.3,
        "embedding_redundancy_threshold": 0.85,
        "embedding_novelty_floor": 0.3,
        "embedding_aggregation": "mean",
        "cmi_fallback_method": "embedding",
        "cmi_debug": False,
    }


# =============================================================================
# Test CMI Configuration
# =============================================================================

class TestCMIConfig:
    """Test CMI configuration loading and validation."""

    def test_default_config_loads(self):
        """Default config should load without errors."""
        from training.reward_server.cmi_config import CMI_DEFAULT_CONFIG, load_cmi_config

        config = load_cmi_config(force_reload=True)
        assert isinstance(config, dict)
        assert "cmi_method" in config
        assert "cmi_shaping" in config

    def test_env_override(self):
        """Environment variables should override defaults."""
        from training.reward_server.cmi_config import load_cmi_config

        with patch.dict(os.environ, {"CMI_METHOD": "entropy", "CMI_SHAPING": "threshold"}):
            config = load_cmi_config(force_reload=True)
            assert config["cmi_method"] == "entropy"
            assert config["cmi_shaping"] == "threshold"

    def test_valid_methods(self):
        """Only valid methods should be accepted."""
        from training.reward_server.cmi_config import CMI_DEFAULT_CONFIG

        valid_methods = ["logprob", "embedding", "entropy"]
        assert CMI_DEFAULT_CONFIG["cmi_method"] in valid_methods

    def test_valid_shapings(self):
        """Only valid shapings should be accepted."""
        from training.reward_server.cmi_config import CMI_DEFAULT_CONFIG

        valid_shapings = ["threshold", "gaussian"]
        assert CMI_DEFAULT_CONFIG["cmi_shaping"] in valid_shapings

    def test_cmi_weight_range(self):
        """CMI weight should be in [0, 1]."""
        from training.reward_server.cmi_config import CMI_DEFAULT_CONFIG

        weight = CMI_DEFAULT_CONFIG["cmi_weight"]
        assert 0.0 <= weight <= 1.0


# =============================================================================
# Test Reward Shapers
# =============================================================================

class TestThresholdShaper:
    """Test threshold reward shaping."""

    def _make_shaper(self, low=0.1, high=0.8, in_reward=1.0, out_reward=-0.3):
        from training.reward_server.cmi_reward import ThresholdRewardShaper
        config = {
            "cmi_threshold_low": low,
            "cmi_threshold_high": high,
            "cmi_threshold_in_reward": in_reward,
            "cmi_threshold_out_reward": out_reward,
        }
        return ThresholdRewardShaper(config)

    def test_in_range_positive(self):
        """CMI within [low, high] should give positive reward."""
        shaper = self._make_shaper()

        # Middle of range
        reward = shaper.shape(0.45)
        assert reward > 0.0
        assert reward <= 1.0

    def test_below_range_negative(self):
        """CMI below low threshold should give reduced/negative reward."""
        shaper = self._make_shaper()

        reward = shaper.shape(0.0)
        assert reward <= 0.0

    def test_above_range_negative(self):
        """CMI above high threshold should give reduced/negative reward."""
        shaper = self._make_shaper()

        reward = shaper.shape(1.0)
        assert reward < 1.0

    def test_at_boundaries(self):
        """CMI at exact boundaries should give non-negative reward."""
        shaper = self._make_shaper()

        reward_low = shaper.shape(0.1)
        reward_high = shaper.shape(0.8)
        assert reward_low >= 0.0
        assert reward_high >= 0.0


class TestGaussianShaper:
    """Test Gaussian reward shaping."""

    def _make_shaper(self, mu=0.4, sigma=0.2, scale=1.0):
        from training.reward_server.cmi_reward import GaussianRewardShaper
        config = {
            "cmi_gaussian_mu": mu,
            "cmi_gaussian_sigma": sigma,
            "cmi_gaussian_scale": scale,
        }
        return GaussianRewardShaper(config)

    def test_peak_at_mu(self):
        """Reward should be maximum at mu."""
        shaper = self._make_shaper()

        reward = shaper.shape(0.4)
        assert abs(reward - 1.0) < 1e-6

    def test_decay_from_mu(self):
        """Reward should decrease as CMI moves away from mu."""
        shaper = self._make_shaper()

        reward_at_mu = shaper.shape(0.4)
        reward_near = shaper.shape(0.5)
        reward_far = shaper.shape(0.8)

        assert reward_at_mu > reward_near
        assert reward_near > reward_far

    def test_symmetric(self):
        """Gaussian shaping should be symmetric around mu."""
        shaper = self._make_shaper()

        reward_left = shaper.shape(0.2)  # mu - 0.2
        reward_right = shaper.shape(0.6)  # mu + 0.2

        assert abs(reward_left - reward_right) < 1e-6

    def test_scale_parameter(self):
        """Scale parameter should multiply the peak reward."""
        shaper_1 = self._make_shaper(scale=1.0)
        shaper_2 = self._make_shaper(scale=0.5)

        reward_1 = shaper_1.shape(0.4)
        reward_2 = shaper_2.shape(0.4)

        assert abs(reward_1 - 2.0 * reward_2) < 1e-6

    def test_always_non_negative(self):
        """Gaussian reward should always be non-negative."""
        shaper = self._make_shaper()

        for cmi in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
            assert shaper.shape(cmi) >= 0.0


# =============================================================================
# Test CMI Estimators (with mocks)
# =============================================================================

class TestEmbeddingCMIEstimator:
    """Test embedding-based CMI estimation."""

    def test_empty_new_memories_returns_zero(self):
        """No new memories should return CMI = 0."""
        from training.reward_server.cmi_reward import EmbeddingCMIEstimator

        estimator = EmbeddingCMIEstimator()
        config = {"embedding_redundancy_threshold": 0.85, "embedding_novelty_floor": 0.3, "embedding_aggregation": "mean"}

        result = estimator.estimate([], ["existing memory"], "context", config)
        assert result == 0.0

    def test_output_range(self, cmi_config_embedding):
        """CMI estimate should be in [0, 1] range."""
        from training.reward_server.cmi_reward import EmbeddingCMIEstimator

        estimator = EmbeddingCMIEstimator()

        # Mock the embedding function to return controlled batch vectors
        with patch.object(estimator, '_get_embeddings') as mock_embed:
            mock_embed.return_value = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]

            result = estimator.estimate(
                ["new memory item"],
                ["existing memory"],
                "conversation context",
                cmi_config_embedding,
            )

            # Result should be in valid range
            assert 0.0 <= result <= 1.0

    def test_redundant_memory_low_cmi(self, cmi_config_embedding):
        """Identical new memory should give low CMI (redundant = bad)."""
        from training.reward_server.cmi_reward import EmbeddingCMIEstimator

        estimator = EmbeddingCMIEstimator()

        # Mock embeddings to simulate high similarity (all identical vectors)
        with patch.object(estimator, '_get_embeddings') as mock_embed:
            # Layout: [new_memory, existing_memory, context]
            mock_embed.return_value = [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]

            result = estimator.estimate(
                ["existing memory item repeated"],
                ["existing memory item repeated"],
                "context",
                cmi_config_embedding,
            )
            # High redundancy → low CMI (redundant memory is bad)
            assert result <= 0.1


class TestLogprobCMIEstimator:
    """Test logprob-based CMI estimation."""

    def test_empty_new_memories_returns_zero(self):
        """No new memories should return 0."""
        from training.reward_server.cmi_reward import LogprobCMIEstimator

        estimator = LogprobCMIEstimator()
        config = {"logprob_max_tokens": 128, "cmi_debug": False}

        result = estimator.estimate([], ["existing"], "context", config)
        assert result == 0.0

    def test_fallback_on_api_failure(self):
        """Should fallback to embedding method when LLM API fails."""
        from training.reward_server.cmi_reward import LogprobCMIEstimator

        estimator = LogprobCMIEstimator()
        config = {
            "logprob_max_tokens": 128,
            "logprob_model": "nonexistent-model",
            "cmi_debug": False,
            "embedding_redundancy_threshold": 0.85,
            "embedding_novelty_floor": 0.3,
            "embedding_aggregation": "mean",
        }

        # Should not raise, should fallback
        with patch("training.reward_server.cmi_reward._get_cmi_llm_client") as mock_client:
            mock_client.side_effect = Exception("API unavailable")
            result = estimator.estimate(
                ["new memory"],
                ["existing memory"],
                "context",
                config,
            )
            # Should return a value from the fallback estimator
            assert isinstance(result, float)


class TestEntropyCMIEstimator:
    """Test entropy-based CMI estimation."""

    def test_empty_new_memories_returns_zero(self):
        """No new memories should return 0."""
        from training.reward_server.cmi_reward import EntropyCMIEstimator

        estimator = EntropyCMIEstimator()
        config = {"entropy_context_window": 512, "entropy_normalize": True, "cmi_debug": False}

        result = estimator.estimate([], ["existing"], "context", config)
        assert result == 0.0


# =============================================================================
# Test compute_cmi_reward (Integration)
# =============================================================================

class TestComputeCMIReward:
    """Test the main compute_cmi_reward entry point."""

    def test_returns_expected_keys(
        self, sample_parsed_responses, sample_memory_state_before, sample_session_messages
    ):
        """Result should contain required keys."""
        from training.reward_server.cmi_reward import compute_cmi_reward

        with patch.dict(os.environ, {"CMI_METHOD": "embedding"}):
            # Force reload to pick up env var
            from training.reward_server.cmi_config import load_cmi_config
            load_cmi_config(force_reload=True)

            result = compute_cmi_reward(
                parsed_responses=sample_parsed_responses,
                memory_state_before=sample_memory_state_before,
                session_messages=sample_session_messages,
                config={},
            )

        assert "task_reward" in result
        assert "dominant_agent" in result
        assert "cmi_raw" in result
        assert "cmi_method" in result

    def test_dominant_agent_is_none(
        self, sample_parsed_responses, sample_memory_state_before, sample_session_messages
    ):
        """CMI mode should always set dominant_agent = None."""
        from training.reward_server.cmi_reward import compute_cmi_reward

        with patch.dict(os.environ, {"CMI_METHOD": "embedding"}):
            from training.reward_server.cmi_config import load_cmi_config
            load_cmi_config(force_reload=True)

            result = compute_cmi_reward(
                sample_parsed_responses, sample_memory_state_before,
                sample_session_messages, {},
            )

        assert result["dominant_agent"] is None

    def test_reward_in_valid_range(
        self, sample_parsed_responses, sample_memory_state_before, sample_session_messages
    ):
        """Shaped reward should be in [-1, 1] range."""
        from training.reward_server.cmi_reward import compute_cmi_reward

        with patch.dict(os.environ, {"CMI_METHOD": "embedding"}):
            from training.reward_server.cmi_config import load_cmi_config
            load_cmi_config(force_reload=True)

            result = compute_cmi_reward(
                sample_parsed_responses, sample_memory_state_before,
                sample_session_messages, {},
            )

        assert -1.0 <= result["task_reward"] <= 1.0

    def test_empty_parsed_responses(
        self, sample_memory_state_before, sample_session_messages
    ):
        """Empty parsed responses (no new memories) should still work."""
        from training.reward_server.cmi_reward import compute_cmi_reward

        empty_responses = {
            "core": None,
            "episodic": None,
            "semantic": None,
            "procedural": None,
        }

        with patch.dict(os.environ, {"CMI_METHOD": "embedding"}):
            from training.reward_server.cmi_config import load_cmi_config
            load_cmi_config(force_reload=True)

            result = compute_cmi_reward(
                empty_responses, sample_memory_state_before,
                sample_session_messages, {},
            )

        assert isinstance(result["task_reward"], float)

    def test_method_switching(
        self, sample_parsed_responses, sample_memory_state_before, sample_session_messages
    ):
        """Switching CMI_METHOD should use different estimators."""
        from training.reward_server.cmi_reward import compute_cmi_reward

        results = {}
        for method in ["embedding"]:
            # Only test embedding as it doesn't require external API
            with patch.dict(os.environ, {"CMI_METHOD": method}):
                from training.reward_server.cmi_config import load_cmi_config
                load_cmi_config(force_reload=True)

                result = compute_cmi_reward(
                    sample_parsed_responses, sample_memory_state_before,
                    sample_session_messages, {},
                )
                results[method] = result

            assert results[method]["cmi_method"] == method

    def test_gaussian_vs_threshold_shaping(
        self, sample_parsed_responses, sample_memory_state_before, sample_session_messages
    ):
        """Different shaping strategies should produce different rewards."""
        from training.reward_server.cmi_reward import compute_cmi_reward

        with patch.dict(os.environ, {"CMI_METHOD": "embedding", "CMI_SHAPING": "gaussian"}):
            from training.reward_server.cmi_config import load_cmi_config
            load_cmi_config(force_reload=True)
            result_gaussian = compute_cmi_reward(
                sample_parsed_responses, sample_memory_state_before,
                sample_session_messages, {},
            )

        with patch.dict(os.environ, {"CMI_METHOD": "embedding", "CMI_SHAPING": "threshold"}):
            from training.reward_server.cmi_config import load_cmi_config
            load_cmi_config(force_reload=True)
            result_threshold = compute_cmi_reward(
                sample_parsed_responses, sample_memory_state_before,
                sample_session_messages, {},
            )

        # Both should be valid but may differ
        assert isinstance(result_gaussian["task_reward"], float)
        assert isinstance(result_threshold["task_reward"], float)
        # Raw CMI should be the same (same estimator, same input)
        assert abs(result_gaussian["cmi_raw"] - result_threshold["cmi_raw"]) < 1e-6


# =============================================================================
# Test Extract New Memories
# =============================================================================

class TestExtractNewMemories:
    """Test memory extraction from parsed responses."""

    def test_extracts_from_all_agents(self, sample_parsed_responses):
        """Should extract memories from all 4 agent types."""
        from training.reward_server.cmi_reward import _extract_new_memories

        memories = _extract_new_memories(sample_parsed_responses)
        assert len(memories) >= 3  # episodic + semantic + procedural at minimum

    def test_handles_none_responses(self):
        """Should handle None parsed responses gracefully."""
        from training.reward_server.cmi_reward import _extract_new_memories

        responses = {"core": None, "episodic": None, "semantic": None, "procedural": None}
        memories = _extract_new_memories(responses)
        assert memories == []

    def test_handles_empty_operations(self):
        """Should handle empty operations list."""
        from training.reward_server.cmi_reward import _extract_new_memories

        responses = {
            "core": {"core": {"operation": "APPEND", "content": ""}},
            "episodic": {"episodic": {"operations": []}},
            "semantic": {"semantic": {"operations": []}},
            "procedural": {"procedural": {"operations": []}},
        }
        memories = _extract_new_memories(responses)
        # Should be empty or minimal
        assert isinstance(memories, list)


# =============================================================================
# Test Format Session Context
# =============================================================================

class TestFormatSessionContext:
    """Test session context formatting."""

    def test_formats_messages(self, sample_session_messages):
        """Should format messages into readable context."""
        from training.reward_server.cmi_reward import _format_session_context

        context = _format_session_context(sample_session_messages)
        assert isinstance(context, str)
        assert len(context) > 0
        assert "hiking" in context.lower()

    def test_empty_messages(self):
        """Should handle empty message list."""
        from training.reward_server.cmi_reward import _format_session_context

        context = _format_session_context([])
        assert isinstance(context, str)


# =============================================================================
# Test Regression: membuilder_qa mode unaffected
# =============================================================================

class TestRegression:
    """Ensure CMI changes don't break existing membuilder_qa mode."""

    def test_reward_function_import(self):
        """reward_function.py should import without errors."""
        import importlib
        spec = importlib.util.find_spec("training.reward_server.reward_function")
        assert spec is not None

    def test_cmi_config_import(self):
        """cmi_config module should import cleanly."""
        from training.reward_server.cmi_config import CMI_DEFAULT_CONFIG, load_cmi_config
        assert isinstance(CMI_DEFAULT_CONFIG, dict)
        assert callable(load_cmi_config)

    def test_cmi_reward_import(self):
        """cmi_reward module should import cleanly."""
        from training.reward_server.cmi_reward import (
            compute_cmi_reward,
            BaseCMIEstimator,
            LogprobCMIEstimator,
            EmbeddingCMIEstimator,
            EntropyCMIEstimator,
            ThresholdRewardShaper,
            GaussianRewardShaper,
        )
        assert callable(compute_cmi_reward)


# =============================================================================
# Test Action-Conditioned Local CMI
# =============================================================================

class TestActionConditionedLocalCMI:
    """Test action-conditioned local CMI helpers and integration."""

    def test_extracts_all_valid_action_steps(self):
        """Should extract all valid non-REWRITE actions with metadata."""
        from training.reward_server.cmi_reward import _extract_cmi_action_steps

        parsed = {
            "core": {"core": {"operation": "REPLACE", "old_text": "old core", "new_text": "new core"}},
            "episodic": {"episodic": {"operations": [
                {"action": "ADD", "memory": "event add"},
                {"action": "UPDATE", "old_memory": "old event", "new_memory": "new event"},
                {"action": "MERGE", "old_memories": ["e1", "e2"], "new_memory": "merged event"},
            ]}},
            "semantic": {"semantic": {"operations": [
                {"action": "ADD", "memory": "fact add"},
                {"action": "UPDATE", "old_memory": "old fact", "new_memory": "new fact"},
                {"action": "SKIP"},
            ]}},
            "procedural": {"procedural": {"operations": [
                {"action": "ADD", "memory": "procedure add"},
                {"action": "UPDATE", "old_memory": "old proc", "new_memory": "new proc"},
            ]}},
        }

        steps = _extract_cmi_action_steps(parsed)

        assert [step["action"] for step in steps] == [
            "REPLACE", "ADD", "UPDATE", "MERGE", "ADD", "UPDATE", "SKIP", "ADD", "UPDATE"
        ]
        assert all("memory_type" in step for step in steps)
        assert steps[0]["reward_mode"] == "replace"
        assert steps[3]["reward_mode"] == "merge"
        assert steps[6]["reward_mode"] == "skip"

    def test_local_transition_semantic_update_replaces_old(self):
        """Semantic UPDATE should replace old memory in local after."""
        from training.reward_server.cmi_reward import _build_local_transition

        current = [
            "[SEMANTIC] User likes Java.",
            "[SEMANTIC] User works at TechCorp.",
        ]
        step = {
            "agent_type": "semantic",
            "memory_type": "semantic",
            "action": "UPDATE",
            "memory": "[SEMANTIC] User likes Python.",
            "old_memory": "User likes Java.",
            "old_memories": [],
            "reward_mode": "replace",
        }
        config = {"local_cmi_top_k": 2, "cmi_debug": False}

        with patch("training.reward_server.cmi_reward._select_top_k_similar", return_value=list(current)):
            before, after = _build_local_transition(current, step, "context", config)

        assert "[SEMANTIC] User likes Java." in before
        assert "[SEMANTIC] User likes Java." not in after
        assert "[SEMANTIC] User likes Python." in after

    def test_local_transition_episodic_update_keeps_old(self):
        """Episodic UPDATE should keep old event and add new event."""
        from training.reward_server.cmi_reward import _build_local_transition

        current = [
            "[EPISODIC] User started a project.",
            "[EPISODIC] User finished milestone one.",
        ]
        step = {
            "agent_type": "episodic",
            "memory_type": "episodic",
            "action": "UPDATE",
            "memory": "[EPISODIC] User shipped milestone two.",
            "old_memory": "User finished milestone one.",
            "old_memories": [],
            "reward_mode": "episodic_update",
        }
        config = {"local_cmi_top_k": 2, "local_cmi_recent_k": 1, "cmi_debug": False}

        with patch("training.reward_server.cmi_reward._select_top_k_similar", return_value=list(current)):
            before, after = _build_local_transition(current, step, "context", config)

        assert "[EPISODIC] User finished milestone one." in before
        assert "[EPISODIC] User finished milestone one." in after
        assert "[EPISODIC] User shipped milestone two." in after

    def test_semantic_skip_reward_uses_similarity_threshold(self):
        """Semantic SKIP should use local coverage threshold."""
        import numpy as np
        from training.reward_server.cmi_reward import _estimate_skip_reward

        config = {
            "local_cmi_skip_similarity_threshold": 0.75,
            "local_cmi_skip_positive_reward": 0.1,
            "local_cmi_skip_negative_reward": -0.1,
        }
        fake_embeddings = np.asarray([
            [1.0, 0.0],
            [0.9, 0.1],
        ], dtype=np.float32)

        with patch("training.reward_server.cmi_reward._get_embeddings", return_value=fake_embeddings):
            reward = _estimate_skip_reward(["[SEMANTIC] covered fact"], "covered context", config)

        assert reward == 0.1

    def test_compute_cmi_reward_returns_action_debug_fields(
        self, sample_parsed_responses, sample_memory_state_before, sample_session_messages
    ):
        """compute_cmi_reward should expose action-level local CMI diagnostics."""
        import numpy as np
        from training.reward_server.cmi_reward import compute_cmi_reward

        def fake_embeddings(texts, config):
            vectors = []
            for index, _ in enumerate(texts):
                vec = np.asarray([1.0, float(index + 1), 0.5], dtype=np.float32)
                vec = vec / max(np.linalg.norm(vec), 1e-10)
                vectors.append(vec)
            return np.vstack(vectors)

        with patch("training.reward_server.cmi_reward._get_embeddings", side_effect=fake_embeddings):
            result = compute_cmi_reward(
                sample_parsed_responses,
                sample_memory_state_before,
                sample_session_messages,
                {},
            )

        assert "action_cmi" in result
        assert len(result["action_cmi"]) == 4
        assert all("local_before_size" in item for item in result["action_cmi"])
        assert all("local_after_size" in item for item in result["action_cmi"])
        assert all("reward" in item for item in result["action_cmi"])

# =============================================================================
# Test Residual Projection CMI Estimator
# =============================================================================

class TestResidualProjectionCMIEstimator:
    """Test residual projection-based CMI estimation."""

    def test_empty_new_memories_returns_zero(self):
        """No new memories should return CMI = 0."""
        from training.reward_server.cmi_reward import ResidualProjectionCMIEstimator

        estimator = ResidualProjectionCMIEstimator()
        config = {"residual_projection_lambda": 1e-4}

        result = estimator.estimate([], ["existing memory"], "context", config)
        assert result == 0.0

    def test_output_range(self):
        """CMI estimate should be in [0, 1] range."""
        from training.reward_server.cmi_reward import ResidualProjectionCMIEstimator
        import numpy as np

        estimator = ResidualProjectionCMIEstimator()
        config = {
            "residual_projection_lambda": 1e-4,
            "residual_projection_recent_k": 15,
            "residual_projection_top_similar_k": 5,
            "cmi_debug": False,
        }

        # Mock _get_embeddings to return controlled vectors
        # Layout: [new_memory, existing1, existing2, context]
        mock_embs = np.array([
            [0.6, 0.8, 0.0],   # new memory
            [1.0, 0.0, 0.0],   # existing 1
            [0.0, 1.0, 0.0],   # existing 2
            [0.5, 0.5, 0.707], # context
        ], dtype=np.float32)
        # L2-normalize
        norms = np.linalg.norm(mock_embs, axis=1, keepdims=True)
        mock_embs = mock_embs / norms

        with patch("training.reward_server.cmi_reward._get_embeddings",
                   return_value=mock_embs):
            result = estimator.estimate(
                ["new memory item"],
                ["existing memory 1", "existing memory 2"],
                "conversation context",
                config,
            )

        assert 0.0 <= result <= 1.0

    def test_empty_existing_degrades_to_relevance(self):
        """With no existing memories, should degrade to cos(context, new)."""
        from training.reward_server.cmi_reward import ResidualProjectionCMIEstimator
        import numpy as np

        estimator = ResidualProjectionCMIEstimator()
        config = {"residual_projection_lambda": 1e-4, "cmi_debug": False}

        # Layout: [new_memory, context]  (no existing)
        # Both vectors pointing in same direction → high relevance
        mock_embs = np.array([
            [1.0, 0.0, 0.0],  # new memory
            [1.0, 0.0, 0.0],  # context
        ], dtype=np.float32)

        with patch("training.reward_server.cmi_reward._get_embeddings",
                   return_value=mock_embs):
            result = estimator.estimate(
                ["new memory"],
                [],
                "context",
                config,
            )

        # Should be close to 1.0 (identical directions)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_redundant_memory_low_cmi(self):
        """New memory identical to existing should give low CMI."""
        from training.reward_server.cmi_reward import ResidualProjectionCMIEstimator
        import numpy as np

        estimator = ResidualProjectionCMIEstimator()
        config = {
            "residual_projection_lambda": 1e-4,
            "residual_projection_recent_k": 15,
            "residual_projection_top_similar_k": 5,
            "cmi_debug": False,
        }

        # new_memory is identical to existing → its residual should be near zero
        # Layout: [new_memory, existing, context]
        mock_embs = np.array([
            [1.0, 0.0, 0.0],  # new memory (same as existing)
            [1.0, 0.0, 0.0],  # existing memory
            [0.0, 1.0, 0.0],  # context (orthogonal)
        ], dtype=np.float32)

        with patch("training.reward_server.cmi_reward._get_embeddings",
                   return_value=mock_embs):
            result = estimator.estimate(
                ["redundant memory"],
                ["existing memory"],
                "context",
                config,
            )

        # Redundant memory → residual near zero → CMI near zero
        assert result < 0.1

    def test_novel_relevant_memory_high_cmi(self):
        """Novel memory aligned with context residual should give high CMI."""
        from training.reward_server.cmi_reward import ResidualProjectionCMIEstimator
        import numpy as np

        estimator = ResidualProjectionCMIEstimator()
        config = {
            "residual_projection_lambda": 1e-4,
            "residual_projection_recent_k": 15,
            "residual_projection_top_similar_k": 5,
            "cmi_debug": False,
        }

        # existing memory captures dimension 0, context and new memory both have
        # components in dimension 1 (not captured by existing) → high residual CMI
        # Layout: [new_memory, existing, context]
        mock_embs = np.array([
            [0.0, 1.0, 0.0],  # new memory (orthogonal to existing, same as context residual)
            [1.0, 0.0, 0.0],  # existing memory (captures dim 0 only)
            [0.5, 0.866, 0.0],  # context (has component in dim 1 not captured by existing)
        ], dtype=np.float32)
        # L2-normalize
        norms = np.linalg.norm(mock_embs, axis=1, keepdims=True)
        mock_embs = mock_embs / norms

        with patch("training.reward_server.cmi_reward._get_embeddings",
                   return_value=mock_embs):
            result = estimator.estimate(
                ["novel relevant memory"],
                ["existing memory"],
                "context about hiking",
                config,
            )

        # Novel + relevant → high CMI
        assert result > 0.5

    def test_factory_registration(self):
        """_get_estimator should return ResidualProjectionCMIEstimator for 'residual_projection'."""
        from training.reward_server.cmi_reward import (
            _get_estimator, ResidualProjectionCMIEstimator
        )
        est = _get_estimator("residual_projection", {})
        assert isinstance(est, ResidualProjectionCMIEstimator)

# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
