"""Reward server for verl-based RL training."""

from typing import Any

__all__ = [
    "compute_memory_reward",
    "compute_score_batch",
    "FormatChecker",
    "AttributionWeightManager",
    "compute_attribution_weights",
    "ActionParser",
    "ConstraintsChecker",
]


def __getattr__(name: str) -> Any:
    """Lazy-load package exports to keep lightweight submodule imports fast."""
    if name in {"compute_memory_reward", "compute_score_batch"}:
        from .reward_function import compute_memory_reward, compute_score_batch

        return {
            "compute_memory_reward": compute_memory_reward,
            "compute_score_batch": compute_score_batch,
        }[name]

    if name == "FormatChecker":
        from .format_checker import FormatChecker

        return FormatChecker

    if name in {"AttributionWeightManager", "compute_attribution_weights"}:
        from .attribution import AttributionWeightManager, compute_attribution_weights

        return {
            "AttributionWeightManager": AttributionWeightManager,
            "compute_attribution_weights": compute_attribution_weights,
        }[name]

    if name == "ActionParser":
        from .action_parser import ActionParser

        return ActionParser

    if name == "ConstraintsChecker":
        from .constraints_checker import ConstraintsChecker

        return ConstraintsChecker

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
