"""Agentic RL training components for MemBuilder.

This package implements multi-turn agentic RL training where each session
becomes a turn in a trajectory, and CMI reward is computed per-turn for
dense credit assignment.

Key components:
- AgenticRLDataset: Dataset that groups sessions into trajectories
- MemoryBuilderAgentLoop: Multi-turn agent loop with in-loop CMI reward
- TrajectoryRewardManager: Aggregates per-turn rewards into reward tensors
"""
