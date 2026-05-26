#!/usr/bin/env python3
"""
Prepare Agentic RL Training Data.

Converts raw LongMemEval / LoCoMo datasets into per-session parquet files
compatible with AgenticRLDataset. No expert trajectories required.

Each row = one session within a conversation, with fields:
  - conversation_id: str
  - session_index: int
  - session: {messages: [...], timestamp: str}
  - state_before: {core_memory: str, episodic: [], semantic: [], procedural: []}
  - questions: [{question: str, answer: str, type: str}, ...]

The state_before field is initialized as empty for every session (the agent
loop builds memory incrementally from scratch). This avoids dependency on
expert trajectories or self-play state generation.

Usage:
    python scripts/prepare_rl_data_agentic.py \\
        --dataset longmemeval \\
        --output-file data/longmemeval_agentic_train.parquet

    python scripts/prepare_rl_data_agentic.py \\
        --dataset longmemeval \\
        --split train \\
        --output-file data/longmemeval_agentic_train.parquet

    python scripts/prepare_rl_data_agentic.py \\
        --dataset longmemeval \\
        --split val \\
        --output-file data/longmemeval_agentic_val.parquet
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_longmemeval(
    dataset_path: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load LongMemEval conversations."""
    if dataset_path:
        path = Path(dataset_path)
    else:
        candidates = [
            Path(__file__).parent.parent / "data" / "longmemeval" / "longmemeval_s_cleaned.json",
            Path(__file__).parent.parent / "data" / "longmemeval" / "longmemeval_s.json",
        ]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Cannot find LongMemEval. Tried: {[str(c) for c in candidates]}")

    print(f"Loading LongMemEval from: {path}")
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())

    # Apply split filter if specified
    selected_ids = None
    if split:
        splits_file = Path(__file__).parent.parent / "data" / "longmemeval" / "splits" / "longmemeval_splits.json"
        if splits_file.exists():
            with open(splits_file) as f:
                splits = json.load(f)
            selected_ids = set(str(sid) for sid in splits.get(split, []))
            print(f"  Using split '{split}': {len(selected_ids)} conversations")
        else:
            print(f"  Warning: splits file not found at {splits_file}, using all data")

    conversations = []
    for item in data:
        conv_id = str(item.get("conversation_id", item.get("id", "")))
        if selected_ids and conv_id not in selected_ids:
            continue

        sessions = []
        for sess in item.get("sessions", []):
            messages = sess.get("messages", [])
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            sessions.append({
                "messages": messages,
                "timestamp": sess.get("timestamp", ""),
            })

        questions = item.get("questions", [])
        if not isinstance(questions, list):
            questions = []

        if sessions:
            conversations.append({
                "conv_id": conv_id,
                "sessions": sessions,
                "questions": questions,
            })

    if num_samples and len(conversations) > num_samples:
        conversations = conversations[:num_samples]

    print(f"  Loaded {len(conversations)} conversations")
    return conversations


def conversations_to_rows(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten conversations into per-session rows for parquet."""
    rows = []
    for conv in conversations:
        conv_id = conv["conv_id"]
        sessions = conv["sessions"]
        questions = conv["questions"]

        for session_idx, session in enumerate(sessions):
            rows.append({
                "conversation_id": conv_id,
                "session_index": session_idx,
                "session": json.dumps(session, ensure_ascii=False),
                "state_before": json.dumps({
                    "core_memory": "",
                    "episodic": [],
                    "semantic": [],
                    "procedural": [],
                }, ensure_ascii=False),
                "questions": json.dumps(questions, ensure_ascii=False),
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare agentic RL training data")
    parser.add_argument("--dataset", default="longmemeval", choices=["longmemeval"])
    parser.add_argument("--dataset-path", default=None, help="Path to dataset file")
    parser.add_argument("--split", default=None, help="Dataset split: train/val/test")
    parser.add_argument("--num-samples", type=int, default=None, help="Max conversations")
    parser.add_argument(
        "--output-file",
        default="data/longmemeval_agentic_train.parquet",
        help="Output parquet path",
    )
    args = parser.parse_args()

    if args.dataset == "longmemeval":
        conversations = load_longmemeval(args.dataset_path, args.split, args.num_samples)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    rows = conversations_to_rows(conversations)
    print(f"Generated {len(rows)} session rows from {len(conversations)} conversations")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(str(output_path), index=False)
    print(f"Saved to: {output_path}")

    # Print summary stats
    sessions_per_conv = df.groupby("conversation_id").size()
    print(f"\nSummary:")
    print(f"  Conversations: {sessions_per_conv.shape[0]}")
    print(f"  Total sessions: {len(df)}")
    print(f"  Sessions/conv: mean={sessions_per_conv.mean():.1f}, "
          f"min={sessions_per_conv.min()}, max={sessions_per_conv.max()}")


if __name__ == "__main__":
    main()
