#!/usr/bin/env python3
"""Prepare training data for agentic RL mode.

Converts longmemeval_s_cleaned.json into per-session parquet rows
grouped by conversation_id (question_id).

Input:  longmemeval_s_cleaned.json (500 questions × ~48 sessions each)
Output: longmemeval_agentic_train.parquet

Each row represents one session within a conversation trajectory:
  - conversation_id: question_id
  - session_index: 0-based index within the conversation
  - extra_info:
      - session: {messages: [...], timestamp: "..."}
      - session_index: int
      - questions: [{question, answer, ...}]  (same for all rows in conv)
      - state_before: {}  (empty — agent loop starts from scratch)

Usage:
    python scripts/prepare_agentic_data.py \
        --input ./data/longmemeval_s_cleaned.json \
        --output ./data/longmemeval_agentic_train.parquet \
        --max-sessions 64

    # Dry run to check data
    python scripts/prepare_agentic_data.py \
        --input ./data/longmemeval_s_cleaned.json \
        --output /tmp/agentic_test.parquet \
        --max-sessions 10 --max-questions 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq


def load_longmemeval(input_path: str) -> List[Dict[str, Any]]:
    """Load longmemeval_s_cleaned.json."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions from {input_path}")
    return data


def convert_to_session_rows(
    questions_data: List[Dict[str, Any]],
    max_sessions: int = 64,
    max_questions: int = 0,
) -> List[Dict[str, Any]]:
    """Convert question-level data to per-session rows.

    Each question becomes a trajectory with multiple session rows.
    All sessions in a trajectory share the same conversation_id.
    """
    rows: List[Dict[str, Any]] = []

    if max_questions > 0:
        questions_data = questions_data[:max_questions]

    for item in questions_data:
        question_id = item["question_id"]
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [])

        # Build QA question info (attached to all sessions in this trajectory)
        qa_questions = [{
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "question_type": item.get("question_type", ""),
            "question_date": item.get("question_date", ""),
            "answer_session_ids": item.get("answer_session_ids", []),
        }]

        # Truncate sessions
        n_sessions = min(len(sessions), max_sessions)

        for session_idx in range(n_sessions):
            session_messages = sessions[session_idx]
            timestamp = dates[session_idx] if session_idx < len(dates) else ""

            extra_info = {
                "session": {
                    "messages": session_messages,
                    "timestamp": timestamp,
                },
                "session_index": session_idx,
                "questions": qa_questions,
                "state_before": {},  # Empty — agent loop starts from scratch
            }

            rows.append({
                "conversation_id": question_id,
                "session_index": session_idx,
                "extra_info": json.dumps(extra_info, ensure_ascii=False),
                "data_source": "memory_management_agentic",
            })

    return rows


def write_parquet(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Write rows to parquet file."""
    schema = pa.schema([
        ("conversation_id", pa.string()),
        ("session_index", pa.int32()),
        ("extra_info", pa.string()),
        ("data_source", pa.string()),
    ])

    arrays = {
        "conversation_id": pa.array([r["conversation_id"] for r in rows], type=pa.string()),
        "session_index": pa.array([r["session_index"] for r in rows], type=pa.int32()),
        "extra_info": pa.array([r["extra_info"] for r in rows], type=pa.string()),
        "data_source": pa.array([r["data_source"] for r in rows], type=pa.string()),
    }

    table = pa.table(arrays, schema=schema)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    print(f"Written {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare agentic RL training data")
    parser.add_argument(
        "--input", "-i",
        default="./data/longmemeval_s_cleaned.json",
        help="Path to longmemeval_s_cleaned.json",
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/longmemeval_agentic_train.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--max-sessions", type=int, default=64,
        help="Max sessions per conversation (default: 64)",
    )
    parser.add_argument(
        "--max-questions", type=int, default=0,
        help="Max questions to process (0=all, for testing)",
    )
    args = parser.parse_args()

    data = load_longmemeval(args.input)
    rows = convert_to_session_rows(data, args.max_sessions, args.max_questions)

    # Stats
    from collections import Counter
    conv_ids = [r["conversation_id"] for r in rows]
    conv_counts = Counter(conv_ids)
    print(f"\nStats:")
    print(f"  Trajectories (conversations): {len(conv_counts)}")
    print(f"  Total session rows: {len(rows)}")
    print(f"  Sessions/trajectory: min={min(conv_counts.values())}, "
          f"max={max(conv_counts.values())}, "
          f"avg={sum(conv_counts.values())/len(conv_counts):.1f}")

    write_parquet(rows, args.output)
    print("\nDone! Ready for agentic RL training.")


if __name__ == "__main__":
    main()
