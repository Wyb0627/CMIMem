#!/usr/bin/env python3
"""
Prepare RL Training Data for Online Rollout (CMI-Mem).

Unlike prepare_rl_data.py which requires expert trajectories, this script
creates training data directly from raw datasets using the base model itself
to generate state snapshots (memory_before for each session).

Key differences from expert-trajectory-based preparation:
- No expert_output in ground_truth (CMI reward doesn't need it)
- state_before_path generated via base model self-play
- Optionally includes QA questions (for cmi_plus_qa mode)

Input: Raw dataset (longmemeval / locomo)
Output: data/memory_rl_train_online.parquet

The parquet file contains:
- prompt: Full agent prompt with state and session
- reward_model: Configuration for reward computation (state_before_path, etc.)
- ability: Task type marker
- data_source: "memory_management_online"

Usage:
    # Generate with base model rollout
    python scripts/prepare_rl_data_online.py \
        --dataset longmemeval \
        --output-file data/memory_rl_train_online.parquet \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --num-samples 50

    # Dry run (no LLM calls, just validate data loading)
    python scripts/prepare_rl_data_online.py \
        --dataset locomo \
        --output-file /tmp/test.parquet \
        --num-samples 2 --dry-run

    # With QA generation for cmi_plus_qa mode
    python scripts/prepare_rl_data_online.py \
        --dataset longmemeval \
        --output-file data/memory_rl_train_online.parquet \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --add-qa --qa-per-session 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_system import MemorySystem
from config import QA_GENERATION_MODEL

try:
    from llm_client_internal import create_llm_client, DEFAULT_PROVIDER
except ImportError:
    from llm_client import create_llm_client, DEFAULT_PROVIDER


# =============================================================================
# Dataset Loading
# =============================================================================

def _resolve_locomo_path(dataset_path: Optional[str] = None) -> Path:
    """Resolve path to LoCoMo dataset file."""
    if dataset_path:
        return Path(dataset_path)
    candidates = [
        Path(__file__).parent.parent / "data" / "locomo_mc10.json",
        Path(__file__).parent.parent.parent / "locomo" / "locomo_mc10" / "data" / "locomo_mc10.json",
        Path(__file__).parent.parent.parent / "locomo" / "locomo_mc10" / "data" / "locomo_mc10.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot find LoCoMo dataset. Tried: {[str(c) for c in candidates]}"
    )


def _resolve_longmemeval_path(dataset_path: Optional[str] = None) -> Path:
    """Resolve path to LongMemEval dataset file."""
    if dataset_path:
        return Path(dataset_path)
    candidates = [
        Path(__file__).parent.parent / "data" / "longmemeval" / "longmemeval_s_cleaned.json",
        Path(__file__).parent.parent.parent / "longmemeval" / "longmemeval_s.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot find LongMemEval dataset. Tried: {[str(c) for c in candidates]}"
    )


def _locomo_conv_sessions(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract sessions from a LoCoMo conversation sample."""
    sessions = []
    observations = sample.get("observation", [])
    for obs in observations:
        if isinstance(obs, dict) and obs.get("content"):
            messages = obs.get("content", [])
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            sessions.append({
                "messages": messages,
                "timestamp": obs.get("timestamp"),
            })
    return sessions


def load_conversations(
    dataset: str,
    dataset_path: Optional[str] = None,
    conv_ids: Optional[List[str]] = None,
    num_samples: Optional[int] = None,
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load conversations from dataset.

    Returns list of dicts with keys: conv_id, sessions
    Each session has: messages (list of {role, content}), timestamp (optional)
    """
    conversations = []

    if dataset == "locomo":
        path = _resolve_locomo_path(dataset_path)
        print(f"Loading LoCoMo from: {path}")

        if str(path).endswith(".jsonl"):
            with open(path) as f:
                data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = list(data.values())

        for item in data:
            cid = item.get("conversation_id", item.get("id", ""))
            if conv_ids and cid not in conv_ids:
                continue
            conversations.append({
                "conv_id": str(cid),
                "sessions": _locomo_conv_sessions(item),
            })

    elif dataset == "longmemeval":
        path = _resolve_longmemeval_path(dataset_path)
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
                selected_ids = set(splits.get(split, []))
                print(f"  Using split '{split}': {len(selected_ids)} conversations")

        for item in data:
            cid = item.get("conversation_id", item.get("id", ""))
            if conv_ids and str(cid) not in conv_ids:
                continue
            if selected_ids and str(cid) not in selected_ids:
                continue

            sessions = []
            for sess in item.get("sessions", []):
                messages = sess.get("messages", [])
                if isinstance(messages, str):
                    messages = [{"role": "user", "content": messages}]
                sessions.append({
                    "messages": messages,
                    "timestamp": sess.get("timestamp"),
                })
            if sessions:
                conversations.append({
                    "conv_id": str(cid),
                    "sessions": sessions,
                })
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: locomo, longmemeval")

    if num_samples and len(conversations) > num_samples:
        conversations = conversations[:num_samples]

    print(f"Loaded {len(conversations)} conversations")
    return conversations


# =============================================================================
# Base Model Rollout (State Snapshot Generation)
# =============================================================================

def generate_state_snapshots(
    conversations: List[Dict[str, Any]],
    model: str,
    output_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Dict[int, str]]:
    """Run base model through each session to generate memory state snapshots.

    For each conversation, processes sessions sequentially:
    1. Save memory state BEFORE processing session i → state_i/
    2. Run 4 agents on session i messages
    3. Apply agent outputs to memory
    4. Repeat for session i+1

    Args:
        conversations: List of {conv_id, sessions}
        model: Base model name for LLM client
        output_dir: Directory to store state snapshots
        dry_run: If True, skip LLM calls and create empty states

    Returns:
        Dict mapping conv_id → {session_idx: state_before_path}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    state_paths: Dict[str, Dict[int, str]] = {}

    for conv_idx, conv in enumerate(conversations):
        conv_id = conv["conv_id"]
        sessions = conv["sessions"]
        conv_dir = output_dir / conv_id / "states"
        conv_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{conv_idx + 1}/{len(conversations)}] {conv_id}: "
              f"{len(sessions)} sessions")

        state_paths[conv_id] = {}

        if dry_run:
            # Create empty state directories without LLM calls
            for session_idx in range(len(sessions)):
                state_dir = conv_dir / f"state_{session_idx}"
                state_dir.mkdir(exist_ok=True)
                # Write minimal core_memory.json
                with open(state_dir / "core_memory.json", "w") as f:
                    json.dump({"human": ""}, f)
                state_paths[conv_id][session_idx] = str(state_dir)
            continue

        # Real rollout with base model
        llm_client = create_llm_client(provider=DEFAULT_PROVIDER, model=model)
        memory = MemorySystem(llm_client=llm_client)

        for session_idx, session in enumerate(sessions):
            # Save state BEFORE this session
            state_dir = conv_dir / f"state_{session_idx}"
            memory.save_state(str(state_dir))
            state_paths[conv_id][session_idx] = str(state_dir)

            # Process session with base model
            messages = session.get("messages", [])
            if not messages:
                continue

            try:
                user_id = conv_id
                timestamp = session.get("timestamp")
                metadata = {"timestamp": timestamp} if timestamp else None

                # Run memory construction (all 4 agents)
                memory.add(
                    messages=messages,
                    user_id=user_id,
                    metadata=metadata,
                )
            except Exception as e:
                print(f"    ⚠️ Session {session_idx} failed: {str(e)[:80]}. "
                      f"Continuing with current state.")

    return state_paths


# =============================================================================
# Prompt Building
# =============================================================================

def build_agent_prompt(
    memory: MemorySystem,
    agent_type: str,
    messages: List[Dict[str, Any]],
    user_id: str,
    timestamp: Optional[str] = None,
) -> str:
    """Build the full prompt for a specific agent type.

    Replicates the prompt construction logic from MemorySystem's agent methods.
    """
    from prompts import (
        CORE_MEMORY_PROMPT, EPISODIC_MEMORY_PROMPT,
        SEMANTIC_MEMORY_PROMPT, PROCEDURAL_MEMORY_PROMPT,
    )
    from config import CORE_MEMORY_CHAR_LIMIT

    formatted_messages = memory._format_messages(messages)

    if agent_type == "core":
        current_core = memory.core_memory.human or "[Empty]"
        usage = memory.core_memory.get_usage()
        prompt = CORE_MEMORY_PROMPT.format(
            current_core_memory=current_core,
            char_limit=CORE_MEMORY_CHAR_LIMIT,
            usage_percent=f"{usage:.1f}",
            new_session=formatted_messages,
        )
    else:
        # Get existing memories for context
        existing = memory._get_existing_memories(agent_type, user_id, messages)
        existing_str = "\n".join(
            f"- [{i}] {mem}" for i, mem in enumerate(existing[:50])
        ) if existing else "(no existing memories)"

        if agent_type == "episodic":
            prompt_template = EPISODIC_MEMORY_PROMPT
        elif agent_type == "semantic":
            prompt_template = SEMANTIC_MEMORY_PROMPT
        else:  # procedural
            prompt_template = PROCEDURAL_MEMORY_PROMPT

        # Build prompt with existing memories and new session
        prompt = prompt_template.format(
            existing_memories=existing_str,
            new_session=formatted_messages,
        )

    return prompt


# =============================================================================
# Parquet Record Building
# =============================================================================

def build_parquet_records(
    conversations: List[Dict[str, Any]],
    state_paths: Dict[str, Dict[int, str]],
    model: str,
    add_qa: bool = False,
    qa_per_session: int = 5,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Build verl-compatible parquet records.

    For each session in each conversation, creates 4 records (one per agent).
    """
    agent_order = ["core", "episodic", "semantic", "procedural"]
    agent_index_map = {a: i for i, a in enumerate(agent_order)}
    records: List[Dict[str, Any]] = []

    qa_generator = None
    qa_llm_client = None
    if add_qa and not dry_run:
        from qa_generator import QAGenerator
        qa_llm_client = create_llm_client(provider=DEFAULT_PROVIDER, model=QA_GENERATION_MODEL)
        qa_generator = QAGenerator(llm_client=qa_llm_client, num_questions=qa_per_session)

    for conv in conversations:
        conv_id = conv["conv_id"]
        sessions = conv["sessions"]
        conv_state_paths = state_paths.get(conv_id, {})

        for session_idx, session in enumerate(sessions):
            messages = session.get("messages", [])
            timestamp = session.get("timestamp")
            if not messages:
                continue

            state_before_path = conv_state_paths.get(session_idx, "")
            session_group_id = f"{conv_id}_sess{session_idx}"

            # Generate QA questions if requested
            qa_questions: List[Dict[str, Any]] = []
            if add_qa and qa_generator and qa_llm_client and state_before_path:
                try:
                    mem = MemorySystem(llm_client=qa_llm_client)
                    mem.load_state(state_before_path)

                    current_session_text = mem._format_messages(messages)
                    episodic = [m for m in mem.vector_store.memories if m.startswith("[EPISODIC]")]
                    semantic = [m for m in mem.vector_store.memories if m.startswith("[SEMANTIC]")]
                    procedural = [m for m in mem.vector_store.memories if m.startswith("[PROCEDURAL]")]

                    qa_questions = qa_generator.generate_qa(
                        current_session=current_session_text,
                        session_timestamp=str(timestamp or session_idx),
                        core_memory=mem.core_memory.human,
                        episodic_memories="\n".join(episodic),
                        semantic_memories="\n".join(semantic),
                        procedural_memories="\n".join(procedural),
                    )
                except Exception as e:
                    print(f"    ⚠️ QA generation failed for {conv_id}_sess{session_idx}: "
                          f"{str(e)[:60]}")
                    qa_questions = []

            # Build prompts for each agent
            # Load memory state for prompt construction
            if not dry_run and state_before_path:
                try:
                    prompt_client = create_llm_client(provider=DEFAULT_PROVIDER, model=model)
                    prompt_memory = MemorySystem(llm_client=prompt_client)
                    prompt_memory.load_state(state_before_path)
                except Exception:
                    prompt_memory = MemorySystem(
                        llm_client=create_llm_client(provider=DEFAULT_PROVIDER, model=model)
                    )
            else:
                prompt_memory = None

            for agent_type in agent_order:
                # Build prompt
                if prompt_memory and not dry_run:
                    try:
                        full_prompt = build_agent_prompt(
                            prompt_memory, agent_type, messages, conv_id, timestamp
                        )
                    except Exception:
                        full_prompt = f"[{agent_type}] Process the following session messages."
                else:
                    full_prompt = f"[{agent_type}] Process the following session messages."

                prompt_messages = [{"role": "user", "content": full_prompt}]

                # Build ground truth (no expert_output for online mode)
                current_agent_gt = {
                    "agent_type": agent_type,
                    "conversation_id": conv_id,
                    "session_index": session_idx,
                    "qa_questions": qa_questions,
                    "new_session_messages": messages,
                    "state_before_path": state_before_path,
                    # No expert_output — CMI reward doesn't need it
                }

                # Other agents' ground truth (for session grouping in reward)
                other_agents_gt = [
                    {
                        "agent_type": at,
                        "conversation_id": conv_id,
                        "session_index": session_idx,
                    }
                    for at in agent_order if at != agent_type
                ]

                ground_truth_ordered = [current_agent_gt] + other_agents_gt

                agent_reward_config = {
                    "style": "custom",
                    "conversation_id": conv_id,
                    "session_index": session_idx,
                    "agent_type": agent_type,
                    "ground_truth": ground_truth_ordered,
                    "qa_questions": qa_questions,
                    "state_before_path": state_before_path,
                    "new_session_messages": messages,
                    # Mark as online mode (no expert)
                    "online_mode": True,
                }

                record = {
                    "prompt": prompt_messages,
                    "ability": "memory_management",
                    "reward_model": agent_reward_config,
                    "data_source": "memory_management_online",
                    "meta": {
                        "conversation_id": conv_id,
                        "session_index": session_idx,
                        "agent_type": agent_type,
                        "agent_index": agent_index_map.get(agent_type, -1),
                        "session_group_id": session_group_id,
                        "is_grouped_training": True,
                        "online_mode": True,
                    },
                }
                records.append(record)

    return records


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare RL training data for online rollout (no expert trajectories)"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["longmemeval", "locomo"],
                        help="Dataset name")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Custom path to dataset file")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output parquet file path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model for state snapshot generation")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of conversations to process")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split (for longmemeval)")
    parser.add_argument("--conv-ids", type=str, nargs="+", default=None,
                        help="Specific conversation IDs to process")
    parser.add_argument("--states-dir", type=str, default=None,
                        help="Directory for state snapshots (default: auto)")
    parser.add_argument("--add-qa", action="store_true",
                        help="Generate QA pairs (for cmi_plus_qa mode)")
    parser.add_argument("--qa-per-session", type=int, default=5,
                        help="Number of QA pairs per session")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls, create minimal data for testing")
    parser.add_argument("--skip-rollout", action="store_true",
                        help="Skip rollout if states already exist")

    args = parser.parse_args()

    # Determine states directory
    if args.states_dir:
        states_dir = Path(args.states_dir)
    else:
        output_path = Path(args.output_file)
        states_dir = output_path.parent / f"states_{args.dataset}_online"

    print(f"=" * 60)
    print(f"Online RL Data Preparation (CMI-Mem)")
    print(f"=" * 60)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Model:       {args.model}")
    print(f"  Output:      {args.output_file}")
    print(f"  States dir:  {states_dir}")
    print(f"  Num samples: {args.num_samples or 'all'}")
    print(f"  Add QA:      {args.add_qa}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"=" * 60)

    # Step 1: Load conversations
    print("\n[Step 1] Loading conversations...")
    conversations = load_conversations(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        conv_ids=args.conv_ids,
        num_samples=args.num_samples,
        split=args.split,
    )

    if not conversations:
        print("ERROR: No conversations loaded. Check dataset path.")
        sys.exit(1)

    total_sessions = sum(len(c["sessions"]) for c in conversations)
    print(f"  Total sessions: {total_sessions}")

    # Step 2: Generate state snapshots via base model rollout
    print("\n[Step 2] Generating state snapshots...")

    # Check if states already exist (for --skip-rollout)
    if args.skip_rollout and states_dir.exists():
        print("  --skip-rollout: Loading existing states...")
        state_paths: Dict[str, Dict[int, str]] = {}
        for conv in conversations:
            conv_id = conv["conv_id"]
            conv_states_dir = states_dir / conv_id / "states"
            state_paths[conv_id] = {}
            for session_idx in range(len(conv["sessions"])):
                state_dir = conv_states_dir / f"state_{session_idx}"
                if state_dir.exists():
                    state_paths[conv_id][session_idx] = str(state_dir)
        loaded_count = sum(len(v) for v in state_paths.values())
        print(f"  Loaded {loaded_count} existing state snapshots")
    else:
        start_time = time.time()
        state_paths = generate_state_snapshots(
            conversations=conversations,
            model=args.model,
            output_dir=states_dir,
            dry_run=args.dry_run,
        )
        elapsed = time.time() - start_time
        snapshot_count = sum(len(v) for v in state_paths.values())
        print(f"  Generated {snapshot_count} state snapshots in {elapsed:.1f}s")

    # Step 3: Build parquet records
    print("\n[Step 3] Building parquet records...")
    records = build_parquet_records(
        conversations=conversations,
        state_paths=state_paths,
        model=args.model,
        add_qa=args.add_qa,
        qa_per_session=args.qa_per_session,
        dry_run=args.dry_run,
    )

    if not records:
        print("ERROR: No records generated.")
        sys.exit(1)

    # Step 4: Save to parquet
    print(f"\n[Step 4] Saving {len(records)} records to parquet...")
    df = pd.DataFrame(records)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_file, engine="pyarrow", index=False)

    # Statistics
    print(f"\n{'=' * 60}")
    print(f"Done! Saved {len(df)} records to {args.output_file}")
    print(f"{'=' * 60}")
    print(f"  Conversations: {len(conversations)}")
    print(f"  Sessions:      {total_sessions}")
    print(f"  Records:       {len(df)} (4 agents × {total_sessions} sessions)")
    print(f"  With QA:       {args.add_qa}")

    # Per-agent breakdown
    agent_counts = defaultdict(int)
    for r in records:
        meta = r.get("meta", {})
        if isinstance(meta, dict):
            agent_counts[meta.get("agent_type", "unknown")] += 1
    print(f"  By agent type:")
    for agent_type in ["core", "episodic", "semantic", "procedural"]:
        print(f"    {agent_type}: {agent_counts.get(agent_type, 0)}")

    return df


if __name__ == "__main__":
    main()
