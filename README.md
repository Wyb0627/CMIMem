# CMI-Mem: Action-Conditioned Conditional Mutual Information for Memory-Augmented LLM Training

> Reinforcement-learning framework for training large language models that build, organize, and retrieve long-term memory, supervised by a dense **Conditional Mutual Information (CMI)** reward signal.

---

## Overview

**CMI-Mem** addresses a central problem in memory-augmented LLM training: terminal QA-only rewards are sparse, high-variance, and provide little credit for the *intermediate* memory-construction actions that ultimately determine answer quality. CMI-Mem replaces — or complements — the sparse terminal reward with an **action-conditioned conditional mutual information** signal that quantifies, at each step, how much each memory-write or retrieval action increases the policy's information about the eventual answer.

Key features:

- **Four-partition cognitive memory.** Memory is partitioned into *Core / Episodic / Semantic / Procedural* slots, following the cognitive-architecture decomposition for language agents. Each partition has a distinct write schema, retrieval policy, and budget.
- **Dense CMI reward.** A residual-projection estimator computes per-step CMI between the executed action and the answer-bearing latent, providing dense per-token credit assignment that is robust to small batch sizes.
- **GRPO reinforcement learning.** Group Relative Policy Optimization is used as the underlying RL algorithm; advantages are normalized within prompt groups, eliminating the need for a learned value baseline.
- **Two rollout regimes.** A *batch* mode (per-action parallel rollout with batched reward) for fast training, and an *agentic* mode (multi-turn agent loop with per-action online reward) for evaluating long-horizon behavior.
- **End-to-end pipeline.** Supervised fine-tuning, GRPO RL, and three-benchmark evaluation (LongMemEval / PerLTQA / LoCoMo) are all driven by the same training entry script.

---

## Requirements

| Component       | Recommended                                               |
| --------------- | --------------------------------------------------------- |
| OS              | Linux (Ubuntu 20.04+) or macOS (evaluation only)          |
| Python          | `>= 3.10`                                                 |
| CUDA            | `>= 12.1` (for GPU training)                              |
| GPUs (training) | 8× A100 / H100 (80 GB) for full-scale GRPO RL             |
| GPUs (eval)     | 1× A10 / A100 sufficient                                  |
| Disk            | `>= 200 GB` (datasets + checkpoints + rollout buffers)    |
| RAM             | `>= 64 GB` (FAISS index resident in memory)               |

---

## Installation

```bash
# 1. Clone
git clone https://github.com/<anonymous>/cmi-mem.git
cd cmi-mem

# 2. Create a clean conda environment
conda create -n cmi-mem python=3.10 -y
conda activate cmi-mem

# 3. Install PyTorch matched to your CUDA toolkit
#    (example for CUDA 12.1; adjust as needed)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Install project dependencies
pip install -r requirements.txt

# 5. (Optional) install verl from source if the PyPI release lags behind
git clone https://github.com/volcengine/verl.git
pip install -e verl

# 6. Pre-download base policy model
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
    --local-dir ./models/Qwen3-4B-Instruct-2507
```

> **OpenAI-compatible LLM endpoint.** The judge / answer / extractor modules call an OpenAI-compatible chat completions API. Export `OPENAI_API_KEY` and (optionally) `OPENAI_BASE_URL` before launching training or evaluation.

---

## Project Structure

```
cmi-mem/
├── training/
│   ├── reward_server/        # CMI reward & QA reward implementations
│   │   ├── cmi_reward.py     # Residual-projection CMI estimator
│   │   ├── reward_function.py# verl custom_reward_function entry
│   │   ├── action_parser.py  # Parses memory write / retrieval actions
│   │   ├── format_checker.py # Format / schema validation
│   │   └── server.py         # Optional HTTP reward server
│   ├── agentic/              # Multi-turn agent loop (agentic RL)
│   │   ├── agent_loop.py     # Per-session rollout driver
│   │   ├── reward_manager.py # Per-action online reward dispatch
│   │   └── embedding_service.py
│   ├── rl/                   # GRPO / ADRPO RL utilities
│   └── sft/                  # Supervised fine-tuning configs
├── eval/
│   ├── runner.py             # Unified evaluator for all benchmarks
│   ├── datasets.py           # LongMemEval / PerLTQA / LoCoMo loaders
│   ├── llm_judge.py          # LLM-as-judge correctness scoring
│   ├── metrics.py            # F1 / EM / judge-score aggregation
│   └── rag_baseline.py       # Vanilla RAG baseline
├── scripts/                  # Data prep, conversion, and launcher scripts
├── verl/                     # Local fork of the verl RL framework
├── configs/                  # Embedding / runtime configuration
├── data/                     # Raw and processed datasets (gitignored)
├── memory_system.py          # Four-partition memory store + FAISS retrieval
├── llm_client.py             # OpenAI-compatible LLM client
├── prompts*.py               # Memory-construction / QA prompt templates
├── train_pipeline.sh         # End-to-end training driver
└── requirements.txt
```

---

## Data Preparation

CMI-Mem trains and evaluates on three long-term-memory QA benchmarks.

### Datasets

| Benchmark       | Description                                          | Source                          |
| --------------- | ---------------------------------------------------- | ------------------------------- |
| **LongMemEval** | Multi-session conversational memory QA              | `xiaowu0162/long-mem-eval`      |
| **PerLTQA**     | Personal long-term QA with structured personas       | `Elfsong/PerLTQA`               |
| **LoCoMo**      | Long-context conversational memory                   | `snap-research/locomo`          |

Place the downloaded raw files under `data/<benchmark>/`.

### Pipeline

```bash
# 1. Convert raw conversations into normalized session JSON
python scripts/process_perltqa.py        --input data/perltqa --output data/perltqa/perltqa_v2_standard.json
python scripts/process_locomo.py         --input data/locomo  --output data/locomo/locomo_conversations.json

# 2. Build the parquet shards consumed by the verl RL trainer
python scripts/prepare_rl_data.py \
    --dataset longmemeval \
    --output  data/longmemeval/longmemeval_4agent_rl_with_session_qa.parquet

# 3. (Optional) inject curriculum ordering / ICL exemplars
python scripts/reorder_rl_data_curriculum.py --in_parquet ... --out_parquet ...
python scripts/inject_icl_into_parquet.py    --in_parquet ... --out_parquet ...

# 4. Build expert trajectories for SFT (DashScope-compatible / OpenAI-compatible)
python scripts/generate_expert_trajectories.py \
    --dataset longmemeval --num_workers 16 \
    --output data/expert_trajectories_format_sft/longmemeval/

python scripts/convert_trajectories_to_sft.py \
    --input  data/expert_trajectories_format_sft/longmemeval/ \
    --output data/memory_building_format_sft_all.json
```

---

## Training

The driver script `train_pipeline.sh` orchestrates all stages and exposes them through environment-variable switches.

### Stage 1 — Supervised Fine-Tuning (SFT)

SFT teaches the base model the four-partition memory schema and the action-emission format. Reward design is *not* involved at this stage; this only ensures the policy can produce well-formed actions before RL.

```bash
ENABLE_SFT=1 ENABLE_RL=0 \
SFT_MODEL_PATH=./models/Qwen3-4B-Instruct-2507 \
SFT_DATA=data/memory_building_format_sft_all.json \
bash train_pipeline.sh sft
```

A reference DeepSpeed ZeRO-2 configuration is provided at `training/sft/ds_z2_config.json`, and an example single-node launcher at `training/sft/train_example.sh`.

### Stage 2 — GRPO Reinforcement Learning (CMI Reward)

This is the core stage. The policy is rolled out in *batch* mode: for each session, the four memory-construction agents emit actions in parallel; per-action CMI rewards are then computed and combined with a terminal QA-correctness reward.

```bash
# CMI + QA combined reward (recommended setting)
MEMBUILDER_REWARD_MODE=cmi_plus_qa \
CMI_WEIGHT=0.3 \
CMI_METHOD=resproj \
ROLLOUT_MODE=batch \
TRAIN_CONFIG=membuilder_cmi_grpo \
ENABLE_SFT=0 ENABLE_RL=1 \
bash train_pipeline.sh train
```

Reward modes (`MEMBUILDER_REWARD_MODE`):

| Mode           | Signal                                                 |
| -------------- | ------------------------------------------------------ |
| `qa`           | Terminal QA correctness only (sparse baseline)        |
| `cmi`          | Per-action CMI only                                    |
| `cmi_plus_qa`  | Convex combination of CMI and terminal QA (default)   |

CMI estimators (`CMI_METHOD`):

| Method     | Notes                                                  |
| ---------- | ------------------------------------------------------ |
| `resproj`  | Residual projection (default, robust to small batches) |
| `logprob`  | Direct log-probability ratio                           |
| `kernel`   | Kernel density estimate                                |

The full configuration surface is exposed in the Hydra config used by verl; see `verl/trainer/config/` for available `TRAIN_CONFIG` presets.

### Stage 3 — Agentic RL Training

Agentic mode replaces the parallel batch rollout with a multi-turn agent loop, where the policy interacts step-by-step with the memory store and observes the consequences of each action before emitting the next. Per-action CMI rewards are computed online inside the loop.

```bash
ROLLOUT_MODE=agentic \
TRAIN_CONFIG=membuilder_agentic_grpo \
MEMBUILDER_REWARD_MODE=cmi_plus_qa \
ENABLE_SFT=0 ENABLE_RL=1 \
bash train_pipeline.sh train
```

The agent-loop driver is in `training/agentic/agent_loop.py`; per-action reward dispatch is in `training/agentic/reward_manager.py`.

---

## Evaluation

A single unified runner evaluates a checkpoint on any of the three benchmarks.

```bash
# Convert verl checkpoint to HuggingFace format
bash scripts/convert_verl_to_hf.sh <verl_ckpt_dir> <hf_output_dir>

# LongMemEval
python -m eval.runner \
    --benchmark longmemeval \
    --model_path <hf_output_dir> \
    --data_path  data/longmemeval/splits/test.json \
    --output     outputs/longmemeval_eval.jsonl

# PerLTQA
python -m eval.runner \
    --benchmark perltqa \
    --model_path <hf_output_dir> \
    --data_path  data/perltqa/perltqa_v2_standard.json \
    --output     outputs/perltqa_eval.jsonl

# LoCoMo
python -m eval.runner \
    --benchmark locomo \
    --model_path <hf_output_dir> \
    --data_path  data/locomo/locomo_conversations.json \
    --output     outputs/locomo_eval.jsonl
```

Metrics (F1, EM, LLM-judge correctness) are aggregated by `eval/metrics.py`. The LLM judge uses an OpenAI-compatible chat endpoint; configure via `OPENAI_API_KEY` / `OPENAI_BASE_URL`.

### RAG Baseline

A vanilla retrieval-augmented baseline (no learned memory construction) is provided for reference:

```bash
python -m eval.rag_baseline \
    --benchmark longmemeval \
    --data_path data/longmemeval/splits/test.json \
    --top_k 5 \
    --output outputs/rag_baseline_longmemeval.jsonl
```

---

## Core Algorithm

### 1. Four-Partition Memory

The memory store is decomposed into four typed partitions:

- **Core** — stable, identity-defining facts (one-shot writes, monotonically growing).
- **Episodic** — time-stamped events tied to specific sessions.
- **Semantic** — generalized factual knowledge distilled across sessions.
- **Procedural** — reusable strategies and routines.

Each partition has its own write schema, retrieval policy, and capacity budget. Implementation: `memory_system.py`.

### 2. Action-Conditioned CMI Reward

For an action `a_t` emitted at step `t` against a session `s` and target answer `y`, CMI-Mem estimates

```
I(a_t ; y | s_<t)
```

i.e. the conditional mutual information between the action and the answer given the prior session context. Three estimators are supported (see `training/reward_server/cmi_reward.py`):

- **Residual projection** — projects answer-bearing residuals onto the action's representation; numerically stable for small RL batches.
- **Log-probability ratio** — `log p(y | s_<t, a_t) − log p(y | s_<t)` under the policy.
- **Kernel** — Gaussian-kernel CMI estimator (used in ablation only).

The per-step CMI is then combined with the terminal QA reward via a tunable weight `CMI_WEIGHT`, applied as a dense per-token shaping signal during GRPO advantage computation.

### 3. Agentic Loop

In agentic mode, the policy interacts with the memory store turn-by-turn. Each turn the agent observes the current memory state, emits a single action (write to a partition / retrieve / answer), and receives an immediate CMI reward conditioned on the still-unrevealed answer. This regime tests whether the policy learns *online credit assignment* rather than relying on the parallel batched approximation. See `training/agentic/agent_loop.py`.

---

## License

Released under the MIT License. See [LICENSE](./LICENSE) for the full text.
