# CMI-Mem · MemBuilder + Action-Conditioned CMI Reward + Agentic RL

[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-red)]()
[![ModelScope](https://img.shields.io/badge/ModelScope-CMIMem--4B-blue)]()

A long-term memory construction framework built upon [MemBuilder (ACL 2026)](https://arxiv.org/abs/2601.05488), with two core improvements:

1. **Action-Conditioned Local CMI Reward**: Replaces the QA judge with Conditional Mutual Information as the RL reward signal, enabling dense rewards without requiring expert trajectories.
2. **Agentic RL Training Framework**: Transforms the 4-agent batch rollout into a multi-turn agent loop, computing per-action CMI rewards for fine-grained credit assignment.

This repository provides the core memory system, evaluation framework, and data preparation tools for local development and experimentation. Training is based on the **[verl](https://github.com/volcengine/verl)** framework with the GRPO algorithm.

---

## Table of Contents

1. [Core Technical Approach](#1-core-technical-approach)
2. [Differences from Original MemBuilder](#2-differences-from-original-membuilder)
3. [Training Mode Switching](#3-training-mode-switching)
4. [Environment Setup](#4-environment-setup)
5. [Data Preparation](#5-data-preparation)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Repository Structure](#8-repository-structure)
9. [Troubleshooting](#9-troubleshooting)
10. [Citation](#10-citation)

---

## 1. Core Technical Approach

### 1.1 Action-Conditioned Local CMI Reward

**Problem**: The original MemBuilder uses a QA judge as the task reward, which has two limitations:
- Relies on an external LLM judge (expensive, slow, unstable)
- Scalar reward is broadcast to all 4 agents, unable to distinguish individual contributions

**Solution**: Use Conditional Mutual Information (CMI) to measure the information gain of each memory action:

```
CMI(action) = I(M_after ; Context | M_before)
```

Where `M_before` / `M_after` are the memory states before/after the action, and `Context` is the current session text.

**Key Design**:
- **Action-Conditioned**: Computes CMI per action rather than at the session level; each ADD/UPDATE/MERGE operation has an independent reward signal
- **Local Neighborhood**: Computes only on the local memory neighborhood affected by the action (top-k similar entries), avoiding full memory computation overhead
- **Multiple CMI Estimation Methods**:
  - `kernel_mi` (recommended): RBF kernel density estimation based on embedding space mutual information
  - `embedding`: Cosine similarity difference, fastest but slightly less accurate
  - `logprob`: LLM log-probability difference
  - `entropy`: Perplexity reduction
- **Reward Shaping**: `gaussian` (bell curve) or `threshold` (fixed interval), mapping raw CMI to shaped rewards in [-1, 1]

**Implementation**: `training/reward_server/cmi_reward.py`

### 1.2 Agentic RL Training Framework

**Problem**: In the original MemBuilder batch rollout mode, 4 agents generate independently and rewards are computed after merging, failing to capture temporal dependencies between actions.

**Solution**: Leverage verl's `AgentLoopManager` to implement multi-turn agentic RL:

```
Trajectory = [Session_0, Session_1, ..., Session_N]  (one conversation)
    └── Turn_t = Session_t:
          1. Build prompt (current memory state + session context)
          2. Model generation: unified JSON output (core + episodic + semantic + procedural)
          3. Parse + execute actions → update memory state
          4. Compute turn-level CMI reward (dense)
```

**Key Design**:

- **Unified Output**: The model outputs a single JSON per turn containing operations for all 4 memory types, replacing the original 4 independent agent generations
- **Deterministic Execution Order**: Actions are executed in core → episodic → semantic → procedural order to ensure reproducibility
- **Lightweight Memory State** (`LightMemoryState`): Uses keyword-overlap approximate retrieval during rollout (no FAISS / no embedding API calls); real embedding CMI is only used during reward computation
- **Dense Reward**: Each turn ends with a CMI reward (not just at trajectory end), written to the corresponding position in `reward_tensor` via `TrajectoryRewardManager`
- **Context Length Control**: Each turn inputs full core memory + top-k similar episodic/semantic/procedural memories + current session text

**Implementation**: `training/agentic/` directory

### 1.3 Dual-Mode Training Architecture

Switch via the `ROLLOUT_MODE` environment variable; the two modes are completely independent:

| Dimension | Batch Mode (Default) | Agentic Mode |
|---|---|---|
| **Env Variable** | `ROLLOUT_MODE=batch` | `ROLLOUT_MODE=agentic` |
| **Rollout** | Standard batch vLLM, 4 agents generate separately | `AgentLoopManager`, multi-turn sequential |
| **Model Output** | 4 independent JSONs (one per agent) | 1 unified JSON (4 agents combined) |
| **Reward** | CMI / QA / CMI+QA (`BatchRewardManager`) | Per-turn CMI (`TrajectoryRewardManager`) |
| **Data** | per-session parquet (`prepare_rl_data_online.py`) | per-session parquet (`prepare_rl_data_agentic.py`) |
| **Config** | `cmi_grpo.yaml` | `agentic_grpo.yaml` |

---

## 2. Differences from Original MemBuilder

| Dimension | Original MemBuilder | This Repository |
|---|---|---|
| **Reward Signal** | QA judge (external LLM evaluates memory answer quality) | Action-Conditioned Local CMI (no external judge needed) |
| **Agent Architecture** | 4 independent agents generating separately (core/episodic/semantic/procedural) | Unified model single output + Agentic RL multi-turn |
| **Credit Assignment** | Scalar task_reward broadcast to all agents | Per-action CMI reward, dense per-turn |
| **Expert Trajectory** | Required (for both SFT and RL data) | Not required in Agentic mode |
| **LLM Backend** | Claude / OpenAI | DashScope (`deepseek-v4-pro` + `Qwen3-Embedding-4B`) |
| **Training Cluster** | Local Ray | Local / custom cluster (verl-based) |
| **Reward Computation** | External HTTP reward server | In-process (worker process directly connects to DashScope) |
| **Embedding** | Remote API (text-embedding-v4) | Local GPU model (`Qwen3-Embedding-4B`), avoids rate-limiting |

---

## 3. Training Mode Switching

### 3.1 Batch Mode (CMI reward, default)

```bash
# Switch reward formula via environment variables
MEMBUILDER_REWARD_MODE=cmi           # CMI only (default)
MEMBUILDER_REWARD_MODE=membuilder_qa # Original QA judge
MEMBUILDER_REWARD_MODE=cmi_plus_qa   # alpha*CMI + (1-alpha)*QA
```

### 3.2 Agentic Mode (multi-turn CMI reward)

```bash
# 1. Prepare data (convert from existing LongMemEval, no expert trajectory needed)
python scripts/prepare_rl_data_agentic.py \
    --split train \
    --output-file ./data/longmemeval_agentic_train.parquet

# 2. Launch Agentic RL training
ROLLOUT_MODE=agentic python -m verl.trainer.main_ppo --config <your_grpo_config>.yaml
```

### 3.3 CMI Hyperparameter Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `CMI_METHOD` | `kernel_mi` | CMI estimation method: `kernel_mi` / `embedding` / `logprob` / `entropy` |
| `CMI_SHAPING` | `gaussian` | Reward shaping: `gaussian` / `threshold` |
| `CMI_WEIGHT` | `0.1` | CMI+QA combination weight (only in `cmi_plus_qa` mode) |
| `CMI_KERNEL_MI_TOP_K` | `10` | Local neighborhood top-k recall count |
| `CMI_KERNEL_MI_BANDWIDTH` | `auto` | RBF kernel bandwidth (`auto` = Silverman's rule) |
| `CMI_KERNEL_MI_EMBEDDING_MODEL` | `Qwen3-Embedding-4B` | CMI-specific embedding model path |

---

## 4. Environment Setup

### 4.1 Credentials

| Variable | Required | Description |
|---|---|---|
| `DASHSCOPE_API_KEY` | ✅ | DashScope API key for LLM and embedding services |
| `OPENAI_API_KEY` | ⭕ | Alternative API key (if using OpenAI-compatible endpoints) |

```bash
export DASHSCOPE_API_KEY="<your_dashscope_key>"
```

### 4.2 Dependencies

| Purpose | Install Command |
|---|---|
| All dependencies | `pip install -r requirements.txt` |

### 4.3 verl Framework Setup

This project depends on [verl](https://github.com/volcengine/verl) for RL training. Install it first, then apply our patches:

```bash
# 1. Install verl from source (recommended) or pip
pip install verl
# OR: git clone https://github.com/volcengine/verl.git && cd verl && pip install -e .

# 2. Apply CMI-Mem patches (required for Agentic RL support)
cd <verl_install_dir>
git apply /path/to/this/repo/patches/0001-fix-agent-loop-non-tensor-batch-serialization.patch
git apply /path/to/this/repo/patches/0002-register-trajectory-reward-manager.patch
```

**What the patches do:**
- **Patch 0001**: Fixes a serialization bug in `AgentLoopWorker` where `np.array([v])` fails for complex objects (dicts, lists). Uses `np.empty(1, dtype=object)` instead.
- **Patch 0002**: Registers `TrajectoryRewardManager` in verl's reward manager registry, enabling the `"trajectory"` reward manager name used by Agentic RL. Wrapped in try/except so batch-only users are unaffected.

---

## 5. Data Preparation

### 5.1 Batch Mode Data (standard 4-agent)

```bash
# Generate expert trajectory + parquet (for Batch mode)
export DASHSCOPE_API_KEY="<your_key>"
python scripts/generate_expert_trajectories.py --split train --expert-model deepseek-v4-pro
python scripts/prepare_rl_data_online.py --split train --output-file ./data/rl_train.parquet
```

### 5.2 Agentic Mode Data (no expert trajectory needed)

```bash
# Convert directly from raw LongMemEval JSON, no expert trajectory dependency
python scripts/prepare_rl_data_agentic.py \
    --split train \
    --output-file ./data/longmemeval_agentic_train.parquet
```

### 5.3 MemoryAgentBench RL Data

Build RL training data from the MemoryAgentBench dataset using the full expert-trajectory pipeline:

```bash
# Full run (default: 5 contexts/split, official QA)
bash scripts/build_memoryagentbench_rl_data.sh

# Smoke test (1 context, end-to-end)
DRY_RUN=1 bash scripts/build_memoryagentbench_rl_data.sh

# Tune sampling / QA source
SAMPLES_PER_SPLIT=10 QA_SOURCE=model bash scripts/build_memoryagentbench_rl_data.sh

# Only run specific stages
STAGES=traj,parquet bash scripts/build_memoryagentbench_rl_data.sh
```

Pipeline stages (`STAGES` env, comma-separated; default: all):
- `traj`: Generate expert trajectories via `generate_expert_trajectories.py --dataset memoryagentbench`
- `parquet`: Prepare RL data via `prepare_rl_data.py`
- `qa`: Inject official QA (`QA_SOURCE=official`, default) or model-generated QA (`QA_SOURCE=model`)
- `strip`: Strip expert output → `*_cmi_no_expert.parquet` (pure CMI, no expert)
- `curriculum`: Reorder by difficulty → `*_curriculum.parquet` (for GRPO curriculum learning)

Output variants:
- `*_all.parquet`: With expert_output + QA (CMI+QA / expert mode)
- `*_cmi_no_expert.parquet`: Expert stripped (pure CMI, no expert)
- `*_curriculum.parquet`: Difficulty-ordered for GRPO

### 5.4 SFT Data (optional)

```bash
# Generate expert trajectory → convert to alpaca format
python scripts/generate_expert_trajectories.py --split sft --expert-model deepseek-v4-pro ...
python scripts/convert_trajectories_to_sft.py --trajectory-dir ... --output-file memory_building_sft.json
```

---

## 6. Training

Training uses the **verl** framework with GRPO. Configure your training via YAML config files under `verl/trainer/config/`.

### 6.1 Batch RL (default)

```bash
export DASHSCOPE_API_KEY="<your_key>"

# Set reward mode
export MEMBUILDER_REWARD_MODE=cmi           # CMI only (default)
# export MEMBUILDER_REWARD_MODE=membuilder_qa  # Original QA judge
# export MEMBUILDER_REWARD_MODE=cmi_plus_qa    # alpha*CMI + (1-alpha)*QA

# Launch training with verl
python -m verl.trainer.main_ppo --config verl/trainer/config/<your_config>.yaml
```

### 6.2 Agentic RL

```bash
# Switch to Agentic RL
export ROLLOUT_MODE=agentic
python -m verl.trainer.main_ppo --config verl/trainer/config/<your_agentic_config>.yaml
```

### 6.3 SFT Training

```bash
# See training/sft/train_example.sh for a local SFT example
bash training/sft/train_example.sh
```

### 6.4 Reward Computation Modes

| Variable | Value | Description |
|---|---|---|
| `MEMBUILDER_TASK_REWARD_MODE` | `inproc` (default) | Worker process directly connects to DashScope |
| | `api` | HTTP call to local reward server |
| | `simulate` | Heuristic reward (for smoke testing) |
| `MEMBUILDER_REWARD_MODE` | `cmi` (default) | Action-Conditioned Local CMI |
| | `membuilder_qa` | Original QA judge reward |
| | `cmi_plus_qa` | alpha × CMI + (1-alpha) × QA |
| `ROLLOUT_MODE` | `batch` (default) | Standard 4-agent batch rollout |
| | `agentic` | Multi-turn agent loop |

---

## 7. Evaluation

### 7.1 LongMemEval Evaluation

```bash
# LongMemEval test split
python -m eval.runner --dataset longmemeval --split test \
    --model deepseek-v4-pro --judge-model deepseek-v4-pro

# Using a trained model (two-step: trained model builds memory → strong model answers)
bash scripts/launch_vllm_openai_server.sh /path/to/hf_model 8000 1
python -m eval.runner --mode build --provider vllm --base-url http://localhost:8000/v1 \
    --model /path/to/hf_model --dataset longmemeval --split test \
    --db-path ./faiss_data/longmemeval/my_model
python -m eval.runner --mode answer --provider openai --model deepseek-v4-pro \
    --dataset longmemeval --split test --db-path ./faiss_data/longmemeval/my_model
```

### 7.2 MemoryAgentBench Evaluation

Evaluate on the MemoryAgentBench dataset with two selectable pipelines:

**RAG Pipeline** (no vLLM needed, pure API):
```bash
python -m eval.run_memoryagentbench \
    --pipeline rag --split Accurate_Retrieval \
    --data-dir /path/to/MemoryAgentBench \
    --output-dir logs/mab_rag
```

**Memory Pipeline** (requires a chat backend such as local vLLM serving the checkpoint):
```bash
# First launch vLLM server
bash scripts/launch_vllm_openai_server.sh /path/to/hf_model 8000 1

# Then run evaluation
python -m eval.run_memoryagentbench \
    --pipeline memory --split Accurate_Retrieval \
    --provider vllm --base-url http://localhost:8000/v1 --model /path/to/hf_model \
    --data-dir /path/to/MemoryAgentBench \
    --output-dir logs/mab_memory
```

**Key arguments**:
- `--split`: Dataset split to evaluate. Accepts a single split name (`Accurate_Retrieval`, `Test_Time_Learning`, `Long_Range_Understanding`, `Conflict_Resolution`), `all`, or a `+`-separated list (e.g., `Test_Time_Learning+Long_Range_Understanding`)
- `--pipeline`: `rag` (chunk context → FAISS top-k retrieval → answer) or `memory` (build structured memory → search → answer)
- `--chunk-char-size`: Token chunk size for context → pseudo session conversion (default: 4096)
- `--top-k`: Number of retrieved chunks for RAG pipeline (default: 5)
- `--judge-model`: LLM judge model for binary accuracy evaluation
- `--answer-model`: Answer generation model (defaults to `qwen3.7-max`)
- `--splits-file`: Path to `memoryagentbench_splits.json` for context-level RL/test partition filtering
- `--worker-rank` / `--worker-count`: Multi-worker data sharding for distributed evaluation
- `--parallel` / `--workers`: Enable parallel processing with specified worker count

**Output**: Results are saved to `{output_dir}/memoryagentbench_{pipeline}_{split}_{timestamp}/` containing:
- `summary.json`: Aggregated metrics including LLM-judge accuracy and official MemoryAgentBench metrics
- `results.json`: Per-sample detailed results

The evaluation reports both:
1. **LLM-judge binary accuracy** (for cross-dataset comparison with LongMemEval/LoCoMo)
2. **MemoryAgentBench official metrics** (substring / exact_match / F1 / ROUGE / recsys recall), routed by dataset source

### 7.3 Checkpoint Conversion

```bash
# verl checkpoint → HuggingFace format
python scripts/convert_trajectories_to_sft.py --trajectory-dir checkpoints/ --output-file output.json
```

---

## 8. Repository Structure

`+` = newly added relative to original MemBuilder, `*` = modified

```
MemBuilder/
├── training/
│   ├── agentic/                       # ★ Agentic RL core module
│   │   ├── agent_loop.py              #   MemoryBuilderAgentLoop (multi-turn agent loop)
│   │   ├── dataset.py                 #   AgenticRLDataset (sessions → trajectories)
│   │   ├── prompts.py                 #   Unified prompt template (4 agents combined output)
│   │   ├── reward_manager.py          #   TrajectoryRewardManager (per-turn dense reward)
│   │   └── agent_loop_config.yaml     #   Agent loop registration config
│   │
│   ├── reward_server/                 # ★ Reward computation
│   │   ├── reward_function.py         #   Main reward entry (CMI / QA / CMI+QA mode switching)
│   │   ├── cmi_reward.py              #   Action-Conditioned Local CMI implementation
│   │   ├── cmi_config.py              #   CMI config loading
│   │   ├── cmi_reward_config.json     #   CMI default hyperparameters
│   │   ├── action_parser.py           #   Parse policy output to structured actions
│   │   ├── format_checker.py          #   Format validation
│   │   ├── attribution.py             #   Attribution weight computation
│   │   ├── constraints_checker.py     #   Constraint checker
│   │   └── start_server.sh            #   Local reward server launcher
│   │
│   ├── rl/
│   │   └── adrpo.py                   #   ADRPO algorithm implementation
│   │
│   └── sft/
│       └── train_example.sh           #   Local SFT training example
│
├── patches/                           #   verl framework patches
│   ├── 0001-fix-agent-loop-non-tensor-batch-serialization.patch
│   └── 0002-register-trajectory-reward-manager.patch
│
├── scripts/
│   ├── prepare_rl_data_agentic.py     #   Agentic data preparation (no expert trajectory)
│   ├── prepare_rl_data_online.py      #   Batch data preparation
│   ├── build_memoryagentbench_rl_data.sh  #   MemoryAgentBench RL data builder
│   ├── prepare_agentic_data.py        #   Agentic data preparation utility
│   ├── generate_expert_trajectories.py #   Expert trajectory generation
│   ├── convert_trajectories_to_sft.py #   Trajectory → SFT format conversion
│   ├── launch_vllm_openai_server.sh   #   Launch local vLLM server
│   ├── process_locomo.py              #   LoCoMo dataset processor
│   └── process_perltqa.py             #   PerlTQA dataset processor
│
├── eval/                              #   Evaluation module
│   ├── runner.py                      #   Evaluation entry point
│   ├── datasets.py                    #   Dataset loading
│   ├── llm_judge.py                   #   LLM judge
│   ├── metrics.py                     #   Metric computation
│   ├── rag_baseline.py                #   RAG baseline evaluation
│   ├── memoryagentbench_data.py       #   MemoryAgentBench dataset loader & official metrics
│   ├── run_memoryagentbench.py        #   MemoryAgentBench evaluation runner
│   ├── answering_prompt.py            #   Answering prompt templates
│   └── configs/                       #   Evaluation config files
│       ├── cmimem_qwen37max.yaml
│       └── cmimem_qwen3_4b.yaml
│
├── memory_system.py                   #   Multi-dimensional memory system (core/episodic/semantic/procedural)
├── prompts.py                         #   4-agent prompt templates
├── config.py                          #   Model constants (overridable via env vars)
├── llm_client.py                      #   DashScope / OpenAI-compatible LLM & Embedding client
├── qa_generator.py                    #   Synthetic QA generation
└── requirements.txt                   #   All dependencies
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Unknown reward manager: trajectory` | `TrajectoryRewardManager` module not imported | Auto-import added in `verl/workers/reward_manager/__init__.py` |
| Reward always returns 0 | Worker cannot access reward server | Use `MEMBUILDER_TASK_REWARD_MODE=inproc` (default) |
| `DASHSCOPE_API_KEY not set` | Missing API key | `export DASHSCOPE_API_KEY="<your_key>"` |
| Embedding rate-limit errors | Remote API batch limits | Set `EMBEDDING_PROVIDER=local` to use local GPU model |
| Context length exceeded | Input too long for model | Reduce `MEMORY_CONSTRUCTION_TOP_K` or enable truncation |

---

## 10. Citation

```bibtex
@article{membuilder2025,
  title={MemBuilder: Building Long-Term Memory with Conditional Mutual Information Rewards},
  author={...},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
