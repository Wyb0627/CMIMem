#!/bin/bash
# =============================================================================
# CMI-Mem End-to-End Training Pipeline (Open-Source Reference)
#
# Run an SFT (optional) -> GRPO RL training loop for CMI-Mem locally on a single
# node or via a multi-node Ray launcher. All paths and credentials are exposed
# as environment variables so you can wire this script into any cluster scheduler.
#
# Usage:
#   bash scripts/run_training.sh [sft|train|all|help]
#
# Required environment variables:
#   DATA_DIR              Root dir for datasets (parquet/json files). Expected:
#                           ${DATA_DIR}/longmemeval/longmemeval_4agent_rl.parquet
#                           ${DATA_DIR}/longmemeval/memory_building_sft.json
#   MODEL_DIR             Base HF/ModelScope model dir or repo id.
#                         Default: Qwen/Qwen3-4B-Instruct-2507
#   OUTPUT_DIR            Where checkpoints, metrics, logs are written.
#                         Default: ./outputs
#   EMBEDDING_MODEL_PATH  Local sentence-transformer path or repo id used by
#                         the embedding service. Default: ./models/embedding
#
# Optional environment variables (selected; see below for the full list):
#   ENABLE_SFT=0|1, ENABLE_RL=0|1     Pipeline stage switches
#   WORLD_SIZE, N_GPUS                 Distributed sizing
#   ROLLOUT_MODE=batch|agentic         Rollout strategy (default: batch)
#   TRAIN_CONFIG=<name>                Hydra config under configs/ (default: grpo_training)
#   MEMBUILDER_REWARD_MODE             cmi (default) | cmi_plus_qa | qa | membuilder_qa
#   CMI_METHOD                         embedding (default) | logprob | kernel_mi |
#                                       residual_projection | entropy
#   CMI_SHAPING                        gaussian (default) | threshold
#   CMI_WEIGHT                         alpha for cmi_plus_qa (default: 0.5)
#   EMBEDDING_PROVIDER                 local (default) | remote
#   ANSWER_API_BASE_URL / ANSWER_API_KEY / ANSWER_MODEL / JUDGE_MODEL
#                                       OpenAI-compatible LLM API for QA reward
#
# Notes:
#   - This is the reference open-source pipeline; cluster submission, OSS
#     upload/download and proprietary credentials have been intentionally
#     removed. Adapt the `run_training` function to your scheduler if needed.
#   - For multi-node Ray, set WORLD_SIZE>1 and RAY_ADDRESS=<host:port>.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ==================== Path defaults ==========================================
DATA_DIR="${DATA_DIR:-./data}"
MODEL_DIR="${MODEL_DIR:-Qwen/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-./models/embedding}"
mkdir -p "${OUTPUT_DIR}"

# ==================== Pipeline switches ======================================
ENABLE_SFT="${ENABLE_SFT:-0}"
ENABLE_RL="${ENABLE_RL:-1}"

# ==================== Distributed config =====================================
WORLD_SIZE="${WORLD_SIZE:-1}"
N_GPUS="${N_GPUS:-8}"
JOB_NAME="${JOB_NAME:-cmi_mem_grpo}"

# ==================== SFT config (used only when ENABLE_SFT=1) ===============
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${MODEL_DIR}}"
SFT_DATA="${SFT_DATA:-${DATA_DIR}/longmemeval/memory_building_sft.json}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${OUTPUT_DIR}/sft}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-1}"
SFT_LR="${SFT_LR:-1.0e-5}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-32}"
SFT_MAX_LEN="${SFT_MAX_LEN:-32768}"

# ==================== RL config ==============================================
ROLLOUT_MODE="${ROLLOUT_MODE:-batch}"
# Hydra config name. We ship a single generic GRPO config (configs/grpo_training.yaml).
# Override via TRAIN_CONFIG=<name> if you maintain additional configs (e.g. an
# `agentic_training` variant for multi-turn agent rollouts).
TRAIN_CONFIG="${TRAIN_CONFIG:-grpo_training}"
CONFIG_DIR="${CONFIG_DIR:-${REPO_ROOT}/configs}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/longmemeval/longmemeval_4agent_rl.parquet}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-${OUTPUT_DIR}/rl/${JOB_NAME}}"
mkdir -p "${RL_OUTPUT_DIR}"

# ==================== Reward / CMI config ====================================
MEMBUILDER_REWARD_MODE="${MEMBUILDER_REWARD_MODE:-cmi}"
MEMBUILDER_TASK_REWARD_MODE="${MEMBUILDER_TASK_REWARD_MODE:-inproc}"
CMI_METHOD="${CMI_METHOD:-embedding}"
CMI_SHAPING="${CMI_SHAPING:-gaussian}"
CMI_WEIGHT="${CMI_WEIGHT:-0.5}"
REWARD_MAX_CONCURRENT="${REWARD_MAX_CONCURRENT:-16}"
CMI_KERNEL_MI_EMBEDDING_MODEL="${CMI_KERNEL_MI_EMBEDDING_MODEL:-${EMBEDDING_MODEL_PATH}}"

# ==================== LLM API config (QA reward / answer / judge) ============
ANSWER_API_BASE_URL="${ANSWER_API_BASE_URL:-https://api.openai.com/v1}"
ANSWER_API_KEY="${ANSWER_API_KEY:-}"
ANSWER_MODEL="${ANSWER_MODEL:-gpt-4o-mini}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
# LLM retry / timeout knobs (raise under heavy rate-limiting).
LLM_RETRY_MAX_ATTEMPTS="${LLM_RETRY_MAX_ATTEMPTS:-12}"
LLM_CLIENT_TIMEOUT="${LLM_CLIENT_TIMEOUT:-600}"

# ==================== Embedding config =======================================
EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-local}"
OPENAI_EMBEDDINGS_BASE_URL="${OPENAI_EMBEDDINGS_BASE_URL:-}"
OPENAI_EMBEDDINGS_API_KEY="${OPENAI_EMBEDDINGS_API_KEY:-EMPTY}"
LOCAL_EMBEDDING_DEVICE="${LOCAL_EMBEDDING_DEVICE:-cuda}"
LOCAL_EMBEDDING_BATCH_SIZE="${LOCAL_EMBEDDING_BATCH_SIZE:-64}"

# ==================== Compatibility shims ====================================
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export UVLOOP_NO_IMPORT="${UVLOOP_NO_IMPORT:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

# Forward all knobs to the trainer process.
export DATA_DIR MODEL_DIR OUTPUT_DIR EMBEDDING_MODEL_PATH N_GPUS WORLD_SIZE
export EMBEDDING_PROVIDER OPENAI_EMBEDDINGS_BASE_URL OPENAI_EMBEDDINGS_API_KEY \
       LOCAL_EMBEDDING_DEVICE LOCAL_EMBEDDING_BATCH_SIZE
export MEMBUILDER_REWARD_MODE MEMBUILDER_TASK_REWARD_MODE REWARD_MAX_CONCURRENT \
       CMI_METHOD CMI_SHAPING CMI_WEIGHT CMI_KERNEL_MI_EMBEDDING_MODEL
export ANSWER_API_BASE_URL ANSWER_API_KEY ANSWER_MODEL JUDGE_MODEL \
       LLM_RETRY_MAX_ATTEMPTS LLM_CLIENT_TIMEOUT

# =============================================================================
# Stage 1: Supervised Fine-Tuning (optional)
# =============================================================================
run_sft() {
    if [ "${ENABLE_SFT}" != "1" ]; then
        echo "[INFO] ENABLE_SFT=${ENABLE_SFT}, skipping SFT stage."
        return 0
    fi

    echo "============================================================"
    echo "[SFT] Supervised fine-tuning"
    echo "  base model      : ${SFT_MODEL_PATH}"
    echo "  training data   : ${SFT_DATA}"
    echo "  output dir      : ${SFT_OUTPUT_DIR}"
    echo "  num epochs      : ${SFT_NUM_EPOCHS}"
    echo "  global batch    : ${SFT_BATCH_SIZE}"
    echo "  learning rate   : ${SFT_LR}"
    echo "============================================================"
    mkdir -p "${SFT_OUTPUT_DIR}"

    # Use torchrun for single-node multi-GPU SFT. Adapt for your scheduler if
    # running on multiple nodes.
    torchrun \
        --standalone \
        --nproc_per_node="${N_GPUS}" \
        training/sft/train_sft.py \
        --model_name_or_path "${SFT_MODEL_PATH}" \
        --train_file "${SFT_DATA}" \
        --output_dir "${SFT_OUTPUT_DIR}" \
        --num_train_epochs "${SFT_NUM_EPOCHS}" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps "$(( SFT_BATCH_SIZE / N_GPUS ))" \
        --learning_rate "${SFT_LR}" \
        --max_seq_length "${SFT_MAX_LEN}" \
        --bf16 true \
        --logging_steps 10 \
        --save_strategy epoch
}

# =============================================================================
# Stage 2: GRPO Reinforcement Learning
# =============================================================================
run_training() {
    if [ "${ENABLE_RL}" != "1" ]; then
        echo "[INFO] ENABLE_RL=${ENABLE_RL}, skipping RL stage."
        return 0
    fi

    echo "============================================================"
    echo "[RL] GRPO training (verl)"
    echo "  hydra config    : ${TRAIN_CONFIG}"
    echo "  job name        : ${JOB_NAME}"
    echo "  rollout mode    : ${ROLLOUT_MODE}"
    echo "  reward mode     : ${MEMBUILDER_REWARD_MODE}"
    echo "  cmi method      : ${CMI_METHOD}"
    echo "  base model      : ${MODEL_DIR}"
    echo "  train data      : ${TRAIN_DATA}"
    echo "  output dir      : ${RL_OUTPUT_DIR}"
    echo "  world size      : ${WORLD_SIZE}"
    echo "  gpus / node     : ${N_GPUS}"
    echo "============================================================"

    DATE_TIME="$(date +%Y%m%d_%H%M)"
    METRICS_LOG_DIR="${RL_OUTPUT_DIR}/${DATE_TIME}"
    METRICS_LOG_PATH="${METRICS_LOG_DIR}/metrics.jsonl"
    mkdir -p "${METRICS_LOG_DIR}"
    export VERL_FILE_LOGGER_PATH="${METRICS_LOG_PATH}"
    echo "[INFO] metrics will be appended to ${METRICS_LOG_PATH}"

    # Clear stale Python bytecode caches before launch.
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true

    # Hydra overrides resolve paths from env vars at parse time; configs in
    # verl/trainer/config/*.yaml may reference these via ${oc.env:...}.
    HYDRA_OVERRIDES=(
        "actor_rollout_ref.model.path=${MODEL_DIR}"
        "data.train_files=${TRAIN_DATA}"
        "trainer.default_local_dir=${RL_OUTPUT_DIR}"
        "trainer.experiment_name=${JOB_NAME}_${ROLLOUT_MODE}_${MEMBUILDER_REWARD_MODE}"
    )

    if [ "${WORLD_SIZE}" -gt 1 ]; then
        # Multi-node: assumes a Ray cluster is already up. Set RAY_ADDRESS to
        # the head node, e.g. RAY_ADDRESS=10.0.0.1:6379.
        : "${RAY_ADDRESS:?RAY_ADDRESS must be set when WORLD_SIZE>1}"
        export RAY_ADDRESS
    fi

    python3 -m verl.trainer.main_ppo \
        --config-path "${CONFIG_DIR}" \
        --config-name "${TRAIN_CONFIG}" \
        "${HYDRA_OVERRIDES[@]}"

    echo ""
    echo "[OK] training finished. checkpoints under ${RL_OUTPUT_DIR}"
}

# =============================================================================
# Entrypoint
# =============================================================================
COMMAND="${1:-help}"
case "${COMMAND}" in
    sft)   run_sft ;;
    train) run_training ;;
    all)   run_sft; run_training ;;
    help|*)
        # Print the leading docstring (first banner block).
        sed -n '2,42p' "${BASH_SOURCE[0]}"
        exit 0
        ;;
esac
