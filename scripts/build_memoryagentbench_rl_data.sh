#!/usr/bin/env bash
# =============================================================================
# Build RL training data from MemoryAgentBench (full expert-trajectory pipeline)
# =============================================================================
#
# Mirrors scripts/build_rl_data_s_cleaned.sh but sources data from
# MemoryAgentBench (long-document context -> 4096-token pseudo sessions),
# samples a small subset per split, and runs the SAME 4-agent expert pipeline.
#
# Pipeline stages (STAGES env, comma-separated; default: all):
#   traj       generate_expert_trajectories.py --dataset memoryagentbench
#   parquet    prepare_rl_data.py            (NO --add-qa by default)
#   qa         inject official QA            (QA_SOURCE=official, default)
#                or prepare_rl_data --add-qa (QA_SOURCE=model)
#   strip      strip_expert_from_parquet.py  -> *_cmi_no_expert.parquet
#   curriculum reorder_rl_data_curriculum.py -> *_curriculum.parquet
#
# The strip + curriculum stages produce the "other modes" the user asked for:
#   - *_all.parquet            : with expert_output + QA  (CMI+QA / expert)
#   - *_cmi_no_expert.parquet  : expert stripped          (pure CMI, no expert)
#   - *_curriculum.parquet     : difficulty-ordered for GRPO
#
# Usage:
#   # full run (default: 5 contexts/split, official QA)
#   bash scripts/build_memoryagentbench_rl_data.sh
#
#   # smoke test (1 context, end-to-end)
#   DRY_RUN=1 bash scripts/build_memoryagentbench_rl_data.sh
#
#   # tune sampling / QA source
#   SAMPLES_PER_SPLIT=10 QA_SOURCE=model bash scripts/build_memoryagentbench_rl_data.sh
#
#   # only run specific stages
#   STAGES=traj,parquet bash scripts/build_memoryagentbench_rl_data.sh
# =============================================================================
set -euo pipefail

# ---------- arg parsing (flags) ----------
DRY_RUN=${DRY_RUN:-0}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '1,46p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown flag: $1 (use --dry-run or --help)" >&2; exit 2 ;;
    esac
done

# ---------- python interpreter selection ----------
# 系统 /usr/bin/python3 通常是 3.6，缺现代依赖。优先 PYTHON -> conda -> fallback。
if [[ -n "${PYTHON:-}" ]]; then
    PY="${PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PY="${CONDA_PREFIX}/bin/python"
elif [[ -x /opt/conda/envs/python3.10/bin/python ]]; then
    PY=/opt/conda/envs/python3.10/bin/python
elif command -v python3 >/dev/null; then
    PY=$(command -v python3)
else
    echo "[ERROR] No python3 interpreter found. Install conda or set PYTHON=." >&2
    exit 1
fi
echo "[setup] Using python: ${PY} ($(${PY} --version 2>&1))"

# ---------- dependency precheck (skipped for speed) ----------
echo "[setup] Skipping package check (disabled)."

# ---------- input / output paths ----------
# MemoryAgentBench dataset root directory.
MAB_DATA_DIR=${MAB_DATA_DIR:-./data/MemoryAgentBench}
OUTPUT_DIR=${OUTPUT_DIR:-./data/MemoryAgentBench}
TRAJ_DIR=${TRAJ_DIR:-${OUTPUT_DIR}/expert_trajectories_mab}

# ---------- sampling config ----------
SAMPLES_PER_SPLIT=${SAMPLES_PER_SPLIT:-5}
SAMPLE_SEED=${SAMPLE_SEED:-42}
CHUNK_TOKENS=${CHUNK_TOKENS:-4096}
# Splits file: when set, overrides random sampling with explicit context list.
SPLITS_FILE=${SPLITS_FILE:-data/memoryagentbench_splits/memoryagentbench_splits.json}
SPLITS_KEY=${SPLITS_KEY:-rl}

# ---------- chat/QA model config ----------
EXPERT_MODEL=${EXPERT_MODEL:-qwen3.7-max}
QA_GEN_MODEL=${QA_GEN_MODEL:-qwen3.7-max}
QA_PER_SESSION=${QA_PER_SESSION:-5}
# QA_SOURCE: official (inject dataset QA, default) | model (re-generate via expert)
QA_SOURCE=${QA_SOURCE:-official}

# ---------- curriculum config ----------
CURRICULUM_BATCH_SIZE=${CURRICULUM_BATCH_SIZE:-32}
CURRICULUM_NUM_TIERS=${CURRICULUM_NUM_TIERS:-3}

# ---------- DashScope OpenAI-compatible endpoint ----------
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-${DASHSCOPE_API_KEY:?must set DASHSCOPE_API_KEY or OPENAI_API_KEY}}

# ---------- embedding config (read from configs/embedding.yaml) ----------
# Single source of truth: configs/embedding.yaml (env > yaml precedence).
# Reused verbatim from build_rl_data_s_cleaned.sh.
EMBED_CONFIG=${EMBED_CONFIG:-configs/embedding.yaml}
if [[ ! -f "${EMBED_CONFIG}" ]]; then
    echo "[ERROR] embedding config not found: ${EMBED_CONFIG}" >&2
    exit 1
fi
echo "[setup] reading embedding config: ${EMBED_CONFIG}"

EMBED_ENV_LINES=$(${PY} - "${EMBED_CONFIG}" <<'PYEOF'
import os, sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}

def emit(k, v):
    if v is None or v == "":
        return
    if os.environ.get(k):
        return
    s = str(v).replace("'", "'\\''")
    print(f"export {k}='{s}'")

provider = (cfg.get("provider") or "remote").strip().lower()
emit("EMBEDDING_PROVIDER_RESOLVED", provider)

remote = cfg.get("remote") or {}
local = cfg.get("local") or {}
workers = cfg.get("workers") or {}

if provider == "local":
    emit("EMBEDDING_PROVIDER", "local")
    emit("LOCAL_EMBEDDING_MODEL", local.get("model"))
    emit("LOCAL_EMBEDDING_DEVICE", local.get("device"))
    emit("LOCAL_EMBEDDING_BATCH_SIZE", local.get("batch_size"))
    emit("LOCAL_EMBEDDING_CACHE_DIR", local.get("cache_dir"))
    auto_dl = local.get("auto_download")
    if auto_dl is not None:
        emit("LOCAL_EMBEDDING_AUTO_DOWNLOAD", "1" if auto_dl else "0")
    emit("WORKERS", workers.get("local", 1))
else:
    emit("EMBED_MODEL", remote.get("model"))
    emit("OPENAI_EMBEDDINGS_BATCH_SIZE", remote.get("batch_size"))
    emit("WORKERS", workers.get("remote", 4))
PYEOF
)
eval "${EMBED_ENV_LINES}"

# Defaults for anything neither in env nor yaml.
EMBED_MODEL=${EMBED_MODEL:-text-embedding-v4}
WORKERS=${WORKERS:-24}

if [[ "${EMBEDDING_PROVIDER:-}" == "local" ]]; then
    echo "[setup] embeddings: LOCAL  model=${LOCAL_EMBEDDING_MODEL}" \
         "device=${LOCAL_EMBEDDING_DEVICE:-auto} batch=${LOCAL_EMBEDDING_BATCH_SIZE:-64}"
else
    echo "[setup] embeddings: REMOTE model=${EMBED_MODEL}" \
         "(batch<=${OPENAI_EMBEDDINGS_BATCH_SIZE:-10})"
fi

# Remote embedding service discovery (align with train_pipeline.sh):
# if OPENAI_EMBEDDINGS_BASE_URL is unset, try the OSS-published service address.
if [[ -z "${OPENAI_EMBEDDINGS_BASE_URL:-}" && "${EMBEDDING_PROVIDER:-}" != "local" ]]; then
    DISCOVERY_DIR="${DISCOVERY_DIR:-./services}"
    DISCOVERY_TXT="${DISCOVERY_DIR}/embedding_service_address.txt"
    DISCOVERY_JSON="${DISCOVERY_DIR}/embedding_service_address.json"
    if [[ -f "${DISCOVERY_TXT}" ]]; then
        OPENAI_EMBEDDINGS_BASE_URL=$(cat "${DISCOVERY_TXT}" | tr -d '[:space:]')
        echo "[setup] embed service: ${OPENAI_EMBEDDINGS_BASE_URL} (auto-discovered from ${DISCOVERY_TXT})"
    elif [[ -f "${DISCOVERY_JSON}" ]]; then
        OPENAI_EMBEDDINGS_BASE_URL=$(${PY} -c "import json; d=json.load(open('${DISCOVERY_JSON}')); print(d.get('base_url','') or d.get('url','') or d.get('address',''))")
        echo "[setup] embed service: ${OPENAI_EMBEDDINGS_BASE_URL} (auto-discovered from ${DISCOVERY_JSON})"
    fi
fi
export OPENAI_EMBEDDINGS_BASE_URL=${OPENAI_EMBEDDINGS_BASE_URL:-${OPENAI_BASE_URL}}
export OPENAI_EMBEDDINGS_API_KEY=${OPENAI_EMBEDDINGS_API_KEY:-${OPENAI_API_KEY}}

# ---------- override config.py defaults via env vars ----------
export SFT_EXPERT_MODEL="${EXPERT_MODEL}"
export QA_GENERATION_MODEL="${QA_GEN_MODEL}"
export EMBEDDING_MODEL="${EMBED_MODEL}"

# ---------- stages selection ----------
STAGES=${STAGES:-traj,parquet,qa,strip,curriculum}
# Submit-side may replace ',' with '+' to survive nebulactl --env parsing.
STAGES="${STAGES//+/,}"

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "=================================================================="
echo "[config] MemoryAgentBench RL data build"
echo "=================================================================="
echo "  data dir          : ${MAB_DATA_DIR}"
echo "  output dir        : ${OUTPUT_DIR}"
echo "  traj dir          : ${TRAJ_DIR}"
echo "  splits file       : ${SPLITS_FILE:-<none>} (key=${SPLITS_KEY})"
echo "  samples/split     : ${SAMPLES_PER_SPLIT} (seed=${SAMPLE_SEED}, fallback if no splits file)"
echo "  chunk tokens      : ${CHUNK_TOKENS}"
echo "  expert model      : ${EXPERT_MODEL}"
echo "  QA source         : ${QA_SOURCE} (qa/session=${QA_PER_SESSION})"
echo "  workers           : ${WORKERS}"
echo "  stages            : ${STAGES}"
echo "=================================================================="

# =============================================================================
# DRY-RUN: 1 context end-to-end smoke test.
# =============================================================================
if [[ "${DRY_RUN}" == "1" ]]; then
    DRY_TRAJ_DIR=${TRAJ_DIR}_dryrun
    DRY_QA_FILE=${OUTPUT_DIR}/memoryagentbench_official_qa_dryrun.json
    DRY_ALL=${OUTPUT_DIR}/memoryagentbench_rl_all_dryrun.parquet
    DRY_QA_OUT=${OUTPUT_DIR}/memoryagentbench_rl_with_qa_dryrun.parquet
    DRY_CMI=${OUTPUT_DIR}/memoryagentbench_rl_cmi_no_expert_dryrun.parquet
    DRY_CURR=${OUTPUT_DIR}/memoryagentbench_rl_curriculum_dryrun.parquet

    echo ""
    echo "=================================================================="
    echo "[DRY-RUN] 1 context/split is too many; forcing samples-per-split=1"
    echo "=================================================================="

    echo "[DRY-RUN] === Stage traj ==="
    ${PY} scripts/generate_expert_trajectories.py \
        --dataset memoryagentbench \
        --mab-data-dir "${MAB_DATA_DIR}" \
        --samples-per-split 1 \
        --sample-seed "${SAMPLE_SEED}" \
        --chunk-tokens "${CHUNK_TOKENS}" \
        --expert-model "${EXPERT_MODEL}" \
        --provider openai \
        --output-dir "${DRY_TRAJ_DIR}" \
        --qa-out-file "${DRY_QA_FILE}" \
        --parallel --workers "${WORKERS}" \
        --skip-existing

    echo "[DRY-RUN] === Stage parquet ==="
    ${PY} scripts/prepare_rl_data.py \
        --trajectories-dir "${DRY_TRAJ_DIR}/memoryagentbench" \
        --output-file "${DRY_ALL}"

    echo "[DRY-RUN] === Stage qa (${QA_SOURCE}) ==="
    if [[ "${QA_SOURCE}" == "official" ]]; then
        ${PY} scripts/inject_official_qa_into_parquet.py \
            --input "${DRY_ALL}" \
            --qa-file "${DRY_QA_FILE}" \
            --output "${DRY_QA_OUT}" \
            --qa-per-context "${QA_PER_SESSION}"
    else
        ${PY} scripts/prepare_rl_data.py \
            --trajectories-dir "${DRY_TRAJ_DIR}/memoryagentbench" \
            --output-file "${DRY_QA_OUT}" \
            --add-qa --qa-per-session "${QA_PER_SESSION}"
    fi

    echo "[DRY-RUN] === Stage strip ==="
    ${PY} scripts/strip_expert_from_parquet.py \
        --input "${DRY_QA_OUT}" \
        --output "${DRY_CMI}"

    echo "[DRY-RUN] === Stage curriculum ==="
    ${PY} scripts/reorder_rl_data_curriculum.py \
        --input "${DRY_CMI}" \
        --output "${DRY_CURR}" \
        --batch-size "${CURRICULUM_BATCH_SIZE}" \
        --num-tiers "${CURRICULUM_NUM_TIERS}"

    echo ""
    echo "[DRY-RUN done] artifacts:"
    ls -lh "${DRY_QA_FILE}" "${DRY_ALL}" "${DRY_QA_OUT}" "${DRY_CMI}" "${DRY_CURR}" 2>/dev/null || true
    echo ""
    echo "Inspect schema:"
    echo "  ${PY} -c \"import pandas as pd; df=pd.read_parquet('${DRY_CURR}'); print(df.columns.tolist()); print(df.iloc[0])\""
    exit 0
fi

# =============================================================================
# File paths for the full run.
# =============================================================================
QA_FILE=${OUTPUT_DIR}/memoryagentbench_official_qa.json
ALL_PARQUET=${OUTPUT_DIR}/memoryagentbench_rl_all.parquet
WITH_QA_PARQUET=${OUTPUT_DIR}/memoryagentbench_rl_with_qa.parquet
CMI_PARQUET=${OUTPUT_DIR}/memoryagentbench_rl_cmi_no_expert.parquet
CURRICULUM_PARQUET=${OUTPUT_DIR}/memoryagentbench_rl_curriculum.parquet

# =============================================================================
# Stage 1: expert trajectories (4-agent) on sampled MemoryAgentBench contexts
# =============================================================================
if [[ ",${STAGES}," == *,traj,* ]]; then
    echo ""
    echo "=================================================================="
    echo "[traj] Generate expert trajectories"
    echo "=================================================================="
    TRAJ_CMD="${PY} scripts/generate_expert_trajectories.py \
        --dataset memoryagentbench \
        --mab-data-dir ${MAB_DATA_DIR} \
        --chunk-tokens ${CHUNK_TOKENS} \
        --expert-model ${EXPERT_MODEL} \
        --provider openai \
        --output-dir ${TRAJ_DIR} \
        --qa-out-file ${QA_FILE} \
        --parallel --workers ${WORKERS} \
        --skip-existing"
    if [[ -n "${SPLITS_FILE}" && -f "${SPLITS_FILE}" ]]; then
        echo "  using splits file: ${SPLITS_FILE} (key=${SPLITS_KEY})"
        TRAJ_CMD="${TRAJ_CMD} --context-ids-file ${SPLITS_FILE} --context-ids-key ${SPLITS_KEY}"
    else
        echo "  using random sampling: ${SAMPLES_PER_SPLIT}/split (seed=${SAMPLE_SEED})"
        TRAJ_CMD="${TRAJ_CMD} --samples-per-split ${SAMPLES_PER_SPLIT} --sample-seed ${SAMPLE_SEED}"
    fi
    eval ${TRAJ_CMD}
fi

# =============================================================================
# Stage 2: prepare_rl_data -> *_all.parquet (NO --add-qa by default)
# =============================================================================
if [[ ",${STAGES}," == *,parquet,* ]]; then
    echo ""
    echo "=================================================================="
    echo "[parquet] Convert trajectories -> ${ALL_PARQUET}"
    echo "=================================================================="
    if [[ "${QA_SOURCE}" == "model" ]]; then
        # When using model-generated QA, fold it in at this stage.
        ${PY} scripts/prepare_rl_data.py \
            --trajectories-dir "${TRAJ_DIR}/memoryagentbench" \
            --output-file "${ALL_PARQUET}" \
            --add-qa --qa-per-session "${QA_PER_SESSION}"
    else
        ${PY} scripts/prepare_rl_data.py \
            --trajectories-dir "${TRAJ_DIR}/memoryagentbench" \
            --output-file "${ALL_PARQUET}"
    fi
fi

# =============================================================================
# Stage 3: QA injection (official) -> *_with_qa.parquet
#   official: inject dataset QA into the no-QA parquet
#   model   : QA already folded in Stage 2; just alias all -> with_qa
# =============================================================================
if [[ ",${STAGES}," == *,qa,* ]]; then
    echo ""
    echo "=================================================================="
    echo "[qa] QA source = ${QA_SOURCE}"
    echo "=================================================================="
    if [[ "${QA_SOURCE}" == "official" ]]; then
        ${PY} scripts/inject_official_qa_into_parquet.py \
            --input "${ALL_PARQUET}" \
            --qa-file "${QA_FILE}" \
            --output "${WITH_QA_PARQUET}" \
            --qa-per-context "${QA_PER_SESSION}"
    else
        # model QA already in ALL_PARQUET; copy bytes via python (OSS-safe).
        SRC_PATH="${ALL_PARQUET}" DST_PATH="${WITH_QA_PARQUET}" ${PY} - <<'PYEOF'
import os
src = os.environ["SRC_PATH"]; dst = os.environ["DST_PATH"]
with open(src, "rb") as f:
    data = f.read()
with open(dst, "wb") as f:
    f.write(data); f.flush(); os.fsync(f.fileno())
print(f"[qa] model-QA parquet copied -> {dst} ({len(data)} bytes)")
PYEOF
    fi
fi

# =============================================================================
# Stage 4: strip expert_output -> *_cmi_no_expert.parquet (pure CMI mode)
# =============================================================================
if [[ ",${STAGES}," == *,strip,* ]]; then
    echo ""
    echo "=================================================================="
    echo "[strip] Strip expert_output -> ${CMI_PARQUET}"
    echo "=================================================================="
    ${PY} scripts/strip_expert_from_parquet.py \
        --input "${WITH_QA_PARQUET}" \
        --output "${CMI_PARQUET}"
fi

# =============================================================================
# Stage 5: curriculum reorder -> *_curriculum.parquet
# =============================================================================
if [[ ",${STAGES}," == *,curriculum,* ]]; then
    echo ""
    echo "=================================================================="
    echo "[curriculum] Reorder by difficulty -> ${CURRICULUM_PARQUET}"
    echo "=================================================================="
    ${PY} scripts/reorder_rl_data_curriculum.py \
        --input "${CMI_PARQUET}" \
        --output "${CURRICULUM_PARQUET}" \
        --batch-size "${CURRICULUM_BATCH_SIZE}" \
        --num-tiers "${CURRICULUM_NUM_TIERS}"
fi

echo ""
echo "=================================================================="
echo "[done] Final parquet files:"
echo "=================================================================="
ls -lh "${OUTPUT_DIR}"/memoryagentbench_rl_*.parquet 2>/dev/null || true
echo ""
echo "RL training data ready. Mode mapping:"
echo "  CMI+QA / expert  -> ${WITH_QA_PARQUET}"
echo "  pure CMI         -> ${CMI_PARQUET}"
echo "  GRPO curriculum  -> ${CURRICULUM_PARQUET}"
