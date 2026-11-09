#!/bin/bash
# =============================================================================
# Stage 3: Scale-Wise High-Resolution Post-Training (1664x960, 2LR + 2HR)
# =============================================================================
# Usage:
#   bash scripts/train_stage3_scale_wise.sh [config_path] [extra args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    CONFIG_PATH="$1"
    shift
else
    CONFIG_PATH="configs/stage3_scale_wise_960p/scale_wise_2lr2hr_960p.yaml"
fi

DATASET_PKG="${REPO_ROOT}/packages/dataset"
export PYTHONPATH="${REPO_ROOT}/packages/ltx-distillation/src:${REPO_ROOT}/packages/ltx-causal/src:${REPO_ROOT}/packages/ltx-core/src:${REPO_ROOT}/packages/ltx-pipelines/src:${DATASET_PKG}${PYTHONPATH:+:${PYTHONPATH}}"

_DETECTED_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
NPROC_PER_NODE="${NPROC_PER_NODE:-${_DETECTED_GPUS:-8}}"
if ! [ "${NPROC_PER_NODE:-0}" -gt 0 ] 2>/dev/null; then
    NPROC_PER_NODE=8
fi
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

NUM_CPUS="${NUM_CPUS:-128}"
export OMP_NUM_THREADS=$((NUM_CPUS / NPROC_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_DISTRIBUTED_TIMEOUT_SEC="${TORCH_DISTRIBUTED_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG_TAG="$(basename "${CONFIG_PATH%.*}")"
LOG_FILE="${LOG_DIR}/${CONFIG_TAG}_${TIMESTAMP}_node${NODE_RANK}.log"

echo "========================================================"
echo "Stage 3: Scale-Wise 2LR+2HR Post-Training (1664x960)"
echo "Config:  ${CONFIG_PATH}"
echo "Nodes:   ${NNODES} | GPUs/Node: ${NPROC_PER_NODE}"
echo "Log:     ${LOG_FILE}"
echo "========================================================"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m ltx_distillation.train_stage4_scale_wise \
    --config_path "${CONFIG_PATH}" \
    "$@" 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
