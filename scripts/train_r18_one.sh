#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/train_r18_one.sh MODEL [extra train_dut_anti_uav.py arguments]" >&2
    echo "Models: original-r18 anti-r18 anti-wtconv anti-hifi anti-mrfpn anti-p2" >&2
    exit 2
fi

MODEL="$1"
shift

case "${MODEL}" in
    original-r18|anti-r18|anti-wtconv|anti-hifi|anti-mrfpn|anti-p2) ;;
    *)
        echo "Unknown R18 experiment: ${MODEL}" >&2
        exit 2
        ;;
esac

cd "${PROJECT_ROOT}"

# Bind physical GPU IDs before Python (and PyTorch) starts. Selecting a device
# inside the trainer can be too late if CUDA was initialized while constructing
# the model, causing concurrent queue jobs to share the wrong GPU.
if [[ "${DEVICE}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    export CUDA_VISIBLE_DEVICES="${DEVICE}"
fi

ARGS=(
    --mode train
    --models "${MODEL}"
    --device "${DEVICE}"
)

# INIT_WEIGHTS is deliberately optional because the author repository does not
# publish the paper's initialization checkpoint. A concrete path is required;
# a boolean `pretrained=true` does not load weights for YAML-created models in
# this RTDETR-main branch.
if [[ -n "${INIT_WEIGHTS:-}" ]]; then
    if [[ "${DEVICE}" == *,* ]]; then
        echo "INIT_WEIGHTS is not supported by the generated DDP launcher; use the GPU queue mode" >&2
        echo "(one GPU per experiment) when checkpoint initialization is required." >&2
        exit 2
    fi
    ARGS+=(--weights "${INIT_WEIGHTS}")
fi

echo "[$(date '+%F %T')] starting ${MODEL} on device ${DEVICE} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
"${PYTHON_BIN}" train_dut_anti_uav.py "${ARGS[@]}" "$@"
echo "[$(date '+%F %T')] completed ${MODEL}"
