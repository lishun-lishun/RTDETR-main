#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Three GPUs jointly train one model at a time. Global batch 24 gives each GPU
# the paper train batch of 8. This changes the optimizer's global batch from 8
# to 24; use the GPU queue for a direct paper-batch comparison.
DEVICE="${DEVICE:-1,2,3}"
BATCH="${BATCH:-24}"
LR0="${LR0:-0.0001}"
WARMUP_ITERS="${WARMUP_ITERS:-667}"

export DEVICE
# torchrun executes a temporary Python file outside the repository. Put this
# checkout first so every rank imports the integrated local Ultralytics code,
# not a separately installed package from site-packages.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Sequential DDP: devices=${DEVICE}, global_batch=${BATCH}, lr0=${LR0}, warmup_iters=${WARMUP_ITERS}"
echo "Local source: ${PROJECT_ROOT}"
echo "For global train batch 8, use train_r18_3gpu_queue.sh (one model per GPU)."

bash "${SCRIPT_DIR}/train_r18_sequence.sh" \
    --batch "${BATCH}" \
    --lr0 "${LR0}" \
    --warmup-iters "${WARMUP_ITERS}" \
    "$@"
