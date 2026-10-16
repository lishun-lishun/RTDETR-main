#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Three GPUs jointly train one model at a time. Effective global batch 24 uses
# two micro-batches of 12, keeping each A30 at four images per forward pass.
DEVICE="${DEVICE:-1,2,3}"
BATCH="${BATCH:-24}"
MICRO_BATCH="${MICRO_BATCH:-12}"
LR0="${LR0:-0.0001}"
WARMUP_ITERS="${WARMUP_ITERS:-667}"

export DEVICE
export CUDA_VISIBLE_DEVICES="${DEVICE}"
# torchrun executes a temporary Python file outside the repository. Put this
# checkout first so every rank imports the integrated local Ultralytics code,
# not a separately installed package from site-packages.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Sequential DDP: devices=${DEVICE}, effective_batch=${BATCH}, micro_batch=${MICRO_BATCH}, lr0=${LR0}, warmup_updates=${WARMUP_ITERS}"
echo "Local source: ${PROJECT_ROOT}"
echo "For global train batch 8, use train_r18_3gpu_queue.sh (one model per GPU)."

bash "${SCRIPT_DIR}/train_r18_sequence.sh" \
    --batch "${BATCH}" \
    --micro-batch "${MICRO_BATCH}" \
    --lr0 "${LR0}" \
    --warmup-iters "${WARMUP_ITERS}" \
    "$@"
