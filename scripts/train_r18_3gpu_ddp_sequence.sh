#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Three GPUs jointly train one model at a time. Global batch 12 gives each GPU
# the same local batch 4 used by the single-GPU profile. Warmup is divided by
# three so it covers approximately the same number of input images.
DEVICE="${DEVICE:-1,2,3}"
BATCH="${BATCH:-12}"
LR0="${LR0:-0.0001}"
WARMUP_ITERS="${WARMUP_ITERS:-667}"

export DEVICE

echo "Sequential DDP: devices=${DEVICE}, global_batch=${BATCH}, lr0=${LR0}, warmup_iters=${WARMUP_ITERS}"
echo "For the closest comparison with single-GPU runs, prefer train_r18_3gpu_queue.sh."

bash "${SCRIPT_DIR}/train_r18_sequence.sh" \
    --batch "${BATCH}" \
    --lr0 "${LR0}" \
    --warmup-iters "${WARMUP_ITERS}" \
    "$@"
