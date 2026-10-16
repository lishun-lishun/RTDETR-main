#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Three GPUs jointly train one model at a time. Batch 24 is an intentional
# large-batch variant of the paper's batch 8, not an exact reproduction.
DEVICE="${DEVICE:-1,2,3}"
BATCH="${BATCH:-24}"
MICRO_BATCH="${MICRO_BATCH:-12}"
LR0="${LR0:-0.0001}"
LRF="${LRF:-0.1}"
WARMUP_ITERS="${WARMUP_ITERS:-667}"

IFS=',' read -r -a GPUS <<< "${DEVICE}"
if [[ ${#GPUS[@]} -ne 3 || ! "${GPUS[0]}" =~ ^[0-9]+$ || ! "${GPUS[1]}" =~ ^[0-9]+$ || ! "${GPUS[2]}" =~ ^[0-9]+$ || "${GPUS[0]}" == "${GPUS[1]}" || "${GPUS[0]}" == "${GPUS[2]}" || "${GPUS[1]}" == "${GPUS[2]}" ]]; then
    echo "DEVICE must contain three distinct physical GPU IDs, e.g. DEVICE=1,2,3" >&2
    exit 2
fi
if [[ ! "${BATCH}" =~ ^[0-9]+$ || ! "${MICRO_BATCH}" =~ ^[0-9]+$ || ${BATCH} -lt 1 || ${MICRO_BATCH} -lt 1 || $((BATCH % MICRO_BATCH)) -ne 0 || $((MICRO_BATCH % 3)) -ne 0 ]]; then
    echo "BATCH must be divisible by MICRO_BATCH, and MICRO_BATCH must be divisible by 3" >&2
    exit 2
fi

export DEVICE
export CUDA_VISIBLE_DEVICES="${DEVICE}"
export PYTHONUNBUFFERED=1
# torchrun executes a temporary Python file outside the repository. Put this
# checkout first so every rank imports the integrated local Ultralytics code,
# not a separately installed package from site-packages.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/runs/anti-detr/logs/ddp_${RUN_ID}}"
mkdir -p "${LOG_DIR}"

MODELS=(original-r18 anti-r18 anti-wtconv anti-hifi anti-p2 anti-mrfpn)
echo "Sequential DDP: devices=${DEVICE}, effective_batch=${BATCH}, micro_batch=${MICRO_BATCH}, accumulation=$((BATCH / MICRO_BATCH)), lr0=${LR0}, cosine_lrf=${LRF}, warmup_updates=${WARMUP_ITERS}"
echo "Model order: ${MODELS[*]}"
echo "Local source: ${PROJECT_ROOT}"
echo "Logs: ${LOG_DIR} (also streamed to this terminal)"
echo "Paper train batch is 8; this is a batch-24 tuning preset, not an exact paper reproduction."

for model in "${MODELS[@]}"; do
    log_file="${LOG_DIR}/${model}.log"
    echo "[$(date '+%F %T')] starting ${model} on GPUs ${DEVICE}; log: ${log_file}"
    if bash "${SCRIPT_DIR}/train_r18_one.sh" "${model}" \
        --batch "${BATCH}" \
        --micro-batch "${MICRO_BATCH}" \
        --lr0 "${LR0}" \
        --cos-lr \
        --lrf "${LRF}" \
        --warmup-bias-lr 0.0 \
        --warmup-iters "${WARMUP_ITERS}" \
        "$@" 2>&1 | tee -a "${log_file}"; then
        echo "[$(date '+%F %T')] completed ${model}"
    else
        echo "[$(date '+%F %T')] FAILED ${model}; later experiments were not started. See ${log_file}" >&2
        exit 1
    fi
done

echo "[$(date '+%F %T')] all R18 experiments completed"
