#!/usr/bin/env bash
# Nine R18-family DUT-Anti-UAV experiments, three at a time on separate GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_LIST="${GPU_LIST:-1,2,3}"
MICRO_BATCH="${MICRO_BATCH:-8}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

# Ordered priority: baseline; small-object scale and fusion; isolated encoder,
# backbone and upsampling changes; then a joint backbone+neck change.
CONFIGS=(
    rtdetr-r18.yaml
    rtdetr-p2.yaml
    rtdetr-RFPN.yaml
    rtdetr-AIFI-DHSA.yaml
    rtdetr-RFAConv.yaml
    rtdetr-DySample.yaml
    rtdetr-CARAFE.yaml
    rtdetr-WDBB.yaml
    rtdetr-WTConv.yaml
)

if [[ ${#GPUS[@]} -ne 3 ]]; then
    echo "GPU_LIST must contain exactly three GPU IDs, e.g. GPU_LIST=1,2,3" >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ || -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
        echo "GPU_LIST must contain three distinct non-negative GPU IDs: ${GPU_LIST}" >&2
        exit 2
    fi
    SEEN_GPUS["${gpu}"]=1
done
if [[ ! "${MICRO_BATCH}" =~ ^[1-9][0-9]*$ ]] || (( 8 % MICRO_BATCH != 0 )); then
    echo "MICRO_BATCH must be a positive divisor of the effective batch 8 (1, 2, 4 or 8)." >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
echo "Preflighting nine model configs and DUT training settings..."
"${PYTHON_BIN}" train_dut_anti_uav.py --mode train \
    --rt-detr-config "${CONFIGS[@]}" --device "${GPUS[0]}" \
    --batch 8 --micro-batch "${MICRO_BATCH}" "$@" --dry-run

echo "GPU order: ${GPUS[*]}"
echo "Experiment order: ${CONFIGS[*]}"
echo "Three ordered waves; each GPU runs one single-GPU experiment per wave."
echo "Per experiment: effective batch 8, physical micro-batch ${MICRO_BATCH}, paper-profile settings."
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: no training processes started."
    exit 0
fi

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/runs/rtdetr-dut/logs/${RUN_ID}}"
mkdir -p "${LOG_DIR}"
echo "Logs: ${LOG_DIR}"
echo "Each log line is also streamed to this terminal with its experiment/GPU prefix."

for ((start = 0; start < ${#CONFIGS[@]}; start += 3)); do
    wave=$((start / 3 + 1))
    pids=()
    labels=()
    log_files=()
    echo "[$(date '+%F %T')] Starting wave ${wave}/3"

    for ((slot = 0; slot < 3; slot++)); do
        config="${CONFIGS[start + slot]}"
        gpu="${GPUS[slot]}"
        label="${config%.yaml}@GPU${gpu}"
        log_file="${LOG_DIR}/${config%.yaml}_gpu${gpu}.log"
        echo "  ${label} -> ${log_file}"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export PYTHONUNBUFFERED=1
            "${PYTHON_BIN}" -u train_dut_anti_uav.py --mode train \
                --rt-detr-config "${config}" --device "${gpu}" \
                --batch 8 --micro-batch "${MICRO_BATCH}" "$@" 2>&1 \
                | sed -u "s/^/[${label}] /" | tee "${log_file}"
        ) &
        pids+=("$!")
        labels+=("${label}")
        log_files+=("${log_file}")
    done

    failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[i]}"; then
            echo "[$(date '+%F %T')] Completed ${labels[i]}"
        else
            status=$?
            echo "[$(date '+%F %T')] FAILED ${labels[i]} (exit ${status})" >&2
            echo "----- last 60 lines: ${log_files[i]} -----" >&2
            tail -n 60 "${log_files[i]}" >&2
            failed=1
        fi
    done
    if (( failed )); then
        echo "Wave ${wave} failed. Later waves were not started; inspect ${LOG_DIR}." >&2
        exit 1
    fi
done

echo "[$(date '+%F %T')] All nine experiments completed."
