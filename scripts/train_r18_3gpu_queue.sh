#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GPU_LIST="${GPU_LIST:-1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

MODELS=(
    original-r18
    anti-r18
    anti-wtconv
    anti-hifi
    anti-p2
    anti-mrfpn
)

if [[ ${#GPUS[@]} -eq 0 ]]; then
    echo "GPU_LIST must contain at least one GPU id" >&2
    exit 2
fi
for gpu in "${GPUS[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU id in GPU_LIST=${GPU_LIST}: ${gpu}" >&2
        exit 2
    fi
done

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/runs/anti-detr/logs/${RUN_ID}}"
mkdir -p "${LOG_DIR}"

echo "GPU queue: ${GPUS[*]}"
echo "Model order: ${MODELS[*]}"
echo "Logs: ${LOG_DIR}"
echo "Each experiment keeps the single-GPU batch/LR profile."

# Dispatch models in ordered waves. Up to one single-GPU experiment runs on
# each GPU, preserving the original per-experiment optimization settings while
# reducing total suite wall time.
for ((start = 0; start < ${#MODELS[@]}; start += ${#GPUS[@]})); do
    pids=()
    labels=()
    wave=$((start / ${#GPUS[@]} + 1))
    echo "[$(date '+%F %T')] starting wave ${wave}"

    for ((slot = 0; slot < ${#GPUS[@]} && start + slot < ${#MODELS[@]}; slot++)); do
        model="${MODELS[start + slot]}"
        gpu="${GPUS[slot]}"
        log_file="${LOG_DIR}/${model}_gpu${gpu}.log"
        echo "  dispatch ${model} -> GPU ${gpu} (${log_file})"
        DEVICE="${gpu}" bash "${SCRIPT_DIR}/train_r18_one.sh" "${model}" "$@" >"${log_file}" 2>&1 &
        pids+=("$!")
        labels+=("${model}@GPU${gpu}")
    done

    failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[i]}"; then
            echo "[$(date '+%F %T')] completed ${labels[i]}"
        else
            status=$?
            echo "[$(date '+%F %T')] FAILED ${labels[i]} (exit ${status})" >&2
            failed=1
        fi
    done
    if [[ ${failed} -ne 0 ]]; then
        echo "Wave ${wave} failed; inspect ${LOG_DIR}. Later models were not started." >&2
        exit 1
    fi
done

echo "[$(date '+%F %T')] all R18 experiments completed"
