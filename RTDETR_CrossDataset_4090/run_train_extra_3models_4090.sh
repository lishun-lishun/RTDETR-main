#!/usr/bin/env bash
# Multi-GPU Batch Launcher for Exp1_BaseHRNet, Exp31_Exp28_InnerNWD, and Exp32_Exp28Plus_Full
# across Det-Fly, TIB-Net, and NPS-Drones on RTX 4090 server.
# On a 6-GPU server: Wave 1 runs Det-Fly (GPUs 0-2) + TIB-Net (GPUs 3-5); Wave 2 runs NPS-Drones (GPUs 0-2).
# Automatically adapts wave size if fewer/more GPUs are detected via `nvidia-smi -L`.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

EPOCHS=${EPOCHS:-150}
NPS_EPOCHS=${NPS_EPOCHS:-150}
BATCH=${BATCH:-8}
SEED=${SEED:-0}

if [ ! -f "dataset/Det-Fly.yaml" ] || [ ! -f "dataset/TIB-Net.yaml" ] || [ ! -f "dataset/NPS-Drones.yaml" ]; then
  echo ">>> Missing standardized dataset YAML(s). Running prepare_datasets_4090.py first..."
  python RTDETR_CrossDataset_4090/prepare_datasets_4090.py --repo . --seed "$SEED"
fi

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -lt 1 ]; then
  NUM_GPUS=1
fi
echo ">>> Detected ${NUM_GPUS} GPU(s) on server. Scheduling 9 extra runs in waves of ${NUM_GPUS}..."

TASKS=(
  "Det-Fly Exp1_BaseHRNet ${EPOCHS} log_detfly_exp1.txt"
  "Det-Fly Exp31_Exp28_InnerNWD ${EPOCHS} log_detfly_exp31.txt"
  "Det-Fly Exp32_Exp28Plus_Full ${EPOCHS} log_detfly_exp32.txt"
  "TIB-Net Exp1_BaseHRNet ${EPOCHS} log_tib_exp1.txt"
  "TIB-Net Exp31_Exp28_InnerNWD ${EPOCHS} log_tib_exp31.txt"
  "TIB-Net Exp32_Exp28Plus_Full ${EPOCHS} log_tib_exp32.txt"
  "NPS-Drones Exp1_BaseHRNet ${NPS_EPOCHS} log_nps_exp1.txt"
  "NPS-Drones Exp31_Exp28_InnerNWD ${NPS_EPOCHS} log_nps_exp31.txt"
  "NPS-Drones Exp32_Exp28Plus_Full ${NPS_EPOCHS} log_nps_exp32.txt"
)

IDX=0
for ENTRY in "${TASKS[@]}"; do
  read -r DS MDL EP LOG <<< "$ENTRY"
  GPU_ID=$((IDX % NUM_GPUS))
  echo ">>> [GPU ${GPU_ID}] Starting ${DS} / ${MDL} (epochs=${EP}) -> ${LOG}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python RTDETR_CrossDataset_4090/train_cross_extra_4090.py \
    --dataset "$DS" --model "$MDL" \
    --device 0 --epochs "$EP" --batch "$BATCH" --seed "$SEED" --execute > "$LOG" 2>&1 &

  IDX=$((IDX + 1))
  if [ $((IDX % NUM_GPUS)) -eq 0 ]; then
    wait
  fi
done
wait

echo ">>> All 9 extra cross-dataset runs completed! Run bash RTDETR_CrossDataset_4090/run_eval_cross_4090.sh next."
