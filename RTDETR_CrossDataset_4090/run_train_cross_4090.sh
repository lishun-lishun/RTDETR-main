#!/usr/bin/env bash
# GPU-Count-Adaptive Cross-Dataset Training Launcher on 4090 Server
# Trains 3 core models per dataset across TIB-Net, Det-Fly, and NPS-Drones:
#   1) RT-DETR-R18                (Baseline, pretrained=false matching e0_baseline)
#   2) Exp27_StarFast_NativeRepC3 (Best Fast Backbone: HRNetStarFast)
#   3) Exp28_StarFast_HiLoHaar    (Best Backbone + Neck Champion)
# Auto-detects available GPUs via `nvidia-smi -L` and dispatches in waves!
# Usage: bash RTDETR_CrossDataset_4090/run_train_cross_4090.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-8}
SEED=${SEED:-0}
R18_WEIGHTS=${R18_WEIGHTS:-""}

echo ">>> Step 1: Standardizing TIB-Net, Det-Fly, NPS-Drones..."
python RTDETR_CrossDataset_4090/prepare_datasets_4090.py --repo . --seed "$SEED"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -lt 1 ]; then
  NUM_GPUS=1
fi
echo ">>> Detected ${NUM_GPUS} GPU(s) on server. Scheduling 9 runs in waves of ${NUM_GPUS}..."

TASKS=(
  "TIB-Net RT-DETR-R18 log_tib_r18.txt"
  "TIB-Net Exp27_StarFast_NativeRepC3 log_tib_exp27.txt"
  "TIB-Net Exp28_StarFast_HiLoHaar log_tib_exp28.txt"
  "Det-Fly RT-DETR-R18 log_detfly_r18.txt"
  "Det-Fly Exp27_StarFast_NativeRepC3 log_detfly_exp27.txt"
  "Det-Fly Exp28_StarFast_HiLoHaar log_detfly_exp28.txt"
  "NPS-Drones RT-DETR-R18 log_nps_r18.txt"
  "NPS-Drones Exp27_StarFast_NativeRepC3 log_nps_exp27.txt"
  "NPS-Drones Exp28_StarFast_HiLoHaar log_nps_exp28.txt"
)

IDX=0
for ENTRY in "${TASKS[@]}"; do
  read -r DS MDL LOG <<< "$ENTRY"
  GPU_ID=$((IDX % NUM_GPUS))
  EXTRA=()
  if [ "$MDL" = "RT-DETR-R18" ] && [ -n "$R18_WEIGHTS" ]; then
    EXTRA=(--r18-weights "$R18_WEIGHTS")
  fi
  echo ">>> [GPU ${GPU_ID}] Starting ${DS} / ${MDL} -> ${LOG}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} python RTDETR_CrossDataset_4090/train_cross_4090.py \
    --dataset "$DS" --model "$MDL" "${EXTRA[@]}" \
    --device 0 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" --execute > "$LOG" 2>&1 &

  IDX=$((IDX + 1))
  if [ $((IDX % NUM_GPUS)) -eq 0 ]; then
    wait
  fi
done
wait

echo ">>> All 9 cross-dataset training runs completed! Run bash RTDETR_CrossDataset_4090/run_eval_cross_4090.sh next."
