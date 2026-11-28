#!/usr/bin/env bash
# 8-GPU Parallel Dynamic Scheduler for Final Paper Suite (4 Datasets x 4 Models = 16 Runs)
# Outputs strictly to: final_model/<Dataset>/<Model>/
# Uses `python -u` (unbuffered logs), `OMP_NUM_THREADS=4`, and tracks real exit codes!
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-0}"
LOG_DIR="$REPO_ROOT/final_model/_logs"
mkdir -p "$LOG_DIR"

# 16 jobs ordered heaviest-first so 8 GPUs finish balanced!
JOBS=(
  "NPS-Drones Final_Ours_Full"
  "NPS-Drones Imp1_Backbone_HRNetStarFast"
  "NPS-Drones Imp2_Neck_HiLoHaar"
  "NPS-Drones Imp3_Loss_InnerNWD"
  "Det-Fly Final_Ours_Full"
  "Det-Fly Imp1_Backbone_HRNetStarFast"
  "Det-Fly Imp2_Neck_HiLoHaar"
  "Det-Fly Imp3_Loss_InnerNWD"
  "DUT-Anti-UAV Final_Ours_Full"
  "DUT-Anti-UAV Imp1_Backbone_HRNetStarFast"
  "DUT-Anti-UAV Imp2_Neck_HiLoHaar"
  "DUT-Anti-UAV Imp3_Loss_InnerNWD"
  "TIB-Net Final_Ours_Full"
  "TIB-Net Imp1_Backbone_HRNetStarFast"
  "TIB-Net Imp2_Neck_HiLoHaar"
  "TIB-Net Imp3_Loss_InnerNWD"
)

echo ">>> Launching ${#JOBS[@]} Final Paper Suite runs across 8 GPUs (GPU 0-7)..."
echo ">>> Hyperparams: EPOCHS=$EPOCHS | BATCH=$BATCH | WORKERS=$WORKERS | SEED=$SEED"

STATUS_FILE="$LOG_DIR/run_status.txt"
rm -f "$STATUS_FILE"

# Split 16 jobs across 8 GPU workers (2 jobs per GPU: 1 heavy + 1 light!)
for gpu in 0 1 2 3 4 5 6 7; do
  (
    for idx in $gpu $((gpu + 8)); do
      if [ "$idx" -lt "${#JOBS[@]}" ]; then
        read -r ds model <<< "${JOBS[$idx]}"
        log_file="$LOG_DIR/${ds}_${model}.log"
        echo "[GPU $gpu] START: $ds / $model -> $log_file"
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu \
          python -u RTDETR_Final_Paper_Suite/train_final_suite.py \
            --dataset "$ds" \
            --model "$model" \
            --epochs "$EPOCHS" \
            --batch "$BATCH" \
            --workers "$WORKERS" \
            --seed "$SEED" \
            --execute > "$log_file" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
          echo "[OK]   GPU=$gpu | $ds | $model" | tee -a "$STATUS_FILE"
        else
          echo "[FAIL] GPU=$gpu | $ds | $model (exit=$rc, see $log_file)" | tee -a "$STATUS_FILE"
        fi
      fi
    done
  ) &
done
wait

echo ""
echo "================ Final Paper Suite Run Status ================"
sort "$STATUS_FILE"
if grep -q "^\[FAIL\]" "$STATUS_FILE"; then
  echo "[WARNING] Some runs failed! Check logs in $LOG_DIR and re-run with --resume."
  exit 1
else
  echo "[ALL 16 RUNS SUCCEEDED] Models saved under: $REPO_ROOT/final_model/"
fi
