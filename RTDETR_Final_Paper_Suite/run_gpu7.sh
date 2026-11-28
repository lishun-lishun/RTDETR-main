#!/usr/bin/env bash
# GPU 7 Sequential 2-Task Launcher (Final Paper Suite)
#   Task 1: Det-Fly / Final_Ours_Full
#   Task 2: TIB-Net / Final_Ours_Full
# Usage: bash RTDETR_Final_Paper_Suite/run_gpu7.sh [--resume]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-0}"
LOG_DIR="$REPO_ROOT/final_model/_logs"
mkdir -p "$LOG_DIR"

run_one() {
  local ds="$1"
  local mdl="$2"
  shift 2
  local log_file="$LOG_DIR/${ds}_${mdl}.log"
  echo "================================================================================"
  echo "[GPU 7] START: $ds / $mdl (epochs=$EPOCHS, batch=$BATCH, seed=$SEED)"
  echo "[GPU 7] LOG:   $log_file"
  echo "================================================================================"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=7 \
    python -u RTDETR_Final_Paper_Suite/train_final_suite.py \
      --dataset "$ds" \
      --model "$mdl" \
      --epochs "$EPOCHS" \
      --batch "$BATCH" \
      --micro-batch "$BATCH" \
      --workers "$WORKERS" \
      --seed "$SEED" \
      --device 0 \
      --execute "$@" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    echo "[GPU 7] [OK]   Finished $ds / $mdl"
  else
    echo "[GPU 7] [FAIL] $ds / $mdl exited with code $rc (see $log_file)"
  fi
  return "$rc"
}

RC1=0
RC2=0
run_one "Det-Fly" "Final_Ours_Full" "$@" || RC1=$?
run_one "TIB-Net" "Final_Ours_Full" "$@" || RC2=$?

echo "================================================================================"
echo "[GPU 7] SUMMARY: Det-Fly/Final_Ours_Full (rc=$RC1) | TIB-Net/Final_Ours_Full (rc=$RC2)"
echo "================================================================================"
if [ "$RC1" -ne 0 ] || [ "$RC2" -ne 0 ]; then
  exit 1
fi
