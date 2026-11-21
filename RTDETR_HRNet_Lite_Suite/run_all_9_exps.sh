#!/usr/bin/env bash
# Dual-GPU Sequential Batch Runner for All 9 HRNet-Lite Experiments on NVIDIA A30
# Usage: bash RTDETR_HRNet_Lite_Suite/run_all_9_exps.sh
set -e

EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-8}
SEED=${SEED:-0}

echo "Launching GPU 0 queue (Exp1, Exp2, Exp4, Exp6, Exp8) and GPU 1 queue (Exp3, Exp5, Exp7, Exp9)..."

(
  for EXP in Exp1_BaseHRNet Exp2_PruneS4 Exp4_SlimNeck Exp6_StarHRNet Exp8_RepGhostHaar; do
    echo ">>> [GPU 0] Starting $EXP ..."
    CUDA_VISIBLE_DEVICES=0 python RTDETR_HRNet_Lite_Suite/train_hrnet_suite.py \
      --exp "$EXP" --device 0 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" --execute
  done
) > runs_gpu0_hrnet_suite.log 2>&1 &

(
  for EXP in Exp3_PConvRep Exp5_FastHRFull Exp7_Lite192Dec Exp9_UltraLite; do
    echo ">>> [GPU 1] Starting $EXP ..."
    CUDA_VISIBLE_DEVICES=1 python RTDETR_HRNet_Lite_Suite/train_hrnet_suite.py \
      --exp "$EXP" --device 0 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" --execute
  done
) > runs_gpu1_hrnet_suite.log 2>&1 &

wait
echo "All 9 experiments completed! Logs saved to runs_gpu0_hrnet_suite.log and runs_gpu1_hrnet_suite.log."
