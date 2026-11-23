#!/usr/bin/env bash
# 6-GPU Parallel Launcher for Cross-Dataset Generalization (Det-Fly, NPS-Drones, TIB-Net)
# Compares RT-DETR-R18 vs Exp12_StarHR_Balanced with 100% identical hyperparameters.
# Verified against `runs/uav_ablation/e0_baseline/seed0/args.yaml`: baseline RT-DETR-R18
# used `pretrained: false` (YAML from scratch). Optional `R18_WEIGHTS` can be set if needed.
set -e

EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-8}
SEED=${SEED:-0}
DATA_ROOT=${DATA_ROOT:-/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav}
R18_WEIGHTS=${R18_WEIGHTS:-""}

R18_EXTRA=()
if [ -n "$R18_WEIGHTS" ]; then
  R18_EXTRA=(--r18-weights "$R18_WEIGHTS")
fi

echo ">>> Step 1: Preparing standardized YOLO datasets & YAMLs from $DATA_ROOT ..."
python RTDETR_CrossDataset_Suite/prepare_cross_datasets.py --repo . --data-root "$DATA_ROOT" --seed "$SEED"

echo ">>> Step 2: Launching 6 parallel GPU runs (GPUs 0-5)..."
CUDA_VISIBLE_DEVICES=0 python RTDETR_CrossDataset_Suite/train_cross_dataset.py --dataset Det-Fly --model RT-DETR-R18 "${R18_EXTRA[@]}" --device 0 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" --execute > log_detfly_r18.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python RTDETR_CrossDataset_Suite/train_cross_dataset.py --dataset Det-Fly --model Exp12_StarHR_Balanced --device 0 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" --execute > log_detfly_exp12.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python RTDETR_CrossDataset_Suite/train_cross_dataset.py --dataset NPS-Drones --model RT-DETR-R18 "${R18_EXTRA[@]}" --device 0 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" --execute > log_nps_r18.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python RTDETR_CrossDataset_Suite/train_cross_dataset.py --dataset NPS-Drones --model Exp12_StarHR_Balanced --device 0 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" --execute > log_nps_exp12.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 python RTDETR_CrossDataset_Suite/train_cross_dataset.py --dataset TIB-Net --model RT-DETR-R18 "${R18_EXTRA[@]}" --device 0 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" --execute > log_tib_r18.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 python RTDETR_CrossDataset_Suite/train_cross_dataset.py --dataset TIB-Net --model Exp12_StarHR_Balanced --device 0 --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" --execute > log_tib_exp12.txt 2>&1 &

wait
echo ">>> Step 3: Running unified full-metric evaluation on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python RTDETR_CrossDataset_Suite/eval_cross_dataset.py --device 0 --seed "$SEED"
