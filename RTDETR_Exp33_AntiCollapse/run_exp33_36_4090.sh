#!/usr/bin/env bash
set -e
# 4-GPU parallel ablation launcher for Exp33-Exp36 on DUT-Anti-UAV (RTX 4090)
# Adjust DATA_YAML if your DUT-Anti-UAV yaml path differs!
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

DATA_YAML="${DATA_YAML:-/tmp/dut_anti_uav_4090.yaml}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-8}"
SEED="${SEED:-0}"

if [ ! -f "$DATA_YAML" ]; then
cat << 'EOF' > /tmp/dut_anti_uav_4090.yaml
path: /home/aigc/disk1/LiShun/muchuan/DUT-Anti-UAV
train: images/train
val: images/val
test: images/test
names:
  0: drone
EOF
DATA_YAML="/tmp/dut_anti_uav_4090.yaml"
fi

echo ">>> Launching Exp33-Exp36 in parallel on GPUs 0-3 (DATA=$DATA_YAML, EPOCHS=$EPOCHS, BATCH=$BATCH, SEED=$SEED)..."
CUDA_VISIBLE_DEVICES=0 python RTDETR_Exp33_AntiCollapse/train_exp33.py --exp Exp33_FCGHaar_Only --data "$DATA_YAML" --epochs "$EPOCHS" --batch "$BATCH" --micro-batch "$BATCH" --seed "$SEED" --device 0 --execute > log_exp33_fcghaar.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python RTDETR_Exp33_AntiCollapse/train_exp33.py --exp Exp34_AntiCollapseDec_Only --data "$DATA_YAML" --epochs "$EPOCHS" --batch "$BATCH" --micro-batch "$BATCH" --seed "$SEED" --device 0 --execute > log_exp34_anticollapse.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python RTDETR_Exp33_AntiCollapse/train_exp33.py --exp Exp35_TightCoupledNWD_Only --data "$DATA_YAML" --epochs "$EPOCHS" --batch "$BATCH" --micro-batch "$BATCH" --seed "$SEED" --device 0 --execute > log_exp35_tightnwd.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python RTDETR_Exp33_AntiCollapse/train_exp33.py --exp Exp36_Full_FCG_AntiCollapse_Coupled --data "$DATA_YAML" --epochs "$EPOCHS" --batch "$BATCH" --micro-batch "$BATCH" --seed "$SEED" --device 0 --execute > log_exp36_full.txt 2>&1 &
wait
echo ">>> All 4 experiments (Exp33-Exp36) finished!"
