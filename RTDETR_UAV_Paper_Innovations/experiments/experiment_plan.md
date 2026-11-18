# 第八阶段：公平实验设计与多轮消融矩阵方案（Experiment Plan）

## 1. 训练前公平性核查清单（Fairness Pre-Flight Checklist）

鉴于历史消融实验中出现过“E0 仅跑 124 轮、TinyV2 仅跑 68–123 轮、V1 跑 150 轮”以及“验证集 `0.65–0.66` 与历史测试集 `0.707` 混比”的教训，本实验方案强制锁定以下 **8 项绝对一致协议**：

| 协议维度 | 统一锁定参数 | 科学依据与防踩坑要求 |
| :--- | :--- | :--- |
| **Dataset Split** | `DUT-Anti-UAV` 官方划分（Train: 5200 / Val: 2600 / Test: 2200 张共 2245 实例） | 严禁跨 Val 与 Test 混比；模型选择仅看 Val，最终论文主表统一报告 Test（2200 张）。 |
| **Input Size** | `640 × 640` | 训练、验证、测试与速度评测全程锁定 `imgsz=640`。 |
| **Optimizer & LR** | `AdamW(lr0=1e-4, lrf=0.1, weight_decay=1e-4, warmup_epochs=3.0, cos_lr=True)` | 与 RT-DETR 官方及 Ultralytics 最佳基线超参数严格一致。 |
| **Batch Size** | `batch = 16`（单卡 NVIDIA A30 24GB AMP 模式） | 保持 BN 统计量与每轮迭代步数（$5200 / 16 = 325\text{ steps/epoch}$）完全一致。 |
| **Training Epochs** | **`epochs = 150, patience = 0`** | **强制设置 `patience=0` 关闭早停**！确保所有模型完整经历前 135 轮 Mosaic 强增强与最后 15 轮（`close_mosaic=15`）原始尺度精细收敛。 |
| **Initialization** | 统一从相同的 `rtdetr-r18.pt` 预训练主干初始化；所有新增模块采用零初始化/保守门控 | 保证第 0 轮起点一致，不破坏 PResNet18-vd 预训练特征提取能力。 |
| **Evaluation Script** | 统一调用 `experiments/evaluate.py`（结合 Ultralytics `RTDETRValidator` 与 `pycocotools`） | 同时输出标准 COCO 尺度指标与 AI-TOD 自定义 Tiny 指标，绝不混淆定义。 |
| **Random Seeds** | 初筛使用 `seed = 0`；核心基线 `E0` 与最优模型 `E6, E7` 使用 `seed ∈ {0, 1, 2}` 复测 | 报告 `Mean ± Std`，验证改进具备统计显著性而非随机种子偶然波动。 |

---

## 2. Small 与 Tiny 目标的面积定义规范（杜绝口径混淆）

在 `640 × 640` 标准评测分辨率下，我们在 `experiments/evaluate.py` 中明确区分并同时报告以下两套面积定义：
1. **COCO 标准像素面积定义（主表必报）**：
   - `COCO Small AP (AP_s)`：边界框像素面积 $\text{Area} < 32^2 = 1,024\text{ px}^2$（归一化面积 $< 0.0025$）；
   - `COCO Medium AP (AP_m)`：$32^2 \le \text{Area} < 96^2\text{ px}^2$（$1,024 \le \text{Area} < 9,216\text{ px}^2$）；
   - `COCO Large AP (AP_l)`：$\text{Area} \ge 96^2 = 9,216\text{ px}^2$。
2. **自定义 Tiny / Very-Tiny 像素面积定义（遵循 AI-TOD 国际标准，专门衡量极端远距离无人机）**：
   - `Custom Very-Tiny AP (AP_vt)`：$\text{Area} < 8^2 = 64\text{ px}^2$；
   - `Custom Tiny AP (AP_t)`：$\text{Area} < 16^2 = 256\text{ px}^2$。

---

## 3. 两轮递进式消融实验矩阵（E0–E7，未训练状态声明）

> **科研诚信声明**：当前实验矩阵中的模型处于**代码与集成验证已通过、尚未在 NVIDIA A30 上跑满 150 Epoch 训练**阶段，因此所有精度指标均严格标记为 **`未训练 (Untrained)`**，绝不编造任何预测 mAP 数字。参数量（Params）与计算量（GFLOPs）为通过 `experiments/benchmark.py` 实测的真实数值。

| 实验编号 | 实验配置名称 | WCA-SFP (Inno 1) | FAGA (Inno 2) | UAD-CR (Inno 3) | SAH-NWD & CW-LSD (Inno 4) | 配置文件路径 | 实测 Params (M) | 实测 GFLOPs | 训练状态 / mAP50:95 / AP75 / AP_small |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **E0** | Original RT-DETR-R18 | - | - | - | - | `E0_baseline_rtdetr_r18.yaml` | 20.08M (Base) | 56.9G (Base) | **未训练**（待在统一 150 Epoch 协议下复测） |
| **E1** | + WCA-SFP (浅层小波对比度桥) | ✓ | - | - | - | `E1_inno1_wcasfp.yaml` | +0.088M | +0.74G | **未训练** |
| **E2** | + FAGA (频空门控与亚像素相位对齐) | - | ✓ | - | - | `E2_inno2_faga.yaml` | +0.105M | +0.52G | **未训练** |
| **E3** | + UAD-CR (不确定性选点与方差保底解码) | - | - | ✓ | - | `E3_inno3_uadcr.yaml` | +0.398M | +0.02G | **未训练** |
| **E4** | + SAH-NWD & CW-LSD (尺度自适应匹配与自蒸馏) | - | - | - | ✓ | `E4_inno4_sahnwd.yaml` | **+0.000M** | **+0.00G** | **未训练** |
| **E5** | Combo A: 特征与颈部协同 (Inno 1 + Inno 2) | ✓ | ✓ | - | - | `E5_combo_A_feature_neck.yaml` | +0.193M | +1.26G | **未训练** |
| **E6** | Combo B: 解码与监督协同 (Inno 3 + Inno 4) | - | - | ✓ | ✓ | `E6_combo_B_decoder_loss.yaml` | +0.398M | +0.02G | **未训练** |
| **E7** | Final Joint Model (全部四项创新联合) | ✓ | ✓ | ✓ | ✓ | `E7_final_joint_model.yaml` | +0.591M | +1.28G | **未训练**（含 Seed 0/1/2 三种子复测计划） |

---

## 4. 服务器端（NVIDIA A30 24GB）一键执行命令清单

```bash
# 1. 安装四个创新模块与配置到 RTDETR-main
python integration/install.py --repo /path/to/RTDETR-main

# 2. 第一轮独立模块消融训练（E0–E4，统一 150 Epochs）
for EXP in E0 E1 E2 E3 E4; do
    python experiments/train.py --exp $EXP --data dut-anti-uav.yaml --weights weights/rtdetr-r18.pt --epochs 150 --batch 16 --device 0 --seed 0
done

# 3. 第二轮组合与最终模型消融训练（E5–E7）
for EXP in E5 E6 E7; do
    python experiments/train.py --exp $EXP --data dut-anti-uav.yaml --weights weights/rtdetr-r18.pt --epochs 150 --batch 16 --device 0 --seed 0
done

# 4. 核心模型 3 随机种子显著性复测（E0 与 E7，Seed 1 & 2）
for SEED in 1 2; do
    python experiments/train.py --exp E0 --data dut-anti-uav.yaml --weights weights/rtdetr-r18.pt --epochs 150 --batch 16 --device 0 --seed $SEED
    python experiments/train.py --exp E7 --data dut-anti-uav.yaml --weights weights/rtdetr-r18.pt --epochs 150 --batch 16 --device 0 --seed $SEED
done

# 5. 统一在测试集（2200 张图像）上评估并输出 COCO Small + Custom Tiny AP
python experiments/evaluate.py --weights runs/uav_paper_ablation/E7_seed0/weights/best.pt --data dut-anti-uav.yaml --split test --device 0 --output-json runs/uav_paper_ablation/E7_test_metrics.json

# 6. 统一在 NVIDIA A30 上运行 FP16 速度基准测试（50 次预热 + 500 次正式计时）
python experiments/benchmark.py --warmup 50 --iters 500 --imgsz 640 --fp16 --output runs/uav_paper_ablation/a30_speed_benchmark.json
```
