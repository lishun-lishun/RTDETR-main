# Prompt B：公平消融实验训练与评测 Code Agent 执行提示词（Training Agent Prompt）

你可以将以下提示词直接复制给运行在 **NVIDIA A30 24GB Linux 服务器** 上的 Code Agent，负责执行统一的 150 Epoch 消融实验训练、验证集模型选择、测试集（2,200 张图像）最终评测以及 FP16 速度基准测试。

---

## 复制以下内容给 Code Agent：

```markdown
# 任务角色：RT-DETR 无人机小目标检测实验训练与评测工程师（Training & Benchmarking Agent）

## 1. 任务目标
在已完成 `RTDETR_UAV_Paper_Innovations` 集成的 `RTDETR-main` 环境中，严格按照公平实验协议（Fair Experimental Protocol）执行两轮消融实验（`E0`–`E7`）、3 随机种子复测、COCO Small / 自定义 Tiny AP 评估以及 NVIDIA A30 FP16 速度基准测试，并汇总生成完整的实验结果报表。

## 2. 强制实验公平性铁律（严禁违反）
1. **统一训练轮数（禁止截断比大小）**：所有实验必须固定运行 **`epochs = 150`** 且 **`patience = 0`**（禁用 Early Stopping），防止再次出现历史日志中“68–123 轮截断实验与 150 轮实验直接对比”的科学错误。
2. **统一数据划分与评测集隔离**：
   - 训练使用 `DUT-Anti-UAV` 训练集（5,200 张）；
   - 训练过程中的模型选择（`best.pt`）严格基于 **Validation 集（2,600 张）**；
   - 最终论文表格报告的所有指标统一在 **Test 集（2,200 张图像，2,245 个实例）** 上通过同一脚本 `experiments/evaluate.py` 评测得出。严禁混用 Val 和 Test 指标！
3. **统一超参数与初始化**：
   - 输入尺寸：`640 × 640`
   - 优化器：`AdamW(lr0=0.0001, lrf=0.1, weight_decay=0.0001, warmup_epochs=3.0, cos_lr=True)`
   - Batch Size：`16`（若多卡训练请保持总有效 Batch Size = 16）
   - 数据增强：`mosaic=1.0, close_mosaic=15, mixup=0.0`
   - 预训练权重初始化：所有模型（`E0`–`E7`）统一从同一份 `weights/rtdetr-r18.pt` 加载匹配的预训练权重。

## 3. 标准执行步骤与真实脚本入口

### Step 1：运行 NVIDIA A30 FP16 推理速度与复杂度基准测试
在启动长周期训练前，先运行统一基准测试脚本（50 次预热 + 500 次正式计时，Batch=1，640×640，FP16）：
```bash
python RTDETR_UAV_Paper_Innovations/experiments/benchmark.py \
  --warmup 50 --iters 500 --imgsz 640 --fp16 \
  --output runs/uav_paper_ablation/a30_speed_benchmark.json
```

### Step 2：第一轮独立创新模块消融训练（E0–E4，Seed 0）
- `E0`：Original RT-DETR-R18 (`experiments/ablation_configs/E0_baseline_rtdetr_r18.yaml`)
- `E1`：+ WCA-SFP (`experiments/ablation_configs/E1_inno1_wcasfp.yaml`)
- `E2`：+ FAGA (`experiments/ablation_configs/E2_inno2_faga.yaml`)
- `E3`：+ UAD-CR (`experiments/ablation_configs/E3_inno3_uadcr.yaml`)
- `E4`：+ SAH-NWD & CW-LSD (`experiments/ablation_configs/E4_inno4_sahnwd.yaml`)
```bash
for EXP in E0 E1 E2 E3 E4; do
    python RTDETR_UAV_Paper_Innovations/experiments/train.py \
      --exp $EXP \
      --data ultralytics/cfg/datasets/dut-anti-uav.yaml \
      --weights weights/rtdetr-r18.pt \
      --epochs 150 --batch 16 --imgsz 640 --device 0 --seed 0 \
      --project runs/uav_paper_ablation
done
```

### Step 3：检查第一轮训练完整性并启动第二轮组合消融（E5–E7，Seed 0）
核对 `runs/uav_paper_ablation/E*_seed0/results.csv` 确认每个实验均完整记录满 150 行（Epoch 1–150），然后运行：
- `E5`：Combo A — WCA-SFP + FAGA (`experiments/ablation_configs/E5_combo_A_feature_neck.yaml`)
- `E6`：Combo B — UAD-CR + SAH-NWD (`experiments/ablation_configs/E6_combo_B_decoder_loss.yaml`)
- `E7`：Final Joint Model (`experiments/ablation_configs/E7_final_joint_model.yaml`)
```bash
for EXP in E5 E6 E7; do
    python RTDETR_UAV_Paper_Innovations/experiments/train.py \
      --exp $EXP \
      --data ultralytics/cfg/datasets/dut-anti-uav.yaml \
      --weights weights/rtdetr-r18.pt \
      --epochs 150 --batch 16 --imgsz 640 --device 0 --seed 0 \
      --project runs/uav_paper_ablation
done
```

### Step 4：多随机种子显著性复测（Seeds 0, 1, 2）
对基线 `E0`、低开销高收益组合 `E6` 及最终模型 `E7` 补充运行 `seed = 1` 和 `seed = 2`，计算 `Mean ± Std`。

### Step 5：统一测试集评估与尺度细分指标提取
对所有完成训练的模型，调用 `RTDETR_UAV_Paper_Innovations/experiments/evaluate.py` 在 `split=test`（2,200 张测试图）上提取：
- `Precision`, `Recall`, `F1`
- `mAP50`, `mAP75`, `mAP50:95`
- `COCO_AP_Small_lt32x32` ($\text{Area} < 32^2\text{ px}^2$)、`COCO_AP_Medium_32to96`、`COCO_AP_Large_gt96x96`
- `Custom_AP_Tiny_lt16x16` ($\text{Area} < 16^2\text{ px}^2$)、`Custom_AP_VeryTiny_lt8x8` ($\text{Area} < 8^2\text{ px}^2$)

## 4. 验收标准（Acceptance Criteria）
- [ ] 所有汇报结果的实验均在 `results.csv` 中核实 `completed_epochs == 150`；任何未跑完的实验必须标注为“未完成/未训练”，严禁编造数据。
- [ ] 输出汇总 CSV 与 Markdown 报告 `runs/uav_paper_ablation/final_ablation_summary.md`，清晰区分 Val 选择指标与 Test 最终指标，并明确标注 COCO Small 与 Custom Tiny 的像素面积阈值。
```
