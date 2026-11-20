# Prompt A：创新点 8–12（基于原始 RT-DETR 独立接入）与 E4 修复 Code Agent 执行提示词

以下提示词已严格对齐您的 GitHub 仓库（[`lishun-lishun/RTDETR-main`](https://github.com/lishun-lishun/RTDETR-main) 的 `codex/uav-research-integration` 分支），可直接复制给服务器端的 Code Agent 执行。

---

## 复制以下内容给 Code Agent：

```markdown
# 任务角色：RT-DETR 无人机小目标检测创新点 8–12 独立接入与 E4 修复工程师（Integration Agent）

## 1. 任务目标
你正在操作基于 Ultralytics YOLOv8.0.201 的 `RTDETR-main` 仓库（`codex/uav-research-integration` 分支），运行环境为 Python 3.10、PyTorch 2.3.1+cu121、CUDA 12.1、NVIDIA A30 24GB。
请将新版 `RTDETR_UAV_Paper_Innovations.zip` 解压覆盖至仓库根目录，完成：
1. **同步修复创新点 4（`SAHNWDRTDETRLoss`）分类损失缩小 300 倍的 Bug**（在 `_get_loss_class` 中补充 `* nq`）；
2. **在原始 `rtdetr-r18.yaml` 上分别独立接入互不干扰的创新点 8、9、10、11、12**；
3. **运行 GPU 冒烟测试（FP32/AMP、GT=0/1/3、EMA `deepcopy`）并校验 `E11`–`E15` 训练配置**。

## 2. 五个新增独立创新模块与配置映射（全部基于原始 `rtdetr-r18.yaml`，互不干扰）
1. **E11 / 创新点 8（`UAVDenseO2OAugmenter` + `AuxO2MRTDETRLoss`）**：
   - 模型配置：`ultralytics/cfg/models/rt-detr/rtdetr-r18-inno8-denseo2o.yaml`（`E11_inno8_denseo2o.yaml`）
   - 架构改动：推理结构与原始 `rtdetr-r18.yaml` **100% 完全一致（+0.00M 参数，+0.00 GFLOPs）**；仅在训练期启用小目标软羽化 Copy-Paste 密集 O2O 增强 + Top-3 辅助一对多监督（解决 `DUT-Anti-UAV` 单图 1.02 目标极度稀疏问题）。
2. **E12 / 创新点 9（`SAFDRDecoder` + `AWLSDDistillLoss`）**：
   - 模型配置：`ultralytics/cfg/models/rt-detr/rtdetr-r18-inno9-safdr.yaml`（`E12_inno9_safdr.yaml`）
   - 架构改动：Layer 0–25 完全不变，仅将 **Layer 26** 替换为 `SAFDRDecoder`（D-FINE / FGUDet 风格非均匀 16-bin 细粒度分布回归 + 面积加权跨层定位自蒸馏）。
3. **E13 / 创新点 10（`TOLFDecoder` + `TOLFUncertaintyLoss`）**：
   - 模型配置：`ultralytics/cfg/models/rt-detr/rtdetr-r18-inno10-tolf.yaml`（`E13_inno10_tolf.yaml`）
   - 架构改动：Layer 0–25 完全不变，仅将 **Layer 26** 替换为 `TOLFDecoder`（4 维边界对数方差头 + 噪声鲁棒拉普拉斯 NLL 损失与推理期不确定性置信度校准）。
4. **E14 / 创新点 11（`DySample2x` + `HWDDownsample`）**：
   - 模型配置：`ultralytics/cfg/models/rt-detr/rtdetr-r18-inno11-dysample-hwd.yaml`（`E14_inno11_dysample_hwd.yaml`）
   - 架构改动：Backbone（Layer 0–7）与 `RTDETRDecoder`（Layer 26）完全不变，仅将 Neck 的上采样 **Layer 11、16** 替换为 `DySample2x`，将 Neck 下采样 **Layer 20、23** 替换为 `HWDDownsample`（消除特征金字塔采样混叠，保留预训练 R18 骨干加载）。
5. **E15 / 创新点 12（`SMAIFI`）**：
   - 模型配置：`ultralytics/cfg/models/rt-detr/rtdetr-r18-inno12-smaifi.yaml`（`E15_inno12_smaifi.yaml`）
   - 架构改动：仅将 **Layer 9** 由 `AIFI` 替换为 `SMAIFI`（局部峰值旁路 + 频谱调制 FFN，防止 98% 天空背景稀释 `P5` 微小目标响应），其余 26 层完全不变。

## 3. 标准执行步骤

### Step 1：解压覆盖并运行原地幂等升级脚本 `install.py`
```bash
unzip -o RTDETR_UAV_Paper_Innovations.zip -d .
python RTDETR_UAV_Paper_Innovations/integration/install.py --repo .
# 连续执行第二次验证幂等性
python RTDETR_UAV_Paper_Innovations/integration/install.py --repo .
git status
git diff ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py
```

### Step 2：确保 `ultralytics/assets/bus.jpg` 存在（防止 AMP 自检报错）
```bash
if [ ! -f ultralytics/assets/bus.jpg ]; then
  mkdir -p ultralytics/assets
  python -c "import cv2, numpy as np; cv2.imwrite('ultralytics/assets/bus.jpg', np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))"
fi
```

### Step 3：运行真实 `RTDETRDetectionModel` 冒烟测试（验证基线 + 修复后 E4 + 创新点 8–12）
```bash
python RTDETR_UAV_Paper_Innovations/integration/smoke_test_8_to_12.py --repo . --device cuda
```
确认：
- 所有 7 个配置在 `GT=0, 1, 3`、FP32 与 AMP (`GradScaler`) 下 Loss 与梯度全部有限（无 `NaN/Inf`）；
- 修复后的 `rtdetr-r18-inno4-sahnwd.yaml` 的 `cls_loss` 恢复至正常量级（`> 0.05`）；
- 每个配置训练步后执行 `copy.deepcopy(model).eval()` 均无报错，且输出形状严格为 `[2, 300, 5]`。

### Step 4：启动同条件公平对比训练（先补跑 E0 基线，再测 E4 及 E11–E15）
```bash
# 核对配置（已统一设置 cos_lr=True, lrf=0.1, mosaic=1.0, close_mosaic=15, deterministic=False）
for exp in E0 E4 E11 E12 E13 E14 E15; do
  python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp $exp --dry-run
done

# 正式启动训练示例（请根据空闲 GPU 卡号 --device 0/1/2/3 并行调度）：
python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp E0  --data dataset/DUT-Anti-UAV.yaml --weights weights/rtdetr-r18.pt --device 0
python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp E11 --data dataset/DUT-Anti-UAV.yaml --weights weights/rtdetr-r18.pt --device 1
python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp E12 --data dataset/DUT-Anti-UAV.yaml --weights weights/rtdetr-r18.pt --device 2
python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp E13 --data dataset/DUT-Anti-UAV.yaml --weights weights/rtdetr-r18.pt --device 3
```

## 4. 验收标准
- [ ] `install.py` 原地升级成功，幂等且不破坏原 `rtdetr-r18.yaml` 与 `innovation_1..7`。
- [ ] `smoke_test_8_to_12.py` 全部 `PASSED`。
- [ ] 如实汇报终端输出与 `git diff` 结果。
```
