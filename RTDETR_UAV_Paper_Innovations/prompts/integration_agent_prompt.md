# Prompt A：创新点 5、6、7（基于原始 RT-DETR 独立接入与验证）Code Agent 执行提示词

以下提示词已针对您 GitHub 仓库（[`lishun-lishun/RTDETR-main`](https://github.com/lishun-lishun/RTDETR-main)，含 `codex/uav-research-integration` 分支最新提交 `0169ae7`、`2415a23`、`de559ef`、`4d7e695`、`7b472a9`、`6e9274d`）的实际代码结构进行了对齐，可直接复制给服务器上的 Code Agent 执行。

---

## 复制以下内容给 Code Agent：

```markdown
# 任务角色：RT-DETR 无人机小目标检测创新点 5、6、7 独立接入与验证工程师（Integration Agent）

## 1. 背景与任务目标
你正在操作基于 Ultralytics YOLOv8.0.201 的 `RTDETR-main` 仓库（分支 `codex/uav-research-integration` 或 `main`），运行环境为 Python 3.10、PyTorch 2.3.1+cu121、CUDA 12.1、NVIDIA A30 24GB。
当前仓库已同步此前的修复（包括 `0169ae7` EMA deepcopy 图缓存清理、`2415a23` AMP FP32 保护、`4d7e695` `self.nc = self.yaml['nc']` 初始化、`7b472a9` 本地 `ultralytics` 导入路径校验）。
现在需要将新版 `RTDETR_UAV_Paper_Innovations/` 中的 **创新点 5、创新点 6、创新点 7** 分别独立接入到**原始 `RT-DETR-R18`（`rtdetr-r18.yaml`）** 上，保证三个模块互不干扰，并逐一验证它们在 FP32/AMP、不同 GT 数量（0/1/3）及 `ModelEMA` `deepcopy` 下的正确性。

## 2. 三个独立创新模块与配置文件说明（均直接构建在原始 `rtdetr-r18.yaml` 上，互不干扰）
1. **创新点 5（`DMBARepC3` / `DMBABlock` — 方向性运动模糊各向异性解耦重聚模块）**：
   - 源码：`RTDETR_UAV_Paper_Innovations/innovation_5/module.py` -> 安装至 `ultralytics/nn/uav_innovations/innovation_5.py`
   - 独立模型配置：`ultralytics/cfg/models/rt-detr/rtdetr-r18-inno5-dmba.yaml`（对应消融配置 `E8_inno5_dmba.yaml`）
   - 接入位置：仅将原始 `rtdetr-r18.yaml` 的 **Layer 19 (`P3/8`, 80×80)** 与 **Layer 22 (`P4/16`, 40×40)** 由 `RepC3` 替换为 `DMBARepC3`，其余层与 `RTDETRDecoder`（Layer 26 `[[19, 22, 25], ...]`）与原始 R18 完全一致。
2. **创新点 6（`SSSCRepC3` / `SSSCLite` — 融合后有符号稀疏语义校准模块）**：
   - 源码：`RTDETR_UAV_Paper_Innovations/innovation_6/module.py` -> 安装至 `ultralytics/nn/uav_innovations/innovation_6.py`
   - 独立模型配置：`ultralytics/cfg/models/rt-detr/rtdetr-r18-inno6-sssc.yaml`（对应消融配置 `E9_inno6_sssc.yaml`）
   - 接入位置：仅将原始 `rtdetr-r18.yaml` 的 **Layer 19 (`P3/8`, $K_r=128$)** 与 **Layer 22 (`P4/16`, $K_r=64$)** 由 `RepC3` 替换为 `SSSCRepC3`，其余层与原始 R18 完全一致。
3. **创新点 7（`GLRoIDecoder` / `P2RoIReSampler` — 解码器内嵌全局-局部高分辨率 P2 ROI 重采样模块）**：
   - 源码：`RTDETR_UAV_Paper_Innovations/innovation_7/module.py` -> 安装至 `ultralytics/nn/uav_innovations/innovation_7.py`
   - 独立模型配置：`ultralytics/cfg/models/rt-detr/rtdetr-r18-inno7-glroi.yaml`（对应消融配置 `E10_inno7_glroi.yaml`）
   - 接入位置：保持原始 `rtdetr-r18.yaml` 第 0–25 层完全不变，仅将 **Layer 26** 替换为 `[[4, 19, 22, 25], 1, GLRoIDecoder, [nc, 256, 300, 4, 8, 3]]`（读取第 4 层 `P2/4` 对第 0 层解码器筛出的 Top-16 微小目标 Query 进行 $5\times 5$ 亚像素局部重采样，不增加 Encoder Token 数）。

## 3. 标准执行步骤

### Step 1：解压覆盖并运行原地升级安装器 `install.py`
```bash
unzip -o RTDETR_UAV_Paper_Innovations.zip -d .
python RTDETR_UAV_Paper_Innovations/integration/install.py --repo .
# 再次运行验证幂等性（不会重复插入 # >>> [UAV_PAPER_INNOVATIONS_BEGIN] >>> 块）
python RTDETR_UAV_Paper_Innovations/integration/install.py --repo .
git status
git diff ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py
```
检查确认：
- `ultralytics/nn/uav_innovations/__init__.py` 已导出 `DMBARepC3`, `DMBABlock`, `SSSCRepC3`, `SSSCLite`, `GLRoIDecoder`, `P2RoIReSampler`（同时保留 `innovation_1..4` 的导出与 `0169ae7`、`2415a23` 修复）；
- `ultralytics/nn/tasks.py` 的 `parse_model` 已包含 `elif m in (DMBARepC3, SSSCRepC3):`（处理 `c1, c2, n`）和 `elif m in (UADCRDecoder, GLRoIDecoder):`（插入 `[ch[x] for x in f]`），且 `RTDETRDetectionModel.__init__` 保留 `self.nc = self.yaml['nc']`。

### Step 2：运行创新点 5、6、7 独立单元测试
```bash
PYTHONPATH=RTDETR_UAV_Paper_Innovations:. pytest \
  RTDETR_UAV_Paper_Innovations/innovation_5/test_module.py \
  RTDETR_UAV_Paper_Innovations/innovation_6/test_module.py \
  RTDETR_UAV_Paper_Innovations/innovation_7/test_module.py -v
```

### Step 3：运行真实 `RTDETRDetectionModel` 冒烟测试（对齐 `tools/smoke_uav_paper_real.py` 标准）
```bash
python RTDETR_UAV_Paper_Innovations/integration/smoke_test_5_6_7.py --repo . --device cuda
```
该脚本将依次对 `rtdetr-r18.yaml`（基线）、`rtdetr-r18-inno5-dmba.yaml`、`rtdetr-r18-inno6-sssc.yaml`、`rtdetr-r18-inno7-glroi.yaml` 验证：
1. 本地 `ultralytics.__file__` 路径一致性校验；
2. `GT = 0, 1, 3` 且 `AMP = False, True`（配合 `GradScaler`）下的 `model.loss(batch)` 前向与反向传播，确保全部 Loss 与参数梯度有限（无 `NaN/Inf`）；
3. 训练步后立即执行 `copy.deepcopy(model).eval()`（模拟 `ModelEMA` 与 Checkpoint 保存），验证无计算图缓存报错，且推理输出形状严格为 `[2, 300, 5]`。
（你也可以将 `'rtdetr-r18-inno5-dmba.yaml'`, `'rtdetr-r18-inno6-sssc.yaml'`, `'rtdetr-r18-inno7-glroi.yaml'` 加入 `tools/smoke_uav_paper_real.py` 的 `CONFIGS` 列表中一并运行测试。）

### Step 4：启动独立消融训练（E8 / E9 / E10）
验证通过后，可使用统一训练入口分别测试三个模块在 `DUT-Anti-UAV` 上的独立效果：
```bash
# 先执行 dry-run 核对配置
python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp E8 --dry-run
python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp E9 --dry-run
python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp E10 --dry-run

# 正式启动训练（示例：分别测试 E8=DMBA, E9=SSSC-Lite, E10=GLRoI）
python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp E8 --data ultralytics/cfg/datasets/dut-anti-uav.yaml --weights weights/rtdetr-r18.pt
python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp E9 --data ultralytics/cfg/datasets/dut-anti-uav.yaml --weights weights/rtdetr-r18.pt
python RTDETR_UAV_Paper_Innovations/experiments/train.py --exp E10 --data ultralytics/cfg/datasets/dut-anti-uav.yaml --weights weights/rtdetr-r18.pt
```

## 4. 验收标准
- [ ] `install.py` 原地升级成功，`git diff` 干净且不破坏已有 `innovation_1..4` 的任何修复（`0169ae7`, `2415a23`, `4d7e695`, `7b472a9`）。
- [ ] `innovation_5/test_module.py`、`innovation_6/test_module.py`、`innovation_7/test_module.py` 全部 `PASSED`。
- [ ] `smoke_test_5_6_7.py` 在 `GT=0,1,3`、FP32 与 AMP、`copy.deepcopy` 下全部 `PASSED`。
```
