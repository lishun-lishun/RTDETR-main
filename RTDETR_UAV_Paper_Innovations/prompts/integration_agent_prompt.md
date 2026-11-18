# Prompt A：代码集成与验证 Code Agent 执行提示词（Integration Agent Prompt）

你可以将以下提示词直接复制给运行在 **Linux 服务器（Python 3.10 + PyTorch 2.3.1+cu121 + NVIDIA A30 24GB）** 上的 Code Agent，完成四个创新点向 `RTDETR-main` 项目的无损接入与全链路验证。

---

## 复制以下内容给 Code Agent：

```markdown
# 任务角色：RT-DETR 无人机小目标检测代码集成与验证工程师（Integration Agent）

## 1. 任务目标
你正在操作基于 Ultralytics YOLOv8.0.201 修改的 `RTDETR-main` 仓库，运行环境为 Python 3.10、PyTorch 2.3.1+cu121、CUDA 12.1、NVIDIA A30 24GB。
当前目录下已解压交付包 `RTDETR_UAV_Paper_Innovations/`。你的任务是将四个经过新颖性审查的论文创新模块安全、幂等地接入 `RTDETR-main`，并完成模块注册、YAML 解析、Tensor Shape 校验、Forward/Backward/AMP 测试以及 Checkpoint 保存与重加载验证。

## 2. 涉及的真实文件路径与配置入口
1. **新增创新模块源码（位于交付包内）**：
   - `RTDETR_UAV_Paper_Innovations/innovation_1/module.py`：包含 `WCASFP`（小波-对比度各向异性浅层先验桥）与 `ShallowEllipseAuxLoss`（训练期高斯椭圆浅层辅助监督）。
   - `RTDETR_UAV_Paper_Innovations/innovation_2/module.py`：包含 `FAGAFusion`（频空解耦门控与有界亚像素相位对齐跨尺度融合）。
   - `RTDETR_UAV_Paper_Innovations/innovation_3/module.py`：包含 `UADCRDecoder`、`VarianceFlooredMSDeformAttn`（不确定性校准查询选择与上下文方差保底可变形解码器）。
   - `RTDETR_UAV_Paper_Innovations/innovation_4/module.py`：包含 `SAHNWDRTDETRLoss` 与 `ScaleAdaptiveHungarianMatcher`（尺度自适应 Wasserstein-InnerGIoU 连续匹配、广义 MAL 软标签与跨层定位自蒸馏）。
2. **安装与回滚脚本**：
   - 安装器：`RTDETR_UAV_Paper_Innovations/integration/install.py`
   - 回滚器：`RTDETR_UAV_Paper_Innovations/integration/rollback.py`
   - 补丁文件：`RTDETR_UAV_Paper_Innovations/integration/integration.patch`
3. **目标仓库受影响的源码文件（仅 2 个文件修改，严禁覆盖原始配置与权重）**：
   - `ultralytics/nn/uav_innovations/`（新增独立子包目录）
   - `ultralytics/nn/modules/__init__.py`（导出 `WCASFP`, `FAGAFusion`, `UADCRDecoder`）
   - `ultralytics/nn/tasks.py`（在 `parse_model` 中注册上述三个模块，并通过钩子在 `RTDETRDetectionModel.init_criterion` 和 `loss` 中支持 `SAHNWDRTDETRLoss` 与 `ShallowEllipseAuxLoss`）
   - `ultralytics/cfg/models/rt-detr/`（新增 `rtdetr-r18-inno1-wcasfp.yaml`、`rtdetr-r18-inno2-faga.yaml`、`rtdetr-r18-inno3-uadcr.yaml`、`rtdetr-r18-inno4-sahnwd.yaml`、`rtdetr-r18-combo-a.yaml`、`rtdetr-r18-combo-b.yaml`、`rtdetr-r18-uav-paper-full.yaml`；**严禁修改或覆盖原始 `rtdetr-r18.yaml`**）。

## 3. 标准执行步骤

### Step 1：执行幂等安装并验证 Git Diff
```bash
python RTDETR_UAV_Paper_Innovations/integration/install.py --repo .
# 再次执行一次，验证幂等性（不得重复插入代码块）
python RTDETR_UAV_Paper_Innovations/integration/install.py --repo .
git status
git diff ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py
```

### Step 2：运行四个独立创新点的单元测试与端到端集成测试
```bash
PYTHONPATH=RTDETR_UAV_Paper_Innovations:. pytest \
  RTDETR_UAV_Paper_Innovations/innovation_1/test_module.py \
  RTDETR_UAV_Paper_Innovations/innovation_2/test_module.py \
  RTDETR_UAV_Paper_Innovations/innovation_3/test_module.py \
  RTDETR_UAV_Paper_Innovations/innovation_4/test_module.py \
  RTDETR_UAV_Paper_Innovations/integration/test_integration.py -v
```

### Step 3：在真实 `RTDETRDetectionModel` 上验证 YAML 解析、AMP、Backward 与 Checkpoint
编写并执行一段快速验证命令，逐一加载 `E0` 至 `E7` 的 YAML 配置文件，验证：
1. 输入 `img = torch.randn(2, 3, 640, 640, device='cuda')`；
2. 分别构造 Empty GT（0 个目标）、Single GT（1 个微小目标 $w=h=0.012$）、Multiple GT（3 个目标）三种 Batch；
3. 在 FP32 与 `torch.cuda.amp.autocast(True)` 下分别执行 `model.loss(batch)`、反向传播 `loss.backward()`、`optimizer.step()`；
4. 验证所有新增可学习参数（`u_gamma`, `offset_head`, `beta_low`, `beta_high`, `u_eta`, `enc_quality_head`, `target_gate`）梯度非零且无 `NaN/Inf`；
5. 验证保存 `torch.save(model.state_dict(), '/tmp/test_ckpt.pt')` 并重新加载后，推理模式 `model.eval()` 输出形状严格为 `[B, 300, 5]`（单类别 `nc=1`）；
6. 验证当所有模块 `enabled=False` 时，模型输出与原始 `rtdetr-r18.yaml` 的最大绝对误差 $\le 10^{-5}$。

## 4. 验收标准（Acceptance Criteria）
- [ ] `git diff` 仅包含 `# >>> [UAV_PAPER_INNOVATIONS_BEGIN] >>>` 标记包裹的最小接入代码，原始 `rtdetr-r18.yaml`、`weights/`、`runs/` 完好无损。
- [ ] 5 个测试脚本全部 `PASSED`（0 failed）。
- [ ] `RTDETRDetectionModel` 在 GPU FP32 与 AMP 下对 `E0–E7` 全部通过 Forward + Loss + Backward + Checkpoint Reload 验证。
```
