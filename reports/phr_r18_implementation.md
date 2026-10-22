# PHR-ResNet18 实现报告

> RT-DETR-R18 骨干的重新设计：**PHR**（Partial High-Resolution Residual Backbone，部分并行高分辨率残差骨干）。
>
> 本报告仅记录实现与工程事实，**不对性能或学术创新性作任何预先判断**。所有 mAP/FPS 结论须以服务器实测为准。本次任务只研究 **Backbone**，不修改 Neck / Decoder。

---

## 1. 结构分析（改动位置）

保持 RT-DETR-R18 原始 Backbone 的 P2/P3/P4/P5 主干（`ConvNormLayer` / `Blocks` / `BasicBlock`）完全不变，仅在 **Backbone 之后、Neck 之前** 插入一个可插拔的 PHR 模块，它：

- 读取已计算好的 **P2 / P3 / P4** 三个尺度特征；
- 内部维护一条轻量的 **1/4 高分辨率残差路径**；
- 产出一个「增强 P3」，作为 Neck P3 侧向输入（`input_proj.0`）的唯一上游。

| 插入点 | 模块 | 位置 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| Backbone→Neck 之间 | PHR | 第 8 层（读 P2=4, P3=5, P4=6） | `[P2, P3, P4]` | 增强 P3 `[B,128,H/8,W/8]` |

未改动项（明确约束）：

- 保留原始 `ConvNormLayer` / `Blocks` / `BasicBlock`（PResNet-18 骨干）。
- 保留原始 `AIFI` / `RepC3` / `RTDETRDecoder`，**不修改 Neck / Decoder**。
- Decoder 仍为三尺度 **P3/P4/P5**，不新增 P2 检测头、不增加 Query、不增加辅助损失。
- 不修改已有 Baseline / BAPS / RDCF / LSCE / ADCF 的 YAML、权重、日志。
- 不调整学习率 / Epoch / Batch / 输入尺寸 / 检测损失。

## 2. 模块结构

### 2.1 HRBlock（高分辨率残差块）

保持 1/4 尺度、通道数不变；无全局自注意力，轻量、AMP 友好。

```text
x (B, h, H/4, W/4)
  └─ DW Conv3x3 (groups=h) -> Pointwise 1x1 -> BatchNorm -> ReLU -> + x
```

内部层正常初始化（Kaiming）；「模块级恒等」由 PHR 的零初始化 `gamma` 保证，而非零化本块。

### 2.2 PHR（部分并行高分辨率残差骨干）

```text
P2 (B,64,H/4,W/4)   P3 (B,128,H/8,W/8)   P4 (B,256,H/16,W/16)
        │                    │                    │
   hr_init (1x1, 64→h=32)    │                    │
        │                    │                    │
        v                    │                    │
       H0 @1/4               │                    │
        │                    │                    │
   ┌────┴──── loop ──────────┴─────────┐          │
   │  for i in range(num_hr_blocks):   │          │
   │      H = HRBlock_i(H)             │          │
   │      if P3 feedback pending:      │          │
   │          H = H + beta3*up(p3_proj(P3))  @1/4 │
   │      if P4 feedback pending:      │          │
   │          H = H + beta4*up(p4_proj(P4))  @1/4 │
   └───────────────────────────────────┘          │
        │
   downsample (3x3 s2, h→h) + BN + ReLU   @1/8
        │
   delta (1x1, h→128)  ->  Delta_P3
        │
   gamma = alpha_max * tanh(raw_gamma)      # 有界、零初始化
        │
   return P3 + gamma * Delta_P3             # 增强 P3
```

反馈注入点随三个递进版本布局：

| 版本 | num_hr_blocks | P3 反馈 | P4 反馈 | 布局 |
| --- | --- | --- | --- | --- |
| PHR-1 | 2 | ✗ | ✗ | `H0 → block0 → block1 → 补偿` |
| PHR-2 | 2 | ✓ | ✗ | `H0 → block0 → +P3 → block1 → 补偿` |
| PHR-3 | 3 | ✓ | ✓ | `H0 → block0 → +P3 → block1 → +P4 → block2 → 补偿` |

关键点：

- `hr_init = ConvNormLayer(c_p2, h, 1, 1, act='relu')`（P2→H0 @1/4）。
- 反馈仅当对应开关为 `True` 时才创建（`p3_proj`+`p3_norm`+`raw_beta3` / `p4_proj`+`p4_norm`+`raw_beta4`），`F.interpolate(..., mode='bilinear', align_corners=False)` 上采样到 H 实际尺寸。
- 补偿路径 `downsample(3x3 s2)→down_norm→down_act(ReLU)→delta(1x1 h→c_p3)` 正常初始化。
- `gamma = alpha_max*tanh(raw_gamma)`、`beta = beta_max*tanh(raw_beta)`，scale 参数零初始化 → **初始化时精确恒等**（`P3_enhanced == P3`）。
- 多输入：YAML `from` 列表 `[p2_idx, p3_idx, p4_idx]`，`forward` 接收单个 list 参数。

## 3. 代码文件

| 文件 | 变更 |
| --- | --- |
| `ultralytics/nn/modules/phr.py` | **新增**：`PHR` / `HRBlock` |
| `ultralytics/nn/modules/__init__.py` | 新增 `from .phr import *` |
| `ultralytics/nn/tasks.py` | `parse_model` 新增 `PHR` 分支 |
| `ultralytics/cfg/models/rt-detr/rtdetr_r18_phr1.yaml` | **新增**（PHR-1，无反馈） |
| `ultralytics/cfg/models/rt-detr/rtdetr_r18_phr2.yaml` | **新增**（PHR-2，P3 反馈） |
| `ultralytics/cfg/models/rt-detr/rtdetr_r18_phr3.yaml` | **新增**（PHR-3，P3+P4 反馈） |
| `tests/test_phr.py` | **新增**：单元测试 |
| `tools/validate_phr.py` | **新增**：训练前静态校验（构建 + 参数量/FLOPs + 恒等 + 前向） |
| `tools/train_r18_phr.py` | **新增**：服务器训练脚本 |
| `tools/evaluate_r18_phr.py` | **新增**：服务器评估脚本 |
| `tools/benchmark_r18_phr.py` | **新增**：服务器性能基准（含 baseline 对照） |
| `tools/collect_phr_diagnostics.py` | **新增**：PHR 模块内部行为统计 |

### `parse_model` 适配

```python
elif m is PHR:
    c1 = [ch[x] for x in f]
    c2 = c1[1]          # c_p3（输出与 P3 对齐）
    args = [*c1, *args] # (c_p2, c_p3, c_p4, hr_channels, num_hr_blocks, ...)
```

`PHR` 的 `from` 是列表 `[p2_idx, p3_idx, p4_idx]`，故 `c1 = [ch[x] for x in f]` 收集三路输入通道；`c2 = c1[1]`（输出与 P3 对齐）。YAML `args` 依次为 `(hr_channels, num_hr_blocks, use_p3_feedback, use_p4_feedback)`。

## 4. YAML 连接层号（关键）

Backbone 层 0–7 与原始 R18 完全一致；PHR 固定插在第 8 层，其后所有层号相对原始 R18 全部 **+1**。

| 配置 | PHR 插入层（args） | `input_proj.0` 读 | P5 `input_proj.2` 读 | Decoder `from` |
| --- | --- | --- | --- | --- |
| Baseline `rtdetr-r18.yaml` | — | `[5]`（P3） | `[-1]` | `[19, 22, 25]` |
| PHR-1 | 第 8 层 `[32,2,False,False]` | `[8]`（增强 P3） | `[7]`（P5） | `[20, 23, 26]` |
| PHR-2 | 第 8 层 `[32,2,True,False]` | `[8]` | `[7]` | `[20, 23, 26]` |
| PHR-3 | 第 8 层 `[32,3,True,True]` | `[8]` | `[7]` | `[20, 23, 26]` |

层号核对要点：

- PHR 读 `[4, 5, 6]`（P2/P3/P4），输出为第 8 层。
- P5 的 `input_proj.2` 由 `[-1]` 改为 `[7]`（因为 PHR 已占据 `-1`）。
- P3 的 `input_proj.0` 由 `[5]` 改为 `[8]`（读增强 P3）。
- 三个版本 PHR 均在 layer 8，下游（input_proj / AIFI / lateral / upsample / concat / RepC3 / downsample）层号相对原始 R18 整体 +1。
- `Decoder.input_proj` 的 `in_channels` 应为 `[256, 256, 256]`（R18 全三尺度 256），由单元测试与 `validate_phr.py` 双重断言。

## 5. 参数量 / GFLOPs

- PHR（`hr_channels=32`）新增参数量极小：`hr_init` 64→32、每个 `HRBlock` 为 DW3x3(32) + PW1x1(32) + BN(32)、`downsample` 32→32 s2、`delta` 32→128，反馈分支（若开启）另加 `p3_proj` 128→32 / `p4_proj` 256→32。
- 相对 R18 基准（约 20M 参数量级）占比极小；PHR-1 < PHR-2 < PHR-3（反馈分支 + 更深路径依次增加）。

**精确值以服务器 `tools/benchmark_r18_phr.py` 实测为准**（该脚本同时输出 params / GFLOPs / FPS / 延迟，并与 baseline 对照）。

## 6. 前反向测试（单元测试 `tests/test_phr.py`）

| 测试 | 覆盖 |
| --- | --- |
| `test_hrblock_shape_and_residual` | HRBlock 保持通道与空间尺寸 |
| `test_hrblock_keeps_channels_and_spatial` | 多种通道/尺寸下的形状不变 |
| `test_phr_output_shape[fb3,fb4,nblk]` | 三版本 `(F,F,2)/(T,F,2)/(T,T,3)` 输出 shape == P3 |
| `test_phr_identity_at_init[...]` | 初始化时 `out == p3`（`torch.equal`，精确恒等） |
| `test_phr_gradient_flow_gamma_then_delta` | `raw_gamma` 首步梯度非零；`delta` 首步梯度为 0；推进 scale 后 `delta.weight` 梯度非零 |
| `test_phr_feedback_gradient_after_unlock[...]` | 解锁 gamma 后 `raw_beta3`/`p3_proj`（及 P4 时 `raw_beta4`/`p4_proj`）梯度非零 |
| `test_phr_feedback_convs_created_only_when_enabled` | 反馈分支仅在开关开启时创建 |
| `test_phr_forward_hook_receives_list` | forward hook 的 `inp[0]` 是 len==3 的 list |
| `test_yaml_instantiation_and_decoder_channels[yaml]` | 3 份 YAML 可实例化，`decoder.input_proj` 为 `[256,256,256]`，恰好 1 个 PHR |
| `test_full_model_forward[yaml][imgsz]` | 640/480 全模型前向，eval 输出 tuple 且 `out[0].shape=(1,300,4+nc)` |

本地无 Python/torch 环境，运行时验证须在服务器 GPU 上执行（见 §9）。

## 7. 初始化梯度检查（ReZero 约定）

沿用 LSCE/ADCF 阶段验证过的 **ReZero 风格**：

- 投影层（`hr_init` / `HRBlock` 内部层 / `downsample` / `delta` / 反馈投影）用默认（Kaiming）初始化 → 非零权重。
- 仅 scale 参数（`raw_gamma` / `raw_beta3` / `raw_beta4`）零初始化 → 保证初始化时 `P3_enhanced == P3` 精确恒等，同时 `raw_gamma` 首步即收到非零梯度。

即：**正常初始化投影 + 零初始化 scale**（二者不可同时零初始化）。`initialize_weights()` 对 `nn.Conv2d` 分支为 `pass`（不重置），因此投影层默认初始化得以保留。

## 8. AMP / FP16 兼容

- 纯 conv / BN / interpolate / tanh 运算，无固定卷积核、无 `.detach()`、无整数索引依赖。
- `F.interpolate(..., mode='bilinear', align_corners=False)` 兼容 FP16。
- BN 使用 `torch_utils` 约定的 `eps=1e-3, momentum=0.03`，ReLU `inplace=False`（避免干扰 hook 采集）。

## 9. 服务器训练 / 评估 / 基准 / 统计命令

> 服务器路径 `/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main`，环境 `ultralytics_rtdetr`。

```bash
# 训练前静态校验（构建 + 参数量/FLOPs + 恒等 + 前向，不训练）
python tools/validate_phr.py --device 0

# 训练（--train 才真正启动；默认 dry-run 打印计划）
python tools/train_r18_phr.py --model phr1 --device 0 --train
# 按规格书通常不跑 PHR-2/3 的训练（仅代码 + YAML 验证）：
# python tools/train_r18_phr.py --model phr2 --device 1 --train
# python tools/train_r18_phr.py --model phr3 --device 0,1,2 --train

# 评估（默认找 best.pt，输出 CSV 汇总）
python tools/evaluate_r18_phr.py --model phr1 --split test --device 0

# 性能基准（含 baseline 对照）
python tools/benchmark_r18_phr.py --model all --device 0

# 模块行为统计（判断模块是否"激活"：gamma>0、residual_ratio>0）
python tools/collect_phr_diagnostics.py --model phr1 --split val --samples 64
```

脚本约定（与 `train_dut_anti_uav.py` / LSCE / ADCF 一致）：

- YAML 由 `ultralytics/cfg/models/rt-detr/` 下文件名直接选择。
- 每个模型写入独立 `runs/<project_dir>/train/<name>`，不覆盖、不混用；已存在目录默认跳过（`--resume`/`--force` 继续）。
- `--weights` 从检查点初始化；`--resume` 从 `latest.pt` 恢复。

## 10. 配置一致性

训练参数统一读取 `ultralytics/cfg/anti_detr_default.yaml`（与 Baseline/BAPS/RDCF/LSCE/ADCF 完全一致）：

`epochs=150, patience=0, batch=8, micro_batch=2, test_batch=16, imgsz=640, cache=false, workers=4, pretrained=false, optimizer=AdamW, seed=0, deterministic=true, cos_lr=false, close_mosaic=0, amp=false, lr0=0.0001, lrf=1.0, momentum=0.9, weight_decay=0.0001, warmup_epochs=2000, warmup_momentum=0.8, warmup_bias_lr=0.0, max_det=300, mosaic/mixup/copy_paste=0.0`

### warmup 机制说明（重要）

本项目 `warmup_epochs` 字段实际表示 **warmup iterations**。训练脚本按 `batch // micro_batch`（= 8 // 2 = 4）累加：

```text
gradient_accumulation = 4
warmup_micro_steps    = warmup_epochs * accumulation = 2000 * 4 = 8000
```

即实际 warmup 为 **8000 个 micro-step**。本实现完全复用该机制，未擅自改动。

## 11. 未完成项 / 已知问题

- 按规格书 **只训练 PHR-1**；PHR-2 / PHR-3 仅完成代码 + YAML + 静态校验（`validate_phr.py` / `test_phr.py`），**不启动训练**（训练入口保留以支持后续复现）。
- 本地无 Python/torch 环境，单元测试与数值验证均待服务器 GPU 执行。
- `get_errors` 报告的 "line too long (>79)" 均为 pycodestyle 风格提示（含 docstring/注释），**无 Python 语法错误**，不影响运行。

## 12. 消融矩阵

| 配置 | hr_channels | num_hr_blocks | P3 反馈 | P4 反馈 | 关键变量 | 目的 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | — | — | — | — | — | 对照 |
| PHR-1 | 32 | 2 | ✗ | ✗ | 高分辨率路径主效果 | 骨干高分辨率残差 |
| PHR-2 | 32 | 2 | ✓ | ✗ | P3 语义反馈 | 跨尺度信息回流 |
| PHR-3 | 32 | 3 | ✓ | ✓ | P3+P4 反馈、更深路径 | 多尺度反馈 + 深度 |

---

*实现日期见文件提交记录。所有性能结论待服务器实测，本文档不预设任何优越性。*
