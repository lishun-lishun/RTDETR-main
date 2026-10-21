# LSCE + ADCF 实现报告

> RT-DETR-R18 的两个可插拔改进模块：**LSCE**（Local Structural Contrast Encoding，骨干局部结构对比编码）与 **ADCF**（Asymmetric Disagreement-aware Cross-scale Fusion，颈部非对称跨尺度证据融合）。
>
> 本报告仅记录实现与工程事实，**不对性能或学术创新性作任何预先判断**。所有 mAP/FPS 结论须以服务器实测为准。

---

## 1. 结构分析（改动位置）

保持 RT-DETR-R18 原始结构不变，仅在两个固定位置做可插拔插入：

| 插入点 | 模块 | 位置 | 输入 | 输出 |
| --- | --- | --- | --- | --- |
| Backbone | LSCE | P2 阶段（`Blocks` 层之后） | P2 特征 `[B,64,H/4,W/4]` | 增强 P2（同通道、同分辨率） |
| Neck | ADCF | FPN 的 P3 融合块（`X3 = RepC3`）之后 | `[F3, Y4]` | 增强 P3（`[B,256,H/8,W/8]`） |

未改动项（明确约束）：

- 保留原始 `ConvNormLayer` / `Blocks` / `BasicBlock`（PResNet-18 骨干）。
- 保留原始 `AIFI` / `RepC3` / `RTDETRDecoder`。
- Decoder 仍为三尺度 **P3/P4/P5**，不新增 P2 检测头、不增加 Query。
- 不修改已有 Baseline / BAPS / RDCF 的 YAML、权重、日志。
- 不调整学习率 / Epoch / Batch / 输入尺寸 / 检测损失（除 E6 可选项，见 §11）。

## 2. 模块结构

### 2.1 LSCE（Backbone，P2 局部结构对比编码）

核心思路：用每个像素的「中心外观」与其「环形背景」的对比表达局部结构，放大天空背景下的小目标（如无人机）。

```text
P2 (B,64,H/4,W/4)
  └─ reduce (1x1 conv, 64→h=32)                    -> F
       ├─ center: DW3x3 + BN + ReLU + PW1x1        -> F_center
       └─ ring5 / ring9（replicate 填充保尺寸平均池化）
            C5 = (F_center - ring5)/(|F_center|+|ring5|+ε)
            C9 = (F_center - ring9)/(|F_center|+|ring9|+ε)
  contrast_feature = contrast_fuse(concat(C5[,C9]))  (1x1 conv, 2h/h→h)
  gate = σ(1x1(ReLU(1x1(concat(C5[,C9], grad_x, grad_y, var)))))   # 背景复杂度门控（可选）
  delta = delta(gate * contrast_feature)            (1x1 conv, h→64)
  alpha = alpha_max * tanh(raw_alpha)               # raw_alpha 零初始化
  out = x + alpha * tanh(delta)
```

关键点：

- `_avg_pool_replicate` 用 **replicate 填充**，保证 `25*avg5 - 9*avg3` / `81*avg9 - 9*avg3` 是精确的环形和，无边界伪影。
- `use_complexity` 控制是否加入梯度幅值 + 局部方差门控（E4 关闭）。
- `multi_scale` 控制 ring5+ring9（E4 关闭，仅 ring5）。

### 2.2 ADCF（Neck，P3/P4 非对称跨尺度证据融合）

核心思路：共享证据头读 P3、P4 得到两个「校准」的证据图 S3/S4，用「一致性」路由粗尺度语义、用「分歧」路由细尺度局部，两条路径独立门控。

```text
F3 (B,256,H/8,W/8), F4 (B,256,H/16,W/16)
  d3 = d3_proj(F3)  (1x1 conv 256→h=64)
  d4 = upsample(d4_proj(F4))                       # 对齐到 H/8
  evidence（共享头，3x3 conv 64→64→1）
       s3 = σ(evidence(d3));  s4 = σ(evidence(d4))
  agreement    = s3 * s4          -> 语义路径: phi_semantic(d4)
  disagreement = relu(s3 - s4)    -> 局部路径: phi_local(d3)
  a_s = alpha_max * tanh(raw_alpha_semantic)       # 零初始化
  a_l = alpha_max * tanh(raw_alpha_local)          # 零初始化
  out = f3 + a_s * delta_semantic(agreement*phi_sem) + a_l * delta_local(disagreement*phi_loc)
```

- `use_evidence=False` 时退化为 E5 的普通学习门控融合：`g = σ(gate(concat(d3,d4)))`，`fused = phi_fused(g*d3 + (1-g)*d4)`，`out = f3 + a_s*delta_semantic(fused)`。
- `use_center_supervision`（E6）当前保留为开关，**辅助中心监督损失尚未接线**（见 §11）。

## 3. 代码文件

| 文件 | 变更 |
| --- | --- |
| `ultralytics/nn/modules/lsce_adcf.py` | **新增**：`LSCE` / `ADCF` 及辅助函数 `_avg_pool_replicate` / `_ring_mean` |
| `ultralytics/nn/modules/__init__.py` | 新增 `from .lsce_adcf import *` |
| `ultralytics/nn/tasks.py` | `parse_model` 新增 `LSCE` / `ADCF` 分支 |
| `ultralytics/cfg/models/rt-detr/rtdetr_r18_lsce.yaml` | **新增**（E1） |
| `ultralytics/cfg/models/rt-detr/rtdetr_r18_adcf.yaml` | **新增**（E2） |
| `ultralytics/cfg/models/rt-detr/rtdetr_r18_lsce_adcf.yaml` | **新增**（E3） |
| `ultralytics/cfg/models/rt-detr/rtdetr_r18_lsce_e4.yaml` | **新增**（E4 消融） |
| `ultralytics/cfg/models/rt-detr/rtdetr_r18_adcf_e5.yaml` | **新增**（E5 消融） |
| `ultralytics/cfg/models/rt-detr/rtdetr_r18_adcf_e6.yaml` | **新增**（E6 消融，损失未接线） |
| `tests/test_lsce_adcf.py` | **新增**：单元测试 |
| `tools/train_r18_lsce_adcf.py` | **新增**：服务器训练脚本 |
| `tools/evaluate_r18_lsce_adcf.py` | **新增**：服务器评估脚本 |
| `tools/benchmark_r18_lsce_adcf.py` | **新增**：服务器性能基准 |
| `tools/collect_lsce_adcf_statistics.py` | **新增**：模块内部行为统计 |

### `parse_model` 适配

```python
elif m is LSCE:
    c1 = ch[f]
    c2 = c1
    args = [c1, *args]
elif m is ADCF:
    c1 = [ch[x] for x in f]
    c2 = c1[0]
    args = [*c1, *args]
```

`ADCF` 的 `from` 是列表 `[f3_idx, f4_idx]`，故 `c1 = [ch[x] for x in f]` 收集两路输入通道；`c2 = c1[0]`（输出与 P3 对齐）。`LSCE` 为单输入。

## 4. YAML 连接层号（关键）

| 配置 | LSCE 插入层 | ADCF 插入层 | Decoder `from` | 说明 |
| --- | --- | --- | --- | --- |
| Baseline `rtdetr-r18.yaml` | — | — | `[19, 22, 25]` | 原始，未改 |
| E1 `rtdetr_r18_lsce.yaml` | 第 5 层（读 P2=4） | — | `[20, 23, 26]` | 下游全部 +1 |
| E2 `rtdetr_r18_adcf.yaml` | — | 第 20 层（读 F3=19, Y4=15） | `[20, 23, 26]` | ADCF 输出即 P3 |
| E3 `rtdetr_r18_lsce_adcf.yaml` | 第 5 层 | 第 21 层（读 F3=20, Y4=16） | `[21, 24, 27]` | 累计 +2 |
| E4 `rtdetr_r18_lsce_e4.yaml` | 第 5 层（`[32,1.0,False,False]`） | — | `[20, 23, 26]` | 单尺度 ring5 + 无复杂度门控 |
| E5 `rtdetr_r18_adcf_e5.yaml` | — | 第 20 层（`[64,1.0,False,False]`） | `[20, 23, 26]` | 普通门控融合 |
| E6 `rtdetr_r18_adcf_e6.yaml` | — | 第 20 层（`[64,1.0,True]`） | `[20, 23, 26]` | 中心监督开关（损失未接线） |

层号核对要点：每插入一个模块，其后所有层号 +1；`input_proj` / `lateral_convs` / `downsample_convs` 的 `from` 索引已同步改指正确的特征层。`Decoder.input_proj` 的 `in_channels` 应为 `[256, 256, 256]`（R18 全三尺度 256），由单元测试断言。

## 5. 参数量 / GFLOPs

- LSCE（`hidden_channels=32`）新增约 **0.1M** 参数量（`reduce` 64→32、`center` 3x3 DW、`contrast_fuse` 2·32→32、`gate` 1x1、`delta` 32→64）。
- ADCF（`hidden_channels=64`）新增约 **0.3M** 参数量（`d3_proj`/`d4_proj` 256→64、`evidence`/`phi_*` 3x3、`delta_*` 64→256）。
- 两者相对 R18 基准（约 20M 参数量级）占比极小。

**精确值以服务器 `tools/benchmark_r18_lsce_adcf.py` 实测为准**（该脚本同时输出 params / GFLOPs / FPS / 延迟，并与 baseline 对照）。

## 6. 前反向测试（单元测试 `tests/test_lsce_adcf.py`）

| 测试 | 覆盖 |
| --- | --- |
| `test_lsce_shape_and_identity` | 输入输出同 shape；**初始化时 `out == x`（恒等）** |
| `test_adcf_shape_and_identity` | `use_evidence=True/False` 均恒等、shape 正确 |
| `test_lsce_gradient_flow` | `raw_alpha` 首步梯度非零；推进 scale 后 `delta.weight` 梯度非零 |
| `test_adcf_gradient_flow` | 同上（两条路径 scale + delta 梯度） |
| `test_yaml_instantiation_and_decoder_channels` | 6 份 YAML 均可实例化，`decoder.input_proj` 为 `[256,256,256]` |
| `test_full_model_forward` | 640/480 全模型前向，输出形状正确 |
| `test_amp_fp16_forward` | FP16 前向（无 CUDA 时 skip） |

本地无 Python/torch 环境，运行时验证须在服务器 GPU 上执行（见 §9）。

## 7. 初始化梯度检查（ReZero 修复）

**已修复的关键梯度流 bug**：此前 LSCE 的 `delta` 与 ADCF 的 `delta_semantic`/`delta_local` 被 `nn.init.zeros_` 零初始化，且 scale factor 也零初始化，导致投影层首步无梯度、永远无法激活。

修正为 **ReZero 风格**：

- 投影层（`delta` 等）改用默认（Kaiming）初始化 → 非零权重。
- 仅 scale factor（`raw_alpha` / `raw_alpha_semantic` / `raw_alpha_local`）保持零初始化 → 保证初始化时 `out == x` 恒等。

即：**正常初始化投影 + 零初始化 scale factor**（二者不可同时零初始化）。`initialize_weights()` 在模型构造末尾会再次 Kaiming 初始化投影层（非零），不影响本修复。

## 8. AMP / FP16 兼容

- 固定 Sobel 核通过 `FixedDepthwiseConv` 注册为 buffer，forward 时按 `x.dtype` 转换。
- 所有归一化用稳定 ε（`eps=1e-4`），环形均值用 replicate 填充避免零边界。
- `ADCF` 的 `F.interpolate(..., mode='bilinear', align_corners=False)` 兼容 FP16。

## 9. 服务器训练 / 评估 / 基准 / 统计命令

> 服务器路径 `/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main`，环境 `ultralytics_rtdetr`。

```bash
# 训练（--train 才真正启动；默认 dry-run 打印计划）
python tools/train_r18_lsce_adcf.py --model lsce --device 0 --train
python tools/train_r18_lsce_adcf.py --model adcf --device 1 --train
python tools/train_r18_lsce_adcf.py --model lsce_adcf --device 0,1,2 --train

# 消融
python tools/train_r18_lsce_adcf.py --model e4_lsce_simple --device 0 --train
python tools/train_r18_lsce_adcf.py --model e5_adcf_plain --device 0 --train

# 评估（默认找 best.pt，输出 CSV 汇总）
python tools/evaluate_r18_lsce_adcf.py --model lsce_adcf --split test --device 0

# 性能基准（含 baseline 对照）
python tools/benchmark_r18_lsce_adcf.py --model all --device 0

# 模块行为统计（判断模块是否"激活"：alpha>0、residual_ratio>0）
python tools/collect_lsce_adcf_statistics.py --model lsce_adcf --split val --samples 64
```

脚本约定（与 `train_dut_anti_uav.py` 一致）：

- YAML 由 `ultralytics/cfg/models/rt-detr/` 下文件名直接选择。
- 每个模型写入独立 `runs/<project_dir>/train/<name>`，不覆盖、不混用；已存在目录默认跳过（`--resume`/`--force` 继续）。
- `--weights` 从检查点初始化；`--resume` 从 `latest.pt` 恢复。

## 10. 配置一致性

训练参数统一读取 `ultralytics/cfg/anti_detr_default.yaml`（与 Baseline/BAPS/RDCF 完全一致）：

`epochs=150, patience=0, batch=8, micro_batch=2, test_batch=16, imgsz=640, cache=false, workers=4, pretrained=false, optimizer=AdamW, seed=0, deterministic=true, cos_lr=false, close_mosaic=0, amp=false, lr0=0.0001, lrf=1.0, momentum=0.9, weight_decay=0.0001, warmup_epochs=2000, warmup_momentum=0.8, warmup_bias_lr=0.0, max_det=300, mosaic/mixup/copy_paste=0.0`

### warmup 机制说明（重要）

本项目 `warmup_epochs` 字段实际表示 **warmup iterations**。训练脚本按 `batch // micro_batch`（= 8 // 2 = 4）累加：

```text
gradient_accumulation = 4
warmup_micro_steps    = warmup_epochs * accumulation = 2000 * 4 = 8000
```

即实际 warmup 为 **8000 个 micro-step**。本实现完全复用该机制，未擅自改动。

## 11. 未完成项 / 已知问题

- **E6 中心监督损失未接线**：`use_center_supervision=True` 的 ADCF 目前仍是纯特征融合块，辅助目标中心监督损失尚未接入 trainer（YAML 头已标注 `TRAINER WIRING PENDING`）。默认检测损失保持不变。E6 仅作为未来可选项预留。
- 本地无 Python/torch 环境，单元测试与数值验证均待服务器 GPU 执行。
- `get_errors` 报告的 "line too long (>79)" 均为 pycodestyle 风格提示（含 docstring/注释），**无 Python 语法错误**，不影响运行。

## 12. 消融矩阵

| 配置 | LSCE | ADCF | 关键变量 | 目的 |
| --- | --- | --- | --- | --- |
| Baseline | ✗ | ✗ | — | 对照 |
| E1 | ✓ 完整 | ✗ | LSCE 主效果 | 骨干局部对比 |
| E2 | ✗ | ✓ 完整 | ADCF 主效果 | 颈部跨尺度证据融合 |
| E3 | ✓ 完整 | ✓ 完整 | 联合效果 | 联合增益 |
| E4 | ✓ 单尺度/无门控 | ✗ | multi_scale、use_complexity | LSCE 消融 |
| E5 | ✗ | ✓ 普通门控 | use_evidence | ADCF 证据路径消融 |
| E6 | ✗ | ✓ 中心监督（未接线） | use_center_supervision | 辅助监督预留 |

---

*实现日期见文件提交记录。所有性能结论待服务器实测，本文档不预设任何优越性。*
