# RT-DETR-R18 CSP Stage 实现记录

## 修改范围

新增 `ultralytics/nn/modules/csp_context.py`，并在 `ultralytics/nn/modules/__init__.py`、`ultralytics/nn/tasks.py` 注册。原始 `rtdetr-r18.yaml`、通用 `BasicBlock`、Head、AIFI、CCFM、RepC3、Decoder、损失均未改动。配置生成器 `tools/generate_r18_csp_context_yaml.py` 仅替换原始 YAML 的 Stage3/Stage4 行，保留 Head 的文本和顶层 0–26 层索引。

Stage3 `CSPResidualStage(64,128,2,3,.75)`：原项目 `BasicBlock(64,128,stride=2,shortcut=False,variant='d')` 下采样，1×1 投影后固定分为 96/32 通道，处理支经过两个 96 通道 `BasicBlock(stride=1,shortcut=True)`，拼接后 1×1 融合，并执行 `Z3 + gamma3 * fusion`。Stage4 `CSPContextStage(128,256,2,4,.75,7)`：相同语义的原始下采样块、192/64 分流、两个 `ContextBlock`、1×1 融合及独立 `gamma4`。ContextBlock 为 DWConv7×7+BN、192→384→192 的逐点卷积、GELU 和块内残差；它借鉴大核 Depthwise 的设计思路，不是完整 ConvNeXt Block 复刻。两个 gamma 均使用 `0.3*tanh(raw_gamma)`，初始有效值 0.05，让处理支在第一次反向传播即可得到梯度。

相比原始 Stage3 的两个全通道 BasicBlock，新 Stage3 是一个全通道下采样块加两个部分通道处理块；深度和成本并不相同。新 Stage4 用部分通道大核 ContextBlock 代替原始第二个全通道 BasicBlock，保留第一块的下采样/shortcut/BN/激活语义。新结构初始化时不与原始 Baseline 数值等价。

## 配置与层连接

| 配置 | Stage3 (layer 5) | Stage4 (layer 6) | Stage5 (layer 7) | Head |
|---|---|---|---|---|
| `rtdetr_r18_csp_stage3.yaml` | CSPResidual | 原始 | 原始 | 原样 |
| `rtdetr_r18_csp_stage4.yaml` | 原始 | CSPContext 7×7 | 原始 | 原样 |
| `rtdetr_r18_csp_context.yaml` | CSPResidual | CSPContext 7×7 | 原始 | 原样 |
| `rtdetr_r18_csp_stage4_k3.yaml` | 原始 | CSPContext 3×3 | 原始 | 原样 |
| `rtdetr_r18_csp_stage3_ratio05.yaml` | CSPResidual 0.5 | 原始 | 原始 | 原样 |

Decoder 源索引保持 `[19,22,25]`、300 queries。P3/P4/P5 期望分别为 `[B,128,H/8,W/8]`、`[B,256,H/16,W/16]`、`[B,512,H/32,W/32]`；Decoder 输入为对应 256 通道特征。YAML 文本与通道拓扑测试已通过，实际完整模型 Forward Hook 待 Linux 服务器执行。

## 本地测试与复杂度

`tests/test_csp_context_r18.py`：6 项中 4 项通过；完整 Ultralytics 模型项因本机缺 `cv2` 跳过，CUDA AMP 项因本机无 CUDA 跳过。已通过的测试覆盖独立前向、原始下采样行为、96/32 与 192/64 分流、3/5/7 大核、两步梯度、BN 状态与 state_dict 往返、YAML Head 一致性。`tools/train_r18_csp_context.py --model stage3 --device 0 --dry-run` 通过。

在 CPU 上用同一个 PyTorch profiler 对独立 Stage 做 batch1、640 输入对应特征尺寸的测量。原始完整记录见 `reports/csp_context_stage_complexity.json`。这里的 FLOPs 为 profiler 支持算子的**下界**，不是完整模型性能：

| 独立 Stage | 原始参数 | CSP 参数 | 参数变化 | 原始支持 GFLOPs | CSP 支持 GFLOPs |
|---|---:|---:|---:|---:|---:|
| Stage3 | 525,568 | 595,969 | +70,401 | 6.713 | 7.605 |
| Stage4 | 2,099,712 | 1,366,785 | −732,927 | 6.712 | 4.361 |

单 Stage 变化不能代替完整模型参数、MACs、FPS 或 GPU 显存。服务器运行 `benchmark_r18_csp_context.py` 时须对 Baseline 和各新 checkpoint 使用同一 FP32、640、batch1、warmup 和计时范围。参数降低不保证 FPS 提高。

## 训练协议与尚未验证的内容

独立的 `csp_context_default.yaml` 为 150 epoch、640、AdamW、初始 LR 1e-4、`lrf=1`、AMP 关闭、batch/micro-batch 均为8、梯度累积1、有效batch8。原始共享配置仍保留微批量2供其他实验使用。当前 Trainer 将 `warmup_epochs` 用作**微批量迭代数**，因此 CSP 运行的 2000 次微批量迭代就是 2000 次优化器迭代。真实每轮 LR、loss 和验证指标写入 `results.csv`，结束时复制到 `epoch_history.json`。

本地未找到原始 Baseline 的 `args.yaml`、`results.csv` 或 checkpoint，无法核对其全部增强和调度，也无法确认历史 20,083,028 参数统计采用的类别数。训练脚本接受 `--baseline-args` 核对核心设置并沿用其余设置。若历史协议不同，需要同协议 Baseline，不应把旧的 0.707/0.816 作为严格公平对照。没有运行完整训练、整网 benchmark、CUDA/AMP 或检测 AP；精度收益未知。
