# Anti-DETR 与 DUT-Anti-UAV 对照实验

本实验套件包括官方原始 RT-DETR-R18 基线，以及作者仓库 `anti-detr/cfg` 发布的
全部 5 个消融配置。Anti-DETR 论文为 *Anti-DETR: End-to-end anti-drone visual
detection network based on wavelet convolution*（Neurocomputing 674, 2026，
DOI: `10.1016/j.neucom.2026.132935`）。

## 数据与模型

服务器数据路径已配置为：

```text
/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/DUT-Anti-UAV/DUT-Anti-UAV
```

数据配置文件是 `dataset/DUT-Anti-UAV.yaml`。数据集保持官方划分：5,200 张训练图、
2,600 张验证图、2,200 张测试图，类别为 `UAV`。

实验名称与模型配置：

| 实验名 | 配置 |
| --- | --- |
| `original-r18` | 原始 `rt-detr/rtdetr-r18.yaml` 基线 |
| `anti-r18` | 作者发布的 R18 对照配置 |
| `anti-wtconv` | GWRN/WTConv 消融配置 |
| `anti-hifi` | HIFI 消融配置 |
| `anti-mrfpn` | MRFPN 消融配置 |
| `anti-p2` | P2 四尺度消融配置 |

作者 YAML 中的 `nc: 80` 被原样保留；训练器读取 DUT 数据后会自动以 `nc: 1`
构建检测头。因此模型配置保持作者默认值，实际训练输出仍是单类别。

## 一致性结论与固定实验参数

当前工程能够严格保持作者仓库公开的 5 份模型 YAML，并让六组实验使用完全相同的
数据划分和训练参数；但不能称为论文完整模型的严格复现。原因见本文最后的“源仓库
缺失项说明”。尤其是作者没有发布论文训练超参数文件，因此
`ultralytics/cfg/anti_detr_default.yaml` 固定的是本 RTDETR-main 分支中与
RT-DETR/Paddle 对齐的默认协议，而不是冒充论文原始超参数。

| 参数 | 值 |
| --- | --- |
| epochs / batch / image size | `100 / 4 / 640` |
| optimizer | `AdamW` |
| initial LR / final LR factor | `0.0001 / 1.0` |
| momentum / weight decay | `0.9 / 0.0001` |
| warmup | `2000` iterations |
| AMP / mosaic / mixup | `False / 0 / 0` |
| seed / deterministic | `0 / True` |
| max detections / queries | `300 / 300` |

训练器还采用与 Paddle RT-DETR 对齐的 `max_norm=0.1` 梯度裁剪，并将名义 batch
设为实际 batch，避免梯度累积改变各实验的有效 batch。

所有模型当前都从 YAML 随机初始化。此分支对 YAML 新建模型使用
`pretrained: true` 时并不会自动加载任何 checkpoint，因此配置中明确写为
`pretrained: false`。如果论文实际使用了预训练初始化，必须取得作者使用的具体
checkpoint 后通过 `INIT_WEIGHTS=/path/to/weights.pt` 显式指定，不能仅用布尔值代替。

## 运行

先检查将要运行的模型和参数（不会加载 PyTorch）：

```bash
python train_dut_anti_uav.py --mode list
python train_dut_anti_uav.py --mode train --models all --device 0 --dry-run
```

### 推荐：GPU 1、2、3 并行实验队列

这个模式同时运行最多 3 个实验，每个实验独占一张卡：第一批按顺序分配
`original-r18 -> GPU 1`、`anti-r18 -> GPU 2`、`anti-wtconv -> GPU 3`，全部完成后
再启动第二批 `anti-hifi`、`anti-p2`、`anti-mrfpn`。因为每个实验仍保持单卡
`batch=4`、`lr0=1e-4` 和 `warmup=2000`，它最适合要求结果与单卡实验接近的场景。

```bash
GPU_LIST=1,2,3 bash scripts/train_r18_3gpu_queue.sh
```

各实验日志单独写入 `runs/anti-detr/logs/<时间>/<模型>_gpu<编号>.log`，并行输出不会
互相混杂。如果一批中任一实验失败，脚本等待该批其他实验退出后停止，不会继续启动
下一批。

### 可选：三卡 DDP 顺序训练

这个模式让 GPU 1、2、3 共同训练一个模型，六个模型仍逐个顺序执行：

```bash
DEVICE=1,2,3 bash scripts/train_r18_3gpu_ddp_sequence.sh
```

默认全局 batch 为 12，即每卡 batch 为 4；warmup 从 2000 次缩放为 667 次，使其
覆盖的样本数与单卡接近。DDP 会减少每个 epoch 的优化器更新次数，因此即使速度更快，
结果也不可能与单卡数值完全一致。需要最公平的单卡对照时，应优先使用上面的三卡实验
队列。为保证稳定性，DDP 模式仍关闭 AMP，并保持 `lr0=1e-4`。

DDP 临时入口会把当前项目根目录放到 `sys.path` 和 `PYTHONPATH` 首位，确保三个 rank
加载本项目集成了 `ConvNormLayer`、`HIFI`、`MCAM` 等模块的源码，而不是 conda 环境中
另行安装的官方 `ultralytics` 包。

### 其他运行方式

训练一个实验：

```bash
DEVICE=0 bash scripts/train_r18_one.sh anti-hifi
```

每个模型使用独立 Python 进程，依次训练基线及全部 5 个作者配置：

```bash
DEVICE=0 bash scripts/train_r18_sequence.sh
```

如果只运行作者仓库中有真实模块实现的配置、排除使用兼容替代层的 MRFPN：

```bash
DEVICE=0 bash scripts/train_r18_released_impls.sh
```

脚本后的参数会继续传给 Python 入口，例如临时修改 workers：

```bash
DEVICE=0 bash scripts/train_r18_sequence.sh --workers 8
```

若有确定的初始化权重，可以设置 `INIT_WEIGHTS`。它会逐个对各结构执行按名称和形状
匹配的权重迁移；不同结构无法匹配的层仍会随机初始化。该参数适用于单卡和三卡实验
队列，不适用于生成临时入口的 DDP 模式：

```bash
INIT_WEIGHTS=/path/to/rtdetr-r18.pt DEVICE=0 \
  bash scripts/train_r18_sequence.sh
```

在官方 test split 上依次测试全部模型。脚本自动寻找各实验最新的 `best.pt`：

```bash
DEVICE=0 SPLIT=test bash scripts/test_r18_sequence.sh
```

汇总指标保存在 `runs/anti-detr/test/test_summary.csv`。如需临时改变显存相关参数，
可以传入 `--batch`、`--imgsz` 或 `--workers`；正式论文对照实验应保持默认值。

## 源仓库缺失项说明

- `HIFI` 复用目标项目已有的 `TransformerEncoderLayer_DHSA`；经 AST 比对，两份
  实现一致，输入输出均保持 `BCHW -> BCHW`。
- 作者仓库的 `rtdetr-MRFPN.yaml` 使用 `MCAM`，论文将其描述为 AGCAU，但仓库
  所有提交中都没有提供该模块实现。当前 `MCAM` 是基于目标工程 `SBA` 的可运行
  接口兼容层，能保持 `[BCHW, BCHW] -> BCHW` 的 batch、尺度和通道契约，但不能
  声称复现论文 AGCAU 的内部算法。要严格复现该项，需要作者补充 MCAM/AGCAU
  源码或论文中的完整结构公式。
- 论文描述的 GWRN 包含全局多通道注意力；作者代码虽然定义了 `MPA`，发布的
  `rtdetr-WTConv.yaml` 和 `BasicBlock_WTConv` 却没有连接该注意力模块。
- 作者仓库没有发布把 GWRN、MRFPN 和 HIFI 合并在一起的最终 `anti-detr.yaml`，
  也没有训练脚本、初始化 checkpoint 或论文超参数文件。因此这里可以严格复现
  作者公开的 5 份 YAML 和目标工程默认训练协议，但在缺少上述源码时不能宣称严格
  复现论文的完整最终 Anti-DETR 网络或论文数值结果。
