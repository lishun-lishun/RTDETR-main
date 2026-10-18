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
缺失项说明”。`ultralytics/cfg/anti_detr_default.yaml` 中的 epochs、训练和测试 batch、
图像尺寸、workers、优化器、初始学习率、momentum 与 weight decay 按用户提供的论文
实验表设置；表中未给出的 warmup、数据增强、初始化等仍是本工程配置，不能据此保证
论文结果完全复现。该脚本目前只使用 DUT-Anti-UAV；Det-Fly 需要单独的数据配置和实验。

| 参数 | 值 |
| --- | --- |
| epochs / train batch / test batch / image size | `150 / 8 / 16 / 640` |
| workers | `4` |
| optimizer | `AdamW` |
| initial LR / final LR factor | `0.0001 / 1.0`（后者是本工程设置） |
| momentum / weight decay | `0.9 / 0.0001` |
| warmup | `2000` iterations（本工程设置） |
| AdamW bias warmup 初始学习率 | `0`（修正原先的 `0.1`，本工程设置） |
| AMP / mosaic / mixup | `False / 0 / 0` |
| seed / deterministic | `0 / True` |
| max detections / queries | `300 / 300` |

训练器采用 `max_norm=0.1` 梯度裁剪。A30 24 GB 上以 batch 8 和 micro batch 4
训练均已触发 CUDA 显存不足，因此默认每次前向使用 `micro_batch=2`，累积四次梯度后更新一次，
有效训练 batch 仍为 8。warmup 的微批次数相应变为 8000，使其覆盖的样本数与原先
batch 8、2000 次更新相近。梯度累积与单次 batch 8 的数值计算不会完全相同。

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
再启动第二批 `anti-hifi`、`anti-p2`、`anti-mrfpn`。队列脚本明确传入单卡实际 batch 8，
梯度累积 1 次，保持有效训练 batch 8 和 `lr0=1e-4`；这与默认配置文件里为显存不足
预留的 micro batch 2 不同。启动脚本会在导入 PyTorch
前设置 `CUDA_VISIBLE_DEVICES`，保证每个进程绑定到指定的物理 GPU。

```bash
GPU_LIST=1,2,3 bash scripts/train_r18_3gpu_queue.sh
```

各实验日志单独写入 `runs/anti-detr/logs/<时间>/<模型>_gpu<编号>.log`，并行输出不会
互相混杂。如果一批中任一实验失败，脚本等待该批其他实验退出后停止，不会继续启动
下一批。A30 24 GB 上单卡实际 batch 8 尚未验证能否容纳；如果发生 OOM，需改用
`--micro-batch 2` 等梯度累积方案（有效 batch 仍为 8），或使用更大显存的 GPU。

### 三卡 DDP 顺序训练（一次只训练一个实验）

这个模式让 GPU 1、2、3 共同训练一个模型，六个模型仍逐个顺序执行：

```bash
DEVICE=1,2,3 bash scripts/train_r18_3gpu_ddp_sequence.sh
```

按 batch 24 的要求，默认每次前向总 micro batch 12（每卡 4），累积两次梯度后
更新一次。warmup 从本工程 batch 8 的 2000 次缩放为 667 次更新，使其覆盖的
训练样本数接近 batch 8 的设置。仍保持 `epochs=150`、`lr0=1e-4`、AdamW、
`momentum=0.9`、`weight_decay=1e-4`、`imgsz=640` 和关闭 AMP。
为改善后期收敛，这个 batch-24 预设启用 cosine 学习率衰减，末期学习率为初始值的
`0.1`；这是工程调参，不是论文已证实的最优设置。batch 24 每轮的优化器更新次数
约为 batch 8 的三分之一，不能保证复现论文结果。训练进度实时显示在终端，同时每个
实验单独保存到 `runs/anti-detr/logs/ddp_<时间>/<模型>.log`；前一个实验成功后才
启动下一个，失败时停止。如果某个模型仍显存不足，可用
`MICRO_BATCH=6 DEVICE=1,2,3 bash scripts/train_r18_3gpu_ddp_sequence.sh`，
此时每卡 micro batch 为 2、累积 4 次，有效全局 batch 仍为 24，但速度会降低。
需要严格使用全局 batch 8 时，应使用上面的三卡实验队列。

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

测试 batch 默认 16；汇总指标保存在 `runs/anti-detr/test/test_summary.csv`。
如需临时改变参数，可传入 `--batch`（有效训练 batch）、`--micro-batch`（实际训练
batch）、`--test-batch`（测试）、`--imgsz` 或 `--workers`。若 micro batch 2 仍然
超出显存，可用 `--micro-batch 1`，有效 batch 继续保持 8，但训练会更慢；
还应使用 `nvidia-smi` 检查卡上是否有其他进程占用显存。

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
  也没有训练脚本或初始化 checkpoint。这里按用户提供的论文实验表设置主要超参数，
  但在缺少上述源码与其余训练细节时，不能宣称严格复现论文完整网络或数值结果。
