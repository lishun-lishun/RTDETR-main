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

## 固定实验参数

`ultralytics/cfg/anti_detr_paper.yaml` 固定了目标工程中与 RT-DETR/Paddle 对齐的
默认协议，六组实验使用完全相同的参数：

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

## 运行

先检查将要运行的模型和参数（不会加载 PyTorch）：

```bash
python train_dut_anti_uav.py --mode list
python train_dut_anti_uav.py --mode train --models all --device 0 --dry-run
```

依次训练基线及全部 5 个 Anti-DETR 配置：

```bash
python train_dut_anti_uav.py --mode train --models all --device 0
```

只训练一个配置：

```bash
python train_dut_anti_uav.py --mode train --models anti-hifi --device 0
```

在官方 test split 上依次测试全部模型。脚本自动寻找各实验最新的 `best.pt`：

```bash
python train_dut_anti_uav.py --mode test --models all --split test --device 0
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
  也没有训练脚本或论文超参数文件。因此这里可以严格复现作者公开的 5 份 YAML
  消融结构和目标工程默认训练协议，但在缺少上述源码时不能宣称严格复现论文的
  完整最终 Anti-DETR 网络。
