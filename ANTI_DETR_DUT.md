# anti-detr 配置与 DUT-Anti-UAV

`anti-detr/cfg` 中的 5 个模型配置已集成到：

```text
ultralytics/cfg/models/anti-detr/
```

这些配置保持原网络的层连接、检测尺度和 RT-DETR 解码器参数，并针对
DUT-Anti-UAV 将类别数设为 1（`UAV`）。模型输入仍为 Ultralytics 标准
`B x 3 x H x W` 张量；所有新增兼容层都保留 batch 维。解码器也沿用原配置
的 300 个 query。

数据配置为 `dataset/DUT-Anti-UAV.yaml`，对应当前工作区中的 5,200 张训练图、
2,600 张验证图和 2,200 张测试图。

## 训练

在 `RTDETR-main` 目录执行：

```powershell
python train_dut_anti_uav.py --model rtdetr-HIFI --batch 4 --device 0
```

可选模型：

- `rtdetr-r18`
- `rtdetr-WTConv`
- `rtdetr-HIFI`
- `rtdetr-p2`
- `rtdetr-MRFPN`

若显存不足，可减小 `--batch`；这只改变 batch 数量，不改变每个样本的输入或
模型输出结构。

## 兼容映射

- `HIFI` 复用目标项目已有的 `TransformerEncoderLayer_DHSA` 实现；两者代码和
  输入输出契约一致。
- 上游 `anti-detr` 的 `rtdetr-MRFPN.yaml` 引用了 `MCAM`，但其 `model.py` 未提供
  该类。本集成将 `MCAM` 显式映射到目标项目的 `SBA` 双尺度融合实现，接口为
  `[BCHW, BCHW] -> BCHW`，并保持 batch 维、目标尺度和目标通道数。
