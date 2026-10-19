# RT-DETR 模型库的 DUT-Anti-UAV 实验入口

`ultralytics/cfg/models/rt-detr/` 中的 428 份 YAML 是通用模型定义，不是数据集配置。
它们原有的 `nc: 80` **不需要批量改成 1**：本项目的 RT-DETR 训练器读取
`dataset/DUT-Anti-UAV.yaml` 后，以 `nc: 1` 重新构建检测头。保留原始 YAML
便于与上游比较，也避免破坏 COCO 权重的结构检查。

现在可用 `--rt-detr-config` 选择该目录中的任意一个或多个 YAML 文件名；入口沿用
DUT 数据路径、150 epochs、640 图像、AdamW、有效训练 batch 8 和测试 batch 16。
例如（先做不加载 PyTorch 的参数检查）：

```bash
python train_dut_anti_uav.py --mode train --rt-detr-config rtdetr-p2.yaml --device 1 --dry-run
python train_dut_anti_uav.py --mode train --rt-detr-config rtdetr-p2.yaml --device 1
python train_dut_anti_uav.py --mode test --rt-detr-config rtdetr-p2.yaml --device 1 --split test
```

默认训练是有效 batch 8、实际 micro batch 2、梯度累积 4；若确认单卡显存足够且需
严格一次前向 batch 8，可给训练命令加 `--batch 8 --micro-batch 8`。部分特殊 YAML
依赖自定义 CUDA 算子、额外 Python 包或预训练骨干，静态 YAML 可解析不代表服务器
上一定能构建或训练；应先单模型试跑。新实验存放在
`runs/rtdetr-dut/train/rt-detr-<配置名>/`，测试汇总写到
`runs/rtdetr-dut/test/test_summary.csv`，不会覆盖此前的 Anti-DETR 实验。

## 结构分类

运行 `python scripts/catalog_rtdetr_models.py` 会更新
`RTDETR_DUT_MODEL_CATALOG.csv`，逐一列出 428 个配置、参考基线、改动层编号和
`backbone`、`encoder`、`neck`、`decoder` 标签。分类只比较 YAML 结构，不代表
精度结论，也不检查算子能否在服务器构建。这里将 `head` 的第 2 层作为编码器、
最后一层作为检测解码器，其余融合层归入 neck；不能简单把 YAML 的整个 `head`
段都叫作 neck。

| 配置 | 主要改动 | 用途 |
| --- | --- | --- |
| `rtdetr-r18.yaml` | R18 基线 | 所有消融的共同对照 |
| `rtdetr-RFAConv.yaml` | 仅 backbone | 测试空间感受野卷积 |
| `rtdetr-WDBB.yaml` | 仅 backbone | 测试 WDBB 骨干块 |
| `rtdetr-AIFI-DHSA.yaml` | 仅 encoder | 测试直方图注意力，不应误称 neck |
| `rtdetr-RFPN.yaml` | 仅 neck | 测试三尺度特征融合 |
| `rtdetr-p2.yaml` | neck + decoder 输入 | 增加高分辨率 P2 检测尺度 |
| `rtdetr-WTConv.yaml` | backbone + neck | 同时换骨干块和 neck 下采样，不能当纯 backbone 消融 |
| `rtdetr-r50.yaml` | 更大基线架构 | 与 R18 算力和参数量不同，不宜混为单模块改进 |

## 建议实验顺序

1. `rtdetr-r18.yaml`：先取得相同训练设置下的对照结果。若已有完全相同的结果，可以复用。
2. `rtdetr-p2.yaml`：DUT 的小目标场景值得优先测试 P2，但高分辨率分支会增加显存和计算量。
3. `rtdetr-RFPN.yaml`：单独检验 neck 融合，便于判断 P2 增益是否来自多尺度处理。
4. `rtdetr-AIFI-DHSA.yaml`：单独检验 encoder 注意力，不与 neck 混淆。
5. `rtdetr-RFAConv.yaml`：单独检验 backbone 改动。
6. `rtdetr-WTConv.yaml`：最后测试骨干与 neck 联合小波改动；如有增益，再拆分消融。

优先用验证集 mAP50-95 选模型，再对锁定的模型在 test 集上只评估一次。保持数据划分、
种子、预训练初始化、学习率日程和 batch 完全一致，否则不能把差异归因于模块。
