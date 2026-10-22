# PHR-ResNet18 实验报告（骨架）

> 本文档为实验记录骨架，**所有数值待服务器实测后填写**。当前仅提供记录结构与对照基准，不预设任何性能结论。

---

## 1. 实验目标

在 DUT-Anti-UAV 数据集上评估 PHR（Partial High-Resolution Residual Backbone）对 RT-DETR-R18 的精度与速度影响，验证「仅改 Backbone、不触碰 Neck / Decoder」下的增益空间。

## 2. 对照基准（Baseline）

> 来源：此前已完成的 RT-DETR-R18 基线训练，**本任务不重训 Baseline**。

| 指标 | 值 |
| --- | --- |
| mAP50 | 0.966 |
| mAP75 | 0.816 |
| mAP50:95 | 0.707 |

## 3. 训练 / 评估配置

- 数据集：`dataset/DUT-Anti-UAV.yaml`（nc=1，UAV）。
- 训练参数：`ultralytics/cfg/anti_detr_default.yaml`（与 Baseline/BAPS/RDCF/LSCE/ADCF 完全一致，见 `phr_r18_implementation.md` §10）。
- 输入尺寸：640×640；评估 max_det=300。

## 4. 精度结果（待填）

| 配置 | mAP50 | mAP75 | mAP50:95 | 权重路径 |
| --- | --- | --- | --- | --- |
| Baseline（复用） | 0.966 | 0.816 | 0.707 | — |
| PHR-1 | 待填 | 待填 | 待填 | `runs/r18_phr1/train/*/weights/best.pt` |
| PHR-2 | 待填 | 待填 | 待填 | `runs/r18_phr2/train/*/weights/best.pt` |
| PHR-3 | 待填 | 待填 | 待填 | `runs/r18_phr3/train/*/weights/best.pt` |

> 按规格书只训练 PHR-1；PHR-2/3 若后续训练再补录。

## 5. 效率结果（待填）

| 配置 | params | GFLOPs | FPS | latency_ms | 设备 |
| --- | --- | --- | --- | --- | --- |
| Baseline | 待填 | 待填 | 待填 | 待填 | — |
| PHR-1 | 待填 | 待填 | 待填 | 待填 | — |
| PHR-2 | 待填 | 待填 | 待填 | 待填 | — |
| PHR-3 | 待填 | 待填 | 待填 | 待填 | — |

> 由 `tools/benchmark_r18_phr.py --model all --device 0` 实测产出（见 `runs/phr_benchmark.csv`）。

## 6. PHR 模块激活诊断（待填）

> 由 `tools/collect_phr_diagnostics.py` 产出（见 `runs/phr_statistics.json`），用于判断 PHR 是否真正学习到贡献而非退化回恒等。

| 配置 | gamma | beta3 | beta4 | residual_ratio_mean |
| --- | --- | --- | --- | --- |
| PHR-1 | 待填 | n/a | n/a | 待填 |
| PHR-2 | 待填 | 待填 | n/a | 待填 |
| PHR-3 | 待填 | 待填 | 待填 | 待填 |

判定标准：`gamma` 显著 > 0 且 `residual_ratio_mean > 0` 表明补偿路径被激活。

## 7. 结论（待填）

- 精度增益：待填。
- 速度/算力代价：待填。
- PHR-1/2/3 消融对比：待填。

---

*所有数值须在服务器 `ultralytics_rtdetr` 环境实测后回填，本骨架不预设任何优越性。*
