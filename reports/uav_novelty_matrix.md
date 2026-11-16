# UAV RT-DETR R18 候选机制查新（2026-11-16）

本表比较**具体机制**，不把相关领域论文等同于相同实现。结论是进入服务器消融前的风险判断，不是性能或可发表性证明。当前代码基于 `9636869af908610c7aa84c4cfca3f2e33479af41`。

| 候选模块 | 相关原始论文与实际机制 | 重叠风险与本实现差异 | 必需消融 / 论文定位 |
|---|---|---|---|
| M1 `CenterRingEvidence` | [QueryDet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html) 由粗定位稀疏引导高分辨率检测；[TQ-DETR, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667564/) 使用目标引导 Query 分配与边界感知多尺度融合。 | **高**：稀疏读取 P2 细节、边界信息均有相近动机。本实现的九点框相对采样与中心/外围差分发生在最终 Query 之后，且不增加 Transformer 尺度；尚未检索到与此九点公式完全相同的论文，但不能据此声称原创。 | E1 对 E2、E6 对 E7、E2 对 E7；同时看 AP75、中心误差、延迟。只有真实证据稳定胜过零证据，才考虑作为组合贡献。 |
| M2 `BoundedGeometryRefiner` | [D-FINE, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html) 以分布精细化及自蒸馏改善定位；[UGS, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html) 用离散定位、梯度稳定及不确定区域细化。 | **高**：框残差细化已有大量先例。本实现是最终层后处理的像素有界残差，不是 D-FINE 的分布回归，也没有 UGS 的不确定性机制。独立新颖性弱。 | E0 对 E1；若 E1 不改善，淘汰 M2；若仅 M1+M2 改善，M2 作为证据利用器。 |
| M3 `QualityReranker` | [RT-DETR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html) 已使用 IoU 感知 Query 选择；本仓库原 Loss 的 VFL 已用匹配 IoU 作为分类软标签。 | **高**：当前只是对所有类别加同一个 Query logit 残差，借原 VFL 间接训练；没有独立 IoU 质量标签或质量头，不能称“质量估计创新”。 | E0 对 E3；测评分与 IoU 的 Spearman、排序错误及 AP，而非仅看 AP50。无稳定增量则淘汰。 |
| M4 `SizeAwareRTDETRDetectionLoss` | [UGS, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html) 研究小目标定位梯度；[RT-DETRv3, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Wang_RT-DETRv3_Real-Time_End-to-End_Object_Detection_with_Hierarchical_Dense_Positive_Supervision_WACV_2025_paper.pdf) 加密集正样本监督；[DEIM, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html) 改进 O2O 匹配。 | **中高**：尺度敏感几何损失不是新方向。本实现保留原 Hungarian、L1、GIoU，只加以 GT 宽高归一化的中心和 log 尺度 SmoothL1；与 v3/DEIM 的密集匹配机制不同。 | E0 对 E4；比较多尺度 AP 和中心误差。更适合作为辅助训练基线，不宜单列主贡献。 |

## 2026 年 UAV 方向近作

- [FA-DETR, Neurocomputing 2026](https://www.sciencedirect.com/science/article/abs/pii/S0925231226010301)：特征增强与自适应融合，靠近多尺度细节方向；需避免把“加入浅层特征”本身当贡献。
- [DG-DETR, Image and Vision Computing 2026](https://www.sciencedirect.com/science/article/pii/S0262885626002970)：密度引导动态 Query，侧重 Query 预算，与固定 300 Query 路线不同。
- [TQ-DETR, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667564/)：目标引导动态 Query 和边界感知融合，是 M1/P2 最接近的 UAV 方向参照。
- [MSFE-DETR, Scientific Reports 2026](https://www.nature.com/articles/s41598-026-69495-2)：RT-DETR-R18 的多尺度特征增强 UAV 检测。
- [UAV-DETR, arXiv 2026 预印本](https://arxiv.org/abs/2603.22841)：面向 DUT-ANTI-UAV 的 WTConv 主干与滑窗编码器；跨论文指标仅作背景，不与本仓库历史数值直接比较。

## 当前判断

尚无证据可把四个模块都写成独立论文创新。优先验证“最终 Query 引导的稀疏 P2/P3 局部证据能否改善小目标定位且保持合理延迟”这一可检验命题；有增量后再看有界修正是否必要。M3、M4 先视为辅助消融。若证据采样相对 Query-only 无增量，应删除 M1 的方法主张；若 M3/M4 无稳定 Val 增量，应从最终方法中移除。此处“淘汰”是实验决策，当前尚未训练，不能预先判负。
