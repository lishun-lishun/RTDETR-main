# V3 方法新颖性初判（2026-11-17）

| 候选 | 最近工作 | 实质差异与风险 | 实验决定条件 |
| --- | --- | --- | --- |
| GAP | [Bian 等 CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bian_Feature_Information_Driven_Position_Gaussian_Distribution_Estimation_for_Tiny_Object_CVPR_2025_paper.html) 已提出位置高斯分布图；[RT-DETRv3 WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Wang_RT-DETRv3_Real-Time_End-to-End_Object_Detection_with_Hierarchical_Dense_Positive_Supervision_WACV_2025_paper.html) 已用训练期密集正样本监督 | 本实现直接监督已有 Encoder 全位置分类 logits，不增加检测头。高斯标注本身不新；若 Top300 GT 覆盖率已接近饱和，论文动机不足 | 测 GT 覆盖率、GAP 对 Encoder 分类头梯度及三种子 Val AP |
| HNR | [DEIM CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html) 提升匹配数量和质量；原 RT-DETR VFL 已压低未匹配 Query | 仅追加正 Query 与远离 GT 的困难背景排序；没有新匹配机制，可能与 VFL 重复 | 统计误检类型、困难背景分数及 Recall 变化 |
| PLM | [D-FINE ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html) 的 GO-LSD 已处理跨层定位知识 | 本实现只是固定 Query 的最终层单调误差正则，没有分布细化。重复风险高 | 先统计末层比中层差的比例；若很低则淘汰 |
| BQR | [UGS ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html) 针对小目标定位梯度不稳 | 本实现对 GT 做固定半像素偏移后平均 GIoU；无显式标注不确定度估计，学术新颖性弱 | 测标注量化误差、AP75/85/90 和梯度稳定性 |

[UAV-DETR 2026](https://arxiv.org/abs/2603.22841) 已报告 DUT-Anti-UAV，重点是骨干、注意力、融合及 Inner-CIoU/NWD；[STRUAV-DETR 2026](https://ieeexplore.ieee.org/abstract/document/11684786/) 也在 DUT-Anti-UAV 评估；[TQ-DETR 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667564/) 在 VisDrone 而非本任务，采用目标引导 Query 分配。因此 V3 不宜主张“首次改进 RT-DETR 小目标监督”。当前全部只是待验证假设，不能称为论文贡献。
