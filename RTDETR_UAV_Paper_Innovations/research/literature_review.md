# 04 前沿文献调研报告（2023–2026 Literature Review）

## 1. 文献调研范围与综述框架

围绕**可见光复杂背景下的远距离微小无人机（Anti-UAV）实时检测**，本报告系统检索并核对了 2023–2026 年（重点聚焦 2025–2026 年）发表在 CVPR、ICLR、ECCV、AAAI、WACV、IROS、IEEE T-PAMI、IEEE T-ITS、ISPRS、Sensors 及 arXiv 上的 **22 篇代表性论文**，涵盖五大核心方向：
- **方向 A：实时与端到端 Transformer 目标检测**（RT-DETR, RT-DETRv2, RT-DETRv3, D-FINE, DEIM, DEIMv2, DINO）
- **方向 B：微小目标与无人机检测架构**（UAV-DETR 2026, TQ-DETR 2026, ReDHF 2026, ViM-DETR 2026, UAV-DETR 2025, Dome-DETR 2025, DQ-DETR 2024, Drone-DETR 2024）
- **方向 C：复杂背景抑制与空频协同表征**（FSDC-DETR 2026, D3R-DETR 2026, HS-FPN 2025, WTConv 2024）
- **方向 D：训练分配、匹配优化与自蒸馏监督**（NWD, Inner-IoU, MAL, GO-LSD, EGSS）
- **方向 E：反无人机基准与高效推理机制**（DUT-Anti-UAV 基准特性与 Train-time-only 增强）

---

## 2. 22 篇核心前沿文献全量核对表

| 序号 | 论文简称 | 完整标题 | 作者与年份 | 发表渠道 | 论文 / 官方代码链接 | 核心创新机制 | 数据集与关键指标 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **RT-DETR** | DETRs Beat YOLOs on Real-time Object Detection | Zhao et al., 2024 | CVPR 2024 | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html) / [GitHub](https://github.com/lyuwenyu/RT-DETR) | 高效混合编码器（AIFI + CCFM）、Uncertainty-minimal Query Selection、可调解码层数 | COCO val: R18 46.5% AP @ 217 FPS; R50 53.1% AP @ 108 FPS |
| 2 | **RT-DETRv2** | RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer | Lv et al., 2024 | arXiv 2024 | [Paper](https://arxiv.org/abs/2407.17140) / [GitHub](https://github.com/lyuwenyu/RT-DETR) | 分尺度动态采样点数设置、离散采样算子适配、动态数据增强调度（末期关闭强增强） | COCO val: R18 47.9% AP (+1.4%), AP_S 30.3% |
| 3 | **RT-DETRv3** | RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision | Wang et al., 2025 | WACV 2025 | [Paper](https://www.alphaxiv.org/abs/2409.08475) | 训练期专用的 CNN 辅助一对多密集监督分支、自注意力扰动多查询组、共享权重解码分支（推理零开销） | COCO val: R18 48.1% AP (+1.6% over RT-DETR-R18)，延迟完全不变 |
| 4 | **D-FINE** | D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement | Peng et al., 2025 | ICLR 2025 Spotlight | [Paper](https://arxiv.org/html/2410.13842v1) / [GitHub](https://github.com/Peterande/D-FINE) | 细粒度分布精化（FDR，将定点回归转化为四条边离散概率分布的逐层残差更新）+ 全局最优定位自蒸馏（GO-LSD） | COCO val: D-FINE-L 54.0% AP @ 124 FPS; 迁移至各类 DETR 提升 +2.0%~5.3% AP |
| 5 | **DEIM** | DEIM: DETR with Improved Matching for Fast Convergence | Huang et al., 2025 | CVPR 2025 | [Paper](https://arxiv.org/html/2412.04234v3) / [GitHub](https://github.com/Intellindust-AI-Lab/DEIM) | Dense O2O 密集一对一匹配 + **Matchability-Aware Loss (MAL)**（将 VFL 的软标签 $q$ 替换为 $q^\gamma$，解决低质量匹配梯度消失） | COCO val: DEIM-D-FINE-L 54.7% AP；CrowdHuman 上 AP_S 提升 +4.2%，AP75 提升 +3.5% |
| 6 | **DEIMv2** | Real-Time Object Detection Meets DINOv3 | Huang et al., 2025 | arXiv 2025 | [Paper](https://arxiv.org/pdf/2509.20787) | 结合 DINOv3 视觉基础模型特征与轻量级 CNN 的双分支实时检测架构与空间精化模块 | COCO val: 全尺寸 SOTA 性能，但双分支直接拼接存在高低频混叠 |
| 7 | **UAV-DETR (2026)** | UAV-DETR: DETR for Anti-Drone Target Detection | Yang et al. (NWPU), 2026 | arXiv 2026 (Mar 2026) | [Paper](https://arxiv.org/html/2603.22841v1) / [GitHub](https://github.com/wd-sir/UAVDETR) | **专门针对反无人机与 DUT-Anti-UAV**：WTConv 主干块、SWSA-IFI 滑动窗口编码器、ECFRFN（SBA+RepNCSPELAN4）颈部、**InnerCIoU-NWD 混合损失** | **DUT-Anti-UAV**: mAP50=96.17%, mAP75=75.71%, mAP50:95=67.15% (vs RT-DETR 66.67%), Params=11.89M |
| 8 | **TQ-DETR (2026)** | Target-Guided Query Allocation and Boundary-Aware Multi-Scale Feature Fusion for Object Detection in UAV Imagery | Chen et al., 2026 | Sensors 2026 (Sep 2026) | [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667564/) | **Target-Oriented Branch (TOB)** 类别无关前景先验、**Dynamic Query Allocation (DQ, 300+100)**、**OG-MBF ($P_2 \to P_3$ 边界门控残差注入，不增加 Transformer 层数)** | VisDrone2019: AP 30.21% (+1.79%), AP50 50.28% (+2.75%), AP75 30.50% (+2.14%), AP_S 21.06% (+1.81%) |
| 9 | **ReDHF (2026)** | Reviving the Discarded High-Resolution Feature for Transformer-Based Tiny Object Detection | Wang et al., 2026 | arXiv 2026 (Nov 2026) | [Paper](https://arxiv.org/html/2611.07994v1) | 跨尺度可变形融合（CSDF）、**椭圆引导浅层辅助监督（EGSS，训练期专用，推理期移除）**、全尺度查询交互（FSQI） | AI-TOD-v2: 35.7% AP (+1.4%), VisDrone: 40.6% AP (+1.6%), SODA-D: 38.7% AP (+1.5%) |
| 10 | **FSDC-DETR (2026)** | FSDC-DETR: A Frequency-Spatial Domain Collaborative DETR for Small Object Detection | Liu et al., 2026 | arXiv 2026 (Aug 2026) | [Paper](https://arxiv.org/html/2607.05176v3) / [GitHub](https://github.com/nevereverinsomnia/FSDC-DETR) | 双分支频空自适应融合（DBFSAF）、分流频空特征融合（SFS-FF）、**基于 DWT 小波与分组卷积的动态下采样（FSD-Down）** | VisDrone: 31.1% AP (+6.4%), AP_S 21.0% (+6.8%); AI-TODv2: 32.3% AP (+6.6%) |
| 11 | **ViM-DETR (2026)** | ViM-DETR: efficient anti-UAV detection via hybrid vision Mamba and frequency-aware fusion | SPIE 2026 | SPIE Proc. 2026 (May 2026) | [Paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/14181/141810J/ViM-DETR--efficient-anti-UAV-detection-via-hybrid-vision/10.1117/12.3115669.full) | **直接面向 DUT-Anti-UAV**：混合 Vision Mamba 主干、小波反池化与互融合模块（MFM）、**Matchability-Aware Loss (MAL)** | **DUT-Anti-UAV**: Recall 95.88%, mAP50:95 70.24% (+1.29% over baseline) |
| 12 | **D3R-DETR (2026)** | D3R-DETR: DETR with Dual-Domain Density Refinement for Tiny Object Detection | Jan 2026 | arXiv 2026 | [Paper](https://arxiv.org/html/2601.02747v1) | 空洞卷积空间上下文与分数阶 Gabor 核频域特征协同提取（D2FM），重构高精度目标密度图引导查询生成 | AI-TOD-v2 上显著超越现有微小目标 DETR 基线 |
| 13 | **UAV-DETR (2025)** | UAV-DETR: Efficient End-to-End Object Detection for Unmanned Aerial Vehicle Imagery | Zhang et al. (Fudan), 2025 | IEEE/RSJ IROS 2025 | [Paper](https://arxiv.org/html/2501.01855v3) / [GitHub](https://github.com/ValiantDiligent/UAV-DETR) | 多尺度频域增强融合（MSFF-FE）、频域聚焦下采样（FD）、**基于可学习 2D 偏移与 GridSample 的语义对齐校准（SAC）**、Inner-SIoU 损失 | VisDrone: R18 29.8% AP (+3.1%), AP50 48.8% (+4.2%); UAVVaste: 37.0% AP |
| 14 | **HS-FPN (2025)** | HS-FPN: High Frequency and Spatial Perception FPN for Tiny Object Detection | Shi et al., 2025 | AAAI 2025 | [Paper](https://arxiv.org/html/2412.10116v2) / [GitHub](https://github.com/ShiZican/HS-FPN) | 高频感知模块（HFP，高通滤波生成空间与通道掩码突出微小目标）+ 空间依赖感知模块（SDP，跨层级像素级相关性对齐） | AI-TOD: Faster R-CNN +2.0% AP, Cascade R-CNN +3.4% AP |
| 15 | **Dome-DETR (2025)** | Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection | Hu et al. (USTC), 2025 | ACM MM 2025 | [Paper](https://www.researchgate.net/publication/391658551_Dome-DETR_DETR_with_Density-Oriented_Feature-Query_Manipulation_for_Efficient_Tiny_Object_Detection) | 轻量级密度聚焦提取器（DeFE）、掩码窗口稀疏注意力（MWAS）、渐进自适应查询初始化（PAQI） | AI-TOD-v2 +3.3% AP, VisDrone +2.5% AP |
| 16 | **DQ-DETR (2024)** | DQ-DETR: DETR with Dynamic Query for Tiny Object Detection | Huang et al., 2024 | ECCV 2024 | [GitHub](https://github.com/hoiliu-0801/dq-detr) | 分类计数模块动态决定查询数量（如 300/500/900/1500）+ 密度图增强编码器位置先验 | AI-TOD-v2: 30.2% AP，但面向高密度多目标，计算开销高达 1805 GFLOPs |
| 17 | **Drone-DETR (2024)** | Drone-DETR: Efficient Small Object Detection for Remote Sensing Image Using Enhanced RT-DETR Model | Kong et al., 2024 | Sensors 2024 | [Paper](https://www.mdpi.com/1424-8220/24/17/5496) | 引入浅层 P2 特征层至颈部、ESDNet 与双路径特征融合注意力（EDF-FAM） | VisDrone: AP50 53.9% (+8.1%)，但参数量增至 28.7M，计算量高达 128.3 GFLOPs |
| 18 | **BAP-DETR (2025)** | BAP-DETR: Efficient drone object detection network based on bipartite attentive processing and dual fusion encoder | Wang et al., 2025 | CVIU 2025 | [DOI](https://doi.org/10.1016/j.cviu.2025.104565) | 二分注意力处理与双融合编码器，增强无人机小目标多尺度上下文交互 | VisDrone 与无人机数据集显著提升小目标召回 |
| 19 | **WTConv (2024)** | Wavelet Convolutions for Large Receptive Fields | Finder et al., 2024 | ECCV 2024 | [Paper](https://arxiv.org/abs/2407.05848) | 利用级联 2D Haar 离散小波变换（DWT）与小核深度可分离卷积在对数级扩展感受野的同时分离高低频分量 | ImageNet / COCO 上以极少参数量大幅提升结构形状响应 |
| 20 | **NWD (2022)** | A Normalized Gaussian Wasserstein Distance for Tiny Object Detection | Wang et al., 2022 | ISPRS J. P&RS 2022 | [Paper](https://www.alphaxiv.org/abs/2110.13389) | 将边界框建模为二维高斯分布 $\mathcal{N}(\mu, \Sigma)$，用归一化二阶 Wasserstein 距离替代 IoU，消除微小目标的位置偏移阶跃敏感性 | AI-TOD 上相比标准 IoU 分配与损失提升 +6.7 AP |
| 21 | **Inner-IoU (2023)** | Inner-IoU: More Effective Intersection over Union Loss with Auxiliary Bounding Box | Zhang et al., 2023 | arXiv 2023 | [Paper](https://arxiv.org/abs/2311.02877) | 通过缩放因子 $r \in [0.5, 1.5]$ 构造辅助内部/外部边界框计算 IoU，调节高/低 IoU 样本回归梯度幅值 | 对微小目标采用 $r > 1$ 放大梯度，对高 IoU 样本采用 $r < 1$ 加速精细收敛 |
| 22 | **DUT-Anti-UAV (2022)** | Vision-Based Anti-UAV Detection and Tracking | Zhao et al., 2022 | IEEE T-ITS 2022 | [Paper](https://www.researchgate.net/publication/360804679_Vision-based_Anti-UAV_Detection_and_Tracking) / [GitHub](https://github.com/wangdongdut/DUT-Anti-UAV) | 构建 10,000 张可见光反无人机检测数据集（Train 5200 / Val 2600 / Test 2200，共 2245 个测试实例），揭示小尺度与复杂天际线背景瓶颈 | 基线 Faster R-CNN / YOLOv5 / DETR 在复杂背景与极小尺度下漏检严重 |

---

## 3. 八篇核心代表作深度剖析与局限性批判

### 3.1 UAV-DETR (NWPU, arXiv 2026.03) —— 反无人机 RT-DETR 的直接竞品
- **核心机制**：在 RT-DETR-R18 上进行四处改造：(1) 将 ResNet18 BasicBlock 替换为级联 `WTConv Block`；(2) 将 AIFI 替换为滑动窗口自注意力 `SWSA-IFI`；(3) 用包含选择性边界聚合（`SBA`）和 `RepNCSPELAN4` 的 `ECFRFN` 替换 CCFM，引入 P2–P5 四尺度；(4) 采用固定权重的 `InnerCIoU + NWD` 混合损失。
- **在 DUT-Anti-UAV 上的表现**：`mAP50:95` 从 66.67% 提升至 67.15%（+0.48%），`mAP75` 从 75.50% 提升至 75.71%（仅 +0.21%），而计算量从 56.9 GFLOPs 增加到 **65.2–66.7 GFLOPs（+17.2% 开销）**。
- **深刻局限性（我们的突破口）**：
  1. **替换主干破坏预训练且损伤语义**：其 Table 4 消融明确暴露，引入 `WTConv` 主干（Model O+A+B+C）导致 `mAP50:95` 从 60.61% **下跌至 59.19%（-1.42%）**！在仅 5,200 张训练图的 DUT-Anti-UAV 上大改主干极易导致过拟合或欠收敛。
  2. **忽视了 Decoder 内部的采样点坍缩与 Query 选择缺陷**：UAV-DETR 完全没有修改 `RTDETRDecoder`，因此在 DUT-Anti-UAV 上的 `mAP75` 仅提高了微不足道的 0.21%。
  3. **静态混合损失系数 $\lambda$**：`InnerCIoU` 与 `NWD` 采用全尺度固定加权，未根据目标真实像素面积动态调节，且未同步改进 `HungarianMatcher` 的代价矩阵！

### 3.2 TQ-DETR (Sensors 2026.09) —— 目标引导查询分配与 P2$\to$P3 边界融合
- **核心机制**：(1) 提出 `OG-MBF`，通过局部卷积、膨胀卷积与 $X - \text{AvgPool}_{3\times 3}(X)$ 高频边缘增强提取 $P_2$ 特征，经粗略目标显著图 $M^{(0)}$ 门控后以可学习残差系数 $\gamma$（初始化为 0.1）注入 $P_3$，**不增加 Transformer 编码尺度（保持 8400 tokens）**；(2) 提出 `TOB`（$13\times 13$ DWConv 预测前景概率图 $M$）对 $P_3^*, P_4, P_5$ 进行约束残差重标定 $F_k' = F_k \odot (1 + \alpha_k R_k(M))$（$\alpha_{\max}=0.2$）；(3) 提出 `DQ` 动态查询分配，将 300 个语义查询扩展为 300+100=400 个查询。
- **深刻局限性（我们的突破口）**：
  1. **增加 Query 数量至 400 导致帧率暴跌 35%**：TQ-DETR Table 8 显示，将查询数从 300 增至 400 并执行在线去重补齐（`Dedup` + `Fill`）使推理速度从 81.39 FPS 暴跌至 52.88 FPS，且破坏了本任务要求的 **300 Queries 输出契约**。
  2. **面向密集航拍（平均每图 48.4 个目标）而非稀疏反无人机（每图 1~2 个目标）**：在 DUT-Anti-UAV 中每张图通常只有 1 个微小无人机，300 个 Query 预算绰绰有余；真正的问题不是“300 个不够分”，而是“单靠分类响应评分无法区分低对比度真目标与高纹理背景伪影”。
  3. **各向同性平均池化高通滤波对方向性杂波无选择性**：$X - \text{AvgPool}_{3\times 3}(X)$ 会同等地放大水平云层边缘与建筑直线，不如 2D 离散小波（DWT）能够显式分离水平（LH）、垂直（HL）与对角高频分量（HH，微小多旋翼无人机的核心各向异性局部极值特征）。

### 3.3 D-FINE (ICLR 2025 Spotlight) —— 细粒度分布精化与全局最优定位自蒸馏
- **核心机制**：(1) **FDR**：将 Decoder 每层的边界框偏移 $\Delta b$ 重新定义为四条边（上下左右）在 $N=32$ 个非均匀离散桶上的概率分布 $\mathbf{Pr}^l(n) = \text{Softmax}(\Delta \text{logits}^l + \text{logits}^{l-1})$，捕捉边界模糊与定位不确定性；(2) **GO-LSD**：将最后一层 Decoder 精化后的高质量定位结果作为软教师，自蒸馏监督前序 Decoder 层，并提出 **Target Gating Layer** 替代交叉注意力后的简单残差相加。
- **深刻局限性（我们的突破口）**：
  1. **FDR 在单类别微小目标上存在“相对尺度量化过粗”问题**：D-FINE 的偏移量按初始框宽高 $\{H, H, W, W\}$ 缩放，且 $4 \times (N+1) = 132$ 维回归头改变了原始 `dec_bbox_head` 的 4 维输出签名与 Checkpoint 权重格式。
  2. **启发改进**：我们可以在**严格保持 4 维边界框输出接口与原始 Checkpoint 兼容**的前提下，提炼 D-FINE 的两大精髓：(a) 在 Decoder 层间引入轻量级**尺度感知边界不确定性残差门控与自适应采样扩张**；(b) 在 `RTDETRDetectionLoss` 中引入**零推理开销的跨层定位自蒸馏（Cross-Layer Localization Self-Distillation）**，用第 3 层（末层）的高精度预测框与不确定性分布直接指导 Encoder 预测头与第 1、2 层 Decoder！

### 3.4 DEIM (CVPR 2025) & ViM-DETR (SPIE 2026) —— 匹配质量感知损失 (MAL)
- **核心机制**：指出 DETR 中 `VarifocalLoss (VFL)` 的致命缺陷：当正样本匹配质量较低（如微小目标初期 $\text{IoU} = 0.05$）时，VFL 的软标签 $q = \text{IoU} = 0.05$ 极小且乘在正样本项前，导致正样本梯度几乎为 0。DEIM 提出 **MAL**，将软标签改为 $q^\gamma$ 并移除前景权重衰减，使低 IoU 正样本也能获得充足的分类-定位联合拉动梯度；ViM-DETR 在 DUT-Anti-UAV 上验证了 MAL 的有效性。
- **深刻局限性（我们的突破口）**：
  - DEIM 的 MAL 仍然使用**标准点集 IoU** 来计算质量分数 $q$！对于 $6\times 6$ 像素的无人机，当预测框偏离 6 个像素（完全无交集，$\text{IoU}=0$）时，即便取 $q^\gamma$，仍然有 $0^\gamma = 0$，依然会被当作纯负样本惩罚！
  - **我们的实质创新**：将 **尺度自适应归一化 Wasserstein 距离（NWD）与 Inner-IoU 的连续质量流形** 注入软标签构造与 Hungarian Matcher 代价矩阵中，定义广义连续匹配质量 $q_{\text{hybrid}} = \beta(s)\cdot \text{IoU} + (1-\beta(s))\cdot \text{NWD}$，从根本上消灭 $\text{IoU}=0$ 时的软标签死区！

### 3.5 UAV-DETR (Fudan, IROS 2025) & FSDC-DETR (arXiv 2026.08) —— 空频协同与可变形对齐
- **核心机制**：UAV-DETR (2025) 提出了 `SAC`（Semantic Alignment and Calibration），利用卷积预测 2D 偏移 $\Delta$ 并通过 `F.grid_sample` 对齐上下层特征；FSDC-DETR (2026) 提出了基于 2D Haar 小波（DWT）与分组卷积的 `FSD-Down` 下采样以及分流频空融合 `SFS-FF`。
- **深刻局限性（我们的突破口）**：
  - UAV-DETR (2025) 的 `SAC` 对双路特征同时做无约束 `grid_sample` 且伴随 $31\times 31$ 超大卷积核，导致 FPS 从 183 跌至 124（下降 32%）；FSDC-DETR 的全尺度 FFT/IFFT 在非 2 的幂次或 TensorRT 部署时兼容性较差，且计算量较大。
  - **我们的实质创新**：在 CCFM 跨尺度融合中设计轻量级**频空解耦门控与有界亚像素相位对齐模块（FAGA）**，仅对上采样深层特征学习有界亚像素偏移 $\Delta \in [-1, 1]$（由 $\tanh$ 约束并零初始化），结合局部高频能量门控抑制上采样背景泛滥，既消除最近邻插值混叠，又仅增加不到 0.15M 参数。

### 3.6 ReDHF (arXiv 2026.11) & HS-FPN (AAAI 2025) —— 浅层高频特征复活与椭圆辅助监督
- **核心机制**：ReDHF 证明了在 DETR 中复活被丢弃的高分辨率浅层特征并通过训练期椭圆引导辅助监督（EGSS）抑制浅层背景噪声，可稳定提升微小目标 AP；同时揭示了一个极其重要的反直觉结论（Appendix G.2 & Table 3）：**用浅层高分辨率特征直接做 Query Initialization 会导致 AP 下降（从 32.8 跌至 31.9），而将其经过语义过滤后注入 $P_3$ 并辅助训练期监督则能稳定涨点**！

---

## 4. 前沿共性总结与 DUT-Anti-UAV 专属突破方向

综合上述 22 篇前沿文献与第三阶段的历史实验审计，我们可以清晰地回答第五阶段前的核心问题：
1. **哪些问题仍未得到充分解决？**
   - 现有方法要么在主干/颈部堆砌重型注意力或增加 P2 解码层导致推理速度大幅下降（如 Drone-DETR、TQ-DETR 400 queries），要么只改损失函数而忽视了 RT-DETR 在**浅层 P2 细节丢弃、CCFM 最近邻上采样错位、Top-300 纯分类选点盲区、以及 MSDeformAttn 微小框采样坍缩**这四个环环相扣的结构性瓶颈。
2. **哪些机制对 DUT-Anti-UAV 特别有效且适合提出实质性创新？**
   - **方向 1（P2$\to$P3 零序列增长高频对比度注入）**：不增加 Transformer 编码尺度（保持 8400 tokens），利用无参数 2D Haar 小波高频子带与中心-环绕（Center-Surround）局部对比度能量从 Layer 4 (`P2`) 提取微小无人机边缘，经语义门控注入 `P3`，并配以训练期高斯椭圆辅助监督（推理期移除）。
   - **方向 2（CCFM 亚像素无混叠相位对齐与背景抑制）**：修正 `nn.Upsample(nearest)` 带来的跨尺度错位与大面积云层/树冠背景扩散。
   - **方向 3（保持 300 Query 契约的不确定性感知选点与方差保底可变形采样）**：打破 `MSDeformAttn` 在微小参考框下的采样坍缩，赋予极小框“最小上下文采样半径下界（Context Variance Floor）”，并在选点时融合分类置信度与定位质量先验。
   - **方向 4（全训练期、零推理开销的尺度自适应 Wasserstein-InnerGIoU 连续匹配、广义 MAL 软标签与跨层定位自蒸馏）**：彻底打通 Matcher $\to$ 分类软标签 $\to$ 回归损失 $\to$ 跨层自蒸馏的微小目标优化闭环。
