# 03 历史消融实验深度审计与失败原因分析报告（Historical Experiment Analysis）

## 1. 历史实验全局盘点与审计范围

在 [`codex/uav-research-integration`](https://github.com/lishun-lishun/RTDETR-main/tree/codex/uav-research-integration) 分支及过往多轮改进尝试中，研究团队围绕 `RT-DETR-R18` 先后设计并测试了数十个模块与三个版本的系统性方案。为避免重复踩坑，本报告首先对所有历史模块与 `runs/uav_ablation/` 实验记录进行分类审计。

### 1.1 历史改进模块全景分类表

| 演进阶段 / 类别 | 涉及的具体模块名称 | 核心设计意图 | 实验状态与实际表现归因 |
| :--- | :--- | :--- | :--- |
| **早期局部特征与注意力尝试** | `BAPS`, `RDCF`, `LSCE`, `ADCF`, `PHR`, `SCRE`, `SPLF` | 在 Backbone 或 Neck 中加入局部空间对比度增强、空洞卷积、边缘重标定或金字塔池化。 | **已测试，提升微弱或下降**：多数模块在 $P_3/P_4/P_5$ 上堆叠卷积或通道/空间注意力，既未利用 $P_2$（$160\times 160$）的原始高频边缘，又破坏了预训练特征分布，且增加了冗余参数。 |
| **通用 YOLO/DETR 算子移植** | `SA-RepBlock`, `SP-ASFF`, `Res2-BasicBlock`, `Lite-GDNeck` | 替换 `BasicBlock` 或 `RepC3`，引入自适应空间特征融合（ASFF）或 Gather-and-Distribute 颈部。 | **已测试，性价比低**：`Res2-BasicBlock` 改变了主干拓扑，导致预训练权重无法对齐；`SP-ASFF` 与 `Lite-GDNeck` 在 Neck 引入过多跨尺度重加权分支，与 RT-DETR 原生 CCFM + Decoder 跨尺度变形注意力功能重复。 |
| **Backbone 轻量化与 CSP 改造** | `CSPResidualStage`, `CSPContextStage`, `LDW-ResNet18`, `R01–R16` | 用深度可分离卷积或 CSP 结构重构 ResNet18 主干以降低参数量。 | **已测试，微小目标精度受损**：正如此前实验及 [UAV-DETR (2026)](https://arxiv.org/html/2603.22841v1) Table 4 消融所示，单纯对 ResNet18 进行激进的通道压缩或 DWConv 替换（如 `O+A+B+C`），会导致语义容量下降，`mAP50:95` 反而下跌约 1.4%。 |
| **高分辨率骨干替换** | `HRNetV2-W18` 及其变体 | 全程保持高分辨率并行多分支特征提取（测试集 `mAP50=0.977, mAP75=0.838, mAP50:95=0.723, AP_small=0.628`）。 | **有效但计算代价高**：证明了**高分辨率特征保留（Stride 4）对提升 `AP75` (+2.2%) 和 `AP_small` (+3.0%) 至关重要**，但 HRNet 并行四分支显著增加了推理延迟与显存占用，偏离了保持 R18 实时性的目标。 |
| **第一版创新系统 (V1)** | `CenterRingEvidence`, `BoundedGeometryRefiner`, `QualityReranker`, `SizeAwareGeometryLoss` | 引入中心环背景对比度证据、有界几何修正器、后处理质量重排序与尺寸感知几何损失（部分实验跑满 150 轮）。 | **部分有效但存在机制冲突**：`SizeAwareGeometryLoss` 改善了小目标梯度，但 `QualityReranker` 在推理后处理阶段强行改变得分排序，若与训练期 Hungarian Matcher 分配准则不一致，会在高置信度区间引发分类与定位错位；`BoundedGeometryRefiner` 的硬截断限制了跨层大步长修正。 |
| **第二版创新系统 (V2 - TinyV2)** | `LPQ` (Local Prior Query), `PDN` (Pyramid Detail Neck), `TCM` (Target Contrast Module), `SNWD` (Scale NWD) | 联合优化局部查询先验、细节颈部、对比度模块与缩放 NWD 损失。 | **实验协议截断（仅记录到 68–123 轮）**：不能简单判定算法无效！由于训练在 68–123 轮提前终止（未进入最后 20 轮学习率充分衰减与 Mosaic 关闭阶段），其 `mAP50:95` 与跑满 150 轮的 V1 无法直接横向比较。此外，纯 `SNWD` 在收紧高 IoU 边界（`AP75`）时不如 `GIoU/Inner-GIoU` 敏锐。 |
| **第三版研究方案 (V3)** | `GAP` (Global-Local Attention/Prior), `HNR` (Hard Negative Repulsion), `PLM` (Progressive Localization Module), `BQR` (Boundary Quality Refinement) | 针对困难负样本抑制、渐进定位与边界质量重校准进行设计。 | **代码已设计 / 部分待完整统一协议验证**：绝对不能将尚未在统一 150 轮公平协议下跑完的 V3 模块直接归结为算法失败，但其多模块同时介入 Decoder 内部容易引起训练初期梯度不稳定，需要更干净的残差门控初始化（Zero/Small-gain Init）。 |

---

## 2. 历史训练数据与实验协议不一致性剖析

通过对历史实验记录（`results.csv`、`training_config.json`、`args.yaml`）的交叉核对，我们发现过往实验存在三个严重的**实验协议陷阱（Experimental Protocol Pitfalls）**，直接掩盖了真实算法规律：

### 2.1 陷阱一：验证集（Val）与测试集（Test）基线混淆
- **现象**：历史报告中原始 `RT-DETR-R18` 的结果为 `mAP50 = 0.966, mAP75 = 0.816, mAP50:95 = 0.707, AP_small = 0.598`；而在 `codex/uav-research-integration` 分支的 `runs/uav_ablation/` 中，基线 `E0` 及各消融模型的 `mAP50:95` 均在 `0.65–0.66` 左右徘徊。
- **根因诊断**：
  1. **数据集划分难度差异**：DUT-Anti-UAV 的验证集（2,600 张图像，2,621 个实例）与测试集（2,200 张图像，2,245 个实例）在目标尺度分布与复杂背景占比上存在显著差异（验证集中包含更多极端微小与夜间/逆光困难样本）。`0.707` 是早期工程在 **Test 集**（且可能伴随特定预训练权重与满轮次 EMA 收敛）上的评测结果，而 `0.65–0.66` 是新消融分支在 **Validation 集** 上的训练过程指标。
  2. **结论**：**严禁拿新消融实验的验证集 `0.66` 与历史测试集的 `0.707` 直接对比并宣称“所有改进均大幅退步”**。所有消融对比必须严格锁定在同一数据划分（统一先看 Val 曲线，最终统一在 Test 集 2,200 张图像上用同一脚本评测）。

### 2.2 陷阱二：训练轮数截断（Epoch Truncation）导致的不公平比较
- **现象**：
  - `E0` 基线日志仅记录到 **124 轮**；
  - `TinyV2`（LPQ/PDN/TCM/SNWD）部分实验仅记录到 **68–123 轮**；
  - 第一版（V1）部分实验则完整训练到了 **150 轮**。
- **根因诊断**：
  - DETR 类模型（特别是包含可变形注意力 `MSDeformAttn` 与匈牙利二分图匹配的架构）具有典型的**后期收敛特性（Late-Stage Convergence）**。在 Cosine 学习率调度下，前 100 轮学习率仍然较高，Hungarian Matcher 的正样本分配仍在探索，且 `close_mosaic`（通常在最后 10–15 轮关闭马赛克增强）尚未触发或刚刚触发。
  - 在第 120–150 轮期间，随着学习率降至 $10^{-5}$ 量级且图像恢复真实尺度分布，模型的边界框回归精度（`AP75` 与 `mAP50:95`）通常还会出现 **1.5%–3.0% 的显著跃升**。
  - **结论**：将 68–123 轮截断的模型与 150 轮完整收敛的模型直接对比 `best mAP` 是严重的科学错误。本项目的后续实验设计必须强制检查 `completed_epochs == target_epochs`，或在相同 Epoch 切片（如 `@Epoch 60`, `@Epoch 120`, `@Epoch 150`）下进行对齐比较。

### 2.3 陷阱三：预训练权重破坏与无门控冷启动（Cold-Start Shock）
- **现象**：许多修改了 Backbone（如 `LDW-ResNet18`, `Res2-BasicBlock`）或直接替换 Neck 核心算子的实验，在训练前 30 轮 Loss 明显高于原始 R18，最终精度反而不如基线。
- **根因诊断**：
  - 当直接修改 Backbone 内部结构时，原有的预训练权重（COCO / ImageNet 初始化）因张量形状或拓扑不匹配而被迫随机初始化；
  - 当在 Neck 或 Decoder 中直接串联随机初始化的新模块（未采用残差零初始化 $\gamma=0$ 或保守门控 $\alpha_{\max}=0.2$）时，随机特征噪声会在训练初期直接冲垮 `RTDETRDecoder` 的 Query Selection 与 `MSDeformAttn` 的采样偏移学习，导致二分图匹配陷入局部劣解。

---

## 3. 核心科学问题回答（七问七答）

### 问题 1：之前的模块为什么没有明显提升？
1. **空间分辨率天花板未破**：绝大多数早期模块（`BAPS`, `RDCF`, `LSCE`, `SA-RepBlock`）只在 $P_3(80\times 80)$、$P_4(40\times 40)$、$P_5(20\times 20)$ 上做文章。对于 $640\times 640$ 图像中 $4\times 4 \sim 8\times 8$ 像素的远距离无人机，经过 8 倍下采样到达 $P_3$ 时仅剩 $0.5\sim 1$ 个像素，高频边缘早已在 Stage 2 $\to$ Stage 3 的步长下采样中丢失。巧妇难为无米之炊，在丢失细节的 $P_3$ 上堆叠再多注意力也无法凭空恢复亚像素边界。
2. **定位精化链路瓶颈未解**：DUT-Anti-UAV 上原始 R18 的 `mAP50` 已经高达 **0.966**（说明目标大体位置检出率已经很高），真正的短板在于 **`mAP75` (0.816) 与 `mAP50:95` (0.707) 之间的巨大落差（相差 25.9 个百分点！）** 以及 **`AP_small` (0.598)**。之前的模块大多致力于“增强分类响应”，却忽视了 `MSDeformAttn` 在微小参考框下的**采样点坍缩**以及固定坐标回归对模糊边界的不确定性失明。

### 问题 2：究竟是 Backbone 特征不足，还是 Decoder / 监督问题？
**答案是：浅层细节传递断层（P2$\to$P3）+ Decoder 微小框采样坍缩 + 小目标匹配/回归梯度失衡三者共同构成的耦合瓶颈，其中 Decoder 与训练监督是提升 `AP75` 和 `mAP50:95` 的决定性杠杆。**
- 证据 1：`HRNetV2-W18` 将 `AP_small` 从 0.598 提升至 0.628，`mAP75` 从 0.816 提升至 0.838，证明 **Stride-4 (`P2`) 的高分辨率空间信息是必须回收的**；但无需更换整个 Backbone，只需将 R18 第 4 层已有的 `P2` 高频细节经对比度门控注入 `P3` 即可（如 [TQ-DETR 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667564/) 与 [ReDHF 2026](https://arxiv.org/html/2611.07994v1) 所证实）。
- 证据 2：[UAV-DETR (2026)](https://arxiv.org/html/2603.22841v1) 的消融实验（Table 4）表明，仅改进回归损失（Inner-CIoU + NWD）就能在不增加任何推理开销的情况下将 `mAP75` 提升 **+4.16%**；[D-FINE (ICLR 2025)](https://arxiv.org/html/2410.13842v1) 证明将 Decoder 定点回归改为细粒度分布精化与自蒸馏可提升 **+2.0%~5.3% AP**。

### 问题 3：哪些机制可能已经被原始 RT-DETR 解决？
- **深层全局语义上下文建模**：原始 RT-DETR 的 `AIFI` 已经在 $P_5$ 上通过多头自注意力建立了全局长程依赖。因此在 Backbone 深层或 $P_5$ 后再堆叠复杂的全局注意力（如 Non-local、ViT Block、重型自注意力）纯属冗余。
- **基础多尺度双向融合**：CCFM 已经包含了完整的 Top-Down FPN 与 Bottom-Up PAN 双向 `RepC3` 路径，因此简单增加更多跨层连线（如 BiFPN、Lite-GDNeck、SP-ASFF）带来的边际收益极低。

### 问题 4：哪些模块与已有方法功能重复？
- `SA-RepBlock` / `CSPResidualStage` 与 CCFM 原生的 `RepC3`（含结构重参数化 `RepConv`）功能高度重叠；
- `SPLF` / `RDCF` 多分支空洞卷积与 `DeformableTransformerDecoder` 中的多尺度可变形采样在感受野扩展上功能重叠；
- 单纯的 `SNWD` 与 `SizeAwareGeometryLoss` 在一阶梯度平滑上功能重叠，且缺乏对高 IoU 阶段严格边界对齐（`AP75`）的二阶段接力机制。

### 问题 5：哪些失败来自实验协议而非算法设计？
- `TinyV2`（LPQ/PDN/TCM/SNWD）在 68–123 轮被提前截断，未经历完整的 Cosine 尾段收敛与无增强微调；
- 跨 Validation 集（`mAP50:95 ≈ 0.65–0.66`）与历史 Test 集（`mAP50:95 = 0.707`）的错位对比；
- 修改 Backbone 导致的预训练权重失效（与加载了完整预训练权重的 R18 对比不公平）。

### 问题 6：哪些研究方向不值得继续投入？
1. **不再替换或大改 ResNet18 主干内部结构**（如 `Res2-BasicBlock`、`LDW-ResNet18`）：既破坏预训练权重，又容易损伤特征容量。
2. **不再将 $P_2$ ($160\times 160$) 直接作为第 4 个特征层送入 Transformer Encoder/Decoder**：这会使 token 总数从 $8,400$ 暴增至 $34,000$（增加 4.05 倍），彻底摧毁 R18 的实时性。
3. **不再设计脱离训练期 Matcher 的独立推理期后处理重排序器（如独立 `QualityReranker`）**：推理期启发式改分容易破坏端到端二分图建立的置信度校准。

### 问题 7：哪些现象能够成为本论文的核心研究动机（Research Motivation）？
我们可以提炼出极具说服力的**三大核心科学现象**作为论文引言与方法章节的逻辑主线：
1. **高检出率与低高精度定位的悖论（High-$AP_{50}$ vs. Low-$AP_{75}$ Paradox）**：在 DUT-Anti-UAV 上，RT-DETR-R18 的 `mAP50` 高达 $96.6\%$，但 `mAP75` 骤降至 $81.6\%$、`AP_small` 仅为 $59.8\%$。这说明模型“看得到无人机的大致方位”，但由于 **$P_2$ 浅层高频边界被丢弃** 以及 **CCFM 最近邻上采样的跨层亚像素错位**，导致预测框无法紧贴微小无人机的真实物理边界。
2. **微弱目标在 Top-K 查询选择与可变形采样中的“双重坍缩”（Dual Collapse in Query Selection & Deformable Sampling）**：一方面，仅靠分类置信度的 Top-300 选点容易漏掉低对比度/模糊无人机，且默认最小锚框（$0.05 \times 640 = 32\text{px}$）远大于微小目标；另一方面，当参考框收缩至微小尺寸（$<10\text{px}$）时，`MSDeformAttn` 的采样偏移量 $\Delta p \propto 0.5 \cdot w_{\text{ref}}$ 坍缩至单个特征网格内部，导致 Decoder 失去局部上下文对比能力。
3. **一对一匹配中的“IoU 悬崖与 VFL 梯度饥饿”（IoU Cliff & VFL Gradient Starvation）**：微小目标 1–2 像素的抖动即引发 GIoU 剧烈震荡与 Hungarian 匹配跳变，同时导致 Varifocal Loss 的 IoU 软标签趋近于 0，使微小正样本在训练早期丧失有效监督梯度。
