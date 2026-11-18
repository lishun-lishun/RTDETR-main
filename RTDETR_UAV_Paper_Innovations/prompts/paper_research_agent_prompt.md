# Prompt C：科研论文撰写与审稿人抗辩 Code Agent 执行提示词（Paper Research Agent Prompt）

你可以将以下提示词直接复制给负责论文撰写的科研 Code Agent，基于本项目生成的研究报告、数学公式与实测消融数据完成高质量英文/中文学术论文（目标期刊/会议：*IEEE T-ITS*, *IEEE T-CSVT*, *Pattern Recognition*, *ICRA*, *CVPR/ICCV*）。

---

## 复制以下内容给 Code Agent：

```markdown
# 任务角色：计算机视觉顶刊/顶会论文主笔与审稿抗辩专家（Paper Research Agent）

## 1. 任务目标
基于 `RTDETR_UAV_Paper_Innovations/` 中的完整方法设计、数学推导、源码实现与实验结果，撰写一篇面向**可见光复杂背景下远距离微小无人机实时端到端检测**的高水平学术论文，并完成审稿人潜在质疑的预判与防御。

## 2. 核心参考资料路径（位于交付包内）
- 项目与基线源码级解析：`RTDETR_UAV_Paper_Innovations/research/project_understanding.md`
- 历史失败实验归因分析：`RTDETR_UAV_Paper_Innovations/research/historical_experiment_analysis.md`
- 2023–2026 前沿文献综述（22篇）：`RTDETR_UAV_Paper_Innovations/research/literature_review.md`
- 文献创新性对照矩阵：`RTDETR_UAV_Paper_Innovations/research/novelty_matrix.csv`
- 四大创新点完整数学公式与审稿自查：`RTDETR_UAV_Paper_Innovations/research/paper_method_design.md`
- 实验计划与评测协议：`RTDETR_UAV_Paper_Innovations/experiments/experiment_plan.md`

## 3. 论文核心叙事逻辑（Narrative Arc — 严禁写成四个孤立模块的简单拼接）

### 3.1 引言与核心科学动机（Introduction & Motivation）
- **切入点**：反无人机（Counter-UAV）视觉感知需要在复杂天际线、浓密树冠、城市建筑与飞鸟干扰下实时精确定位远距离微小无人机（在 $640\times 640$ 图像中常不足 $8\times 8\text{px}$）。
- **核心发现（Empirical Paradox）**：在 `DUT-Anti-UAV` 基准上，`RT-DETR-R18` 虽然取得了较高的粗略检出率（$mAP_{50} = 96.6\%$），但在严格定位指标 $mAP_{75}$（$81.6\%$）与小目标指标 $AP_{\text{small}}$（$59.8\%$）上出现断崖式下跌。
- **根因诊断（Four-Stage Cascaded Bottleneck）**：
  1. **浅层边界截断**：直接丢弃 $P_2/4$（$160\times 160$）特征导致亚像素边缘丢失，而直接将 $P_2$ 送入 Transformer 会使 Token 数从 $8,400$ 暴增 $4.05\times$ 至 $34,000$；
  2. **跨尺度混叠与亚像素错位**：CCFM 的最近邻上采样引入 $0.5\sim 1.5$ 网格相位偏移，使微小目标激活弥散为双峰；
  3. **可变形采样点向单像素坍缩（Rank-1 Sampling Collapse）**：当参考框收敛至微小无人机（$w_r \approx 0.009$）时，`MSDeformAttn` 的采样偏移 $\Delta p \propto 0.5\cdot w_r$ 小于特征网格步长，4 个采样点退化至同一像素内部；
  4. **一对一匹配的 IoU 悬崖与 VFL 梯度饥饿**：微小框 2 像素偏移即引发 GIoU 崩塌与软标签 $q \to 0$ 的梯度死区。

### 3.2 方法章节结构设计（Methodology — 聚合成三大紧密咬合的核心贡献）
为避免审稿人批评“模块堆砌（Incremental A+B+C+D）”，在论文正文中应将四个工程模块组织为 **三项层层递进的核心方法贡献**：
1. **Section 3.2：空频对比度浅层先验与亚像素相位对齐特征金字塔（整合 WCA-SFP 与 FAGA）**
   - 阐述如何通过正交 2D Haar 小波对角高频加权与中心-环绕相对对比度（`WCA-SFP`），在**零 Token 增长（保持 8,400 tokens）**的前提下将 $P_2$ 亚像素边缘注入 $P_3$，并配合训练期高斯椭圆辅助监督抑制背景杂波；
   - 阐述如何通过非对称零初始化、$\tanh$ 有界约束的亚像素相位对齐与无 FFT 频空正交双向门控（`FAGA`），消除跨层特征错位。
2. **Section 3.3：不确定性校准查询选择与上下文方差保底可变形解码器（UAD-CR）**
   - 给出 `MSDeformAttn` 在微小参考框下的采样坍缩定理及证明；
   - 提出层级自适应上下文采样方差保底公式 $w_{\text{eff}}^{(l)} = \sqrt{w_r^2 + (\eta_l / W_l)^2}$ 与保持 300 查询预算的定位质量-分类联合选点机制。
3. **Section 3.4：尺度自适应 Wasserstein-InnerGIoU 连续匹配与跨层定位自蒸馏（SAH-NWD & CW-LSD，纯训练期零推理开销）**
   - 阐述连续指数尺度门控 $\beta(s_g) = \exp(-s_g / \tau_s)$ 如何统一协调 `ScaleAdaptiveHungarianMatcher`、广义连续软标签 `GC-MAL`、动态辅助框回归以及跨层连续高斯定位自蒸馏 `CW-LSD`。

### 3.3 相关工作对比重点（Related Work）
必须深入对比并明确区分本方法与 2024–2026 最相近工作：
- 对比 **UAV-DETR (NWPU, arXiv 2026)**：指出其替换 ResNet18 主干破坏预训练导致掉点、未改进 Decoder 采样坍缩、且使用静态混合损失权重；
- 对比 **TQ-DETR (Sensors 2026)**：指出其将查询数扩充至 400 导致 FPS 下降 $35\%$，且各向同性平均池化边缘提取无法区分点状无人机与线性云层边缘；
- 对比 **D-FINE (ICLR 2025)** 与 **DEIM (CVPR 2025)**：指出 D-FINE 的 132 维离散分布头在极小目标上相对尺度量化过粗，而 DEIM 的 MAL 在 $\text{IoU}=0$ 时依然存在梯度死区。

## 4. 验收标准（Acceptance Criteria）
- [ ] 所有数学公式符号体系全局统一，变量维度（如 $\mathbb{R}^{B\times C\times H\times W}$、$\mathbb{R}^{B\times 8400\times 256}$）与实际 PyTorch 代码严格一致。
- [ ] 所有引用文献真实可查（来自 `novelty_matrix.csv`），严禁虚构任何文献或实验数值。
- [ ] 包含专门的“Limitations and Computational Overhead Analysis”小节，客观呈现参数量（$+2.9\%$）、GFLOPs（$+2.2\%$）与高精度定位收益的权衡。
```
