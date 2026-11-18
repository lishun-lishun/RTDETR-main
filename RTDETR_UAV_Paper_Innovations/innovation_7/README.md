# 创新点 7：解码器内嵌全局-局部高分辨率 P2 ROI 重采样模块 (`GLRoIDecoder`)

## 1. 研究动机与解决的核心问题
2025—2026 年的无人机运动/小目标检测代表作（如 [**GL-YOMO** (IEEE TAES 2025)](https://arxiv.org/html/2411.02582v1) 与 [**MSM-YOLO** (Sensors 2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12986653/)）证明：**“先全局粗定位、再对微小目标局部裁剪放大精检（Global-to-Local Zoom-in）”** 能有效规避全图下采样带来的小目标边缘稀释，显著提升 `Recall` 与高阈值定位精度 `AP75`。然而，外部两阶段裁剪重检（Two-Stage Crop-and-Redetect）破坏了 `RT-DETR` 的端到端单次前向推理结构，且延迟翻倍。与此同时，`RT-DETR-R18` 的 `PResNet18-vd` 骨干网在每次前向传播时均已计算出 **Layer 4 (`P2/4`, $160\times 160$, 64 通道)** 高分辨率特征，却将其直接丢弃。

## 2. 参考前沿文献（2025—2026）
- [**GL-YOMO** (IEEE TAES, Oct 2025)](https://arxiv.org/html/2411.02582v1)：全局检测与局部 ROI 自适应重检协同机制。
- [**MSM-YOLO / Spatio-Temporal Feature Fusion** (Sensors, Feb 2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12986653/)：针对微弱小无人机的 $160\times 160$ 局部候选区域重采样与双线性插值定位修正。
- [**ReDHF** (arXiv, 2026)](https://arxiv.org/html/2611.07994v1)：在不增加 Encoder Token 的前提下唤醒被丢弃的 $P_2$ 高分辨率特征用于 Decoder 细化。

## 3. 数学模型与前向公式
1. **严格保持 8,400 Encoder Tokens 与 300 Queries**：
   `GLRoIDecoder` 接收 `[P2, P3, P4, P5]`（对应 Layer `[4, 19, 22, 25]`），仅将 `[P3, P4, P5]`（共 $80^2+40^2+20^2=8,400$ 个 Token）送入编码器投影与 Top-300 Query 选择。
2. **Decoder 第 0 层粗定位后的小目标优先评分（Small-Object Priority Routing）**：
   设第 0 层解码器对 300 个匹配 Query 输出粗预测框 $B_i^{(0)} = (c_{x,i}, c_{y,i}, w_i, h_i)$ 与分类概率 $p_i^{(0)} = \max_c \sigma(s_{i,c}^{(0)})$，定义小目标重采样优先级：
   $$\psi_i = p_i^{(0)} \cdot \exp\!\left(-\frac{w_i h_i}{\tau_{\text{area}}}\right), \quad \mathcal{M} = \text{TopK}(\psi, M=16)$$
3. **FP32 保护的向量化 $P_2$ 局部子像素网格重采样（$5\times 5$ Grid Sample）**：
   对选出的 $M=16$ 个微小目标候选框，在 $160\times 160$ 的 $F_{P2}$ 上构建带上下文扩展系数 $\kappa=1.5$（且最小跨度保底为 $4\text{ px}$）的 $5\times 5$ 子像素采样网格 $\mathcal{G}_{5\times 5}(B_m^{(0)})$，一次性批量采样局部高分辨率特征块：
   $$R_m = \text{GridSample}_{\text{FP32}}\big(\text{Stem}(P_2),\, \mathcal{G}_{5\times 5}(B_m^{(0)})\big) \in \mathbb{R}^{64\times 5\times 5}$$
4. **局部-全局特征残差与亚像素参考框修正**：
   $$\mathbf{z}_m^{\text{roi}} = \text{DWConv}_{5\times 5}(R_m), \quad Q_m^{(1)} = Q_m^{(0)} + \alpha_{\max}\tanh(u_{\text{feat}}) \cdot p_m^{(0)} \cdot \text{MLP}_Q([\mathbf{z}_m^{\text{roi}}, Q_m^{(0)}])$$
   $$\text{logit}(B_m^{\text{ref}}) \leftarrow \text{logit}(B_m^{(0)}) + b_{\max}\tanh(u_{\text{box}}) \cdot p_m^{(0)} \cdot \tanh\big(\text{MLP}_B([\mathbf{z}_m^{\text{roi}}, Q_m^{(0)}])\big)$$
   随后送入第 1、2 层解码器完成精细回归。

## 4. 在原始 `rtdetr-r18.yaml` 上的独立接入位置
- 配置文件：`ultralytics/cfg/models/rt-detr/rtdetr-r18-inno7-glroi.yaml`
- 保持原始 `rtdetr-r18.yaml` 第 0–25 层完全不变，仅将 **Layer 26** 设为 `[[4, 19, 22, 25], 1, GLRoIDecoder, [nc, 256, 300, 4, 8, 3]]`，与创新点 1–6 完全独立互不干扰。
