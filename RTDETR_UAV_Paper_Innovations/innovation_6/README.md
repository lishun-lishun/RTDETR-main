# 创新点 6：融合后有符号稀疏语义校准模块 (SSSC-Lite / `SSSCRepC3`)

## 1. 研究动机与解决的核心问题
在复杂背景反无人机检测（[**DUT-Anti-UAV**](https://github.com/wangdongdut/DUT-Anti-UAV) 中的树林、城市建筑天际线、高压塔等场景）中，跨尺度特征融合后的高分辨率层（`P3/8` $80\times 80$ 与 `P4/16` $40\times 40$）包含大量由高频背景边缘引起的强伪影激活（Hard Negatives）。传统注意力模块仅支持非负权重缩放（$\sigma(\cdot) \in (0,1)$）或正残差叠加，无法主动抵消强背景杂波尖峰，导致 Encoder Top-300 Query 选择阶段混入大量高分背景假阳性。

## 2. 参考前沿文献（2025—2026）
- [**SSC-DETR** (Remote Sensing, Nov 2026)](https://www.mdpi.com/2072-4292/18/21/3858/pdf)：提出融合后有符号稀疏语义校准（Signed Sparse Semantic Calibration, SSSC）与零初始化限幅偏移采样（$|\Delta \mathbf{p}| \le 0.25$）。
- [**Temporal/Statistical Anomalies Drone Detection** (Sensors, Dec 2025)](https://www.mdpi.com/1424-8220/26/1/170)：利用局部统计异常度（Z-Score）快速定位稀疏候选区域。

## 3. 数学模型与前向公式
给定融合特征 $F \in \mathbb{R}^{B\times C\times H\times W}$：
1. **FP32 保护的统计异常显著性稀疏路由（Top-$K_r$ 选择）**：
   $$E(x,y) = \frac{1}{C}\sum_{c=1}^C |F_c(x,y)|, \quad Z_{\text{anom}}(x,y) = \tanh\!\left(\frac{E(x,y) - \mu_E}{\sigma_E + \epsilon}\right)$$
   $$S = \sigma\big(\mathcal{H}_{\text{sal}}([F, Z_{\text{anom}}])\big) \in (0,1)^{B\times 1\times H\times W}, \quad \mathcal{I}_{K_r} = \text{TopK}(S, K_r=128)$$
2. **零初始化有界亚像素对齐采样（$|\Delta\mathbf{p}_k| \le 0.25$）**：
   $$\Delta\mathbf{p}_k = 0.25 \tanh(\mathbf{W}_{\text{off}} \mathbf{f}_k), \quad \tilde{\mathbf{f}}_k = \text{GridSample}_{\text{FP32}}(F, \mathbf{p}_k + \Delta\mathbf{p}_k)$$
3. **L2 范数受限的双极性有符号残差校准与稀疏回写**：
   $$\mathbf{r}_k = \tanh(\mathbf{W}_{\text{sign}}\tilde{\mathbf{f}}_k) \odot (\mathbf{W}_{\text{mag}}\tilde{\mathbf{f}}_k), \quad \hat{\mathbf{r}}_k = \mathbf{r}_k \cdot \min\!\left(1, \frac{\tau_{\text{norm}}}{\|\mathbf{r}_k\|_2 + \epsilon}\right)$$
   $$F_{\text{out}}(\mathbf{p}_k) = F(\mathbf{p}_k) + \beta_{\max}\tanh(u_{\beta}) \cdot S(\mathbf{p}_k) \cdot \hat{\mathbf{r}}_k$$
   正残差分量强化微弱真无人机响应，负残差分量主动削峰抑制树梢/建筑边缘伪影。

## 4. 在原始 `rtdetr-r18.yaml` 上的独立接入位置
- 配置文件：`ultralytics/cfg/models/rt-detr/rtdetr-r18-inno6-sssc.yaml`
- 仅将原始 `rtdetr-r18.yaml` 的 **Layer 19**（`P3/8`, $80\times 80$, $K_r=128$）与 **Layer 22**（`P4/16`, $40\times 40$, $K_r=64$）由 `RepC3` 替换为 `SSSCRepC3`，其余层与 `RTDETRDecoder` 完全保持原始结构，与创新点 1–5、7 互不干扰。
