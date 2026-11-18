# 创新点 5：方向性运动模糊各向异性解耦重聚模块 (DMBA / `DMBARepC3`)

## 1. 研究动机与解决的核心问题
在可见光反无人机场景（[**DUT-Anti-UAV**](https://github.com/wangdongdut/DUT-Anti-UAV)）中，高速移动的小型无人机在快门曝光时间内会产生沿飞行速度矢量 $\vec{v}$ 的**各向异性线性运动拖影（Anisotropic Motion Blur）**。原本 $6\times 6\text{ px}$ 的机身能量被摊薄为 $6\times 16\text{ px}$ 的低对比度条带。原始 `RT-DETR-R18` 颈部 `RepC3` 仅堆叠各向同性的 $3\times 3$ 与 $1\times 1$ 卷积，无法沿运动轨迹方向将弥散能量重新聚拢至目标质心，导致预测框沿拖影方向过度膨胀或中心偏离，严重制约高 IoU 阈值指标（`AP75` 与 `mAP50:95`）。

## 2. 参考前沿文献（2025—2026）
- [**ViM-DETR** (SPIE, May 2026)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/14181/141810J/ViM-DETR--efficient-anti-UAV-detection-via-hybrid-vision/10.1117/12.3115669.full)：针对 `DUT-Anti-UAV` 严重运动模糊引入卷积门控线性单元（CGLU）。
- [**DCM-DETR** (Symmetry, Feb 2026)](https://www.mdpi.com/2073-8994/18/3/397)：方向性上下文建模（Directional Context Modelling）。
- [**UAV-DETR** (Sensors, Jul 2025)](https://www.mdpi.com/1424-8220/25/15/4582)：CSP-OmniKernel 多方向条带核特征融合。

## 3. 数学模型与前向公式
给定 `RepC3` 瓶颈隐层特征 $X \in \mathbb{R}^{B\times C_h\times H\times W}$（$C_h = \lfloor e \cdot C_2 \rfloor$）：
1. **四方向各向异性条带核组（$0^\circ, 90^\circ, 45^\circ, 135^\circ$）**：
   $$B_{0^\circ} = \text{DWConv}_{1\times K}(X), \quad B_{90^\circ} = \text{DWConv}_{K\times 1}(X), \quad B_{45^\circ} = \text{DWConv}_{3\times 3, d=2}^{\text{diag}}(X), \quad B_{135^\circ} = \text{DWConv}_{3\times 3, d=2}^{\text{anti-diag}}(X)$$
2. **FP32 保护的逐像素运动轨迹软路由**：
   $$\mathbf{w}_{\text{dir}}(x,y) = \text{Softmax}_{\text{FP32}}\big(\mathcal{G}_{\text{dir}}(X)\big) \in \Delta^3, \quad F_{\text{dir}} = \text{SiLU}\!\left(\text{BN}\!\left(\sum_{d=1}^4 w_{\text{dir}}^{(d)} \odot B_d\right)\right)$$
3. **中心-拖影差分 CGLU 能量重聚与零初始化有界残差注入**：
   $$F_{\text{deblur}} = \text{PWConv}_{\text{out}}\Big(\text{SiLU}(\text{PWConv}_1(F_{\text{dir}})) \odot \sigma\big(\text{DWConv}_{3\times 3}(X - F_{\text{dir}})\big)\Big)$$
   $$F_{\text{out}} = F_{\text{base}} + \gamma_{\max}\tanh(u_{\gamma}) \cdot F_{\text{deblur}}, \quad u_{\gamma}(0) = 0$$

## 4. 在原始 `rtdetr-r18.yaml` 上的独立接入位置
- 配置文件：`ultralytics/cfg/models/rt-detr/rtdetr-r18-inno5-dmba.yaml`
- 仅将原始 `rtdetr-r18.yaml` 的 **Layer 19**（`P3/8`, $80\times 80$）与 **Layer 22**（`P4/16`, $40\times 40$）由 `RepC3` 替换为 `DMBARepC3`，其余所有层与 `RTDETRDecoder`（Layer 26 `[[19, 22, 25], ...]`）完全不变，与创新点 1–4、6、7 互不干扰。
