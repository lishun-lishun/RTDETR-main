# 创新点 2：频空解耦门控与有界亚像素相位对齐跨尺度融合（FAGA）

## 1. 模块定位与核心贡献
- **中文名称**：频空解耦门控与有界亚像素相位对齐跨尺度融合模块
- **英文名称**：Frequency-Decoupled Gating and Bounded Sub-Pixel Phase-Aligned Cross-Scale Fusion (`FAGAFusion`)
- **解决痛点**：原始 `RT-DETR-R18` 的 CCFM 颈部在 Top-Down 融合（Layer 11–13、Layer 16–18）时采用 `nn.Upsample(mode='nearest')` + `Concat`，不仅在跨层特征间引入 $0.5\sim 1.5$ 个网格的亚像素相位错位（导致微小目标响应弥散为双峰，严重损害 `AP75`），还会将深层大面积云层/树冠背景激活扩散至浅层。
- **核心机制**：
  1. **非对称有界亚像素相位对齐（Asymmetric Bounded Sub-Pixel Phase Warping）**：以浅层高分辨率特征 $F_{\text{low}}$ 为几何锚点，对上采样深层特征预测零初始化、$\tanh$ 有界约束（$\Delta \mathbf{p} \in [-r_{\max}, +r_{\max}]$，默认 $r_{\max}=1.5$ 像素）的二维亚像素偏移场并执行双线性重采样；
  2. **无 FFT 的频空正交双向门控**：分解深层低频语义包络 $S_{\text{sem}}$ 与浅层高频边缘残差 $B_{\text{edge}}$ 进行正交双向调制，100% 兼容 AMP 半精度与 TensorRT/ONNX 导出；
  3. **零初始化恒等启动**：训练第 0 步时偏移场与调制增益严格为 0，前向等价于原始 `Concat`，彻底避免预训练特征冷启动震荡。

## 2. 接口契约与张量形状
- **输入**：`[f_high, f_low]`，各为 `[B, 256, H, W]`（或 `f_high` 为 `[B, 256, H/2, W/2]`）。
- **输出**：`out: [B, 512, H, W]`，原地替代 `rtdetr-r18.yaml` 的 Layer 13 和 Layer 18 `Concat` 节点。
- **开关控制**：`enabled=False` 时严格执行 `torch.cat([f_high, f_low], dim=1)`。

## 3. 运行单元测试
```bash
pytest innovation_2/test_module.py -v
```
