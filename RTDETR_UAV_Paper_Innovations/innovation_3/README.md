# 创新点 3：不确定性感知查询选择与上下文方差保底可变形解码器（UAD-CR）

## 1. 模块定位与核心贡献
- **中文名称**：不确定性感知查询选择与上下文方差保底可变形解码器
- **英文名称**：Uncertainty-Aware Query Selection and Context-Variance-Floored Deformable Decoder (`UADCRDecoder`)
- **解决痛点**：
  1. **源码级缺陷 A（Query 选择纯分类偏置）**：Ultralytics 的 `RTDETRDecoder._get_decoder_input` 仅按 `enc_outputs_scores.max(-1)` 选择 Top-300 查询，忽略了定位不确定性与前景显著性；
  2. **源码级缺陷 B（`MSDeformAttn` 微小框采样点向单像素坍缩）**：当参考框收敛至微小无人机（如 $6\times 6\text{px}$，归一化宽高 $w_r \approx 0.009$）时，采样偏移乘子 $0.5 \cdot w_r$ 远小于特征图网格间距（$1/80 = 0.0125$），导致 4 个采样点全部坍缩在同一个网格像素内（Rank-1 Collapse），丧失边界与上下文感知能力。
- **核心机制**：
  1. **不确定性校准 Top-300 查询选择**：通过 $s_i^{\text{UAD}} = p_i \cdot (1 + \lambda_q q_i \exp(-|p_i - q_i|))$ 融合分类置信度与定位质量先验，严格保持 **300 个查询预算**；
  2. **层级自适应上下文采样方差保底（Context Variance Floor）**：在 `MSDeformAttn` 中定义 $w_{\text{eff}}^{(l)} = \sqrt{w_r^2 + (\eta_l / W_l)^2}$，强制微小参考框张开至少 $\eta_l \approx 1.6$ 个网格步长的局部上下文探测环，而对大中目标（$w_r \gg \eta_l / W_l$）几乎无影响；
  3. **目标解耦交叉注意力门控与不确定性阻尼精化**：全部新增层采用零初始化，100% 兼容原始 `RTDETRDecoder` 预训练权重加载。

## 2. 接口契约与张量形状
- **输入**：`[X3, F4, F5]` 列表，形状分别为 `[B, 256, 80, 80]`, `[B, 256, 40, 40]`, `[B, 256, 20, 20]`。
- **输出**：训练期返回 `(dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)`；推理期返回 `(y, raw)`，其中 `y: [B, 300, 4 + nc]`。
- **开关控制**：调用 `decoder.set_enabled(False)` 或初始化 `enabled=False` 时，严格恢复原始 `RTDETRDecoder` 行为。

## 3. 运行单元测试
```bash
pytest innovation_3/test_module.py -v
```
