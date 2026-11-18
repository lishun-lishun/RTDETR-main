# 创新点 1：小波-对比度各向异性浅层先验桥（WCA-SFP）

## 1. 模块定位与核心贡献
- **中文名称**：小波-对比度各向异性浅层先验桥（含训练期高斯椭圆辅助监督）
- **英文名称**：Wavelet-Contrast Anisotropic Shallow Feature Prior (`WCASFP`)
- **解决痛点**：原始 `RT-DETR-R18` 完全丢弃了 Backbone Layer 4 的 $P_2/4$（$160\times 160$）高分辨率特征，导致 $4\times 4\sim 12\times 12$ 像素的远距离微小无人机在进入 $P_3/8$（$80\times 80$）时丧失了亚像素边界梯度；而直接将 $P_2$ 送入 Transformer 会使 Token 数从 $8,400$ 暴增至 $34,000$。
- **核心机制**：
  1. **正交 2D Haar 小波无损下采样**：无参数分解出低频近似 $U_{LL}$ 与三方向高频细节 $\{U_{LH}, U_{HL}, U_{HH}\}$，并对对角高频子带 $U_{HH}$ 赋予高权重以突出紧凑点状无人机、抑制单向水平/垂直背景边缘；
  2. **中心-环绕相对局部对比度（`CenterSurroundContrast`）**：计算 $3\times 3$ 中心与 $5\times 5$ 空洞环绕背景的归一化相对对比度 $D_{\text{cs}}$；
  3. **语义-对比度联合门控注入**：生成目标显著图 $M_{\text{obj}}$，以有界可学习增益 $\alpha_{\max}\sigma(u_\gamma)$（初始化为 $0.10$，上限 $0.30$）注入 $P_3$，**不增加任何 Transformer Token 数量**；
  4. **训练期高斯-椭圆浅层辅助损失（`ShallowEllipseAuxLoss`）**：训练期监督 $Z_{\text{obj}}$，推理期零开销。

## 2. 接口契约与张量形状
- **输入**：`[p2, p3]` 列表，其中 `p2: [B, 64, 160, 160]`，`p3: [B, 128, 80, 80]`（或单张量 `p3`）。
- **输出**：`p3_enh: [B, 256, 80, 80]`，完美替代原始 `rtdetr-r18.yaml` 的第 17 层（`input_proj.0`），后续层索引 `18~26` 完全不变。
- **开关控制**：设置 `enabled=False` 时，严格退化为 `self.p3_proj(p3)`，与原始 R18 输出一致。

## 3. 运行单元测试
```bash
pytest innovation_1/test_module.py -v
```
