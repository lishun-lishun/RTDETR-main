# RT-DETR DUT-Anti-UAV V3：基于真实消融日志的稀疏目标监督扩展

**代码性质：训练期可拆卸研究原型。不是完整工程，不保证mAP提升，不得直接宣称四项学术创新。**

## 1. 研究来源与诊断

完整数字见 `reports/actual_log_analysis.md`。日志来源为用户仓库 `codex/uav-research-integration/runs/uav_ablation`，2026-11-17已读取13组CSV。**训练过程val mAP50:95 与此前历史test 0.707不能直接比较**。E0仅124轮，第一批E1~E8有150轮；TinyV2四个模型为123/120/70/68轮。应先完成E0公平对照。

前两版反馈：P2局部采样、Query-only残差、评分残差、尺度感知损失没有清晰的可复现增益。TinyV2的像素DN在120轮比同轮R18低0.00697。旧质量分支并非独立IoU质量头，原始VFL已经使用IoU目标。故本方案**不修改推理架构和Query数，不重复尝试NWD、P2采样、DN噪声改写或Matcher中心代价**。

## 2. 从代表性论文出发的四个可拆卸机制

| ID | 名称 | 主要借鉴 | 本版具体改动 | 部署增量 |
|---|---|---|---|---|
| GAP | Gaussian Anchor Prior：多尺度高斯锚点监督 | Bian et al., CVPR 2025，位置高斯分布；RT-DETRv3, WACV 2025，训练期密集正样本监督 | **直接对现有Encoder的全位置 `enc_score_head` 输出施加尺度自适应高斯监督**，不增加推理head、保持原Top300机制 | 0 |
| HNR | Hard-Negative Ranking：稀疏场景难负样本排序 | DEIM, CVPR 2025，稀疏匹配与Matchability-Aware Loss | 保留Hungarian一对一匹配，用匹配Query分数与远离GT的高置信假阳性Query做margin排序 | 0 |
| PLM | Progressive Localization Monotonicity：跨Decoder层定位单调约束 | D-FINE, ICLR 2025，GO-LSD多层定位学习 | 检查同一matched Query在不同Decoder层的误差，仅对**最终层不如中间层**的情况施加非负惩罚；不改分布回归 | 0 |
| BQR | Boundary Quantization Robustness：半像素定位量化鲁棒监督 | UGS, ICCV 2025，微小目标定位梯度稳定 | 对匹配GT施加确定性±0.5像素中心扰动，计算期望GIoU；保留原L1+GIoU和Matcher | 0 |

GAP与HNR最有希望围绕“**DUT单目标/稀疏目标下的Encoder候选覆盖与困难背景抑制**”形成一个统一研究假设。PLM、BQR是辅助验证，不应预先宣称其为独立论文创新。

### 2.1 主要数学形式

**GAP**：对第l层位置u和GT中心c、宽高w/h构造 `G_l(u)=max_j exp(-1/2 * [(u_x-c_x)^2/σ_x²+(u_y-c_y)^2/σ_y²])`，每一类按GT合并，再对Encoder完整分类logits做软标签Focal-weighted BCE。关键是监督**全位置候选生成**，与Top300后端主loss互补。

**HNR**：对每个Hungarian匹配正Query，取GT以外空间、IoU低于阈值的TopK困难背景Query，优化 `softplus(margin + score_hard - score_positive)`；保持一对一主监督。

**PLM**：对同Query的最后层误差e_L、早期层e_l，引入 `relu(e_L - stop_gradient(e_l) + margin)`，只促进最终层。误差基于目标对角线归一化中心偏差与对数宽高偏差。

**BQR**：对每个小目标GT的中心在x/y轴各移动±0.5像素，求 `mean_δ [1-GIoU(box_pred, shift(GT,δ))]`。这是额外正则，标注本身不被改变。

全部系数默认0。**注意：一些机制与已有研究有明显重叠，不能仅更换名字就声称原创**。

## 3. 文件说明

```text
RTDETR_DUT_ResearchV3/
  src/
    dut_sparse_signals.py       # 四项纯PyTorch训练Loss，单元测试可直接运行
    dut_sparse_supervision.py   # RTDETRDetectionLoss子类；短时捕获Encoder分数
  install_v3.py                # 默认Dry Run，--apply才安装；备份tasks.py
  run_v3.py                    # 默认Dry Run，--execute才训练
  analyze_uploaded_runs.py     # 对真实CSV按同一Epoch重新比较，不覆盖原有结果
  tests/test_signals.py        # CPU测试（数值、梯度、空GT等）
  reports/actual_log_analysis.md
  reports/actual_summary.csv
  CODE_AGENT_PROMPT.md
```

安装器不会复制完整项目、不修改原始R18模型YAML、不覆盖旧runs。只向`ultralytics/models/utils/`添加两个新文件，在`ultralytics/nn/tasks.py`的criterion选择与predict调用处增加**明确的可选分支**，并生成6个新YAML。

## 4. 如何安装

在服务器将压缩包解压后：

```bash
python install_v3.py --repo /home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main
# 检查dry run的目标目录与补丁预期
python install_v3.py --repo /home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main --apply
```

强烈建议先创建独立Git分支。

安装后应该有：

```text
ultralytics/models/utils/dut_sparse_signals.py
ultralytics/models/utils/dut_sparse_supervision.py
ultralytics/cfg/models/rt-detr/rtdetr_r18_dut_v3_gap.yaml
ultralytics/cfg/models/rt-detr/rtdetr_r18_dut_v3_hnr.yaml
ultralytics/cfg/models/rt-detr/rtdetr_r18_dut_v3_plm.yaml
ultralytics/cfg/models/rt-detr/rtdetr_r18_dut_v3_bqr.yaml
ultralytics/cfg/models/rt-detr/rtdetr_r18_dut_v3_gap_hnr.yaml
ultralytics/cfg/models/rt-detr/rtdetr_r18_dut_v3_all.yaml
```

**GAP的前向hook只在训练前向的context manager中短暂生效，随后立即删除**。这避免常驻hook的Python局部闭包导致的EMA深拷贝或Checkpoint序列化问题。所有四项训练信号在`RTDETRDetectionModel.loss()`正常返回的字典内加入，参与总loss；正常推理时不存在额外计算。

## 5. 先测试再训练

```bash
python -m pytest -q tests/test_signals.py
python -m py_compile src/*.py install_v3.py run_v3.py
# 在项目根目录执行，下述两个命令用于验证原始及opt-in模型能构建
python - <<'PY'
from ultralytics import RTDETR
from pathlib import Path
root=Path('ultralytics/cfg/models/rt-detr')
for f in ('rtdetr-r18.yaml','rtdetr_r18_dut_v3_gap.yaml'):
    model=RTDETR(str(root/f))
    print(f,'OK', type(model.model.model[-1]).__name__)
PY
```

随后需要在真正的`ultralytics_rtdetr`环境执行含GT、空GT、DN、AMP、Backward、EMA保存加载及Validation的完整集成测试。**目前只通过独立模块CPU测试和模拟仓库安装测试，未在用户服务器A30上跑通整个训练流程。**

## 6. 训练（默认不会启动）

```bash
python run_v3.py --repo /home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main --exp gap --device 0
# 自检通过后显式加--execute
python run_v3.py --repo /home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main --exp gap --device 0 --execute
```

与历史`train_uav_ablation.py`保持大体同样配置：Seed0、AdamW、img640、Batch8、AMP false、pretrained false、150轮、warmup设置记录为2000 micro-batch iterations。请Code Agent确认**真实**Warmup实现、GT坐标与解析器后再开始大规模实验。不要用正在训练的E0第124轮与完成的150轮改进作最终结论；需要完整150轮/更长对照。只在val上选权重，test用于最终报告。

## 7. 关键风险

1. GAP可能与原始Encoder一对一Query损失的分类目标发生竞争；先小权重、记录两者梯度余弦相似度。
2. HNR强化了已有VFL负样本压力，可能增加Precision但降低Recall；必须同时报告PR和FN变化。
3. PLM约束可能只是增加训练复杂度，且与已有多层自蒸馏重叠明显。
4. BQR的标注微扰可能不利于极严格AP85/AP90；必须逐阈值评价，不要将其当成确定增益。
5. 新增系数只用于一次明确的Val网格，不能频繁读取test挑选。
6. `run_v3.py`可选追加完整训练，但初次只做1batch学习率/梯度数值自检。
7. 单一DUT数据集、单一随机种子不足以证明方法原创性或普适有效。

## 8. 文献起点（完整查新仍需Code Agent完成）

- DEIM, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html
- Feature Information Driven Position Gaussian Distribution Estimation, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Bian_Feature_Information_Driven_Position_Gaussian_Distribution_Estimation_for_Tiny_Object_CVPR_2025_paper.html
- RT-DETRv3, WACV 2025: https://openaccess.thecvf.com/content/WACV2025/html/Wang_RT-DETRv3_Real-Time_End-to-End_Object_Detection_with_Hierarchical_Dense_Positive_Supervision_WACV_2025_paper.html
- D-FINE, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html
- UGS, ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.html
- UAV-DETR, 2026 (DUT-Anti-UAV): https://arxiv.org/abs/2603.22841
- STRUAV-DETR, IEEE IoTJ Sep 2026 (DUT-Anti-UAV): https://ieeexplore.ieee.org/abstract/document/11684786/
- TQ-DETR, Oct 2026 (VisDrone, **不是**DUT反无人机): https://pmc.ncbi.nlm.nih.gov/articles/PMC13667564/

**研究路线建议：真正可能形成论文的不是四个模块简单相加，而是围绕“稀疏目标下Encoder候选覆盖与背景混淆”建立机制和数据证据；PLM/BQR按消融结果保留。**
