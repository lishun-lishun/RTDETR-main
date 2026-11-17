# Code Agent Prompt：RTDETR-main真实训练结果诊断、论文查新与V3创新模块集成

## 身份与目标
你是高级计算机视觉研究工程师，熟悉PyTorch 2.3、Ultralytics YOLOv8.0.201 RT-DETR、DETR Hungarian matching、Encoder query selection、辅助Decoder层训练以及无人机微小目标检测。用户有A30 24GB多卡Linux服务器，希望从现有DUT-Anti-UAV任务形成**有明确机制解释、能够发表论文而不仅是微弱涨点**的方法。不得以“保证提高准确率”替代实验。

实际工程：`/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main`。
研究分支：`codex/uav-research-integration`。
数据：`dataset/DUT-Anti-UAV.yaml`，test曾记录2200张、2245框，常用640输入，单类别UAV。
原始历史测试基线：AP50=.966，AP75=.816，AP50:95=.707；但本次上传的`results.csv`属于验证集，数值约.65，**禁止混淆二者**。
源码真实入口：`ultralytics/nn/modules/head.py`的`RTDETRDecoder`，`ultralytics/nn/tasks.py`的`RTDETRDetectionModel`，`ultralytics/models/utils/loss.py`的`RTDETRDetectionLoss`，`ultralytics/models/utils/ops.py`的`HungarianMatcher`。

## Task A：实际检查本地Skills
先列举可用Skills，并读取相关`SKILL.md`。优先考虑`academic-research`/`literature-review`/`novelty`/`computer-vision`/`pytorch`/`rigor-reviewer`/`ml-training-recipes`一类能力，但**只能使用实际存在的Skill，不得虚构名字**。输出`reports/v3_selected_skills.md`。

## Task B：严谨审计已上传两版训练日志
读取：
`runs/uav_ablation/e0_baseline/seed0/`、`e1_queryonly`~`e8_p2_all`、`tinyv2_m1_peak`~`tinyv2_m4_nwd`全部`results.csv`,`args.yaml`,`training_config.json`。
先阅读本包的`reports/actual_log_analysis.md`，但以仓库中**实际记录**为准，确认本包摘要是否准确。
重点复核：
1. E0只有124个Epoch，E1~E8全部150；TinyV2分别123、120、70、68个Epoch。按**第68/70/120/123轮**同轮对比，禁止用未完成E0的峰值和其他150轮模型比最终效果。
2. TinyV2 M2在第120轮0.64666，E0第120轮0.65363，落后0.00697；M1第123轮0.65489，E0第123轮0.65386，仅+0.00103。
3. E1和E4在150轮约0.660，但E0尚无150轮记录；其领先幅度不能证明独立增益。
4. E5/E8组合不如单项的具体Epoch趋势，是否有反向梯度竞争。第一版QualityReranker仅由原有VFL训练，非独立的定位质量标签头。
5. TinyV2 M3/M4未完成，不得下最终失败结论；检查SNWD项是否使val/giou_loss不可直接跨模型比较。
6. 所有训练`pretrained=false`、seed0、AdamW、batch8、640、AMP关闭、lr0=1e-4、lrf=1、warmup_epochs=2000。**检查Warmup真实单位**以及e0的代码commit与e1不同的影响。
7. 检查`dataset/DUT-Anti-UAV.yaml`分割；不要把验证集结果当test，也不要把历史的其他工程test 0.707与现在的val 0.65直接比较。
8. 输出分类错误/误检/漏检、GT尺寸直方图、按IoU0.50/0.75/0.85/0.90 AP、Top300 encoder GT覆盖率及匹配质量。能复用本仓库工具优先复用，不重复造轮子。
输出`reports/v3_prior_runs_audit.md`、`reports/v3_prior_runs.csv`、曲线图与完整命令；未经实际验证的内容标“假设”。

## Task C：系统研究已公开论文
必须实际阅读摘要、方法、消融及官方源码（可用时），优先：
- CVPR 2025 DEIM，Dense O2O与MAL；
- WACV 2025 RT-DETRv3，Encoder密集监督与训练辅助分支；
- CVPR 2025 Bian et al. 信息驱动位置高斯分布；
- ICLR 2025 D-FINE，FDR与GO-LSD；
- ICCV 2025 UGS，微小目标定位梯度稳定；
- 2026 UAV-DETR（DUT-Anti-UAV）；
- 2026 STRUAV-DETR（DUT-Anti-UAV）；
- 2026 TQ-DETR（VisDrone，**任务不同**，可用于查重）；
- 2026最新微小目标、困难背景、目标检测监督论文。
输出方法对比矩阵，特别标注：什么已经属于成熟已知机制、哪些是本项目具体改造、与已有方法实质相同的地方。
不允许将直接实现已有论文称为原创；审稿人质疑风险高的模块必须清晰标注。

## Task D：接入本包四个训练期候选模块
压缩包`RTDETR_DUT_ResearchV3.zip`内：
- `src/dut_sparse_signals.py`
- `src/dut_sparse_supervision.py`
- `install_v3.py`
- `run_v3.py`
- `tests/test_signals.py`
- `README_zh.md`

实现了：
1. GAP：基于P3/P4/P5各位置的高斯GT先验监督，直接作用Encoder全部分类logits（不是只作用Top300），不新增推理分支。
2. HNR：对Hungarian匹配正Query与困难背景假阳性Query进行分数Margin排序；排除接近GT的候选，防止误将几何上有效的候选当作困难背景。
3. PLM：同一Query跨Decoder层的几何误差单调性约束，只对最终误差反而升高的情况惩罚；原始多层辅助Loss保留。
4. BQR：对微小GT施加对称±0.5像素中心扰动后的GIoU平均监督，增强对标注量化的鲁棒性。

这些都是**假设型候选**，不是已验证涨点的论文四大贡献。

接入要求：
- 第一执行`python install_v3.py --repo <项目路径>`做Dry Run，核查完整diff，创建独立Git分支后执行`--apply`。
- 原始`rtdetr-r18.yaml`不变；只额外生成6个`rtdetr_r18_dut_v3_*.yaml`。
- 只改`ultralytics/nn/tasks.py`中的opt-in criterion选择与一次训练predict的Context Manager；其余推理架构保持原样。
- **核验Forward Hook只在训练forward期间注册并删除**，不能持久存在于EMA/Checkpoint，不能将整个Decoder意外注册为criterion的子module。
- 关闭新功能时，原始R18保持完全相同的训练与推理流程；禁用开关的代码路径不应改变旧权重或原始结果。
- `DUTSparseCriterion.forward`通过基类Loss保留原有Hungarian、VFL、L1、GIoU、DN、辅助层处理，新四项Loss只附加于字典，默认0。
- 确保验证与推理阶段不执行新增计算。`DUTSparseSignals`只应在训练Loss中调用。
- 数据GT为归一化cxcywh；Encoder每尺度分数shape由实际input projection获得。若stride不是8/16/32或尺度不匹配，必须明确失败，不要静默假设。
- 对所有新增Loss分别记录量级、梯度范数、与主Loss梯度余弦相似度、Encoder Top300 GT recall。若新增项与基础监督持续相反，要减小权重或停止该方向。
- 对多GT与空GT训练、类别索引、AMP、梯度有限、二次前向、Checkpoint+EMA Deepcopy/序列化、Resume、独立Eval走完整测试。

## Task E：测试与修复
严格先运行：
```bash
python -m pytest -q tests/test_signals.py
python install_v3.py --repo /home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main
```
然后在独立分支安装后，编写额外集成测试，对真实模型`RTDETR('...rtdetr_r18_dut_v3_gap.yaml')`执行单张/多张forward、loss、backward、保存与加载。重点测试：
1. GAP开时`dense_logits`确实捕获完整8400个点（640输入典型P3:6400、P4:1600、P5:400）。
2. GAP反向确实更新`enc_score_head`参数，不是loss已加入但无梯度。
3. 不改变原有300个Decoder Query、DN分组、`RTDETRValidator`后处理或`predictions`形状。
4. 训练空GT和GT=1/2/10合法，不产生Inf/NaN；AMP稳定。
5. `torch.save()`完整EMA / model对象成功；之前代码中的Local Hook Pickling风险不得出现。
6. 同一个checkpoint同样图像在**原始R18**和**关闭全部Loss的V3**下输出相同（容差精确定义）。
7. GPU显存和训练速度可接受；所有额外计算只在训练中发生。

## Task F：科学消融和提前终止条件
现有基线不完整：**首先完成原始R18同协议150轮**，建议再建立200轮对照以观察尚未收敛的曲线。
独立实验：
```text
V0 original R18 complete 150 (and optional 200)
V1 GAP only
V2 HNR only
V3 PLM only
V4 BQR only
V5 GAP + HNR (仅当前两项有效时)
V6 all four (仅逐项有效后)
```
同样配置：`batch8,img640,seed0,AdamW,lr0=1e-4,lrf=1,warmup_epochs=2000 (核验单位),pretrained=false,amp=false`；与之前`training_config.json`一致，任何改动必须另做baseline对照。
输出所有实验的`args.yaml`、`results.csv`、`best.pt`、验证集AP50、AP75、AP50:95、Recall、按目标尺寸分组AP，模型及推理FPS、训练耗时。
如果第40轮就显著落后原始同轮基线，并出现训练崩溃/梯度冲突，允许依据预先定义条件早停，但不要机械地把40轮指标与150轮指标比较。
若Val增益小于预先定义阈值或仅seed0有效，不宣称提升。通过筛选的方案至少3 seeds；最后对test评估一次，禁止反复用test调超参数。

## Task G：更深入的科学问题论证
尤其需要判断：
- 目标稀疏是否真导致Top300 Encoder正样本覆盖低？若候选覆盖已接近100%，GAP不应成为核心创新点。
- 高置信假阳性究竟来自背景干扰还是框定位/置信度重排问题？若不存在显著硬负样本问题，不应把HNR写作主创新。
- 最后Decoder层是否经常比中间层定位差？若几乎无倒退，PLM几乎无作用。
- GT标注是否存在显著1像素不确定性？若标注精确且目标足够大，BQR的合理性弱。

最终给出四项的保留/淘汰表，论文应优先选择**一条统一主线+2个有价值的贡献**，不要把4项机械拼接成论文。

## Task H：最终交付
必须实际提供：
1. Skills检查报告和2025—2026查新矩阵。
2. 历史两版训练曲线的同口径分析，以及导致波动和过拟合的证据。
3. 修复后的实际代码差异，不能只给伪代码或模块描述。
4. 所有编译、CPU、GPU、模型序列化、Val评估的通过/失败日志。
5. 单项及组合实验的运行入口、独立输出目录和消融表。
6. 训练前必须确认并展示用户选择的GPU可用，默认Dry Run，不自动跑150轮。
7. 论文方法逻辑、机制可视化建议及审稿人可能的质疑。
8. 任何未跑出的数字必须写“未运行/未验证”，禁止编造。

**执行顺序：审计真实日志→查新与机制论证→小规模接入验证→公平基线→单项实验→多种子→论文判断。**
