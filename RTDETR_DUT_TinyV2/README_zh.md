# DUT-Anti-UAV / RT-DETR-R18：TinyV2创新代码包

**代码参照**：用户仓库 `lishun-lishun/RTDETR-main`，`main` Git tree SHA `9636869af908610c7aa84c4cfca3f2e33479af41`，于2026年11月16日读取关键源码。**状态**：可移植研究原型；未在用户A30环境中验证训练或精度；不能保证涨点。

## 为什么改变方向

用户已经验证多个Backbone改动（BAPS/RDCF/CSP/LDW/R01—R16等）无稳定收益。故本版与上一版 P2空间采样+有界细化+重排不同，围绕RT-DETR的四个瓶颈：编码器Top-K选Query、正负去噪、Hungarian匹配、微小目标回归。全部组件均采用原始R18+AIFI+CCFM+原始Deformable Decoder；**未新增P2 head、未新增Query、未替换主干**。

## 四个独立可选的机制

1. **M1 LPQ（Local-Peak Query Ranking）**：对Encoder原本的分类logit，依据各特征层的局部高响应峰加入**小幅正奖励**再选择Top-300。相比旧SAQS，本方法不设定固定P3/P4/P5比例；只改变候选竞争中的局部显著性。借鉴RT-DETR混合Query选择、DINO的Mixed Query Selection及小目标高响应优先的文献动机。关闭时完整调用原 `_get_decoder_input`。
2. **M2 PDN（Pixel-Calibrated Denoising）**：借鉴DINO对比去噪训练，在`get_cdn_group`生成正常DN结构后，保持DN正负Query索引和Mask，**只重新构造可配置像素级扰动的参考框**。针对6—16像素目标与大目标共用比例扰动造成的噪声不均衡。仅训练使用；测试返回原Decoder路径。要求与标准DN单独对照。
3. **M3 TCM（Tiny-conditioned Center Matching）**：保留原匹配类损失及bbox、GIoU代价，增加由GT真实像素尺度控制的归一化中心距离，只有微小框获得较大权重。注意代码中使用GT仅限训练。Hungarian使用SciPy与原工程保持一致。
4. **M4 SNWD（Size-modulated NWD Supervision）**：保留原`L1+GIoU`，在小目标上加入**渐进权重的像素空间NWD项**。原仓库已有普通NWD开关，因此本次新增不是“发明NWD”，而是研究目标尺寸与稳定性调制在DETR多层监督中的收益。训练时通过原`RTDETRDetectionLoss`机制覆盖主层、辅助层和DN层，不改推理模型。

**区别于上一版**：不再依赖局部P2采样、Box Refiner、额外Quality head，全部围绕Query和监督机制展开。

## 论文来源（请逐项在原文和同数据集上对照）

- Yang et al., **UAV-DETR: DETR for Anti-Drone Target Detection**, arXiv:2603.22841 (2026)，专门报告了DUT-ANTI-UAV上的Precision/F1收益；其WTConv、SWSA-IFI、ECFRFN、Inner-CIoU/NWD不能直接等同于本包的新颖性。代码：https://github.com/wd-sir/UAVDETR
- Huang et al., **DEIM: DETR with Improved Matching for Fast Convergence**, CVPR 2025：Dense O2O与MAL，作者论文：https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html
- Wang et al., **RT-DETRv3**, WACV 2025：分层密集正样本监督，论文：https://openaccess.thecvf.com/content/WACV2025/html/Wang_RT-DETRv3_Real-Time_End-to-End_Object_Detection_with_Hierarchical_Dense_Positive_Supervision_WACV_2025_paper.html
- Zhang et al., **DINO**, ICLR 2023：Contrastive Denoising与Mixed Query Selection，论文：https://mlanthology.org/iclr/2023/zhang2023iclr-dino/
- Xu et al., **Detecting Tiny Objects in Aerial Images: A Normalized Wasserstein Distance and a New Benchmark**, ISPRS J. P&RS 2022，论文：https://www.sciencedirect.com/science/article/pii/S0924271622001599
- Peng et al., **D-FINE**, ICLR 2025：更强定位对照，但与本包实现相独立。论文：https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html

## 目录

```
RTDETR_DUT_TinyV2/
├── src/tiny_uav_decoder.py     # M1和M2
├── src/tiny_uav_loss.py        # M3和M4
├── install_tiny_v2.py          # 防覆盖安装器：先dry-run
├── run_tiny_v2.py              # 单实验训练入口（默认dry-run）
├── tests/test_tinyv2.py        # 需要完整仓库与ultralytics环境
├── tests/test_isolated_math.py # 无项目依赖的数值公式检查
├── CODE_AGENT_PROMPT.md
└── README_zh.md
```

## 安装方式

假设压缩包已解压至 `/tmp/RTDETR_DUT_TinyV2`，仓库在 `/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main`：

```bash
cd /tmp/RTDETR_DUT_TinyV2
python install_tiny_v2.py --repo /home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main
# 核对显示的文件清单；确保git status没有意外未提交的变更，再执行
python install_tiny_v2.py --repo /home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main --apply
cd /home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/RTDETR-main
python -m pytest -q /tmp/RTDETR_DUT_TinyV2/tests/test_tinyv2.py
python /tmp/RTDETR_DUT_TinyV2/run_tiny_v2.py --yaml ultralytics/cfg/models/rt-detr/rtdetr_r18_tinyv2_m3_match.yaml --name m3_seed42
# 经确认统一训练协议后才增加 --execute
```

**提示**：原仓库可能存在 `warmup_epochs=2000` 的历史配置。运行前先统一核查此参数及优化器、预训练/EMA、数据集划分。`run_tiny_v2.py`中默认超参数是示范值，不是对历史基线的自动复现。

安装器只修改 `ultralytics/nn/tasks.py` 和 `ultralytics/nn/modules/__init__.py` 两个原有源码文件，并各自生成`.bak`备份；另新增两个Python文件和8份独立YAML，不修改原始 `rtdetr-r18.yaml`。安装器依赖源码文字锚点，若仓库版本改变则明确报错，不会猜测。

## 实验矩阵

| 文件后缀 | 开启内容 |
|---|---|
| off_control | 与原始方法同参数但改为子类，验证输出一致 |
| m1_peak | M1 |
| m2_dn | M2 |
| m3_match | M3 |
| m4_nwd | M4 |
| m1_m3 | M1+M3 |
| m2_m4 | M2+M4 |
| all4 | 四模块联合（只有单模块有效后尝试） |

建议从相同权重、同训练协议启动，先用40 epoch与40 epoch对照筛选，再用150 epoch和多seed确认；**只用验证集选型，最后一次使用测试集**。与2026年UAV-DETR论文跨论文比较时需要统一Split、数据预处理、指标和硬件；不能直接将其+1.4% Precision与本地mAP比较。

## 必须明确的不足

- 本地只有PyTorch CPU，没有原仓库完整依赖及A30，因此这里只能做纯公式与模拟集成检查；**实际模型构建、训练、评估需要在你的原环境执行**。
- M1只有对局部峰值排名做启发式微调，有可能改变Query质量甚至下降。M2 DN随机重构可能导致训练噪声分布转移；若效果变差应退回默认DINO式DN。
- M3、M4均为已知检测技术的条件化改造，不应称作已证明的论文原创。请通过与标准NWD、基础Center Cost、DEIM-MAL和DINO-DN的对照建立价值。
- 核心学术命题建议定为**面向稀疏微小反无人机目标的Query初始化—匹配—监督一致性**，四模块可形成两项核心贡献，但必须经查新与实测后才可以作为论文成果。
