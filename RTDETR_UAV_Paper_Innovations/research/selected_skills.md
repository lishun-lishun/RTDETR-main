# 02 已安装 Skills 审计与选择报告（Selected Skills Report）

## 1. 本地环境 Skills 扫描结果

按照任务要求，首先对当前运行环境中的本地目录（`.agents/skills/`、`.codex/skills/`、`.claude/skills/` 及 `$HOME` 下的对应路径）以及系统预置的 Agent Skills 进行了全面扫描与核对：

1. **工作区与用户目录扫描**：
   - 执行 `ls -la .agents/skills/ .codex/skills/ .claude/skills/ ~/.agents/skills/ ~/.codex/skills/ ~/.claude/skills/`：当前沙箱工作目录 `/working_dir/c_a51f1d057f65c040` 与 `/home/spark` 下未挂载外部 `.agents/skills` 或 `.codex/skills` 文件夹（如外部环境中可能存在的 `rigor-reviewer` 或 `ml-training-recipes` 在当前沙箱容器中未安装，严格遵守“不得虚构未安装的 Skills”原则，不伪造调用记录）。
2. **环境中实际可用并已验证的 Skills**：
   - 通过技能检索工具读取了当前环境实际安装的 `SKILL.md` 文档，确认了面向**代码与文件生成工程规范**、**文献与网页深度检索**及**跨文件分析**的核心能力模块。

## 2. 适用本项目的 Skills 职责与调用策略

| 序号 | 实际核实的 Skill 标识 / 能力域 | 核心职责说明 | 在本任务中的具体应用场景 |
| :--- | :--- | :--- | :--- |
| 1 | `file-generation`（源码与科研文档工程生成规范） | 规范化管理 Python 源码模块（`.py`）、配置文件（`.yaml`）、对照矩阵（`.csv`）、研究报告（`.md`）及交付产物的多阶段规划、构建与程序化校验流程。 | 指导 `innovation_1`–`innovation_4` 源码实现、`integration.patch` 生成、`novelty_matrix.csv` 构建及 `RTDETR_UAV_Paper_Innovations.zip` 打包验证。 |
| 2 | `web-browsing-v2`（学术文献与开源代码检索路由指南） | 指导在 `google:search` 与 `google:browse` 之间进行高效链路组合：先通过精确关键词检索 arXiv / IEEE / CVPR / ICLR / Sensors 论文与 GitHub 源码，再通过全文浏览提取精确数学公式、实验表格与函数签名。 | 用于检索并核对 2024–2026 年 22 篇代表性论文（如 D-FINE、DEIM、UAV-DETR 2025/2026、TQ-DETR 2026、ReDHF 2026、FSDC-DETR 2026）及 Ultralytics YOLOv8.0.201 RT-DETR 核心源码。 |
| 3 | `file-understanding`（大型代码库与实验日志解析策略） | 提供基于 `find`、`rg`、`git grep`、结构化脚本解析 CSV/JSON/YAML 日志与张量形状追踪的高效审计方法，避免盲目遍历全仓文件导致上下文污染。 | 用于解析 `rtdetr-r18.yaml`、`RTDETRDecoder`、`RTDETRDetectionLoss` 调用图，以及对 `runs/uav_ablation/` 历史训练轮数与实验协议进行严格归因。 |

## 3. 科研审稿与实验设计内建准则（Rigor & Recipe Guardrails）

虽然本地 `.agents/skills/` 目录下未单独挂载外部 `rigor-reviewer` 与 `ml-training-recipes` 文件，但我们在执行全流程时将严格内嵌以下四项科研严谨性准则：
1. **反缝合怪准则（Anti-Frankenstein Principle）**：拒绝将互不相干的注意力模块（如 CBAM + SE + ECA）简单堆叠。所有四个创新点必须围绕“**远距离微小无人机在降采样、跨尺度融合、稀疏变形采样与一对一匹配中的信号退化与定位失配**”这一统一核心科学问题展开。
2. **训练协议公平性准则（Fair Training Recipe）**：固定输入分辨率 `640×640`、固定优化器 `AdamW(lr=1e-4, weight_decay=1e-4)`、固定训练轮数（150 Epochs）、固定预训练初始化策略与 EMA 衰减率，严禁跨不同训练轮数（如 68 轮 vs 150 轮）或跨数据集划分（Validation vs Test）直接比大小。
3. **契约兼容与可拆卸准则（Detachment & Contract Invariance）**：所有创新模块必须提供 `enabled: bool = True` 开关；当 `enabled=False` 时，前向传播在数学上严格退化为原始 `RT-DETR-R18`（误差 $< 10^{-6}$），且不破坏 `[B, 300, 4+nc]` 的输出契约。
4. **审稿人视角自查（Reviewer Self-Critique）**：对每一项创新点明确指出其与最接近前沿工作（如 [UAV-DETR 2026](https://arxiv.org/html/2603.22841v1)、[TQ-DETR 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667564/)、[D-FINE 2025](https://arxiv.org/html/2410.13842v1)、[DEIM 2025](https://arxiv.org/html/2412.04234v3)）的本质数学差异、潜在计算开销与适用边界。
