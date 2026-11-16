# TinyV2 接入审查与服务器实验清单

## 基准与状态

- 接入前分支：`codex/uav-research-integration`，HEAD `5607e4a011c2efc8dc8f67985c4e6f13a09fa7bc`。
- 安装器先 Dry Run，发现原 `parse_model` 锚点已包含上一版 `UAVResearchDecoder`；适配后再次 Dry Run，再执行 `--apply`。原始 `rtdetr-r18.yaml`、权重和 `runs` 未修改。
- 本机 Python 缺少 `torch`，`python -m pytest -q RTDETR_DUT_TinyV2/tests/test_tinyv2.py` 在测试收集阶段报 `ModuleNotFoundError: No module named 'torch'`。因此实际模型构建、FP32/AMP、Forward/Backward、Checkpoint 和 A30 延迟均**未运行**。只通过 Python 编译、YAML 静态校验、安装器重复 Dry Run 和训练命令 Dry Run。不得将其写为运行通过。
- 本包的 `run_tiny_v2.py` 是示例入口，默认 warmup=3，与本仓库 R18 协议不一致。实验使用 `tools/train_uav_ablation.py`。

## 接入点

| 模块 | 代码 | 开关为零时 | 推理开销 |
| --- | --- | --- | --- |
| M1 LPQ | `TinyUAVDecoder._get_decoder_input` 的 Encoder 排名 | 调用原方法 | 有局部均值池化与排序分数计算 |
| M2 PDN | `TinyUAVDecoder.forward/_pixel_dn` | 调用原 `forward/get_cdn_group` | 无 |
| M3 TCM | `TinyCenterHungarianMatcher.forward` | 用原 Matcher | 无 |
| M4 SNWD | `TinyUAVRTDETRLoss._get_loss_bbox` | 用原 L1+GIoU | 无 |

所有 YAML 使用原 Backbone/Neck 与 `[[19,22,25],1,TinyUAVDecoder,[nc,256,300,4,8,3]]`，即仅 P3/P4/P5、300 个正常 Query。DN Query 由训练时 `get_cdn_group` 生成，经 `dn_num_split` 与正常 Query 分开进入 Loss。M2 沿用原始 DN 元数据和注意力 Mask，仅改参考框。M3 和 M4 只在 Loss 中使用 GT。

## 服务器验证命令

在仓库根目录、已配置 PyTorch 的环境中运行：

```bash
python -m pytest -q RTDETR_DUT_TinyV2/tests/test_tinyv2.py RTDETR_DUT_TinyV2/tests/test_isolated_functions.py RTDETR_DUT_TinyV2/tests/test_isolated_math.py
python -m pytest -q RTDETR_DUT_TinyV2/tests/test_tinyv2_server.py
python tools/train_uav_ablation.py --model tinyv2_off_control --device 0 --epochs 40 --batch 8 --micro-batch 8 --seed 0
```

上述测试覆盖模型 YAML 解析、Decoder state_dict 兼容、关闭开关的输出一致性、Checkpoint 保存加载、Batch 1/2、0/1/多 GT、DN 索引、Matcher、NWD 1 像素偏移与梯度；CUDA 专项文件覆盖完整模型 FP32/AMP 训练步和梯度有限性。服务器仍须补跑原始 R18 权重迁移和 A30 同步延迟；本地无这些结果。

## 统一训练命令

选择 `--device 0` 或其他 CUDA 编号。以下默认从同一随机种子构造模型；若使用原始 R18 初始化权重，**每条命令都加相同的** `--init-weights /absolute/path/to/rtdetr-r18.pt`。建议同样加 `--baseline-args /absolute/path/to/original/args.yaml` 核对历史参数。短程 40 Epoch 必须与同协议原始 R18 一同运行。

```bash
python tools/train_uav_ablation.py --model e0_baseline        --device 0 --epochs 40 --batch 8 --micro-batch 8 --seed 0 --execute
python tools/train_uav_ablation.py --model tinyv2_off_control --device 0 --epochs 40 --batch 8 --micro-batch 8 --seed 0 --execute
python tools/train_uav_ablation.py --model tinyv2_m1_peak     --device 0 --epochs 40 --batch 8 --micro-batch 8 --seed 0 --execute
python tools/train_uav_ablation.py --model tinyv2_m2_dn       --device 0 --epochs 40 --batch 8 --micro-batch 8 --seed 0 --execute
python tools/train_uav_ablation.py --model tinyv2_m3_match    --device 0 --epochs 40 --batch 8 --micro-batch 8 --seed 0 --execute
python tools/train_uav_ablation.py --model tinyv2_m4_nwd      --device 0 --epochs 40 --batch 8 --micro-batch 8 --seed 0 --execute
```

第二批只在单项有实际增益后执行，四模块联合只在其独立项稳定有效后执行：

```bash
python tools/train_uav_ablation.py --model tinyv2_m1_m3 --device 0 --epochs 40 --batch 8 --micro-batch 8 --seed 0 --execute
python tools/train_uav_ablation.py --model tinyv2_m2_m4 --device 0 --epochs 40 --batch 8 --micro-batch 8 --seed 0 --execute
python tools/train_uav_ablation.py --model tinyv2_all4  --device 0 --epochs 40 --batch 8 --micro-batch 8 --seed 0 --execute
```

正式训练将同一命令的 `--epochs 40` 改为 `--epochs 150`，对有价值的模型分别改 `--seed 0`、`1`、`2`，并为每个种子运行相应 R18 基线。协议来自 `ultralytics/cfg/csp_context_default.yaml`：AdamW、lr0=0.0001、lrf=1、weight decay=0.0001、warmup_epochs=2000（本 Trainer 中是 micro-batch 迭代）、AMP=false、imgsz=640、batch=micro-batch=8。若历史 R18 的实际 `args.yaml` 不同，应先统一协议。

验证集评估示例（脚本默认 `val`，测试集仅用于最终报告）：

```bash
python tools/evaluate_tinyv2.py --weights runs/uav_ablation/tinyv2_m1_peak/seed0/weights/best.pt --device 0 --split val --batch 8 --name m1_seed0_val
```

该评估器报告 AP50、AP75、AP50:95。Small/Medium/Large AP、自定义微小框定位误差、GFLOPs 和严格 A30 Batch1 FP16 同步延迟尚无统一实现，不能以其他字段冒充；正式论文结果需要在服务器按统一评估器补齐。

## 消融表

| 模型 | AP50 | AP75 | AP50:95 | Small AP | 结论 |
| --- | --- | --- | --- | --- | --- |
| R18 40ep 同协议 | 未运行 | 未运行 | 未运行 | 未运行 | 待对照 |
| off_control | 未运行 | 未运行 | 未运行 | 未运行 | 验证零开关 |
| M1 | 未运行 | 未运行 | 未运行 | 未运行 | 待实验 |
| M2 | 未运行 | 未运行 | 未运行 | 未运行 | 待实验 |
| M3 | 未运行 | 未运行 | 未运行 | 未运行 | 待实验 |
| M4 | 未运行 | 未运行 | 未运行 | 未运行 | 待实验 |
| M1+M3 | 未运行 | 未运行 | 未运行 | 未运行 | 条件启动 |
| M2+M4 | 未运行 | 未运行 | 未运行 | 未运行 | 条件启动 |
| all4 | 未运行 | 未运行 | 未运行 | 未运行 | 条件启动 |

用户提供的历史 R18 AP50=.966、AP75=.816、AP50:95=.707 不是本协议当前重跑结果，不能直接填入上表。

## 新颖性初判

| 模块 | 最接近工作 | 当前判断 |
| --- | --- | --- |
| M1 | [DINO](https://arxiv.org/abs/2203.03605) 已提出混合 Query 选择；2026 年 [ROI-DETR](https://www.nature.com/articles/s41598-026-72701-w) 也针对小目标改进 Query 选择 | 局部峰值奖励是可检验的实现假设，单独作为强论文原创性不足；需证明 GT Query 覆盖率与跨尺度稳定收益 |
| M2 | [DINO](https://arxiv.org/abs/2203.03605) 的对比去噪与本仓库 CDN | 像素标定的噪声分布是应用改造；需直接对照原 DN、仅像素扰动与不同正负幅度 |
| M3 | [DEIM](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html) 的匹配质量，[RT-DETRv3](https://openaccess.thecvf.com/content/WACV2025/html/Wang_RT-DETRv3_Real-Time_End-to-End_Object_Detection_with_Hierarchical_Dense_Positive_Supervision_WACV_2025_paper.html) 的密集监督 | 具体 GT 尺度条件化中心代价有可检验的机制差异，但仍需对照均匀 Center Cost、原 Matcher 和 DEIM 思路 |
| M4 | 原 [NWD](https://arxiv.org/abs/2206.13996)、同数据集 [UAV-DETR 2026](https://arxiv.org/abs/2603.22841) 已用 NWD；本仓库已有普通 NWD 开关 | 重叠风险最高；只可声称尺度调制的消融价值，不能声称发明 NWD |

[D-FINE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html) 通过分布细化和自蒸馏提高定位精度，与本实现机制不同，但应作为强定位参照。当前四项均**未有性能证据**，因此没有可保留或淘汰的实证结论。论文主线暂可假设为“稀疏微小目标的 Query 初始化与匹配/监督一致性”，待同协议消融和多种子验证后再确定贡献。
