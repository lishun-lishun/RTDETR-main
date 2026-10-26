# 实验顺序与服务器命令

## 实验矩阵

| 编号 | `--model` | 内容 | 执行顺序 |
|---|---|---|---|
| E0/B0/N0 | 原始 `rtdetr-r18.yaml` | 已有 Baseline checkpoint | 复用，不重训 |
| E1/B1 | `res2basic` | Stage3 单处 scale=4 + 原 CCFM | 第一轮 |
| E2/N1 | `litegd` | 原 Backbone + 完整 Lite-GDNeck | 第一轮 |
| E3 | `res2basic_litegd` | 联合模型 | E1/E2 有价值后 |
| B2 | `res2basic_scale2` | 单处 scale=2 | 可选 |
| B3 | `res2basic_both` | Stage3 两处 | 可选 |
| B4 | `res2basic_stage34` | Stage3、Stage4 各一处 | 可选 |
| B5 | `res2basic_independent` | 与 B1 同宽度，关闭层级连接 | 机制消融 |
| N2 | `litegd_high_only` | 仅高层 Gather/Distribute | 机制消融 |
| N3 | `litegd_low_only` | 仅低层 Gather/Distribute | 机制消融 |
| N4 | `litegd_local_only` | 仅本尺度投影/细化 | 容量对照 |
| N5 | `litegd_p3_only` | 融合三层，仅向 P3 注入 | 尺度分配消融 |

共 12 种条件，包括沿用的 E0；新增 11 个 YAML。不要默认训练全部。首轮 E1、E2 和 B5；E3 按验证结果再决定。首轮 seed 0，有效结构至少再用两个种子复现。所有运行使用同一数据划分和训练协议。

## 单卡训练

在 Linux 项目根目录执行。若希望使用物理 GPU 1，设置 `CUDA_VISIBLE_DEVICES=1` 后脚本中的 `--device 0` 指向这张卡。优先传入真实 Baseline 的 `args.yaml`，例如下列 `--baseline-args`；若找不到该文件，先核对仓库内 `anti_detr_default.yaml` 与旧训练记录。

```bash
CUDA_VISIBLE_DEVICES=1 python -u tools/train_r18_res2_gd.py --model res2basic --device 0 --seed 0 --baseline-args /path/to/baseline/args.yaml
CUDA_VISIBLE_DEVICES=1 python -u tools/train_r18_res2_gd.py --model litegd --device 0 --seed 0 --baseline-args /path/to/baseline/args.yaml
CUDA_VISIBLE_DEVICES=1 python -u tools/train_r18_res2_gd.py --model res2basic_independent --device 0 --seed 0 --baseline-args /path/to/baseline/args.yaml
```

E3 与其他消融的命令只需替换 `--model` 为上表名称。例如联合模型：

```bash
CUDA_VISIBLE_DEVICES=1 python -u tools/train_r18_res2_gd.py --model res2basic_litegd --device 0 --seed 0 --baseline-args /path/to/baseline/args.yaml
```

输出路径为 `runs/res2basic_litegd/<model>/seed<seed>/`。已有目录会阻止新训练。恢复同一运行：

```bash
CUDA_VISIBLE_DEVICES=1 python -u tools/train_r18_res2_gd.py --model res2basic --device 0 --seed 0 --baseline-args /path/to/baseline/args.yaml --resume runs/res2basic_litegd/res2basic/seed0/weights/last.pt
```

训练输出包含 `training_config.json`、`model_summary.json`、`args.yaml`、`results.csv`、`epoch_history.json`、权重和 `execution_log.txt`。`results.csv` 中的 `lr/pg*` 和训练 loss 是 Trainer 每轮真实记录；检查学习率曲线后再解释实验结果。

## 验证、测试和分析

验证集与测试集分别运行，`--annotations` 必须匹配所选 split；同一个模型分别使用新的输出目录：

```bash
python tools/evaluate_r18_res2_gd.py --model res2basic --weights runs/res2basic_litegd/res2basic/seed0/weights/best.pt --annotations /path/to/val.json --split val --device 0 --output-dir runs/res2basic_litegd/res2basic/seed0/eval_val
python tools/evaluate_r18_res2_gd.py --model res2basic --weights runs/res2basic_litegd/res2basic/seed0/weights/best.pt --annotations /path/to/test.json --split test --device 0 --output-dir runs/res2basic_litegd/res2basic/seed0/eval_test
python tools/benchmark_r18_res2_gd.py --weights runs/res2basic_litegd/res2basic/seed0/weights/best.pt --device cuda:0 --output runs/res2basic_litegd/res2basic/seed0/benchmark_results.json
python tools/diagnose_r18_res2_gd.py --weights runs/res2basic_litegd/res2basic/seed0/weights/best.pt --image /path/to/example.jpg --annotations /path/to/val.json --device cuda:0 --output-dir runs/res2basic_litegd/res2basic/seed0/diagnostics
```

自定义 Tiny/Small/Medium/Large 分组按 640 letterbox 后框的最大边分为 ≤8、(8,16]、(16,32]、>32 像素，**不是** COCO 官方 APs/APm/APl。脚本使用 COCOeval 分组计算 AP；`localization_analysis.csv` 是候选框定位诊断，不能替代 AP。测速固定 FP32、batch 1、640，记录 GPU 型号、warmup、循环数、前向延迟/FPS/显存；不含预处理、传输及后处理。profiler FLOPs 是其支持算子的下界。

筛选时记录整体 AP50/AP75/AP50:95、各尺寸 AP/Recall/匹配 IoU、参数量、FLOPs 下界和速度。若 Medium 或 Large AP50:95 比 Baseline 低超过 0.003，单独标记并结合 B5、N2–N5 与内部残差统计分析。0.003 是筛选阈值，不代表统计显著性。目标 mAP50:95 >0.707、mAP75 ≥0.816 尚未验证。
