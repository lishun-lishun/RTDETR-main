# CSP Stage 实验与 Linux 命令

在 Linux 项目根目录运行，物理 GPU 1 对应 `CUDA_VISIBLE_DEVICES=1 ... --device 0`。先把本地修改同步到服务器，运行：

```bash
CUDA_VISIBLE_DEVICES=1 python -m unittest discover -s tests -p test_csp_context_r18.py -v
```

第一轮仅优先 B1、B2；已有 B0 Baseline 复用已有 checkpoint，不重训。B3 联合模型只在 B1/B2 的验证集结果有明确、可解释的收益后运行。B4 只改 Stage4 kernel=3；B5 只改 Stage3 process_ratio=0.5。

```bash
# B1: Stage3 CSPResidual
CUDA_VISIBLE_DEVICES=1 python -u tools/train_r18_csp_context.py --model stage3 --device 0 --seed 0

# B2: Stage4 CSPContext (7x7)
CUDA_VISIBLE_DEVICES=1 python -u tools/train_r18_csp_context.py --model stage4 --device 0 --seed 0

# B3: combined, conditional
CUDA_VISIBLE_DEVICES=1 python -u tools/train_r18_csp_context.py --model combined --device 0 --seed 0

# B4: Stage4 3x3 kernel ablation
CUDA_VISIBLE_DEVICES=1 python -u tools/train_r18_csp_context.py --model stage4_k3 --device 0 --seed 0

# B5: Stage3 0.5 process ratio ablation
CUDA_VISIBLE_DEVICES=1 python -u tools/train_r18_csp_context.py --model stage3_ratio05 --device 0 --seed 0
```

若有原始 Baseline `args.yaml`，建议在各命令末尾追加 `--baseline-args /path/to/args.yaml`。脚本保存到 `runs/csp_context_r18/<model>/seed0/`，已有目录会阻止新训练。恢复同一运行示例：

```bash
CUDA_VISIBLE_DEVICES=1 python -u tools/train_r18_csp_context.py --model stage3 --device 0 --seed 0 --resume runs/csp_context_r18/stage3/seed0/weights/last.pt
```

训练目录含 `training_config.json`、`model_summary.json`、`args.yaml`、`results.csv`、`epoch_history.json`、权重和 `execution_log.txt`。使用新的输出目录分别验证和最终测试；标注 JSON 必须对应 split：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/evaluate_r18_csp_context.py --model stage3 --weights runs/csp_context_r18/stage3/seed0/weights/best.pt --annotations /path/to/val.json --split val --device 0 --output-dir runs/csp_context_r18/stage3/seed0/eval_val
CUDA_VISIBLE_DEVICES=1 python tools/evaluate_r18_csp_context.py --model stage3 --weights runs/csp_context_r18/stage3/seed0/weights/best.pt --annotations /path/to/test.json --split test --device 0 --output-dir runs/csp_context_r18/stage3/seed0/eval_test
CUDA_VISIBLE_DEVICES=1 python tools/benchmark_r18_csp_context.py --weights runs/csp_context_r18/stage3/seed0/weights/best.pt --device cuda:0 --output runs/csp_context_r18/stage3/seed0/benchmark_results.json
CUDA_VISIBLE_DEVICES=1 python tools/diagnose_r18_csp_context.py --weights runs/csp_context_r18/stage3/seed0/weights/best.pt --image /path/to/example.jpg --annotations /path/to/val.json --device cuda:0 --output runs/csp_context_r18/stage3/seed0/csp_diagnostics.json
```

`evaluate` 写整体 AP50/AP75/AP50:95、P/R、自定义尺寸组 AP/Recall/匹配 IoU 和候选框定位表。Tiny/Small/Medium/Large 用 640 letterbox 后最大边 ≤8、(8,16]、(16,32]、>32 像素，**不是** COCO 官方 APs/APm/APl。正式 AP 以 COCOeval 为准，候选框定位统计仅作诊断。结构筛选用验证集，测试集用于最终评估。

原始 R18 的已有权重也可用 `evaluate_r18_csp_context.py --model baseline --weights /path/to/original/best.pt` 在相同协议下重新评估；`benchmark_r18_csp_context.py` 同样接受该权重。不会重新训练 Baseline。
