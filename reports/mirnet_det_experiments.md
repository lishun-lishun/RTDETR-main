# MIRNet-Det experiment matrix

| Model | Backbone configuration | Purpose |
|---|---|---|
| `mir_base` | no cross-scale exchange | Stem and branch baseline |
| `mir_dual` | two S8/S16 exchanges | Dual-resolution ablation |
| `mir_full` | dual plus two S8/S16/S32 exchanges | Full MIRNet-Det |
| `mir_stride` | full exchanges with stride-2 S4 to S8 control | PixelUnshuffle ablation |
| `mir_nofusion` | exchange blocks with cross-scale paths disabled | Fusion-off diagnostic |
| `mir_zero_fusion` | exchange paths enabled with fixed alpha=0 | Zero-fusion diagnostic |

The first pass uses the checked-in R18 protocol: 150 epochs, batch 8, 640
input, AdamW, `lr0=1e-4`, `lrf=1.0`, weight decay `1e-4`, seed 0,
`amp=false`, and `warmup_epochs=2000`. Supply the historical baseline
`args.yaml` with `--baseline-args` to make the tool reject protocol drift.

Train one configuration at a time:

```bash
python tools/train_mirnet_det.py --model mir_base --device 0
python tools/train_mirnet_det.py --model mir_dual --device 0
python tools/train_mirnet_det.py --model mir_full --device 0
python tools/train_mirnet_det.py --model mir_stride --device 0
```

The tool runs the MIRNet unit tests and a 128px model/head preflight before a
new training run. It writes configuration, model summary, per-epoch history,
learning-rate columns and execution log under `runs/mirnet_det/<model>/seed0`.
It never starts training merely by importing the script and refuses to
overwrite an existing run.

Final held-out test evaluation:

```bash
python tools/evaluate_mirnet_det.py --model mir_full \
  --weights runs/mirnet_det/mir_full/seed0/weights/best.pt \
  --split test --device 0 --output-dir runs/mirnet_det/mir_full/eval_test
```

Pass `--split val` during model selection. If a COCO annotation JSON is
available, also pass `--annotations` to generate custom Tiny/Small/Medium/Large
AP, recall, matched IoU and localization CSV files.

Benchmark and diagnose the full model only after its checkpoint exists:

```bash
python tools/benchmark_mirnet_det.py --weights runs/mirnet_det/mir_full/seed0/weights/best.pt \
  --device cuda:0 --precision fp32 --warmup 50 --iterations 200 \
  --output runs/mirnet_det/mir_full/benchmark_fp32.json
python tools/benchmark_mirnet_det.py --weights runs/mirnet_det/mir_full/seed0/weights/best.pt \
  --device cuda:0 --precision fp16 --warmup 50 --iterations 200 \
  --output runs/mirnet_det/mir_full/benchmark_fp16.json
python tools/diagnose_mirnet_det.py --weights runs/mirnet_det/mir_full/seed0/weights/best.pt \
  --device cuda:0 --output runs/mirnet_det/mir_full/branch_statistics.json
```
