# LDW-ResNet18 experiment protocol

The comparison matrix is E0 original/original, E1 LateDown Stage4, E2
LateDown Stage5, E3 both LateDown, E4 width matched EarlyDown Stage4, and E5
width matched EarlyDown Stage5. Use the five YAML files directly; each keeps
the original RT-DETR head and decoder configuration.

The checked-in protocol is read from `ultralytics/cfg/csp_context_default.yaml`:
150 epochs, batch 8, 640 input, AdamW, `lr0=1e-4`, `lrf=1.0`, `amp=false`, seed
0, and `warmup_epochs=2000`. The repository comment says this trainer interprets
warmup as micro-batch iterations. Before claiming a fair baseline comparison,
pass the historical `args.yaml` with `--baseline-args` and let the training
tool reject mismatches.

## Commands

```bash
python tools/train_r18_ldw.py --model stage4 --device 0
python tools/train_r18_ldw.py --model early_stage4 --device 0
python tools/train_r18_ldw.py --model stage5 --device 0
python tools/train_r18_ldw.py --model early_stage5 --device 0
python tools/train_r18_ldw.py --model full --device 0
```

Each run writes its own directory under `runs/ldw_r18/<model>/seed0` and
preserves `training_config.json`, `model_summary.json`, `epoch_history.json`,
`results.csv`, and `execution_log.txt`. The train command does not overwrite an
existing run and supports `--resume` only from that run's `weights/last.pt`.

## Evaluation and profiling

```bash
python tools/evaluate_r18_ldw.py --model full --weights runs/ldw_r18/full/seed0/weights/best.pt --split test --device 0 --output-dir runs/ldw_r18/full/eval_test
python tools/benchmark_r18_ldw.py --weights runs/ldw_r18/full/seed0/weights/best.pt --device cuda:0 --precision fp32 --output runs/ldw_r18/full/benchmark_results.json
python tools/benchmark_r18_ldw.py --weights runs/ldw_r18/full/seed0/weights/best.pt --device cuda:0 --precision fp16 --output runs/ldw_r18/full/benchmark_fp16.json
python tools/diagnose_r18_ldw.py --weights runs/ldw_r18/full/seed0/weights/best.pt --device cuda:0 --output runs/ldw_r18/full/stage_statistics.json
```

The benchmark reports parameters, trainable parameters, MACs/GFLOPs when
`thop` is available, mean/median/P95 latency, FPS, and peak GPU memory. It
measures raw model forward time only and excludes preprocessing, transfer and
NMS. The evaluation tool defaults to the held-out `test` split for the final
checkpoint. Pass `--split val` explicitly for model selection and ablations.
Accuracy and tiny/small/medium/large AP must be generated from the real
validation/test annotations; no fabricated result is included here.
