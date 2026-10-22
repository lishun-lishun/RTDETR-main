# SCRE / SPLF A30 experiment runbook

Run from the Linux `RTDETR-main` directory with the existing
`ultralytics_rtdetr` environment. The dataset path is configured in
`dataset/DUT-Anti-UAV.yaml`. Replace the two baseline paths below with the
**completed original R18 run's** actual files; do not substitute an anti-detr
variant or a newly guessed profile.

```bash
BASE_ARGS=/absolute/path/to/original-r18/args.yaml
BASE_BEST=/absolute/path/to/original-r18/weights/best.pt
python -m unittest discover -s tests -p test_scre_splf.py -v
python tools/train_r18_scre_splf.py --model scre_splf --device 1 --protocol frozen --baseline-args "$BASE_ARGS" --baseline-weights "$BASE_BEST" --diagnostic-epochs 10 --train
python tools/diagnose_r18_scre_splf.py --model scre_splf --protocol frozen --split val --device 1
python tools/evaluate_r18_scre_splf.py --model scre_splf --protocol frozen --split val --device 1
```

The launcher is a dry-run unless `--train` is present. It refuses to overwrite
an existing experiment directory. Inspect `candidate_recall.csv` (Top16/32/64,
one-P3-cell radius), auxiliary losses in `results.csv`, F4/F5 same-forward
identity in `feature_statistics.json`, and medium/large validation AP before
formal training. If Top32 coverage is weak, investigate the center targets or
candidate mechanism first. The frozen result is diagnostic only.

After that gate, train separate scratch variants one at a time (or schedule on
separate GPUs if resources permit); the baseline is **not** retrained:

```bash
python tools/train_r18_scre_splf.py --model scre --device 1 --baseline-args "$BASE_ARGS" --train
python tools/train_r18_scre_splf.py --model splf --device 2 --baseline-args "$BASE_ARGS" --train
python tools/train_r18_scre_splf.py --model scre_splf --device 3 --baseline-args "$BASE_ARGS" --train
python tools/train_r18_scre_splf.py --model no_offset --device 1 --baseline-args "$BASE_ARGS" --train
python tools/train_r18_scre_splf.py --model no_protect --device 2 --baseline-args "$BASE_ARGS" --train
```

The five YAMLs implement E1 through E5 respectively. E0 is the already
trained original R18. The optional teacher experiment E6 is not present.
Each train run saves Ultralytics `best.pt`/`last.pt`, `args.yaml`,
`results.csv`, `training_config.json`, `model_summary.json`, and
`execution_log.txt` under `runs/r18_scre_splf/<protocol>/<model>/train`.
Check `best.pt` exists before test evaluation.

Evaluate the same test split and benchmark with the same device, batch,
resolution, warmup, and repeat count for every model:

```bash
python tools/evaluate_r18_scre_splf.py --model baseline --weights "$BASE_BEST" --split test --device 1
python tools/evaluate_r18_scre_splf.py --model scre --split test --device 1
python tools/evaluate_r18_scre_splf.py --model splf --split test --device 1
python tools/evaluate_r18_scre_splf.py --model scre_splf --split test --device 1
python tools/evaluate_r18_scre_splf.py --model no_offset --split test --device 1
python tools/evaluate_r18_scre_splf.py --model no_protect --split test --device 1
python tools/diagnose_r18_scre_splf.py --model scre_splf --split test --device 1
python tools/benchmark_r18_scre_splf.py --model baseline --weights "$BASE_BEST" --device 1 --batch 1 --imgsz 640 --warmup 20 --repeats 100
python tools/benchmark_r18_scre_splf.py --model scre_splf --device 1 --batch 1 --imgsz 640 --warmup 20 --repeats 100
```

Run the benchmark command with `--model scre`, `splf`, `no_offset`, and
`no_protect` as well. Evaluation writes `evaluation_metrics.json` and
`size_group_metrics.csv`; diagnostics write `candidate_recall.csv` and
`feature_statistics.json`; benchmarking writes `benchmark_results.json`.
Their `execution_log.txt` files record completion. Existing output directories
are protected against overwrite; choose a new `--output` path for a repeat.

The size groups are mutually exclusive by maximum side in 640 model-input
pixels: Tiny ≤8, Small (8,16], Medium (16,32], Large >32. The custom
101-point AP uses global confidence ranking, per-image one-to-one IoU
matching, and out-of-group GT ignore. It is **not** official COCO area AP;
candidate recall is a separate non-AP diagnostic. The requested comparison
must report GT count, AP50, AP75, AP50:95, recall, matched IoU and false
positives for every group, plus original Ultralytics overall metrics. Compare
medium/large AP50:95 changes against baseline explicitly, flagging a decline
greater than 0.003. Small AP/overall AP75 targets are research hypotheses,
not expected outputs guaranteed by the code.
