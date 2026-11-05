# DLA34-RTDETR experiment protocol

## Planned experiments

| ID | Backbone | Neck | Initialization |
|---|---|---|---|
| E0 | original R18 | original CCFM | historical R18 baseline |
| E1 | HRNetV2-W18 | original CCFM | separate HRNet baseline |
| E2 | DLA34 | original CCFM | scratch, seed 0 first |
| E3 | DLA34 | original CCFM | optional local DLA-34 pretrained checkpoint |

E2 is the fair structural comparison with scratch R18.  E3 must be reported
as a separate initialization condition and must not be presented as the same
comparison as E2.

## Training protocol

The scripts read `ultralytics/cfg/csp_context_default.yaml`: 150 epochs,
batch 8, 640 input, AdamW, `lr0=1e-4`, `lrf=1.0`, weight decay `1e-4`, seed
0, AMP disabled, and warmup 2000 micro-batch iterations.  A historical R18
`args.yaml` can be passed with `--baseline-args`; the script checks the core
fields against the checked-in profile before training.  Results and the
per-epoch learning-rate columns are saved under `runs/dla34_rtdetr/`.

## Evaluation and test split policy

Model selection uses `--split val`.  The held-out test set is used only for
the final confirmed run with the selected checkpoint.  The evaluator defaults
to `--split test` so that a test command is explicit in the run metadata; pass
`--split val` for validation during development.

## Complexity and size reporting

`benchmark_rtdetr_dla34.py` uses the same profiler path and defaults as the
other backbone tools: 640x640, batch 1, 50 warmups and 200 timed iterations,
with an explicit FP32/FP16 setting.  `evaluate_rtdetr_dla34.py` records native
AP50, AP75, AP50:95, precision and recall and can add Tiny/Small/Medium/Large
groups defined by the largest box side after scaling to the 640 coordinate
system.  It does not claim COCO's official area groups.

## Current limits

No GPU training, external reference checkpoint equivalence, or DLA34 metric
has been claimed from the Windows checkout.  Those values require the Linux
A30 run and are written by the scripts into the selected output directory.
