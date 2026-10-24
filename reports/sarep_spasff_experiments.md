# SA-Rep / SP-ASFF experiment matrix

| ID | YAML / launcher model | Backbone change | Final F3 change | Status |
|---|---|---|---|---|
| E0 | `rtdetr-r18.yaml` | None | None | Previously trained; do not retrain. Local checkpoint/log unavailable. |
| E1 | `sarep` | Stage3 block 2 Conv2 SA-Rep | None | First priority; pending server training. |
| E1b | `sarep_both` | Stage3 block 1+2 Conv2 | None | Optional configuration only. |
| E2 | `spasff` | None | Protected difference residual | First priority; pending server training. |
| E3 | `sarep_spasff` | E1 | E2 | Train only if independent results justify it. |
| E4 | `sarep_stage34` | Stage3+Stage4 block 2 Conv2 | None | Optional configuration only. |
| E5 | `asff_direct` | None | Direct fused-feature replacement of F3 | Optional control; same F3/F4/F5 sources and alignment as E2. |
| E6 | `spasff_no_difference` | None | Bounded residual using fused feature without subtracting projected F3 | Optional control. |

All new runs use 150 epochs, one GPU, effective batch 8, image size 640, AdamW, lr0=0.0001, lrf=1, weight decay=0.0001, seed=0, AMP off, random model initialization. Supply the original R18 `args.yaml` to confirm all other settings. No measured results have been filled in.

Evaluation requires the matching COCO GT JSON on the server:

```bash
python -u tools/evaluate_r18_sarep_spasff.py --model sarep --weights runs/sarep_spasff/sarep/weights/best.pt --annotations /path/to/val.json --split val --device 0 --output-dir runs/sarep_spasff/sarep/evaluation
python -u tools/diagnose_r18_sarep_spasff.py --weights runs/sarep_spasff/sarep/weights/best.pt --image /path/to/image.jpg --annotations /path/to/val.json --output runs/sarep_spasff/sarep/diagnostics.json
```

The size groups use each COCO box's maximum side after the 640-input letterbox scale `min(640/W, 640/H)`; padding changes position but not box size. COCOeval area-range ignores out-of-group GT and filters unmatched predictions by the same group definition. Mean matched IoU is a separate greedy, one-to-one localization diagnostic with a 0.5 IoU threshold; it is not AP. `size_group_metrics.csv` records GT count, AP50, AP75, AP50:95, recall, and matched IoU. `localization_analysis.csv` records prediction scores and IoUs. SP-ASFF diagnostics report full-map scale weight mean/std, foreground/background means and size-group means when annotations are supplied, plus same-shape residual RMS and L2 ratios.

| Model | Seed | Overall AP50 | AP75 | AP50:95 | Tiny AP50:95 | Small AP50:95 | Medium AP50:95 | Large AP50:95 | FPS | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E0 | 0 | 0.966 (user reported) | 0.816 (user reported) | 0.707 (user reported) | — | — | — | — | — | Local log unavailable; verify protocol. |
| E1 | 0 | — | — | — | — | — | — | — | — | Pending. |
| E2 | 0 | — | — | — | — | — | — | — | — | Pending. |
| E3 | 0 | — | — | — | — | — | — | — | — | Conditional. |
