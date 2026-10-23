# RB-HRNet / LAF port for RTDETR-main

The new code is in **this** `RTDETR-main` Ultralytics tree. The neighboring
`RT-DETR/rtdetr_pytorch` implementation supplied the audited HRNetV2-W18
stage, transition, branch and fusion structure; the model graph and training
entry here use Ultralytics YAML and `train_dut_anti_uav.py`.

The Ultralytics tree previously contained no historical HRNet YAML, weight or
training log. The old HRNet-vs-R18 reported AP values therefore cannot be
verified as a controlled Ultralytics comparison. The new YAMLs initialize
HRNet from scratch (`pretrained: false`), matching this tree's checked-in
Anti-DETR training profile. The supplied historical HRNet checkpoint was not
found locally and is not silently loaded.

## Architecture

- Standard HRNetV2-W18: 1/4, 1/8, 1/16, 1/32 channels 18/36/72/144.
- RB-HRNet-A: 24/36/72/144.
- RB-HRNet-B: 24/36/64/112.
- The HRNet layer returns P3/P4/P5 only. `HRNetFeature` selects each scale;
  the existing 1x1 projections supply 256 channels to the original FPN/PAN.
- LAF-0 changes only the P4-to-P3 top-down upsample to learned bilinear
  `grid_sample`. Zero offsets give regular bilinear sampling, not nearest.
- LAF-1 keeps the complete original FPN/PAN and adds an isolated F4-to-F3
  alignment after it. `0.3*tanh(alpha_raw)` starts at zero, so its decoder F3
  is initially exactly unchanged. Decoder F4/F5 keep the original path.
- Every new YAML retains P3/P4/P5 and the original 300-query RTDETRDecoder.
  No P2 detector head or new loss was added.

The new modules are in `ultralytics/nn/modules/hrnet.py`, `rb_hrnet.py` and
`laf.py`. `ultralytics/nn/tasks.py` handles their input/output channel
contracts. The five candidate YAMLs and a plain HRNet control are under
`ultralytics/cfg/models/rt-detr/`.

## Training

From `RTDETR-main` on the server, with physical GPU 1:

```bash
python -u tools/train_rb_hrnet_laf.py --model rb_hrnet_a --device 1
python -u tools/train_rb_hrnet_laf.py --model rb_hrnet_b --device 1
python -u tools/train_rb_hrnet_laf.py --model hrnet_laf0 --device 1
python -u tools/train_rb_hrnet_laf.py --model hrnet_laf1 --device 1
```

The wrapper invokes the existing `train_dut_anti_uav.py` with 150 epochs,
one GPU, effective batch 8, micro-batch 8, 640 input, AdamW, LR 0.0001,
weight decay 0.0001, seed 0 and AMP disabled. This fork interprets the profile
`warmup_epochs: 2000` as **2000 optimizer iterations**; with accumulation 1
that remains 2000 iterations. Runs go to `runs/rb-hrnet-laf/train/`, separately
from prior results. Use `--dry-run` to inspect the exact command first.
Train `rb_hrnet_laf1` only after examining the independent experiments.

## Verification and remaining server work

The local standalone tests passed HRNet stage/branch output and gradient
checks, LAF coordinate sampling and two-step gradient propagation, initial
LAF-1 identity, shape-matched partial weight loading, and YAML decoder
contracts. Python compilation and training dry-run passed. The full
Ultralytics build/forward and CUDA AMP tests are included but skipped locally
because the available Windows Python environments lack this zoo's complete
dependencies (`cv2`, modern `timm.layers`, and others), or a CUDA device.
Run on the
server before training:

```bash
python -m unittest discover -s tests -p test_rb_hrnet_laf.py -v
```

The formal AP, GPU latency, checkpoint loading coverage, AMP and size-group
metrics are not claimed yet. `tools/evaluate_rb_hrnet_laf.py` runs validation
and outputs overall COCO AP and Tiny/Small/Medium/Large maximum-edge AP at
640 input. `benchmark_rb_hrnet_laf.py` reports profiler-supported FLOPs as a
lower bound. `diagnose_rb_hrnet_laf.py` reports offsets and F3 residual on
one image. The evaluator writes `localization_analysis.csv` with detection
confidence and the maximum same-class GT IoU as a diagnostic; it is not an
AP calculation. Foreground/background residuals remain to be completed after
server data collection.
