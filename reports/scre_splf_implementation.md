# RT-DETR-R18 SCRE / SPLF implementation review

## Scope and topology

The five added YAMLs are generated from `rtdetr-r18.yaml` by
`tools/generate_scre_splf_yaml.py`. The original R18 backbone, AIFI, CCFM, and
three-scale 300-query decoder rows remain unchanged, except for necessary
index shifts and the new late F3 connection. The first ordinary neck layer is
explicitly connected to original P5 (layer 7), not the inserted SCRE layer.

At 640 input the intended routes are P2 layer 4 (64×160×160) and P3 layer 5
(128×80×80) into SCRE layer 8; original P3 remains the Stage4 input. Original
P4/P5 remain on the normal backbone/CCFM route. SCRE outputs 64 sparse
channels and one center-logit channel on the P3 grid. E1 uses fixed-gain late
injection; E3/E4/E5 use SPLF after complete CCFM. SPLF only changes final F3.
Decoder input is `[F3_new, F4_original, F5_original]`; no P2 decoder scale or
extra queries are added. E2 uses an independent F3-based candidate/feature
head, not SCRE or P2.

SCRE selects at most 32 distinct spatial positions per image. It samples 5×5
P2 patches with `grid_sample(align_corners=False)`, applies a 1×1 projection
and depthwise/pointwise local encoding on sampled patches only, predicts a
bounded P3-cell offset, and uses normalized bilinear `scatter_add` to create
M3. `TopK` selection itself is not differentiable; a GT-center focal loss
supervises the heatmap directly. Nearest eligible candidate matching within
one P3 cell provides one-to-one offset supervision. If no candidate covers a
small GT, it has no offset loss; candidate recall is therefore a prerequisite
diagnostic, not a guaranteed property.

SPLF uses `F3 + 0.1*tanh(alpha_raw)*sigmoid(center_logits)*tanh(project(M3)/rms)`.
`alpha_raw=0` gives exact initial identity. The new residual is elementwise
bounded in magnitude by 0.1. On the first optimization step, the detection
loss reaches the gain, while direct center/offset auxiliary supervision reaches
SCRE; once the gain moves, detection gradients also reach M3. This is an
initialization behavior, not a guarantee that small-object AP will improve.

The total loss is the original RT-DETR loss plus separately logged weighted
center, offset, and protection terms. Defaults in each YAML are
`lambda_center=0.05`, `lambda_offset=0.1`, `lambda_protect=0.01`. E2 disables
offset, E1 disables protection, E4 removes the offset head/loss, and E5
disables protection. The protection target is current-forward original CCFM
F3; it penalizes only the new F3 residual outside a small-box-plus-margin
region. It does **not** prevent the original backbone/CCFM from changing
during scratch training, nor does it prove medium/large AP preservation.
Teacher distillation (optional E6) is not implemented or enabled.

## Coordinate and training-protocol audit

Targets come from the actual training batch's normalized `bboxes` and `img`
dimensions, after the fork's resize/augmentation. This fork's RT-DETR dataset
uses **stretch to square**, not letterbox (`RTDETRDataset.load_image(...,
rect_mode=False)` and `v8_transforms(..., stretch=True)`). The targets are
aligned to the actual 640 model input, but this is a meaningful difference
from the task's letterbox wording. Do not change this transform for only the
new experiments; changing it would invalidate baseline parity.

`warmup_epochs` is misleadingly named here. In `ultralytics/engine/trainer.py`,
`nw = self.args.warmup_epochs` and `nw` is compared with the training batch
iteration count. Thus a value of 2000 means **2000 iterations**, not 2000
epochs. The new launcher copies the actual baseline `args.yaml` and does not
silently correct this. Inspect that file and the completed baseline log before
claiming a paper-faithful schedule.

The training launcher requires the real baseline `args.yaml` and checks
150 epochs, 640 image, AdamW, lr0 1e-4, lrf 1.0, weight decay 1e-4, seed 0, AMP off,
pretrained off, and effective train batch 8. Other training settings are
copied from that record. The frozen diagnostic is **not** a fair 150-epoch
comparison: it transfers the baseline main-path weights and freezes original
backbone/CCFM/decoder parameters. Formal scratch runs initialize afresh and
must be compared only against the original scratch baseline.

## Verification status and limitations

Local Windows checks: Python compilation and YAML topology/custom-AP unit
tests pass. This local Python environment has no PyTorch, so module forward,
autograd, CUDA/AMP, full-decoder hook, and actual training tests cannot run
here; the nine Torch tests are skipped locally. `--train` invokes all preflight
tests on the Linux server before starting. **Do not treat a local skip as GPU
validation.** No server training, validation, benchmark, or measured
parameter/GFLOP/AP improvement is claimed in this report.

On the A30, first run the complete test command in the experiments report and
inspect the full-size hook assertions. Use the benchmark tool to record true
forward latency/peak memory. Its PyTorch-profiler FLOPs and MACs are marked
*incomplete estimates* because sparse kernels such as TopK, grid sampling and
scatter are not fully counted. Report those timings separately; do not call
them exact whole-model GFLOPs.
