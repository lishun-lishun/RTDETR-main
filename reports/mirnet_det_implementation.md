# MIRNet-Det implementation

## Implemented files

- `ultralytics/nn/modules/mirnet_det.py`: `MIRStem`,
  `SpatialRearrangeDownsample`, `StrideDownsample`,
  `EfficientInvertedBlock`, `DualResolutionExchange`,
  `TriResolutionExchange`, `MIRDetBackbone`, and `FeatureSelect`.
- `ultralytics/nn/tasks.py`: parser handling for the multi-output backbone and
  explicit feature selectors. The original RT-DETR AIFI, CCFM, RepC3 and
  RTDETRDecoder remain unchanged.
- Four independent model YAML files: base, dual, full and stride control.
- `tests/test_mirnet_det.py`: geometry, dynamic sizes, exchange routes,
  gradients, full decoder routing and checkpoint round-trip coverage.
- Linux training, evaluation, benchmark and diagnosis tools under `tools/`.

## Data flow

`MIRDetBackbone` returns three tensors in one list:

```text
P3: [B, 128, H/8,  W/8]
P4: [B, 256, H/16, W/16]
P5: [B, 512, H/32, W/32]
```

The YAML immediately selects these outputs with `FeatureSelect(0/1/2)`. The
original RT-DETR head consumes those three distinct tensors and feeds decoder
indices `[15, 18, 21]` with hidden width 256 and 300 queries.

Dual and tri exchanges update all branches from the same old states in each
round. Neighbouring projections use 1x1 plus interpolation for up paths and
3x3 stride-2 convolutions for down paths. No global attention, P2 detector,
additional loss or query change was introduced.

## Verification status

Python AST compilation and YAML parsing pass on the local Windows checkout.
The local environment does not have PyTorch installed, so runtime unit tests,
forward/backward checks, GPU benchmarks and training have not been run here.
Run the real test suite on the Python 3.10/PyTorch 2.3.1 A30 environment:

```bash
python -m unittest discover -s tests -p 'test_mirnet_det.py' -v
```

Do not report accuracy, FLOPs or latency until those commands complete on the
target environment.
