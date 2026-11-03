# LDW-ResNet18 implementation

## What changed

- Added `LateDownStage`, `WidthMatchedEarlyDownStage`, `ResidualBlock`, and
  `DownsampleProjection` in `ultralytics/nn/modules/ldw.py`.
- Registered both stage classes in `ultralytics/nn/modules/__init__.py` and
  added their `c1/c2/hidden_channels` handling to `parse_model()`.
- Added independent Stage4, Stage5, full, width matched control, and dilation
  YAML files. The original `rtdetr-r18.yaml` and `BasicBlock` are unchanged.
- The head remains the original RT-DETR head, including decoder input indices
  `[19, 22, 25]`, 300 queries, AIFI, CCFM and RepC3 blocks.
- Added `tests/test_ldw_r18.py` and four Linux ready tools under `tools/`.

## Shape contract

For a 640 input, the backbone contract is P3 `[B,128,80,80]`, P4
`[B,256,40,40]`, and P5 `[B,512,20,20]`. The decoder receives three 256
channel feature maps at those same strides. The tests use a 128 input to keep
CPU preflight inexpensive and check the equivalent `[16,8,4]` spatial sizes.

## Downsampling order

`LateDownStage` executes the configured stride-1 residual blocks and then one
3x3 stride-2 convolution. `WidthMatchedEarlyDownStage` executes one 3x3
stride-2 convolution, the same number of stride-1 residual blocks, and a 1x1
projection when the final width differs. A matching residual shortcut is an
identity and does not allocate a redundant 1x1 projection.

## Validation status

Static Python AST compilation and YAML parsing pass. The local Windows runtime
does not have PyTorch installed, so `python -m unittest tests/test_ldw_r18.py -v`
cannot import `torch`. No GPU, latency, FLOPs, training, or accuracy result is
reported from this machine.

Run the real checks on the A30 environment after installing the repository's
Python 3.10/PyTorch 2.3.1 requirements:

```bash
python -m unittest discover -s tests -p 'test_ldw_r18.py' -v
python tools/train_r18_ldw.py --model stage4 --device 0 --dry-run
python tools/train_r18_ldw.py --model stage5 --device 0 --dry-run
python tools/train_r18_ldw.py --model full --device 0 --dry-run
```

Training is intentionally not started by these changes.
