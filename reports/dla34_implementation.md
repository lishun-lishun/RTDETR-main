# DLA34-RTDETR implementation report

## Reference and scope

The backbone follows the UCB Drive DLA reference implementation at
`https://github.com/ucbdrive/dla/blob/master/dla.py`.  The retained DLA-34
definition is `levels=[1,1,1,2,2,1]`, `channels=[16,32,64,128,256,512]`,
the stride-1 7x7 stem, `DLABasicBlock`, recursive `DLATree`, `DLARoot`,
`level_root`, MaxPool downsampling and residual projection.  Classification
average pooling and the FC layer are not part of the detection backbone.

The reference project is BSD 3-Clause licensed, Copyright (c) 2018 Fisher Yu.
Redistribution of source or binary forms must retain that copyright notice,
the three BSD conditions, and the disclaimer; the copyright holder's name may
not be used for endorsement.  The complete license is available in the
reference repository's `LICENSE` file.

## Files changed

- `ultralytics/nn/modules/dla34_backbone.py`: DLA-34 implementation and an
  explicit local-checkpoint loader.
- `ultralytics/nn/modules/__init__.py`: exports DLA modules.
- `ultralytics/nn/tasks.py`: parser support for a tuple-valued DLA output.
- `ultralytics/cfg/models/rt-detr/rtdetr_dla34.yaml`: complete RT-DETR graph.
- `tests/test_dla34_rtdetr.py`: structure, shape, routing, gradient and
  checkpoint tests.
- `tools/{train,evaluate,benchmark,diagnose}_rtdetr_dla34.py` and
  `tools/dla34_common.py`: reproducible training and measurement tools.

## RT-DETR connection

The DLA layer returns `(level3, level4, level5)` as `(P3, P4, P5)` with
channels `(128, 256, 512)` and strides `(8, 16, 32)`.  Three explicit
`FeatureSelect` nodes preserve graph edges into the original CCFM.  The head
keeps AIFI, both CCFM directions, RepC3, `RTDETRDecoder`, 300 queries and
hidden dimension 256.  At a 128x128 preflight input, decoder features are
`(1,256,16,16)`, `(1,256,8,8)`, `(1,256,4,4)` and the decoder `from` indices
are `[15,18,21]`.

## Pretrained policy

Scratch training is the default (`pretrained: false`).  A local official DLA
checkpoint can be supplied with `--pretrained-weights`; the loader strips an
optional `module.` prefix, ignores classification `fc`/pool keys, and records
provided, loaded, missing and unexpected keys.  No checkpoint is downloaded
automatically.

## Verification status

Static Python compilation, YAML parsing and `git diff --check` pass in the
Windows checkout.  Runtime PyTorch tests must be run on the Linux A30
environment because this checkout's Windows Python does not have PyTorch
installed.  The training script runs the DLA test suite and model preflight
before starting a new experiment unless `--skip-tests` is explicitly given.
