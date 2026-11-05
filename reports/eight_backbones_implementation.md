# RT-DETR eight-backbone platform

## Candidate registry

| ID | Registry name | Source | Feature source |
|---|---|---|---|
| B1 | `hgnetv2_b2` | timm | runtime `feature_info` |
| B2 | `convnextv2_pico` | timm | runtime `feature_info` |
| B3 | `fasternet_t1` | timm | runtime `feature_info` |
| B4 | `repvit_m1_1` | timm | runtime `feature_info` |
| B5 | `efficientvit_b1` | timm | runtime `feature_info` |
| B6 | `pvt_v2_b1` | timm | runtime `feature_info` |
| B7 | `lsknet_t` | official LSKNet implementation already present in this checkout | reductions 8/16/32 |
| B8 | `pkinetv2_t` | official PKINet-v2 implementation ported to pure PyTorch | reductions 8/16/32 |

DLA34 remains a separate experiment and is not included in this sweep.

Reference links: [timm](https://github.com/huggingface/pytorch-image-models),
[LSKNet](https://github.com/zcablii/LSKNet), and
[PKINet](https://github.com/PKINet/PKINet).  B1--B6 use the installed timm
registry and therefore inherit the exact implementation/version available on
the training server.  B7/B8 use the local copies derived from those official
repositories; no MMRotate training framework is imported.
The PKINet repository is Apache-2.0 licensed; the port keeps the source
attribution and implementation link in this report.

The timm adapter calls `timm.list_models()` and first tries the explicit base
name, then a checkpoint suffix with the same stem.  It records the resolved
name and source channels.  A missing model raises an error listing similar
installed names instead of silently selecting a different network.

## Adapter and graph

`ultralytics/nn/modules/backbone_adapter.py` selects actual reductions 8, 16
and 32 and applies independent 1x1 Conv+BatchNorm projections to
`(128,256,512)`. It checks every forward shape against the input size and
does not fuse or enhance the three features.  The eight YAMLs use one adapter
layer followed by explicit `FeatureSelect(0/1/2)` nodes.  The head is copied
from the original RT-DETR R18 graph, with AIFI, top-down and bottom-up CCFM,
RepC3, decoder hidden dimension 256 and 300 queries unchanged.

`ultralytics/nn/modules/pkinetv2_backbone.py` ports the official PKINet-v2-T
configuration and its five-branch PKS module directly to PyTorch.  It avoids
the MMRotate registry and does not require MMCV.  The older PKINet module in
`ultralytics/nn/backbone/pkinet.py` remains available for existing configs.

## Reproducibility

`tools/train_backbone.py` reads `ultralytics/cfg/csp_context_default.yaml` and
records the effective batch, micro-batch, accumulation and warmup unit. The
default protocol is random initialization, seed 0, 150 epochs, effective
batch 8, 640 input, AdamW, `lr0=1e-4`, `lrf=1.0`, weight decay `1e-4`, AMP
off, and warmup 2000 micro-batch iterations.  The historical R18 args file
can be checked with `--baseline-args`.

`tools/launch_backbone_sweep.py` defaults to dry-run.  With `--execute` it
runs the full eight-model unit suite once, binds one child process to each
physical GPU through `CUDA_VISIBLE_DEVICES`, queues excess jobs, writes a
manifest containing PID/timestamps/return codes/Git state/config hashes, and
isolates every model's output directory.  `--stop-model` sends SIGTERM to the
recorded child PID; `--resume-models` resumes only named jobs with an existing
`last.pt`.

## Validation status

The Windows checkout has no PyTorch runtime, so GPU forward, AMP, timing and
training results must be produced on the Linux A30 server.  Static Python,
AST, YAML and whitespace checks are run locally.  No metric or speed result
is claimed before the server test suite and evaluation scripts produce it.

## Commands

One model can be trained with:

```bash
python tools/train_backbone.py --model hgnetv2_b2 --device 0 --seed 0
```

Replace `--model` with any registry key.  The complete eight-model sweep is
planned with:

```bash
python tools/launch_backbone_sweep.py --models all --gpus 0,1,2,3 \
  --seed 0 --max-parallel 4
```

Review the dry-run manifest, then add `--execute`.  Development evaluation
uses `tools/evaluate_backbone.py --split val`; the held-out final run uses
`--split test`.  `tools/benchmark_backbone.py` defaults to 640 input, batch 1,
50 warmups and 200 timed iterations.
