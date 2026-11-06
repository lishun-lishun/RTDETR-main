# RT-DETR-R18 independent backbone variants

This experiment family starts from `ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml`.
The original YAML remains unchanged. Every R01-R08 YAML contains the same eight
backbone slots and the original RT-DETR head, with decoder routes `[19, 22, 25]`.

| ID | Model | Single backbone change |
|---|---|---|
| E0 | baseline | Original R18 |
| E1 | r18_stemconv | Conv3x3/BN/ReLU stride-2 stem downsampling |
| E2 | r18_aastem | stride-1 MaxPool plus fixed 3x3 BlurPool |
| E3 | r18_resnetd | Explicit ResNet-D downsampling shortcuts |
| E4 | r18_p3deep | One extra Stage3 BasicBlock |
| E5 | r18_p3wide | Stage3 output width 160 |
| E6 | r18_p4dilated | Stage4 second block branch2b dilation 2 |
| E7 | r18_p4dcn | Stage4 second block branch2b torchvision DeformConv2d |
| E8 | r18_p5pconv | Stage5 second block branch2a PartialConv3, ratio 0.25 |

The custom modules are isolated in `ultralytics/nn/modules/r18_backbone_variants.py`.
The shared parser only adds these classes to the model registry; the original
`BasicBlock` and `Blocks` behavior is preserved for historical models. R07
requires a working torchvision `DeformConv2d` operator and fails explicitly if
the matching C++/CUDA implementation is unavailable.

Run the structural tests before queuing training:

```bash
python -m unittest discover -s tests -p test_r18_eight_variants.py -v
```

Training, evaluation, benchmarking and diagnosis are provided by
`tools/train_r18_variants.py`, `tools/launch_r18_eight.py`,
`tools/evaluate_r18_variants.py`, `tools/benchmark_r18_variants.py` and
`tools/diagnose_r18_variants.py`. The launcher is dry-run by default.
The summary reserves a reference-only row for the existing `rtdetr_hrnet_w18.yaml`;
its metrics must be regenerated with the same evaluator before making a numerical
claim against the R18 family.
