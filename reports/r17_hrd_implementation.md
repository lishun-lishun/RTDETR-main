# HRD-R18 (R17) implementation

The three ablations are registered as independent YAMLs:

- tdetr-r17a-cbr.yaml: original single stream, P5 reduced to 256 channels.
- tdetr-r17b-hrd-512.yaml: original 64/128/256/512 widths plus HR stream.
- tdetr-r17c-hrd-cbr.yaml: HR stream plus 64/128/256/256 widths.

HRStreamInit uses PixelUnshuffle(2) followed by a 1x1 projection. HRUpdate3/4
use nearest resizing only, and HRInjectP3/P4 use zero-initialized terminal
projection BatchNorm together with zero gamma, preserving the original P3/P4
features at initialization. The RT-DETR neck and decoder are unchanged.

Run python tools/verify_r17_hrd.py --device 0 on the CUDA training machine,
then benchmark with python tools/benchmark_r17_hrd.py --device 0 --execute.
