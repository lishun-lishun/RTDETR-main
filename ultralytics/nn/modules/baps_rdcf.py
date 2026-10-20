"""BAPS and RDCF pluggable modules for the RT-DETR-R18 backbone/neck improvements.

BAPS (Blur-Adaptive Pixel-preserving Sampling) enhances the P3 feature from the
backbone with a high-frequency detail path (PixelUnshuffle) and an anti-aliased
low-pass path (BlurPool), fused by a lightweight learned gate. The residual
projection is zero-initialized so that at initialization the module is an exact
identity mapping (P3_enhanced == P3).

RDCF (Reliability-guided Detail Compensation Fusion) fuses detail cues from P2,
the current P3 feature and P4, weighted by a learned per-pixel reliability map,
again with a zero-initialized residual projection (F3_enhanced == F3 at init).

Both modules are:
    * pluggable: registered by name in parse_model via ``globals()[m]``;
    * multi-input: declared with a ``from`` list in the YAML (``f`` is a list);
    * AMP/FP16-safe: fixed kernels are cast to the input dtype on the fly;
    * free of heavy dependencies (torch only);
    * gradient-preserving: no ``.detach()`` is ever used on the residual path.

References:
    * Zhang, R. "Making Convolutional Networks Shift-Invariant Again" (BlurPool).
    * Shi et al. "Real-Time Single Image and Video Super-Resolution Using an
      Efficient Sub-Pixel Convolutional Neural Network" (PixelShuffle/Unshuffle).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import ConvNormLayer

__all__ = ('BAPS', 'RDCF', 'BlurPool')


class FixedDepthwiseConv(nn.Module):
    """Depthwise convolution with a fixed (non-learnable) kernel, e.g. a blur or
    gradient operator. The kernel is registered as a buffer and cast to the input
    dtype in ``forward`` so that it remains safe under AMP/FP16 autocast."""

    def __init__(self, channels, kernel, stride=1):
        super().__init__()
        self.stride = stride
        k = kernel.to(torch.float32).view(1, 1, *kernel.shape)
        self.register_buffer('kernel', k.repeat(int(channels), 1, 1, 1))

    def forward(self, x):
        c = x.shape[1]
        kernel = self.kernel[:c].to(x.dtype)
        return F.conv2d(x, kernel, stride=self.stride, padding=1, groups=c)


class BlurPool(FixedDepthwiseConv):
    """Anti-aliased low-pass downsampling (binomial 3x3 kernel, normalized to 1).

    With ``stride=2`` this halves the spatial resolution while suppressing aliasing;
    with ``stride=1`` it acts as a low-pass filter at the native resolution (used to
    build the high-frequency residual statistic in BAPS)."""

    def __init__(self, channels, stride=2):
        kernel = torch.tensor([1.0, 2.0, 1.0,
                               2.0, 4.0, 2.0,
                               1.0, 2.0, 1.0]).view(3, 3)
        kernel = kernel / kernel.sum()
        super().__init__(channels, kernel, stride=stride)


def _sobel_x():
    return torch.tensor([[-1.0, 0.0, 1.0],
                         [-2.0, 0.0, 2.0],
                         [-1.0, 0.0, 1.0]])


def _sobel_y():
    return torch.tensor([[-1.0, -2.0, -1.0],
                         [0.0, 0.0, 0.0],
                         [1.0, 2.0, 1.0]])


class BAPS(nn.Module):
    """Blur-Adaptive Pixel-preserving Sampling.

    Inputs (from a YAML ``from`` list ``[p2_idx, p3_idx]``):
        P2: [B, c_p2, H/4, W/4]
        P3: [B, c_p3, H/8, W/8]
    Output:
        P3_enhanced: [B, c_p3, H/8, W/8]

    Paths:
        1. Detail path: P2 -> 1x1 conv (c_p2->hidden) -> PixelUnshuffle(2) ->
           3x3 depthwise conv -> 1x1 conv (4*hidden->c_p2).
        2. Anti-alias path: P2 -> BlurPool(stride=2) -> 1x1 conv (c_p2->c_p2).

    The gate receives the two branch features together with the P2 high-frequency
    residual statistic, P2 horizontal+vertical gradient statistics and a compressed
    P3 context, and outputs per-pixel softmax weights over the two branches. The
    fused feature is projected back to ``c_p3`` through a zero-initialized 1x1 conv
    and added to P3.
    """

    def __init__(self, c_p2, c_p3, hidden_channels=32):
        super().__init__()
        c_p2 = int(c_p2)
        c_p3 = int(c_p3)
        h = int(hidden_channels)

        # --- Detail-preserving path (PixelUnshuffle) ---
        self.p2_reduce = ConvNormLayer(c_p2, h, 1, 1, act='relu')
        self.pixel_unshuffle = nn.PixelUnshuffle(2)  # h -> 4*h, H/4 -> H/8
        self.detail_dw = nn.Conv2d(h * 4, h * 4, 3, 1, 1, groups=h * 4, bias=False)
        self.detail_bn = nn.BatchNorm2d(h * 4)
        self.detail_pw = ConvNormLayer(h * 4, c_p2, 1, 1, act='relu')

        # --- Anti-aliasing path ---
        self.blur = BlurPool(c_p2, stride=2)
        self.anti_conv = ConvNormLayer(c_p2, c_p2, 1, 1, act='relu')

        # --- Gate cues ---
        self.blur1 = BlurPool(c_p2, stride=1)          # low-pass at P2 resolution
        self.grad_x = FixedDepthwiseConv(c_p2, _sobel_x())
        self.grad_y = FixedDepthwiseConv(c_p2, _sobel_y())
        self.pool2 = nn.AvgPool2d(2, 2)                 # H/4 -> H/8

        self.cue_detail = ConvNormLayer(c_p2, h, 1, 1, act='relu')
        self.cue_anti = ConvNormLayer(c_p2, h, 1, 1, act='relu')
        self.cue_hf = ConvNormLayer(c_p2, h, 1, 1, act='relu')
        self.cue_grad = ConvNormLayer(c_p2, h, 1, 1, act='relu')
        self.cue_p3 = ConvNormLayer(c_p3, h, 1, 1, act='relu')

        gate_in = h * 5
        self.gate = nn.Sequential(
            ConvNormLayer(gate_in, h, 3, 1, act='relu'),
            nn.Conv2d(h, 2, 1, bias=True),
        )

        # --- Zero-initialized residual projection ---
        self.delta = nn.Conv2d(c_p2, c_p3, 1, bias=True)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x):
        p2, p3 = x

        # Detail path
        d = self.p2_reduce(p2)                       # [B, h, H/4, W/4]
        d = self.pixel_unshuffle(d)                  # [B, 4h, H/8, W/8]
        d = F.relu(self.detail_bn(self.detail_dw(d)))
        f_detail = self.detail_pw(d)                 # [B, c_p2, H/8, W/8]

        # Anti-aliasing path
        f_anti = self.anti_conv(self.blur(p2))       # [B, c_p2, H/8, W/8]

        # High-frequency residual statistic of P2
        hf = self.pool2((p2 - self.blur1(p2)).abs())  # [B, c_p2, H/8, W/8]

        # Horizontal + vertical gradient statistics of P2
        gx = self.grad_x(p2).abs()
        gy = self.grad_y(p2).abs()
        grad = self.pool2(gx + gy)                    # [B, c_p2, H/8, W/8]

        # Compressed P3 context
        p3c = self.cue_p3(p3)                         # [B, h, H/8, W/8]

        gate_in = torch.cat([
            self.cue_detail(f_detail),
            self.cue_anti(f_anti),
            self.cue_hf(hf),
            self.cue_grad(grad),
            p3c,
        ], dim=1)                                     # [B, 5h, H/8, W/8]

        logits = self.gate(gate_in)                   # [B, 2, H/8, W/8]
        w = torch.softmax(logits, dim=1)              # [B, 2, H/8, W/8]

        f_fused = w[:, 0:1] * f_detail + w[:, 1:2] * f_anti
        delta = self.delta(f_fused)                   # [B, c_p3, H/8, W/8]
        return p3 + delta


class RDCF(nn.Module):
    """Reliability-guided Detail Compensation Fusion.

    Inputs (from a YAML ``from`` list ``[f3_idx, p2_idx, f4_idx]``):
        F3: [B, c_p3, H/8, W/8]
        P2: [B, c_p2, H/4, W/4]
        F4: [B, c_p4, H/16, W/16]
    Output:
        F3_enhanced: [B, c_p3, H/8, W/8]

    Three detail projections are aligned to the F3 resolution (using the actual
    spatial sizes, never a hard-coded image size):
        D2 = 1x1 conv(BlurPool(P2))
        D3 = 1x1 conv(F3)
        D4 = upsample(1x1 conv(F4))
    A reliability map R in (0.1, 1.0) is predicted from D2, D3, D4 and their
    pairwise absolute differences; the detail D2 is weighted by R and projected
    back to c_p3 through a zero-initialized 1x1 conv before being added to F3.
    """

    def __init__(self, c_p3, c_p2, c_p4, hidden_channels=64):
        super().__init__()
        c_p3 = int(c_p3)
        c_p2 = int(c_p2)
        c_p4 = int(c_p4)
        h = int(hidden_channels)

        self.blur2 = BlurPool(c_p2, stride=2)
        self.d2_conv = ConvNormLayer(c_p2, h, 1, 1, act='relu')
        self.d3_conv = ConvNormLayer(c_p3, h, 1, 1, act='relu')
        self.d4_conv = ConvNormLayer(c_p4, h, 1, 1, act='relu')

        rel_in = h * 5
        self.rel = nn.Sequential(
            ConvNormLayer(rel_in, h, 3, 1, act='relu'),
            nn.Conv2d(h, 1, 1, bias=True),
        )

        # Zero-initialized residual projection
        self.delta = nn.Conv2d(h, c_p3, 1, bias=True)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x):
        f3, p2, f4 = x

        d2 = self.d2_conv(self.blur2(p2))             # [B, h, H/8, W/8]
        d3 = self.d3_conv(f3)                          # [B, h, H/8, W/8]
        d4 = self.d4_conv(f4)                          # [B, h, H/16, W/16]
        d4 = F.interpolate(d4, size=d3.shape[2:], mode='bilinear', align_corners=False)

        rel_in = torch.cat([d2, d3, d4, (d2 - d3).abs(), (d2 - d4).abs()], dim=1)
        logits = self.rel(rel_in)                      # [B, 1, H/8, W/8]
        r = 0.1 + 0.9 * torch.sigmoid(logits)          # [B, 1, H/8, W/8]

        delta = self.delta(r * d2)                     # [B, c_p3, H/8, W/8]
        return f3 + delta
