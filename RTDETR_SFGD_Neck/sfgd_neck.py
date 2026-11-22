# -*- coding: utf-8 -*-
"""
SFGD-Neck: Semantic-Guided Frequency-Decoupled Gated Neck for FastStar-HRNet + RT-DETR
======================================================================================
Original Innovation 2 specifically coupled to `FastStar-HRNet` (`[P3(36), P4(72), P5(144)]`):

1. `FreqDecoupledBridge` (Backbone-to-Neck Frequency-Decoupled Semantic Denoising Bridge):
   - Addresses the 7.1x blind channel expansion (`36 -> 256`) and background high-frequency
     clutter (foliage/building edges) introduced by Haar folding at `P3 (80x80)`.
   - Decomposes `P3` via parameter-free 3x3 average low-pass (`F_LF`) and Laplacian residual
     (`F_HF = P3 - F_LF`), then uses coarse semantic features from `P4 (40x40)` to generate a
     lightweight spatial-channel gate `G_sem` that suppresses non-target background high-frequency
     clutter while preserving isolated tiny UAV sub-pixel edges.
   - Zero-Shift initialized (`gamma=0`) + 100% `switch_to_deploy()` Conv+BN fusion, zero `grid_sample`.

2. `SoftplusZeroConcatFusion` (Zero-Concat Softplus Semantic-Gated Cross-Scale Fusion):
   - Replaces static `Concat` (`256+256 -> 512ch`) in FPN/PAN with semantic channel screening
     + FP32-safe Softplus-normalized learnable weighting (`w1, w2 = Softplus(a1), Softplus(a2)`).
   - Cuts input channel width of all 4 downstream `RepC3` blocks in HALF (`512 -> 256`),
     reducing Neck parameters by ~1.2M while suppressing cross-scale background conflict.

3. `DualFreqAIFI` (Local-High-Frequency + Global-Context Dual-Branch AIFI):
   - Enhances compact `AIFI(512, 8)` on `P5 (20x20)` with a parallel grouped-3x3 local contrast
     residual branch (Zero-Shift BN init, fused into 1 plain conv at deploy time).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .hrnet_lite_suite import _fuse_conv_bn_pair, fuse_sequential_conv_bn
except ImportError:
    from hrnet_lite_suite import _fuse_conv_bn_pair, fuse_sequential_conv_bn

try:
    from .transformer import AIFI
except ImportError:
    from ultralytics.nn.modules.transformer import AIFI

__all__ = [
    "FreqDecoupledBridge",
    "SoftplusZeroConcatFusion",
    "DualFreqAIFI",
]


class FreqDecoupledBridge(nn.Module):
    """
    Backbone-to-Neck Semantic-Guided High-Frequency Denoising Bridge for `P3 (36ch -> c_out)`.
    Takes `[f_p3, f_p4]` from `FastStar-HRNet`:
      - `f_lf = AvgPool2d(3, 1, 1)(f_p3)` (smooth structural base)
      - `f_hf = f_p3 - f_lf`              (zero-param high-frequency detail residual)
      - `g_sem = sigmoid(Upsample(Conv1x1(f_p4)))` (coarse semantic spatial-channel prior)
      - `out = Proj_LF(f_lf) + sigmoid(gamma) * g_sem * Proj_HF(f_hf)`
    """

    def __init__(self, c_p3: int, c_p4: int, c_out: int):
        super().__init__()
        self.c_p3 = int(c_p3)
        self.c_p4 = int(c_p4)
        self.c_out = int(c_out)
        self.fused = False

        # Low-frequency structural projection (3x3 spatial context expansion 36 -> c_out)
        self.lf_conv = nn.Conv2d(self.c_p3, self.c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.lf_bn = nn.BatchNorm2d(self.c_out)

        # High-frequency residual projection (1x1 preserving sharp isolated UAV points)
        self.hf_conv = nn.Conv2d(self.c_p3, self.c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.hf_bn = nn.BatchNorm2d(self.c_out)
        # Zero-Shift init so Epoch-0 starts smoothly without injecting noisy untrained HF gradients
        nn.init.constant_(self.hf_bn.weight, 0.0)
        nn.init.constant_(self.hf_bn.bias, 0.0)

        # Semantic gate from P4 (40x40 -> 80x80): bottlenecked 1x1 conv for low params & high A30 speed
        gate_mid = max(16, self.c_out // 8)
        self.sem_gate: nn.Module = nn.Sequential(
            nn.Conv2d(self.c_p4, gate_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(gate_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_mid, self.c_out, kernel_size=1, bias=True),
        )
        self.low_pass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def switch_to_deploy(self) -> None:
        if self.fused:
            return
        if hasattr(self, "lf_bn") and isinstance(self.lf_bn, nn.BatchNorm2d):
            self.lf_conv = _fuse_conv_bn_pair(self.lf_conv, self.lf_bn)
            delattr(self, "lf_bn")
        if hasattr(self, "hf_bn") and isinstance(self.hf_bn, nn.BatchNorm2d):
            self.hf_conv = _fuse_conv_bn_pair(self.hf_conv, self.hf_bn)
            delattr(self, "hf_bn")
        self.sem_gate = fuse_sequential_conv_bn(self.sem_gate)
        for p in self.parameters():
            if p.is_leaf:
                p.requires_grad_(False)
        self.fused = True

    def fuse(self) -> "FreqDecoupledBridge":
        self.switch_to_deploy()
        return self

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        f_p3, f_p4 = inputs[0], inputs[1]
        f_lf = self.low_pass(f_p3)
        f_hf = f_p3 - f_lf

        gate_coarse = self.sem_gate(f_p4)
        gate_up = F.interpolate(gate_coarse, size=f_p3.shape[-2:], mode="nearest")
        calc_dtype = torch.float64 if f_p3.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=f_p3.device.type, enabled=False):
            g_sem = torch.sigmoid(gate_up.to(calc_dtype)).to(f_p3.dtype)
            scale = torch.sigmoid(self.gamma.to(calc_dtype)).to(f_p3.dtype)

        if self.fused:
            y_lf = self.lf_conv(f_lf)
            y_hf = self.hf_conv(f_hf)
        else:
            y_lf = self.lf_bn(self.lf_conv(f_lf))
            y_hf = self.hf_bn(self.hf_conv(f_hf))

        return y_lf + scale * (g_sem * y_hf)


class SoftplusZeroConcatFusion(nn.Module):
    """
    Zero-Concat Semantic-Screened Softplus-Gated Cross-Scale Fusion.
    Replaces `Concat` (`[x_sem, x_det]` -> `2C`) with a `C`-channel gated fusion:
      - Semantic channel screening: `g_c = sigmoid(Conv1x1(GAP(x_sem)))` filters noisy channels in `x_det`
      - Stable Softplus-normalized weights (`w1, w2 > 0`, FP32-safe under AMP):
        `out = (w1 * x_sem + w2 * (g_c * x_det)) / (w1 + w2 + eps)`
    Cuts downstream `RepC3` input channels from `512 -> 256` (or `384 -> 192`), saving ~1.2M Neck params!
    """

    def __init__(self, c_sem: int, c_det: int, c_out: int):
        super().__init__()
        self.c_out = int(c_out)
        self.fused = False
        self.proj_sem = (
            nn.Sequential(nn.Conv2d(c_sem, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
            if int(c_sem) != int(c_out)
            else nn.Identity()
        )
        self.proj_det = (
            nn.Sequential(nn.Conv2d(c_det, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
            if int(c_det) != int(c_out)
            else nn.Identity()
        )
        mid = max(16, self.c_out // 8)
        self.chan_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c_out, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, self.c_out, kernel_size=1, bias=True),
        )
        # Learnable Softplus fusion logits (initialized to equal 0.5 / 0.5 weight)
        self.w_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def switch_to_deploy(self) -> None:
        if self.fused:
            return
        if isinstance(self.proj_sem, nn.Sequential):
            self.proj_sem = fuse_sequential_conv_bn(self.proj_sem)
        if isinstance(self.proj_det, nn.Sequential):
            self.proj_det = fuse_sequential_conv_bn(self.proj_det)
        for p in self.parameters():
            if p.is_leaf:
                p.requires_grad_(False)
        self.fused = True

    def fuse(self) -> "SoftplusZeroConcatFusion":
        self.switch_to_deploy()
        return self

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x_sem = self.proj_sem(inputs[0])
        x_det = self.proj_det(inputs[1])
        if x_sem.shape[-2:] != x_det.shape[-2:]:
            x_sem = F.interpolate(x_sem, size=x_det.shape[-2:], mode="nearest")

        gate_raw = self.chan_gate(x_sem)
        calc_dtype = torch.float64 if x_sem.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x_sem.device.type, enabled=False):
            g_c = torch.sigmoid(gate_raw.to(calc_dtype)).to(x_sem.dtype)
            w = F.softplus(self.w_logits.to(calc_dtype))
            w_norm = w / (w.sum() + 1e-6)
            w0 = w_norm[0].to(x_sem.dtype)
            w1 = w_norm[1].to(x_sem.dtype)

        return w0 * x_sem + w1 * (g_c * x_det)


class DualFreqAIFI(nn.Module):
    """
    Dual-Frequency Enhanced Compact AIFI on `P5 (20x20)`:
    Combines standard global self-attention `AIFI(c1, cm, num_heads)` with a lightweight
    parallel local high-frequency contrast branch (`Conv3x3(groups=4) + Zero-Shift BN`),
    fully fused into a single grouped 3x3 conv at inference time.
    """

    def __init__(self, c1: int, cm: int = 512, num_heads: int = 8):
        super().__init__()
        self.aifi = AIFI(c1, cm=cm, num_heads=num_heads)
        g = 4 if (int(c1) % 4 == 0) else 1
        self.local_conv = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=g, bias=False)
        self.local_bn = nn.BatchNorm2d(c1)
        nn.init.constant_(self.local_bn.weight, 0.0)
        nn.init.constant_(self.local_bn.bias, 0.0)
        self.act = nn.SiLU(inplace=True)
        self.fused = False

    def switch_to_deploy(self) -> None:
        if self.fused:
            return
        if hasattr(self, "local_bn") and isinstance(self.local_bn, nn.BatchNorm2d):
            self.local_conv = _fuse_conv_bn_pair(self.local_conv, self.local_bn)
            delattr(self, "local_bn")
        for p in self.parameters():
            if p.is_leaf:
                p.requires_grad_(False)
        self.fused = True

    def fuse(self) -> "DualFreqAIFI":
        self.switch_to_deploy()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_global = self.aifi(x)
        if self.fused:
            y_local = self.act(self.local_conv(x))
        else:
            y_local = self.act(self.local_bn(self.local_conv(x)))
        return y_global + y_local
