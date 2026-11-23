# -*- coding: utf-8 -*-
"""
RT-DETR Multi-Paradigm Neck Innovation Arena (Exp 20 - Exp 25)
==============================================================
Five orthogonal, hardware-aware, zero-`grid_sample` Neck paradigms built specifically for
`FastStar-HRNet` + RT-DETR on DUT-Anti-UAV small UAV detection (all 100% `switch_to_deploy()`
Conv+BN fusible, 0 residual gradients, AMP FP16 safe):

  1. `GatherDistributeGate` (Exp 20):
     Single-hop tri-scale Gather-and-Distribute consensus hub (`[P3, P4, P5] -> P4 -> [P3, P4, P5]`)
     eliminating multi-hop FPN/PAN attenuation on tiny targets.
  2. `OrthogonalFreqFilter` (Exp 21):
     Exact lossless 2D Haar DWT -> Learnable 4-subband (`LL, LH, HL, HH`) frequency-channel gating
     -> Exact Inverse Haar DWT (IDWT) restoring full spatial resolution with zero aliasing!
  3. `HiLoLinearAIFI` (Exp 22):
     High/Low-frequency decoupled AIFI: High-freq branch uses parallel 3x3 + 5x5 local contrast
     grouped convs; Low-freq branch pools 20x20 -> 10x10 (16x less attention FLOPs!) for global context.
  4. `StarC3` (Exp 23):
     Unified StarNet Neck block replacing heavy `RepC3` bottlenecks with hardware-aware `StarFastBlock`s.
  5. `MultiScaleContrastModule` (Exp 24):
     Physics-inspired Center-Surround Local Contrast (`X - AvgPool3(X)` & `X - AvgPool7(X)`) highlighting
     isolated dim 6x6-16x16 UAVs against cluttered clouds, foliage, and buildings with near-zero latency.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .hrnet_lite_suite import StarFastBlock, _fuse_conv_bn_pair, fuse_sequential_conv_bn
except ImportError:
    from hrnet_lite_suite import StarFastBlock, _fuse_conv_bn_pair, fuse_sequential_conv_bn

__all__ = [
    "GatherDistributeGate",
    "OrthogonalFreqFilter",
    "HiLoLinearAIFI",
    "StarC3",
    "MultiScaleContrastModule",
]


class GatherDistributeGate(nn.Module):
    """
    Exp 20: Lightweight Tri-Scale Gather-and-Distribute Injection Module.
    Takes `[x_target, x_p3, x_p4, x_p5]` (or `[x_target, x_ctx1, x_ctx2]`):
      - Aligns all context streams to `x_target`'s resolution via fast avg_pool / nearest upsample
      - Computes a shared global consensus feature via 1x1 fusion + `StarFastBlock`
      - Injects gated multi-scale consensus back into `x_target` (Zero-Shift BN init).
    """

    def __init__(self, c_list: list[int], c_out: int):
        super().__init__()
        self.c_out = int(c_out)
        self.fused = False
        c_total = sum(int(c) for c in c_list)
        self.gather_proj = nn.Conv2d(c_total, self.c_out, kernel_size=1, bias=False)
        self.gather_bn = nn.BatchNorm2d(self.c_out)
        self.star = StarFastBlock(self.c_out, groups=4, expand=1.5)
        self.dist_conv = nn.Conv2d(self.c_out, self.c_out, kernel_size=1, bias=False)
        self.dist_bn = nn.BatchNorm2d(self.c_out)
        nn.init.constant_(self.dist_bn.weight, 0.0)
        nn.init.constant_(self.dist_bn.bias, 0.0)
        self.act = nn.SiLU(inplace=True)

    def switch_to_deploy(self) -> None:
        if getattr(self, "fused", False):
            return
        if hasattr(self, "gather_bn") and isinstance(self.gather_bn, nn.BatchNorm2d):
            self.gather_proj = _fuse_conv_bn_pair(self.gather_proj, self.gather_bn)
            delattr(self, "gather_bn")
        self.star.switch_to_deploy()
        if hasattr(self, "dist_bn") and isinstance(self.dist_bn, nn.BatchNorm2d):
            self.dist_conv = _fuse_conv_bn_pair(self.dist_conv, self.dist_bn)
            delattr(self, "dist_bn")
        for p in self.parameters():
            if p.is_leaf:
                p.requires_grad_(False)
        self.fused = True

    def fuse(self) -> "GatherDistributeGate":
        self.switch_to_deploy()
        return self

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        target = inputs[0]
        h, w = target.shape[-2:]
        aligned = [target]
        for feat in inputs[1:]:
            if feat.shape[-2:] != (h, w):
                if feat.shape[-2] > h:
                    feat = F.adaptive_avg_pool2d(feat, (h, w))
                else:
                    feat = F.interpolate(feat, size=(h, w), mode="nearest")
            aligned.append(feat)
        cat = torch.cat(aligned, dim=1)
        if getattr(self, "fused", False):
            g = self.act(self.gather_proj(cat))
            g = self.star(g)
            return target + self.dist_conv(g)
        g = self.act(self.gather_bn(self.gather_proj(cat)))
        g = self.star(g)
        return target + self.dist_bn(self.dist_conv(g))


class OrthogonalFreqFilter(nn.Module):
    """
    Exp 21 & Exp 25: Lossless 2D Haar DWT -> Learnable 4-Band Frequency Gating -> Exact Inverse Haar IDWT.
    Explicitly separates low-frequency sky/cloud background (`LL`) from horizontal/vertical/diagonal
    high-frequency tiny UAV signatures (`LH, HL, HH`), applies lightweight channel-band gating,
    and reconstructs full spatial resolution via exact orthogonal IDWT (zero downsampling loss!).
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.fused = False
        self.in_proj: nn.Module = (
            nn.Sequential(nn.Conv2d(self.c1, self.c2, 1, bias=False), nn.BatchNorm2d(self.c2))
            if self.c1 != self.c2
            else nn.Identity()
        )
        # Lightweight 4-band frequency gate (operates on pooled 4*C frequency descriptor)
        mid = max(16, self.c2 // 4)
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c2 * 4, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, self.c2 * 4, kernel_size=1, bias=True),
        )
        # Learnable spatial refinement on reconstructed high-freq residual (Zero-Shift init)
        self.out_conv = nn.Conv2d(self.c2, self.c2, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(self.c2)
        nn.init.constant_(self.out_bn.weight, 0.0)
        nn.init.constant_(self.out_bn.bias, 0.0)

    def switch_to_deploy(self) -> None:
        if getattr(self, "fused", False):
            return
        if isinstance(self.in_proj, nn.Sequential):
            self.in_proj = fuse_sequential_conv_bn(self.in_proj)
        if hasattr(self, "out_bn") and isinstance(self.out_bn, nn.BatchNorm2d):
            self.out_conv = _fuse_conv_bn_pair(self.out_conv, self.out_bn)
            delattr(self, "out_bn")
        for p in self.parameters():
            if p.is_leaf:
                p.requires_grad_(False)
        self.fused = True

    def fuse(self) -> "OrthogonalFreqFilter":
        self.switch_to_deploy()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        h, w = x.shape[-2:]
        pad_h, pad_w = h % 2, w % 2
        xp = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else x

        # Exact 2D Haar DWT
        x00 = xp[:, :, 0::2, 0::2]
        x01 = xp[:, :, 0::2, 1::2]
        x10 = xp[:, :, 1::2, 0::2]
        x11 = xp[:, :, 1::2, 1::2]
        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (x00 + x01 - x10 - x11)
        hl = 0.5 * (x00 - x01 + x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)

        bands = torch.cat([ll, lh, hl, hh], dim=1)  # [B, 4C, H/2, W/2]
        gate_raw = self.freq_gate(bands)
        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            g = (2.0 * torch.sigmoid(gate_raw.to(calc_dtype))).to(x.dtype)  # centered at 1.0
        ll_g, lh_g, hl_g, hh_g = (bands * g).chunk(4, dim=1)

        # Exact Inverse 2D Haar DWT (IDWT) restoring full HxW resolution (pure out-of-place, autograd-safe)
        y00 = 0.5 * (ll_g + lh_g + hl_g + hh_g)
        y01 = 0.5 * (ll_g + lh_g - hl_g - hh_g)
        y10 = 0.5 * (ll_g - lh_g + hl_g - hh_g)
        y11 = 0.5 * (ll_g - lh_g - hl_g + hh_g)
        b_sz, c_sz, h2, w2 = y00.shape
        rec = torch.stack([y00, y01, y10, y11], dim=2).reshape(b_sz, c_sz, 2, 2, h2, w2).permute(0, 1, 4, 2, 5, 3).reshape(b_sz, c_sz, h2 * 2, w2 * 2)
        if pad_h or pad_w:
            rec = rec[:, :, :h, :w]

        if getattr(self, "fused", False):
            return x + self.out_conv(rec)
        return x + self.out_bn(self.out_conv(rec))


class HiLoLinearAIFI(nn.Module):
    """
    Exp 22 & Exp 25: High/Low-Frequency Decoupled Fast AIFI on `P5 (20x20)`.
    - High-Frequency Half (`C/2`): Parallel grouped 3x3 + 5x5 local contrast convolutions (zero quadratic cost!)
    - Low-Frequency Half (`C/2`): 2x2 AvgPool (`20x20 -> 10x10`, 16x less MHSA FLOPs!) + 4-head Global Attention
    - Compact FFN (`cm=512`) with residual LayerNorm.
    """

    def __init__(self, c1: int, cm: int = 512, num_heads: int = 4):
        super().__init__()
        self.c1 = int(c1)
        self.c_hi = self.c1 // 2
        self.c_lo = self.c1 - self.c_hi
        self.fused = False

        g_hi = 4 if (self.c_hi % 4 == 0) else 1
        self.hi_conv3 = nn.Conv2d(self.c_hi, self.c_hi, kernel_size=3, stride=1, padding=1, groups=g_hi, bias=False)
        self.hi_bn3 = nn.BatchNorm2d(self.c_hi)
        self.hi_conv5 = nn.Conv2d(self.c_hi, self.c_hi, kernel_size=5, stride=1, padding=2, groups=g_hi, bias=False)
        self.hi_bn5 = nn.BatchNorm2d(self.c_hi)

        # Low-frequency pooled global attention (100 tokens instead of 400 -> 16x faster!)
        nh = num_heads if (self.c_lo % num_heads == 0) else 2
        self.lo_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lo_attn = nn.MultiheadAttention(self.c_lo, nh, dropout=0.0, batch_first=True)
        self.lo_norm = nn.LayerNorm(self.c_lo)

        self.merge_conv = nn.Conv2d(self.c1, self.c1, kernel_size=1, bias=False)
        self.merge_bn = nn.BatchNorm2d(self.c1)
        self.act = nn.GELU()

        # Compact Conv-FFN (1x1 -> GELU -> 1x1, faster on A30 than permute+Linear)
        self.ffn1 = nn.Conv2d(self.c1, cm, kernel_size=1, bias=False)
        self.ffn_bn1 = nn.BatchNorm2d(cm)
        self.ffn2 = nn.Conv2d(cm, self.c1, kernel_size=1, bias=False)
        self.ffn_bn2 = nn.BatchNorm2d(self.c1)
        nn.init.constant_(self.ffn_bn2.weight, 0.0)
        nn.init.constant_(self.ffn_bn2.bias, 0.0)

    def switch_to_deploy(self) -> None:
        if getattr(self, "fused", False):
            return
        for cname, bname in (
            ("hi_conv3", "hi_bn3"),
            ("hi_conv5", "hi_bn5"),
            ("merge_conv", "merge_bn"),
            ("ffn1", "ffn_bn1"),
            ("ffn2", "ffn_bn2"),
        ):
            if hasattr(self, bname) and isinstance(getattr(self, bname), nn.BatchNorm2d):
                setattr(self, cname, _fuse_conv_bn_pair(getattr(self, cname), getattr(self, bname)))
                delattr(self, bname)
        for p in self.parameters():
            if p.is_leaf:
                p.requires_grad_(False)
        self.fused = True

    def fuse(self) -> "HiLoLinearAIFI":
        self.switch_to_deploy()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_hi, x_lo = torch.split(x, [self.c_hi, self.c_lo], dim=1)

        # High-frequency local branch
        if getattr(self, "fused", False):
            y_hi = self.act(self.hi_conv3(x_hi) + self.hi_conv5(x_hi))
        else:
            y_hi = self.act(self.hi_bn3(self.hi_conv3(x_hi)) + self.hi_bn5(self.hi_conv5(x_hi)))

        # Low-frequency pooled global branch
        lo_p = self.lo_pool(x_lo) if (h >= 2 and w >= 2) else x_lo
        hp, wp = lo_p.shape[-2:]
        tokens = lo_p.flatten(2).transpose(1, 2)  # [B, N, C_lo]
        attn_out, _ = self.lo_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.lo_norm(tokens + attn_out)
        y_lo = tokens.transpose(1, 2).reshape(b, self.c_lo, hp, wp)
        if (hp, wp) != (h, w):
            y_lo = F.interpolate(y_lo, size=(h, w), mode="nearest")

        merged = torch.cat([y_hi, y_lo], dim=1)
        if getattr(self, "fused", False):
            h_feat = x + self.merge_conv(merged)
            return h_feat + self.ffn2(self.act(self.ffn1(h_feat)))
        h_feat = x + self.merge_bn(self.merge_conv(merged))
        return h_feat + self.ffn_bn2(self.ffn2(self.act(self.ffn_bn1(self.ffn1(h_feat)))))


class StarC3(nn.Module):
    """
    Exp 23 & Exp 25: Hardware-Aware StarNet Neck Block replacing heavy `RepC3`.
    Uses zero-Identity deploy-fusible `StarFastBlock`s inside a CSP residual structure.
    """

    def __init__(self, c1: int, c2: int, n: int = 2, e: float = 0.5):
        super().__init__()
        c_ = max(32, int(c2 * e))
        self.fused = False
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_)
        self.m = nn.Sequential(*[StarFastBlock(c_, groups=4, expand=1.5) for _ in range(max(1, int(n)))])
        self.cv3 = nn.Conv2d(c_, c2, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def switch_to_deploy(self) -> None:
        if getattr(self, "fused", False):
            return
        for cname, bname in (("cv1", "bn1"), ("cv2", "bn2"), ("cv3", "bn3")):
            if hasattr(self, bname) and isinstance(getattr(self, bname), nn.BatchNorm2d):
                setattr(self, cname, _fuse_conv_bn_pair(getattr(self, cname), getattr(self, bname)))
                delattr(self, bname)
        for blk in self.m:
            if hasattr(blk, "switch_to_deploy"):
                blk.switch_to_deploy()
        for p in self.parameters():
            if p.is_leaf:
                p.requires_grad_(False)
        self.fused = True

    def fuse(self) -> "StarC3":
        self.switch_to_deploy()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "fused", False):
            return self.act(self.cv3(self.m(self.act(self.cv1(x))) + self.act(self.cv2(x))))
        return self.act(self.bn3(self.cv3(self.m(self.act(self.bn1(self.cv1(x)))) + self.act(self.bn2(self.cv2(x))))))


class MultiScaleContrastModule(nn.Module):
    """
    Exp 24 & Exp 25: Physics-Inspired Center-Surround Local Contrast Enhancement Module.
    Computes zero-parameter multi-scale center-surround contrast maps:
      - `d3 = x - AvgPool(3x3)(x)` (captures 4x4 - 8x8 very-tiny UAV points)
      - `d7 = x - AvgPool(7x7)(x)` (captures 10x10 - 24x24 small UAV silhouettes against local sky/clutter)
    Fuses `[d3, d7]` with a lightweight gated 1x1 projection (Zero-Shift BN init, 100% fused at deploy).
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.fused = False
        self.base_conv = nn.Conv2d(self.c1, self.c2, kernel_size=1, bias=False)
        self.base_bn = nn.BatchNorm2d(self.c2)

        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool7 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.contrast_conv = nn.Conv2d(self.c1 * 2, self.c2, kernel_size=1, bias=False)
        self.contrast_bn = nn.BatchNorm2d(self.c2)
        nn.init.constant_(self.contrast_bn.weight, 0.0)
        nn.init.constant_(self.contrast_bn.bias, 0.0)
        self.act = nn.SiLU(inplace=True)

    def switch_to_deploy(self) -> None:
        if getattr(self, "fused", False):
            return
        if hasattr(self, "base_bn") and isinstance(self.base_bn, nn.BatchNorm2d):
            self.base_conv = _fuse_conv_bn_pair(self.base_conv, self.base_bn)
            delattr(self, "base_bn")
        if hasattr(self, "contrast_bn") and isinstance(self.contrast_bn, nn.BatchNorm2d):
            self.contrast_conv = _fuse_conv_bn_pair(self.contrast_conv, self.contrast_bn)
            delattr(self, "contrast_bn")
        for p in self.parameters():
            if p.is_leaf:
                p.requires_grad_(False)
        self.fused = True

    def fuse(self) -> "MultiScaleContrastModule":
        self.switch_to_deploy()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d3 = x - self.pool3(x)
        d7 = x - self.pool7(x)
        contrast = torch.cat([d3, d7], dim=1)
        if getattr(self, "fused", False):
            return self.base_conv(x) + self.act(self.contrast_conv(contrast))
        return self.base_bn(self.base_conv(x)) + self.act(self.contrast_bn(self.contrast_conv(contrast)))
