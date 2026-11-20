# -*- coding: utf-8 -*-
"""
V2 Reconstructed Innovation A: Dual-Scale Coordinate-Preserving WCA-SFP v2 (WCASFPv2)
=====================================================================================
Empirical Basis from Real A30 Test Results (`DUT-Anti-UAV`, 2200 test images):
    - `E1 (WCA-SFP)` was the #1 overall winner on test set:
      mAP50: 0.960 -> 0.966 (+0.6%), mAP75: 0.796 -> 0.803 (+0.7%), mAP50:95: 0.702 -> 0.704 (+0.2%).
    - Key empirical takeaway: Orthogonal Haar DWT preserves exact sub-pixel grid alignment
      (ZERO `grid_sample` coordinate drift!), which is why `mAP75` rose significantly.

V2 Targeted Upgrades over E1:
    1. Dual-Scale Coverage (`P2->P3` at Layer 17 AND `P3->P4` at Layer 12):
       Extends the proven coordinate-preserving Haar DWT prior to both `P3/8` (80x80) and
       `P4/16` (40x40), helping both tiny (<16px) and small (16-32px) UAVs.
    2. Zero-Shift Bipolar Clutter Gate (inspired by E9's 0.983 Precision without grid_sample):
       Adds a strictly pixel-aligned `tanh` signed channel-spatial gate so high-frequency
       background clutter (tree tips / building corners) is suppressed with zero coordinate shift.
    3. Full FP32 AMP stability (`autocast(enabled=False)`) & EMA `deepcopy` safety.
"""

from __future__ import annotations
import copy
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class _HaarDWT2D(nn.Module):
    """Parameter-free orthogonal 2D Haar DWT (Parseval energy-preserving, stride=2)."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        u_ll = 0.5 * (x00 + x01 + x10 + x11)
        u_lh = 0.5 * (x00 + x01 - x10 - x11)
        u_hl = 0.5 * (x00 - x01 + x10 - x11)
        u_hh = 0.5 * (x00 - x01 - x10 + x11)
        return u_ll, u_lh, u_hl, u_hh


class _FP32CenterSurroundContrast(nn.Module):
    """FP32-protected local Center-Surround contrast (aligned with commit 2415a23)."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = float(eps)
        ring = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32) / 8.0
        self.register_buffer("ring_kernel", ring.view(1, 1, 3, 3), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        mu_cen = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        ring_w = self.ring_kernel.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        x_pad = F.pad(x, (2, 2, 2, 2), mode="replicate")
        mu_sur = F.conv2d(x_pad, ring_w, bias=None, stride=1, padding=0, dilation=2, groups=c)
        with torch.autocast(device_type=x.device.type, enabled=False):
            contrast = (mu_cen.float() - mu_sur.float()).abs() / (mu_sur.float().abs() + self.eps)
            d_cs = torch.tanh(contrast).to(dtype=x.dtype)
        return d_cs


class WCASFPv2(nn.Module):
    """
    Dual-Scale Coordinate-Preserving Wavelet-Contrast Prior v2 (Innovation V2-A).
    Works seamlessly at BOTH:
      - Layer 17 (`from=[4, 5]`: P2 64ch 160x160 + P3 128ch 80x80 -> 256ch 80x80)
      - Layer 12 (`from=[5, 6]`: P3 128ch 80x80 + P4 256ch 40x40 -> 256ch 40x40)
    Strictly zero `grid_sample` coordinate drift + zero-shift bipolar clutter suppression.
    """

    def __init__(
        self,
        c1: Union[int, List[int], Tuple[int, ...]],
        c2: int = 256,
        alpha_max: float = 0.30,
        init_gain: float = 0.10,
        diagonal_weight: float = 2.0,
        enabled: bool = True,
    ):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            self.c_hi, self.c_lo = int(c1[0]), int(c1[1])
            self.has_hi = True
        else:
            self.c_hi, self.c_lo = int(c1), int(c1)
            self.has_hi = False

        self.c_out = int(c2)
        self.c_mid = max(32, self.c_out // 2)
        self.alpha_max = float(alpha_max)
        self.diagonal_weight = float(diagonal_weight)
        self.enabled = bool(enabled)

        # Baseline-compatible 1x1 projection on target scale
        self.lo_proj = nn.Sequential(
            nn.Conv2d(self.c_lo, self.c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_out),
        )

        # High-resolution donor projection before Haar DWT
        self.hi_proj = nn.Sequential(
            nn.Conv2d(self.c_hi, self.c_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_mid),
            nn.SiLU(inplace=True),
        )

        self.dwt = _HaarDWT2D()
        self.cs_contrast = _FP32CenterSurroundContrast()

        self.hf_fuse = nn.Sequential(
            nn.Conv2d(self.c_mid * 3, self.c_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c_mid, self.c_mid, kernel_size=3, padding=1, groups=self.c_mid, bias=False),
            nn.BatchNorm2d(self.c_mid),
            nn.SiLU(inplace=True),
        )

        self.lo_sem = nn.Sequential(
            nn.Conv2d(self.c_out, self.c_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_mid),
            nn.SiLU(inplace=True),
        )

        # Spatial objectness gate (feeds ShallowEllipseAuxLoss during training)
        self.gate_head = nn.Sequential(
            nn.Conv2d(self.c_mid * 2 + 1, self.c_mid // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.c_mid // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c_mid // 2, 1, kernel_size=1, bias=True),
        )

        # Zero-Shift Bipolar Signed Clutter Gate (suppresses high-freq background clutter without warping!)
        self.signed_clutter_gate = nn.Sequential(
            nn.Conv2d(self.c_mid * 2, self.c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_out),
        )

        self.detail_refine = nn.Sequential(
            nn.Conv2d(self.c_mid * 2, self.c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c_out, self.c_out, kernel_size=3, padding=1, groups=self.c_out, bias=False),
            nn.BatchNorm2d(self.c_out),
        )

        ratio = min(max(init_gain / max(alpha_max, 1e-4), 1e-3), 1.0 - 1e-3)
        self.u_gamma = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32))
        self.last_obj_logits: Optional[torch.Tensor] = None
        self.last_obj_map: Optional[torch.Tensor] = None

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            f_hi, f_lo = x[0], x[1]
        else:
            f_lo, f_hi = x, None

        lo_base = self.lo_proj(f_lo)
        if not self.enabled:
            self.last_obj_logits = None
            self.last_obj_map = None
            return lo_base

        _, _, h, w = lo_base.shape
        if f_hi is None:
            hi_in = F.interpolate(f_lo, size=(h * 2, w * 2), mode="bilinear", align_corners=False)
        elif f_hi.shape[-2:] != (h * 2, w * 2):
            hi_in = F.interpolate(f_hi, size=(h * 2, w * 2), mode="bilinear", align_corners=False)
        else:
            hi_in = f_hi

        u = self.hi_proj(hi_in)
        u_ll, u_lh, u_hl, u_hh = self.dwt(u)

        with torch.autocast(device_type=lo_base.device.type, enabled=False):
            e_sq = (
                u_lh.float().pow(2) + u_hl.float().pow(2) + self.diagonal_weight * u_hh.float().pow(2)
            ).mean(dim=1, keepdim=True)
            e_wave = torch.sqrt(e_sq + 1e-6).to(dtype=lo_base.dtype)

        f_hf = self.hf_fuse(torch.cat([u_lh, u_hl, u_hh], dim=1))
        d_cs = self.cs_contrast(u_ll)
        f_hf_mod = f_hf * (1.0 + d_cs)

        s_lo = self.lo_sem(lo_base)
        z_obj = self.gate_head(torch.cat([s_lo, f_hf_mod, e_wave], dim=1))
        m_obj = torch.sigmoid(z_obj)

        self.last_obj_logits = z_obj if self.training else None
        self.last_obj_map = m_obj.detach() if self.training else None

        cat_detail = torch.cat([u_ll, f_hf_mod], dim=1)
        # Bipolar signed modulation in (-1, 1) allows suppressing hard-negative clutter spikes
        bipolar_sign = torch.tanh(self.signed_clutter_gate(cat_detail))
        detail_feat = self.detail_refine(cat_detail * m_obj) * bipolar_sign

        with torch.autocast(device_type=lo_base.device.type, enabled=False):
            gain = (self.alpha_max * torch.sigmoid(self.u_gamma.float())).to(dtype=lo_base.dtype)
        return lo_base + gain * detail_feat

    def __deepcopy__(self, memo):
        self.last_obj_logits = None
        self.last_obj_map = None
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
