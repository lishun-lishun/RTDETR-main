# -*- coding: utf-8 -*-
"""
Innovation 5: Directional Motion-Blur Anisotropic Deblurring Block (DMBA / DMBARepC3)
=====================================================================================
Target Problem:
    Fast-moving small UAVs in visible-light counter-UAV scenarios (`DUT-Anti-UAV`)
    frequently suffer from directional linear motion blur during exposure. A 6x6 px
    drone body is smeared into an elongated, low-contrast anisotropic streak along its
    velocity vector. Standard isotropic 3x3 convolutions in `RepC3` fail to re-concentrate
    this smeared energy toward the target centroid, causing bounding box over-expansion
    or missed detections under strict IoU thresholds (`AP75`, `mAP50:95`).

Core Mechanism (Inspired by ViM-DETR 2026, DCM-DETR 2026 & UAV-DETR OmniKernel 2025):
    1. Drop-in Replacement for `RepC3` on Original `rtdetr-r18.yaml` (Layers 19 & 22):
       Preserves the exact `RepC3` forward path when `enabled=False`, and initializes
       the residual injection gate to 0 so Epoch-0 output matches original `RepC3`.
    2. Four-Orientation Anisotropic Strip-Kernel Bank (0°, 90°, 45°, 135°):
       Uses parallel depthwise strip convolutions (1xK, Kx1) and diagonal-initialized
       dilated depthwise kernels to capture motion-blurred streaks along any trajectory.
    3. Trajectory-Aware Soft Router & CGLU Deblurring Concentrator:
       Estimates local motion orientation weights in FP32 (`autocast(enabled=False)`)
       and applies a Convolutional Gated Linear Unit (CGLU) that sharpens center-vs-trail
       contrast to re-concentrate smeared blur energy back to the UAV center.
    4. Full AMP (FP16) & ModelEMA (`deepcopy`) Safety:
       All normalizations/softmax run in FP32; no autograd graph tensors are cached on `self`.
"""

from __future__ import annotations
import copy
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _autopad(k: int | tuple[int, int], p: Optional[int | tuple[int, int]] = None, d: int = 1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else tuple(d * (x - 1) + 1 for x in k)
    if p is None:
        p = k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)
    return p


class _StdConvNormAct(nn.Module):
    """Standard Conv2d + BatchNorm2d + SiLU matching Ultralytics `Conv`."""

    def __init__(self, c1: int, c2: int, k: int | tuple[int, int] = 1, s: int = 1, p=None, g: int = 1, d: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _RepConvBlock(nn.Module):
    """Standard RepConv block (3x3 + 1x1 + identity BN) used inside Ultralytics `RepC3`."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.bn = nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None
        self.conv1 = _StdConvNormAct(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = _StdConvNormAct(c1, c2, 1, s, p=0, g=g, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)


class DMBABlock(nn.Module):
    """
    Directional Motion-Blur Anisotropic Deblurring Sub-Block (DMBA).
    Models 4 principal motion-blur orientations (0°, 90°, 45°, 135°), estimates pixel-wise
    trajectory weights in FP32, and re-concentrates smeared streak energy via CGLU gating.
    """

    def __init__(
        self,
        channels: int,
        strip_k: int = 7,
        gamma_max: float = 0.35,
        init_gain: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.strip_k = int(strip_k)
        self.gamma_max = float(gamma_max)
        self.enabled = bool(enabled)

        pad_k = self.strip_k // 2
        # 1. Four-orientation depthwise strip/diagonal kernels
        # 0 deg (horizontal motion blur): 1 x K
        self.dw_h = nn.Conv2d(channels, channels, kernel_size=(1, self.strip_k), padding=(0, pad_k), groups=channels, bias=False)
        # 90 deg (vertical motion blur): K x 1
        self.dw_v = nn.Conv2d(channels, channels, kernel_size=(self.strip_k, 1), padding=(pad_k, 0), groups=channels, bias=False)
        # 45 deg & 135 deg (diagonal motion blur): 3x3 dilated (d=2 -> 5x5 effective span)
        self.dw_d1 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
        self.dw_d2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
        self.bn_dir = nn.BatchNorm2d(channels)

        # 2. Pixel-wise orientation router (outputs 4 orientation logits per pixel)
        mid_c = max(channels // 4, 16)
        self.dir_router = nn.Sequential(
            nn.Conv2d(channels, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(),
            nn.Conv2d(mid_c, 4, kernel_size=3, padding=1, bias=True),
        )

        # 3. CGLU (Convolutional Gated Linear Unit) deblurring concentrator
        self.cglu_value = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.cglu_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.cglu_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Bounded residual parameter u_gamma (initialized to 0 -> exact identity at Epoch 0)
        self.u_gamma = nn.Parameter(torch.tensor(float(init_gain), dtype=torch.float32))
        self._init_diagonal_priors()

    def _init_diagonal_priors(self) -> None:
        """Initializes diagonal depthwise kernels along 45° and 135° lines."""
        with torch.no_grad():
            self.dw_d1.weight.zero_()
            self.dw_d2.weight.zero_()
            for i in range(3):
                self.dw_d1.weight[:, 0, i, i] = 1.0 / 3.0
                self.dw_d2.weight[:, 0, i, 2 - i] = 1.0 / 3.0
            nn.init.zeros_(self.dir_router[-1].weight)
            nn.init.zeros_(self.dir_router[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        # Compute 4 directional responses
        b_h = self.dw_h(x)
        b_v = self.dw_v(x)
        b_d1 = self.dw_d1(x)
        b_d2 = self.dw_d2(x)

        # Orientation routing in FP32 for AMP stability
        raw_logits = self.dir_router(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            w_dir = F.softmax(raw_logits.float(), dim=1).to(x.dtype)

        f_dir = (
            w_dir[:, 0:1] * b_h
            + w_dir[:, 1:2] * b_v
            + w_dir[:, 2:3] * b_d1
            + w_dir[:, 3:4] * b_d2
        )
        f_dir = F.silu(self.bn_dir(f_dir))

        # Center-vs-trail deblurring difference gate (CGLU)
        diff_highpass = x - f_dir
        val = self.cglu_value(f_dir)
        gate = torch.sigmoid(self.cglu_gate(diff_highpass))
        f_deblur = self.cglu_proj(val * gate)

        with torch.autocast(device_type=x.device.type, enabled=False):
            scale = (self.gamma_max * torch.tanh(self.u_gamma.float())).to(x.dtype)

        return x + scale * f_deblur


class DMBARepC3(nn.Module):
    """
    Drop-in Replacement for Ultralytics `RepC3` with Directional Motion-Blur Anisotropic
    Deblurring (Innovation 5).
    Compatible with YAML signature: `[-1, 3, DMBARepC3, [256, 0.5, 7, 0.35, 0.0, True]]`
    or standard `[-1, 3, DMBARepC3, [256, 0.5]]`.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 3,
        e: float = 0.5,
        strip_k: int = 7,
        gamma_max: float = 0.35,
        init_gain: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.c_ = c_
        self.enabled = bool(enabled)

        # Standard RepC3 backbone branches (using local or Ultralytics Conv/RepConv)
        try:
            from ultralytics.nn.modules.conv import Conv, RepConv
            self.cv1 = Conv(c1, c_, 1, 1)
            self.cv2 = Conv(c1, c_, 1, 1)
            self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
            self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()
        except Exception:
            self.cv1 = _StdConvNormAct(c1, c_, 1, 1)
            self.cv2 = _StdConvNormAct(c1, c_, 1, 1)
            self.m = nn.Sequential(*[_RepConvBlock(c_, c_) for _ in range(n)])
            self.cv3 = _StdConvNormAct(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

        self.dmba = DMBABlock(
            channels=c_,
            strip_k=strip_k,
            gamma_max=gamma_max,
            init_gain=init_gain,
            enabled=enabled,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.m(self.cv1(x)) + self.cv2(x)
        if self.enabled:
            feat = self.dmba(feat)
        return self.cv3(feat)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
