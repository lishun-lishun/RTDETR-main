# -*- coding: utf-8 -*-
"""
Innovation 6: Signed Sparse Semantic Calibration (SSSC-Lite / SSSCRepC3)
========================================================================
Target Problem:
    In visible-light anti-UAV scenes (`DUT-Anti-UAV`), cluttered backgrounds such as
    building corners, tree branches, and powerlines produce high-energy false-positive
    activations on high-resolution feature maps (`P3/8` 80x80 and `P4/16` 40x40) after
    cross-scale concatenation. Conventional positive-only attention mechanisms amplify
    both true tiny UAVs and hard-negative clutter, causing false alarms in the Top-300
    decoder query selection.

Core Mechanism (Inspired by SSC-DETR Nov 2026 & Statistical Anomaly Drone Detection Dec 2025):
    1. Statistical Anomaly Saliency Routing (FP32 Protected):
       Combines learned saliency logits with standardized spatial z-score anomaly
       $Z = (E - \mu_E) / (\sigma_E + \epsilon)$ in FP32 to select Top-K_r (e.g., K_r=128)
       candidate object/clutter tokens while leaving dense background untouched.
    2. Zero-Initialized Bounded Sub-Pixel Offset Alignment (`|Δp| <= 0.25`):
       Predicts zero-initialized sub-pixel offsets bounded by 0.25 grid cells and samples
       locally aligned features via `F.grid_sample` in FP32.
    3. Norm-Constrained Bipolar Signed Residual Calibration:
       Generates signed residuals $r_k \in (-\tau, \tau)^C$ via $\tanh(W_{sign} f_k) \odot W_{mag} f_k$
       with L2 norm clipping. Positive residuals reinforce weak tiny UAV cues, while negative
       residuals actively cancel false-positive background clutter spikes.
    4. Drop-in Replacement for `RepC3` on Original `rtdetr-r18.yaml` (Layers 19 & 22):
       Preserves exact layer indices `[19, 22, 25]` and `ModelEMA` `deepcopy` compatibility.
"""

from __future__ import annotations
import copy
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .innovation_5 import _StdConvNormAct, _RepConvBlock


class SSSCLite(nn.Module):
    """
    Signed Sparse Semantic Calibration Module (SSSC-Lite).
    Selects Top-K_r sparse spatial tokens based on statistical anomaly + learned saliency,
    aligns local context via zero-initialized bounded sub-pixel GridSample (`|Δp| <= 0.25`),
    and scatters L2-norm-constrained bipolar signed residuals back to the feature map.
    """

    def __init__(
        self,
        channels: int,
        top_k: int = 128,
        max_offset: float = 0.25,
        norm_tau: float = 2.0,
        beta_max: float = 0.35,
        init_gain: float = 0.0,
        eps: float = 1e-4,
        enabled: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.top_k = int(top_k)
        self.max_offset = float(max_offset)
        self.norm_tau = float(norm_tau)
        self.beta_max = float(beta_max)
        self.eps = float(eps)
        self.enabled = bool(enabled)

        mid_c = max(channels // 4, 16)
        # 1. Saliency scoring head
        self.saliency_head = nn.Sequential(
            nn.Conv2d(channels + 1, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(),
            nn.Conv2d(mid_c, 1, kernel_size=1, bias=True),
        )

        # 2. Zero-initialized sub-pixel offset predictor (2D offset per selected token)
        self.offset_head = nn.Linear(channels, 2, bias=True)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

        # 3. Bipolar signed residual generator: tanh(W_sign * f) * (W_mag * f)
        self.sign_branch = nn.Sequential(
            nn.Linear(channels, mid_c, bias=False),
            nn.LayerNorm(mid_c),
            nn.SiLU(),
            nn.Linear(mid_c, channels, bias=True),
        )
        self.mag_branch = nn.Sequential(
            nn.Linear(channels, mid_c, bias=False),
            nn.LayerNorm(mid_c),
            nn.SiLU(),
            nn.Linear(mid_c, channels, bias=True),
        )

        # Bounded residual gate parameter u_beta (init 0 -> exact identity at Epoch 0)
        self.u_beta = nn.Parameter(torch.tensor(float(init_gain), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        b, c, h, w = x.shape
        hw = h * w
        k_r = min(self.top_k, hw)

        # 1. Statistical anomaly z-score in FP32 to prevent AMP underflow/overflow
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            energy = xf.abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
            mu_e = energy.mean(dim=(2, 3), keepdim=True)
            std_e = energy.std(dim=(2, 3), keepdim=True, unbiased=False)
            z_anom = torch.tanh((energy - mu_e) / (std_e + self.eps))  # [B, 1, H, W] bounded in (-1, 1)

        sal_in = torch.cat([x, z_anom.to(x.dtype)], dim=1)  # [B, C+1, H, W]
        sal_logits = self.saliency_head(sal_in)  # [B, 1, H, W]
        sal_prob = torch.sigmoid(sal_logits)     # [B, 1, H, W]

        # Flatten spatial tokens: [B, HW, 1]
        sal_flat = sal_prob.view(b, hw)
        _, topk_idx = torch.topk(sal_flat, k=k_r, dim=1, largest=True, sorted=False)  # [B, K_r]

        # Gather Top-K_r token features: [B, K_r, C]
        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, c)  # [B, K_r, C]
        f_sel = torch.gather(x_flat, dim=1, index=idx_expand)  # [B, K_r, C]

        # 2. Predict bounded sub-pixel offsets |Δp| <= max_offset (0.25 grid cells)
        raw_off = self.offset_head(f_sel)  # [B, K_r, 2]
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta_p = self.max_offset * torch.tanh(raw_off.float())  # [B, K_r, 2] ingrid cells
            # Compute normalized grid coordinates in [-1, 1] for F.grid_sample(align_corners=False)
            iy = (topk_idx // w).float()
            ix = (topk_idx % w).float()
            gx = ((ix + 0.5 + delta_p[..., 0]) / float(w)) * 2.0 - 1.0
            gy = ((iy + 0.5 + delta_p[..., 1]) / float(h)) * 2.0 - 1.0
            sample_grid = torch.stack([gx.clamp(-1.0, 1.0), gy.clamp(-1.0, 1.0)], dim=-1).unsqueeze(2)  # [B, K_r, 1, 2]
            f_aligned = F.grid_sample(
                x.float(),
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).squeeze(3).transpose(1, 2)  # [B, K_r, C]
            f_aligned = f_aligned.to(x.dtype)

        # 3. Bipolar signed residual with L2 norm constraint
        r_sign = torch.tanh(self.sign_branch(f_aligned))  # [B, K_r, C] in (-1, 1)
        r_mag = self.mag_branch(f_aligned)                # [B, K_r, C]
        r_raw = r_sign * r_mag

        with torch.autocast(device_type=x.device.type, enabled=False):
            r_fp32 = r_raw.float()
            r_norm = torch.linalg.vector_norm(r_fp32, ord=2, dim=-1, keepdim=True)
            clip_coef = torch.clamp(self.norm_tau / (r_norm + self.eps), max=1.0)
            r_clipped = (r_fp32 * clip_coef).to(x.dtype)
            gate_scale = (self.beta_max * torch.tanh(self.u_beta.float())).to(x.dtype)

        # Weight by saliency confidence at selected tokens so gradients flow through saliency_head
        sal_sel = torch.gather(sal_flat, dim=1, index=topk_idx).unsqueeze(-1).to(x.dtype)  # [B, K_r, 1]
        delta_sparse = gate_scale * sal_sel * r_clipped  # [B, K_r, C]

        # Scatter signed residuals back only to the K_r selected spatial positions
        out_flat = x_flat.scatter_add(dim=1, index=idx_expand, src=delta_sparse)
        out = out_flat.transpose(1, 2).view(b, c, h, w)
        return out


class SSSCRepC3(nn.Module):
    """
    Drop-in Replacement for Ultralytics `RepC3` with Post-Fusion Signed Sparse Semantic
    Calibration (Innovation 6).
    Compatible with YAML signature: `[-1, 3, SSSCRepC3, [256, 0.5, 128, 0.25, 2.0, 0.35, 0.0, True]]`
    or standard `[-1, 3, SSSCRepC3, [256, 0.5]]`.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 3,
        e: float = 0.5,
        top_k: int = 128,
        max_offset: float = 0.25,
        norm_tau: float = 2.0,
        beta_max: float = 0.35,
        init_gain: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.c_ = c_
        self.enabled = bool(enabled)

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

        self.sssc = SSSCLite(
            channels=c2,
            top_k=top_k,
            max_offset=max_offset,
            norm_tau=norm_tau,
            beta_max=beta_max,
            init_gain=init_gain,
            enabled=enabled,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv3(self.m(self.cv1(x)) + self.cv2(x))
        if self.enabled:
            out = self.sssc(out)
        return out

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
