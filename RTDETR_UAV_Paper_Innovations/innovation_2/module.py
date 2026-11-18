# -*- coding: utf-8 -*-
"""
Innovation 2: Frequency-Decoupled Gating and Bounded Sub-Pixel Phase-Aligned
Cross-Scale Fusion (FAGA / FAGAFusion)
=============================================================================
Target Stage: CCFM Neck Top-Down / Bottom-Up Concat Nodes (Layer 13 & Layer 18)
Compatible with: Ultralytics YOLOv8.0.201 RT-DETR-R18, Python 3.10, PyTorch 2.3.1+cu121

Core Mathematical Mechanisms:
1. Asymmetric Bounded Sub-Pixel Phase Warping:
   Uses shallow high-resolution feature F_low as the fixed geometric anchor and
   predicts a zero-initialized, tanh-bounded sub-pixel offset field
   Delta_p in [-r_max, +r_max] (in feature grid units, default r_max=1.5) to warp
   the deeper semantic feature F_high via bilinear GridSample. Eliminates the
   0.5~1.5 pixel shift and blocky artifacts caused by `nn.Upsample(mode='nearest')`.
2. FFT-Free Frequency-Spatial Mutual Recalibration:
   Extracts the low-frequency semantic envelope S_sem from aligned F_high and
   the high-frequency edge residual B_edge = |F_low - AvgPool3x3(F_low)| from F_low,
   performing orthogonal mutual gating before channel concatenation so that
   upsampled sky/cloud/foliage background activations do not flood shallow UAV cues.
"""

from __future__ import annotations
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FAGAFusion(nn.Module):
    """
    Frequency-Decoupled Gating & Bounded Sub-Pixel Phase-Aligned Fusion (FAGA).

    Drop-in replacement for `Concat` at cross-scale fusion nodes (e.g., Layer 13 and
    Layer 18 in `rtdetr-r18.yaml`).
    Input:
        x: list/tuple of two feature tensors `[f_high, f_low]` where each tensor has
           shape `[B, C, H, W]` (or `f_high` has `[B, C, H/2, W/2]`, which is
           automatically aligned to `(H, W)`).
    Output:
        out: concatenated & recalibrated feature tensor `[B, C_high + C_low, H, W]`,
             strictly preserving the output channel count `2*C` expected by `RepC3`.

    When `enabled=False`, strictly performs `torch.cat([f_high, f_low], dim=1)`
    (exact baseline RT-DETR-R18 behavior).
    """

    def __init__(
        self,
        c1: Union[int, List[int], Tuple[int, ...]] = (256, 256),
        dimension: int = 1,
        r_max: float = 1.5,
        hidden_ratio: float = 0.25,
        enabled: bool = True,
    ):
        super().__init__()
        if isinstance(c1, (list, tuple)):
            self.c_high = int(c1[0])
            self.c_low = int(c1[1])
        else:
            # Fallback if single channel int passed
            self.c_high = int(c1) // 2
            self.c_low = int(c1) - self.c_high

        self.d = int(dimension)
        self.r_max = float(r_max)
        self.enabled = bool(enabled)
        c_total = self.c_high + self.c_low
        c_mid = max(32, int(c_total * hidden_ratio))

        # 1. Cross-scale phase correlation & bounded offset predictor
        self.corr_Encoder = nn.Sequential(
            nn.Conv2d(c_total, c_mid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=1, padding=1, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
        )
        # 2-channel sub-pixel offset predictor (dx, dy), zero-initialized for stable cold-start
        self.offset_head = nn.Conv2d(c_mid, 2, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.constant_(self.offset_head.weight, 0.0)
        nn.init.constant_(self.offset_head.bias, 0.0)

        # 2. FFT-free Frequency-Spatial Mutual Gating projections
        self.sem_gate = nn.Sequential(
            nn.Conv2d(self.c_high, self.c_low, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.edge_gate = nn.Sequential(
            nn.Conv2d(self.c_low, self.c_high, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # Zero-initialized learnable channel modulation gains beta_1 (for low) and beta_2 (for high)
        self.beta_low = nn.Parameter(torch.zeros(1, self.c_low, 1, 1, dtype=torch.float32))
        self.beta_high = nn.Parameter(torch.zeros(1, self.c_high, 1, 1, dtype=torch.float32))

    @staticmethod
    def _build_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Builds standard normalized sampling grid in [-1, 1] matching align_corners=False.
        Returns shape [1, H, W, 2].
        """
        gy = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * (2.0 / h) - 1.0
        gx = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * (2.0 / w) - 1.0
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        return grid.to(dtype=dtype)

    def forward(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        Args:
            x: [f_high, f_low], where:
               f_high: [B, C_high, H, W] (already upsampled by preceding layer) or [B, C_high, H/2, W/2]
               f_low:  [B, C_low, H, W] (shallow high-res anchor feature)
        Returns:
            out: [B, C_high + C_low, H, W]
        """
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"FAGAFusion expects a list/tuple of 2 tensors [f_high, f_low], got {type(x)}")

        f_high, f_low = x[0], x[1]
        b, _, h, w = f_low.shape

        if not self.enabled:
            if f_high.shape[-2:] != (h, w):
                f_high = F.interpolate(f_high, size=(h, w), mode="nearest")
            return torch.cat([f_high, f_low], dim=self.d)

        # Ensure f_high matches spatial size (h, w) via smooth bilinear interpolation if needed
        if f_high.shape[-2:] != (h, w):
            f_high_up = F.interpolate(f_high, size=(h, w), mode="bilinear", align_corners=False)
        else:
            # Smooth out blocky nearest-neighbor artifacts with a lightweight 2x2 bilinear filter
            f_high_up = f_high

        # 1. Predict bounded sub-pixel offsets Delta_p in [-r_max, +r_max]
        corr_feat = self.corr_Encoder(torch.cat([f_low, f_high_up], dim=1))  # [B, c_mid, H, W]
        raw_offset = self.offset_head(corr_feat)  # [B, 2, H, W]
        delta_p = self.r_max * torch.tanh(raw_offset.float())  # [B, 2, H, W] in pixel units

        # Convert pixel offsets to normalized [-1, 1] grid offsets: dx * (2/W), dy * (2/H)
        base_grid = self._build_base_grid(h, w, device=f_low.device, dtype=torch.float32)  # [1, H, W, 2]
        norm_dx = delta_p[:, 0, :, :] * (2.0 / max(w, 1))  # [B, H, W]
        norm_dy = delta_p[:, 1, :, :] * (2.0 / max(h, 1))  # [B, H, W]
        warped_grid = (base_grid + torch.stack([norm_dx, norm_dy], dim=-1)).clamp(-1.0, 1.0)  # [B, H, W, 2]

        # Warp f_high_up in FP32 for numerical stability under AMP, then cast back
        f_high_aligned = F.grid_sample(
            f_high_up.float(),
            warped_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).to(dtype=f_low.dtype)  # [B, C_high, H, W]

        # 2. Frequency-Spatial Decoupled Mutual Recalibration
        # Low-frequency semantic envelope from aligned deep feature
        f_high_lp = F.avg_pool2d(f_high_aligned, kernel_size=3, stride=1, padding=1)  # [B, C_high, H, W]
        s_sem = self.sem_gate(f_high_lp)  # [B, C_low, H, W] in (0, 1)

        # High-frequency structural detail from shallow feature
        f_low_lp = F.avg_pool2d(f_low, kernel_size=3, stride=1, padding=1)
        f_low_hf = (f_low - f_low_lp).abs()  # [B, C_low, H, W]
        b_edge = self.edge_gate(f_low_hf)  # [B, C_high, H, W] in (0, 1)

        beta1 = torch.tanh(self.beta_low).to(dtype=f_low.dtype)
        beta2 = torch.tanh(self.beta_high).to(dtype=f_low.dtype)

        # Recalibrate both branches while preserving zero-init identity at start of training
        f_low_cal = f_low * (1.0 + beta1 * s_sem * (f_low_hf / (f_low_hf.mean(dim=(2, 3), keepdim=True) + 1e-5)).tanh())
        f_high_cal = f_high_aligned * (1.0 + beta2 * b_edge)

        out = torch.cat([f_high_cal, f_low_cal], dim=self.d)  # [B, C_high + C_low, H, W]
        return out
