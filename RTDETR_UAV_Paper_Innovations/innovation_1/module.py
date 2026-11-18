# -*- coding: utf-8 -*-
"""
Innovation 1: Wavelet-Contrast Anisotropic Shallow Feature Prior (WCA-SFP)
==========================================================================
Target Stage: Backbone Layer 4 (P2/4, 160x160) -> Neck Layer 17 (P3/8, 80x80)
Compatible with: Ultralytics YOLOv8.0.201 RT-DETR-R18, Python 3.10, PyTorch 2.3.1+cu121

Core Mathematical Mechanisms:
1. Orthogonal 2D Haar Wavelet Stride-2 Decomposition on P2:
   Decomposes P2 [B, C2, 2H, 2W] into {U_LL, U_LH, U_HL, U_HH} [B, Ch, H, W]
   without aliasing or learnable downsampling information loss.
2. Anisotropic Diagonal-Weighted High-Frequency Energy (E_wave) &
   Center-Surround Local Contrast (D_cs):
   Emphasizes compact point/blob-like UAV targets (strong in diagonal HH sub-band
   and local center-surround difference) while suppressing 1D extended background
   edges (clouds, horizon lines, building roofs).
3. Semantic-Contrast Co-Gated Residual Injection into P3:
   Injects filtered high-frequency boundary cues into P3 with a bounded residual
   gain alpha_max * sigmoid(gamma) (initialized to 0.1), keeping the 3-scale
   Transformer token count strictly at 8,400 (zero sequence length expansion).
4. Train-Time Gaussian-Ellipse Shallow Auxiliary Supervision (ShallowEllipseAuxLoss):
   Supervises the spatial objectness logits Z_obj during training only, with zero
   inference-time overhead.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletTransform2D(nn.Module):
    """
    Parameter-free Orthogonal 2D Haar Discrete Wavelet Transform (DWT) with stride 2.
    Safe under both FP32 and AMP (FP16/BF16).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        # Construct 2x2 orthogonal Haar filters: LL, LH, HL, HH
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        # filter_bank: [4, 1, 2, 2] -> repeat for depthwise grouped conv [4*C, 1, 2, 2]
        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("kernels", kernels, persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, 2H, 2W]
        Returns:
            u_ll, u_lh, u_hl, u_hh: each of shape [B, C, H, W]
        """
        # x: [B, C, 2H, 2W]
        b, c, h2, w2 = x.shape
        if h2 % 2 != 0 or w2 % 2 != 0:
            x = F.pad(x, (0, w2 % 2, 0, h2 % 2), mode="replicate")
        weight = self.kernels.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)  # [4*C, 1, 2, 2]
        out = F.conv2d(x, weight, bias=None, stride=2, padding=0, groups=c)  # [B, 4*C, H, W]
        out = out.view(b, c, 4, out.shape[-2], out.shape[-1])  # [B, C, 4, H, W]
        u_ll = out[:, :, 0]  # [B, C, H, W]
        u_lh = out[:, :, 1]  # [B, C, H, W]
        u_hl = out[:, :, 2]  # [B, C, H, W]
        u_hh = out[:, :, 3]  # [B, C, H, W]
        return u_ll, u_lh, u_hl, u_hh


class CenterSurroundContrast(nn.Module):
    """
    Computes normalized local center-surround contrast D_cs on feature maps
    to highlight isolated point/tiny UAV targets against locally smooth or
    slowly varying sky/cloud/foliage backgrounds.
    """

    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        # Fixed 3x3 dilated surround kernel (8 surrounding ring neighbors at distance 2)
        ring = torch.tensor(
            [[1.0, 1.0, 1.0],
             [1.0, 0.0, 1.0],
             [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        ) / 8.0
        self.register_buffer("ring_kernel", ring.view(1, 1, 3, 3), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            d_cs: [B, C, H, W] in range [0, 1)
        """
        # x: [B, C, H, W]
        _, c, _, _ = x.shape
        # The small denominator can overflow FP16 gradients even when tanh
        # keeps the forward values finite. Keep this ratio and its backward in FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            mu_cen = F.avg_pool2d(x_fp32, kernel_size=3, stride=1, padding=1)
            ring_w = self.ring_kernel.to(device=x.device).repeat(c, 1, 1, 1)
            x_pad = F.pad(x_fp32, (2, 2, 2, 2), mode="replicate")
            mu_sur = F.conv2d(x_pad, ring_w, bias=None, stride=1, padding=0, dilation=2, groups=c)
            contrast = (mu_cen - mu_sur).abs() / (mu_sur.abs() + self.eps)
            d_cs = torch.tanh(contrast)
        return d_cs.to(dtype=x.dtype)


class WCASFP(nn.Module):
    """
    Wavelet-Contrast Anisotropic Shallow Feature Prior (WCA-SFP) Module.

    Can be used in two modes in Ultralytics YAML:
    Mode A (Dual-input bridge [from=[4, 5]]):
        Receives [p2, p3] where p2 is [B, C2, 160, 160] and p3 is [B, C3, 80, 80],
        and outputs enhanced P3 feature of shape [B, c2_out, 80, 80].
    Mode B (Single-input P3 enhancer [from=5]):
        If only p3 [B, C3, 80, 80] is passed, performs intra-scale wavelet-contrast
        enhancement and outputs [B, c2_out, 80, 80].

    When `enabled=False`, strictly degenerates to a standard 1x1 projection Conv+BN
    on P3 (exact baseline RT-DETR-R18 Layer 17 behavior).
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
            if len(c1) != 2:
                raise ValueError(f"WCASFP expects 1 or 2 inputs [P2, P3], got {len(c1)} channels: {c1}")
            self.c_p2, self.c_p3 = int(c1[0]), int(c1[1])
            self.has_p2 = True
        else:
            self.c_p2, self.c_p3 = int(c1), int(c1)
            self.has_p2 = False

        self.c_out = int(c2)
        self.c_mid = max(32, self.c_out // 2)
        self.alpha_max = float(alpha_max)
        self.diagonal_weight = float(diagonal_weight)
        self.enabled = bool(enabled)

        # Baseline-compatible P3 projection (matches Layer 17 Conv(c_p3, c_out, 1, 1, bias=False) + BN)
        self.p3_proj = nn.Sequential(
            nn.Conv2d(self.c_p3, self.c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c_out),
        )

        # P2 channel projection before Haar wavelet decomposition
        self.p2_proj = nn.Sequential(
            nn.Conv2d(self.c_p2, self.c_mid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c_mid),
            nn.SiLU(inplace=True),
        )

        self.dwt = HaarWaveletTransform2D(self.c_mid)
        self.cs_contrast = CenterSurroundContrast(self.c_mid)

        # Fuse directional high-frequency subbands [U_LH, U_HL, U_HH] -> c_mid
        self.hf_fuse = nn.Sequential(
            nn.Conv2d(self.c_mid * 3, self.c_mid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c_mid, self.c_mid, kernel_size=3, stride=1, padding=1, groups=self.c_mid, bias=False),
            nn.BatchNorm2d(self.c_mid),
            nn.SiLU(inplace=True),
        )

        # Semantic projection of P3 for gating
        self.p3_sem = nn.Sequential(
            nn.Conv2d(self.c_out, self.c_mid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c_mid),
            nn.SiLU(inplace=True),
        )

        # Target-guided spatial gate predictor -> outputs single-channel objectness logits Z_obj
        self.gate_head = nn.Sequential(
            nn.Conv2d(self.c_mid * 2 + 1, self.c_mid // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.c_mid // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c_mid // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # Refinement block before residual injection into P3
        self.detail_refine = nn.Sequential(
            nn.Conv2d(self.c_mid * 2, self.c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c_out, self.c_out, kernel_size=3, stride=1, padding=1, groups=self.c_out, bias=False),
            nn.BatchNorm2d(self.c_out),
        )

        # Learnable logit parameter u_gamma such that alpha_max * sigmoid(u_gamma) == init_gain
        ratio = min(max(init_gain / max(alpha_max, 1e-4), 1e-3), 1.0 - 1e-3)
        init_u = math.log(ratio / (1.0 - ratio))
        self.u_gamma = nn.Parameter(torch.tensor(init_u, dtype=torch.float32))

        # Cached spatial objectness logits & map from most recent forward pass (for auxiliary supervision)
        self.last_obj_logits: Optional[torch.Tensor] = None
        self.last_obj_map: Optional[torch.Tensor] = None

    def get_residual_gain(self) -> torch.Tensor:
        """Returns bounded scalar residual gain in (0, alpha_max)."""
        return self.alpha_max * torch.sigmoid(self.u_gamma)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        Args:
            x: Either [p2, p3] list/tuple where p2 is [B, C2, 2H, 2W] and p3 is [B, C3, H, W],
               or single tensor p3 [B, C3, H, W].
        Returns:
            p3_enh: [B, c_out, H, W] (e.g. [B, 256, 80, 80] for 640x640 input)
        """
        if isinstance(x, (list, tuple)):
            p2, p3 = x[0], x[1]
        else:
            p3 = x
            p2 = None

        # Base projected P3: [B, c_out, H, W]
        p3_base = self.p3_proj(p3)

        if not self.enabled:
            self.last_obj_logits = None
            self.last_obj_map = None
            return p3_base

        b, _, h, w = p3_base.shape

        if p2 is None:
            # Upsample P3 by 2x if standalone single-input mode is used
            p2_in = F.interpolate(p3, size=(h * 2, w * 2), mode="bilinear", align_corners=False)
        else:
            if p2.shape[-2:] != (h * 2, w * 2):
                p2_in = F.interpolate(p2, size=(h * 2, w * 2), mode="bilinear", align_corners=False)
            else:
                p2_in = p2

        # 1. Project P2 and apply orthogonal 2D Haar DWT -> [B, c_mid, H, W]
        u = self.p2_proj(p2_in)  # [B, c_mid, 2H, 2W]
        u_ll, u_lh, u_hl, u_hh = self.dwt(u)  # each [B, c_mid, H, W]

        # 2. Compute anisotropic diagonal-weighted wavelet energy E_wave: [B, 1, H, W]
        energy_sq = (
            u_lh.float().pow(2)
            + u_hl.float().pow(2)
            + self.diagonal_weight * u_hh.float().pow(2)
        ).mean(dim=1, keepdim=True)
        e_wave = torch.sqrt(energy_sq + 1e-6).to(dtype=p3_base.dtype)  # [B, 1, H, W]

        # 3. Directional high-frequency feature fusion + Center-Surround contrast modulation
        f_hf = self.hf_fuse(torch.cat([u_lh, u_hl, u_hh], dim=1))  # [B, c_mid, H, W]
        d_cs = self.cs_contrast(u_ll)  # [B, c_mid, H, W]
        f_hf_mod = f_hf * (1.0 + d_cs)  # [B, c_mid, H, W]

        # 4. Semantic-Contrast Co-Gating
        s_p3 = self.p3_sem(p3_base)  # [B, c_mid, H, W]
        gate_in = torch.cat([s_p3, f_hf_mod, e_wave], dim=1)  # [B, 2*c_mid + 1, H, W]
        z_obj = self.gate_head(gate_in)  # [B, 1, H, W]
        m_obj = torch.sigmoid(z_obj)  # [B, 1, H, W]

        # Only the logits are needed for training supervision. Never retain
        # autograd tensors on the module used by EMA/checkpoint deepcopy.
        self.last_obj_logits = z_obj if self.training else None
        self.last_obj_map = m_obj.detach() if self.training else None

        # 5. Filtered boundary injection into P3 with conservative residual gain
        detail_feat = self.detail_refine(torch.cat([u_ll, f_hf_mod], dim=1) * m_obj)  # [B, c_out, H, W]
        gain = self.get_residual_gain().to(dtype=p3_base.dtype)
        p3_enh = p3_base + gain * detail_feat  # [B, c_out, H, W]
        return p3_enh


class ShallowEllipseAuxLoss(nn.Module):
    """
    Train-Time-Only Gaussian-Ellipse Shallow Auxiliary Supervision for WCA-SFP.
    Generates size-filtered elliptical Gaussian soft targets on the P3 grid (80x80)
    and computes Quality Focal + Soft Dice loss on `WCASFP.last_obj_logits`.
    Zero inference overhead.
    """

    def __init__(
        self,
        r_max_norm: float = 0.20,
        min_sigma: float = 1.2,
        kappa: float = 0.5,
        loss_weight: float = 0.25,
    ):
        super().__init__()
        self.r_max_norm = float(r_max_norm)  # normalized sqrt(w*h) threshold (0.20 = 128px on 640 image)
        self.min_sigma = float(min_sigma)
        self.kappa = float(kappa)
        self.loss_weight = float(loss_weight)

    @torch.no_grad()
    def build_gaussian_ellipse_targets(
        self,
        shape_hw: Tuple[int, int],
        batch_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Args:
            shape_hw: (H, W) of P3 feature map, e.g. (80, 80)
            batch_bboxes: [num_gts, 4] normalized (cx, cy, w, h) in [0, 1]
            batch_idx: [num_gts] integer image index in [0, B-1]
            batch_size: int B
            device: torch.device
        Returns:
            target_map: [B, 1, H, W] in [0, 1]
        """
        h, w = shape_hw
        target_map = torch.zeros((batch_size, 1, h, w), dtype=torch.float32, device=device)
        if batch_bboxes.numel() == 0:
            return target_map

        gy = torch.arange(h, dtype=torch.float32, device=device).view(1, h, 1) + 0.5
        gx = torch.arange(w, dtype=torch.float32, device=device).view(1, 1, w) + 0.5

        for b in range(batch_size):
            mask_b = batch_idx == b
            if not torch.any(mask_b):
                continue
            boxes = batch_bboxes[mask_b].float()  # [M, 4]
            scales = torch.sqrt((boxes[:, 2] * boxes[:, 3]).clamp(min=1e-8))
            valid = scales <= self.r_max_norm
            if not torch.any(valid):
                valid = torch.ones_like(scales, dtype=torch.bool)
            boxes = boxes[valid]  # [K, 4]

            cx = boxes[:, 0].view(-1, 1, 1) * w
            cy = boxes[:, 1].view(-1, 1, 1) * h
            bw = (boxes[:, 2].view(-1, 1, 1) * w * self.kappa).clamp(min=self.min_sigma)
            bh = (boxes[:, 3].view(-1, 1, 1) * h * self.kappa).clamp(min=self.min_sigma)

            dist_sq = ((gx - cx) / bw).pow(2) + ((gy - cy) / bh).pow(2)  # [K, H, W]
            gauss = torch.exp(-0.5 * dist_sq)  # [K, H, W]
            target_map[b, 0] = gauss.max(dim=0).values

        return target_map

    def forward(
        self,
        obj_logits: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes auxiliary shallow supervision loss.
        Args:
            obj_logits: [B, 1, H, W] from WCASFP.last_obj_logits
            batch: dict with 'bboxes' [N, 4] and 'batch_idx' [N]
        Returns:
            scalar loss tensor
        """
        if obj_logits is None:
            return torch.tensor(0.0)

        b, _, h, w = obj_logits.shape
        device = obj_logits.device
        bboxes = batch.get("bboxes", torch.zeros((0, 4), device=device)).to(device)
        if "batch_idx" in batch:
            batch_idx = batch["batch_idx"].to(device=device, dtype=torch.long).view(-1)
        elif "gt_groups" in batch:
            gt_groups = batch["gt_groups"]
            batch_idx = torch.cat(
                [torch.full((g,), i, dtype=torch.long, device=device) for i, g in enumerate(gt_groups)]
            ) if sum(gt_groups) > 0 else torch.zeros((0,), dtype=torch.long, device=device)
        else:
            batch_idx = torch.zeros(( len(bboxes),), dtype=torch.long, device=device)

        targets = self.build_gaussian_ellipse_targets((h, w), bboxes, batch_idx, b, device)
        logits_fp32 = obj_logits.float()
        probs = torch.sigmoid(logits_fp32)

        # Quality Focal Binary Cross-Entropy
        bce = F.binary_cross_entropy_with_logits(logits_fp32, targets, reduction="none")
        focal_weight = (targets - probs).abs().pow(2.0)
        pos_scale = torch.where(targets > 0.05, 4.0, 1.0)
        loss_focal = (bce * focal_weight * pos_scale).mean()

        # Soft Dice on foreground support (targets > 0.1)
        fg_mask = (targets > 0.1).float()
        inter = (probs * fg_mask).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + fg_mask.sum(dim=(1, 2, 3))
        loss_dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

        return self.loss_weight * (loss_focal + 0.5 * loss_dice)
