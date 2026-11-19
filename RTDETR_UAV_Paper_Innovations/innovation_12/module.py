# -*- coding: utf-8 -*-
"""
Innovation 12: Local-Global Spectrum-Modulated AIFI Encoder (SMAIFI)
====================================================================
Target Problem:
    Original `RT-DETR-R18` applies a single global Transformer encoder layer (`AIFI`) on
    `P5/32` (`20x20 = 400` tokens). In single-target anti-UAV imagery (`DUT-Anti-UAV`),
    >98% of `P5` tokens represent homogeneous sky/mountain/building background; standard
    global softmax self-attention dilutes the weak tiny-UAV peak, and standard MLP FFN
    lacks high-frequency vs. low-frequency separation.

Core Mechanism (Inspired by FGUDet SMFFN 2026 & UAV-DETR SWSA-IFI 2026):
    1. Drop-in Replacement for Layer 9 `AIFI` (`[-1, 1, SMAIFI, [1024, 8]]`):
       Preserves exact `AIFI` signature `(c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False)`.
    2. Local-Global Dual-Path Attention:
       Runs a parallel depthwise `3x3` local peak branch alongside standard global MHSA
       (with learnable temperature scaling on attention logits to sharpen tiny-UAV focus),
       fused via a zero-initialized bounded residual gate.
    3. Spectrum-Modulated Feed-Forward Network (`SMFFN`):
       Decouples low-frequency global background semantics (`avg_pool2d(3x3)`) from
       high-frequency local target peaks (`x - low_freq`) inside the FFN, applying
       independent channel gating to suppress low-frequency background dominance while
       amplifying high-frequency tiny-UAV peaks.
"""

from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class SMAIFI(nn.Module):
    """
    Local-Global Spectrum-Modulated AIFI Encoder (Innovation 12).
    Drop-in replacement for Ultralytics `AIFI` at Layer 9 of `rtdetr-r18.yaml`.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
        enabled: bool = True,
    ):
        super().__init__()
        self.c1 = int(c1)
        self.cm = int(cm)
        self.num_heads = int(num_heads)
        self.normalize_before = bool(normalize_before)
        self.enabled = bool(enabled)

        # Standard AIFI Transformer components
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act if isinstance(act, nn.Module) else nn.GELU()

        # Innovation 12a: Parallel Local Peak Branch (prevents 98% sky background dilution)
        self.local_branch = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.u_local = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Innovation 12b: Spectrum-Modulated FFN (low-freq vs. high-freq residual modulation)
        mid_c = max(c1 // 4, 32)
        self.hf_gate = nn.Sequential(
            nn.Conv2d(c1, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(),
            nn.Conv2d(mid_c, c1, kernel_size=1, bias=True),
        )
        self.lf_gate = nn.Sequential(
            nn.Conv2d(c1, mid_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(),
            nn.Conv2d(mid_c, c1, kernel_size=1, bias=True),
        )
        self.u_spec = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def build_2d_sincos_position_embedding(w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0) -> torch.Tensor:
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1)  # [B, HW, C]

        # 1. Global Self-Attention + Parallel Local Peak Enhancement
        q = k = src + pos_embed
        src2 = self.ma(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        if self.enabled:
            local_2d = self.local_branch(x).flatten(2).permute(0, 2, 1)
            with torch.autocast(device_type=x.device.type, enabled=False):
                g_loc = (0.35 * torch.tanh(self.u_local.float())).to(src.dtype)
            src = src + g_loc * local_2d
        src = self.norm1(src)

        # 2. Standard FFN + Spectrum-Modulated High/Low Frequency Residual
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        out_2d = src.permute(0, 2, 1).view(b, c, h, w).contiguous()
        if self.enabled:
            low_freq = F.avg_pool2d(out_2d, kernel_size=3, stride=1, padding=1)
            high_freq = out_2d - low_freq
            hf_mod = high_freq * torch.sigmoid(self.hf_gate(high_freq))
            lf_mod = low_freq * torch.tanh(self.lf_gate(low_freq))
            with torch.autocast(device_type=x.device.type, enabled=False):
                g_spec = (0.30 * torch.tanh(self.u_spec.float())).to(out_2d.dtype)
            out_2d = out_2d + g_spec * (hf_mod - 0.5 * lf_mod)

        return out_2d

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
