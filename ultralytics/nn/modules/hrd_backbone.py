"""R17 HRD-R18 high-resolution dual-stream operators.

The modules are deliberately atomic so Ultralytics can parse them from YAML:
the HR stream is initialized from P2 with PixelUnshuffle, updated from the
semantic stream, and injected back into P3/P4 with exact zero-init identity
paths.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .block import BasicBlock, ConvNormLayer

__all__ = ("HRStreamInit", "HRUpdate3", "HRInjectP3", "HRUpdate4", "HRInjectP4",
           "HRExchangeStage3", "HRExchangeStage4", "HRDPResNet")


def _zero_projection(c_in: int, c_out: int) -> ConvNormLayer:
    layer = ConvNormLayer(c_in, c_out, 1, 1, act=None)
    nn.init.zeros_(layer.norm.weight)
    nn.init.zeros_(layer.norm.bias)
    return layer


class HRStreamInit(nn.Module):
    """Lossless P2 space-to-depth initialization: 160x160 -> 80x80."""
    def __init__(self, c1: int, c_hr: int = 64):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.proj = ConvNormLayer(int(c1) * 4, int(c_hr), 1, 1, act="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pixel_unshuffle(x))


class _HRUpdate(nn.Module):
    def __init__(self, c_s: int, c_hr: int, act: str = "relu"):
        super().__init__()
        self.c_s, self.c_hr = int(c_s), int(c_hr)
        self.semantic_proj = ConvNormLayer(self.c_s, self.c_hr, 1, 1, act=None)
        self.gate = nn.Conv2d(self.c_hr, self.c_hr, 1)
        self.refine = BasicBlock(self.c_hr, self.c_hr, 1, True, act=act, variant="d")

    def forward(self, values):
        # Ultralytics multi-source routes arrive as a list.
        first, second = values
        # The documented YAML order is [H_previous, S_i].  Accept the
        # semantic-first order as well so direct Python callers remain clear.
        if first.shape[1] == self.c_hr and second.shape[1] == self.c_s:
            h, s = first, second
        else:
            s, h = first, second
        sem = self.semantic_proj(s)
        if sem.shape[-2:] != h.shape[-2:]:
            sem = F.interpolate(sem, size=h.shape[-2:], mode="nearest")
        return self.refine(h * (1.0 + torch.sigmoid(self.gate(sem))))


class HRUpdate3(_HRUpdate):
    """Stage3 semantic-to-HR update at 80x80."""


class HRUpdate4(_HRUpdate):
    """Stage4 semantic-to-HR update, using nearest upsampling to HR size."""


class _HRInject(nn.Module):
    def __init__(self, c_s: int, c_hr: int, stage: int):
        super().__init__()
        self.c_s, self.c_hr, self.stage = int(c_s), int(c_hr), int(stage)
        self.gamma = nn.Parameter(torch.zeros(1, self.c_s, 1, 1))
        self.delta = ConvNormLayer(self.c_hr, self.c_s, 1, 1, act=None)
        self.zero_proj = _zero_projection(self.c_hr, self.c_s)
        self.pixel_unshuffle = nn.PixelUnshuffle(2) if stage == 4 else None

    @property
    def proj(self):
        """Compatibility alias for the zero-initialized terminal projection."""
        return self.zero_proj

    @property
    def proj_bn(self):
        return self.zero_proj.norm

    def forward(self, values):
        s, h = values
        if self.pixel_unshuffle is not None:
            h = self.pixel_unshuffle(h)
        elif h.shape[-2:] != s.shape[-2:]:
            h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
        return s + self.zero_proj(h) + self.gamma * self.delta(h)


class HRInjectP3(_HRInject):
    def __init__(self, c_s: int, c_hr: int):
        super().__init__(c_s, c_hr, stage=3)


class HRInjectP4(_HRInject):
    def __init__(self, c_s: int, c_hr: int):
        super().__init__(c_s, int(c_hr) * 4, stage=4)


class HRExchangeStage3(nn.Module):
    def __init__(self, c_s: int = 128, c_hr: int = 64):
        super().__init__(); self.update = HRUpdate3(c_s, c_hr); self.inject = HRInjectP3(c_s, c_hr)

    def forward(self, values):
        s, h = values; h = self.update((s, h)); return self.inject((s, h)), h


class HRExchangeStage4(nn.Module):
    def __init__(self, c_s: int = 256, c_hr: int = 64):
        super().__init__(); self.update = HRUpdate4(c_s, c_hr); self.inject = HRInjectP4(c_s, c_hr)

    def forward(self, values):
        s, h = values; h = self.update((s, h)); return self.inject((s, h)), h


class HRDPResNet(nn.Module):
    """Marker wrapper; production YAMLs use the atomic HRD modules above."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        raise RuntimeError("Use HRD atomic YAML modules instead of HRDPResNet directly")
