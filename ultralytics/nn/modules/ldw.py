"""Late and early downsampling stages for the LDW-ResNet18 experiments.

The modules in this file are deliberately independent from :class:`BasicBlock`.
That keeps the historical RT-DETR R18 implementation unchanged while making the
location of the stride-2 operation explicit and easy to inspect with hooks.
"""

from __future__ import annotations

import torch
from torch import nn

from .block import BasicBlock, ConvNormLayer, get_activation

__all__ = (
    "DownsampleProjection",
    "ResidualBlock",
    "LateDownStage",
    "WidthMatchedEarlyDownStage",
)


class DownsampleProjection(nn.Module):
    """A 1x1 BN projection used only when residual shapes differ."""

    def __init__(self, c1: int, c2: int, stride: int = 1):
        super().__init__()
        self.conv = ConvNormLayer(c1, c2, 1, stride, act=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ResidualConv(nn.Module):
    """Conv-BN-activation with optional dilation, kept local to LDW stages."""

    def __init__(self, c1: int, c2: int, stride: int, dilation: int, act):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride, padding=dilation, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(BasicBlock):
    """A ResNet-18 two-convolution block with an explicit shortcut.

    With matching channels and stride 1 the shortcut is an identity.  This is
    important for the width-matched LDW variants: no extra 1x1 projection is
    inserted before the late downsample.
    """

    expansion = 1

    def __init__(
        self,
        c1: int,
        c2: int,
        stride: int = 1,
        act: str | nn.Module | None = "relu",
        dilation: int = 1,
    ):
        nn.Module.__init__(self)
        if min(c1, c2, stride, dilation) <= 0:
            raise ValueError("channels, stride and dilation must be positive")
        self.stride = stride
        self.dilation = dilation
        self.shortcut = c1 == c2 and stride == 1
        if dilation == 1:
            # Reuse the project's exact ResNet-18 block for the common case,
            # while forcing identity shortcuts for width-matched stride-1 blocks.
            super().__init__(c1, c2, stride, self.shortcut, act=act, variant="d")
        else:
            self.branch2a = _ResidualConv(c1, c2, stride, dilation, act)
            self.branch2b = _ResidualConv(c2, c2, 1, dilation, None)
            self.short = nn.Identity() if self.shortcut else DownsampleProjection(c1, c2, stride)
            self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = x if self.shortcut else self.short(x)
        return self.act(self.branch2b(self.branch2a(x)) + short)


class _StageDownsample(nn.Module):
    """The single explicit stride-2 convolution at a stage boundary."""

    def __init__(self, c1: int, c2: int, dilation: int = 1, act: str | nn.Module | None = "relu"):
        super().__init__()
        if min(c1, c2, dilation) <= 0:
            raise ValueError("channels and dilation must be positive")
        self.stride = 2
        self.conv = nn.Conv2d(c1, c2, 3, 2, padding=dilation, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class LateDownStage(nn.Module):
    """Compute residual features at full resolution, then downsample once."""

    def __init__(
        self,
        c1: int,
        c2: int,
        hidden_channels: int,
        num_blocks: int = 2,
        stage_num: int | None = None,
        dilation: int = 1,
        act: str | nn.Module | None = "relu",
    ):
        super().__init__()
        if min(c1, c2, hidden_channels, num_blocks, dilation) <= 0:
            raise ValueError("channels, num_blocks and dilation must be positive")
        self.c1, self.c2 = c1, c2
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.stage_num = stage_num
        self.dilation = dilation
        self.input_projection = (
            DownsampleProjection(c1, hidden_channels) if c1 != hidden_channels else nn.Identity()
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_channels, hidden_channels, 1, act=act, dilation=dilation)
            for _ in range(num_blocks)
        ])
        self.late_downsample = _StageDownsample(hidden_channels, c2, dilation=dilation, act=act)

    @property
    def residual_blocks(self):
        """Alias used by diagnostics without registering duplicate modules."""
        return self.blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.blocks(x)
        return self.late_downsample(x)


class WidthMatchedEarlyDownStage(nn.Module):
    """Downsample first, compute with matched width, then project to stage width."""

    def __init__(
        self,
        c1: int,
        c2: int,
        hidden_channels: int,
        num_blocks: int = 2,
        stage_num: int | None = None,
        dilation: int = 1,
        act: str | nn.Module | None = "relu",
    ):
        super().__init__()
        if min(c1, c2, hidden_channels, num_blocks, dilation) <= 0:
            raise ValueError("channels, num_blocks and dilation must be positive")
        self.c1, self.c2 = c1, c2
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.stage_num = stage_num
        self.dilation = dilation
        self.early_downsample = _StageDownsample(c1, hidden_channels, dilation=dilation, act=act)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_channels, hidden_channels, 1, act=act, dilation=dilation)
            for _ in range(num_blocks)
        ])
        self.output_projection = (
            DownsampleProjection(hidden_channels, c2) if hidden_channels != c2 else nn.Identity()
        )

    @property
    def residual_blocks(self):
        return self.blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.early_downsample(x)
        x = self.blocks(x)
        return self.output_projection(x)
