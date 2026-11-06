"""Local, mutually-exclusive RT-DETR-R18 backbone variants.

Each class in this file changes only the requested R18 backbone operation.  The
original ``BasicBlock`` and ``Blocks`` implementations remain untouched so
that historical models continue to load exactly as before.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from .block import BasicBlock, Blocks, ConvNormLayer, get_activation

try:  # torchvision is optional until the R07 model is instantiated.
    from torchvision.ops import DeformConv2d
except Exception:  # pragma: no cover - depends on the local torch/vision build
    DeformConv2d = None

__all__ = (
    "StemConvDown", "BlurPool", "AAStemPool", "ResNetDDownsample",
    "ResNetDBasicBlock", "ResNetDBlocks", "DilatedConvNorm", "DilatedBasicBlock",
    "P4DilatedBlocks", "DCNConvNorm", "DeformableBasicBlock", "P4DeformableBlocks",
    "PartialConv3", "PConvBasicBlock", "P5PConvBlocks",
)


class StemConvDown(nn.Module):
    """R01: learnable 3x3 stride-2 replacement for the R18 stem pool."""

    def __init__(self, ch_in: int, ch_out: int = 64, kernel_size: int = 3,
                 stride: int = 2, padding: int = 1):
        super().__init__()
        if ch_in != ch_out:
            raise ValueError(f"StemConvDown requires equal input/output channels, got {ch_in}->{ch_out}")
        if kernel_size != 3 or stride != 2 or padding != 1:
            raise ValueError("StemConvDown is fixed to Conv3x3(stride=2,padding=1)")
        self.conv = nn.Conv2d(ch_in, ch_out, 3, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation("relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class BlurPool(nn.Module):
    """Fixed depthwise anti-aliased downsampling filter."""

    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        if stride != 2:
            raise ValueError("BlurPool in the R02 experiment uses stride=2")
        kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
        self.register_buffer("kernel", kernel[None, None], persistent=True)
        self.channels = int(channels)
        self.stride = int(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(dtype=x.dtype, device=x.device).expand(self.channels, 1, 3, 3)
        return F.conv2d(x, kernel, stride=self.stride, padding=1, groups=self.channels)


class AAStemPool(nn.Module):
    """R02: stride-1 max pooling followed by fixed BlurPool stride 2."""

    def __init__(self, ch_in: int, ch_out: int = 64):
        super().__init__()
        if ch_in != ch_out:
            raise ValueError(f"AAStemPool requires equal input/output channels, got {ch_in}->{ch_out}")
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.blurpool = BlurPool(ch_in, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blurpool(self.maxpool(x))


class ResNetDDownsample(nn.Module):
    """Average-pool then 1x1 projection used by the R03 shortcuts."""

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.pool = nn.AvgPool2d(2, 2, 0, ceil_mode=True)
        self.conv = ConvNormLayer(ch_in, ch_out, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class ResNetDBasicBlock(nn.Module):
    """BasicBlock with an explicit ResNet-D downsampling shortcut."""

    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__()
        self.shortcut = shortcut
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        if shortcut:
            self.short = nn.Identity()
        elif variant == "d" and stride == 2:
            self.short = ResNetDDownsample(ch_in, ch_out)
        else:
            self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch2b(self.branch2a(x))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class ResNetDBlocks(Blocks):
    """R03 stage wrapper; only this variant opts into explicit D shortcuts."""

    def __init__(self, ch_in, ch_out, block, count, stage_num, act="relu", variant="d"):
        super().__init__(ch_in, ch_out, ResNetDBasicBlock, count, stage_num, act=act, variant="d")


class DilatedConvNorm(nn.Module):
    """ConvNormLayer-equivalent block with an explicit dilation value."""

    def __init__(self, ch_in: int, ch_out: int, dilation: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, padding=dilation, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DilatedBasicBlock(BasicBlock):
    """R06 BasicBlock whose second 3x3 convolution uses dilation 2."""

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__(ch_in, ch_out, stride, shortcut, act=act, variant=variant)
        self.branch2b = DilatedConvNorm(ch_out, ch_out, dilation=2)


class P4DilatedBlocks(nn.Module):
    """R06 Stage4 with dilation applied only to its second block."""

    def __init__(self, ch_in, ch_out, block, count, stage_num, act="relu", variant="d"):
        if count != 2 or stage_num != 4:
            raise ValueError("P4DilatedBlocks is fixed to the two-block Stage4 R18 experiment")
        super().__init__()
        self.blocks = nn.ModuleList([
            BasicBlock(ch_in, ch_out, 2, False, act=act, variant=variant),
            DilatedBasicBlock(ch_out, ch_out, 1, True, act=act, variant=variant),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class DCNConvNorm(nn.Module):
    """R07 offset-predicting DeformConv2d replacement for one 3x3 branch."""

    def __init__(self, channels: int):
        super().__init__()
        if DeformConv2d is None:
            raise RuntimeError(
                "R07 requires torchvision.ops.DeformConv2d; install a torchvision build "
                "matching the active PyTorch/CUDA environment instead of silently falling back.")
        self.offset = nn.Conv2d(channels, 18, 3, 1, 1, bias=True)
        self.deform = DeformConv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.deform(x, self.offset(x)))


class DeformableBasicBlock(BasicBlock):
    """R07 BasicBlock with DCNv1-style offset sampling in branch2b only."""

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__(ch_in, ch_out, stride, shortcut, act=act, variant=variant)
        if ch_in != ch_out:
            raise ValueError("DeformableBasicBlock is intended for the non-downsampling Stage4 block")
        self.branch2b = DCNConvNorm(ch_out)


class P4DeformableBlocks(nn.Module):
    """R07 Stage4 with deformable sampling only in its second block."""

    def __init__(self, ch_in, ch_out, block, count, stage_num, act="relu", variant="d"):
        if count != 2 or stage_num != 4:
            raise ValueError("P4DeformableBlocks is fixed to the two-block Stage4 R18 experiment")
        super().__init__()
        self.blocks = nn.ModuleList([
            BasicBlock(ch_in, ch_out, 2, False, act=act, variant=variant),
            DeformableBasicBlock(ch_out, ch_out, 1, True, act=act, variant=variant),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class PartialConv3(nn.Module):
    """R08 partial 3x3 convolution followed by a 1x1 channel mixer."""

    def __init__(self, ch_in: int, ch_out: int, ratio: float = 0.25, act="relu"):
        super().__init__()
        if not 0.0 < ratio < 1.0:
            raise ValueError("PartialConv3 ratio must be between 0 and 1")
        self.partial_channels = max(1, int(round(ch_in * ratio)))
        self.identity_channels = ch_in - self.partial_channels
        self.partial = nn.Conv2d(self.partial_channels, self.partial_channels, 3, 1, 1, bias=False)
        self.merge = nn.Conv2d(ch_in, ch_out, 1, 1, bias=False)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active, identity = torch.split(x, (self.partial_channels, self.identity_channels), dim=1)
        return self.act(self.norm(self.merge(torch.cat((self.partial(active), identity), dim=1))))


class PConvBasicBlock(BasicBlock):
    """R08 BasicBlock with PConv replacing only branch2a."""

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d", ratio=0.25):
        super().__init__(ch_in, ch_out, stride, shortcut, act=act, variant=variant)
        if stride != 1 or ch_in != ch_out:
            raise ValueError("PConvBasicBlock is intended for the second non-downsampling Stage5 block")
        self.branch2a = PartialConv3(ch_in, ch_out, ratio=ratio, act=act)


class P5PConvBlocks(nn.Module):
    """R08 Stage5 with PConv only in its second block."""

    def __init__(self, ch_in, ch_out, block, count, stage_num, act="relu", variant="d"):
        if count != 2 or stage_num != 5:
            raise ValueError("P5PConvBlocks is fixed to the two-block Stage5 R18 experiment")
        super().__init__()
        self.blocks = nn.ModuleList([
            BasicBlock(ch_in, ch_out, 2, False, act=act, variant=variant),
            PConvBasicBlock(ch_out, ch_out, 1, True, act=act, variant=variant, ratio=0.25),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
