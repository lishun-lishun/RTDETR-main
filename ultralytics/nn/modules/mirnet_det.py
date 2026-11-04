"""MIRNet-Det multi-resolution backbone modules.

The implementation keeps the RT-DETR neck and decoder untouched.  The
backbone returns three tensors at strides 8, 16 and 32; ``FeatureSelect``
provides explicit graph edges from that multi-output tensor to the original
head.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from .block import ConvNormLayer, get_activation

__all__ = (
    "MIRStem",
    "SpatialRearrangeDownsample",
    "StrideDownsample",
    "EfficientInvertedBlock",
    "DualResolutionExchange",
    "TriResolutionExchange",
    "MIRDetBackbone",
    "FeatureSelect",
)


class MIRStem(nn.Module):
    """Two 3x3 stride-2 convolutions producing a stride-4 feature map."""

    def __init__(self, in_channels: int = 3, out_channels: int = 48, act="silu"):
        super().__init__()
        hidden = 32 if out_channels != 32 else out_channels
        self.conv1 = ConvNormLayer(in_channels, hidden, 3, 2, act=act)
        self.conv2 = ConvNormLayer(hidden, out_channels, 3, 2, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class SpatialRearrangeDownsample(nn.Module):
    """Pixel-unshuffle downsample followed by 1x1 channel mixing."""

    def __init__(self, in_channels: int, out_channels: int, act="silu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downscale_factor = 2
        self.proj = ConvNormLayer(in_channels * 4, out_channels, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(f"SpatialRearrangeDownsample expected {self.in_channels} channels, got {x.shape[1]}")
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("SpatialRearrangeDownsample requires even spatial dimensions")
        return self.proj(F.pixel_unshuffle(x, self.downscale_factor))


class StrideDownsample(nn.Module):
    """Conventional 3x3 stride-2 control used by the E5 ablation."""

    def __init__(self, in_channels: int, out_channels: int, act="silu"):
        super().__init__()
        self.conv = ConvNormLayer(in_channels, out_channels, 3, 2, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EfficientInvertedBlock(nn.Module):
    """Small inverted residual block with a depthwise spatial convolution."""

    def __init__(self, channels: int, expansion_ratio: float = 2, kernel_size: int = 3):
        super().__init__()
        if channels <= 0 or expansion_ratio <= 0 or kernel_size % 2 == 0:
            raise ValueError("channels and expansion_ratio must be positive; kernel_size must be odd")
        hidden = max(int(round(channels * expansion_ratio)), channels)
        padding = kernel_size // 2
        self.channels = channels
        self.hidden_channels = hidden
        self.expand = ConvNormLayer(channels, hidden, 1, 1, act="silu")
        self.depthwise = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(hidden, hidden, kernel_size, 1, padding, groups=hidden, bias=False)),
            ("norm", nn.BatchNorm2d(hidden)),
            ("act", get_activation("silu")),
        ]))
        self.project = ConvNormLayer(hidden, channels, 1, 1, act=None)
        self.use_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.project(self.depthwise(self.expand(x)))
        return x + value if self.use_residual else value


class _ScaleProject(nn.Module):
    """Projection used by an exchange direction."""

    def __init__(self, c1: int, c2: int, stride: int = 1):
        super().__init__()
        self.conv = ConvNormLayer(c1, c2, 3 if stride == 2 else 1, stride, act=None)

    def forward(self, x: torch.Tensor, size=None) -> torch.Tensor:
        x = self.conv(x)
        return F.interpolate(x, size=size, mode="nearest") if size is not None and x.shape[-2:] != size else x


class DualResolutionExchange(nn.Module):
    """Synchronous bidirectional exchange between S8 and S16 branches."""

    def __init__(self, c8: int = 128, c16: int = 224, fusion_strength: float = 0.1,
                 enable_fusion: bool = True, expansion_ratio: float = 2):
        super().__init__()
        self.c8, self.c16 = c8, c16
        self.enable_fusion = enable_fusion
        if enable_fusion:
            self.s16_to_s8 = _ScaleProject(c16, c8, stride=1)
            self.s8_to_s16 = _ScaleProject(c8, c16, stride=2)
            self.alpha_8 = nn.Parameter(torch.tensor(float(fusion_strength)))
            self.alpha_16 = nn.Parameter(torch.tensor(float(fusion_strength)))
            if fusion_strength == 0:
                self.alpha_8.requires_grad_(False)
                self.alpha_16.requires_grad_(False)
        self.block8 = EfficientInvertedBlock(c8, expansion_ratio)
        self.block16 = EfficientInvertedBlock(c16, expansion_ratio)
        self.last_residuals = {}

    def forward(self, s8: torch.Tensor, s16: torch.Tensor):
        old_s8, old_s16 = s8, s16
        if self.enable_fusion:
            from16 = self.s16_to_s8(old_s16, size=old_s8.shape[-2:])
            from8 = self.s8_to_s16(old_s8, size=old_s16.shape[-2:])
            new_s8 = old_s8 + self.alpha_8 * from16
            new_s16 = old_s16 + self.alpha_16 * from8
            self.last_residuals = {"s16_to_s8": from16.detach(), "s8_to_s16": from8.detach()}
        else:
            new_s8, new_s16 = old_s8, old_s16
            self.last_residuals = {}
        # Both branches are updated from old_s8/old_s16, never sequentially.
        return self.block8(new_s8), self.block16(new_s16)


class TriResolutionExchange(nn.Module):
    """Synchronous nearest-neighbour exchange between adjacent S8/S16/S32."""

    def __init__(self, c8: int = 128, c16: int = 224, c32: int = 384,
                 fusion_strength: float = 0.1, enable_fusion: bool = True,
                 expansion_ratio: float = 2):
        super().__init__()
        self.c8, self.c16, self.c32 = c8, c16, c32
        self.enable_fusion = enable_fusion
        if enable_fusion:
            self.s16_to_s8 = _ScaleProject(c16, c8, 1)
            self.s8_to_s16 = _ScaleProject(c8, c16, 2)
            self.s32_to_s16 = _ScaleProject(c32, c16, 1)
            self.s16_to_s32 = _ScaleProject(c16, c32, 2)
            self.alpha_8 = nn.Parameter(torch.tensor(float(fusion_strength)))
            self.alpha_16_from8 = nn.Parameter(torch.tensor(float(fusion_strength)))
            self.alpha_16_from32 = nn.Parameter(torch.tensor(float(fusion_strength)))
            self.alpha_32 = nn.Parameter(torch.tensor(float(fusion_strength)))
            if fusion_strength == 0:
                for alpha in (self.alpha_8, self.alpha_16_from8, self.alpha_16_from32, self.alpha_32):
                    alpha.requires_grad_(False)
        self.block8 = EfficientInvertedBlock(c8, expansion_ratio)
        self.block16 = EfficientInvertedBlock(c16, expansion_ratio)
        self.block32 = EfficientInvertedBlock(c32, expansion_ratio)
        self.last_residuals = {}

    def forward(self, s8: torch.Tensor, s16: torch.Tensor, s32: torch.Tensor):
        old_s8, old_s16, old_s32 = s8, s16, s32
        if self.enable_fusion:
            from16 = self.s16_to_s8(old_s16, size=old_s8.shape[-2:])
            from8 = self.s8_to_s16(old_s8, size=old_s16.shape[-2:])
            from32 = self.s32_to_s16(old_s32, size=old_s16.shape[-2:])
            from16_down = self.s16_to_s32(old_s16, size=old_s32.shape[-2:])
            new_s8 = old_s8 + self.alpha_8 * from16
            new_s16 = old_s16 + self.alpha_16_from8 * from8 + self.alpha_16_from32 * from32
            new_s32 = old_s32 + self.alpha_32 * from16_down
            self.last_residuals = {
                "s16_to_s8": from16.detach(), "s8_to_s16": from8.detach(),
                "s32_to_s16": from32.detach(), "s16_to_s32": from16_down.detach(),
            }
        else:
            new_s8, new_s16, new_s32 = old_s8, old_s16, old_s32
            self.last_residuals = {}
        return self.block8(new_s8), self.block16(new_s16), self.block32(new_s32)


class FeatureSelect(nn.Module):
    """Explicitly select P3/P4/P5 from a multi-output backbone."""

    def __init__(self, index: int):
        super().__init__()
        if index not in (0, 1, 2):
            raise ValueError("FeatureSelect index must be 0, 1 or 2")
        self.index = index

    def forward(self, features):
        if not isinstance(features, (tuple, list)) or len(features) != 3:
            raise ValueError("FeatureSelect expects three backbone outputs")
        return features[self.index]


class MIRDetBackbone(nn.Module):
    """MIRNet-Det backbone returning P3/P4/P5 at strides 8/16/32."""

    def __init__(
        self,
        stem_channels: int = 48,
        s8_channels: int = 128,
        s16_channels: int = 224,
        s32_channels: int = 384,
        dual_repeats: int = 2,
        tri_repeats: int = 2,
        expansion_ratio: float = 2,
        use_pixel_unshuffle: bool = True,
        enable_dual_exchange: bool = True,
        enable_tri_exchange: bool = True,
        fusion_strength: float = 0.1,
        enable_exchange_fusion: bool = True,
        in_channels: int = 3,
        output_p3_channels: int = 128,
        output_p4_channels: int = 256,
        output_p5_channels: int = 512,
    ):
        super().__init__()
        if min(stem_channels, s8_channels, s16_channels, s32_channels, dual_repeats, tri_repeats) <= 0:
            raise ValueError("MIRNet channels and repeat counts must be positive")
        self.s8_channels, self.s16_channels, self.s32_channels = s8_channels, s16_channels, s32_channels
        self.fusion_strength = fusion_strength
        self.enable_exchange_fusion = enable_exchange_fusion
        self.stem = MIRStem(in_channels, stem_channels)
        self.use_pixel_unshuffle = use_pixel_unshuffle
        self.spatial_downsample = (SpatialRearrangeDownsample(stem_channels, s8_channels)
                                   if use_pixel_unshuffle else StrideDownsample(stem_channels, s8_channels))
        self.s8_blocks = nn.Sequential(*[
            EfficientInvertedBlock(s8_channels, expansion_ratio) for _ in range(2)
        ])
        self.s16_generate = StrideDownsample(s8_channels, s16_channels)
        self.s32_generate = StrideDownsample(s16_channels, s32_channels)
        self.dual_exchange = nn.ModuleList([
            DualResolutionExchange(s8_channels, s16_channels, fusion_strength=fusion_strength,
                                   enable_fusion=enable_exchange_fusion, expansion_ratio=expansion_ratio)
            for _ in range(dual_repeats)
        ]) if enable_dual_exchange else nn.ModuleList()
        self.tri_exchange = nn.ModuleList([
            TriResolutionExchange(s8_channels, s16_channels, s32_channels,
                                  fusion_strength=fusion_strength,
                                  enable_fusion=enable_exchange_fusion,
                                  expansion_ratio=expansion_ratio)
            for _ in range(tri_repeats)
        ]) if enable_tri_exchange else nn.ModuleList()
        self.s16_blocks = nn.Sequential(*[
            EfficientInvertedBlock(s16_channels, expansion_ratio) for _ in range(2)
        ]) if not enable_dual_exchange else nn.Identity()
        self.s32_blocks = nn.Sequential(*[
            EfficientInvertedBlock(s32_channels, expansion_ratio) for _ in range(2)
        ]) if not enable_tri_exchange else nn.Identity()
        self.p3 = nn.Identity() if s8_channels == output_p3_channels else ConvNormLayer(s8_channels, output_p3_channels, 1, 1, act=None)
        self.p4 = ConvNormLayer(s16_channels, output_p4_channels, 1, 1, act=None)
        self.p5 = ConvNormLayer(s32_channels, output_p5_channels, 1, 1, act=None)
        self.last_states = None

    def forward(self, x: torch.Tensor):
        stem = self.stem(x)
        s8 = self.s8_blocks(self.spatial_downsample(stem))
        s16 = self.s16_generate(s8)
        if self.dual_exchange:
            for exchange in self.dual_exchange:
                s8, s16 = exchange(s8, s16)
        else:
            s16 = self.s16_blocks(s16)
        s32 = self.s32_generate(s16)
        if self.tri_exchange:
            for exchange in self.tri_exchange:
                s8, s16, s32 = exchange(s8, s16, s32)
        else:
            s32 = self.s32_blocks(s32)
        self.last_states = (s8, s16, s32)
        return [self.p3(s8), self.p4(s16), self.p5(s32)]
