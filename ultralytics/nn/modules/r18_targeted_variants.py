"""Targeted R18 backbone variants for DUT-Anti-UAV (R09--R16).

The classes in this file are intentionally small, local operators.  They do
not alter the RT-DETR neck, decoder or loss.  Stage wrappers keep the original
R18 feature contract (P3/P4/P5) while making the changed operation explicit in
the module graph.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .block import BasicBlock, Blocks

__all__ = (
    "LIPDown", "PhaseDown", "ABSR", "LCR", "CoordRefine", "SKResidual",
    "SemanticDetailFeedback", "SFR", "LIPStage3Blocks", "PhaseStage3Blocks",
    "ABSRStage3Blocks", "LCRStage3Blocks", "CoordRefineStage3Blocks",
    "SKStage4Blocks", "SDFStage4Blocks", "SFRStage3Blocks",
    "AdaptiveBackgroundSmoothingResidual", "LocalContrastResidual",
    "CoordinateRefine", "SparseForegroundResidual",
    "UniformPool", "UniformPhaseDown",
)


class LIPDown(nn.Module):
    """Learned local importance pooling with a fixed stride-two reduction."""

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 2,
                 uniform: bool = False):
        super().__init__()
        if kernel_size != 3 or stride != 2:
            raise ValueError("LIPDown is defined for a 3x3 stride-2 operator")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.uniform = bool(uniform)
        self.logit = nn.Conv2d(channels, 9, 3, 1, 1, bias=True)
        self.logits = self.logit
        self.weight_predictor = self.logit
        nn.init.zeros_(self.logit.weight)
        nn.init.zeros_(self.logit.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=3, padding=1, stride=2)
        oh = (h + 1) // 2
        ow = (w + 1) // 2
        patches = patches.view(b, c, 9, oh, ow)
        if self.uniform:
            weights = patches.new_full((b, 1, 9, oh, ow), 1.0 / 9.0)
        else:
            logits = self.logit(x)[..., ::2, ::2][..., :oh, :ow]
            weights = torch.softmax(logits.float(), dim=1).to(dtype=x.dtype).unsqueeze(1)
        return (patches * weights).sum(dim=2)


class PhaseDown(nn.Module):
    """Phase-aware stride-two downsampling based on PixelUnshuffle."""

    def __init__(self, channels: int, uniform: bool = False):
        super().__init__()
        self.channels = int(channels)
        self.uniform = bool(uniform)
        self.weight = nn.Conv2d(channels, 4, 1, 1, 0, bias=True)
        self.phase_weight = self.weight
        self.weight_predictor = self.weight
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
        phases = F.pixel_unshuffle(x, 2).view(x.shape[0], 4, self.channels,
                                                x.shape[-2] // 2,
                                                x.shape[-1] // 2)
        if self.uniform:
            weights = phases.new_full((x.shape[0], 4, 1, phases.shape[-2],
                                       phases.shape[-1]), 0.25)
        else:
            weights = torch.softmax(self.weight(x).float(), dim=1).to(dtype=x.dtype).unsqueeze(2)
        return (phases * weights).sum(dim=1)


class UniformPool(LIPDown):
    """Fixed-uniform counterpart used by the LIP ablation control."""

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__(channels, kernel_size=kernel_size, stride=stride, uniform=True)


class UniformPhaseDown(PhaseDown):
    """Fixed-uniform counterpart used by the phase ablation control."""

    def __init__(self, channels: int):
        super().__init__(channels, uniform=True)


class ABSR(nn.Module):
    """Adaptive background smoothing residual for Stage3/P3."""

    def __init__(self, channels: int, gamma_init: float = 0.01):
        super().__init__()
        self.channels = int(channels)
        self.smooth = nn.AvgPool2d(5, 1, 2, count_include_pad=False)
        self.saliency_predictor = nn.Conv2d(channels, 1, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = self.smooth(x)
        protect = torch.sigmoid(self.saliency_predictor(x))
        residual = (1.0 - protect) * (smooth - x)
        self._last_protect = protect.detach()
        self._last_smooth_residual = residual.detach()
        return x + self.gamma * residual

    @property
    def diagnostics(self):
        p = getattr(self, "_last_protect", None)
        r = getattr(self, "_last_smooth_residual", None)
        return {
            "protect_mean": None if p is None else float(p.mean()),
            "protect_std": None if p is None else float(p.std(unbiased=False)),
            "smooth_residual_rms": None if r is None else float(r.square().mean().sqrt()),
            "target_mean": None if p is None else float(p.mean()),
            "background_mean": None if p is None else float((1.0 - p).mean()),
        }


class LCR(nn.Module):
    """Local contrast residual enhancement for Stage3/P3."""

    def __init__(self, channels: int, gamma_init: float = 0.01):
        super().__init__()
        self.channels = int(channels)
        self.local_mean = nn.AvgPool2d(5, 1, 2, count_include_pad=False)
        self.mask = nn.Conv2d(channels, 1, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contrast = x - self.local_mean(x)
        mask = torch.sigmoid(self.mask(contrast))
        self._last_contrast = contrast.detach()
        self._last_mask = mask.detach()
        return x + self.gamma * mask * contrast

    @property
    def diagnostics(self):
        c = getattr(self, "_last_contrast", None)
        return {"local_contrast_rms": None if c is None else float(c.square().mean().sqrt())}


class CoordRefine(nn.Module):
    """Coordinate Attention residual refinement, applied once at P3."""

    def __init__(self, channels: int, reduction: int = 32, gamma_init: float = 0.01):
        super().__init__()
        self.channels = int(channels)
        hidden = max(8, channels // int(reduction))
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        h_pool = x.mean(dim=3, keepdim=True)
        w_pool = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = torch.cat((h_pool, w_pool), dim=2)
        y = self.act(self.norm(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)
        refined = x * torch.sigmoid(self.conv_h(y_h)) * torch.sigmoid(self.conv_w(y_w))
        return x + self.gamma * (refined - x)


class SKResidual(nn.Module):
    """Selective-kernel residual branch used by the second Stage4 block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = int(channels)
        self.branch3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.branch5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False)
        self.conv3 = self.branch3
        self.conv5 = self.branch5
        hidden = max(8, channels // int(reduction))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(hidden, channels * 2, 1, bias=True))
        self.last_branch_weights: Optional[torch.Tensor] = None

    @property
    def branch_weights(self):
        """Last detached ``[B,2,C,1,1]`` softmax weights for diagnostics."""
        return self.last_branch_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch3(x)
        b = self.branch5(x)
        u = a + b
        logits = self.fc(self.pool(u)).view(x.shape[0], 2, self.channels, 1, 1)
        weights = torch.softmax(logits.float(), dim=1).to(dtype=x.dtype)
        self.last_branch_weights = weights.detach()
        return a * weights[:, 0] + b * weights[:, 1]


class SemanticDetailFeedback(nn.Module):
    """Feed P4 semantic context back into P3 without changing P4 itself."""

    def __init__(self, p3_channels: int = 128, p4_channels: int = 256,
                 gamma_init: float = 0.01):
        super().__init__()
        self.p3_channels = int(p3_channels)
        self.p4_channels = int(p4_channels)
        self.project_p4 = nn.Conv2d(p4_channels, p3_channels, 1, bias=False)
        self.gate = nn.Conv2d(p3_channels * 2, 1, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def refine(self, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        semantic = self.project_p4(p4)
        semantic = F.interpolate(semantic, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.gate(torch.cat((p3, semantic), dim=1)))
        self._last_gate = gate.detach()
        return p3 + self.gamma * gate * semantic

    def forward(self, p3: torch.Tensor, p4: Optional[torch.Tensor] = None):
        # Model graphs pass multi-input sources as a single Python list.  Keep
        # the two-argument form convenient for direct unit tests as well.
        if p4 is None:
            if not isinstance(p3, (tuple, list)) or len(p3) != 2:
                raise TypeError("SemanticDetailFeedback expects (P3, P4)")
            p3, p4 = p3
        return self.refine(p3, p4)


class SFR(nn.Module):
    """Sparse foreground residual with soft saliency (no hard top-k)."""

    def __init__(self, channels: int, gamma_init: float = 0.01):
        super().__init__()
        self.channels = int(channels)
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.saliency = nn.Conv2d(channels, 1, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.pw(self.dw(x))
        saliency = torch.sigmoid(self.saliency(x))
        self._last_saliency = saliency.detach()
        return x + self.gamma * saliency * residual

    @property
    def diagnostics(self):
        s = getattr(self, "_last_saliency", None)
        return {"saliency_mean": None if s is None else float(s.mean()),
                "saliency_std": None if s is None else float(s.std(unbiased=False))}


def _blocks_no_downsample(ch_in, ch_out, block, count, act, variant):
    """Construct a regular R18 stage whose first block has stride one."""
    return Blocks(ch_in, ch_out, block, count, 2, act=act, variant=variant)


class LIPStage3Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, block=BasicBlock, count=2, stage_num=3,
                 act="relu", variant="d", uniform=False):
        super().__init__()
        # Keep the compact ablation YAML form [..., act, true] compatible with
        # the full [..., act, variant, uniform] form.
        if isinstance(variant, bool) and uniform is False:
            uniform, variant = variant, "d"
        self.down = LIPDown(ch_in, uniform=bool(uniform))
        self.blocks = _blocks_no_downsample(ch_in, ch_out, block, count, act, variant)

    def forward(self, x):
        return self.blocks(self.down(x))


class PhaseStage3Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, block=BasicBlock, count=2, stage_num=3,
                 act="relu", variant="d", uniform=False):
        super().__init__()
        if isinstance(variant, bool) and uniform is False:
            uniform, variant = variant, "d"
        self.down = PhaseDown(ch_in, uniform=bool(uniform))
        self.blocks = _blocks_no_downsample(ch_in, ch_out, block, count, act, variant)

    def forward(self, x):
        return self.blocks(self.down(x))


class _PostStage3(nn.Module):
    post_cls = nn.Identity

    def __init__(self, ch_in, ch_out, block=BasicBlock, count=2, stage_num=3,
                 act="relu", variant="d", gamma_init=0.01):
        super().__init__()
        self.blocks = Blocks(ch_in, ch_out, block, count, stage_num, act=act, variant=variant)
        self.post = self.post_cls(ch_out, gamma_init=gamma_init)

    def forward(self, x):
        return self.post(self.blocks(x))


class ABSRStage3Blocks(_PostStage3):
    post_cls = ABSR


class LCRStage3Blocks(_PostStage3):
    post_cls = LCR


class CoordRefineStage3Blocks(_PostStage3):
    post_cls = CoordRefine


class SFRStage3Blocks(_PostStage3):
    post_cls = SFR


class SKStage4Blocks(nn.Module):
    """Stage4 with SK residual replacing only the second block's branch2b."""

    def __init__(self, ch_in, ch_out, block=BasicBlock, count=2, stage_num=4,
                 act="relu", variant="d"):
        super().__init__()
        if count != 2:
            raise ValueError("SKStage4Blocks is fixed to the two-block R18 Stage4")
        self.blocks = nn.ModuleList([
            block(ch_in, ch_out, 2, False, act=act, variant=variant),
            block(ch_out, ch_out, 1, True, act=act, variant=variant),
        ])
        self.sk = SKResidual(ch_out)
        # branch2b is replaced in place.  The SK operator itself owns only the
        # two depthwise branches and the softmax selector; no extra residual
        # block, BN or activation is inserted after the fusion.
        self.blocks[1].branch2b = self.sk

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class SDFStage4Blocks(nn.Module):
    """Stage4 plus P4-to-P3 semantic feedback, returning ``(P3, P4)``."""

    def __init__(self, ch_in, ch_out, block=BasicBlock, count=2, stage_num=4,
                 act="relu", variant="d", gamma_init=0.01):
        super().__init__()
        self.blocks = Blocks(ch_in, ch_out, block, count, stage_num, act=act, variant=variant)
        self.sdf = SemanticDetailFeedback(ch_in, ch_out, gamma_init=gamma_init)

    def forward(self, p3):
        p4 = self.blocks(p3)
        refined = self.sdf(p3, p4)
        return refined, p4


# Descriptive aliases make the standalone operators easy to discover without
# changing the concise YAML-facing names used by the experiment registry.
AdaptiveBackgroundSmoothingResidual = ABSR
LocalContrastResidual = LCR
CoordinateRefine = CoordRefine
SparseForegroundResidual = SFR

