"""Opt-in CSP reconstruction stages for the original RT-DETR R18 backbone."""
import math

import torch
from torch import nn

from .block import BasicBlock, ConvNormLayer


class ContextBlock(nn.Module):
    """Depthwise large-kernel context with a twofold pointwise expansion."""

    def __init__(self, channels, kernel_size=7, expansion=2):
        super().__init__()
        if kernel_size not in (3, 5, 7):
            raise ValueError('kernel_size must be 3, 5, or 7')
        if channels < 1 or expansion < 1:
            raise ValueError('channels and expansion must be positive')
        self.dwconv = nn.Conv2d(channels, channels, kernel_size,
                                padding=kernel_size // 2, groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.expand = nn.Conv2d(channels, channels * expansion, 1)
        self.act = nn.GELU()
        self.project = nn.Conv2d(channels * expansion, channels, 1)

    def forward(self, x):
        return x + self.project(self.act(self.expand(self.norm(self.dwconv(x)))))


class _CSPStage(nn.Module):
    """Shared split/fusion plumbing; subclasses define the processed branch."""

    def __init__(self, c1, c2, stage_num, process_ratio):
        super().__init__()
        if not 0 < process_ratio < 1:
            raise ValueError('process_ratio must be strictly between 0 and 1')
        if c2 < 2 or stage_num not in (3, 4):
            raise ValueError('CSP stages require c2 >= 2 and stage 3 or 4')
        self.process_channels = min(c2 - 1, max(1, round(c2 * process_ratio)))
        self.bypass_channels = c2 - self.process_channels
        # Exactly the first BasicBlock used by original Blocks(stage_num).
        self.downsample = BasicBlock(c1, c2, stride=2, shortcut=False,
                                     act='relu', variant='d')
        self.split_projection = ConvNormLayer(c2, c2, 1, 1, act='relu')
        self.fusion = ConvNormLayer(c2, c2, 1, 1, act=None)
        self.gamma_max = 0.3
        self.raw_gamma = nn.Parameter(torch.tensor(math.atanh(0.05 / self.gamma_max)))
        self.last_features = None

    @property
    def gamma(self):
        return self.gamma_max * torch.tanh(self.raw_gamma)

    def forward(self, x):
        z = self.downsample(x)
        projected = self.split_projection(z)
        processed, bypass = torch.split(
            projected, (self.process_channels, self.bypass_channels), dim=1)
        processed = self.process(processed)
        fusion = self.fusion(torch.cat((processed, bypass), dim=1))
        result = z + self.gamma * fusion
        if not self.training:
            self.last_features = {name: value.detach() for name, value in
                                  (('downsample', z), ('processed', processed),
                                   ('bypass', bypass), ('fusion', fusion),
                                   ('output', result))}
        return result


class CSPResidualStage(_CSPStage):
    """Original Stage3 downsampling plus partial-channel residual computation."""

    def __init__(self, c1, c2=128, num_partial_blocks=2, stage_num=3,
                 process_ratio=0.75):
        if stage_num != 3 or num_partial_blocks < 1:
            raise ValueError('CSPResidualStage requires stage 3 and positive block count')
        super().__init__(c1, c2, stage_num, process_ratio)
        self.process = nn.Sequential(*(
            BasicBlock(self.process_channels, self.process_channels,
                       stride=1, shortcut=True, act='relu', variant='d')
            for _ in range(num_partial_blocks)))


class CSPContextStage(_CSPStage):
    """Original Stage4 downsampling plus partial-channel context modeling."""

    def __init__(self, c1, c2=256, num_context_blocks=2, stage_num=4,
                 process_ratio=0.75, kernel_size=7):
        if stage_num != 4 or num_context_blocks < 1:
            raise ValueError('CSPContextStage requires stage 4 and positive block count')
        super().__init__(c1, c2, stage_num, process_ratio)
        self.process = nn.Sequential(*(
            ContextBlock(self.process_channels, kernel_size=kernel_size)
            for _ in range(num_context_blocks)))
