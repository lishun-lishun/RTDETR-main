"""Res2Net-inspired hierarchical convolution adapted to the R18 BasicBlock."""
import torch
from torch import nn

from .block import BasicBlock, Blocks, ConvNormLayer


class Res2Conv(nn.Module):
    """One local BasicBlock branch with hierarchical split convolutions."""

    def __init__(self, in_channels, out_channels, stride=1, scale=4, width=None,
                 hierarchical=True, act='relu'):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError('Res2Conv supports scale=2 or scale=4')
        width = out_channels // scale if width is None else width
        if width < 1 or width * scale > 4 * out_channels:
            raise ValueError('invalid group width')
        self.scale, self.width, self.hierarchical = scale, width, hierarchical
        self.pre = ConvNormLayer(in_channels, scale * width, 1, stride, act=act)
        self.groups = nn.ModuleList(
            ConvNormLayer(width, width, 3, 1, act=act) for _ in range(scale - 1))
        self.post = ConvNormLayer(scale * width, out_channels, 1, 1, act=act)
        self.last_group_outputs = None
        self.last_hierarchical_inputs = None

    def forward(self, x):
        chunks = self.pre(x).split(self.width, dim=1)
        outputs = [chunks[0]]
        hierarchical_inputs = []
        for i, conv in enumerate(self.groups, start=1):
            value = chunks[i] + outputs[-1] if self.hierarchical and i > 1 else chunks[i]
            hierarchical_inputs.append(value)
            outputs.append(conv(value))
        if not self.training:
            self.last_group_outputs = [part.detach() for part in outputs]
            self.last_hierarchical_inputs = [part.detach() for part in hierarchical_inputs]
        return self.post(torch.cat(outputs, dim=1))


class Res2BasicBlock(BasicBlock):
    """Preserve BasicBlock Conv2, shortcut, add and final activation."""

    def __init__(self, in_channels, out_channels, stride=1, shortcut=True,
                 act='relu', variant='d', scale=4, width=None, hierarchical=True):
        super().__init__(in_channels, out_channels, stride, shortcut, act, variant)
        self.branch2a = Res2Conv(in_channels, out_channels, stride, scale, width,
                                 hierarchical, act)


class Res2Blocks(Blocks):
    """Keep the original stage layout while selecting individual BasicBlocks."""

    def __init__(self, ch_in, ch_out, count, stage_num, act='relu',
                 replace_indices=(1,), scale=4, width=None, hierarchical=True):
        super().__init__(ch_in, ch_out, BasicBlock, count, stage_num, act=act)
        for index in replace_indices:
            if index < 0 or index >= count:
                raise ValueError(f'invalid Res2BasicBlock index {index}')
            original = self.blocks[index]
            block_in = ch_in if index == 0 else ch_out
            stride = 2 if index == 0 and stage_num != 2 else 1
            replacement = Res2BasicBlock(block_in, ch_out, stride,
                                         shortcut=original.shortcut, act=act,
                                         scale=scale, width=width,
                                         hierarchical=hierarchical)
            # The original second convolution and shortcut are structurally
            # unchanged. New runs initialize from scratch, so no weights copy.
            self.blocks[index] = replacement
