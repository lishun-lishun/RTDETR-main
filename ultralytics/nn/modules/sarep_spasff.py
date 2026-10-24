"""Isolated SA-Rep and scale-protected ASFF experiments for RT-DETR-R18."""

import torch
from torch import nn
from torch.nn import functional as F

from .block import BasicBlock, Blocks


class SARepConv(nn.Module):
    """Three Conv+BN branches with an exact eval-time 3x3 equivalent."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super().__init__()
        if in_channels != out_channels or stride != 1:
            raise ValueError('SARepConv v1 requires equal channels and stride=1')
        if groups < 1 or in_channels % groups or out_channels % groups:
            raise ValueError('groups must divide both channel counts')
        self.in_channels, self.out_channels, self.groups = in_channels, out_channels, groups
        if deploy:
            self.reparam = nn.Conv2d(in_channels, out_channels, 3, 1, 1,
                                     groups=groups, bias=True)
        else:
            self.branch_3x3 = self._branch((3, 3), (1, 1))
            self.branch_1x3 = self._branch((1, 3), (0, 1))
            self.branch_3x1 = self._branch((3, 1), (1, 0))

    def _branch(self, kernel, padding):
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel, stride=1,
                      padding=padding, groups=self.groups, bias=False),
            nn.BatchNorm2d(self.out_channels))

    def forward(self, x):
        if hasattr(self, 'reparam'):
            return self.reparam(x)
        return (self.branch_3x3(x) + self.branch_1x3(x) + self.branch_3x1(x))

    @staticmethod
    def _fuse(branch):
        conv, bn = branch
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weight = conv.weight * scale.reshape(-1, 1, 1, 1)
        conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        bias = bn.bias + (conv_bias - bn.running_mean) * scale
        return weight, bias

    def equivalent_kernel_bias(self):
        if hasattr(self, 'reparam'):
            return self.reparam.weight, self.reparam.bias
        k3, b3 = self._fuse(self.branch_3x3)
        k13, b13 = self._fuse(self.branch_1x3)
        k31, b31 = self._fuse(self.branch_3x1)
        return k3 + F.pad(k13, (0, 0, 1, 1)) + F.pad(k31, (1, 1, 0, 0)), b3 + b13 + b31

    @torch.no_grad()
    def switch_to_deploy(self):
        if hasattr(self, 'reparam'):
            return self
        kernel, bias = self.equivalent_kernel_bias()
        conv = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1,
                         groups=self.groups, bias=True).to(device=kernel.device, dtype=kernel.dtype)
        conv.weight.copy_(kernel)
        conv.bias.copy_(bias)
        self.reparam = conv
        del self.branch_3x3, self.branch_1x3, self.branch_3x1
        return self


class SARepBlocks(Blocks):
    """Build the original Blocks stage and replace only selected branch2b modules."""

    def __init__(self, ch_in, ch_out, count, stage_num, act='relu', replace_indices=(1,)):
        super().__init__(ch_in, ch_out, BasicBlock, count, stage_num, act=act)
        for index in replace_indices:
            if index < 0 or index >= count:
                raise ValueError(f'invalid SA-Rep block index {index}')
            block = self.blocks[index]
            original = block.branch2b
            if original.conv.stride != (1, 1) or original.conv.in_channels != ch_out:
                raise ValueError('selected BasicBlock Conv2 is incompatible with SARepConv')
            block.branch2b = SARepConv(ch_out, ch_out)


class SPASFF(nn.Module):
    """Align final F3/F4/F5, fuse per pixel, and modify only decoder F3."""

    def __init__(self, c3, c4, c5, hidden_channels=64, mode='protected', gamma_max=0.3):
        super().__init__()
        if mode not in {'protected', 'no_difference', 'direct'}:
            raise ValueError(f'unknown SPASFF mode: {mode}')
        self.mode, self.gamma_max = mode, gamma_max
        self.project3 = nn.Conv2d(c3, hidden_channels, 1)
        self.project4 = nn.Conv2d(c4, hidden_channels, 1)
        self.project5 = nn.Conv2d(c5, hidden_channels, 1)
        self.weights = nn.Sequential(nn.Conv2d(3 * hidden_channels, hidden_channels, 1),
                                     nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, 3, 1))
        self.projection = nn.Conv2d(hidden_channels, c3, 1)
        self.raw_gamma = nn.Parameter(torch.zeros(()))
        self.last_weights = None
        self.last_residual = None

    def forward(self, inputs):
        f3, f4, f5 = inputs
        size = f3.shape[-2:]
        x3 = self.project3(f3)
        x4 = F.interpolate(self.project4(f4), size=size, mode='nearest')
        x5 = F.interpolate(self.project5(f5), size=size, mode='nearest')
        weight = torch.softmax(self.weights(torch.cat((x3, x4, x5), dim=1)), dim=1)
        fused = weight[:, 0:1] * x3 + weight[:, 1:2] * x4 + weight[:, 2:3] * x5
        if self.mode == 'direct':
            output = self.projection(fused)
            residual = output - f3
        else:
            difference = fused - x3 if self.mode == 'protected' else fused
            residual = self.gamma_max * torch.tanh(self.raw_gamma) * self.projection(difference)
            output = f3 + residual
        if not self.training:
            self.last_weights = weight.detach()
            self.last_residual = residual.detach()
        return output


def switch_sarep_to_deploy(model):
    """Convert all SARepConv layers of an eval model in place."""
    model.eval()
    converted = 0
    for module in model.modules():
        if isinstance(module, SARepConv):
            if not hasattr(module, 'reparam'):
                module.switch_to_deploy()
                converted += 1
    return converted
