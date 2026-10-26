"""Lightweight three-scale Gather-and-Distribute neck for RT-DETR-R18."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def conv_bn_act(c1, c2, kernel=1, groups=1):
    return nn.Sequential(nn.Conv2d(c1, c2, kernel, padding=kernel // 2,
                                   groups=groups, bias=False),
                         nn.BatchNorm2d(c2), nn.SiLU(inplace=True))


class GDGather(nn.Module):
    """Align three distinct scales and interact with pointwise-depthwise fusion."""

    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(conv_bn_act(3 * channels, channels),
                                  conv_bn_act(channels, channels, 3, channels),
                                  conv_bn_act(channels, channels))

    def forward(self, features, size):
        aligned = [F.adaptive_avg_pool2d(x, size) if x.shape[-2] >= size[0]
                   and x.shape[-1] >= size[1] else
                   F.interpolate(x, size=size, mode='nearest')
                   for x in features]
        return self.fuse(torch.cat(aligned, dim=1))


class GDInject(nn.Module):
    """Content-dependent context injection into a same-scale residual path."""

    def __init__(self, channels, hidden):
        super().__init__()
        self.context = conv_bn_act(hidden, hidden)
        self.local = conv_bn_act(channels, hidden)
        self.gate = nn.Conv2d(2 * hidden, hidden, 1)
        self.output = nn.Conv2d(hidden, channels, 1, bias=False)
        self.output_bn = nn.BatchNorm2d(channels)
        self.raw_gain = nn.Parameter(torch.tensor(math.atanh(0.05 / 0.3)))
        self.gain_max = 0.3
        self.last_residual = None

    def forward(self, local, shared):
        context = F.interpolate(shared, size=local.shape[-2:], mode='nearest')
        context = self.context(context)
        local_hidden = self.local(local)
        gate = torch.sigmoid(self.gate(torch.cat((local_hidden, context), dim=1)))
        correction = self.output_bn(self.output(gate * context + local_hidden))
        residual = self.gain_max * torch.tanh(self.raw_gain) * correction
        if not self.training:
            self.last_residual = residual.detach()
        return local + residual


class LiteGDNeck(nn.Module):
    """Replace CCFM; consume P3, P4 and projected+AIFI P5 and emit F3/F4/F5."""

    def __init__(self, c3, c4, c5, hidden_channels=128, output_channels=256,
                 mode='full', distribute=(True, True, True)):
        super().__init__()
        if mode not in {'full', 'high_only', 'low_only', 'local_only', 'p3_only'}:
            raise ValueError(f'unknown LiteGDNeck mode: {mode}')
        if len(distribute) != 3:
            raise ValueError('distribute must contain P3/P4/P5 switches')
        self.mode, self.distribute = mode, tuple(distribute)
        self.base = nn.ModuleList(conv_bn_act(c, output_channels) for c in (c3, c4, c5))
        self.local_refine = nn.ModuleList(
            conv_bn_act(output_channels, output_channels, 3, output_channels)
            for _ in range(3))
        self.to_hidden = nn.ModuleList(conv_bn_act(output_channels, hidden_channels)
                                       for _ in range(3))
        self.high_gather = GDGather(hidden_channels)
        self.high_inject4 = GDInject(output_channels, hidden_channels)
        self.high_inject5 = GDInject(output_channels, hidden_channels)
        self.low_gather = GDGather(hidden_channels)
        self.low_inject = nn.ModuleList(GDInject(output_channels, hidden_channels)
                                        for _ in range(3))
        self.last_high = None
        self.last_low = None

    def forward(self, inputs):
        p3, p4, p5 = [refine(project(x)) for x, project, refine in
                      zip(inputs, self.base, self.local_refine)]
        if self.mode in {'full', 'high_only'}:
            high = self.high_gather([self.to_hidden[0](p3),
                                     self.to_hidden[1](p4),
                                     self.to_hidden[2](p5)], p5.shape[-2:])
            p4 = self.high_inject4(p4, high) if self.distribute[1] else p4
            p5 = self.high_inject5(p5, high) if self.distribute[2] else p5
            if not self.training:
                self.last_high = high.detach()
        if self.mode in {'full', 'low_only', 'p3_only'}:
            low = self.low_gather([self.to_hidden[0](p3),
                                   self.to_hidden[1](p4),
                                   self.to_hidden[2](p5)], p4.shape[-2:])
            features = [p3, p4, p5]
            switches = self.distribute if self.mode != 'p3_only' else (self.distribute[0], False, False)
            features = [inject(local, low) if enabled else local
                        for local, inject, enabled in zip(features, self.low_inject, switches)]
            if not self.training:
                self.last_low = low.detach()
            return features
        return [p3, p4, p5]


class GDFeature(nn.Module):
    """Select one output of LiteGDNeck for the unchanged RTDETRDecoder API."""

    def __init__(self, index):
        super().__init__()
        if index not in (0, 1, 2):
            raise ValueError('GDFeature index must be 0, 1, or 2')
        self.index = index

    def forward(self, features):
        return features[self.index]
