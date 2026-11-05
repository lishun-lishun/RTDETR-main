"""Official DLA-34 backbone adapted for the local RT-DETR graph.

The implementation follows the reference DLA code from UCB Drive:
https://github.com/ucbdrive/dla/blob/master/dla.py

Only the feature extractor is retained.  The classification average-pool and
fully-connected layers are intentionally absent because RT-DETR consumes the
three spatial feature maps at levels 3, 4 and 5.

The reference DLA source is BSD 3-Clause licensed (Copyright 2018 Fisher Yu).
This file retains the copyright attribution and the BSD conditions in the
repository's accompanying implementation report.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

__all__ = ("DLABasicBlock", "DLARoot", "DLATree", "DLA34Backbone")


class DLABasicBlock(nn.Module):
    """The two 3x3 convolution block used by official DLA-34."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 dilation: int = 1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=padding, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is None:
            residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class DLARoot(nn.Module):
    """Concatenate sibling Tree outputs and aggregate them with a 1x1 root."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 residual: bool):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = bool(residual)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        if not tensors:
            raise ValueError("DLARoot requires at least one tensor")
        out = self.conv(torch.cat(tensors, dim=1))
        out = self.bn(out)
        if self.residual:
            out = out + tensors[0]
        return self.relu(out)


class DLATree(nn.Module):
    """Recursive DLA aggregation tree, kept structurally equivalent to UCB DLA."""

    def __init__(self, levels: int, block: type[nn.Module], in_channels: int,
                 out_channels: int, stride: int = 1, level_root: bool = False,
                 root_dim: int = 0, root_kernel_size: int = 1,
                 dilation: int = 1, root_residual: bool = False):
        super().__init__()
        if levels < 1:
            raise ValueError("DLATree levels must be >= 1")
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels

        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
            self.root = DLARoot(root_dim, out_channels, root_kernel_size,
                                root_residual)
        else:
            self.tree1 = DLATree(
                levels - 1, block, in_channels, out_channels, stride,
                root_dim=0, root_kernel_size=root_kernel_size,
                dilation=dilation, root_residual=root_residual)
            self.tree2 = DLATree(
                levels - 1, block, out_channels, out_channels, 1,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size, dilation=dilation,
                root_residual=root_residual)

        self.level_root = bool(level_root)
        self.root_dim = root_dim
        self.levels = levels
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else None
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None,
                children: list[torch.Tensor] | None = None) -> torch.Tensor:
        # ``children`` is deliberately shared down the recursive tree.  This
        # is the level_root path used by official DLA and is not a Sequential
        # approximation of the aggregation.
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample is not None else x
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            return self.root(x2, x1, *children)
        children.append(x1)
        return self.tree2(x1, children=children)


class DLA34Backbone(nn.Module):
    """Standard DLA-34 feature extractor returning RT-DETR P3/P4/P5.

    The official DLA-34 stem is stride 1.  Consequently the six levels have
    strides ``1, 2, 4, 8, 16, 32`` and channels ``16, 32, 64, 128, 256, 512``.
    ``pretrained`` is opt-in and never downloads weights; provide a local
    checkpoint through ``pretrained_path`` when using it.
    """

# Reference license: BSD 3-Clause, Copyright (c) 2018 Fisher Yu.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the copyright notice, conditions,
# and disclaimer are retained.  Neither the name of Fisher Yu nor contributors
# may be used to endorse products without prior written permission.

    levels = (1, 1, 1, 2, 2, 1)
    channels = (16, 32, 64, 128, 256, 512)

    def __init__(self, pretrained: bool | str | Path = False, in_channels: int = 3,
                 pretrained_path: str | Path | None = None,
                 residual_root: bool = False):
        super().__init__()
        if isinstance(pretrained, (str, Path)):
            if pretrained_path is not None:
                raise ValueError("set either pretrained or pretrained_path, not both")
            pretrained_path = pretrained
            pretrained = True
        if not isinstance(pretrained, bool):
            raise TypeError("pretrained must be bool or a local checkpoint path")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        self.out_channels = (128, 256, 512)
        self.out_strides = (8, 16, 32)
        self.pretrained_load_report = None
        self.base_layer = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=7,
                      stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(self.channels[0], self.channels[0], self.levels[0])
        self.level1 = self._make_conv_level(self.channels[0], self.channels[1], self.levels[1], stride=2)
        self.level2 = DLATree(self.levels[2], DLABasicBlock, self.channels[1], self.channels[2], 2,
                              level_root=False, root_residual=residual_root)
        self.level3 = DLATree(self.levels[3], DLABasicBlock, self.channels[2], self.channels[3], 2,
                              level_root=True, root_residual=residual_root)
        self.level4 = DLATree(self.levels[4], DLABasicBlock, self.channels[3], self.channels[4], 2,
                              level_root=True, root_residual=residual_root)
        self.level5 = DLATree(self.levels[5], DLABasicBlock, self.channels[4], self.channels[5], 2,
                              level_root=True, root_residual=residual_root)
        self._initialize_weights()
        if pretrained:
            if pretrained_path is None:
                raise ValueError("pretrained=True requires a local pretrained_path; automatic downloads are disabled")
            self.load_pretrained(pretrained_path)

    @staticmethod
    def _make_conv_level(in_channels: int, out_channels: int, convs: int,
                         stride: int = 1, dilation: int = 1) -> nn.Sequential:
        modules: list[nn.Module] = []
        for index in range(convs):
            modules.extend((
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=stride if index == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
            in_channels = out_channels
        return nn.Sequential(*modules)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def load_pretrained(self, path: str | Path):
        """Load a local official DLA checkpoint and return a transparent report."""
        checkpoint = torch.load(str(path), map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise TypeError("DLA checkpoint must contain a state-dict mapping")
        normalized = {}
        ignored_keys = []
        stripped_module_keys = []
        for key, value in state.items():
            if key.startswith("module."):
                stripped_module_keys.append(key)
                key = key[7:]
            if key.startswith("fc.") or key.startswith("avgpool."):
                ignored_keys.append(key)
                continue
            normalized[key] = value
        incompatible = self.load_state_dict(normalized, strict=False)
        loaded = len(normalized) - len(incompatible.unexpected_keys)
        loaded_keys = sorted(set(normalized) - set(incompatible.unexpected_keys))
        loaded_modules = sorted({key.rsplit('.', 1)[0] for key in loaded_keys if '.' in key})
        self.pretrained_load_report = {
            "path": str(Path(path).resolve()),
            "provided_keys": len(state),
            "backbone_keys": len(normalized),
            "loaded_keys": loaded,
            "load_ratio": loaded / max(len(normalized), 1),
            "key_mapping": {"module.": ""},
            "module_prefix_keys": len(stripped_module_keys),
            "ignored_classification_keys": ignored_keys,
            "loaded_modules": loaded_modules,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        return self.pretrained_load_report

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.base_layer(x)
        levels = []
        for index in range(6):
            x = getattr(self, f"level{index}")(x)
            levels.append(x)
        return levels[3], levels[4], levels[5]

