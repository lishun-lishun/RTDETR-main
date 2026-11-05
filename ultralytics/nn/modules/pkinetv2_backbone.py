"""Pure-PyTorch PKINet-v2-T backbone.

This is a detection-only port of the official PKINet-v2 implementation.  It
keeps the five-branch PKS module and the published T configuration while
removing MMRotate/MMCV registry dependencies.
"""

from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn

__all__ = ("PKSModule", "PKSBlock", "PKINetV2", "PKINetV2_T", "create_pkinetv2_t")


def _bn(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor()


def _fuse_bn_tensor(conv, bn):
    kernel = conv.weight if conv is not None else torch.zeros(
        bn.num_features, 1, 1, 1, device=bn.weight.device)
    mean, var, gamma, beta, eps = bn.running_mean, bn.running_var, bn.weight, bn.bias, bn.eps
    std = (var + eps).sqrt()
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * scale, beta - mean * gamma / std


class PKSModule(nn.Module):
    """Official PKINet-v2 Poly-Kernel Scope module with five branches."""

    def __init__(self, dim: int, deploy: bool = False, branch_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.deploy = bool(deploy)
        self.branch_scale = float(branch_scale)
        self.max_kernel = 19
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        if self.deploy:
            self.fused_parallel_conv = nn.Conv2d(dim, dim, 19, padding=9, groups=dim)
        else:
            self.branch1_axial = nn.Sequential(
                nn.Conv2d(dim, dim, (1, 19), padding=(0, 9), groups=dim, bias=False),
                nn.Conv2d(dim, dim, (19, 1), padding=(9, 0), groups=dim, bias=False), _bn(dim))
            self.branch2_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, 7, padding=9, dilation=3, groups=dim, bias=False), _bn(dim))
            self.branch3_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, 5, padding=6, dilation=3, groups=dim, bias=False), _bn(dim))
            self.branch4_sparse = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=3, dilation=3, groups=dim, bias=False), _bn(dim))
            self.branch5_dense = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False), _bn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return x * self.conv1(self.fused_parallel_conv(self.conv0(x)))
        feature = self.conv0(x)
        attention = self.branch1_axial(feature)
        attention = attention + self.branch2_sparse(feature)
        attention = attention + self.branch3_sparse(feature)
        attention = attention + self.branch4_sparse(feature)
        attention = attention + self.branch5_dense(feature)
        return x * self.conv1(attention * self.branch_scale)

    def switch_to_deploy(self):
        if self.deploy:
            return
        device = self.conv0.weight.device
        fused_kernel = torch.zeros(self.dim, 1, 19, 19, device=device)
        fused_bias = torch.zeros(self.dim, device=device)
        # The axial branch has two depthwise convolutions.  The reference
        # implementation's deployment path uses their outer product.
        k_first = self.branch1_axial[0].weight
        k_second, b_second = _fuse_bn_tensor(self.branch1_axial[1], self.branch1_axial[2])
        k_axial = torch.matmul(k_second, k_first)
        fused_kernel += k_axial
        fused_bias += b_second

        def add_sparse(branch, size, dilation):
            kernel, bias = _fuse_bn_tensor(branch[0], branch[1])
            center = 9
            small = size // 2
            for row in range(size):
                for col in range(size):
                    fused_kernel[:, :, center + (row - small) * dilation,
                                 center + (col - small) * dilation] += kernel[:, :, row, col]
            return bias

        fused_bias += add_sparse(self.branch2_sparse, 7, 3)
        fused_bias += add_sparse(self.branch3_sparse, 5, 3)
        fused_bias += add_sparse(self.branch4_sparse, 3, 3)
        fused_bias += add_sparse(self.branch5_dense, 3, 1)
        fused_kernel *= self.branch_scale
        fused_bias *= self.branch_scale
        self.fused_parallel_conv = nn.Conv2d(self.dim, self.dim, 19, padding=9, groups=self.dim, bias=True)
        self.fused_parallel_conv.weight.data.copy_(fused_kernel)
        self.fused_parallel_conv.bias.data.copy_(fused_bias)
        del self.branch1_axial, self.branch2_sparse, self.branch3_sparse, self.branch4_sparse, self.branch5_dense
        self.deploy = True


class PKSBlock(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, branch_scale: float = 1.0):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = PKSModule(dim, branch_scale=branch_scale)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return shortcut + x

    def switch_to_deploy(self):
        self.spatial_gating_unit.switch_to_deploy()


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.dwconv(self.fc1(x))))
        return self.drop2(self.fc2(x))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size=3, stride=2, norm_cfg=True):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride, patch_size // 2)
        self.norm = _bn(embed_dim) if norm_cfg else nn.Identity()

    def forward(self, x):
        x = self.norm(self.proj(x))
        return x, x.shape[-2], x.shape[-1]


class PKINetV2Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0, branch_scale=1.0):
        super().__init__()
        self.norm1 = _bn(dim)
        self.norm2 = _bn(dim)
        self.attn = PKSBlock(dim, drop_path=drop_path, branch_scale=branch_scale)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1[:, None, None] * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2[:, None, None] * self.mlp(self.norm2(x)))
        return x

    def switch_to_deploy(self):
        self.attn.switch_to_deploy()


class PKINetV2(nn.Module):
    def __init__(self, embed_dims=(32, 64, 160, 256), depths=(3, 3, 5, 2),
                 mlp_ratios=(8, 8, 4, 4), drop_rate=0.1, drop_path_rate=0.1,
                 branch_scale=1.0):
        super().__init__()
        self.embed_dims = tuple(embed_dims)
        self.depths = tuple(depths)
        self.num_stages = 4
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cursor = 0
        for index in range(self.num_stages):
            patch = OverlapPatchEmbed(
                3 if index == 0 else embed_dims[index - 1], embed_dims[index],
                patch_size=7 if index == 0 else 3, stride=4 if index == 0 else 2)
            blocks = nn.ModuleList([
                PKINetV2Block(embed_dims[index], mlp_ratios[index], drop_rate,
                              dpr[cursor + block], branch_scale)
                for block in range(depths[index])
            ])
            setattr(self, f"patch_embed{index + 1}", patch)
            setattr(self, f"block{index + 1}", blocks)
            setattr(self, f"norm{index + 1}", nn.LayerNorm(embed_dims[index], eps=1e-6))
            cursor += depths[index]
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels // module.groups
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        batch = x.shape[0]
        outputs = []
        for index in range(self.num_stages):
            x, height, width = getattr(self, f"patch_embed{index + 1}")(x)
            for block in getattr(self, f"block{index + 1}"):
                x = block(x)
            # LayerNorm is applied on flattened tokens, but the next stage's
            # patch embedding is a Conv2d and therefore must receive NCHW.
            # Keep the stage output in NCHW after normalisation instead of
            # leaking the intermediate [B, HW, C] token layout downstream.
            x = getattr(self, f"norm{index + 1}")(x.flatten(2).transpose(1, 2))
            x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs

    def switch_to_deploy(self):
        for module in self.modules():
            if module is not self and hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()


class PKINetV2_T(PKINetV2):
    def __init__(self, **kwargs):
        config = dict(embed_dims=(32, 64, 160, 256), depths=(3, 3, 5, 2),
                      mlp_ratios=(8, 8, 4, 4), drop_rate=0.1, drop_path_rate=0.1)
        config.update(kwargs)
        super().__init__(**config)


def create_pkinetv2_t():
    return PKINetV2_T()

