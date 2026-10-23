"""Learned spatial sampling for the stride-16 to stride-8 path."""
import torch
from torch import nn
from torch.nn import functional as F


class DynamicUpsample(nn.Module):
    """Bilinear sampling with offsets in source-feature pixel units."""

    def __init__(self, channels=256, max_offset=0.5):
        super().__init__()
        if max_offset <= 0:
            raise ValueError('max_offset must be positive')
        self.max_offset = float(max_offset)
        self.offset_predictor = nn.Conv2d(channels, 2, 1)
        nn.init.zeros_(self.offset_predictor.weight)
        nn.init.zeros_(self.offset_predictor.bias)
        self.last_offset = None

    def forward(self, source, target_size=None):
        if isinstance(source, (tuple, list)):
            if len(source) != 2:
                raise ValueError('LAF-0 expects [source, target]')
            source, target = source
            target_size = target.shape[-2:]
        if target_size is None:
            raise ValueError('target_size is required for a single source tensor')
        h, w = source.shape[-2:]
        th, tw = target_size
        raw = F.interpolate(self.offset_predictor(source).float(),
                            size=target_size, mode='bilinear', align_corners=False)
        offset = self.max_offset * torch.tanh(raw)
        self.last_offset = offset.detach()
        yy, xx = torch.meshgrid(
            torch.arange(th, device=source.device, dtype=torch.float32),
            torch.arange(tw, device=source.device, dtype=torch.float32),
            indexing='ij')
        # align_corners=False: destination pixel centres map to source centres.
        x = (xx + 0.5) * w / tw - 0.5 + offset[:, 0]
        y = (yy + 0.5) * h / th - 0.5 + offset[:, 1]
        grid = torch.stack(((x + 0.5) * 2 / w - 1,
                            (y + 0.5) * 2 / h - 1), dim=-1)
        result = F.grid_sample(source.float(), grid, mode='bilinear',
                               padding_mode='border', align_corners=False)
        return result.to(source.dtype)


class IsolatedLAF(nn.Module):
    def __init__(self, channels=256, max_offset=0.5, alpha_max=0.3):
        super().__init__()
        if alpha_max <= 0:
            raise ValueError('alpha_max must be positive')
        self.sampler = DynamicUpsample(channels, max_offset)
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1))
        self.alpha_raw = nn.Parameter(torch.zeros(()))
        self.alpha_max = float(alpha_max)
        self.last_delta = None

    def forward(self, f3, f4=None):
        if f4 is None:
            if not isinstance(f3, (tuple, list)) or len(f3) != 2:
                raise ValueError('LAF-1 expects [F3, F4]')
            f3, f4 = f3
        aligned = self.sampler(f4, f3.shape[-2:])
        delta = self.fusion(torch.cat((f3, aligned), dim=1))
        self.last_delta = delta.detach()
        alpha = self.alpha_max * torch.tanh(self.alpha_raw)
        return f3 + alpha * delta
