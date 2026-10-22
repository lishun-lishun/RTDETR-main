"""Sparse small-object side path for the RT-DETR-R18 DUT experiments.

SCRE never replaces the ordinary backbone features. SPLF is placed *after*
CCFM and changes only its final P3 output. Runtime tensors kept in ``last``
are consumed and cleared by RTDETRDetectionModel.loss before EMA/checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

__all__ = ("SCRE", "SPLF", "SCREFixedFusion")


def _bounded_residual(feature, projection):
    """Normalize per spatial location; exact zeros stay zero."""
    delta = projection(feature)
    rms = (delta.float().square().mean(1, keepdim=True) + 1e-6).sqrt()
    return torch.tanh(delta / rms.to(delta.dtype))


class SCRE(nn.Module):
    """P3 top-K candidate selection, P2 local encoding, and sparse P3 scatter."""

    def __init__(self, c_p2, c_p3, hidden_channels=64, topk=32, patch_size=5, use_offset=True):
        super().__init__()
        if patch_size < 1 or patch_size % 2 != 1 or topk < 1:
            raise ValueError("SCRE requires an odd positive patch_size and positive topk")
        self.hidden_channels = int(hidden_channels)
        self.c_p2 = int(c_p2)
        self.topk = int(topk)
        self.patch_size = int(patch_size)
        self.use_offset = bool(use_offset)
        self.center_dw = nn.Conv2d(c_p3, c_p3, 3, padding=1, groups=c_p3, bias=False)
        self.center_head = nn.Conv2d(c_p3, 1, 1)
        nn.init.constant_(self.center_head.bias, -4.6)
        self.p2_proj = nn.Conv2d(c_p2, self.hidden_channels, 1, bias=False)
        self.local_dw = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3,
                                  padding=1, groups=self.hidden_channels, bias=False)
        self.local_pw = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1)
        self.offset_head = nn.Linear(self.hidden_channels, 2) if self.use_offset else None
        if self.offset_head is not None:
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)
        self.last = None

    def _sample_p2(self, p2, centers_x, centers_y):
        b, _, h2, w2 = p2.shape
        k = centers_x.shape[1]
        radius = self.patch_size // 2
        step = torch.arange(-radius, radius + 1, device=p2.device, dtype=p2.dtype)
        dy, dx = torch.meshgrid(step, step, indexing="ij")
        x = (centers_x[:, :, None, None] + dx[None, None]).expand(b, k, self.patch_size, self.patch_size)
        y = (centers_y[:, :, None, None] + dy[None, None]).expand(b, k, self.patch_size, self.patch_size)
        grid = torch.stack((2 * x / w2 - 1, 2 * y / h2 - 1), dim=-1)
        grid = grid.reshape(b, k * self.patch_size, self.patch_size, 2)
        patches = F.grid_sample(p2, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return patches.reshape(b, self.c_p2, k, self.patch_size, self.patch_size)

    @staticmethod
    def _scatter(descriptors, indices, offsets, h, w):
        """Normalized bilinear scatter; values and offset weights retain gradients."""
        b, k, c = descriptors.shape
        x = (indices % w).to(descriptors.dtype) + offsets[..., 0]
        y = torch.div(indices, w, rounding_mode="floor").to(descriptors.dtype) + offsets[..., 1]
        x0, y0 = torch.floor(x), torch.floor(y)
        wx, wy = x - x0, y - y0
        feature = descriptors.new_zeros((b, c, h * w))
        norm = descriptors.new_zeros((b, 1, h * w))
        for ix, iy, weight in (
            (x0, y0, (1 - wx) * (1 - wy)),
            (x0 + 1, y0, wx * (1 - wy)),
            (x0, y0 + 1, (1 - wx) * wy),
            (x0 + 1, y0 + 1, wx * wy),
        ):
            valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            flat = (iy.clamp(0, h - 1).long() * w + ix.clamp(0, w - 1).long())
            weight = (weight * valid.to(weight.dtype)).unsqueeze(1)
            feature = feature.scatter_add(2, flat.unsqueeze(1).expand(-1, c, -1),
                                          descriptors.transpose(1, 2) * weight)
            norm = norm.scatter_add(2, flat.unsqueeze(1), weight)
        # Only normalize collisions. Dividing by a subunit single-candidate
        # weight would cancel the bilinear weight and its offset gradient.
        return (feature / norm.clamp_min(1.0)).reshape(b, c, h, w)

    def forward(self, inputs):
        p2, p3 = inputs
        if p2.shape[0] != p3.shape[0]:
            raise ValueError("P2 and P3 batch sizes must match")
        b, _, h3, w3 = p3.shape
        with record_function("SCRE_center_prediction"):
            center_logits = self.center_head(F.silu(self.center_dw(p3)))
        with record_function("SCRE_topk"):
            scores, indices = center_logits.flatten(2).topk(min(self.topk, h3 * w3), dim=-1)
        scores, indices = scores.squeeze(1), indices.squeeze(1)
        # With align_corners=False, pixel centers occupy continuous coordinates
        # 0.5, 1.5, ...; the P3 cell center maps to the P2 center via ratio.
        x2 = ((indices % w3).to(p2.dtype) + 0.5) * (p2.shape[-1] / w3)
        y2 = (torch.div(indices, w3, rounding_mode="floor").to(p2.dtype) + 0.5) * (p2.shape[-2] / h3)
        with record_function("SCRE_grid_sample"):
            patches = self._sample_p2(p2, x2, y2)
        k = indices.shape[1]
        patches = patches.permute(0, 2, 1, 3, 4).reshape(b * k, self.c_p2,
                                                           self.patch_size, self.patch_size)
        patches = self.p2_proj(patches)
        descriptors = F.silu(self.local_pw(F.silu(self.local_dw(patches))))
        descriptors = descriptors.mean((-2, -1)).reshape(b, k, self.hidden_channels)
        offsets = (self.offset_head(descriptors).tanh() if self.offset_head is not None
                   else descriptors.new_zeros((b, k, 2)))
        gated = descriptors * scores.sigmoid().unsqueeze(-1)
        with record_function("SCRE_scatter"):
            sparse = self._scatter(gated, indices, offsets, h3, w3)
        if self.training or not torch.is_grad_enabled():
            self.last = {"logits": center_logits, "indices": indices, "offsets": offsets}
        else:
            self.last = None
        return torch.cat((sparse, center_logits), dim=1)


class SPLF(nn.Module):
    """Bounded, initially exact-identity compensation of final CCFM P3."""

    def __init__(self, c_f3, c_aux=None, hidden_channels=64, topk=32, alpha_max=0.1):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.topk = int(topk)
        self.alpha_max = float(alpha_max)
        self.alpha_raw = nn.Parameter(torch.zeros(()))
        self.project = nn.Conv2d(self.hidden_channels, c_f3, 1, bias=False)
        self.standalone = c_aux is None
        if not self.standalone and c_aux != self.hidden_channels + 1:
            raise ValueError("SPLF expects SCRE channels [M3, center_logits]")
        if self.standalone:
            self.feature_head = nn.Conv2d(c_f3, self.hidden_channels, 1, bias=False)
            self.center_head = nn.Conv2d(c_f3, 1, 3, padding=1)
            nn.init.constant_(self.center_head.bias, -4.6)
        self.last = None

    def _features(self, inputs):
        if self.standalone:
            f3 = inputs if isinstance(inputs, torch.Tensor) else inputs[0]
            m3 = F.silu(self.feature_head(f3))
            with record_function("SPLF_center_prediction"):
                logits = self.center_head(f3)
            b, _, h, w = logits.shape
            indices = logits.flatten(2).topk(min(self.topk, h * w), dim=-1).indices
            sparse_mask = torch.zeros_like(logits).flatten(2).scatter(2, indices, 1).reshape_as(logits)
            return f3, m3 * sparse_mask, logits
        f3, auxiliary = inputs
        return f3, auxiliary[:, :self.hidden_channels], auxiliary[:, self.hidden_channels:]

    def forward(self, inputs):
        f3, m3, logits = self._features(inputs)
        if m3.shape[-2:] != f3.shape[-2:]:
            raise ValueError("SCRE and final CCFM P3 spatial sizes must match")
        with record_function("SPLF_projection"):
            delta = _bounded_residual(m3, self.project)
        enhanced = f3 + (self.alpha_max * self.alpha_raw.tanh()) * logits.sigmoid() * delta
        if self.training or not torch.is_grad_enabled():
            self.last = {"base": f3, "enhanced": enhanced,
                         "logits": logits if self.standalone else None}
        else:
            self.last = None
        return enhanced


class SCREFixedFusion(SPLF):
    """E1: SCRE with a fixed post-CCFM residual gain, without learned SPLF gain."""

    def __init__(self, c_f3, c_aux, hidden_channels=64, fixed_gain=0.05):
        super().__init__(c_f3, c_aux, hidden_channels=hidden_channels, alpha_max=0.0)
        del self.alpha_raw
        self.fixed_gain = float(fixed_gain)

    def forward(self, inputs):
        f3, m3, logits = self._features(inputs)
        with record_function("SCRE_fixed_projection"):
            delta = _bounded_residual(m3, self.project)
        enhanced = f3 + self.fixed_gain * logits.sigmoid() * delta
        self.last = ({"base": f3, "enhanced": enhanced}
                     if self.training or not torch.is_grad_enabled() else None)
        return enhanced
