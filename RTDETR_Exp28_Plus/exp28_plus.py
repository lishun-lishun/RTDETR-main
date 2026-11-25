# -*- coding: utf-8 -*-
"""
Exp28+ Zero-Inference-Overhead Upgrades (`RepStarFastBlock` + `ScaleAwareInnerNWDLoss`)
=======================================================================================
Strictly 0.00 ms inference latency increase, 0 extra deploy layers,
and 0 extra deploy parameters relative to `Exp28_StarFast_HiLoHaar`!

1. `RepStarFastBlock`:
   - Training: Adds a parallel grouped `1x1` conv + Identity BN alongside `sconv (3x3, groups=4)`
     to recover `Exp6`'s multi-branch optimization & high-IoU (`mAP75`) localization power.
   - Inference (`switch_to_deploy()`): Mathematically fuses `3x3 + 1x1 + Identity` into the
     EXACT SAME single `3x3` grouped convolution (0 extra layers, 0 extra params, 0.00 ms cost!).
2. `ScaleAwareInnerNWDLoss` (Innovation 3 — Training-Only Box Regression Loss):
   - Combines `Inner-GIoU` (`ratio=1.25`, accelerates high-IoU 0.75-0.95 boundary convergence)
     with scale-adaptive `Normalized Wasserstein Distance (NWD)` for tiny 6x6-16x16 UAV boxes.
   - Completely absent at inference time (0.00 ms overhead).
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .hrnet_lite_suite import (
        BaseHRNetV2W18,
        _HighResolutionModule,
        _fuse_conv_bn_pair,
        _fuse_conv_bn_tensors,
    )
except ImportError:
    from hrnet_lite_suite import (
        BaseHRNetV2W18,
        _HighResolutionModule,
        _fuse_conv_bn_pair,
        _fuse_conv_bn_tensors,
    )

__all__ = [
    "RepStarFastBlock",
    "HRNetRepStarFast",
    "inner_giou_nwd_loss",
    "patch_rtdetr_inner_nwd_loss",
]


class RepStarFastBlock(nn.Module):
    """
    Zero-Overhead Structurally Reparameterized `StarFastBlock`:
      - Training: `Conv3x3(g) + Conv1x1(g) + Identity BN` -> merged `1x1x2` Star Op -> `1x1` project
      - Deploy (`switch_to_deploy()`): Identical to `StarFastBlock` — ONLY 3 plain fused convs, 0 BNs!
    """

    def __init__(self, channels: int, groups: int = 4, expand: float = 1.25):
        super().__init__()
        self.channels = int(channels)
        g = groups if (self.channels % groups == 0) else 1
        self.groups = g
        hidden = max(self.channels, int(round(self.channels * expand)))

        # Multi-branch grouped spatial conv (fused into single `self.sconv` at deploy!)
        self.sconv = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=g, bias=False)
        self.sbn = nn.BatchNorm2d(self.channels)
        self.sconv1x1 = nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0, groups=g, bias=False)
        self.sbn1x1 = nn.BatchNorm2d(self.channels)
        self.sbn_id = nn.BatchNorm2d(self.channels)

        # Merged dual-output 1x1 conv for Star multiplication
        self.f12 = nn.Conv2d(self.channels, hidden * 2, kernel_size=1, bias=False)
        self.f12_bn = nn.BatchNorm2d(hidden * 2)

        self.g = nn.Conv2d(hidden, self.channels, kernel_size=1, bias=False)
        self.gn = nn.BatchNorm2d(self.channels)
        self.act6 = nn.ReLU6(inplace=False)
        self.relu = nn.ReLU(inplace=False)
        self.fused = False

        # Zero-Shift residual init
        nn.init.constant_(self.gn.weight, 0.0)
        nn.init.constant_(self.gn.bias, 0.0)

    def switch_to_deploy(self) -> None:
        if getattr(self, "fused", False):
            return
        with torch.no_grad():
            k3, b3 = _fuse_conv_bn_tensors(self.sconv, self.sbn)
            k1, b1 = _fuse_conv_bn_tensors(self.sconv1x1, self.sbn1x1)
            k_eq = k3 + F.pad(k1, [1, 1, 1, 1])
            b_eq = b3 + b1

            # Exact grouped Identity BN kernel
            c_in_g = self.channels // self.groups
            kid = torch.zeros_like(k3)
            for i in range(self.channels):
                kid[i, i % c_in_g, 1, 1] = 1.0
            std_id = torch.sqrt(self.sbn_id.running_var + self.sbn_id.eps)
            t_id = (self.sbn_id.weight / std_id).reshape(-1, 1, 1, 1)
            k_eq = k_eq + kid * t_id
            b_eq = b_eq + (self.sbn_id.bias - self.sbn_id.running_mean * self.sbn_id.weight / std_id)

            fused_sconv = nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.groups, bias=True
            ).to(device=k_eq.device, dtype=k_eq.dtype)
            fused_sconv.weight.copy_(k_eq)
            fused_sconv.bias.copy_(b_eq)
            self.sconv = fused_sconv.requires_grad_(False)

            for attr in ("sbn", "sconv1x1", "sbn1x1", "sbn_id"):
                if hasattr(self, attr):
                    delattr(self, attr)

            if hasattr(self, "f12_bn") and isinstance(self.f12_bn, nn.BatchNorm2d):
                self.f12 = _fuse_conv_bn_pair(self.f12, self.f12_bn)
                delattr(self, "f12_bn")
            if hasattr(self, "gn") and isinstance(self.gn, nn.BatchNorm2d):
                self.g = _fuse_conv_bn_pair(self.g, self.gn)
                delattr(self, "gn")
            for p in self.parameters():
                if p.is_leaf:
                    p.requires_grad_(False)
        self.fused = True

    def fuse(self) -> "RepStarFastBlock":
        self.switch_to_deploy()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "fused", False):
            a, b = self.f12(self.sconv(x)).chunk(2, dim=1)
            return F.relu(x + self.g(F.relu6(a, inplace=False) * b), inplace=False)
        feat = self.sbn(self.sconv(x)) + self.sbn1x1(self.sconv1x1(x)) + self.sbn_id(x)
        a, b = self.f12_bn(self.f12(feat)).chunk(2, dim=1)
        out = self.gn(self.g(F.relu6(a, inplace=False) * b))
        return F.relu(x + out, inplace=False)


class HRNetRepStarFast(BaseHRNetV2W18):
    """
    Exact `Exp28` (`HRNetStarFast`) backbone upgraded with training-only `RepStarFastBlock`.
    After `fuse()`, has IDENTICAL deploy architecture, layers, params, and speed to `HRNetStarFast`!
    """

    def __init__(self, pretrained: bool = True, pretrained_path: Optional[str] = None):
        super().__init__(
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            prune_stage4_p2=True,
            sparse_fuse=True,
            lite_block_type="star_fast",
            lite_min_channels=72,
            stage1_blocks=4,
            stage3_modules=4,
            stage4_modules=2,
            blocks_per_branch=3,
        )
        # Swap StarFastBlock -> RepStarFastBlock in Stage 3/4 wide branches (>=72ch)
        for stage in (self.stage3, self.stage4):
            for mod in stage:
                if not isinstance(mod, _HighResolutionModule):
                    continue
                for b_idx, branch in enumerate(mod.branches):
                    new_blocks = []
                    changed = False
                    for blk in branch:
                        if blk.__class__.__name__ == "StarFastBlock":
                            new_blocks.append(RepStarFastBlock(blk.channels, groups=4, expand=1.25))
                            changed = True
                        else:
                            new_blocks.append(blk)
                    if changed:
                        mod.branches[b_idx] = nn.Sequential(*new_blocks)


# ---------------------------------------------------------------------------
# Innovation 3: Scale-Aware Inner-GIoU + NWD Loss (0.00 ms Inference Cost!)
# ---------------------------------------------------------------------------
def inner_giou_nwd_loss(
    pred_xywh: torch.Tensor,
    gt_xywh: torch.Tensor,
    ratio: float = 1.25,
    nwd_c: float = 0.05,
    nwd_max_weight: float = 0.25,
    area_tau: float = 0.01,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    FP32-safe Scale-Aware `Inner-GIoU + NWD` loss for normalized `[cx, cy, w, h]` boxes.
      - `Inner-GIoU` (`ratio=1.25`): sharpens high-IoU (`mAP75` / `mAP50:95`) boundary regression.
      - `NWD` (`exp(-sqrt(W2^2) / C)`): provides smooth scale-invariant gradients for tiny UAV boxes.
      - Adaptive weight `w_nwd = nwd_max_weight * exp(-gt_area / area_tau)`.
    Returns per-box loss tensor of shape `[N]`.
    """
    with torch.autocast(device_type=pred_xywh.device.type, enabled=False):
        p = pred_xywh.float()
        g = gt_xywh.float()
        px, py, pw, ph = p.unbind(-1)
        gx, gy, gw, gh = g.unbind(-1)
        pw = pw.clamp(min=eps)
        ph = ph.clamp(min=eps)
        gw = gw.clamp(min=eps)
        gh = gh.clamp(min=eps)

        # 1. Inner-IoU (scaled auxiliary boxes by `ratio`)
        pb_x1, pb_x2 = px - pw * ratio * 0.5, px + pw * ratio * 0.5
        pb_y1, pb_y2 = py - ph * ratio * 0.5, py + ph * ratio * 0.5
        gb_x1, gb_x2 = gx - gw * ratio * 0.5, gx + gw * ratio * 0.5
        gb_y1, gb_y2 = gy - gh * ratio * 0.5, gy + gh * ratio * 0.5

        inter_w = (torch.minimum(pb_x2, gb_x2) - torch.maximum(pb_x1, gb_x1)).clamp(min=0.0)
        inter_h = (torch.minimum(pb_y2, gb_y2) - torch.maximum(pb_y1, gb_y1)).clamp(min=0.0)
        inter = inter_w * inter_h
        union = (pw * ph + gw * gh) * (ratio * ratio) - inter + eps
        inner_iou = inter / union

        # Standard enclosing box C for GIoU penalty (unscaled)
        p_x1, p_x2 = px - pw * 0.5, px + pw * 0.5
        p_y1, p_y2 = py - ph * 0.5, py + ph * 0.5
        g_x1, g_x2 = gx - gw * 0.5, gx + gw * 0.5
        g_y1, g_y2 = gy - gh * 0.5, gy + gh * 0.5

        raw_inter = (
            (torch.minimum(p_x2, g_x2) - torch.maximum(p_x1, g_x1)).clamp(min=0.0)
            * (torch.minimum(p_y2, g_y2) - torch.maximum(p_y1, g_y1)).clamp(min=0.0)
        )
        raw_union = pw * ph + gw * gh - raw_inter + eps
        cw = (torch.maximum(p_x2, g_x2) - torch.minimum(p_x1, g_x1)).clamp(min=eps)
        ch = (torch.maximum(p_y2, g_y2) - torch.minimum(p_y1, g_y1)).clamp(min=eps)
        c_area = cw * ch + eps
        inner_giou = inner_iou - (c_area - raw_union) / c_area
        loss_inner_giou = 1.0 - inner_giou

        # 2. Normalized Wasserstein Distance (NWD)
        w2_sq = (px - gx).pow(2) + (py - gy).pow(2) + 0.25 * ((pw - gw).pow(2) + (ph - gh).pow(2))
        nwd = torch.exp(-torch.sqrt(w2_sq + eps) / max(nwd_c, 1e-4))
        loss_nwd = 1.0 - nwd

        # 3. Scale-adaptive blending
        gt_area = (gw * gh).detach()
        w_nwd = nwd_max_weight * torch.exp(-gt_area / max(area_tau, 1e-5))
        return (1.0 - w_nwd) * loss_inner_giou + w_nwd * loss_nwd


def _detr_get_loss_bbox_inner_nwd(self, pred_bboxes, gt_bboxes, postfix=""):
    name_bbox = f"loss_bbox{postfix}"
    name_giou = f"loss_giou{postfix}"
    loss = {}
    if len(gt_bboxes) == 0:
        loss[name_bbox] = torch.tensor(0.0, device=self.device)
        loss[name_giou] = torch.tensor(0.0, device=self.device)
        return loss
    ratio, nwd_c, nwd_max_weight = getattr(self.__class__, "_inner_nwd_cfg", (1.25, 0.05, 0.25))
    loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
    loss_ig_nwd = inner_giou_nwd_loss(
        pred_bboxes, gt_bboxes, ratio=ratio, nwd_c=nwd_c, nwd_max_weight=nwd_max_weight
    )
    loss[name_giou] = self.loss_gain["giou"] * loss_ig_nwd.sum() / len(gt_bboxes)
    return {k: v.squeeze() for k, v in loss.items()}


def patch_rtdetr_inner_nwd_loss(ratio: float = 1.25, nwd_c: float = 0.05, nwd_max_weight: float = 0.25) -> None:
    """
    Cleanly patches Ultralytics `DETRLoss._get_loss_bbox` in memory for training-only
    Scale-Aware Inner-GIoU + NWD supervision. Zero impact on inference or saved weights!
    """
    from ultralytics.models.utils.loss import DETRLoss

    DETRLoss._inner_nwd_cfg = (float(ratio), float(nwd_c), float(nwd_max_weight))
    if getattr(DETRLoss, "_inner_nwd_patched", False):
        return

    DETRLoss._get_loss_bbox = _detr_get_loss_bbox_inner_nwd
    DETRLoss._inner_nwd_patched = True
    print(
        f"[Exp28+] Patched DETRLoss with Scale-Aware Inner-GIoU + NWD "
        f"(ratio={ratio}, nwd_c={nwd_c}, nwd_max_weight={nwd_max_weight}) [0.00 ms inference cost]"
    )
