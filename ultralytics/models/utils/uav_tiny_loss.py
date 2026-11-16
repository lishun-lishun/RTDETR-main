"""Optional matched-box scale-normalized geometry penalty for RT-DETR.

Invoked via RTDETRDetectionModel.init_criterion (opt-in by YAML config).
Uses the base Hungarian matching, VFL, denoising and auxiliary flow unchanged.
Research hypothesis only; standard scale-aware losses already exist in literature.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from .loss import RTDETRDetectionLoss


class SizeAwareRTDETRDetectionLoss(RTDETRDetectionLoss):
    def __init__(self, *args, geometry_weight=0.1, min_target_px=6.0,
                 max_scale_weight=2.0, input_size=640, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_weight = float(geometry_weight)
        self.min_target_px = float(min_target_px)
        self.max_scale_weight = float(max_scale_weight)
        self.input_size = float(input_size)
        if self.min_target_px <= 0 or self.input_size <= 0:
            raise ValueError('Input pixels / min target pixels must be positive')

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        result = super()._get_loss_bbox(pred_bboxes, gt_bboxes, postfix)
        if pred_bboxes.numel() == 0 or self.geometry_weight <= 0:
            return result
        # All boxes normalized xywh. Normalize center error by GT object width/height,
        # but clamp denominator to at least min_target_px to prevent gradient explosion.
        min_wh = self.min_target_px / self.input_size
        gt_wh = gt_bboxes[:, 2:].detach().clamp_min(min_wh)
        center_error = (pred_bboxes[:, :2] - gt_bboxes[:, :2]) / gt_wh
        # Relative log-size error is more scale symmetric than raw width L1.
        logsize_error = torch.log(pred_bboxes[:, 2:].clamp_min(min_wh) / gt_wh)
        sample_area_px = (gt_bboxes[:, 2:] * self.input_size).prod(-1).clamp_min(1.)
        small_priority = (self.min_target_px / sample_area_px.sqrt()).clamp(0.25, self.max_scale_weight)
        # SmoothL1 is zero for exact match; no GIoU or matcher alteration.
        per_box = F.smooth_l1_loss(center_error, torch.zeros_like(center_error), reduction='none').sum(-1)
        per_box += 0.5 * F.smooth_l1_loss(logsize_error, torch.zeros_like(logsize_error), reduction='none').sum(-1)
        penalty = (small_priority * per_box).mean() * self.geometry_weight
        result[f'loss_bbox{postfix}'] = result[f'loss_bbox{postfix}'] + penalty
        return result
