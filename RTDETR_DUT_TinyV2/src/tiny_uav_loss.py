"""M3: tiny target center-aware assignment; M4: size-adaptive NWD geometry.

The original matcher and loss are used unchanged if the new weights are zero.
No extra parameter or computation remains in inference.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.utils.metrics import bbox_iou


class TinyCenterHungarianMatcher(HungarianMatcher):
    def __init__(self, center_weight=0.0, input_pixels=640, tiny_threshold=24,
                 tiny_transition=6, **kwargs):
        super().__init__(**kwargs)
        self.center_weight = float(center_weight)
        self.input_pixels = float(input_pixels)
        self.tiny_threshold = float(tiny_threshold)
        self.tiny_transition = float(tiny_transition)

    @torch.no_grad()
    def forward(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups,
                masks=None, gt_mask=None):
        if self.center_weight == 0 or sum(gt_groups) == 0:
            return super().forward(pred_bboxes, pred_scores, gt_bboxes, gt_cls,
                                   gt_groups, masks, gt_mask)
        if self.with_mask:
            raise NotImplementedError('TinyCenterHungarianMatcher currently supports bbox detection only')
        bs, nq, nc = pred_scores.shape
        p = pred_scores.detach().float().reshape(-1, nc).sigmoid()
        pb = pred_bboxes.detach().float().reshape(-1, 4)
        gt = gt_bboxes.detach().float()
        score = p[:, gt_cls.long()].clamp(1e-6, 1 - 1e-6)
        neg = (1 - self.alpha) * score.pow(self.gamma) * (-torch.log1p(-score))
        pos = self.alpha * (1 - score).pow(self.gamma) * (-score.log())
        cls_cost = pos - neg
        l1_cost = (pb[:, None, :] - gt[None, :, :]).abs().sum(-1)
        giou_cost = 1 - bbox_iou(pb[:, None, :], gt[None, :, :],
                                 xywh=True, GIoU=True).squeeze(-1)
        # Pixel-normalized center error conditioned on GT diagonal.
        diff = (pb[:, None, :2] - gt[None, :, :2]) * self.input_pixels
        center_px = torch.sqrt(diff.square().sum(-1) + 1e-8)
        diag_px = gt[:, 2:4].norm(dim=-1).mul(self.input_pixels).clamp(min=4.0)
        norm = center_px / diag_px[None, :]
        size_px = gt[:, 2:4].amax(-1) * self.input_pixels
        tiny = torch.sigmoid((self.tiny_threshold - size_px) / max(self.tiny_transition, 1e-3))
        extra = self.center_weight * tiny[None, :] * norm / (1 + norm)
        C = (self.cost_gain['class'] * cls_cost +
             self.cost_gain['bbox'] * l1_cost +
             self.cost_gain['giou'] * giou_cost + extra)
        # Fail visibly, do not silently convert NaNs to 0 in experimental matcher.
        if not torch.isfinite(C).all():
            raise FloatingPointError('Non-finite tiny Hungarian matching costs')
        C = C.view(bs, nq, -1).cpu()
        result = []
        offsets = [0]
        for n in gt_groups[:-1]:
            offsets.append(offsets[-1] + int(n))
        for b, (part, offset, n) in enumerate(zip(C, offsets, gt_groups)):
            if n:
                row, col = linear_sum_assignment(part[:, offset:offset + n].numpy())
                result.append((torch.as_tensor(row, dtype=torch.long),
                               torch.as_tensor(col + offset, dtype=torch.long)))
            else:
                result.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
        return result


class TinyUAVRTDETRLoss(RTDETRDetectionLoss):
    """Overrides only matched regression and, optionally, Hungarian matching.

    The base class still computes encoder / decoder auxiliary / DN losses.
    Extra regression penalty is folded into existing loss_giou so that
    RTDETRTrainer logging (three named losses) stays compatible.
    """
    def __init__(self, nc=1, settings=None, **kwargs):
        super().__init__(nc=nc, **kwargs)
        cfg = settings or {}
        self.nwd_weight = float(cfg.get('nwd_weight', 0.0))
        self.nwd_c_px = float(cfg.get('nwd_c_px', 12.0))
        self.input_pixels = float(cfg.get('input_pixels', 640))
        self.tiny_threshold = float(cfg.get('tiny_threshold', 24))
        self.tiny_transition = float(cfg.get('tiny_transition', 6))
        center_weight = float(cfg.get('center_match_weight', 0.0))
        if self.nwd_c_px <= 0 or self.input_pixels <= 0:
            raise ValueError('positive pixel scale and NWD normalization are required')
        if center_weight > 0:
            self.matcher = TinyCenterHungarianMatcher(
                center_weight=center_weight, input_pixels=self.input_pixels,
                tiny_threshold=self.tiny_threshold,
                tiny_transition=self.tiny_transition,
                cost_gain={'class': 2, 'bbox': 5, 'giou': 2})
        # Base repository already offers self.nwd_loss and self.gcd_loss.
        # Keep both off to isolate our scale-conditioning experiment.
        self.nwd_loss = False
        self.gcd_loss = False

    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=''):
        losses = super()._get_loss_bbox(pred_bboxes, gt_bboxes, postfix)
        if self.nwd_weight == 0 or gt_bboxes.numel() == 0:
            return losses
        p = pred_bboxes.float()
        g = gt_bboxes.float()
        # Gaussian W2 closed form for axis-aligned bounding boxes.
        # All input coords are normalized xywh, scale to 640px (or YAML imgsz).
        center_diff = (p[:, :2] - g[:, :2]) * self.input_pixels
        size_diff = (p[:, 2:] - g[:, 2:]) * self.input_pixels * 0.5
        dist = torch.sqrt(center_diff.square().sum(-1) +
                          size_diff.square().sum(-1) + 1e-9)
        nwd = 1 - torch.exp(-dist / self.nwd_c_px)
        gt_side_px = g[:, 2:].amax(-1) * self.input_pixels
        w = torch.sigmoid((self.tiny_threshold - gt_side_px) /
                          max(self.tiny_transition, 1e-3))
        extra = self.nwd_weight * (w * nwd).sum() / max(len(g), 1)
        losses[f'loss_giou{postfix}'] = losses[f'loss_giou{postfix}'] + extra.to(
            losses[f'loss_giou{postfix}'].dtype)
        return losses
