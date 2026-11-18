# -*- coding: utf-8 -*-
"""
Innovation 4: Scale-Adaptive Hybrid NWD-InnerGIoU Matching & Cross-Layer
Localization Self-Distillation (SAH-NWD & CW-LSD)
=========================================================================
Target Stage: HungarianMatcher (`ops.py`) & RTDETRDetectionLoss (`loss.py`)
Inference Overhead: ZERO (+0.00M Params, +0.00 GFLOPs, +0.00 ms Latency)
Compatible with: Ultralytics YOLOv8.0.201 RT-DETR-R18, Python 3.10, PyTorch 2.3.1+cu121

Core Mathematical Mechanisms:
1. Scale-Adaptive Exponential Gate beta(s) & Dynamic Auxiliary Box Inner-GIoU:
   Given GT normalized area s_g = w_g * h_g, defines beta(s_g) = exp(-s_g / tau_s).
   - For tiny UAVs (s_g << tau_s, beta -> 1): smoothly blends Normalized Wasserstein
     Distance (NWD) and expanded Inner-GIoU (with dynamic ratio r(s_g) > 1.0) to
     provide non-vanishing smooth gradients even when boxes do not overlap.
   - For medium/large UAVs (s_g >> tau_s, beta -> 0): smoothly reverts to standard
     GIoU, preserving sharp boundary regression on large targets.
2. Consistent Scale-Adaptive Hungarian Matcher (ScaleAdaptiveHungarianMatcher):
   Aligns the bipartite matching cost matrix with the scale-adaptive hybrid similarity
   so decoder layers do not suffer from matching index oscillation on tiny targets.
3. Generalized Continuous Matchability-Aware Loss (GC-MAL):
   Replaces the discontinuous point-set IoU soft target `q = IoU` in VarifocalLoss
   with `q_hybrid = clip((1 - 0.5*beta)*IoU + 0.5*beta*NWD, 0, 1)^gamma_mal`,
   eliminating the gradient dead-zone when matched tiny boxes have zero overlap.
4. Continuous Wasserstein Cross-Layer Localization Self-Distillation (CW-LSD):
   Uses the final decoder layer's refined bounding boxes as a detached teacher
   to supervise earlier decoder layers and the encoder proposal head via NWD + L1
   distance weighted by teacher matchability/confidence.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def bbox_iou_xywh(
    box1: torch.Tensor,
    box2: torch.Tensor,
    giou: bool = False,
    scale_ratio: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Element-wise or pairwise IoU and GIoU computation for normalized (cx, cy, w, h) boxes.
    Supports optional Inner-IoU dynamic scaling ratio `scale_ratio` (r(s_g)).
    Args:
        box1: [N, 4] or [M, 1, 4] (cx, cy, w, h)
        box2: [N, 4] or [1, K, 4] (cx, cy, w, h)
        giou: whether to return GIoU in addition to standard IoU
        scale_ratio: optional tensor broadcastable to box1[..., 2]
    Returns:
        iou: [N] or [M, K]
        giou_val: [N] or [M, K]
    """
    b1_cx, b1_cy, b1_w, b1_h = box1.unbind(-1)
    b2_cx, b2_cy, b2_w, b2_h = box2.unbind(-1)

    if scale_ratio is not None:
        b1_w = b1_w * scale_ratio
        b1_h = b1_h * scale_ratio
        b2_w = b2_w * scale_ratio
        b2_h = b2_h * scale_ratio

    b1_x1, b1_x2 = b1_cx - b1_w * 0.5, b1_cx + b1_w * 0.5
    b1_y1, b1_y2 = b1_cy - b1_h * 0.5, b1_cy + b1_h * 0.5
    b2_x1, b2_x2 = b2_cx - b2_w * 0.5, b2_cx + b2_w * 0.5
    b2_y1, b2_y2 = b2_cy - b2_h * 0.5, b2_cy + b2_h * 0.5

    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0.0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0.0)
    inter = inter_w * inter_h

    area1 = b1_w.clamp(min=0.0) * b1_h.clamp(min=0.0)
    area2 = b2_w.clamp(min=0.0) * b2_h.clamp(min=0.0)
    union = area1 + area2 - inter + eps
    iou = inter / union

    if not giou:
        return iou, iou

    cw = (torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)).clamp(min=0.0)
    ch = (torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)).clamp(min=0.0)
    c_area = cw * ch + eps
    giou_val = iou - (c_area - union) / c_area
    return iou, giou_val


def normalized_wasserstein_distance(
    box1: torch.Tensor,
    box2: torch.Tensor,
    img_size: float = 640.0,
    constant_c: float = 12.8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes Normalized Gaussian Wasserstein Distance (NWD) in (0, 1] between
    normalized (cx, cy, w, h) boxes scaled to `img_size`.
    """
    b1 = box1 * img_size
    b2 = box2 * img_size
    center_dist_sq = (b1[..., 0] - b2[..., 0]).pow(2) + (b1[..., 1] - b2[..., 1]).pow(2)
    wh_dist_sq = ((b1[..., 2] - b2[..., 2]).pow(2) + (b1[..., 3] - b2[..., 3]).pow(2)) * 0.25
    w2_dist = torch.sqrt(center_dist_sq + wh_dist_sq + eps)
    nwd = torch.exp(-w2_dist / max(constant_c, 1e-3))
    return nwd


class ScaleAdaptiveHungarianMatcher(nn.Module):
    """
    Scale-Adaptive Bipartite Hungarian Matcher for RT-DETR.
    When `enabled=False`, reproduces the exact Ultralytics `HungarianMatcher` cost matrix.
    When `enabled=True`, incorporates scale-adaptive NWD + Inner-GIoU geometric cost
    so tiny UAV targets do not suffer from GIoU plateau/cliff during matching.
    """

    def __init__(
        self,
        cost_gain: Optional[Dict[str, float]] = None,
        use_fl: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0,
        tau_s: float = 0.0025,
        constant_c: float = 12.8,
        img_size: float = 640.0,
        enabled: bool = True,
    ):
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 2.0, "bbox": 5.0, "giou": 2.0}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.alpha = alpha
        self.gamma = gamma
        self.tau_s = float(tau_s)
        self.constant_c = float(constant_c)
        self.img_size = float(img_size)
        self.enabled = bool(enabled)

    @torch.no_grad()
    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: List[int],
        masks: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            pred_bboxes: [B, Nq, 4] in (cx, cy, w, h) normalized [0, 1]
            pred_scores: [B, Nq, nc] raw logits
            gt_bboxes: [num_gts, 4]
            gt_cls: [num_gts]
            gt_groups: list[int] of length B
        Returns:
            list of (pred_idx_tensor, gt_idx_tensor) tuples for each image in batch
        """
        bs, nq, nc = pred_scores.shape
        if sum(gt_groups) == 0:
            return [
                (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
                for _ in range(bs)
            ]

        # Flatten predictions across batch: [B*Nq, ...]
        out_prob = pred_scores.detach().view(-1, nc).sigmoid()  # [B*Nq, nc]
        out_bbox = pred_bboxes.detach().view(-1, 4).float()     # [B*Nq, 4]
        gt_bboxes_f = gt_bboxes.detach().float()                # [num_gts, 4]

        # 1. Classification cost: [B*Nq, num_gts]
        out_prob_gt = out_prob[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (
                (1.0 - self.alpha) * (out_prob_gt ** self.gamma) * (-(1.0 - out_prob_gt + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1.0 - out_prob_gt) ** self.gamma) * (-(out_prob_gt + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob_gt

        # 2. L1 bbox cost: [B*Nq, num_gts]
        cost_bbox = torch.cdist(out_bbox, gt_bboxes_f, p=1)

        # 3. Geometric GIoU / Scale-Adaptive Hybrid NWD-InnerGIoU cost: [B*Nq, num_gts]
        _, giou_mat = bbox_iou_xywh(out_bbox.unsqueeze(1), gt_bboxes_f.unsqueeze(0), giou=True)
        cost_giou = -giou_mat

        if self.enabled:
            # Compute GT area s_g = w_g * h_g -> [1, num_gts]
            s_g = (gt_bboxes_f[:, 2] * gt_bboxes_f[:, 3]).clamp(min=1e-8).unsqueeze(0)
            beta = torch.exp(-s_g / max(self.tau_s, 1e-6))  # [1, num_gts] in (0, 1]
            nwd_mat = normalized_wasserstein_distance(
                out_bbox.unsqueeze(1),
                gt_bboxes_f.unsqueeze(0),
                img_size=self.img_size,
                constant_c=self.constant_c,
            )  # [B*Nq, num_gts]
            # Hybrid geometric cost: smoothly supplements GIoU with NWD on tiny UAV targets
            cost_geom = (1.0 - 0.5 * beta) * cost_giou + 0.5 * beta * (-nwd_mat)
        else:
            cost_geom = cost_giou

        C = (
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox
            + self.cost_gain["giou"] * cost_geom
        )
        C = C.view(bs, nq, -1).cpu()
        C = torch.nan_to_num(C, nan=1e5, posinf=1e5, neginf=-1e5)

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long))
            for i, j in indices
        ]


class SAHNWDRTDETRLoss(nn.Module):
    """
    Scale-Adaptive Hybrid NWD-InnerGIoU & Cross-Layer Localization Self-Distillation Loss.
    Drop-in replacement for Ultralytics `RTDETRDetectionLoss`.
    Preserves exact output dictionary keys (`loss_class`, `loss_bbox`, `loss_giou`,
    `loss_class_aux`, `loss_bbox_aux`, `loss_giou_aux`, plus `*_dn` keys) so that
    `RTDETRDetectionModel.loss()` works without any modification to training log keys!
    """

    def __init__(
        self,
        nc: int = 1,
        loss_gain: Optional[Dict[str, float]] = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = True,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        tau_s: float = 0.0025,
        constant_c: float = 12.8,
        img_size: float = 640.0,
        r_min: float = 1.00,
        r_max: float = 1.30,
        omega_nwd: float = 0.50,
        gamma_mal: float = 1.5,
        lambda_lsd: float = 0.50,
        enabled: bool = True,
    ):
        super().__init__()
        if loss_gain is None:
            loss_gain = {"class": 1.0, "bbox": 5.0, "giou": 2.0, "no_object": 0.1, "mask": 1.0, "dice": 1.0}
        self.nc = int(nc)
        self.loss_gain = loss_gain
        self.aux_loss = bool(aux_loss)
        self.use_fl = bool(use_fl)
        self.use_vfl = bool(use_vfl)
        self.use_uni_match = bool(use_uni_match)
        self.uni_match_ind = int(uni_match_ind)

        self.tau_s = float(tau_s)
        self.constant_c = float(constant_c)
        self.img_size = float(img_size)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.omega_nwd = float(omega_nwd)
        self.gamma_mal = float(gamma_mal)
        self.lambda_lsd = float(lambda_lsd)
        self.enabled = bool(enabled)

        self.matcher = ScaleAdaptiveHungarianMatcher(
            cost_gain={"class": 2.0, "bbox": 5.0, "giou": 2.0},
            use_fl=use_fl,
            tau_s=tau_s,
            constant_c=constant_c,
            img_size=img_size,
            enabled=enabled,
        )
        self.device = torch.device("cpu")

    def set_enabled(self, flag: bool):
        self.enabled = bool(flag)
        self.matcher.enabled = bool(flag)

    @staticmethod
    def _get_index(match_indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _compute_hybrid_quality(
        self,
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes generalized continuous matching quality q_hybrid in [0, 1]
        for matched pairs.
        """
        iou, _ = bbox_iou_xywh(pred_bboxes, gt_bboxes, giou=False)
        iou = iou.clamp(min=0.0, max=1.0)
        if not self.enabled:
            return iou

        s_g = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).clamp(min=1e-8)
        beta = torch.exp(-s_g / max(self.tau_s, 1e-6))
        nwd = normalized_wasserstein_distance(
            pred_bboxes, gt_bboxes, img_size=self.img_size, constant_c=self.constant_c
        )
        q_raw = ((1.0 - 0.5 * beta) * iou + 0.5 * beta * nwd).clamp(min=1e-4, max=1.0)
        # Apply DEIM MAL power scaling q^(gamma_mal / 1.5) to boost low-IoU positive supervision
        power = max(0.5, min(self.gamma_mal / 1.5, 1.5))
        return q_raw.pow(power)

    def _get_loss_class(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        gt_scores: torch.Tensor,
        num_gts: int,
        postfix: str = "",
    ) -> Dict[str, torch.Tensor]:
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1].float()  # [bs, nq, nc]
        gt_scores_3d = gt_scores.view(bs, nq, 1) * one_hot

        pred_fp32 = pred_scores.float()
        prob = pred_fp32.sigmoid()

        if self.use_fl:
            if num_gts > 0 and self.use_vfl:
                if self.enabled:
                    # Generalized Continuous Matchability-Aware VFL (GC-MAL)
                    # Positive weight: q_hybrid, Negative weight: 0.75 * p^gamma
                    weight = gt_scores_3d * one_hot + 0.75 * prob.pow(2.0) * (1.0 - one_hot)
                    bce = F.binary_cross_entropy_with_logits(pred_fp32, gt_scores_3d, reduction="none")
                    loss_cls = (bce * weight).mean(1).sum() / max(num_gts, 1)
                else:
                    # Standard Ultralytics VarifocalLoss
                    weight = gt_scores_3d * one_hot + 0.75 * prob.pow(2.0) * (1.0 - one_hot)
                    bce = F.binary_cross_entropy_with_logits(pred_fp32, gt_scores_3d, reduction="none")
                    loss_cls = (bce * weight).mean(1).sum() / max(num_gts, 1)
            else:
                # Standard Focal Loss when num_gts == 0
                pt = prob * one_hot + (1.0 - prob) * (1.0 - one_hot)
                alpha_t = 0.25 * one_hot + 0.75 * (1.0 - one_hot)
                bce = F.binary_cross_entropy_with_logits(pred_fp32, one_hot, reduction="none")
                loss_cls = (alpha_t * ((1.0 - pt) ** 2.0) * bce).mean(1).sum() / max(num_gts, 1)
        else:
            loss_cls = F.binary_cross_entropy_with_logits(pred_fp32, gt_scores_3d, reduction="none").mean(1).sum()

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(
        self,
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        postfix: str = "",
    ) -> Dict[str, torch.Tensor]:
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"
        loss: Dict[str, torch.Tensor] = {}

        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        pred_f = pred_bboxes.float()
        gt_f = gt_bboxes.float()
        num_gts = len(gt_f)

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_f, gt_f, reduction="sum") / num_gts

        _, std_giou = bbox_iou_xywh(pred_f, gt_f, giou=True)
        loss_std_giou = 1.0 - std_giou

        if not self.enabled:
            loss[name_giou] = self.loss_gain["giou"] * loss_std_giou.sum() / num_gts
        else:
            # Scale-Adaptive Hybrid NWD + Dynamic Inner-GIoU Loss
            s_g = (gt_f[:, 2] * gt_f[:, 3]).clamp(min=1e-8)
            beta = torch.exp(-s_g / max(self.tau_s, 1e-6))  # [N] in (0, 1]
            r_scale = self.r_min + (self.r_max - self.r_min) * beta  # [N] > 1.0 for tiny objects

            _, inner_giou = bbox_iou_xywh(pred_f, gt_f, giou=True, scale_ratio=r_scale)
            loss_inner_giou = 1.0 - inner_giou

            nwd = normalized_wasserstein_distance(
                pred_f, gt_f, img_size=self.img_size, constant_c=self.constant_c
            )
            loss_nwd = 1.0 - nwd

            loss_tiny_branch = self.omega_nwd * loss_nwd + (1.0 - self.omega_nwd) * loss_inner_giou
            loss_hybrid = (1.0 - beta) * loss_std_giou + beta * loss_tiny_branch
            loss[name_giou] = self.loss_gain["giou"] * loss_hybrid.sum() / num_gts

        return {k: v.squeeze() for k, v in loss.items()}

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: List[int],
        postfix: str = "",
        match_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        if match_indices is None:
            match_indices = self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)

        idx, gt_idx = self._get_index(match_indices)
        bs, nq = pred_scores.shape[:2]
        device = pred_scores.device

        targets = torch.full((bs, nq), self.nc, device=device, dtype=gt_cls.dtype)
        gt_scores = torch.zeros((bs, nq), device=device, dtype=torch.float32)

        if len(gt_bboxes) > 0 and len(gt_idx) > 0:
            idx_dev = (idx[0].to(device), idx[1].to(device))
            gt_idx_dev = gt_idx.to(device)
            matched_pred_bboxes = pred_bboxes[idx_dev]
            matched_gt_bboxes = gt_bboxes[gt_idx_dev]
            targets[idx_dev] = gt_cls[gt_idx_dev]
            gt_scores[idx_dev] = self._compute_hybrid_quality(
                matched_pred_bboxes.detach(), matched_gt_bboxes.detach()
            )
        else:
            matched_pred_bboxes = torch.zeros((0, 4), device=device, dtype=pred_bboxes.dtype)
            matched_gt_bboxes = torch.zeros((0, 4), device=device, dtype=gt_bboxes.dtype)

        loss: Dict[str, torch.Tensor] = {}
        loss.update(self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(matched_pred_bboxes, matched_gt_bboxes, postfix))
        return loss, match_indices

    def _compute_cw_lsd_loss(
        self,
        aux_bboxes: torch.Tensor,
        teacher_bboxes: torch.Tensor,
        teacher_scores: torch.Tensor,
        teacher_matches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Continuous Wasserstein Cross-Layer Localization Self-Distillation (CW-LSD).
        Transfers fine-grained localization knowledge from final decoder layer L
        to earlier decoder layers and encoder proposals (0..L-1).
        Args:
            aux_bboxes: [L_aux, B, Nq, 4]
            teacher_bboxes: [B, Nq, 4] (detached final layer boxes)
            teacher_scores: [B, Nq, nc] (detached final layer scores)
            teacher_matches: list of (pred_idx, gt_idx) for final layer
        """
        if (not self.enabled) or self.lambda_lsd <= 0.0 or aux_bboxes.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        t_box = teacher_bboxes.detach().float()  # [B, Nq, 4]
        t_conf = teacher_scores.detach().float().sigmoid().max(dim=-1).values  # [B, Nq]

        # Build teacher distillation weight mask: high weight on matched queries + moderate on confident queries
        weight_mask = 0.25 * t_conf.pow(2.0)  # [B, Nq]
        for b_i, (src_i, _) in enumerate(teacher_matches):
            if len(src_i) > 0:
                weight_mask[b_i, src_i.to(weight_mask.device)] = 1.0

        total_lsd = torch.tensor(0.0, device=self.device)
        num_aux = aux_bboxes.shape[0]
        for l_idx in range(num_aux):
            s_box = aux_bboxes[l_idx].float()  # [B, Nq, 4]
            nwd_l = normalized_wasserstein_distance(
                s_box, t_box, img_size=self.img_size, constant_c=self.constant_c
            )  # [B, Nq]
            l1_l = F.l1_loss(s_box, t_box, reduction="none").sum(dim=-1)  # [B, Nq]
            layer_distill = (weight_mask * ((1.0 - nwd_l) + l1_l)).sum() / (weight_mask.sum() + 1e-6)
            total_lsd = total_lsd + layer_distill

        return self.lambda_lsd * (total_lsd / max(num_aux, 1))

    def _get_loss_aux(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: List[int],
        match_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        postfix: str = "",
        teacher_bboxes: Optional[torch.Tensor] = None,
        teacher_scores: Optional[torch.Tensor] = None,
        teacher_matches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_vec = torch.zeros(3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
            )

        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            loss_, _ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss_vec[0] = loss_vec[0] + loss_[f"loss_class{postfix}"]
            loss_vec[1] = loss_vec[1] + loss_[f"loss_bbox{postfix}"]
            loss_vec[2] = loss_vec[2] + loss_[f"loss_giou{postfix}"]

        # Add CW-LSD self-distillation into loss_giou_aux so backward & logging work seamlessly
        if (
            self.enabled
            and postfix == ""
            and teacher_bboxes is not None
            and teacher_scores is not None
            and teacher_matches is not None
        ):
            loss_lsd = self._compute_cw_lsd_loss(
                pred_bboxes, teacher_bboxes, teacher_scores, teacher_matches
            )
            loss_vec[2] = loss_vec[2] + loss_lsd

        return {
            f"loss_class_aux{postfix}": loss_vec[0],
            f"loss_bbox_aux{postfix}": loss_vec[1],
            f"loss_giou_aux{postfix}": loss_vec[2],
        }

    @staticmethod
    def get_dn_match_indices(
        dn_pos_idx: List[torch.Tensor],
        dn_num_group: int,
        gt_groups: List[int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices

    def forward_single_branch(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: Dict[str, Any],
        postfix: str = "",
        match_indices: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        self.device = pred_bboxes.device
        gt_cls = batch["cls"]
        gt_bboxes = batch["bboxes"]
        gt_groups = batch["gt_groups"]

        total_loss, last_matches = self._get_loss(
            pred_bboxes[-1],
            pred_scores[-1],
            gt_bboxes,
            gt_cls,
            gt_groups,
            postfix=postfix,
            match_indices=match_indices,
        )

        if self.aux_loss and pred_bboxes.shape[0] > 1:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1],
                    pred_scores[:-1],
                    gt_bboxes,
                    gt_cls,
                    gt_groups,
                    match_indices=match_indices,
                    postfix=postfix,
                    teacher_bboxes=pred_bboxes[-1],
                    teacher_scores=pred_scores[-1],
                    teacher_matches=last_matches,
                )
            )
        return total_loss

    def forward(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor],
        batch: Dict[str, Any],
        dn_bboxes: Optional[torch.Tensor] = None,
        dn_scores: Optional[torch.Tensor] = None,
        dn_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Matches exact signature of `RTDETRDetectionLoss.forward(preds, batch, dn_bboxes, dn_scores, dn_meta)`.
        """
        pred_bboxes, pred_scores = preds
        total_loss = self.forward_single_branch(pred_bboxes, pred_scores, batch)

        if dn_meta is not None and dn_bboxes is not None and dn_scores is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])
            dn_loss = self.forward_single_branch(
                dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices
            )
            total_loss.update(dn_loss)
        else:
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in list(total_loss.keys())})

        return total_loss
