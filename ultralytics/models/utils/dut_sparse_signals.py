"""Opt-in *training-only* losses for sparse, tiny UAV detection in RT-DETR.

Four independent hypotheses; not a claim of accuracy or novelty:
GAP: Gaussian anchor prior on all encoder sites (CVPR'25 information prior inspiration)
HNR: Hungarian-positive vs hard negative score ranking (DEIM motivation)
PLM: progressive decoder-layer localization monotonicity (D-FINE motivation)
BQR: quantization-tolerant local box jitter integration (UGS motivation)

No inference module, no new network parameters. All boxes normalized cxcywh.
Works with vanilla PyTorch 2.3+ and does not require ultralytics for unit tests.
"""
from __future__ import annotations
from typing import Mapping, Sequence
import torch
from torch import nn
from torch.nn import functional as F


def _xyxy(x: torch.Tensor) -> torch.Tensor:
    half = x[..., 2:].clamp_min(1.e-8) * .5
    return torch.cat((x[..., :2] - half, x[..., :2] + half), dim=-1)


def pairwise_iou(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """IoU for [N,4] x [M,4] in normalized cxcywh."""
    if x.numel() == 0 or y.numel() == 0:
        return x.new_zeros((x.shape[0], y.shape[0]))
    a = _xyxy(x.float())[:, None, :]
    b = _xyxy(y.float())[None, :, :]
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
    return inter / (area_a + area_b - inter).clamp_min(1.e-8)


def aligned_giou(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Aligned generalized IoU [N]."""
    a = _xyxy(x.float())
    b = _xyxy(y.float())
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
    union = (area_a + area_b - inter).clamp_min(1.e-8)
    enclose_lt = torch.minimum(a[..., :2], b[..., :2])
    enclose_rb = torch.maximum(a[..., 2:], b[..., 2:])
    enclosing = (enclose_rb - enclose_lt).clamp_min(0).prod(-1).clamp_min(1.e-8)
    return inter / union - (enclosing - union) / enclosing


class DUTSparseSignals(nn.Module):
    """Optional training signals, original RT-DETR matched supervision unchanged.

    Arguments from YAML are scalar weights; default 0 means each term disabled.
    Returns 4 individually named scalars. No learnable parameters.
    """

    def __init__(self, gaussian_weight=0., rank_weight=0., monotonic_weight=0.,
                 jitter_weight=0., tiny_threshold_px=24., hard_neg_topk=8,
                 margin=0.25, monotonic_margin=0.025, jitter_px=0.5,
                 min_sigma_cells=0.8, negative_iou=0.25):
        super().__init__()
        self.gaussian_weight = float(gaussian_weight)
        self.rank_weight = float(rank_weight)
        self.monotonic_weight = float(monotonic_weight)
        self.jitter_weight = float(jitter_weight)
        self.tiny_threshold_px = float(tiny_threshold_px)
        self.hard_neg_topk = int(hard_neg_topk)
        self.margin = float(margin)
        self.monotonic_margin = float(monotonic_margin)
        self.jitter_px = float(jitter_px)
        self.min_sigma_cells = float(min_sigma_cells)
        self.negative_iou = float(negative_iou)
        if any(k < 0 for k in (self.gaussian_weight, self.rank_weight,
                               self.monotonic_weight, self.jitter_weight)):
            raise ValueError('All regularization weights must be nonnegative')
        if self.hard_neg_topk < 1 or self.tiny_threshold_px <= 0:
            raise ValueError('Invalid hard-negative topk or tiny threshold')

    @property
    def active(self):
        return any((self.gaussian_weight, self.rank_weight,
                    self.monotonic_weight, self.jitter_weight))

    def _scale_weight(self, gt: torch.Tensor, image_hw: tuple[int, int]):
        h, w = image_hw
        maxside = torch.maximum(gt[..., 2] * w, gt[..., 3] * h).clamp_min(0)
        # 1 for <=threshold, smoothly decreases to 0 for larger instances
        return (self.tiny_threshold_px / maxside.clamp_min(self.tiny_threshold_px)).clamp(0, 1)

    def _dense_gaussian(self, dense_logits: torch.Tensor, shapes: Sequence[tuple[int, int]],
                        batch: Mapping, image_hw: tuple[int,int]):
        if dense_logits is None or not shapes:
            raise ValueError('GAP needs encoder score logits and three actual feature shapes')
        bs, n, nc = dense_logits.shape
        if sum(h*w for h,w in shapes) != n:
            raise ValueError(f'Dense score sites {n} != measured feature shapes {shapes}')
        groups = [int(g) for g in batch['gt_groups']]
        if sum(groups) != len(batch['bboxes']) or bs != len(groups):
            raise ValueError('GT group structure inconsistent')
        logits = dense_logits.float()
        start_gt = 0
        start_site = 0
        level_losses = []
        for h, w in shapes:
            yy, xx = torch.meshgrid((torch.arange(h, device=logits.device, dtype=torch.float32)+0.5)/h,
                                    (torch.arange(w, device=logits.device, dtype=torch.float32)+0.5)/w,
                                    indexing='ij')
            target = torch.zeros((bs, h, w, nc), device=logits.device, dtype=torch.float32)
            start_gt = 0
            for b, count in enumerate(groups):
                for j in range(count):
                    gt = batch['bboxes'][start_gt+j].float()
                    cls = int(batch['cls'][start_gt+j].item())
                    if cls < 0 or cls >= nc:
                        raise ValueError('GT class out of range')
                    # Use localized Gaussian; floor width in feature cells avoids zero coverage.
                    sx = torch.maximum(gt[2]/2, gt.new_tensor(self.min_sigma_cells/w))
                    sy = torch.maximum(gt[3]/2, gt.new_tensor(self.min_sigma_cells/h))
                    blob = torch.exp(-0.5*((xx-gt[0])/sx).square()-0.5*((yy-gt[1])/sy).square())
                    target[b, ..., cls] = torch.maximum(target[b, ..., cls], blob)
                start_gt += count
            s = logits[:, start_site:start_site+h*w].reshape(bs, h, w, nc)
            # Soft foreground targets, hard-negative focal damping; normalize by total weights.
            prob = s.detach().sigmoid()
            weight = target + 0.05 * (1.0-target) * prob.square()
            loss = F.binary_cross_entropy_with_logits(s, target, reduction='none')
            level_losses.append((loss*weight).sum()/weight.sum().clamp_min(1.))
            start_site += h*w
        return torch.stack(level_losses).mean()

    @staticmethod
    def _normalized_error(p: torch.Tensor, g: torch.Tensor, image_hw: tuple[int,int]):
        h, w = image_hw
        pixscale = p.new_tensor((w,h))
        delta = (p[..., :2] - g[..., :2]) * pixscale
        diagonal = (g[..., 2:] * pixscale).square().sum(-1).sqrt().clamp_min(4.0)
        center = delta.square().sum(-1).sqrt() / diagonal
        size = (p[..., 2:].clamp_min(1e-6) / g[..., 2:].clamp_min(1e-6)).log().abs().mean(-1)
        return center + 0.25 * size

    def _hard_negative_rank(self, boxes: torch.Tensor, scores: torch.Tensor,
                            batch: Mapping, matches, image_hw):
        terms = []
        for b, (src_cpu, tgt_cpu) in enumerate(matches):
            if not len(src_cpu):
                continue
            src = src_cpu.to(boxes.device)
            tgt = tgt_cpu.to(boxes.device)
            gt = batch['bboxes'][tgt]
            classes = batch['cls'][tgt].long()
            all_iou = pairwise_iou(boxes[b].detach(), gt)
            for k in range(len(src)):
                cls = classes[k]
                # Exclude candidates close to any GT, not just this positive's GT.
                valid = all_iou.max(dim=1).values < self.negative_iou
                valid[src] = False
                if not valid.any():
                    continue
                logits = scores[b, :, cls]
                hard = logits.detach().masked_fill(~valid, -1.e8).topk(
                    min(self.hard_neg_topk, int(valid.sum().item()))).indices
                # Positive should outrank false positives far from that GT.
                pos = logits[src[k]]
                neg = logits[hard]
                w = self._scale_weight(gt[k:k+1], image_hw).mean()
                terms.append(w * F.softplus(self.margin + neg - pos).mean())
        return torch.stack(terms).mean() if terms else scores.sum()*0.

    def _decoder_monotonic(self, all_boxes: torch.Tensor, batch: Mapping,
                           matches, image_hw):
        # all_boxes [L,B,Q,4]: layer 0 is encoder; layers 1... are decoder.
        if all_boxes.shape[0] < 3:
            return all_boxes.sum()*0.
        out = []
        final = all_boxes[-1]
        for b, (src_cpu, tgt_cpu) in enumerate(matches):
            if not len(src_cpu):
                continue
            src = src_cpu.to(final.device)
            gt = batch['bboxes'][tgt_cpu.to(final.device)]
            last_err = self._normalized_error(final[b,src],gt,image_hw)
            tiny_w = self._scale_weight(gt,image_hw)
            for layer in range(1,all_boxes.shape[0]-1):
                prev_err = self._normalized_error(all_boxes[layer,b,src].detach(),gt,image_hw)
                out.append((tiny_w * F.relu(last_err-prev_err+self.monotonic_margin)).mean())
        return torch.stack(out).mean() if out else final.sum()*0.

    def _box_jitter_expectation(self, boxes: torch.Tensor, batch: Mapping,
                                matches, image_hw):
        h, w = image_hw
        terms = []
        shifts = [(self.jitter_px/w,0.),(-self.jitter_px/w,0.),
                  (0.,self.jitter_px/h),(0.,-self.jitter_px/h)]
        for b, (src_cpu, tgt_cpu) in enumerate(matches):
            if not len(src_cpu):
                continue
            src = src_cpu.to(boxes.device)
            gt = batch['bboxes'][tgt_cpu.to(boxes.device)]
            pred = boxes[b,src]
            tiny_w = self._scale_weight(gt,image_hw)
            values = []
            for dx,dy in shifts:
                jitter_gt = gt.clone()
                delta = gt.new_tensor((dx,dy))
                jitter_gt[:,:2] = (gt[:,:2]+delta).clamp(0.0,1.0)
                values.append((1.0-aligned_giou(pred,jitter_gt)).clamp_min(0.))
            terms.append((torch.stack(values).mean(0)*tiny_w).mean())
        return torch.stack(terms).mean() if terms else boxes.sum()*0.

    def forward(self, all_boxes: torch.Tensor, all_scores: torch.Tensor,
                batch: Mapping, matches,
                dense_logits: torch.Tensor | None = None,
                shapes: Sequence[tuple[int,int]] | None = None,
                image_hw: tuple[int,int]=(640,640)) -> dict[str,torch.Tensor]:
        if all_boxes.ndim != 4 or all_scores.ndim != 4:
            raise ValueError('Expected [L,B,Q,4] and [L,B,Q,C]')
        zero = all_scores.sum()*0.
        out = {'loss_gap':zero,'loss_hnr':zero,'loss_plm':zero,'loss_bqr':zero}
        if self.gaussian_weight:
            out['loss_gap'] = self.gaussian_weight*self._dense_gaussian(dense_logits,shapes,batch,image_hw)
        if self.rank_weight:
            out['loss_hnr'] = self.rank_weight*self._hard_negative_rank(
                all_boxes[-1],all_scores[-1],batch,matches,image_hw)
        if self.monotonic_weight:
            out['loss_plm'] = self.monotonic_weight*self._decoder_monotonic(
                all_boxes,batch,matches,image_hw)
        if self.jitter_weight:
            out['loss_bqr'] = self.jitter_weight*self._box_jitter_expectation(
                all_boxes[-1],batch,matches,image_hw)
        return out
