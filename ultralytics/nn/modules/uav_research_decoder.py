"""Opt-in UAV localization extensions for Ultralytics YOLOv8.0.201 RT-DETR.

The parent RTDETRDecoder has unchanged parameter names and its original output
contract.  Four independently controllable experiments are:
  E1 CenterRingEvidence (query-local P3 or optional sparse P2 sampling)
  E2 BoundedGeometryRefiner (bounded normalized bbox residual)
  E3 QualityReranker (quality residual trained through original VFL)
  E4 optional SizeAwareRTDETRDetectionLoss (separate file)

This is a research prototype, not a claim of novel or improved results.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .head import RTDETRDecoder


class CenterRingEvidence(nn.Module):
    """Sample 1 center + 4 boundary + 4 surrounding points at each query box.

    All box coordinates are normalized xywh in (0,1). Grid normalization uses
    align_corners=False consistently. No ground truth or decoder redesign.
    """

    def __init__(self, in_channels=256, evidence_dim=64):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, evidence_dim, 1, bias=False),
            nn.BatchNorm2d(evidence_dim),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Linear(3 * evidence_dim + 4, evidence_dim),
            nn.LayerNorm(evidence_dim),
            nn.SiLU(),
        )
        self.evidence_dim = evidence_dim

    def forward(self, feat: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 4 or boxes.ndim != 3 or boxes.size(-1) != 4:
            raise ValueError('Expected feat=[B,C,H,W], boxes=[B,K,4] normalized xywh')
        if feat.shape[0] != boxes.shape[0]:
            raise ValueError('Feature/bbox batch mismatch')
        b, n, _ = boxes.shape
        z = self.reduce(feat)
        if n == 0:
            return z.new_zeros((b, 0, self.evidence_dim))
        # Base xywh and offsets are all in normalized image coordinates.
        center = boxes[..., :2]
        wh = boxes[..., 2:].clamp_min(1e-5)
        # center, left/right/top/bottom boundary, and 4 outer corners
        offsets = boxes.new_tensor([
            [0.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [0.0, -0.5], [0.0, 0.5],
            [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0],
        ])
        coords = center.unsqueeze(2) + wh.unsqueeze(2) * offsets.view(1, 1, 9, 2)
        grid = (coords * 2.0 - 1.0).clamp(-1.0, 1.0)
        # [B,C,K*9,1], no per-query Python loop.
        samples = F.grid_sample(z, grid.reshape(b, n * 9, 1, 2), mode='bilinear',
                                padding_mode='border', align_corners=False)
        samples = samples.squeeze(-1).reshape(b, self.evidence_dim, n, 9)
        central = samples[..., 0].transpose(1, 2)
        boundary = samples[..., 1:5].mean(-1).transpose(1, 2)
        context = samples[..., 5:9].mean(-1).transpose(1, 2)
        # Preserve boundary-vs-surround contrast explicitly, with original box shape.
        statistic = torch.cat([central, boundary - context, central - context, boxes], -1)
        return self.project(statistic)


class BoundedGeometryRefiner(nn.Module):
    """Regress a strictly bounded pixel center shift and bounded log-size change."""

    def __init__(self, evidence_dim=64, max_shift_px=1.5, max_log_size=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(evidence_dim + 4, 96), nn.SiLU(),
            nn.Linear(96, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.max_shift_px = float(max_shift_px)
        self.max_log_size = float(max_log_size)

    def forward(self, box: torch.Tensor, evidence: torch.Tensor,
                image_hw: tuple[int, int]) -> torch.Tensor:
        raw = self.net(torch.cat([evidence, box], -1))
        h, w = image_hw
        if min(h, w) <= 0:
            raise ValueError('Invalid image size')
        scale = box.new_tensor([self.max_shift_px / w, self.max_shift_px / h])
        center = box[..., :2] + raw[..., :2].tanh() * scale
        size = box[..., 2:].clamp_min(1e-5) * torch.exp(raw[..., 2:].tanh() * self.max_log_size)
        # Clamp normalized prediction parameters (not xyxy extents).
        # At zero-init this is exactly identical to normal sigmoid outputs.
        return torch.cat([center.clamp(0.0, 1.0), size.clamp(1e-8, 1.0)], -1)


class QualityReranker(nn.Module):
    """Small sigmoid-independent logit correction; original VFL supervises it.

    This is *not* a separately supervised IoU quality predictor. To claim that,
    an additional matched-target quality objective would be required.
    """

    def __init__(self, evidence_dim=64, max_logit_offset=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(evidence_dim + 4, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.max_logit_offset = float(max_logit_offset)

    def forward(self, logits: torch.Tensor, boxes: torch.Tensor, evidence: torch.Tensor):
        delta = self.net(torch.cat([evidence, boxes], -1)).tanh()
        return logits + self.max_logit_offset * delta


class UAVResearchDecoder(RTDETRDecoder):
    """Drop-in substitute for RTDETRDecoder in the repository's rtdetr-r18 YAML.

    Constructor preserves original six positional YAML parameters. The parser
    injects `ch` in position one, just as for RTDETRDecoder. `opts` is an
    optional last YAML mapping, e.g. {"refine": true, "quality": false}.
    """

    def __init__(self, nc=80, ch=(256, 256, 256), hd=256, nq=300,
                 ndp=4, nh=8, ndl=3, opts=None):
        self.opts = opts or {}
        self.p2_source = bool(self.opts.get('p2_source', False))
        if self.p2_source and len(ch) != 4:
            raise ValueError('p2_source requires YAML from=[P2,P3,P4,P5]')
        if not self.p2_source and len(ch) != 3:
            raise ValueError('P3-only source requires YAML from=[P3,P4,P5]')
        # P2 is a sparse side-input only. Base decoder still sees 3 scales.
        super().__init__(nc=nc, ch=ch[1:] if self.p2_source else ch,
                         hd=hd, nq=nq, ndp=ndp, nh=nh, ndl=ndl)
        self.use_refine = bool(self.opts.get('refine', True))
        self.use_quality = bool(self.opts.get('quality', False))
        self.evidence_mode = self.opts.get('evidence_mode', 'sampled')
        if self.evidence_mode not in ('sampled', 'zero'):
            raise ValueError('evidence_mode must be sampled or zero')
        self.topk = int(self.opts.get('topk', 300))
        if self.topk < 1 or self.topk > nq:
            raise ValueError(f'topk must be in [1,{nq}]')
        self.evidence = CenterRingEvidence(ch[0] if self.p2_source else hd, int(self.opts.get('evidence_dim', 64)))
        ed = int(self.opts.get('evidence_dim', 64))
        self.refiner = BoundedGeometryRefiner(ed, self.opts.get('max_shift_px', 1.5),
                                             self.opts.get('max_log_size', 0.15))
        self.reranker = QualityReranker(ed, self.opts.get('max_logit_offset', 1.0))
        self._uav_p3 = None
        self._uav_p2 = None

    def _get_encoder_input(self, x):
        # Faithfully mirror the base implementation; reuse its projected P3
        # instead of applying another expensive 1x1 projection.
        projections = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        self._uav_p3 = projections[0]
        shapes = [[p.shape[2], p.shape[3]] for p in projections]
        return torch.cat([p.flatten(2).permute(0, 2, 1) for p in projections], 1), shapes

    def _modify(self, boxes, logits, feat):
        # boxes [B,N,4], logits [B,N,nc]; no denoising queries at this boundary.
        b, n, _ = boxes.shape
        stride = 4 if self.p2_source else 8
        image_hw = (feat.shape[-2] * stride, feat.shape[-1] * stride)
        if self.topk == n:
            ids = torch.arange(n, device=boxes.device).unsqueeze(0).expand(b, n)
        else:
            ids = logits.detach().amax(dim=-1).topk(self.topk, dim=1).indices
        index_box = ids[..., None].expand(-1, -1, 4)
        sel_boxes = torch.gather(boxes, 1, index_box)
        sel_scores = torch.gather(logits, 1, ids[..., None].expand(-1, -1, logits.size(-1)))
        if self.evidence_mode == 'zero':
            evidence = sel_boxes.new_zeros((*sel_boxes.shape[:2], self.evidence.evidence_dim))
        else:
            evidence = self.evidence(feat, sel_boxes)
        result_boxes = self.refiner(sel_boxes, evidence, image_hw) if self.use_refine else sel_boxes
        result_scores = self.reranker(sel_scores, sel_boxes, evidence) if self.use_quality else sel_scores
        # Scatter creates a differentiable copy; unsampled queries stay unchanged.
        out_boxes = boxes.scatter(1, index_box, result_boxes) if self.use_refine else boxes
        out_scores = logits.scatter(1, ids[..., None].expand(-1, -1, logits.size(-1)),
                                    result_scores) if self.use_quality else logits
        return out_boxes, out_scores

    def forward(self, x, batch=None):
        if self.p2_source:
            self._uav_p2 = x[0]
            x_base = x[1:]
        else:
            x_base = x
        try:
            original = super().forward(x_base, batch=batch)
            detail = self._uav_p2 if self.p2_source else self._uav_p3
        finally:
            # A failed construction-time probe must not retain an activation.
            self._uav_p2 = None
            self._uav_p3 = None
        if not (self.use_refine or self.use_quality):
            return original
        if detail is None:
            raise RuntimeError('Missing local evidence feature')
        if self.training:
            dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta = original
            # Base decoder trains with DN positions *before* normal nq queries.
            # Modify only the LAST layer's ordinary queries, never DN queries.
            last_b = dec_boxes[-1, :, -self.num_queries:, :]
            last_s = dec_scores[-1, :, -self.num_queries:, :]
            fixed_b, fixed_s = self._modify(last_b, last_s, detail)
            dec_boxes = torch.cat([dec_boxes[:-1], torch.cat([
                dec_boxes[-1, :, :-self.num_queries, :], fixed_b], dim=1).unsqueeze(0)], dim=0)
            dec_scores = torch.cat([dec_scores[:-1], torch.cat([
                dec_scores[-1, :, :-self.num_queries, :], fixed_s], dim=1).unsqueeze(0)], dim=0)
            return dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta
        if self.export:
            # Base model exports concatenated [xywh, sigmoid(cls)].
            boxes = original[..., :4]
            scores = torch.logit(original[..., 4:].clamp(min=1e-6, max=1. - 1e-6))
            boxes, scores = self._modify(boxes, scores, detail)
            return torch.cat([boxes, scores.sigmoid()], dim=-1)
        _, raw = original
        dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta = raw
        boxes, scores = self._modify(dec_boxes[-1], dec_scores[-1], detail)
        dec_boxes = torch.cat([dec_boxes[:-1], boxes.unsqueeze(0)], 0)
        dec_scores = torch.cat([dec_scores[:-1], scores.unsqueeze(0)], 0)
        y = torch.cat([boxes, scores.sigmoid()], dim=-1)
        return y, (dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta)
