# -*- coding: utf-8 -*-
"""
Innovation 10: Noise-Robust Uncertainty-Aware Tiny Object Localization (TOLFDecoder & TOLFUncertaintyLoss)
===========================================================================================================
Target Problem:
    Recent empirical studies (`TOLF`, arXiv Jan 2026) reveal that ~34% of tiny object
    annotations contain 1-2 px boundary looseness or motion-blur ambiguity. For a 10x10 px
    UAV in `DUT-Anti-UAV`, a 1 px label noise drops true IoU from 1.00 to 0.67. Optimizing
    hard `L1 + GIoU` forces the detector to overfit noisy annotations, causing `AP75` oscillation.

Core Mechanism (Inspired by TOLF Jan 2026):
    1. Ultra-Lightweight 4-Dim Per-Edge Log-Variance Head (`+0.004M` params):
       Attached alongside each decoder layer's box head, predicts `s = log(sigma^2) in [-4, 2]`
       for `(cx, cy, w, h)` (zero-initialized so Epoch-0 variance = 1.0).
    2. Noise-Robust Laplace/Gaussian NLL + Uncertainty-Modulated Auxiliary Loss (`TOLFUncertaintyLoss`):
       During training, computes heteroscedastic Laplace NLL on matched tiny/small boxes:
           `L_nll = |b_pred - b_gt| * exp(-0.5 * s) + 0.5 * s`
       with clamped log-variance in FP32, automatically attenuating erratic gradients from
       blurred/noisy labels while sharpening clean UAV edges.
    3. Inference-Time Confidence Calibration:
       Calibrates classification confidence via `p_cal = p * exp(-lambda_cal * mean(sigma))`
       so well-localized sharp UAV detections rank above uncertain/blurry false positives.
"""

from __future__ import annotations
import copy
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0.0, max=1.0)
    return torch.log(x.clamp(min=eps) / (1.0 - x).clamp(min=eps))


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TOLFUncertaintyLoss(nn.Module):
    """
    Noise-Robust Heteroscedastic Laplace NLL + Uncertainty Regularization Loss (Train-Time-Only).
    Computes uncertainty-modulated localization loss on the nearest matched predictions
    in FP32, preventing overfitting to 1-2 px annotation noise on tiny UAVs.
    """

    def __init__(
        self,
        loss_weight: float = 0.25,
        tau_area: float = 0.015,
        enabled: bool = True,
    ):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.tau_area = float(tau_area)
        self.enabled = bool(enabled)

    def forward(
        self,
        log_vars_stack: torch.Tensor,
        dec_bboxes: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        num_dn: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            log_vars_stack: [L, B, Nq_total, 4] clamped in [-4.0, 2.0]
            dec_bboxes: [L, B, Nq_total, 4] predicted normalized xywh
            batch: dict containing 'bboxes' [N_gt, 4] and 'batch_idx' [N_gt]
        """
        device = dec_bboxes.device
        if (not self.enabled) or ("bboxes" not in batch) or (batch["bboxes"].numel() == 0):
            return torch.tensor(0.0, device=device, dtype=torch.float32)

        with torch.autocast(device_type=device.type, enabled=False):
            lv = log_vars_stack[:, :, num_dn:, :].float()  # [L, B, 300, 4]
            pb = dec_bboxes[:, :, num_dn:, :].float()      # [L, B, 300, 4]
            gt_b = batch["bboxes"].float().to(device)
            bidx = batch["batch_idx"].long().to(device)

            num_l, bs = pb.shape[0], pb.shape[1]
            total_nll = torch.tensor(0.0, device=device, dtype=torch.float32)
            count = 0

            for b_i in range(bs):
                mask = bidx == b_i
                gb = gt_b[mask]  # [Ng, 4]
                if gb.shape[0] == 0:
                    continue
                # Emphasize tiny/small UAV targets where label noise hurts AP75 most
                areas = (gb[:, 2] * gb[:, 3]).clamp(min=1e-6)
                scale_w = (1.0 + 1.5 * torch.exp(-areas / max(self.tau_area, 1e-4))).unsqueeze(-1)  # [Ng, 1]

                for l_idx in range(num_l):
                    p_l = pb[l_idx, b_i]   # [300, 4]
                    s_l = lv[l_idx, b_i]   # [300, 4]
                    with torch.no_grad():
                        # Match each GT to its closest query by L1 distance
                        dist = torch.cdist(gb, p_l, p=1)  # [Ng, 300]
                        best_q = dist.argmin(dim=1)       # [Ng]

                    sel_p = p_l[best_q]  # [Ng, 4]
                    sel_s = s_l[best_q]  # [Ng, 4]
                    abs_err = (sel_p - gb).abs()
                    # Heteroscedastic Laplace NLL with softplus offset to keep loss strictly >= 0
                    nll = abs_err * torch.exp(-0.5 * sel_s) + 0.5 * F.softplus(sel_s)
                    total_nll = total_nll + (nll * scale_w).mean()
                    count += 1

            if count == 0:
                return torch.tensor(0.0, device=device, dtype=torch.float32)
            return self.loss_weight * (total_nll / float(count))


class TOLFDecoder(nn.Module):
    """
    Noise-Robust Uncertainty-Aware Tiny Object Localization Decoder (Innovation 10).
    Drop-in replacement for Layer 26 `RTDETRDecoder` on Original `rtdetr-r18.yaml`.
    """

    export = False

    def __init__(
        self,
        nc: int = 80,
        ch: Tuple[int, ...] = (256, 256, 256),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 3,
        d_ffn: int = 1024,
        eval_idx: int = -1,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        calib_lambda: float = 0.12,
        enabled: bool = True,
    ):
        super().__init__()
        self.nc = int(nc)
        self.hd = int(hd)
        self.nq = int(nq)
        self.num_queries = self.nq
        self.nl = len(ch)
        self.ndl = int(ndl)
        self.eval_idx = eval_idx if eval_idx >= 0 else ndl + eval_idx
        self.num_denoising = int(nd)
        self.label_noise_ratio = float(label_noise_ratio)
        self.box_noise_scale = float(box_noise_scale)
        self.learnt_init_query = bool(learnt_init_query)
        self.calib_lambda = float(calib_lambda)
        self.enabled = bool(enabled)

        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch
        )

        try:
            from ultralytics.nn.modules.transformer import DeformableTransformerDecoderLayer
            dec_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        except Exception:
            from RTDETR_UAV_Paper_Innovations.innovation_3.module import UADDeformableDecoderLayer
            dec_layer = UADDeformableDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp, enabled=False)
        # Preserve RTDETRDecoder's decoder.layers.* checkpoint keys.
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([copy.deepcopy(dec_layer) for _ in range(ndl)])

        if self.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(self.nc + 1, hd, padding_idx=self.nc)
            init.normal_(self.denoising_class_embed.weight[:-1])

        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, self.nc)
        self.enc_bbox_head = _MLP(hd, hd, 4, num_layers=3)

        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = _MLP(4, 2 * hd, hd, num_layers=2)
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, self.nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([_MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        # Ultra-lightweight 4-dim per-edge log-variance head (+0.004M params total)
        self.dec_logvar_head = nn.ModuleList([nn.Linear(hd, 4) for _ in range(ndl)])

        # Ephemeral training cache (cleared in _uav_loss and __deepcopy__)
        self.last_tolf_logvars: Optional[torch.Tensor] = None
        self.last_tolf_bboxes: Optional[torch.Tensor] = None
        self.last_tolf_num_dn: int = 0

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bias_cls = float(-math.log((1 - 0.01) / 0.01))
        init.constant_(self.enc_score_head.bias, bias_cls)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_, lv_ in zip(self.dec_score_head, self.dec_bbox_head, self.dec_logvar_head):
            init.constant_(cls_.bias, bias_cls)
            init.constant_(reg_.layers[-1].weight, 0.0)
            init.constant_(reg_.layers[-1].bias, 0.0)
            init.constant_(lv_.weight, 0.0)
            init.constant_(lv_.bias, 0.0)
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    def _get_encoder_input(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, List[List[int]]]:
        feats, shapes = [], []
        for i, feat in enumerate(x):
            h, w = feat.shape[2:]
            shapes.append([h, w])
            feats.append(self.input_proj[i](feat).flatten(2).permute(0, 2, 1))
        return torch.cat(feats, dim=1), shapes

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device=torch.device("cpu"), eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))
        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > eps) & (anchors < 1.0 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1.0 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.tensor(torch.inf, dtype=dtype, device=device))
        return anchors, valid_mask

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = feats.shape[0]
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(torch.where(valid_mask, feats, torch.tensor(0.0, dtype=feats.dtype, device=feats.device)))
        enc_outputs_scores = self.enc_score_head(features)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.nq, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype, device=feats.device).unsqueeze(-1).repeat(1, self.nq).view(-1)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.nq, -1)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.nq, -1)
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.nq, -1)
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)
        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def forward(self, x: List[torch.Tensor], batch: Optional[dict] = None):
        feats, shapes = self._get_encoder_input(x)
        dn_embed, dn_bbox, attn_mask, dn_meta = None, None, None, None
        if self.training and self.num_denoising > 0 and batch is not None:
            from ultralytics.models.utils.ops import get_cdn_group
            dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
                batch=batch, num_classes=self.nc, num_queries=self.nq,
                class_embed=self.denoising_class_embed.weight, num_dn=self.num_denoising,
                cls_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale, training=self.training,
            )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        num_dn = dn_meta["dn_num_split"][0] if (dn_meta is not None and "dn_num_split" in dn_meta) else 0

        dec_bboxes, dec_scores, dec_logvars = [], [], []
        output = embed
        spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=feats.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        for i, layer in enumerate(self.decoder.layers):
            ref_sig = refer_bbox.sigmoid()
            query_pos = self.query_pos_head(ref_sig)
            output = layer(output, ref_sig, feats, shapes, None, attn_mask, query_pos)

            score = self.dec_score_head[i](output)
            bbox_delta = self.dec_bbox_head[i](output)
            inter_bbox = torch.sigmoid(bbox_delta + refer_bbox)
            log_var = self.dec_logvar_head[i](output).clamp(min=-4.0, max=2.0)

            if self.training:
                dec_scores.append(score)
                dec_bboxes.append(inter_bbox)
                dec_logvars.append(log_var)
            elif i == self.eval_idx:
                dec_scores.append(score)
                dec_bboxes.append(inter_bbox)
                dec_logvars.append(log_var)
                break

            refer_bbox = _inverse_sigmoid(inter_bbox.detach() if self.training else inter_bbox)

        dec_bboxes_t = torch.stack(dec_bboxes)
        dec_scores_t = torch.stack(dec_scores)
        dec_logvars_t = torch.stack(dec_logvars)

        if self.training and self.enabled:
            self.last_tolf_logvars = dec_logvars_t
            self.last_tolf_bboxes = dec_bboxes_t
            self.last_tolf_num_dn = int(num_dn)
        else:
            self.last_tolf_logvars = None
            self.last_tolf_bboxes = None
            self.last_tolf_num_dn = 0

        if self.training:
            return dec_bboxes_t, dec_scores_t, enc_bboxes, enc_scores, dn_meta
        else:
            probs = dec_scores_t.squeeze(0).sigmoid()
            if self.enabled and self.calib_lambda > 0.0:
                with torch.autocast(device_type=probs.device.type, enabled=False):
                    sigma_mean = torch.exp(0.5 * dec_logvars_t.squeeze(0).float()).mean(dim=-1, keepdim=True)
                    cal_factor = torch.exp(-self.calib_lambda * (sigma_mean - 1.0).clamp(min=0.0))
                    probs = (probs.float() * cal_factor).to(probs.dtype)
            y = torch.cat([dec_bboxes_t.squeeze(0), probs], dim=-1)
            if self.export:
                return y
            return y, (dec_bboxes_t, dec_scores_t, enc_bboxes, enc_scores, dn_meta)

    def __deepcopy__(self, memo):
        self.last_tolf_logvars = None
        self.last_tolf_bboxes = None
        self.last_tolf_num_dn = 0
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
