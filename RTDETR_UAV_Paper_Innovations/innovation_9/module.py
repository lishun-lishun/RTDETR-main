# -*- coding: utf-8 -*-
"""
Innovation 9: Scale-Aware Fine-Grained Distribution Refinement & Area-Weighted LSD (SAFDRDecoder)
==================================================================================================
Target Problem:
    Standard RT-DETR directly regresses 4 scalar coordinates `(cx, cy, w, h)` per decoder
    layer. On blurred or tiny 6x6 px UAVs in `DUT-Anti-UAV`, direct scalar regression suffers
    from high gradient variance and imprecise edge boundaries, limiting `AP75` and `mAP50:95`.

Core Mechanism (Inspired by D-FINE ICLR 2025 Spotlight & FGUDet IEEE 2026):
    1. Non-Uniform Fine-Grained Distribution Refinement (SA-FDR):
       Instead of predicting a single scalar delta, each decoder layer predicts a 4x(B+1)
       discrete probability distribution over the 4 box edge offsets `(Δl, Δt, Δr, Δb)`
       using a non-uniform bin grid `W_bins` that concentrates fine sub-pixel resolution
       near 0 (for tiny UAVs) while covering wider shifts at the tails.
       Distributions are refined residually across layers:
           `logits^(l) = logits^(l-1) + Δlogits^(l)`
       and integrated in FP32 via `p^(l) = softmax(logits^(l))`, `δ^(l) = Σ p^(l)_k * bin_k`.
    2. Area-Weighted Cross-Layer Localization Self-Distillation (`AWLSDDistillLoss`):
       During training, distills the final decoder layer's edge probability distribution
       `p^(L)` to earlier layers `p^(0..L-1)` via KL divergence weighted inversely by
       predicted box area `w_area = 1 + 2 * exp(-area / tau_a)`, forcing shallow decoder
       layers to prioritize tiny UAV boundary precision (+0.00 GFLOPs at inference).
    3. Drop-in Replacement for Layer 26 `RTDETRDecoder` on Original `rtdetr-r18.yaml`:
       Strictly preserves 8,400 encoder tokens, 300 queries, `[B, 300, 4+nc]` output contract,
       AMP FP16 safety, and `ModelEMA` `deepcopy` safety.
"""

from __future__ import annotations
import copy
import math
from typing import List, Optional, Tuple
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


def _build_nonuniform_bins(reg_max: int = 16, reg_scale: float = 4.0) -> torch.Tensor:
    """
    Constructs a symmetric non-uniform bin grid of length `reg_max + 1` in [-0.5, 0.5]
    with high sub-pixel density near 0 (inspired by D-FINE weighting function).
    """
    half = reg_max // 2
    idx = torch.linspace(-1.0, 1.0, steps=reg_max + 1, dtype=torch.float32)
    # Cubic-linear hybrid gives dense bins near 0 and wide bins at ±0.5
    bins = 0.5 * (0.35 * idx + 0.65 * (idx.abs().pow(2.0) * idx.sign())) / max(reg_scale / 4.0, 0.5)
    return bins  # [reg_max + 1]


class AWLSDDistillLoss(nn.Module):
    """
    Area-Weighted Localization Self-Distillation Loss (Train-Time-Only) for SAFDRDecoder.
    Distills final decoder layer's edge probability distribution `p^(L)` to earlier layers
    `p^(0..L-1)` via KL divergence weighted inversely by object box area.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        tau_area: float = 0.015,
        loss_weight: float = 0.35,
        enabled: bool = True,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.tau_area = float(tau_area)
        self.loss_weight = float(loss_weight)
        self.enabled = bool(enabled)

    def forward(
        self,
        edge_logits_list: List[torch.Tensor],
        final_bboxes: torch.Tensor,
        final_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            edge_logits_list: list of L tensors, each [B, Nq_total, 4, reg_max+1]
            final_bboxes: [B, Nq_total, 4]
            final_scores: [B, Nq_total, nc]
        """
        if (not self.enabled) or len(edge_logits_list) < 2:
            return torch.tensor(0.0, device=final_bboxes.device, dtype=torch.float32)

        device = final_bboxes.device
        with torch.autocast(device_type=device.type, enabled=False):
            t = max(self.temperature, 0.5)
            teacher_logits = edge_logits_list[-1].detach().float()  # [B, Nq, 4, K]
            teacher_prob = F.softmax(teacher_logits / t, dim=-1)    # [B, Nq, 4, K]

            # Area-weighted + confidence-weighted mask focusing on small UAVs
            conf = final_scores.detach().float().sigmoid().max(dim=-1).values  # [B, Nq]
            area = (final_bboxes.detach().float()[..., 2] * final_bboxes.detach().float()[..., 3]).clamp(min=1e-6)
            area_weight = 1.0 + 2.0 * torch.exp(-area / max(self.tau_area, 1e-4))  # [B, Nq] in (1, 3]
            q_weight = (conf.pow(1.5) * area_weight).clamp(min=1e-4)  # [B, Nq]
            q_weight_norm = q_weight / (q_weight.sum() + 1e-6)

            total_kl = torch.tensor(0.0, device=device, dtype=torch.float32)
            num_stu = len(edge_logits_list) - 1
            for l_idx in range(num_stu):
                stu_log_prob = F.log_softmax(edge_logits_list[l_idx].float() / t, dim=-1)
                kl_per_edge = F.kl_div(stu_log_prob, teacher_prob, reduction="none").sum(dim=-1).mean(dim=-1)  # [B, Nq]
                total_kl = total_kl + (kl_per_edge * q_weight_norm).sum() * (t * t)

            return self.loss_weight * (total_kl / float(max(num_stu, 1)))


class SAFDRDecoder(nn.Module):
    """
    Scale-Aware Fine-Grained Distribution Refinement Decoder (Innovation 9).
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
        reg_max: int = 16,
        reg_scale: float = 4.0,
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
        self.reg_max = int(reg_max)
        self.num_bins = self.reg_max + 1
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
        # Layer 0 predicts coarse scalar box; all layers also have fine-grained distribution heads
        self.dec_bbox_head = nn.ModuleList([_MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])
        self.fdr_dist_head = nn.ModuleList([_MLP(hd, hd, 4 * self.num_bins, num_layers=3) for _ in range(ndl)])

        # Scale-adaptive bin step modulation
        self.register_buffer("nonuniform_bins", _build_nonuniform_bins(self.reg_max, reg_scale), persistent=False)
        self.u_fdr_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Ephemeral cache for AW-LSD training loss (cleared in _uav_loss and __deepcopy__)
        self.last_fdr_logits: Optional[List[torch.Tensor]] = None
        self.last_fdr_final_boxes: Optional[torch.Tensor] = None
        self.last_fdr_final_scores: Optional[torch.Tensor] = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bias_cls = float(-math.log((1 - 0.01) / 0.01))
        init.constant_(self.enc_score_head.bias, bias_cls)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_, fdr_ in zip(self.dec_score_head, self.dec_bbox_head, self.fdr_dist_head):
            init.constant_(cls_.bias, bias_cls)
            init.constant_(reg_.layers[-1].weight, 0.0)
            init.constant_(reg_.layers[-1].bias, 0.0)
            init.constant_(fdr_.layers[-1].weight, 0.0)
            init.constant_(fdr_.layers[-1].bias, 0.0)
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

    def _integrate_fdr_box(self, base_box: torch.Tensor, dist_logits: torch.Tensor) -> torch.Tensor:
        """
        Integrates edge offset distributions `dist_logits` [B, Nq, 4, K] around `base_box` [B, Nq, 4]
        (cx, cy, w, h) in FP32 with scale-aware step modulation.
        """
        with torch.autocast(device_type=base_box.device.type, enabled=False):
            bf = base_box.float()
            bins = self.nonuniform_bins.to(device=base_box.device, dtype=torch.float32)  # [K]
            prob = F.softmax(dist_logits.float(), dim=-1)  # [B, Nq, 4, K]
            edge_delta = (prob * bins.view(1, 1, 1, -1)).sum(dim=-1)  # [B, Nq, 4] for (dl, dt, dr, db)

            # Scale-aware step amplitude: larger relative precision for small boxes
            scale_gate = 0.5 * torch.sigmoid(self.u_fdr_scale.float())
            cx, cy, bw, bh = bf[..., 0], bf[..., 1], bf[..., 2].clamp(min=1e-4), bf[..., 3].clamp(min=1e-4)
            dl = edge_delta[..., 0] * bw * scale_gate
            dt = edge_delta[..., 1] * bh * scale_gate
            dr = edge_delta[..., 2] * bw * scale_gate
            db = edge_delta[..., 3] * bh * scale_gate

            x1 = (cx - 0.5 * bw + dl).clamp(min=0.0, max=1.0)
            y1 = (cy - 0.5 * bh + dt).clamp(min=0.0, max=1.0)
            x2 = (cx + 0.5 * bw + dr).clamp(min=0.0, max=1.0)
            y2 = (cy + 0.5 * bh + db).clamp(min=0.0, max=1.0)

            new_w = (x2 - x1).clamp(min=1e-4)
            new_h = (y2 - y1).clamp(min=1e-4)
            new_cx = (x1 + x2) * 0.5
            new_cy = (y1 + y2) * 0.5
            refined = torch.stack([new_cx, new_cy, new_w, new_h], dim=-1).clamp(min=1e-4, max=1.0 - 1e-4)
            return refined.to(base_box.dtype)

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

        dec_bboxes, dec_scores = [], []
        edge_logits_list: List[torch.Tensor] = []
        output = embed
        prev_dist_logits: Optional[torch.Tensor] = None
        spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=feats.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        for i, layer in enumerate(self.decoder.layers):
            ref_sig = refer_bbox.sigmoid()
            query_pos = self.query_pos_head(ref_sig)
            output = layer(output, ref_sig, feats, shapes, None, attn_mask, query_pos)

            score = self.dec_score_head[i](output)
            scalar_delta = self.dec_bbox_head[i](output)
            coarse_box = torch.sigmoid(scalar_delta + refer_bbox)

            if self.enabled:
                b, n_tot, _ = output.shape
                delta_logits = self.fdr_dist_head[i](output).view(b, n_tot, 4, self.num_bins)
                cur_logits = delta_logits if prev_dist_logits is None else (prev_dist_logits + delta_logits)
                inter_bbox = self._integrate_fdr_box(coarse_box, cur_logits)
                prev_dist_logits = cur_logits.detach() if self.training else cur_logits
                if self.training:
                    edge_logits_list.append(cur_logits)
            else:
                inter_bbox = coarse_box

            if self.training:
                dec_scores.append(score)
                dec_bboxes.append(inter_bbox)
            elif i == self.eval_idx:
                dec_scores.append(score)
                dec_bboxes.append(inter_bbox)
                break

            refer_bbox = _inverse_sigmoid(inter_bbox.detach() if self.training else inter_bbox)

        dec_bboxes_t = torch.stack(dec_bboxes)
        dec_scores_t = torch.stack(dec_scores)

        if self.training and self.enabled:
            self.last_fdr_logits = edge_logits_list
            self.last_fdr_final_boxes = dec_bboxes_t[-1].detach()
            self.last_fdr_final_scores = dec_scores_t[-1].detach()
        else:
            self.last_fdr_logits = None
            self.last_fdr_final_boxes = None
            self.last_fdr_final_scores = None

        if self.training:
            return dec_bboxes_t, dec_scores_t, enc_bboxes, enc_scores, dn_meta
        else:
            y = torch.cat([dec_bboxes_t.squeeze(0), dec_scores_t.squeeze(0).sigmoid()], dim=-1)
            if self.export:
                return y
            return y, (dec_bboxes_t, dec_scores_t, enc_bboxes, enc_scores, dn_meta)

    def __deepcopy__(self, memo):
        self.last_fdr_logits = None
        self.last_fdr_final_boxes = None
        self.last_fdr_final_scores = None
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
