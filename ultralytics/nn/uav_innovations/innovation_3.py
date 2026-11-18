# -*- coding: utf-8 -*-
"""
Innovation 3: Uncertainty-Aware Query Selection & Context-Variance-Floored
Deformable Decoder (UAD-CR / UADCRDecoder)
==========================================================================
Target Stage: RTDETRDecoder, DeformableTransformerDecoder, MSDeformAttn (Layer 26)
Compatible with: Ultralytics YOLOv8.0.201 RT-DETR-R18, Python 3.10, PyTorch 2.3.1+cu121

Core Mathematical Mechanisms:
1. Quality-Calibrated Uncertainty-Minimal Top-300 Query Selection:
   Augments the encoder output with a lightweight localization quality / objectness
   branch `enc_quality_head(features)` and ranks the 8,400 candidate tokens via:
       s_i = p_i * (1 + lambda_q * q_i * exp(-|p_i - q_i|))
   Strictly preserves the 300-query budget (`nq = 300`) while prioritizing low-contrast
   tiny UAVs with coherent spatial structure and penalizing high-uncertainty background clutter.
2. Scale- & Level-Adaptive Context Variance Floor in MSDeformAttn:
   In standard MSDeformAttn, box-mode sampling offsets are scaled by `0.5 * refer_bbox[..., 2:]`.
   When a reference box converges to a tiny 6x6px UAV (w_r ≈ 0.009), all 4 sampling points
   collapse into a single feature pixel (Rank-1 Sampling Collapse).
   We introduce a level-adaptive Context Variance Floor:
       w_eff^(l) = sqrt(w_r^2 + (eta_l / W_l)^2),  h_eff^(l) = sqrt(h_r^2 + (eta_l / H_l)^2)
   guaranteeing a minimum local context sampling radius of `eta_l` feature grid cells
   around tiny targets without perturbing medium/large targets (where w_r >> eta_l / W_l).
3. Target-Decoupled Cross-Attention Gating & Uncertainty-Damped Box Refinement:
   Incorporates zero-initialized cross-attention gating and per-layer uncertainty-modulated
   bounding box refinement while preserving 100% state_dict key compatibility with
   pretrained `RTDETRDecoder` checkpoints.
"""

from __future__ import annotations
import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse sigmoid (logit) function."""
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Create a ModuleList of n deep-copied modules."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] boxes from (cx, cy, w, h) to (x1, y1, x2, y2)."""
    y = x.clone()
    half_w = x[..., 2] * 0.5
    half_h = x[..., 3] * 0.5
    y[..., 0] = x[..., 0] - half_w
    y[..., 1] = x[..., 1] - half_h
    y[..., 2] = x[..., 0] + half_w
    y[..., 3] = x[..., 1] + half_h
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """Convert [..., 4] boxes from (x1, y1, x2, y2) to (cx, cy, w, h)."""
    y = x.clone()
    y[..., 0] = (x[..., 0] + x[..., 2]) * 0.5
    y[..., 1] = (x[..., 1] + x[..., 3]) * 0.5
    y[..., 2] = (x[..., 2] - x[..., 0]).clamp(min=1e-6)
    y[..., 3] = (x[..., 3] - x[..., 1]).clamp(min=1e-6)
    return y


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: List[List[int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of Multi-Scale Deformable Attention matching Ultralytics utils.py.
    Args:
        value: [bs, len_v, num_heads, embed_dims]
        value_spatial_shapes: list of [H_l, W_l] for l in 0..num_levels-1
        sampling_locations: [bs, num_queries, num_heads, num_levels, num_points, 2] in [0, 1]
        attention_weights: [bs, num_queries, num_heads, num_levels, num_points]
    Returns:
        output: [bs, num_queries, num_heads * embed_dims]
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([h_ * w_ for h_, w_ in value_spatial_shapes], dim=1)
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampling_value_list = []

    for level, (h_, w_) in enumerate(value_spatial_shapes):
        # [bs, H_*W_, num_heads, embed_dims] -> [bs*num_heads, embed_dims, H_, W_]
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, h_, w_)
        )
        # [bs, num_queries, num_heads, num_points, 2] -> [bs*num_heads, num_queries, num_points, 2]
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # Run grid_sample in FP32 if needed for numerical precision, then cast back
        sampling_value_l_ = F.grid_sample(
            value_l_.float(),
            sampling_grid_l_.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).to(dtype=value.dtype)
        sampling_value_list.append(sampling_value_l_)

    # [bs, num_queries, num_heads, num_levels, num_points] -> [bs*num_heads, 1, num_queries, num_levels*num_points]
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def get_cdn_group_uav(
    batch: Optional[Dict[str, torch.Tensor]],
    num_classes: int,
    num_queries: int,
    class_embed: torch.Tensor,
    num_dn: int = 100,
    cls_noise_ratio: float = 0.5,
    box_noise_scale: float = 1.0,
    training: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict]]:
    """
    Contrastive Denoising (CDN) query group generator compatible with Ultralytics `get_cdn_group`.
    Handles empty GT, single GT, and multiple GT across arbitrary batch sizes.
    """
    if (not training) or num_dn <= 0 or batch is None:
        return None, None, None, None

    gt_groups = batch.get("gt_groups", None)
    if gt_groups is None or len(gt_groups) == 0:
        return None, None, None, None

    total_num = sum(gt_groups)
    max_nums = max(gt_groups)
    if max_nums == 0 or total_num == 0:
        return None, None, None, None

    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group

    bs = len(gt_groups)
    gt_cls = batch["cls"].view(-1)          # [total_num]
    gt_bbox = batch["bboxes"]      # [total_num, 4]
    b_idx = batch["batch_idx"].view(-1)     # [total_num]
    device = gt_bbox.device

    dn_cls = gt_cls.repeat(2 * num_group)            # [2*num_group*total_num]
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)       # [2*num_group*total_num, 4]
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # [2*num_group*total_num]

    neg_idx = (
        torch.arange(total_num * num_group, dtype=torch.long, device=device)
        + num_group * total_num
    )

    if cls_noise_ratio > 0:
        mask = torch.rand(dn_cls.shape, device=device) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        if idx.numel() > 0:
            new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=device)
            dn_cls[idx] = new_label

    if box_noise_scale > 0:
        known_bbox = xywh2xyxy(dn_bbox)
        diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0
        rand_part *= rand_sign
        known_bbox = (known_bbox + rand_part * diff).clamp(min=0.0, max=1.0)
        dn_bbox = xyxy2xywh(known_bbox)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)

    num_dn_total = int(max_nums * 2 * num_group)
    dn_cls_embed = class_embed[dn_cls]  # [2*num_group*total_num, hd]
    padding_cls = torch.zeros(bs, num_dn_total, dn_cls_embed.shape[-1], device=device, dtype=class_embed.dtype)
    padding_bbox = torch.zeros(bs, num_dn_total, 4, device=device, dtype=dn_bbox.dtype)

    map_indices = torch.cat([torch.arange(num, dtype=torch.long, device=device) for num in gt_groups])
    pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)
    map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])

    padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
    padding_bbox[(dn_b_idx, map_indices)] = dn_bbox.to(dtype=padding_bbox.dtype)

    tgt_size = num_dn_total + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool, device=device)
    # Matching queries cannot see denoising queries
    attn_mask[num_dn_total:, :num_dn_total] = True
    # Denoising groups cannot see each other
    for i in range(num_group):
        start_i = max_nums * 2 * i
        end_i = max_nums * 2 * (i + 1)
        if i == 0:
            attn_mask[start_i:end_i, end_i:num_dn_total] = True
        elif i == num_group - 1:
            attn_mask[start_i:end_i, :start_i] = True
        else:
            attn_mask[start_i:end_i, end_i:num_dn_total] = True
            attn_mask[start_i:end_i, :start_i] = True

    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
        "dn_num_group": num_group,
        "dn_num_split": [num_dn_total, num_queries],
    }
    return padding_cls, padding_bbox, attn_mask, dn_meta


_get_cdn_group_fallback = get_cdn_group_uav


class MLP(nn.Module):
    """Standard configurable MLP matching Ultralytics `ultralytics.nn.modules.transformer.MLP`."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act: nn.Module = nn.ReLU,
        sigmoid: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid = sigmoid
        self.act = act() if isinstance(act, type) else act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.sigmoid() if self.sigmoid else x


class VarianceFlooredMSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention with Scale- & Level-Adaptive Context Variance Floor.
    Parameter names (`sampling_offsets`, `attention_weights`, `value_proj`, `output_proj`)
    match `ultralytics.nn.modules.transformer.MSDeformAttn` 1-to-1 so pretrained checkpoints
    load seamlessly.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 3,
        n_heads: int = 8,
        n_points: int = 4,
        eta_max: float = 2.5,
        init_eta_grid: float = 1.6,
        variance_floor_enabled: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}")
        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.eta_max = float(eta_max)
        self.variance_floor_enabled = bool(variance_floor_enabled)

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # Level-specific learnable variance floor parameters u_eta_l (shape [n_levels])
        ratio = min(max(init_eta_grid / max(eta_max, 1e-4), 1e-3), 1.0 - 1e-3)
        init_u = math.log(ratio / (1.0 - ratio))
        self.u_eta = nn.Parameter(torch.full((n_levels,), init_u, dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def get_effective_box_span(
        self,
        refer_wh: torch.Tensor,
        value_shapes: List[List[int]],
    ) -> torch.Tensor:
        """
        Computes variance-floored effective sampling span s_eff^(l) for each level l.
        Args:
            refer_wh: [bs, len_q, 1, 2] (w_r, h_r) in [0, 1]
            value_shapes: [[H_0, W_0], ..., [H_{L-1}, W_{L-1}]]
        Returns:
            s_eff: [bs, len_q, 1, n_levels, 1, 2]
        """
        # refer_wh_exp: [bs, len_q, 1, 1, 1, 2]
        refer_wh_exp = refer_wh[:, :, None, :, None, :]
        if not self.variance_floor_enabled:
            return refer_wh_exp

        # grid_step: [n_levels, 2] as (1/W_l, 1/H_l)
        hw = torch.as_tensor(value_shapes, dtype=torch.float32, device=refer_wh.device)  # [L, 2] (H_l, W_l)
        wh_step = (1.0 / hw.flip(-1).clamp(min=1.0))  # [L, 2] (1/W_l, 1/H_l)
        eta_l = (self.eta_max * torch.sigmoid(self.u_eta)).view(self.n_levels, 1)  # [L, 1]
        floor_span = (eta_l * wh_step).view(1, 1, 1, self.n_levels, 1, 2).to(dtype=refer_wh.dtype)

        s_eff = torch.sqrt(refer_wh_exp.pow(2) + floor_span.pow(2) + 1e-8)
        return s_eff

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: List[List[int]],
        value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [bs, len_q, C]
            refer_bbox: [bs, len_q, 1, 2 or 4] in [0, 1]
            value: [bs, len_v, C]
            value_shapes: [[H_0, W_0], ..., [H_{L-1}, W_{L-1}]]
            value_mask: optional [bs, len_v]
        Returns:
            output: [bs, len_q, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points
        )

        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(
                value_shapes, dtype=query.dtype, device=query.device
            ).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            s_eff = self.get_effective_box_span(refer_bbox[..., 2:], value_shapes)
            add = (sampling_offsets / self.n_points) * s_eff * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, got {num_points}.")

        output = multi_scale_deformable_attn_pytorch(
            value, value_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)


class UADDeformableDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer equipped with VarianceFlooredMSDeformAttn
    and zero-initialized Target-Decoupled Cross-Attention Gating.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 3,
        n_points: int = 4,
        eta_max: float = 2.5,
        init_eta_grid: float = 1.6,
        enabled: bool = True,
    ):
        super().__init__()
        self.enabled = bool(enabled)

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention with Context Variance Floor
        self.cross_attn = VarianceFlooredMSDeformAttn(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
            eta_max=eta_max,
            init_eta_grid=init_eta_grid,
            variance_floor_enabled=enabled,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Zero-initialized Target-Decoupled Gating residual modulation (inspired by D-FINE Eq. 7)
        self.target_gate = nn.Linear(d_model * 2, d_model)
        nn.init.constant_(self.target_gate.weight, 0.0)
        nn.init.constant_(self.target_gate.bias, 0.0)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: List[List[int]],
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            embed.transpose(0, 1),
            attn_mask=attn_mask,
        )[0].transpose(0, 1)
        embed = self.norm1(embed + self.dropout1(tgt))

        # 2. Cross attention with Variance Floor + Target-Decoupled Gate
        tgt_cross = self.cross_attn(
            self.with_pos_embed(embed, query_pos),
            refer_bbox.unsqueeze(2) if refer_bbox.dim() == 3 else refer_bbox,
            feats,
            shapes,
            padding_mask,
        )
        if self.enabled:
            gate = torch.tanh(self.target_gate(torch.cat([embed, tgt_cross], dim=-1)))
            embed = self.norm2(embed + self.dropout2(tgt_cross * (1.0 + 0.2 * gate)))
        else:
            embed = self.norm2(embed + self.dropout2(tgt_cross))

        # 3. FFN
        return self.forward_ffn(embed)


class UADDeformableDecoder(nn.Module):
    """
    Iterative Deformable Transformer Decoder with Uncertainty-Damped Box Refinement.
    """

    def __init__(
        self,
        hidden_dim: int,
        decoder_layer: nn.Module,
        num_layers: int,
        eval_idx: int = -1,
        enabled: bool = True,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.enabled = bool(enabled)

        # Lightweight uncertainty-damped edge refinement heads (zero-initialized)
        self.unc_damping_heads = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(num_layers)])
        for head in self.unc_damping_heads:
            nn.init.constant_(head.weight, 0.0)
            nn.init.constant_(head.bias, 0.0)

    def set_enabled(self, flag: bool):
        self.enabled = bool(flag)
        for layer in self.layers:
            layer.enabled = bool(flag)
            layer.cross_attn.variance_floor_enabled = bool(flag)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: List[List[int]],
        bbox_head: nn.ModuleList,
        score_head: nn.ModuleList,
        pos_mlp: nn.Module,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()

        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                refer_bbox,
                feats,
                shapes,
                padding_mask,
                attn_mask,
                pos_mlp(refer_bbox),
            )
            bbox_delta = bbox_head[i](output)
            if self.enabled:
                unc_mod = torch.tanh(self.unc_damping_heads[i](output))
                bbox_delta = bbox_delta * (1.0 + 0.15 * unc_mod)

            refined_bbox = torch.sigmoid(bbox_delta + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox_delta + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class UADCRDecoder(nn.Module):
    """
    Drop-in replacement for Ultralytics `RTDETRDecoder` implementing Innovation 3 (UAD-CR).
    Preserves all attribute names (`nc`, `ch`, `hd`, `num_queries`, `input_proj`, `enc_output`,
    `enc_score_head`, `enc_bbox_head`, `decoder`, `dec_score_head`, `dec_bbox_head`,
    `query_pos_head`, `denoising_class_embed`, `export`) and return signatures:
    - Training mode: returns `(dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)`
    - Inference mode: returns `y` if `self.export` else `(y, x)` where `y` is `[B, 300, 4 + nc]`
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
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        eta_max: float = 2.5,
        init_eta_grid: float = 1.6,
        lambda_q: float = 0.35,
        enabled: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # number of levels (3 for P3, P4, P5)
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl
        self.lambda_q = float(lambda_q)
        self.enabled = bool(enabled)
        self.stride = torch.tensor([8.0, 16.0, 32.0][: self.nl])

        # Backbone feature projection
        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch
        )

        # Transformer module
        decoder_layer = UADDeformableDecoderLayer(
            d_model=hd,
            n_heads=nh,
            d_ffn=d_ffn,
            dropout=dropout,
            act=act,
            n_levels=self.nl,
            n_points=ndp,
            eta_max=eta_max,
            init_eta_grid=init_eta_grid,
            enabled=enabled,
        )
        self.decoder = UADDeformableDecoder(
            hidden_dim=hd,
            decoder_layer=decoder_layer,
            num_layers=ndl,
            eval_idx=eval_idx,
            enabled=enabled,
        )

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Innovation 3: Encoder Localization Quality / Objectness Prior Head (zero-initialized)
        self.enc_quality_head = nn.Linear(hd, 1)
        nn.init.constant_(self.enc_quality_head.weight, 0.0)
        nn.init.constant_(self.enc_quality_head.bias, 0.0)

        # Learnable scale adapter on base grid_size (zero-initialized so step 0 == 0.05)
        self.anchor_scale_delta = nn.Parameter(torch.zeros((self.nl, 2), dtype=torch.float32))

        # Decoder heads
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def set_enabled(self, flag: bool):
        """Toggle Innovation 3 on/off dynamically."""
        self.enabled = bool(flag)
        self.decoder.set_enabled(flag)

    def _reset_parameters(self):
        bias_cls = float(-math.log((1 - 0.01) / 0.01))
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

    def _get_encoder_input(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, List[List[int]]]:
        # x: list of [B, C_i, H_i, W_i] -> projected to [B, hd, H_i, W_i]
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        feats = []
        shapes = []
        for feat in proj_feats:
            h, w = feat.shape[2:]
            feats.append(feat.flatten(2).permute(0, 2, 1))  # [B, H*W, hd]
            shapes.append([h, w])
        feats = torch.cat(feats, dim=1)  # [B, 8400, hd]
        return feats, shapes

    def _generate_anchors(
        self,
        shapes: List[List[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        eps: float = 1e-2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], -1)  # [h, w, 2]
            valid_wh = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh  # [1, h, w, 2]
            base_scale = grid_size * (2.0 ** i)
            if self.enabled:
                # Smooth bounded scale adaptation (starts at 1.0 when anchor_scale_delta == 0)
                scale_factor = torch.exp(0.5 * torch.tanh(self.anchor_scale_delta[i])).to(dtype=dtype)
                wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * base_scale * scale_factor
            else:
                wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * base_scale
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))

        anchors = torch.cat(anchors, 1)  # [1, 8400, 4]
        valid_mask = ((anchors > eps) * (anchors < 1.0 - eps)).all(-1, keepdim=True)  # [1, 8400, 1]
        anchors = torch.log(anchors / (1.0 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: List[List[int]],
        dn_embed: Optional[torch.Tensor] = None,
        dn_bbox: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = feats.shape[0]
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # [bs, 8400, hd]
        enc_outputs_scores = self.enc_score_head(features)  # [bs, 8400, nc]

        # Query selection (Top-300)
        if self.enabled and self.lambda_q > 0.0:
            # Uncertainty-Minimal Quality-Calibrated Selection Score
            cls_prob = enc_outputs_scores.max(-1).values.sigmoid()  # [bs, 8400]
            qual_prob = self.enc_quality_head(features).squeeze(-1).sigmoid()  # [bs, 8400]
            uncertainty_gap = (cls_prob - qual_prob.detach()).abs()
            select_metric = cls_prob * (1.0 + self.lambda_q * qual_prob * torch.exp(-uncertainty_gap))
            # Preserve exact tie-breaking with raw logits when lambda_q * qual_prob is uniform
            topk_ind = torch.topk(select_metric, self.num_queries, dim=1).indices.view(-1)
        else:
            topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)

        batch_ind = (
            torch.arange(end=bs, dtype=topk_ind.dtype, device=feats.device)
            .unsqueeze(-1)
            .repeat(1, self.num_queries)
            .view(-1)
        )

        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)  # [bs, 300, hd]
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)           # [bs, 300, 4]

        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors  # [bs, 300, 4] (unactivated logits)
        enc_bboxes = refer_bbox.sigmoid()                                # [bs, 300, 4] in (0, 1)

        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], dim=1)

        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)  # [bs, 300, nc]

        embeddings = (
            self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            if self.learnt_init_query
            else top_k_features
        )
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()

        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], dim=1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def forward(
        self,
        x: List[torch.Tensor],
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[Tuple, torch.Tensor]:
        """
        Args:
            x: list of 3 feature maps [P3, P4, P5] with shapes:
               P3: [B, C3, 80, 80], P4: [B, C4, 40, 40], P5: [B, C5, 20, 20]
            batch: optional ground-truth dictionary during training
        Returns:
            In training: (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)
            In eval: (y, x) where y has shape [B, 300, 4 + nc]
        """
        feats, shapes = self._get_encoder_input(x)

        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group_uav(
            batch=batch,
            num_classes=self.nc,
            num_queries=self.num_queries,
            class_embed=self.denoising_class_embed.weight,
            num_dn=self.num_denoising,
            cls_noise_ratio=self.label_noise_ratio,
            box_noise_scale=self.box_noise_scale,
            training=self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(
            feats, shapes, dn_embed, dn_bbox
        )

        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        out_tuple = (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)
        if self.training:
            return out_tuple

        # Inference output contract: [B, 300, 4 + nc]
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), dim=-1)
        return y if self.export else (y, out_tuple)
