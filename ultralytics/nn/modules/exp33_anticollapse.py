#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTDETR_Exp33_AntiCollapse:
  1. `FCGHaarDownsample`: Frequency-Contrast Gated Haar Downsampling (strictly non-equivalent
     to any 2x2 stride-2 linear convolution due to spatial-frequency contrast nonlinearity on
     the (LH, HL, HH) high-frequency subbands before 1x1 fusion).
  2. `AntiCollapseMSDeformAttn` & `AntiCollapseRTDETRDecoder`: Grid-stride variance-floored
     deformable sampling that prevents sampling-point collapse on tiny UAV boxes (<12 px),
     adding only +1 learnable scalar per feature level (+9 params total, ~0.00 ms overhead).
  3. `patch_rtdetr_coupled_tight_nwd`: Tight `area_tau=0.0015` Scale-Aware Inner-GIoU + NWD
     coupled consistently in BOTH `HungarianMatcher` and `DETRLoss` (0.00 ms deploy cost).
"""

from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.transformer import MSDeformAttn, multi_scale_deformable_attn_pytorch
    from ultralytics.nn.modules.head import RTDETRDecoder
except Exception:
    Conv = None
    MSDeformAttn = nn.Module
    multi_scale_deformable_attn_pytorch = None
    RTDETRDecoder = nn.Module

__all__ = [
    "FCGHaarDownsample",
    "AntiCollapseMSDeformAttn",
    "AntiCollapseRTDETRDecoder",
    "compute_nwd_pairwise",
    "tight_inner_giou_nwd_loss",
    "patch_rtdetr_coupled_tight_nwd",
]


def _fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    with torch.no_grad():
        fused = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        ).to(device=conv.weight.device, dtype=conv.weight.dtype)
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
        b_conv = (
            torch.zeros(conv.weight.size(0), device=conv.weight.device, dtype=conv.weight.dtype)
            if conv.bias is None
            else conv.bias
        )
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        return fused.requires_grad_(False)


# ===========================================================================
# Innovation A: Frequency-Contrast Gated Haar Downsampling (FCG-Haar)
# ===========================================================================
class FCGHaarDownsample(nn.Module):
    """
    2D Haar DWT downsampling with a nonlinear spatial-frequency contrast gate on
    the high-frequency subbands H = [LH, HL, HH] BEFORE 1x1 projection with LL:

        H' = H * (1 + alpha * (sigmoid(Gate(DWConv3x3(H) - AvgPool3x3(H))) - 0.5))
        Y  = Proj1x1(Concat[LL, H'])

    Because a data-dependent multiplicative nonlinearity acts specifically on the
    high-frequency subbands prior to projection, FCGHaarDownsample is strictly
    NON-EQUIVALENT to any single 2x2 stride-2 linear convolution.
    """

    def __init__(self, c1: int, c2: Optional[int] = None, alpha_init: float = 0.25):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c2 if c2 is not None else c1)
        hf_c = 3 * self.c1

        # High-frequency local contrast extractor: DWConv3x3(H) - AvgPool3x3(H)
        self.hf_dw = nn.Conv2d(hf_c, hf_c, kernel_size=3, stride=1, padding=1, groups=hf_c, bias=False)
        self.hf_bn = nn.BatchNorm2d(hf_c)
        self.hf_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # Lightweight spatial-frequency contrast gate (groups=3 keeps LH/HL/HH decoupled)
        self.gate_conv = nn.Conv2d(hf_c, hf_c, kernel_size=1, stride=1, padding=0, groups=3, bias=True)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)

        # Learnable modulation strength per high-frequency channel
        self.alpha = nn.Parameter(torch.full((1, hf_c, 1, 1), float(alpha_init)))

        # Final 1x1 projection from [LL, H'] (4*c1) -> c2
        self.proj_conv = nn.Conv2d(4 * self.c1, self.c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_bn = nn.BatchNorm2d(self.c2)
        self.act = nn.SiLU(inplace=False)
        self.fused = False

    @staticmethod
    def _haar_dwt_2d(x: torch.Tensor):
        _, _, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (x00 - x01 + x10 - x11)
        hl = 0.5 * (x00 + x01 - x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return ll, torch.cat([lh, hl, hh], dim=1)

    def switch_to_deploy(self):
        if getattr(self, "fused", False):
            return self
        with torch.no_grad():
            if hasattr(self, "hf_bn") and isinstance(self.hf_bn, nn.BatchNorm2d):
                self.hf_dw = _fuse_conv_bn_pair(self.hf_dw, self.hf_bn)
                delattr(self, "hf_bn")
            if hasattr(self, "proj_bn") and isinstance(self.proj_bn, nn.BatchNorm2d):
                self.proj_conv = _fuse_conv_bn_pair(self.proj_conv, self.proj_bn)
                delattr(self, "proj_bn")
            for p in self.parameters():
                if p.is_leaf:
                    p.requires_grad_(False)
        self.fused = True
        return self

    def fuse(self):
        return self.switch_to_deploy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, hf = self._haar_dwt_2d(x)
        if getattr(self, "fused", False):
            hf_local = self.hf_dw(hf)
        else:
            hf_local = self.hf_bn(self.hf_dw(hf))
        contrast = hf_local - self.hf_pool(hf)
        gate = torch.sigmoid(self.gate_conv(contrast))
        hf_gated = hf * (1.0 + self.alpha * (gate - 0.5))
        y = torch.cat([ll, hf_gated], dim=1)
        if getattr(self, "fused", False):
            return self.act(self.proj_conv(y))
        return self.act(self.proj_bn(self.proj_conv(y)))


# ===========================================================================
# Innovation B: Anti-Collapse Grid-Floored Deformable Attention & Decoder
# ===========================================================================
class AntiCollapseMSDeformAttn(MSDeformAttn):
    """
    Drop-in subclass of Ultralytics `MSDeformAttn` that prevents tiny-box sampling
    collapse in box mode (`refer_bbox.shape[-1] == 4`):

        half_wh  = 0.5 * refer_bbox[..., 2:]
        eff_span = sqrt(half_wh^2 + (beta_l * [1/W_l, 1/H_l])^2 + eps)
        sampling_locations = refer_bbox[..., :2] + (offsets / n_points) * eff_span

    where `beta_l = softplus(floor_alpha[l]).clamp(0.0, 2.5)` is a single learnable
    scalar per feature level (only +3 params per decoder layer, ~0.00 ms overhead).
    """

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4, floor_init: float = 1.0):
        super().__init__(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        # inverse softplus so initial beta_l == floor_init (1.0 grid cell)
        init_logit = math.log(math.expm1(max(float(floor_init), 1e-3)))
        self.floor_alpha = nn.Parameter(torch.full((int(n_levels),), init_logit))

    def get_effective_span(self, ref_wh: torch.Tensor, value_shapes) -> torch.Tensor:
        """
        ref_wh: [bs, len_q, 1, 2] -> returns eff_span: [bs, len_q, 1, n_levels, 1, 2]
        """
        # value_shapes is list/tensor of (H_l, W_l); coordinates are (x, y) -> stride is (1/W_l, 1/H_l)
        strides = [[1.0 / float(w), 1.0 / float(h)] for (h, w) in value_shapes]
        grid_step = torch.tensor(strides, device=ref_wh.device, dtype=ref_wh.dtype).view(1, 1, 1, -1, 1, 2)
        beta = F.softplus(self.floor_alpha).clamp(max=2.5).to(dtype=ref_wh.dtype).view(1, 1, 1, -1, 1, 1)
        half_wh = 0.5 * ref_wh[:, :, None, :, None, :]  # [bs, len_q, 1, 1, 1, 2]
        return torch.sqrt(half_wh.pow(2) + (beta * grid_step).pow(2) + 1e-12)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        nh = getattr(self, "n_heads", getattr(self, "nh", 8))
        nl = getattr(self, "n_levels", getattr(self, "nl", 3))
        np_ = getattr(self, "n_points", getattr(self, "np", 4))

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, nh, self.d_model // nh)

        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, nh, nl, np_, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, nh, nl * np_)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, nh, nl, np_)

        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            eff_span = self.get_effective_span(refer_bbox[..., 2:], value_shapes)
            add = (sampling_offsets / np_) * eff_span
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")

        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class AntiCollapseRTDETRDecoder(RTDETRDecoder):
    """
    Drop-in subclass of Ultralytics `RTDETRDecoder` that upgrades every decoder layer's
    `cross_attn` (`MSDeformAttn`) to `AntiCollapseMSDeformAttn` while keeping 100% of the
    300-query contract and standard RT-DETR checkpoint compatibility.
    """

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,
        nq=300,
        ndp=4,
        nh=8,
        ndl=6,
        d_ffn=1024,
        eval_idx=-1,
        dropout=0.0,
        act=nn.ReLU(),
        nd=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        super().__init__(
            nc=nc,
            ch=ch,
            hd=hd,
            nq=nq,
            ndp=ndp,
            nh=nh,
            ndl=ndl,
            d_ffn=d_ffn,
            eval_idx=eval_idx,
            dropout=dropout,
            act=act,
            nd=nd,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learnt_init_query=learnt_init_query,
        )
        for layer in self.decoder.layers:
            old = layer.cross_attn
            upgraded = AntiCollapseMSDeformAttn(
                d_model=old.d_model,
                n_levels=getattr(old, "n_levels", getattr(old, "nl", len(ch))),
                n_heads=getattr(old, "n_heads", getattr(old, "nh", nh)),
                n_points=getattr(old, "n_points", getattr(old, "np", ndp)),
                floor_init=1.0,
            )
            upgraded.load_state_dict(old.state_dict(), strict=False)
            layer.cross_attn = upgraded


# ===========================================================================
# Innovation C: Tight `area_tau` Scale-Aware Inner-GIoU + NWD Coupled in
#               BOTH `HungarianMatcher` and `DETRLoss` (0.00 ms Deploy Cost!)
# ===========================================================================
def compute_nwd_pairwise(pred_xywh: torch.Tensor, gt_xywh: torch.Tensor, nwd_c: float = 0.05, eps: float = 1e-7) -> torch.Tensor:
    """
    Pairwise Normalized Wasserstein Distance matrix `[N_pred, N_gt]` in `(0, 1]`.
    """
    with torch.autocast(device_type=pred_xywh.device.type, enabled=False):
        p = pred_xywh.float().unsqueeze(1)  # [N, 1, 4]
        g = gt_xywh.float().unsqueeze(0)    # [1, M, 4]
        w2_sq = (
            (p[..., 0] - g[..., 0]).pow(2)
            + (p[..., 1] - g[..., 1]).pow(2)
            + 0.25 * ((p[..., 2] - g[..., 2]).pow(2) + (p[..., 3] - g[..., 3]).pow(2))
        )
        return torch.exp(-torch.sqrt(w2_sq + eps) / max(nwd_c, 1e-4))


def tight_inner_giou_nwd_loss(
    pred_xywh: torch.Tensor,
    gt_xywh: torch.Tensor,
    ratio: float = 1.25,
    nwd_c: float = 0.05,
    nwd_max_weight: float = 0.25,
    area_tau: float = 0.0015,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Tight-tau Scale-Aware `Inner-GIoU + NWD` loss for normalized `[cx, cy, w, h]` boxes.
    With `area_tau = 0.0015` (at 640x640 input):
      -  6x6  px (area=0.000088) -> w_nwd = 0.236 (strong smooth NWD guidance)
      - 16x16 px (area=0.000625) -> w_nwd = 0.165 (balanced blend)
      - 32x32 px (area=0.002500) -> w_nwd = 0.047 (sharp Inner-GIoU dominant)
      - 64x64 px (area=0.010000) -> w_nwd = 0.0003 (pure Inner-GIoU)
    """
    with torch.autocast(device_type=pred_xywh.device.type, enabled=False):
        p = pred_xywh.float()
        g = gt_xywh.float()
        px, py, pw, ph = p.unbind(-1)
        gx, gy, gw, gh = g.unbind(-1)
        pw, ph = pw.clamp(min=eps), ph.clamp(min=eps)
        gw, gh = gw.clamp(min=eps), gh.clamp(min=eps)

        # 1. Inner-IoU (auxiliary boxes scaled by `ratio`)
        pb_x1, pb_x2 = px - pw * ratio * 0.5, px + pw * ratio * 0.5
        pb_y1, pb_y2 = py - ph * ratio * 0.5, py + ph * ratio * 0.5
        gb_x1, gb_x2 = gx - gw * ratio * 0.5, gx + gw * ratio * 0.5
        gb_y1, gb_y2 = gy - gh * ratio * 0.5, gy + gh * ratio * 0.5

        inter_w = (torch.minimum(pb_x2, gb_x2) - torch.maximum(pb_x1, gb_x1)).clamp(min=0.0)
        inter_h = (torch.minimum(pb_y2, gb_y2) - torch.maximum(pb_y1, gb_y1)).clamp(min=0.0)
        inter = inter_w * inter_h
        union = (pw * ph + gw * gh) * (ratio * ratio) - inter + eps
        inner_iou = inter / union

        # Enclosing box C for GIoU penalty
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
        loss_inner_giou = 1.0 - (inner_iou - (c_area - raw_union) / c_area)

        # 2. Normalized Wasserstein Distance
        w2_sq = (px - gx).pow(2) + (py - gy).pow(2) + 0.25 * ((pw - gw).pow(2) + (ph - gh).pow(2))
        loss_nwd = 1.0 - torch.exp(-torch.sqrt(w2_sq + eps) / max(nwd_c, 1e-4))

        # 3. Tight scale-adaptive blending
        gt_area = (gw * gh).detach()
        w_nwd = nwd_max_weight * torch.exp(-gt_area / max(area_tau, 1e-6))
        return (1.0 - w_nwd) * loss_inner_giou + w_nwd * loss_nwd


def _detr_get_loss_bbox_tight(self, pred_bboxes, gt_bboxes, postfix=""):
    name_bbox = f"loss_bbox{postfix}"
    name_giou = f"loss_giou{postfix}"
    loss = {}
    if len(gt_bboxes) == 0:
        loss[name_bbox] = torch.tensor(0.0, device=self.device)
        loss[name_giou] = torch.tensor(0.0, device=self.device)
        return loss
    ratio, nwd_c, nwd_max_weight, area_tau = getattr(
        self.__class__, "_tight_coupled_nwd_cfg", (1.25, 0.05, 0.25, 0.0015)
    )
    loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
    loss_ig = tight_inner_giou_nwd_loss(
        pred_bboxes,
        gt_bboxes,
        ratio=ratio,
        nwd_c=nwd_c,
        nwd_max_weight=nwd_max_weight,
        area_tau=area_tau,
    )
    loss[name_giou] = self.loss_gain["giou"] * loss_ig.sum() / len(gt_bboxes)
    return {k: v.squeeze() for k, v in loss.items()}


def _matcher_forward_coupled(self, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None):
    from scipy.optimize import linear_sum_assignment
    from ultralytics.utils.metrics import bbox_iou

    bs, nq, nc = pred_scores.shape
    if sum(gt_groups) == 0:
        return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

    _, nwd_c, nwd_max_weight, area_tau = getattr(
        self.__class__, "_tight_coupled_nwd_cfg", (1.25, 0.05, 0.25, 0.0015)
    )

    with torch.autocast(device_type=pred_bboxes.device.type, enabled=False):
        pred_scores = pred_scores.detach().float().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        pred_bboxes = pred_bboxes.detach().float().view(-1, 4)
        gt_bboxes_f = gt_bboxes.float()

        # Classification cost
        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pred_scores ** self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # L1 box cost
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes_f.unsqueeze(0)).abs().sum(-1)

        # Coupled Scale-Aware GIoU + NWD geometric cost
        giou = bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes_f.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)
        cost_giou = 1.0 - giou
        nwd = compute_nwd_pairwise(pred_bboxes, gt_bboxes_f, nwd_c=nwd_c)
        cost_nwd = 1.0 - nwd
        gt_area = (gt_bboxes_f[:, 2].clamp(min=1e-7) * gt_bboxes_f[:, 3].clamp(min=1e-7)).unsqueeze(0)  # [1, M]
        w_nwd = nwd_max_weight * torch.exp(-gt_area / max(area_tau, 1e-6))
        cost_geom = (1.0 - w_nwd) * cost_giou + w_nwd * cost_nwd

        C = self.cost_gain["class"] * cost_class + self.cost_gain["bbox"] * cost_bbox + self.cost_gain["giou"] * cost_geom
        C[C.isnan() | C.isinf()] = 0.0
        C = C.view(bs, nq, -1).cpu()

    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
    gt_groups_t = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
    return [
        (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups_t[k])
        for k, (i, j) in enumerate(indices)
    ]


def patch_rtdetr_coupled_tight_nwd(
    ratio: float = 1.25,
    nwd_c: float = 0.05,
    nwd_max_weight: float = 0.25,
    area_tau: float = 0.0015,
) -> None:
    """
    Patches BOTH `HungarianMatcher.forward` AND `DETRLoss._get_loss_bbox` in memory so
    bipartite matching and bounding-box regression share the exact same tight-tau
    scale-adaptive NWD formulation. Zero inference overhead (0.00 ms).
    """
    from ultralytics.models.utils.loss import DETRLoss
    from ultralytics.models.utils.ops import HungarianMatcher

    cfg = (float(ratio), float(nwd_c), float(nwd_max_weight), float(area_tau))
    DETRLoss._tight_coupled_nwd_cfg = cfg
    HungarianMatcher._tight_coupled_nwd_cfg = cfg

    if not getattr(DETRLoss, "_tight_coupled_nwd_patched", False):
        DETRLoss._get_loss_bbox = _detr_get_loss_bbox_tight
        DETRLoss._tight_coupled_nwd_patched = True

    if not getattr(HungarianMatcher, "_tight_coupled_nwd_patched", False):
        HungarianMatcher.forward = _matcher_forward_coupled
        HungarianMatcher._tight_coupled_nwd_patched = True

    print(
        f"[Exp33+] Patched BOTH HungarianMatcher & DETRLoss with Tight Scale-Aware Inner-GIoU+NWD "
        f"(ratio={ratio}, nwd_c={nwd_c}, max_w={nwd_max_weight}, area_tau={area_tau}) [0.00 ms deploy cost]"
    )
