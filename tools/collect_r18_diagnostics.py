#!/usr/bin/env python3
"""RT-DETR-R18 诊断脚本 (collect_r18_diagnostics).

在 Linux 服务器上收集三个模型 (E0 baseline / E1 BAPS / E2 RDCF) 的完整诊断数据，
用于分析 BAPS / RDCF 为何未提升 mAP。产出一个 ``r18_diagnostics.zip`` 供下载后离线分析。

用法示例
---------
基本结构审计 + 少量特征采集 (basic 模式, 不跑完整测试集)::

    python tools/collect_r18_diagnostics.py \\
        --baseline-weights runs/rtdetr-r18/train/weights/best.pt \\
        --baps-weights     runs/rtdetr-r18-baps/train/weights/best.pt \\
        --rdcf-weights     runs/rtdetr-r18-rdcf/train/weights/best.pt \\
        --data dataset/DUT-Anti-UAV.yaml --mode basic

完整分析 (full 模式, 含完整测试集预测 + 错误分析)::

    python tools/collect_r18_diagnostics.py \\
        --baseline-weights runs/rtdetr-r18/train/weights/best.pt \\
        --baps-weights     runs/rtdetr-r18-baps/train/weights/best.pt \\
        --rdcf-weights     runs/rtdetr-r18-rdcf/train/weights/best.pt \\
        --data dataset/DUT-Anti-UAV.yaml --mode full

设计要点
--------
* 每次只加载一个模型到 GPU, 用完即 ``del`` 并 ``torch.cuda.empty_cache()``, 控制显存占用。
* 全程 ``model.eval()`` + ``torch.inference_mode()``, 不保存完整特征图, 只在 hook 回调里
  就地计算统计量 (mean/std/L2 范数/余弦相似度/门控饱和比例/残差占比), 内存友好。
* 不修改任何网络结构, 仅通过 ``register_forward_hook`` 采集 BAPS/RDCF 内部行为。
* 复用项目自带的 ``RTDETRDataset`` / ``build_dataloader`` / ``attempt_load_one_weight``,
  保证预处理与后处理坐标换算和官方 val 完全一致 (stretch 到 imgsz 的正方形输入)。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import platform
import sys
import time
import traceback
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.nn.modules.baps_rdcf import BAPS, RDCF
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.torch_utils import select_device as torch_select_device
from ultralytics.data.utils import check_det_dataset

torch_Tensor = torch.Tensor
torch_is_tensor = torch.is_tensor
torch_zeros = torch.zeros
torch_softmax = torch.softmax
torch_sigmoid = torch.sigmoid
torch_argsort = torch.argsort
torch_inference_mode = torch.inference_mode

# --------------------------------------------------------------------------- #
# 日志: 同时输出到 stdout 与 execution_log.txt
# --------------------------------------------------------------------------- #
LOGGER = logging.getLogger("r18_diagnostics")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    _console = logging.StreamHandler(sys.stdout)
    _console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(_console)

_exec_log_entries = []


def log(msg: str) -> None:
    """打印并记录到 execution_log。"""
    LOGGER.info(msg)
    _exec_log_entries.append(msg)


def _file_logger(path: Path) -> None:
    """把日志落盘到 execution_log.txt。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_exec_log_entries) + "\n")


# --------------------------------------------------------------------------- #
# 数值 / 几何工具
# --------------------------------------------------------------------------- #
def _as_float(x):
    try:
        v = float(x)
        if math_is_nan(v) or math_is_inf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def math_is_nan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except TypeError:
        return False


def math_is_inf(v) -> bool:
    try:
        return bool(np.isinf(v))
    except TypeError:
        return False


def tensor_stats(t) -> dict:
    """单个 tensor 的 mean/std/L2 范数 (detach + float32)。"""
    x = t.detach().float()
    return {
        "mean": _as_float(x.mean().cpu()),
        "std": _as_float(x.std().cpu()),
        "l2norm": _as_float(x.norm(2).cpu()),
    }


def cosine(a, b) -> float:
    """展平后两 tensor 的余弦相似度。"""
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    denom = (a.norm(2) * b.norm(2))
    if float(denom) < 1e-12:
        return 0.0
    return _as_float((a @ b) / denom)


def agg_scalars(values) -> dict:
    """把标量列表聚合成统计量。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": _as_float(arr.mean()),
        "std": _as_float(arr.std()),
        "min": _as_float(arr.min()),
        "max": _as_float(arr.max()),
        "n": int(arr.size),
    }


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """两批 xyxy 框的 IoU 矩阵 (numpy)。boxes1[N,4], boxes2[M,4] -> [N,M]。"""
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    b1 = np.asarray(boxes1, dtype=np.float32)
    b2 = np.asarray(boxes2, dtype=np.float32)
    ix1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    iy1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    ix2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    iy2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou = np.zeros_like(inter)
    np.divide(inter, union, out=iou, where=union > 1e-12)
    return iou


def greedy_match(pred_boxes: np.ndarray, pred_cls: np.ndarray, gt_boxes: np.ndarray,
                 gt_cls: np.ndarray, iou_thr: float = 0.5):
    """class-aware 贪心一对一匹配。

    返回:
        tp_pred_idx (list[int]): 被匹配上的预测索引 (每 GT 一个, 或 None)
        tp_iou     (list[float]): 对应 IoU
        matched_gt (list[int]):   对应 GT 索引
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], [], []
    iou = box_iou_np(pred_boxes, gt_boxes)
    # class 掩码
    cls_ok = (np.asarray(pred_cls)[:, None] == np.asarray(gt_cls)[None, :])
    iou = np.where(cls_ok, iou, 0.0)
    # 按 IoU 降序贪心匹配
    used_p, used_g = set(), set()
    order = np.argsort(-iou, axis=None)
    tp_pred_idx, tp_iou, matched_gt = [], [], []
    for flat in order:
        p, g = np.unravel_index(flat, iou.shape)
        if iou[p, g] < iou_thr:
            continue
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        tp_pred_idx.append(int(p))
        tp_iou.append(float(iou[p, g]))
        matched_gt.append(int(g))
    return tp_pred_idx, tp_iou, matched_gt


def laplacian_variance(img_tensor) -> float:
    """输入单张 [3,H,W] float tensor (0-255), 计算灰度 Laplacian 方差作为模糊代理指标。"""
    try:
        import cv2
    except Exception:
        return float("nan")
    gray = img_tensor.float().mean(dim=0).cpu().numpy().astype(np.float32)
    try:
        return _as_float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# 结构审计 + 特征形状采集
# --------------------------------------------------------------------------- #
def safe_json(obj):
    """把任意对象转成可 JSON 序列化 (tensor/numpy/path 兜底)。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _as_float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch_Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def summarize_modules(model) -> list:
    """遍历 model.model (parse_model 产物), 汇总每层信息。"""
    rows = []
    for i, m in enumerate(model.model):
        f = getattr(m, "f", None)
        t = getattr(m, "type", None)
        np_ = getattr(m, "np", None)
        n_params = sum(p.numel() for p in m.parameters())
        rows.append({
            "index": i,
            "type": str(type(m).__name__),
            "from": f,
            "declared_type": t,
            "params": int(n_params),
            "parse_params": int(np_) if np_ is not None else None,
            "has_baps_rdcf": isinstance(m, (BAPS, RDCF)),
        })
    return rows


def collect_feature_shapes(model, device, imgsz) -> list:
    """用 dummy 输入前向一次, 记录每个顶层模块的输入/输出 shape。"""
    shapes = []
    handles = []

    def make_hook(idx, m):
        def hook_fn(_m, inp, out):
            in_shapes = []
            for x in inp:
                if torch_is_tensor(x):
                    in_shapes.append(list(x.shape))
                elif isinstance(x, (list, tuple)):
                    in_shapes.append([[list(xx.shape) for xx in x] if torch_is_tensor(x) else None for x in x])
                else:
                    in_shapes.append(None)
            out_shape = None
            if torch_is_tensor(out):
                out_shape = list(out.shape)
            elif isinstance(out, (list, tuple)):
                out_shape = [list(o.shape) if torch_is_tensor(o) else None for o in out]
            shapes.append({"index": idx, "type": str(type(m).__name__),
                           "input_shapes": in_shapes, "output_shapes": out_shape})
        return hook_fn

    for idx, m in enumerate(model.model):
        handles.append(m.register_forward_hook(make_hook(idx, m)))
    try:
        with torch_inference_mode():
            _ = model(torch_zeros(1, 3, imgsz, imgsz, device=device))
    except Exception as e:  # 前向失败不致命
        log(f"[warn] 特征形状采集前向失败: {e}")
    finally:
        for h in handles:
            h.remove()
    return shapes


# --------------------------------------------------------------------------- #
# BAPS / RDCF 行为采集器
# --------------------------------------------------------------------------- #
class _BAPSCollector:
    """hook BAPS 内部各子模块, 就地聚合统计并支持 GT 区域分析。"""

    def __init__(self, module, name, feat_size):
        self.module = module
        self.name = name
        self.feat_size = feat_size  # (H, W) of P3 feature map
        self.handles = []
        self.p3_stats, self.p2_stats, self.out_stats = [], [], []
        self.detail_stats, self.anti_stats = [], []
        self.detail_anti_cosine = []
        self.w_detail_mean, self.w_detail_std = [], []
        self.w_anti_mean, self.w_anti_std = [], []
        self.w_detail_near0, self.w_detail_near1 = [], []
        self.w_anti_near0, self.w_anti_near1 = [], []
        self.delta_norm, self.p3_norm = [], []
        self.delta_mean, self.delta_std = [], []
        self.logits_detail_mean, self.logits_anti_mean, self.logits_abs_mean = [], [], []
        # 响应图采样 (当前样本)
        self.pending = None
        self._last_detail = None
        self._last_anti = None
        # GT 区域统计累加
        self.region = {"w_detail": {"fg": [], "bg": []}, "delta_norm": {"fg": [], "bg": []}}

        self.handles.append(module.register_forward_hook(self._on_module))
        if hasattr(module, "gate"):
            self.handles.append(module.gate.register_forward_hook(self._on_gate))
        if hasattr(module, "delta"):
            self.handles.append(module.delta.register_forward_hook(self._on_delta))
        if hasattr(module, "detail_pw"):
            self.handles.append(module.detail_pw.register_forward_hook(self._on_detail))
        if hasattr(module, "anti_conv"):
            self.handles.append(module.anti_conv.register_forward_hook(self._on_anti))

    def _on_module(self, m, inp, out):
        p2, p3 = inp
        self.p2_stats.append(tensor_stats(p2))
        self.p3_stats.append(tensor_stats(p3))
        self.out_stats.append(tensor_stats(out))
        self.p3_norm.append(_as_float(p3.detach().float().norm(2).cpu()))

    def _on_gate(self, m, inp, out):
        logits = out.detach().float()  # [B,2,H,W]
        w = torch_softmax(logits, dim=1)  # [B,2,H,W]
        wd = w[:, 0]
        wa = w[:, 1]
        self.w_detail_mean.append(_as_float(wd.mean().cpu()))
        self.w_detail_std.append(_as_float(wd.std().cpu()))
        self.w_anti_mean.append(_as_float(wa.mean().cpu()))
        self.w_anti_std.append(_as_float(wa.std().cpu()))
        self.w_detail_near0.append(_as_float((wd < 0.05).float().mean().cpu()))
        self.w_detail_near1.append(_as_float((wd > 0.95).float().mean().cpu()))
        self.w_anti_near0.append(_as_float((wa < 0.05).float().mean().cpu()))
        self.w_anti_near1.append(_as_float((wa > 0.95).float().mean().cpu()))
        self.logits_detail_mean.append(_as_float(logits[:, 0].mean().cpu()))
        self.logits_anti_mean.append(_as_float(logits[:, 1].mean().cpu()))
        self.logits_abs_mean.append(_as_float(logits.abs().mean().cpu()))
        if self.pending is None:
            self.pending = {}
        self.pending["w_detail"] = wd[0].cpu().numpy().astype(np.float32)

    def _on_delta(self, m, inp, out):
        delta = out.detach().float()
        self.delta_mean.append(_as_float(delta.mean().cpu()))
        self.delta_std.append(_as_float(delta.std().cpu()))
        dn = delta.norm(2, dim=1)  # [B,H,W]
        self.delta_norm.append(_as_float(dn.mean().cpu()))
        if self.pending is None:
            self.pending = {}
        self.pending["delta_norm"] = dn[0].cpu().numpy().astype(np.float32)

    def _on_detail(self, m, inp, out):
        self.detail_stats.append(tensor_stats(out))
        self._last_detail = out.detach().float()

    def _on_anti(self, m, inp, out):
        self.anti_stats.append(tensor_stats(out))
        self._last_anti = out.detach().float()

    def finalize_sample(self, gt_boxes_feat):
        """在一张图 forward 完成后调用, 结算 cosine 与 GT 区域统计。"""
        if self._last_detail is not None and self._last_anti is not None:
            self.detail_anti_cosine.append(cosine(self._last_detail, self._last_anti))
        if self.pending is not None:
            for key in ("w_detail", "delta_norm"):
                if key in self.pending:
                    fg, bg = _region_stats(self.pending[key], gt_boxes_feat)
                    self.region[key]["fg"].append(fg)
                    self.region[key]["bg"].append(bg)
        self.pending = None

    def remove(self):
        for h in self.handles:
            h.remove()

    def to_dict(self):
        return {
            "name": self.name,
            "n_samples": len(self.p3_stats),
            "p3_input": _agg_tensor_stats(self.p3_stats),
            "p2_input": _agg_tensor_stats(self.p2_stats),
            "output": _agg_tensor_stats(self.out_stats),
            "f_detail": _agg_tensor_stats(self.detail_stats),
            "f_anti": _agg_tensor_stats(self.anti_stats),
            "detail_anti_cosine": agg_scalars(self.detail_anti_cosine),
            "gate_weights": {
                "detail": {"mean": agg_scalars(self.w_detail_mean), "std": agg_scalars(self.w_detail_std)},
                "anti": {"mean": agg_scalars(self.w_anti_mean), "std": agg_scalars(self.w_anti_std)},
                "detail_near_0_ratio": agg_scalars(self.w_detail_near0),
                "detail_near_1_ratio": agg_scalars(self.w_detail_near1),
                "anti_near_0_ratio": agg_scalars(self.w_anti_near0),
                "anti_near_1_ratio": agg_scalars(self.w_anti_near1),
            },
            "logits": {
                "detail_mean": agg_scalars(self.logits_detail_mean),
                "anti_mean": agg_scalars(self.logits_anti_mean),
                "abs_mean": agg_scalars(self.logits_abs_mean),
            },
            "residual": {
                "delta_l2norm_mean": agg_scalars(self.delta_norm),
                "p3_l2norm": agg_scalars(self.p3_norm),
                "delta_mean": agg_scalars(self.delta_mean),
                "delta_std": agg_scalars(self.delta_std),
            },
            "gt_region_analysis": {
                "gate_detail_weight": _region_agg(self.region["w_detail"]),
                "compensation_norm": _region_agg(self.region["delta_norm"]),
            },
        }


class _RDCFCollector:
    """hook RDCF 内部各子模块, 就地聚合统计并支持 GT 区域分析。"""

    def __init__(self, module, name, feat_size):
        self.module = module
        self.name = name
        self.feat_size = feat_size
        self.handles = []
        self.f3_stats, self.p2_stats, self.f4_stats, self.out_stats = [], [], [], []
        self.d2_stats, self.d3_stats, self.d4_stats = [], [], []
        self.r_mean, self.r_std, self.r_min, self.r_max = [], [], [], []
        self.r_near_low, self.r_near_high = [], []
        self.delta_norm, self.f3_norm = [], []
        self.delta_mean, self.delta_std = [], []
        self.logits_mean, self.logits_std, self.logits_abs_mean = [], [], []
        self.pending = None
        self.region = {"r": {"fg": [], "bg": []}, "delta_norm": {"fg": [], "bg": []}}

        self.handles.append(module.register_forward_hook(self._on_module))
        if hasattr(module, "rel"):
            self.handles.append(module.rel.register_forward_hook(self._on_rel))
        if hasattr(module, "delta"):
            self.handles.append(module.delta.register_forward_hook(self._on_delta))
        if hasattr(module, "d2_conv"):
            self.handles.append(module.d2_conv.register_forward_hook(self._on_d2))
        if hasattr(module, "d3_conv"):
            self.handles.append(module.d3_conv.register_forward_hook(self._on_d3))
        if hasattr(module, "d4_conv"):
            self.handles.append(module.d4_conv.register_forward_hook(self._on_d4))

    def _on_module(self, m, inp, out):
        f3, p2, f4 = inp
        self.f3_stats.append(tensor_stats(f3))
        self.p2_stats.append(tensor_stats(p2))
        self.f4_stats.append(tensor_stats(f4))
        self.out_stats.append(tensor_stats(out))
        self.f3_norm.append(_as_float(f3.detach().float().norm(2).cpu()))

    def _on_rel(self, m, inp, out):
        logits = out.detach().float()
        r = 0.1 + 0.9 * torch_sigmoid(logits)  # [B,1,H,W]
        self.r_mean.append(_as_float(r.mean().cpu()))
        self.r_std.append(_as_float(r.std().cpu()))
        self.r_min.append(_as_float(r.min().cpu()))
        self.r_max.append(_as_float(r.max().cpu()))
        self.r_near_low.append(_as_float((r < 0.15).float().mean().cpu()))
        self.r_near_high.append(_as_float((r > 0.85).float().mean().cpu()))
        self.logits_mean.append(_as_float(logits.mean().cpu()))
        self.logits_std.append(_as_float(logits.std().cpu()))
        self.logits_abs_mean.append(_as_float(logits.abs().mean().cpu()))
        if self.pending is None:
            self.pending = {}
        self.pending["r"] = r[0, 0].cpu().numpy().astype(np.float32)

    def _on_delta(self, m, inp, out):
        delta = out.detach().float()
        self.delta_mean.append(_as_float(delta.mean().cpu()))
        self.delta_std.append(_as_float(delta.std().cpu()))
        dn = delta.norm(2, dim=1)
        self.delta_norm.append(_as_float(dn.mean().cpu()))
        if self.pending is None:
            self.pending = {}
        self.pending["delta_norm"] = dn[0].cpu().numpy().astype(np.float32)

    def _on_d2(self, m, inp, out):
        self.d2_stats.append(tensor_stats(out))

    def _on_d3(self, m, inp, out):
        self.d3_stats.append(tensor_stats(out))

    def _on_d4(self, m, inp, out):
        self.d4_stats.append(tensor_stats(out))

    def finalize_sample(self, gt_boxes_feat):
        if self.pending is not None:
            for key in ("r", "delta_norm"):
                if key in self.pending:
                    fg, bg = _region_stats(self.pending[key], gt_boxes_feat)
                    self.region[key]["fg"].append(fg)
                    self.region[key]["bg"].append(bg)
        self.pending = None

    def remove(self):
        for h in self.handles:
            h.remove()

    def to_dict(self):
        return {
            "name": self.name,
            "n_samples": len(self.f3_stats),
            "f3_input": _agg_tensor_stats(self.f3_stats),
            "p2_input": _agg_tensor_stats(self.p2_stats),
            "f4_input": _agg_tensor_stats(self.f4_stats),
            "output": _agg_tensor_stats(self.out_stats),
            "d2": _agg_tensor_stats(self.d2_stats),
            "d3": _agg_tensor_stats(self.d3_stats),
            "d4": _agg_tensor_stats(self.d4_stats),
            "reliability": {
                "mean": agg_scalars(self.r_mean),
                "std": agg_scalars(self.r_std),
                "min": agg_scalars(self.r_min),
                "max": agg_scalars(self.r_max),
                "near_low_0_1_ratio": agg_scalars(self.r_near_low),
                "near_high_1_0_ratio": agg_scalars(self.r_near_high),
            },
            "logits": {
                "mean": agg_scalars(self.logits_mean),
                "std": agg_scalars(self.logits_std),
                "abs_mean": agg_scalars(self.logits_abs_mean),
            },
            "residual": {
                "delta_l2norm_mean": agg_scalars(self.delta_norm),
                "f3_l2norm": agg_scalars(self.f3_norm),
                "delta_mean": agg_scalars(self.delta_mean),
                "delta_std": agg_scalars(self.delta_std),
            },
            "gt_region_analysis": {
                "reliability": _region_agg(self.region["r"]),
                "compensation_norm": _region_agg(self.region["delta_norm"]),
            },
        }


def _agg_tensor_stats(stats_list):
    """聚合多张图的 tensor_stats。"""
    if not stats_list:
        return {"mean": None, "std": None, "l2norm": None, "n": 0}
    keys = ("mean", "std", "l2norm")
    out = {"n": len(stats_list)}
    for k in keys:
        out[k] = agg_scalars([s.get(k) for s in stats_list])
    return out


def _region_stats(resp_map: np.ndarray, gt_boxes_feat):
    """给定响应图 [H,W] 与 GT 框 (特征图坐标), 计算前景/背景统计。"""
    H, W = resp_map.shape
    mask = np.zeros((H, W), dtype=bool)
    for box in gt_boxes_feat:
        x1, y1, x2, y2 = box
        x1 = int(np.floor(max(0, x1)))
        y1 = int(np.floor(max(0, y1)))
        x2 = int(np.ceil(min(W, x2)))
        y2 = int(np.ceil(min(H, y2)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    fg = resp_map[mask]
    bg = resp_map[~mask]
    return (
        _as_float(fg.mean()) if fg.size else None,
        _as_float(bg.mean()) if bg.size else None,
    )


def _region_agg(region_dict):
    return {
        "fg": agg_scalars(region_dict["fg"]),
        "bg": agg_scalars(region_dict["bg"]),
    }


def gt_boxes_to_feat(gt_boxes_np, ori_shape, feat_size, imgsz):
    """原图 xyxy -> 特征图坐标 (stretch 输入)。feat_size=(H,W)。"""
    h0, w0 = ori_shape
    fh, fw = feat_size
    scale_x = fw / w0
    scale_y = fh / h0
    boxes = np.asarray(gt_boxes_np, dtype=np.float32).reshape(-1, 4)
    out = boxes.copy()
    out[:, [0, 2]] *= scale_x
    out[:, [1, 3]] *= scale_y
    return out


# --------------------------------------------------------------------------- #
# 模型加载 / 参数统计
# --------------------------------------------------------------------------- #
def load_model(weight: str, device):
    """加载单个 RT-DETR ckpt, 返回 (model, ckpt)。"""
    log(f"加载模型: {weight}")
    t0 = time.time()
    model, ckpt = attempt_load_one_weight(weight, device=device, inplace=True, fuse=False)
    model.eval()
    log(f"  加载完成 ({time.time() - t0:.2f}s), nc={getattr(model, 'nc', '?')}")
    return model, ckpt


def model_param_summary(model) -> dict:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    baps_params, rdcf_params = 0, 0
    n_baps, n_rdcf = 0, 0
    for m in model.modules():
        if isinstance(m, BAPS):
            baps_params += sum(p.numel() for p in m.parameters())
            n_baps += 1
        elif isinstance(m, RDCF):
            rdcf_params += sum(p.numel() for p in m.parameters())
            n_rdcf += 1
    return {
        "total_params": total,
        "trainable_params": trainable,
        "n_top_modules": len(model.model),
        "n_baps_instances": n_baps,
        "n_rdcf_instances": n_rdcf,
        "baps_params": baps_params if n_baps else None,
        "rdcf_params": rdcf_params if n_rdcf else None,
    }


def try_flops(model, device, imgsz):
    """尝试用 thop 计算 GFLOPs, 失败返回 None。"""
    try:
        from thop import profile as thop_profile
        inp = torch_zeros(1, 3, imgsz, imgsz, device=device)
        flops, params = thop_profile(model, inputs=(inp,), verbose=False)
        return {"gflops": _as_float(flops), "params": _as_float(params)}
    except Exception as e:
        log(f"[warn] GFLOPs 计算失败 (thop): {e}")
        return None


# --------------------------------------------------------------------------- #
# 预测分析 (full 模式)
# --------------------------------------------------------------------------- #
def postprocess_rtdetr(y, imgsz, conf_thr):
    """把 RTDETR 输出 [bs,300,4+nc] 转成每图 (boxes_640, cls, score), 按 score 降序。"""
    bs, _, nd = y.shape
    bboxes, scores = y.split((4, nd - 4), dim=-1)
    bboxes = bboxes * imgsz  # 640 空间 xywh
    outs = []
    for i in range(bs):
        b = xywh2xyxy(bboxes[i])  # [300,4] xyxy
        s = scores[i]  # [300,nc]
        score, cls = s.max(-1)
        order = torch_argsort(score, descending=True)
        b, cls, score = b[order], cls[order], score[order]
        if conf_thr and conf_thr > 0:
            keep = score > conf_thr
            b, cls, score = b[keep], cls[keep], score[keep]
        outs.append((b, cls, score))
    return outs


def run_predictions(model, dataloader, device, imgsz, conf_thr, max_images):
    """对数据集前向, 返回逐图像预测 dict 列表。"""
    results = []
    model.eval()
    seen = 0
    with torch_inference_mode():
        for bi, batch in enumerate(dataloader):
            im = batch["img"].to(device)
            y, _ = model(im)
            preds = postprocess_rtdetr(y, imgsz, conf_thr)
            for i in range(im.shape[0]):
                ori_shape = batch["ori_shape"][i]
                h0, w0 = ori_shape
                b, cls, score = preds[i]
                b = b.cpu().numpy()
                cls = cls.cpu().numpy()
                score = score.cpu().numpy()
                # 转回原图坐标
                if b.shape[0]:
                    b = b.copy()
                    b[:, [0, 2]] *= w0 / imgsz
                    b[:, [1, 3]] *= h0 / imgsz
                # GT
                gt = xywh2xyxy(batch["bboxes"][batch["batch_idx"] == i])
                gt = gt.cpu().numpy().astype(np.float32)
                if gt.shape[0]:
                    gt = gt.copy()
                    gt[:, [0, 2]] *= w0
                    gt[:, [1, 3]] *= h0
                gt_cls = batch["cls"][batch["batch_idx"] == i].cpu().numpy().astype(np.int64)
                im_file = batch.get("im_file", [None])[i]
                results.append({
                    "image_id": seen,
                    "path": str(im_file) if im_file is not None else None,
                    "ori_shape": [int(h0), int(w0)],
                    "gt_boxes": gt.tolist(),
                    "gt_classes": gt_cls.tolist(),
                    "pred_boxes": b.tolist(),
                    "pred_classes": cls.astype(int).tolist(),
                    "pred_scores": score.tolist(),
                    "blur_score": laplacian_variance(im[i]),
                })
                seen += 1
                if max_images and seen >= max_images:
                    return results
    return results


def analyze_results(model_name, results, imgsz, iou_thr):
    """对一个模型的预测结果做匹配与统计。"""
    size_groups = {"tiny": [], "small": [], "medium": [], "large": []}
    per_image_rows = []
    loc_rows = []
    tp_iou_all = []

    n_gt_all, n_tp_all, n_pred_all = 0, 0, 0

    # 模糊度分位 (图像级, 三模型图像集合一致, 分位基本一致)
    blur_scores = [r.get("blur_score") for r in results]
    blur_valid = [b for b in blur_scores if b is not None and not math_is_nan(b)]
    if len(blur_valid) >= 3:
        q_low = float(np.percentile(blur_valid, 33.33))
        q_high = float(np.percentile(blur_valid, 66.67))
    else:
        q_low = q_high = None
    blur_groups = {"low": {"n_images": 0, "n_gt": 0, "n_tp": 0},
                   "mid": {"n_images": 0, "n_gt": 0, "n_tp": 0},
                   "high": {"n_images": 0, "n_gt": 0, "n_tp": 0}}

    for r in results:
        h0, w0 = r["ori_shape"]
        gt = np.asarray(r["gt_boxes"], dtype=np.float32).reshape(-1, 4)
        gt_cls = np.asarray(r["gt_classes"], dtype=np.int64).reshape(-1)
        pred = np.asarray(r["pred_boxes"], dtype=np.float32).reshape(-1, 4)
        pred_cls = np.asarray(r["pred_classes"], dtype=np.int64).reshape(-1)
        pred_score = np.asarray(r["pred_scores"], dtype=np.float32).reshape(-1)

        n_gt = gt.shape[0]
        n_pred = pred.shape[0]
        n_gt_all += n_gt
        n_pred_all += n_pred

        tp_idx, tp_iou, matched_gt = greedy_match(pred, pred_cls, gt, gt_cls, iou_thr=iou_thr)
        n_tp = len(tp_idx)
        n_tp_all += n_tp
        tp_iou_all.extend(tp_iou)

        # 定位误差
        for p_i, g_i, iou in zip(tp_idx, matched_gt, tp_iou):
            pb = pred[p_i]
            gb = gt[g_i]
            pw, ph = pb[2] - pb[0], pb[3] - pb[1]
            gw, gh = gb[2] - gb[0], gb[3] - gb[1]
            pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
            gcx, gcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
            center_err = float(np.hypot(pcx - gcx, pcy - gcy))
            w_err = float(abs(pw - gw) / max(gw, 1e-6))
            h_err = float(abs(ph - gh) / max(gh, 1e-6))
            loc_rows.append({
                "model": model_name, "image_id": r["image_id"],
                "iou": float(iou), "center_error_px": center_err,
                "w_rel_error": w_err, "h_rel_error": h_err,
                "gt_w": float(gw), "gt_h": float(gh),
            })

        # 目标尺寸分组 (640 空间像素边长)
        gt_640 = gt.copy()
        gt_640[:, [0, 2]] *= imgsz / w0
        gt_640[:, [1, 3]] *= imgsz / h0
        sizes = np.sqrt(np.clip((gt_640[:, 2] - gt_640[:, 0]) * (gt_640[:, 3] - gt_640[:, 1]), 0, None))
        matched_set = set(matched_gt)
        for gi, s in enumerate(sizes):
            group = ("tiny" if s <= 8 else "small" if s <= 16 else "medium" if s <= 32 else "large")
            size_groups[group].append({
                "matched": gi in matched_set,
                "iou": tp_iou[matched_gt.index(gi)] if gi in matched_set else 0.0,
                "size": float(s),
            })

        # 模糊度分组
        bs = r.get("blur_score")
        blur_group = None
        if bs is not None and not math_is_nan(bs) and q_low is not None:
            if bs <= q_low:
                blur_group = "low"
            elif bs <= q_high:
                blur_group = "mid"
            else:
                blur_group = "high"
            blur_groups[blur_group]["n_images"] += 1
            blur_groups[blur_group]["n_gt"] += n_gt
            blur_groups[blur_group]["n_tp"] += n_tp

        per_image_rows.append({
            "model": model_name, "image_id": r["image_id"], "path": r["path"],
            "ori_h": h0, "ori_w": w0, "n_gt": n_gt, "n_pred": n_pred, "n_tp": n_tp,
            "recall": (n_tp / n_gt) if n_gt else None,
            "precision": (n_tp / n_pred) if n_pred else None,
            "blur_score": bs,
            "blur_group": blur_group,
        })

    size_metrics = {}
    for k, items in size_groups.items():
        n = len(items)
        matched = sum(1 for it in items if it["matched"])
        ious = [it["iou"] for it in items if it["matched"]]
        size_metrics[k] = {
            "n_gt": n,
            "n_tp": matched,
            "recall": (matched / n) if n else None,
            "mean_iou": float(np.mean(ious)) if ious else None,
            "mean_size": float(np.mean([it["size"] for it in items])) if items else None,
        }

    blur_metrics = {}
    for k, acc in blur_groups.items():
        blur_metrics[k] = {
            "n_images": acc["n_images"],
            "n_gt": acc["n_gt"],
            "n_tp": acc["n_tp"],
            "recall": (acc["n_tp"] / acc["n_gt"]) if acc["n_gt"] else None,
        }

    return {
        "model": model_name,
        "n_images": len(results),
        "n_gt": n_gt_all,
        "n_pred": n_pred_all,
        "n_tp": n_tp_all,
        "recall_50": (n_tp_all / n_gt_all) if n_gt_all else None,
        "precision_50": (n_tp_all / n_pred_all) if n_pred_all else None,
        "mean_tp_iou": float(np.mean(tp_iou_all)) if tp_iou_all else None,
        "blur_quantiles": {"q_low": q_low, "q_high": q_high},
        "size_group_metrics": size_metrics,
        "blur_group_metrics": blur_metrics,
    }, per_image_rows, loc_rows


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="RT-DETR-R18 BAPS/RDCF 诊断采集")
    p.add_argument("--baseline-weights", type=str, default=None, help="E0 原始 R18 best.pt 路径")
    p.add_argument("--baps-weights", type=str, default=None, help="E1 BAPS best.pt 路径")
    p.add_argument("--rdcf-weights", type=str, default=None, help="E2 RDCF best.pt 路径")
    p.add_argument("--data", type=str, default="dataset/DUT-Anti-UAV.yaml", help="数据 yaml 路径")
    p.add_argument("--device", type=str, default="0", help="cuda 设备, 如 0 或 cpu")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    p.add_argument("--batch", type=int, default=16, help="预测批大小")
    p.add_argument("--split", type=str, default="test", help="数据集划分 (test/val)")
    p.add_argument("--mode", type=str, default="full", choices=["basic", "full"], help="basic=结构审计+少量特征采集; full=+完整测试集预测分析")
    p.add_argument("--max-images", type=int, default=0, help="full 模式最多推理的图像数, 0=全部")
    p.add_argument("--conf", type=float, default=0.001, help="保存预测的最低置信度")
    p.add_argument("--iou-thr", type=float, default=0.5, help="匹配 IoU 阈值")
    p.add_argument("--feat-samples", type=int, default=32, help="行为分析采样图像数")
    p.add_argument("--output", type=str, default="diagnostics_output", help="输出目录")
    return p.parse_args()


def main():
    args = parse_args()

    weights = {
        "baseline": args.baseline_weights,
        "baps": args.baps_weights,
        "rdcf": args.rdcf_weights,
    }
    weights = {k: v for k, v in weights.items() if v}
    if not weights:
        log("错误: 至少提供一个权重 (--baseline-weights / --baps-weights / --rdcf-weights)")
        sys.exit(2)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch_select_device(args.device, args.batch)
    log(f"设备: {device}")

    # 环境信息
    env = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "mode": args.mode,
        "max_images": args.max_images,
        "conf_thr": args.conf,
        "iou_thr": args.iou_thr,
        "weights": weights,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)

    # 数据 (仅 full 模式 + 行为采样需要)
    data_dict = None
    dataset = None
    if args.mode == "full":
        try:
            data_dict = check_det_dataset(args.data)
            img_path = data_dict.get(args.split)
            if img_path is None:
                log(f"错误: 数据 yaml 中无 '{args.split}' 划分, 可用键: {list(data_dict.keys())}")
                sys.exit(2)
            from ultralytics.models.rtdetr.val import RTDETRDataset as _RTDETRDataset
            from ultralytics.data.build import build_dataloader as _build_dataloader
            from ultralytics.utils import DEFAULT_CFG as _DEFAULT_CFG
            dataset = _RTDETRDataset(
                img_path=img_path,
                imgsz=args.imgsz,
                batch_size=args.batch,
                augment=False,
                hyp=_DEFAULT_CFG,
                rect=False,
                cache=None,
                prefix=f"{args.split}: ",
                data=data_dict,
            )
            log(f"数据集 {args.split}: {len(dataset)} 张图像")
        except Exception as e:
            log(f"错误: 构建数据集失败: {e}\n{traceback.format_exc()}")
            sys.exit(2)

    # ---- 逐模型处理 ----
    configs = {}
    model_summaries = {}
    layer_params_all = {}
    feature_shapes_all = {}
    baps_stats_all = {}
    rdcf_stats_all = {}
    ckpt_audits = {}
    predictions_all = {}
    eval_metrics_all = {}

    for mname, wpath in weights.items():
        log(f"\n{'=' * 70}\n处理模型 [{mname}] : {wpath}\n{'=' * 70}")
        if not Path(wpath).exists():
            log(f"[warn] 权重文件不存在, 跳过: {wpath}")
            continue

        model, ckpt = load_model(wpath, device)

        # --- 结构审计 ---
        ms = model_param_summary(model)
        ms["weight_path"] = wpath
        ms["weight_size_bytes"] = Path(wpath).stat().st_size if Path(wpath).exists() else None
        ms["nc"] = getattr(model, "nc", None)
        ms["names"] = safe_json(getattr(model, "names", None))
        ms["stride"] = safe_json(getattr(model, "stride", None))
        ms["yaml"] = safe_json(getattr(model, "yaml", None))
        ms["flops"] = try_flops(model, device, args.imgsz)
        model_summaries[mname] = ms

        rows = summarize_modules(model)
        layer_params_all[mname] = rows
        feature_shapes_all[mname] = collect_feature_shapes(model, device, args.imgsz)

        # --- 行为分析 (BAPS/RDCF) ---
        baps_collectors, rdcf_collectors = [], []
        feat_size = (args.imgsz // 8, args.imgsz // 8)
        for name, m in model.named_modules():
            if isinstance(m, BAPS):
                baps_collectors.append(_BAPSCollector(m, name, feat_size))
            elif isinstance(m, RDCF):
                rdcf_collectors.append(_RDCFCollector(m, name, feat_size))
        if baps_collectors:
            log(f"  发现 {len(baps_collectors)} 个 BAPS 实例")
        if rdcf_collectors:
            log(f"  发现 {len(rdcf_collectors)} 个 RDCF 实例")

        if baps_collectors or rdcf_collectors:
            _run_behavior_sampling(model, baps_collectors, rdcf_collectors, device, args, data_dict)
            baps_stats_all[mname] = [c.to_dict() for c in baps_collectors]
            rdcf_stats_all[mname] = [c.to_dict() for c in rdcf_collectors]
            for c in baps_collectors:
                c.remove()
            for c in rdcf_collectors:
                c.remove()

        # --- checkpoint / 训练配置审计 ---
        ckpt_audits[mname] = audit_checkpoint(ckpt, wpath)

        # --- full: 完整测试集预测 ---
        if args.mode == "full" and dataset is not None:
            from ultralytics.data.build import build_dataloader as _build_dataloader
            dl = _build_dataloader(dataset, batch=args.batch, workers=min(8, os.cpu_count() or 1), shuffle=False)
            log(f"  推理测试集 (max_images={args.max_images or '全部'})...")
            t0 = time.time()
            results = run_predictions(model, dl, device, args.imgsz, args.conf, args.max_images)
            log(f"  推理完成 ({time.time() - t0:.2f}s), {len(results)} 张")
            predictions_all[mname] = results
            eval_metrics_all[mname], per_img, loc = analyze_results(mname, results, args.imgsz, args.iou_thr)
            eval_metrics_all[mname + "_per_image"] = per_img
            eval_metrics_all[mname + "_localization"] = loc

        # 释放
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 写结果文件 ----
    _write_all(out_dir, env, model_summaries, layer_params_all, feature_shapes_all,
               baps_stats_all, rdcf_stats_all, ckpt_audits, predictions_all, eval_metrics_all, args)

    _file_logger(out_dir / "execution_log.txt")
    _make_zip(out_dir)
    log(f"\n全部完成。输出目录: {out_dir} ; 打包文件: {out_dir.parent / 'r18_diagnostics.zip'}")


def _run_behavior_sampling(model, baps_collectors, rdcf_collectors, device, args, data_dict):
    """用少量样本前向, 驱动 BAPS/RDCF collector 采集统计。"""
    from ultralytics.models.rtdetr.val import RTDETRDataset as _RTDETRDataset
    from ultralytics.data.build import build_dataloader as _build_dataloader
    from ultralytics.utils import DEFAULT_CFG as _DEFAULT_CFG

    img_path = data_dict.get(args.split) if data_dict else None
    if img_path is None:
        log("[warn] 无数据路径, 跳过行为采样")
        return
    try:
        ds = _RTDETRDataset(
            img_path=img_path, imgsz=args.imgsz, batch_size=1, augment=False,
            hyp=_DEFAULT_CFG, rect=False, cache=None, prefix="feat: ", data=data_dict,
        )
    except Exception as e:
        log(f"[warn] 行为采样数据集构建失败: {e}")
        return

    n = min(args.feat_samples, len(ds))
    dl = _build_dataloader(ds, batch=1, workers=min(4, os.cpu_count() or 1), shuffle=False)
    log(f"  行为采样: 前 {n} 张图像")
    feat_size = (args.imgsz // 8, args.imgsz // 8)
    model.eval()
    with torch_inference_mode():
        for bi, batch in enumerate(dl):
            if bi >= n:
                break
            im = batch["img"].to(device)
            _ = model(im)
            # GT -> 特征图坐标, 供区域统计
            gt_boxes_np = xywh2xyxy(batch["bboxes"][batch["batch_idx"] == 0]).cpu().numpy().astype(np.float32)
            ori_shape = batch["ori_shape"][0]
            h0, w0 = ori_shape
            if gt_boxes_np.shape[0]:
                gt_boxes_np[:, [0, 2]] *= w0
                gt_boxes_np[:, [1, 3]] *= h0
            gt_feat = gt_boxes_to_feat(gt_boxes_np, ori_shape, feat_size, args.imgsz)
            for c in baps_collectors:
                c.finalize_sample(gt_feat)
            for c in rdcf_collectors:
                c.finalize_sample(gt_feat)


def audit_checkpoint(ckpt, wpath):
    """审计 checkpoint 里的训练配置, 无法确定的字段标 unknown。"""
    if not ckpt:
        return {"weight": wpath, "error": "no ckpt dict"}
    train_args = ckpt.get("train_args") or {}
    ta = {k: safe_json(v) for k, v in train_args.items()}

    def pick(*keys):
        for k in keys:
            if k in train_args and train_args[k] is not None:
                return train_args[k]
        return "unknown"

    return {
        "weight": wpath,
        "epoch": safe_json(ckpt.get("epoch")),
        "best_fitness": safe_json(ckpt.get("best_fitness")),
        "date": safe_json(ckpt.get("date")),
        "task": safe_json(ckpt.get("task")) or pick("task"),
        "nc": safe_json(ckpt.get("nc")) or pick("nc"),
        "optimizer": pick("optimizer"),
        "initial_lr": pick("lr0", "initial_lr"),
        "batch": pick("batch"),
        "epochs": pick("epochs"),
        "imgsz": pick("imgsz"),
        "seed": pick("seed"),
        "deterministic": pick("deterministic"),
        "data": pick("data"),
        "pretrained_weights": pick("model", "pretrained"),
        "augmentation": {
            "mosaic": pick("mosaic"),
            "mixup": pick("mixup"),
            "copy_paste": pick("copy_paste"),
            "degrees": pick("degrees"),
            "translate": pick("translate"),
            "scale": pick("scale"),
            "fliplr": pick("fliplr"),
            "flipud": pick("flipud"),
            "hsv_h": pick("hsv_h"),
            "hsv_s": pick("hsv_s"),
            "hsv_v": pick("hsv_v"),
        },
        "best_epoch": safe_json(ckpt.get("best_epoch")) or "unknown",
        "train_args_full": ta,
    }


def _write_all(out_dir, env, model_summaries, layer_params_all, feature_shapes_all,
               baps_stats_all, rdcf_stats_all, ckpt_audits, predictions_all, eval_metrics_all, args):
    """把所有结果写成 json / csv。"""
    def dump_json(name, obj):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(safe_json(obj), f, indent=2, ensure_ascii=False)

    dump_json("configs.json", {"cli": vars(args), "env": env})
    dump_json("model_summary.json", model_summaries)
    dump_json("feature_shapes.json", feature_shapes_all)
    dump_json("baps_statistics.json", baps_stats_all)
    dump_json("rdcf_statistics.json", rdcf_stats_all)
    dump_json("checkpoint_audit.json", ckpt_audits)

    # layer_parameters.csv
    with open(out_dir / "layer_parameters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "index", "type", "declared_type", "from", "params", "parse_params", "has_baps_rdcf"])
        for mname, rows in layer_params_all.items():
            for r in rows:
                w.writerow([mname, r["index"], r["type"], r["declared_type"], r["from"],
                            r["params"], r["parse_params"], r["has_baps_rdcf"]])

    if eval_metrics_all:
        # 分离 metrics 与 per-image / localization
        metrics = {k: v for k, v in eval_metrics_all.items()
                   if not k.endswith("_per_image") and not k.endswith("_localization")}
        dump_json("evaluation_metrics.json", metrics)

        # per_image_analysis.csv
        with open(out_dir / "per_image_analysis.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "image_id", "path", "ori_h", "ori_w", "n_gt", "n_pred", "n_tp",
                        "recall", "precision", "blur_score", "blur_group"])
            for mname, rows in eval_metrics_all.items():
                if not mname.endswith("_per_image"):
                    continue
                for r in rows:
                    w.writerow([r["model"], r["image_id"], r["path"], r["ori_h"], r["ori_w"],
                                r["n_gt"], r["n_pred"], r["n_tp"], r["recall"], r["precision"],
                                r["blur_score"], r.get("blur_group")])

        # localization_analysis.csv
        with open(out_dir / "localization_analysis.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "image_id", "iou", "center_error_px", "w_rel_error", "h_rel_error", "gt_w", "gt_h"])
            for mname, rows in eval_metrics_all.items():
                if not mname.endswith("_localization"):
                    continue
                for r in rows:
                    w.writerow([r["model"], r["image_id"], r["iou"], r["center_error_px"],
                                r["w_rel_error"], r["h_rel_error"], r["gt_w"], r["gt_h"]])

        # size_group_metrics.csv
        with open(out_dir / "size_group_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "group", "n_gt", "n_tp", "recall", "mean_iou", "mean_size"])
            for mname, m in metrics.items():
                for group, sgm in (m.get("size_group_metrics") or {}).items():
                    w.writerow([mname, group, sgm.get("n_gt"), sgm.get("n_tp"), sgm.get("recall"),
                                sgm.get("mean_iou"), sgm.get("mean_size")])

        # blur_group_metrics.csv
        with open(out_dir / "blur_group_metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "group", "n_images", "n_gt", "n_tp", "recall"])
            for mname, m in metrics.items():
                for group, bgm in (m.get("blur_group_metrics") or {}).items():
                    w.writerow([mname, group, bgm.get("n_images"), bgm.get("n_gt"),
                                bgm.get("n_tp"), bgm.get("recall")])

    if predictions_all:
        with gzip.open(out_dir / "predictions.jsonl.gz", "wt", encoding="utf-8") as f:
            for mname, results in predictions_all.items():
                for r in results:
                    rec = {"model": mname, **r}
                    f.write(json.dumps(safe_json(rec), ensure_ascii=False) + "\n")


def _make_zip(out_dir: Path):
    zip_path = out_dir.parent / "r18_diagnostics.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.iterdir()):
            if f.is_file():
                zf.write(f, arcname=Path("diagnostics_output") / f.name)
    log(f"已打包: {zip_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("用户中断")
        sys.exit(130)
    except Exception:
        log(f"未捕获异常:\n{traceback.format_exc()}")
        sys.exit(1)
