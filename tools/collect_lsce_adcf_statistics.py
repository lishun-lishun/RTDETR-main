#!/usr/bin/env python3
"""RT-DETR-R18 LSCE / ADCF 内部行为统计 (Linux 服务器端).

对已训练好的 E1/E2/E3 权重, 通过 forward hook 就地采集 LSCE / ADCF 模块的
激活统计, 用于判断模块是否真正 "激活" (而非退化回恒等映射)。产出一份 JSON。

采集的指标
----------
* LSCE:
    - alpha (learnable scale, tanh 后): 应 > 0 才表示模块已学习到贡献。
    - residual_ratio = ||out - x|| / ||x||: 残差项的相对强度。
    - delta_l2norm: 投影输出的平均 L2 范数。
    - gate_mean / gate_saturated_ratio: 背景复杂度门控的均值与饱和比例。
* ADCF:
    - alpha_semantic / alpha_local: 两条路径的 scale。
    - residual_ratio = ||out - f3|| / ||f3||。
    - delta_semantic_l2 / delta_local_l2: 两路径投影的 L2 范数。

用法示例
--------
    python tools/collect_lsce_adcf_statistics.py --model lsce_adcf --split val --samples 64
    python tools/collect_lsce_adcf_statistics.py --model all --split test --device 0

说明
----
* 复用项目 ``RTDETRDataset`` / ``build_dataloader``, 预处理与 val 一致 (stretch 到
  imgsz 正方形, 像素 /255)。
* 全程 eval + inference_mode, 不保存完整特征图, 内存友好。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATA_CONFIG = _PROJECT_ROOT / "dataset" / "DUT-Anti-UAV.yaml"

_MODELS = {
    "lsce": "r18_lsce",
    "adcf": "r18_adcf",
    "lsce_adcf": "r18_lsce_adcf",
    "e4_lsce_simple": "r18_lsce_e4",
    "e5_adcf_plain": "r18_adcf_e5",
    "e6_adcf_center": "r18_adcf_e6",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", choices=("all", *_MODELS), default=("lsce_adcf",))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=64, help="用于统计的前 N 张图。")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights", type=Path, default=None, help="显式权重 (仅单模型)。")
    parser.add_argument("--project", type=Path, default=_PROJECT_ROOT / "runs")
    parser.add_argument("--output", type=Path, default=_PROJECT_ROOT / "runs" / "lsce_adcf_statistics.json")
    return parser.parse_args()


def selected(names):
    if "all" in names:
        return dict(_MODELS)
    return {n: _MODELS[n] for n in names}


def find_best_weights(project_dir):
    hits = list(project_dir.glob("**/weights/best.pt"))
    if not hits:
        raise FileNotFoundError(f"未找到 best.pt: {project_dir}")
    return max(hits, key=lambda p: p.stat().st_mtime)


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": _f(arr.mean()), "std": _f(arr.std()), "n": int(arr.size)}


class LSCECollector:
    def __init__(self, module, name):
        self.name = name
        self.handles = []
        self.alpha = _f(torch.tanh(module.raw_alpha.detach()).cpu()) if hasattr(module, "raw_alpha") else None
        self.in_norm, self.resid_norm = [], []
        self.delta_norm, self.gate_mean, self.gate_sat = [], [], []
        self.handles.append(module.register_forward_hook(self._on_module))
        if hasattr(module, "delta"):
            self.handles.append(module.delta.register_forward_hook(self._on_delta))
        if hasattr(module, "gate"):
            self.handles.append(module.gate.register_forward_hook(self._on_gate))

    def _on_module(self, m, inp, out):
        x = inp[0].detach().float()
        o = out.detach().float()
        self.in_norm.append(_f(x.norm(2).cpu()))
        self.resid_norm.append(_f((o - x).norm(2).cpu()))

    def _on_delta(self, m, inp, out):
        d = out.detach().float()
        self.delta_norm.append(_f(d.norm(2, dim=1).mean().cpu()))

    def _on_gate(self, m, inp, out):
        g = out.detach().float()
        self.gate_mean.append(_f(g.mean().cpu()))
        self.gate_sat.append(_f(((g < 0.05) | (g > 0.95)).float().mean().cpu()))

    def remove(self):
        for h in self.handles:
            h.remove()

    def to_dict(self):
        ratio = [r / n if n and n > 0 else None for r, n in zip(self.resid_norm, self.in_norm)]
        return {
            "name": self.name,
            "alpha": self.alpha,
            "residual_ratio": agg(ratio),
            "delta_l2norm": agg(self.delta_norm),
            "gate_mean": agg(self.gate_mean),
            "gate_saturated_ratio": agg(self.gate_sat),
        }


class ADCFCollector:
    def __init__(self, module, name):
        self.name = name
        self.handles = []
        self.alpha_sem = _f(torch.tanh(module.raw_alpha_semantic.detach()).cpu()) if hasattr(module, "raw_alpha_semantic") else None
        self.alpha_loc = _f(torch.tanh(module.raw_alpha_local.detach()).cpu()) if hasattr(module, "raw_alpha_local") else None
        self.in_norm, self.resid_norm = [], []
        self.delta_sem, self.delta_loc = [], []
        self.handles.append(module.register_forward_hook(self._on_module))
        if hasattr(module, "delta_semantic"):
            self.handles.append(module.delta_semantic.register_forward_hook(self._on_delta_sem))
        if hasattr(module, "delta_local"):
            self.handles.append(module.delta_local.register_forward_hook(self._on_delta_loc))

    def _on_module(self, m, inp, out):
        f3 = inp[0][0].detach().float()
        o = out.detach().float()
        self.in_norm.append(_f(f3.norm(2).cpu()))
        self.resid_norm.append(_f((o - f3).norm(2).cpu()))

    def _on_delta_sem(self, m, inp, out):
        self.delta_sem.append(_f(out.detach().float().norm(2, dim=1).mean().cpu()))

    def _on_delta_loc(self, m, inp, out):
        self.delta_loc.append(_f(out.detach().float().norm(2, dim=1).mean().cpu()))

    def remove(self):
        for h in self.handles:
            h.remove()

    def to_dict(self):
        ratio = [r / n if n and n > 0 else None for r, n in zip(self.resid_norm, self.in_norm)]
        return {
            "name": self.name,
            "alpha_semantic": self.alpha_sem,
            "alpha_local": self.alpha_loc,
            "residual_ratio": agg(ratio),
            "delta_semantic_l2": agg(self.delta_sem),
            "delta_local_l2": agg(self.delta_loc),
        }


def collect_one(model, device, args, data_dict):
    from ultralytics.models.rtdetr.val import RTDETRDataset as _RTDETRDataset
    from ultralytics.data.build import build_dataloader as _build_dataloader
    from ultralytics.utils import DEFAULT_CFG as _DEFAULT_CFG

    collectors = []
    for m in model.modules():
        cls = m.__class__.__name__
        if cls == "LSCE":
            collectors.append(LSCECollector(m, cls))
        elif cls == "ADCF":
            collectors.append(ADCFCollector(m, cls))

    img_path = data_dict.get(args.split)
    ds = _RTDETRDataset(
        img_path=img_path, imgsz=args.imgsz, batch_size=1, augment=False,
        hyp=_DEFAULT_CFG, rect=False, cache=None, prefix="stat: ", data=data_dict,
    )
    n = min(args.samples, len(ds))
    dl = _build_dataloader(ds, batch=1, workers=min(4, os.cpu_count() or 1), shuffle=False)
    model.eval()
    with torch.inference_mode():
        for bi, batch in enumerate(dl):
            if bi >= n:
                break
            im = batch["img"].to(device)
            im = im.float() / 255.0
            _ = model(im)
    result = [c.to_dict() for c in collectors]
    for c in collectors:
        c.remove()
    return result


def main():
    args = parse_args()
    from ultralytics.utils.torch_utils import select_device as torch_select_device
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import attempt_load_one_weight

    device = torch_select_device(args.device)
    data_dict = check_det_dataset(str(DATA_CONFIG))
    models = selected(args.model)
    if args.weights is not None and len(models) != 1:
        raise SystemExit("--weights 仅能与单个模型配合使用。")

    report = {"device": str(device), "split": args.split, "samples": args.samples, "models": {}}
    for name, project_dir in models.items():
        weights = args.weights or find_best_weights(args.project / project_dir)
        print(f"\n{'=' * 24} COLLECT {name} {'=' * 24}")
        model, _ = attempt_load_one_weight(weights, device=device)  # 已 .to(device).float().eval()
        report["models"][name] = {"weights": str(weights), "modules": collect_one(model, device, args, data_dict)}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n统计已保存: {args.output}")
    # 简明摘要
    for name, info in report["models"].items():
        print(f"\n[{name}] {info['weights']}")
        for mod in info["modules"]:
            keys = [k for k in ("alpha", "alpha_semantic", "alpha_local") if k in mod and mod[k] is not None]
            alpha_str = ", ".join(f"{k}={mod[k]:.4f}" for k in keys)
            rr = mod.get("residual_ratio", {}).get("mean")
            print(f"  {mod['name']}: {alpha_str or 'n/a'} residual_ratio_mean={rr if rr is None else round(rr, 5)}")


if __name__ == "__main__":
    main()
