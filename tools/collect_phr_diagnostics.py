#!/usr/bin/env python3
"""RT-DETR-R18 PHR 内部行为统计 (Linux 服务器端).

对已训练好的 PHR-1 / PHR-2 / PHR-3 权重, 通过 forward hook 就地采集 PHR 模块的
激活统计, 用于判断模块是否真正 "激活" (而非退化回恒等映射)。产出一份 JSON。

采集的指标
----------
* gamma (补偿尺度 = alpha_max * tanh(raw_gamma)): 应显著 > 0 才表示 PHR 学习到贡献。
* residual_ratio = ||out - p3|| / ||p3||: 高分辨率补偿残差的相对强度。
* delta_l2norm: 补偿投影 ``delta`` 输出的平均 L2 范数。
* beta3 / beta4 (反馈尺度, 仅在对应开关开启时存在): 反馈路径是否被激活。

用法示例
--------
    python tools/collect_phr_diagnostics.py --model phr1 --split val --samples 64
    python tools/collect_phr_diagnostics.py --model all --split test --device 0

说明
----
* 复用项目 ``RTDETRDataset`` / ``build_dataloader``, 预处理与 val 一致 (stretch 到
  imgsz 正方形, 像素 /255)。
* 全程 eval + inference_mode, 不保存完整特征图, 内存友好。
* PHR 的 forward 输入为单个 list ``[p2, p3, p4]``, 故 forward hook 的 ``inp[0]``
  是 list, ``inp[0][1]`` 才是 p3。
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
    "phr1": "r18_phr1",
    "phr2": "r18_phr2",
    "phr3": "r18_phr3",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", choices=("all", *_MODELS), default=("phr1",))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--samples", type=int, default=64, help="用于统计的前 N 张图。")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights", type=Path, default=None, help="显式权重 (仅单模型)。")
    parser.add_argument("--project", type=Path, default=_PROJECT_ROOT / "runs")
    parser.add_argument("--output", type=Path, default=_PROJECT_ROOT / "runs" / "phr_statistics.json")
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


class PHRCollector:
    def __init__(self, module, name):
        self.name = name
        self.handles = []
        self.gamma = _f(module.alpha_max * torch.tanh(module.raw_gamma.detach()).cpu())
        self.beta3 = _f(module.beta_max * torch.tanh(module.raw_beta3.detach()).cpu()) if module.use_p3_feedback else None
        self.beta4 = _f(module.beta_max * torch.tanh(module.raw_beta4.detach()).cpu()) if module.use_p4_feedback else None
        self.in_norm, self.resid_norm = [], []
        self.delta_norm = []
        self.handles.append(module.register_forward_hook(self._on_module))
        if hasattr(module, "delta"):
            self.handles.append(module.delta.register_forward_hook(self._on_delta))

    def _on_module(self, m, inp, out):
        # PHR 的 forward 输入是单个 list [p2, p3, p4]; inp 为单元素 tuple。
        p3 = inp[0][1].detach().float()
        o = out.detach().float()
        self.in_norm.append(_f(p3.norm(2).cpu()))
        self.resid_norm.append(_f((o - p3).norm(2).cpu()))

    def _on_delta(self, m, inp, out):
        self.delta_norm.append(_f(out.detach().float().norm(2, dim=1).mean().cpu()))

    def remove(self):
        for h in self.handles:
            h.remove()

    def to_dict(self):
        ratio = [r / n if n and n > 0 else None for r, n in zip(self.resid_norm, self.in_norm)]
        return {
            "name": self.name,
            "gamma": self.gamma,
            "beta3": self.beta3,
            "beta4": self.beta4,
            "residual_ratio": agg(ratio),
            "delta_l2norm": agg(self.delta_norm),
        }


def collect_one(model, device, args, data_dict):
    from ultralytics.models.rtdetr.val import RTDETRDataset as _RTDETRDataset
    from ultralytics.data.build import build_dataloader as _build_dataloader
    from ultralytics.utils import DEFAULT_CFG as _DEFAULT_CFG

    collectors = []
    for m in model.modules():
        if m.__class__.__name__ == "PHR":
            collectors.append(PHRCollector(m, "PHR"))

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
            gamma = mod.get("gamma")
            beta_str = ", ".join(f"{k}={mod[k]:.4f}" for k in ("beta3", "beta4") if mod.get(k) is not None)
            rr = mod.get("residual_ratio", {}).get("mean")
            print(
                f"  {mod['name']}: gamma={gamma if gamma is None else round(gamma, 4)} "
                f"{beta_str or ''} residual_ratio_mean={rr if rr is None else round(rr, 5)}"
            )


if __name__ == "__main__":
    main()
