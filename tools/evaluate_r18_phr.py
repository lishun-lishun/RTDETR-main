#!/usr/bin/env python3
"""RT-DETR-R18 PHR 评估脚本 (Linux 服务器端).

对 PHR-1 / PHR-2 / PHR-3 在 DUT-Anti-UAV 的 val/test 划分上运行验证, 汇总
mAP50 / mAP50:95 / mAP75 到一份 CSV。权重默认按项目目录自动查找 best.pt。

用法示例
--------
    python tools/evaluate_r18_phr.py --model phr1 --split val
    python tools/evaluate_r18_phr.py --model all --split test --device 0
    python tools/evaluate_r18_phr.py --model phr1 --weights runs/r18_phr1/train/.../best.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATA_CONFIG = _PROJECT_ROOT / "dataset" / "DUT-Anti-UAV.yaml"
PROFILE_CONFIG = _PROJECT_ROOT / "ultralytics" / "cfg" / "anti_detr_default.yaml"

_MODELS = {
    "phr1": "r18_phr1",
    "phr2": "r18_phr2",
    "phr3": "r18_phr3",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", choices=("all", *_MODELS), default=("phr1",))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights", type=Path, default=None, help="显式权重 (仅单模型)。")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--project", type=Path, default=_PROJECT_ROOT / "runs")
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


def main():
    args = parse_args()
    from ultralytics import RTDETR

    models = selected(args.model)
    if args.weights is not None and len(models) != 1:
        raise SystemExit("--weights 仅能与单个模型配合使用。")

    rows = []
    for name, project_dir in models.items():
        weights = args.weights or find_best_weights(args.project / project_dir)
        print(f"\n{'=' * 24} VAL {name} ({args.split}) {'=' * 24}")
        model = RTDETR(str(weights))
        metrics = model.val(
            data=str(DATA_CONFIG),
            split=args.split,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            max_det=args.max_det,
            device=args.device,
            project=str(args.project / project_dir),
            name="test",
        )
        row = {"model": name, "weights": str(weights), "split": args.split}
        row.update(getattr(metrics, "results_dict", {}))
        speed = getattr(metrics, "speed", {}) or {}
        row.update({f"speed/{k}": v for k, v in speed.items()})
        rows.append(row)

    out_dir = args.project / "phr_evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / f"{args.split}_summary.csv"
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n汇总已保存: {summary}")


if __name__ == "__main__":
    main()
