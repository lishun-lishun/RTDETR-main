"""Evaluate a targeted R09--R16 model on the explicit val or test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from r18_targeted_common import CONFIGS, DATA, config_path, local_package, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        parser.error("weights and data YAML must exist")
    if args.output_dir.exists() and not args.dry_run:
        parser.error(f"output directory already exists: {args.output_dir}")
    payload = {"model": args.model, "yaml": str(config_path(args.model)), "weights": str(args.weights),
               "data": str(args.data), "split": args.split, "device": args.device,
               "batch": args.batch, "imgsz": args.imgsz}
    if args.dry_run:
        print(json.dumps(payload, indent=2)); return
    local_package()
    from ultralytics import RTDETR
    result = RTDETR(str(args.weights)).val(data=str(args.data), split=args.split, device=args.device,
                                           batch=args.batch, imgsz=args.imgsz, max_det=300, save_json=True,
                                           project=str(args.output_dir.parent), name=args.output_dir.name,
                                           exist_ok=False)
    native = dict(getattr(result, "results_dict", {}))
    metrics = {**payload, "precision": native.get("metrics/precision(B)"),
               "recall": native.get("metrics/recall(B)"), "mAP50": native.get("metrics/mAP50(B)"),
               "mAP50:95": native.get("metrics/mAP50-95(B)"), "ultralytics": native,
               "save_dir": str(getattr(result, "save_dir", ""))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "evaluation_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
