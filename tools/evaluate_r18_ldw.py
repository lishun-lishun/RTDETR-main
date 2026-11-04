"""Evaluate an LDW-R18 checkpoint and save native Ultralytics metrics."""

import argparse
import json
from pathlib import Path

from ldw_r18_common import CONFIGS, DATA, config_path, local_package, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CONFIGS, default="full")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    # Final checkpoint evaluation defaults to the held-out test split. Use
    # --split val explicitly while selecting or comparing experiments.
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        parser.error("weights and dataset YAML must exist")
    if args.output_dir.exists() and not args.dry_run:
        parser.error("output directory already exists; choose a new directory")
    payload = {"model": args.model, "yaml": str(config_path(args.model)), "weights": str(args.weights),
               "data": str(args.data), "split": args.split, "device": args.device, "imgsz": 640, "max_det": 300}
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    local_package()
    from ultralytics import RTDETR

    result = RTDETR(str(args.weights)).val(data=str(args.data), split=args.split, device=args.device,
                                           batch=8, imgsz=640, max_det=300, project=str(args.output_dir.parent),
                                           name=args.output_dir.name, exist_ok=False)
    native = dict(getattr(result, "results_dict", {}))
    metrics = {"model": args.model, "weights": str(args.weights), "split": args.split,
               "input_size": 640, "max_detections": 300, "precision": native.get("metrics/precision(B)"),
               "recall": native.get("metrics/recall(B)"), "mAP50": native.get("metrics/mAP50(B)"),
               "mAP50_95": native.get("metrics/mAP50-95(B)"), "ultralytics": native}
    write_json(args.output_dir / "evaluation_metrics.json", metrics)
    # Keep a stable artifact even when the installed evaluator does not expose
    # custom size groups; a later COCO pass can replace this file in-place.
    (args.output_dir / "size_group_metrics.csv").write_text(
        "group,GT Count,AP50,AP75,AP50:95,Recall,Mean Matched IoU\n", encoding="utf-8")
    (args.output_dir / "localization_analysis.csv").write_text(
        "image_id,category_id,score,best_unmatched_iou\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
