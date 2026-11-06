"""Evaluate an R18 variant on validation or test data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from r18_variants_common import CONFIGS, DATA, config_path, local_package, write_json


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
    parser.add_argument("--annotations", type=Path,
                        help="COCO annotations for Tiny/Small/Medium/Large analysis")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        parser.error("weights and data YAML must exist")
    if args.annotations and not args.annotations.is_file():
        parser.error("annotations file does not exist")
    if args.output_dir.exists() and not args.dry_run:
        parser.error(f"output directory already exists: {args.output_dir}")
    payload = {"model": args.model, "yaml": str(config_path(args.model)),
               "weights": str(args.weights), "data": str(args.data), "split": args.split,
               "device": args.device, "batch": args.batch, "imgsz": args.imgsz}
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    local_package()
    from ultralytics import RTDETR

    result = RTDETR(str(args.weights)).val(data=str(args.data), split=args.split,
                                           device=args.device, batch=args.batch, imgsz=args.imgsz,
                                           max_det=300, save_json=True,
                                           project=str(args.output_dir.parent), name=args.output_dir.name,
                                           exist_ok=False)
    native = dict(getattr(result, "results_dict", {}))
    metrics = {**payload,
               "precision": native.get("metrics/precision(B)"),
               "recall": native.get("metrics/recall(B)"),
               "mAP50": native.get("metrics/mAP50(B)"),
               "mAP50:95": native.get("metrics/mAP50-95(B)"),
               "ultralytics": native,
               "save_dir": str(getattr(result, "save_dir", ""))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "evaluation_metrics.json", metrics)
    if args.annotations:
        from pycocotools.coco import COCO
        from evaluate_backbone import matched_ious, measure_groups

        prediction_file = Path(result.save_dir) / "predictions.json"
        if not prediction_file.is_file():
            raise RuntimeError(f"Ultralytics did not create {prediction_file}")
        gt = COCO(str(args.annotations))
        predictions = json.loads(prediction_file.read_text(encoding="utf-8"))
        image_ids = {Path(image["file_name"]).stem: image["id"] for image in gt.dataset["images"]}
        for prediction in predictions:
            key = str(prediction["image_id"])
            if key in image_ids:
                prediction["image_id"] = image_ids[key]
        groups = measure_groups(gt, predictions)
        _, matched = matched_ious(gt, predictions)
        for group in groups:
            group["Mean Matched IoU"] = matched[group["group"]]
        with (args.output_dir / "size_group_metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(groups[0]))
            writer.writeheader()
            writer.writerows(groups)
        with (args.output_dir / "localization_analysis.csv").open("w", newline="", encoding="utf-8") as file:
            rows, _ = matched_ious(gt, predictions)
            columns = list(rows[0]) if rows else ["image_id", "category_id", "score", "best_unmatched_iou"]
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
