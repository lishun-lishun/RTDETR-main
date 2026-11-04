"""Evaluate MIRNet-Det on val/test and optionally compute custom size groups."""

import argparse
import copy
import csv
import json
from pathlib import Path

from mirnet_det_common import CONFIGS, DATA, config_path, local_package, write_json


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def group_name(annotation, image):
    ratio = min(640 / image["width"], 640 / image["height"])
    side = max(annotation["bbox"][2:4]) * ratio
    return "Tiny" if side <= 8 else "Small" if side <= 16 else "Medium" if side <= 32 else "Large"


def coco_metrics(gt, dt, area=None):
    from pycocotools.cocoeval import COCOeval

    evaluator = COCOeval(gt, dt, "bbox")
    if area is not None:
        evaluator.params.areaRng = [[area[0], area[1]]]
        evaluator.params.areaRngLbl = ["custom"]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {"AP50:95": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]),
            "Recall": float(stats[8] if len(stats) > 8 else stats[6])}


def measure_groups(gt, predictions):
    from pycocotools.coco import COCO

    data = copy.deepcopy(gt.dataset)
    images = {item["id"]: item for item in data["images"]}

    def proxy_area(ann):
        image = images[ann["image_id"]]
        ratio = min(640 / image["width"], 640 / image["height"])
        return (max(ann["bbox"][2:4]) * ratio) ** 2

    for ann in data["annotations"]:
        ann["area"] = proxy_area(ann)
    scaled_gt = COCO()
    scaled_gt.dataset = data
    scaled_gt.createIndex()
    scaled_dt = scaled_gt.loadRes(predictions)
    for ann in scaled_dt.dataset["annotations"]:
        ann["area"] = proxy_area(ann)
    scaled_dt.createIndex()
    groups = []
    for name, low, high in (("Tiny", 0, 64), ("Small", 64, 256), ("Medium", 256, 1024), ("Large", 1024, 1e12)):
        row = {"group": name, "GT Count": sum(
            (low < ann["area"] <= high) if low else ann["area"] <= high
            for ann in data["annotations"] if not ann.get("iscrowd", 0))}
        row.update(coco_metrics(scaled_gt, scaled_dt, (low + (1e-6 if low else 0), high)))
        groups.append(row)
    return groups


def matched_ious(gt, predictions):
    images = {item["id"]: item for item in gt.dataset["images"]}
    annotations, detections = {}, {}
    for ann in gt.dataset["annotations"]:
        if not ann.get("iscrowd", 0):
            annotations.setdefault((ann["image_id"], ann["category_id"]), []).append(ann)
    for pred in predictions:
        detections.setdefault((pred["image_id"], pred["category_id"]), []).append(pred)
    rows, matched = [], {name: [] for name in ("Tiny", "Small", "Medium", "Large")}
    for key, preds in detections.items():
        ground, used = annotations.get(key, []), set()
        for pred in sorted(preds, key=lambda item: item["score"], reverse=True):
            candidates = [(iou(pred["bbox"], ann["bbox"]), index, ann)
                          for index, ann in enumerate(ground) if index not in used]
            best = max(candidates, default=(0.0, -1, None), key=lambda item: item[0])
            rows.append({"image_id": key[0], "category_id": key[1], "score": pred["score"],
                         "best_unmatched_iou": best[0]})
            if best[0] >= 0.5:
                used.add(best[1])
                matched[group_name(best[2], images[key[0]])].append(best[0])
    return rows, {name: sum(values) / len(values) if values else None for name, values in matched.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--annotations", type=Path,
                        help="Optional COCO JSON for custom Tiny/Small/Medium/Large metrics")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        parser.error("weights and dataset YAML must exist")
    if args.annotations and not args.annotations.is_file():
        parser.error("annotations file does not exist")
    if args.output_dir.exists() and not args.dry_run:
        parser.error("output directory already exists")
    payload = {"model": args.model, "yaml": str(config_path(args.model)), "weights": str(args.weights),
               "data": str(args.data), "annotations": str(args.annotations) if args.annotations else None,
               "split": args.split, "device": args.device, "imgsz": 640, "max_det": 300}
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    local_package()
    from ultralytics import RTDETR

    result = RTDETR(str(args.weights)).val(data=str(args.data), split=args.split, device=args.device,
                                           batch=8, imgsz=640, max_det=300, save_json=bool(args.annotations),
                                           project=str(args.output_dir.parent), name=args.output_dir.name,
                                           exist_ok=False)
    native = dict(getattr(result, "results_dict", {}))
    mAP75 = None
    if hasattr(result, "box") and getattr(result.box, "all_ap", None) is not None:
        mAP75 = float(result.box.all_ap[:, 5].mean())
    metrics = {**payload, "precision": native.get("metrics/precision(B)"),
               "recall": native.get("metrics/recall(B)"), "mAP50": native.get("metrics/mAP50(B)"),
               "mAP75": mAP75, "mAP50:95": native.get("metrics/mAP50-95(B)"), "ultralytics": native}
    write_json(args.output_dir / "evaluation_metrics.json", metrics)
    groups, rows = [], []
    if args.annotations:
        from pycocotools.coco import COCO

        gt = COCO(str(args.annotations))
        prediction_file = Path(result.save_dir) / "predictions.json"
        if not prediction_file.is_file():
            raise RuntimeError(f"Ultralytics did not create {prediction_file}")
        predictions = json.loads(prediction_file.read_text(encoding="utf-8"))
        ids = {Path(image["file_name"]).stem: image["id"] for image in gt.dataset["images"]}
        if len(ids) != len(gt.dataset["images"]):
            raise RuntimeError("annotation image stems are not unique")
        for pred in predictions:
            stem = str(pred["image_id"])
            if stem not in ids:
                raise RuntimeError(f"prediction image ID not found in annotations: {stem}")
            pred["image_id"] = ids[stem]
        groups = measure_groups(gt, predictions)
        rows, matched = matched_ious(gt, predictions)
        for group in groups:
            group["Mean Matched IoU"] = matched[group["group"]]
    with (args.output_dir / "size_group_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        columns = list(groups[0]) if groups else ["group", "GT Count", "AP50", "AP75", "AP50:95", "Recall", "Mean Matched IoU"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(groups)
    with (args.output_dir / "localization_analysis.csv").open("w", newline="", encoding="utf-8") as file:
        columns = list(rows[0]) if rows else ["image_id", "category_id", "score", "best_unmatched_iou"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"metrics": metrics, "groups": groups}, indent=2, default=str))


if __name__ == "__main__":
    main()
