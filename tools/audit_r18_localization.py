"""Compute AP50--AP95 and per-instance localization errors from predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax-aw/2, ay-ah/2, ax+aw/2, ay+ah/2
    bx1, by1, bx2, by2 = bx-bw/2, by-bh/2, bx+bw/2, by+bh/2
    inter = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1)); union = aw*ah+bw*bh-inter
    return inter/union if union > 0 else 0.0


def _predictions(path: Path):
    value = json.loads(path.read_text(encoding="utf-8")); value = value.get("predictions", value.get("results", [])) if isinstance(value, dict) else value
    output = {}
    for record in value:
        key = str(record.get("image", record.get("file", record.get("path", record.get("image_id", "")))))
        boxes = record.get("boxes", record.get("xywhn", [])); boxes = boxes.get("xywhn", boxes.get("xywh", boxes.get("xyxy", []))) if isinstance(boxes, dict) else boxes
        scores = record.get("scores", record.get("conf", record.get("confidence", []))); classes = record.get("classes", record.get("class", record.get("cls", [])))
        if not isinstance(scores, list): scores = [scores] * len(boxes)
        if not isinstance(classes, list): classes = [classes] * len(boxes)
        output[key] = [{"box": [float(x) for x in box], "score": float(score), "class": int(cls)} for box, score, cls in zip(boxes, scores, classes) if len(box) == 4]
    return output


def _labels(path: Path):
    output = {}
    for file in path.rglob("*.txt") if path.exists() else []:
        rows = []
        for line in file.read_text(encoding="utf-8", errors="ignore").splitlines():
            f = line.split()
            if len(f) == 5:
                try: rows.append({"class": int(f[0]), "box": [float(x) for x in f[1:5]]})
                except ValueError: pass
        output[file.stem] = rows
    return output


def audit(prediction_path: Path, label_dir: Path):
    predictions, ground_truth = _predictions(prediction_path), _labels(label_dir); thresholds = [x/100 for x in range(50,100,5)]; matches = {t: [] for t in thresholds}; instances = []
    for key, preds in predictions.items():
        gts = ground_truth.get(Path(key).stem, []); used = set()
        for pred in sorted(preds, key=lambda x: -x["score"]):
            candidates = [(_iou(pred["box"], gt["box"]), index) for index, gt in enumerate(gts) if index not in used and gt["class"] == pred["class"]]; best, index = max(candidates, default=(0.0, None))
            if index is not None: used.add(index)
            for threshold in thresholds: matches[threshold].append(int(best >= threshold))
            if index is not None:
                gt = gts[index]; px, py, pw, ph = pred["box"]; gx, gy, gw, gh = gt["box"]
                instances.append({"image": key, "iou": best, "score": pred["score"], "center_error": ((px-gx)**2+(py-gy)**2)**0.5, "width_error": abs(pw-gw), "height_error": abs(ph-gh), "small": max(gw,gh) <= 0.05})
    small = [x for x in instances if x["small"]]
    mean = lambda key, rows=instances: sum(x[key] for x in rows)/len(rows) if rows else 0.0
    return {"predictions": len(predictions), "ground_truth_images": len(ground_truth), "metrics": {f"AP{int(t*100):02d}": sum(values)/len(values) if values else 0.0 for t, values in matches.items()}, "matched_instances": len(instances), "mean_best_iou": mean("iou"), "mean_center_error": mean("center_error"), "mean_width_error": mean("width_error"), "mean_height_error": mean("height_error"), "small_object_count": len(small), "small_object_mean_iou": mean("iou", small), "instances": instances}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--labels", type=Path, required=True); parser.add_argument("--output", type=Path, default=Path("reports/r18_targeted_localization.json")); args = parser.parse_args()
    result = audit(args.predictions, args.labels); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"); print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
