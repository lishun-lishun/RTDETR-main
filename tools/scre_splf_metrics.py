"""One-class confidence-ranked detection AP for fixed input-pixel size groups.

Predictions that match out-of-group GT at the current IoU threshold are ignored
for that group's AP (COCO-style area-range behavior). Background detections and
duplicates of in-group GT remain false positives. This is a separate custom
diagnostic, not a replacement for the project's official overall mAP.
"""

import csv
from pathlib import Path


GROUPS = ("Tiny", "Small", "Medium", "Large")


def size_group(width, height):
    edge = max(width, height)
    return "Tiny" if edge <= 8 else "Small" if edge <= 16 else "Medium" if edge <= 32 else "Large"


def iou_xyxy(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def ap101(detections, gt_count):
    if not gt_count:
        return None, None
    detections.sort(key=lambda item: item[0], reverse=True)
    tp = fp = 0
    recalls, precisions = [], []
    for _, matched, _ in detections:
        tp += int(matched)
        fp += int(not matched)
        recalls.append(tp / gt_count)
        precisions.append(tp / (tp + fp))
    ap = sum(max((p for r, p in zip(recalls, precisions) if r >= step / 100), default=0.0)
             for step in range(101)) / 101
    return ap, (tp / gt_count if detections else 0.0)


def evaluate_group(records, group, threshold, fp_confidence=0.25):
    gt_count = sum(sum(label == group for label in record["gt_groups"]) for record in records)
    detections = []
    matched_ious = []
    false_positives = 0
    for record in records:
        boxes = record["gt_boxes"]
        labels = record["gt_groups"]
        used = set()
        predictions = sorted(zip(record["pred_boxes"], record["scores"]),
                             key=lambda item: item[1], reverse=True)
        for predicted, confidence in predictions:
            best_iou, best_index = 0.0, None
            for index, (truth, label) in enumerate(zip(boxes, labels)):
                if label != group or index in used:
                    continue
                overlap = iou_xyxy(predicted, truth)
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            if best_index is not None and best_iou >= threshold:
                used.add(best_index)
                detections.append((confidence, True, best_iou))
                if confidence >= fp_confidence:
                    matched_ious.append(best_iou)
                continue
            if any(iou_xyxy(predicted, truth) >= threshold for truth, label in zip(boxes, labels)
                   if label != group):
                continue
            detections.append((confidence, False, 0.0))
            false_positives += confidence >= fp_confidence
    ap, recall = ap101(detections, gt_count)
    return {"gt_count": gt_count, "ap": ap, "recall": recall,
            "matched_iou": (sum(matched_ious) / len(matched_ious) if matched_ious else None),
            "false_positives": false_positives}


def group_summary(records, fp_confidence=0.25):
    rows = []
    for group in GROUPS:
        values = [evaluate_group(records, group, round(0.5 + i * 0.05, 2), fp_confidence)
                  for i in range(10)]
        ap_values = [value["ap"] for value in values if value["ap"] is not None]
        rows.append({"group": group, "gt_count": values[0]["gt_count"],
                     "ap50": values[0]["ap"], "ap75": values[5]["ap"],
                     "ap50_95": (sum(ap_values) / len(ap_values) if ap_values else None),
                     "recall50": values[0]["recall"],
                     "mean_matched_iou_at_conf": values[0]["matched_iou"],
                     "false_positives_at_conf": values[0]["false_positives"],
                     "fp_confidence": fp_confidence})
    return rows


def write_csv(path, rows):
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
