#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Evaluation Script for RT-DETR UAV Paper Innovations
===========================================================
Evaluates trained checkpoints on Validation (`split=val`) or Test (`split=test`,
2200 images, 2245 instances on DUT-Anti-UAV) and computes both standard Ultralytics
metrics AND scale-specific COCO / Custom Tiny AP metrics:
- Precision, Recall, F1
- mAP50, mAP75, mAP50:95
- COCO Small AP (`area < 32^2 = 1024 px^2` at 640x640)
- COCO Medium AP (`32^2 <= area < 96^2 px^2`)
- COCO Large AP (`area >= 96^2 px^2`)
- Custom Tiny AP (`area < 16^2 = 256 px^2`, AI-TOD Tiny definition)
  and Very-Tiny AP (`area < 8^2 = 64 px^2`, AI-TOD Very-Tiny definition)

Explicitly distinguishes COCO pixel-area definitions from custom Tiny thresholds!
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any

SCALE_DEFINITIONS_PX2 = {
    "Custom_VeryTiny_AI_TOD": (0.0, float(8 * 8)),        # [0, 64) px^2 at 640x640
    "Custom_Tiny_AI_TOD": (0.0, float(16 * 16)),          # [0, 256) px^2 at 640x640
    "COCO_Small": (0.0, float(32 * 32)),                  # [0, 1024) px^2 at 640x640
    "COCO_Medium": (float(32 * 32), float(96 * 96)),      # [1024, 9216) px^2 at 640x640
    "COCO_Large": (float(96 * 96), float(1e10)),          # [9216, inf) px^2 at 640x640
}


def compute_extended_coco_scale_metrics(anno_json: str, pred_json: str) -> Dict[str, float]:
    """
    Runs pycocotools COCOeval with standard COCO areaRng AND custom Tiny areaRng (<16x16 px).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(anno_json)
    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "COCO_mAP50_95": float(coco_eval.stats[0]),
        "COCO_mAP50": float(coco_eval.stats[1]),
        "COCO_mAP75": float(coco_eval.stats[2]),
        "COCO_AP_Small_lt32x32": float(coco_eval.stats[3]),
        "COCO_AP_Medium_32to96": float(coco_eval.stats[4]),
        "COCO_AP_Large_gt96x96": float(coco_eval.stats[5]),
    }

    # Evaluate Custom Tiny (<16x16 = 256 px^2) and Very-Tiny (<8x8 = 64 px^2)
    for label, (amin, amax) in [
        ("Custom_AP_VeryTiny_lt8x8", SCALE_DEFINITIONS_PX2["Custom_VeryTiny_AI_TOD"]),
        ("Custom_AP_Tiny_lt16x16", SCALE_DEFINITIONS_PX2["Custom_Tiny_AI_TOD"]),
    ]:
        eval_custom = COCOeval(coco_gt, coco_dt, "bbox")
        eval_custom.params.areaRng = [[0.0, 1e10], [amin, amax], [amin, amax], [amin, amax]]
        eval_custom.params.areaRngLbl = ["all", "small", "medium", "large"]
        eval_custom.evaluate()
        eval_custom.accumulate()
        eval_custom.summarize()
        metrics[label] = float(eval_custom.stats[3])

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation for RT-DETR UAV Innovations")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/dut-anti-uav.yaml", help="Dataset YAML")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split ('val' for model selection, 'test' for final reporting)")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image resolution (default: 640)")
    parser.add_argument("--batch", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--anno-json", type=str, default="", help="Optional COCO ground-truth JSON for scale-specific AP")
    parser.add_argument("--output-json", type=str, default="evaluation_summary.json", help="Output JSON report path")
    parser.add_argument("--dry-run", action="store_true", help="Print scale definitions and evaluation protocol without running GPU val")
    args = parser.parse_args()

    protocol_info: Dict[str, Any] = {
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "scale_definitions_px2_at_640x640": SCALE_DEFINITIONS_PX2,
    }

    if args.dry_run:
        print(json.dumps(protocol_info, indent=2, ensure_ascii=False))
        return

    from ultralytics import RTDETR

    model = RTDETR(args.weights)
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_json=True,
    )

    summary = {
        "protocol": protocol_info,
        "Precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
        "Recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP75": float(getattr(getattr(results, "box", None), "map75", results.results_dict.get("metrics/mAP75(B)", 0.0))),
        "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
    }

    pred_json = Path(results.save_dir) / "predictions.json"
    if args.anno_json and Path(args.anno_json).exists() and pred_json.exists():
        summary["coco_and_custom_tiny_metrics"] = compute_extended_coco_scale_metrics(
            args.anno_json, str(pred_json)
        )

    Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
