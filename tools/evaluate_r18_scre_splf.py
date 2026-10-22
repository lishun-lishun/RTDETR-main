#!/usr/bin/env python3
"""Evaluate overall RT-DETR metrics and confidence-ranked DUT size-group AP."""

import argparse
import datetime as dt
import sys
from pathlib import Path

from scre_splf_common import DATA_CONFIG, OUTPUT_ROOT, VARIANTS, write_json
from scre_splf_metrics import group_summary, size_group, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", *VARIANTS), required=True)
    parser.add_argument("--weights", type=Path, help="Explicit best.pt; required for baseline")
    parser.add_argument("--protocol", choices=("scratch", "frozen"), default="scratch")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fp-confidence", type=float, default=0.25)
    parser.add_argument("--project-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model == "baseline" and args.weights is None:
        raise SystemExit("Baseline evaluation requires --weights /path/to/best.pt")
    weights = args.weights or (args.project_root / args.protocol / args.model / "train" / "weights" / "best.pt")
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    output = args.output or (args.project_root / args.protocol / args.model / f"eval_{args.split}")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing evaluation: {output}")

    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.val import RTDETRValidator

    class GroupValidator(RTDETRValidator):
        latest = None

        def __init__(self, *positional, **keyword):
            super().__init__(*positional, **keyword)
            self.records = []
            GroupValidator.latest = self

        def update_metrics(self, preds, batch):
            side = self.args.imgsz
            for image_index, pred in enumerate(preds):
                selected = batch["batch_idx"].flatten() == image_index
                boxes = batch["bboxes"][selected].float().cpu().tolist()
                truths, groups = [], []
                for cx, cy, bw, bh in boxes:
                    truths.append([(cx - bw / 2) * side, (cy - bh / 2) * side,
                                   (cx + bw / 2) * side, (cy + bh / 2) * side])
                    groups.append(size_group(bw * side, bh * side))
                raw = pred.float().cpu().tolist()
                self.records.append({"image": str(batch["im_file"][image_index]),
                                     "gt_boxes": truths, "gt_groups": groups,
                                     "pred_boxes": [item[:4] for item in raw],
                                     "scores": [item[4] for item in raw]})
            super().update_metrics(preds, batch)

    started = dt.datetime.now().astimezone()
    model = RTDETR(str(weights))
    metrics = model.val(data=str(DATA_CONFIG), split=args.split, batch=args.batch,
                        imgsz=args.imgsz, device=args.device, workers=4, max_det=300,
                        save_json=False, plots=False, validator=GroupValidator,
                        project=str(output.parent), name=output.name, exist_ok=True)
    validator = GroupValidator.latest
    if validator is None:
        raise RuntimeError("Custom size-group validator did not run")
    output.mkdir(parents=True, exist_ok=True)
    rows = group_summary(validator.records, fp_confidence=args.fp_confidence)
    write_csv(output / "size_group_metrics.csv", rows)
    write_json(output / "evaluation_metrics.json", {
        "model": args.model, "weights": str(weights.resolve()), "split": args.split,
        "data": str(DATA_CONFIG.resolve()), "imgsz": args.imgsz, "batch": args.batch,
        "started_at": started.isoformat(), "finished_at": dt.datetime.now().astimezone().isoformat(),
        "overall_ultralytics": getattr(metrics, "results_dict", {}),
        "speed": getattr(metrics, "speed", {}),
        "custom_size_groups": rows,
        "size_group_definition": "max box side in stretched model-input pixels",
        "custom_ap_note": "101-point confidence-ranked, one-to-one, non-group GT ignored; not official COCO AP",
    })
    with (output / "execution_log.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"[{started.isoformat()}] Evaluation started\n")
        stream.write(f"[{dt.datetime.now().astimezone().isoformat()}] Evaluation finished; images={len(validator.records)}\n")
    print(f"overall={getattr(metrics, 'results_dict', {})}")
    for row in rows:
        print(row)
    print(f"outputs={output}")


if __name__ == "__main__":
    main()
