#!/usr/bin/env python3
"""Measure Top-K small-GT coverage and protected-feature residual strength."""

import argparse
import datetime as dt
import sys
from pathlib import Path

from scre_splf_common import DATA_CONFIG, OUTPUT_ROOT, VARIANTS, ensure_local_ultralytics, write_json
from scre_splf_metrics import write_csv

ensure_local_ultralytics()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--protocol", choices=("scratch", "frozen"), default="scratch")
    parser.add_argument("--project-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--distance-cells", type=float, default=1.0,
                        help="Coverage radius on the P3 grid; 1 cell = 8 pixels at 640")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    weights = args.weights or args.project_root / args.protocol / args.model / "train" / "weights" / "best.pt"
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    output = args.output or args.project_root / args.protocol / args.model / f"diagnose_{args.split}"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing diagnostics: {output}")

    import torch
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from ultralytics.nn.modules.scre_splf import SCRE, SPLF, SCREFixedFusion
    from ultralytics.nn.tasks import RTDETRDetectionModel

    class DiagnosticValidator(RTDETRValidator):
        latest = None

        def __init__(self, *positional, **keyword):
            super().__init__(*positional, **keyword)
            self.coverage = {k: {"gt": 0, "covered": 0, "distances": []} for k in (16, 32, 64)}
            self.feature_sums = {name: [0.0, 0] for name in ("Tiny", "Small", "Medium", "Large", "Background")}
            self.p4_p5_preserved = True
            self.handles = []
            self.forward_cache = {}
            DiagnosticValidator.latest = self

        def init_metrics(self, model):
            super().init_metrics(model)
            network = next((item for item in model.modules()
                            if isinstance(item, RTDETRDetectionModel)), None)
            if network is None:
                raise RuntimeError("Could not access RTDETRDetectionModel for diagnostic hooks")
            self.scre = next((item for item in network.modules() if isinstance(item, SCRE)), None)
            self.fusion = next((item for item in network.modules()
                                if isinstance(item, (SPLF, SCREFixedFusion))), None)
            decoder = network.model[-1]
            p4_index, p5_index = decoder.f[1:]
            for name, index in (("p4", p4_index), ("p5", p5_index)):
                self.handles.append(network.model[index].register_forward_hook(
                    lambda module, incoming, result, name=name: self.forward_cache.__setitem__(name, result)))
            self.handles.append(decoder.register_forward_pre_hook(
                lambda module, incoming: self.forward_cache.__setitem__("decoder", incoming[0])))

        def update_metrics(self, preds, batch):
            source = self.scre.last if self.scre is not None else self.fusion.last
            if source is None or source.get("logits") is None:
                raise RuntimeError("Candidate logits unavailable during validation forward")
            logits = source["logits"]
            h, w = logits.shape[-2:]
            for image_index in range(logits.shape[0]):
                selected = batch["batch_idx"].flatten() == image_index
                boxes = batch["bboxes"][selected].float().cpu().tolist()
                small_centers = []
                for cx, cy, bw, bh in boxes:
                    if max(bw, bh) * self.args.imgsz <= 16:
                        small_centers.append((cx * w, cy * h))
                scores = logits[image_index, 0].flatten()
                for k, values in self.coverage.items():
                    indices = scores.topk(min(k, scores.numel())).indices
                    candidates = torch.stack(((indices % w).float() + 0.5,
                                              torch.div(indices, w, rounding_mode="floor").float() + 0.5), dim=1)
                    for center in small_centers:
                        gt = torch.tensor(center, device=candidates.device)
                        nearest = float(torch.linalg.vector_norm(candidates - gt, dim=1).min())
                        values["gt"] += 1
                        values["covered"] += nearest <= args.distance_cells
                        values["distances"].append(nearest)

                if self.fusion is not None and self.fusion.last is not None:
                    base = self.fusion.last["base"][image_index]
                    enhanced = self.fusion.last["enhanced"][image_index]
                    amplitude = (enhanced - base).abs().float().mean(0)
                    occupied = torch.zeros((h, w), dtype=torch.bool, device=amplitude.device)
                    for cx, cy, bw, bh in boxes:
                        edge = max(bw, bh) * self.args.imgsz
                        group = ("Tiny" if edge <= 8 else "Small" if edge <= 16
                                 else "Medium" if edge <= 32 else "Large")
                        x0 = max(0, int((cx - bw / 2) * w))
                        x1 = min(w, int((cx + bw / 2) * w + 1))
                        y0 = max(0, int((cy - bh / 2) * h))
                        y1 = min(h, int((cy + bh / 2) * h + 1))
                        region = amplitude[y0:y1, x0:x1]
                        if region.numel():
                            self.feature_sums[group][0] += float(region.sum())
                            self.feature_sums[group][1] += region.numel()
                            occupied[y0:y1, x0:x1] = True
                    background = amplitude[~occupied]
                    self.feature_sums["Background"][0] += float(background.sum())
                    self.feature_sums["Background"][1] += background.numel()
            decoder = self.forward_cache.get("decoder")
            self.p4_p5_preserved &= bool(decoder is not None and
                                         decoder[1] is self.forward_cache.get("p4") and
                                         decoder[2] is self.forward_cache.get("p5"))
            super().update_metrics(preds, batch)

    model = RTDETR(str(weights))
    model.val(data=str(DATA_CONFIG), split=args.split, batch=args.batch, imgsz=640,
              device=args.device, workers=4, save_json=False, plots=False,
              validator=DiagnosticValidator, project=str(output.parent), name=output.name, exist_ok=True)
    validator = DiagnosticValidator.latest
    if validator is None:
        raise RuntimeError("Diagnostic validator did not run")
    for handle in validator.handles:
        handle.remove()
    rows = []
    for k, value in validator.coverage.items():
        rows.append({"topk": k, "small_gt_count": value["gt"],
                     "covered_small_gt": value["covered"],
                     "uncovered_small_gt": value["gt"] - value["covered"],
                     "candidate_recall": value["covered"] / max(value["gt"], 1),
                     "average_nearest_distance_cells": (sum(value["distances"]) / len(value["distances"])
                                                        if value["distances"] else None),
                     "coverage_radius_cells": args.distance_cells})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "candidate_recall.csv", rows)
    write_json(output / "feature_statistics.json", {
        "model": args.model, "weights": str(weights.resolve()), "split": args.split,
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "mean_absolute_p3_residual": {key: total / max(count, 1)
                                     for key, (total, count) in validator.feature_sums.items()},
        "p4_p5_identity_preserved_in_same_forward": validator.p4_p5_preserved,
        "candidate_recall": rows,
        "note": "Top64 is counterfactual ranking when the configured SCRE samples only Top32; not AP.",
    })
    (output / "execution_log.txt").write_text(
        f"[{dt.datetime.now().astimezone().isoformat()}] Diagnostics completed\n", encoding="utf-8")
    print(rows)
    print(f"p4_p5_preserved={validator.p4_p5_preserved}")
    print(f"outputs={output}")


if __name__ == "__main__":
    main()
