"""Collect run status and optional R07 offset diagnostics for R18 variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r18_variants_common import CONFIGS, ROOT, RUNS, config_path, local_package, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--output", type=Path, default=RUNS / "diagnostics")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = {"model": args.model, "yaml": str(config_path(args.model)),
               "weights": str(args.weights) if args.weights else None, "device": args.device,
               "imgsz": args.imgsz}
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    if args.weights and not args.weights.is_file():
        parser.error("weights file does not exist")
    result = {**payload, "checks": {}}
    if args.model == "r18_p4dcn":
        local_package()
        import torch
        from ultralytics import RTDETR

        device = torch.device(f"cuda:{args.device}" if str(args.device).isdecimal() else args.device)
        model = RTDETR(str(args.weights) if args.weights else str(config_path(args.model))).model.to(device).eval()
        offsets = []
        for module in model.modules():
            if hasattr(module, "offset") and hasattr(module.offset, "weight"):
                offsets.append(module.offset)
        if not offsets:
            raise RuntimeError("R07 model contains no offset predictor")
        with torch.inference_mode():
            model(torch.zeros(1, 3, args.imgsz, args.imgsz, device=device))
        result["checks"]["offset_predictors"] = len(offsets)
        result["checks"]["offset_weight_max"] = max(float(item.weight.abs().max()) for item in offsets)
        result["checks"]["offset_bias_max"] = max(float(item.bias.abs().max()) for item in offsets)
        result["checks"]["offset_zero_initialized"] = all(
            float(item.weight.abs().max()) == 0.0 and float(item.bias.abs().max()) == 0.0 for item in offsets)
    run_dir = RUNS / args.model
    result["run_files"] = {"exists": run_dir.exists(),
                            "results": str(run_dir / "seed0" / "results.csv"),
                            "evaluation": str(run_dir / "eval" / "evaluation_metrics.json")}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / f"{args.model}_diagnostics.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
