"""Measure parameters and real inference latency for R09--R16."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from r18_targeted_common import CONFIGS, config_path, local_package, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.weights and not args.weights.is_file(): parser.error("weights file does not exist")
    payload = {"model": args.model, "yaml": str(config_path(args.model)),
               "weights": str(args.weights) if args.weights else None, "device": args.device,
               "imgsz": args.imgsz, "batch": args.batch, "warmup": args.warmup,
               "iterations": args.iters, "amp": args.amp}
    if args.dry_run: print(json.dumps(payload, indent=2)); return
    local_package()
    import torch
    from ultralytics import RTDETR
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdecimal() else args.device)
    model = RTDETR(str(args.weights) if args.weights else str(config_path(args.model))).model.to(device).eval()
    inputs = torch.zeros(args.batch, 3, args.imgsz, args.imgsz, device=device)
    params = sum(p.numel() for p in model.parameters())
    gflops = None
    try:
        from thop import profile
        gflops = float(profile(model, inputs=(inputs,), verbose=False)[0] * 2 / args.batch / 1e9)
    except Exception: pass
    enabled = bool(args.amp and device.type == "cuda")
    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast(device_type=inputs.device.type, dtype=torch.float16, enabled=enabled): model(inputs)
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            with torch.autocast(device_type=inputs.device.type, dtype=torch.float16, enabled=enabled): model(inputs)
        if device.type == "cuda": torch.cuda.synchronize(device)
    latency = (time.perf_counter() - start) * 1000 / args.iters
    result = {**payload, "parameters": params, "GFLOPs": gflops, "latency_ms": latency,
              "FPS": args.batch * 1000 / latency}
    args.output.parent.mkdir(parents=True, exist_ok=True); write_json(args.output, result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__": main()
