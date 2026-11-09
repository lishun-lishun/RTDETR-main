"""Benchmark baseline R18 and the three HRD-R18 R17 variants.

Measurements use batch=1 and 640x640 input.  The command is dry-run by
default; pass --execute to run 50 warm-up iterations and 200 timed FP16
iterations on the requested GPU.  Results are written as JSON, CSV, and
Markdown so comparisons use one consistent protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
CONFIGS = {
    "baseline": "rtdetr-r18.yaml",
    "r17a_cbr": "rtdetr-r17a-cbr.yaml",
    "r17b_hrd_512": "rtdetr-r17b-hrd-512.yaml",
    "r17c_hrd_cbr": "rtdetr-r17c-hrd-cbr.yaml",
}

def local_device(value):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    ids = [x.strip() for x in visible.split(",") if x.strip()]
    if value.isdigit() and ids:
        if len(ids) == 1:
            return "0"
        if value in ids:
            return str(ids.index(value))
    return value


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lo, hi = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def benchmark_one(name, path, device_id, imgsz, warmup, iters, amp):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    from ultralytics import RTDETR

    device = torch.device(f"cuda:{device_id}" if str(device_id).isdigit() else device_id)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model = RTDETR(str(path)).model.to(device).eval()
    inputs = torch.zeros(1, 3, imgsz, imgsz, device=device)
    parameters = sum(p.numel() for p in model.parameters())
    gflops = None
    try:
        from thop import profile
        gflops = float(profile(model, inputs=(inputs,), verbose=False)[0] * 2 / 1e9)
    except Exception as exc:
        gflops_error = f"{type(exc).__name__}: {exc}"
    else:
        gflops_error = None
    enabled = bool(amp and device.type == "cuda")
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=enabled):
                model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(iters):
            start = time.perf_counter()
            with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=enabled):
                model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    mean_ms = statistics.fmean(timings)
    result = {
        "model": name,
        "yaml": str(path),
        "device": str(device),
        "batch": 1,
        "imgsz": imgsz,
        "amp_fp16": enabled,
        "warmup": warmup,
        "iterations": iters,
        "parameters": parameters,
        "parameters_m": parameters / 1e6,
        "GFLOPs": gflops,
        "GFLOPs_error": gflops_error,
        "latency_mean_ms": mean_ms,
        "latency_p95_ms": percentile(timings, 0.95),
        "FPS": 1000.0 / mean_ms if mean_ms > 0 else None,
        "gpu_memory_peak_mb": (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                                if device.type == "cuda" else None),
    }
    return result


def write_outputs(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "r17_hrd_benchmark.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    fields = ["model", "parameters_m", "GFLOPs", "latency_mean_ms", "latency_p95_ms", "FPS", "gpu_memory_peak_mb"]
    with (output_dir / "r17_hrd_benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in results)
    lines = [
        "# R17 HRD benchmark",
        "",
        "Protocol: batch=1, input=640x640, FP16 autocast on CUDA, 50 warm-up iterations, 200 timed iterations.",
        "",
        "| Model | Params (M) | GFLOPs | Mean latency (ms) | P95 latency (ms) | FPS | Peak GPU MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        def f(key):
            value = row.get(key)
            return "NA" if value is None else f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
        lines.append(f"| {row['model']} | {f('parameters_m')} | {f('GFLOPs')} | {f('latency_mean_ms')} | {f('latency_p95_ms')} | {f('FPS')} | {f('gpu_memory_peak_mb')} |")
    (output_dir / "r17_hrd_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="comma-separated names or all")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "r17_hrd_benchmark")
    parser.add_argument("--fp32", action="store_true", help="disable FP16 autocast")
    parser.add_argument("--execute", action="store_true", help="run the benchmark; default is dry-run")
    args = parser.parse_args()
    args.device = local_device(args.device)
    names = list(CONFIGS) if args.models == "all" else [x.strip() for x in args.models.split(",") if x.strip()]
    unknown = [x for x in names if x not in CONFIGS]
    if unknown:
        parser.error(f"unknown models {unknown}; choose from {tuple(CONFIGS)}")
    missing = [name for name in names if not (MODEL_DIR / CONFIGS[name]).is_file()]
    if missing:
        parser.error(f"missing R17 YAML files for {missing}")
    payload = {"models": names, "device": args.device, "imgsz": args.imgsz, "warmup": args.warmup,
               "iterations": args.iters, "amp_fp16": not args.fp32,
               "output_dir": str(args.output_dir),
               "configs": {name: str(MODEL_DIR / CONFIGS[name]) for name in names}}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not args.execute:
        print("Dry run only. Add --execute to measure latency.")
        return
    results = [benchmark_one(name, MODEL_DIR / CONFIGS[name], args.device, args.imgsz,
                             args.warmup, args.iters, not args.fp32) for name in names]
    write_outputs(results, args.output_dir)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
