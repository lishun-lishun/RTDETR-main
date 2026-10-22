#!/usr/bin/env python3
"""Same-device, same-shape RT-DETR forward benchmark with sparse-op timings."""

import argparse
import datetime as dt
import statistics
import time
from pathlib import Path

from scre_splf_common import OUTPUT_ROOT, VARIANTS, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", *VARIANTS), required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--protocol", choices=("scratch", "frozen"), default="scratch")
    parser.add_argument("--project-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch < 1 or args.warmup < 1 or args.repeats < 2:
        parser.error("batch and warmup must be positive; repeats must be >=2")
    weights = args.weights or (args.project_root / args.protocol / args.model / "train" / "weights" / "best.pt")
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")
    output = args.output or (args.project_root / args.protocol / args.model / "benchmark")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing benchmark: {output}")

    import torch
    from torch.profiler import ProfilerActivity, profile
    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import select_device

    device = select_device(args.device)
    module = RTDETR(str(weights)).model.to(device).eval()
    sample = torch.zeros(args.batch, 3, args.imgsz, args.imgsz, device=device)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    with torch.inference_mode():
        for _ in range(args.warmup):
            module.predict(sample)
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        latencies = []
        for _ in range(args.repeats):
            synchronize()
            start = time.perf_counter()
            module.predict(sample)
            synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, with_flops=True) as prof:
            module.predict(sample)
            synchronize()

    events = prof.key_averages()
    sections = {}
    for label in ("SCRE_center_prediction", "SPLF_center_prediction", "SCRE_topk",
                  "SCRE_grid_sample", "SCRE_scatter", "SPLF_projection",
                  "SCRE_fixed_projection"):
        matches = [event for event in events if event.key == label]
        if matches:
            event = matches[0]
            sections[label] = {"cpu_time_ms": event.cpu_time_total / 1000,
                               "cuda_time_ms": getattr(event, "cuda_time_total", 0) / 1000}
    counted_flops = sum(getattr(event, "flops", 0) or 0 for event in events)
    sorted_ms = sorted(latencies)
    result = {
        "model": args.model, "weights": str(weights.resolve()),
        "device": str(device), "batch": args.batch, "imgsz": args.imgsz,
        "warmup": args.warmup, "repeats": args.repeats,
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "parameters": sum(p.numel() for p in module.parameters()),
        "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted_ms[int(0.95 * (len(sorted_ms) - 1))],
        "fps_images_per_second": args.batch * 1000 / statistics.mean(latencies),
        "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
        "profiler_counted_flops": counted_flops,
        "profiler_counted_gflops": counted_flops / 1e9,
        "profiler_macs_approx": counted_flops / 2,
        "flops_scope_warning": "PyTorch profiler counts supported conv/matmul ops only; "
                               "Top-K, grid_sample and scatter may not be included. "
                               "Do not present these as exact whole-model GFLOPs/MACs.",
        "selected_sparse_op_timings": sections,
        "timing_scope": "model.predict forward only; excludes image loading, resizing and NMS",
    }
    output.mkdir(parents=True)
    write_json(output / "benchmark_results.json", result)
    (output / "execution_log.txt").write_text(
        f"[{result['timestamp']}] Benchmark completed for {weights.resolve()}\n", encoding="utf-8")
    print(result)
    print(f"outputs={output}")


if __name__ == "__main__":
    main()
