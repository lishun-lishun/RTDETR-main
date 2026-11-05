"""Benchmark DLA34-RTDETR raw forward latency and full-model complexity."""

import argparse
import json
import statistics
import time
from pathlib import Path

from dla34_common import CONFIGS, config_path, local_package, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path)
    source.add_argument("--model", choices=CONFIGS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", "--runs", dest="iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.weights and not args.weights.is_file():
        parser.error("weights file does not exist")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations positive")
    if args.output.exists():
        parser.error("output already exists")
    local_package()
    import torch
    from ultralytics import RTDETR
    from ultralytics.nn.tasks import RTDETRDetectionModel

    source_name = str(args.weights) if args.weights else str(config_path(args.model))
    network = (RTDETR(source_name).model if args.weights else
               RTDETRDetectionModel(source_name, nc=1, verbose=False)).to(args.device).eval()
    device = torch.device(args.device)
    if args.precision == "fp16":
        if device.type != "cuda":
            parser.error("fp16 benchmark requires CUDA")
        network = network.half()
    sample = torch.randn(1, 3, 640, 640, device=device,
                         dtype=torch.float16 if args.precision == "fp16" else torch.float32)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            network(sample)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(args.iterations):
            sync()
            start = time.perf_counter()
            network(sample)
            sync()
            timings.append((time.perf_counter() - start) * 1000)
    try:
        import thop
        macs, _ = thop.profile(network, inputs=(sample,), verbose=False)
        macs, gflops = float(macs), float(2 * macs / 1e9)
    except Exception:
        try:
            with torch.profiler.profile(with_flops=True) as profile:
                with torch.inference_mode():
                    network(sample)
            flops = sum((event.flops or 0) for event in profile.key_averages())
            macs, gflops = float(flops / 2), float(flops / 1e9)
        except Exception:
            macs, gflops = None, None
    ordered = sorted(timings)
    report = {"source": source_name, "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "precision": args.precision.upper(), "input_shape": [1, 3, 640, 640],
              "warmup": args.warmup, "iterations": args.iterations,
              "scope": "full RT-DETR forward; excludes preprocessing, transfer and NMS",
              "parameters": sum(p.numel() for p in network.parameters()),
              "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
              "MACs": macs, "GFLOPs": gflops,
              "mean_latency_ms": statistics.mean(timings), "median_latency_ms": statistics.median(timings),
              "p95_latency_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
              "fps_batch1": 1000 / statistics.mean(timings),
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    write_json(args.output, report)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

