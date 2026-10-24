"""Benchmark batch-one model latency, memory and supported profiler FLOPs."""
import argparse
import json
from pathlib import Path
import statistics
import time

from sarep_spasff_common import local_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file() or args.output.exists() or args.runs < 1:
        parser.error('weights must exist; output must be new; runs must be positive')
    local_package()
    import torch
    from ultralytics import RTDETR
    model = RTDETR(str(args.weights)).model.to(args.device).eval()
    tensor = torch.randn(1, 3, 640, 640, device=args.device)
    is_cuda = args.device.startswith('cuda')
    def sync():
        if is_cuda:
            torch.cuda.synchronize(args.device)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(args.device)
    with torch.inference_mode():
        for _ in range(10):
            model(tensor)
        sync()
        times = []
        for _ in range(args.runs):
            sync()
            started = time.perf_counter()
            model(tensor)
            sync()
            times.append(1000 * (time.perf_counter() - started))
        with torch.profiler.profile(with_flops=True) as profile:
            model(tensor)
            sync()
    flops = sum(item.flops for item in profile.key_averages())
    report = {'weights': str(args.weights), 'device': args.device,
              'input_shape': [1, 3, 640, 640], 'parameters': sum(p.numel() for p in model.parameters()),
              'profiler_supported_flops_lower_bound': flops,
              'profiler_supported_macs_lower_bound': flops / 2,
              'mean_latency_ms': statistics.mean(times),
              'median_latency_ms': statistics.median(times),
              'fps_batch1': 1000 / statistics.mean(times),
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(args.device) if is_cuda else None,
              'runs': args.runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
