"""Measure raw FP32 model forward speed on 1x3x640x640, excluding I/O and NMS."""
import argparse
import json
from pathlib import Path
import statistics
import time

from res2_gd_common import local_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file() or args.output.exists() or args.runs < 1 or args.warmup < 0:
        parser.error('weights must exist, output must be new, runs positive, warmup nonnegative')
    local_package()
    import torch
    from ultralytics import RTDETR

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    network = RTDETR(str(args.weights)).model.to(device).float().eval()
    sample = torch.randn(1, 3, 640, 640, device=device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            network(sample)
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(args.runs):
            sync()
            start = time.perf_counter()
            network(sample)
            sync()
            times.append((time.perf_counter() - start) * 1000)
        with torch.profiler.profile(with_flops=True) as profile:
            network(sample)
            sync()
    flops = sum(row.flops for row in profile.key_averages())
    report = {
        'weights': str(args.weights), 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'precision': 'FP32', 'input_shape': [1, 3, 640, 640],
        'batch_size': 1, 'warmup': args.warmup, 'runs': args.runs,
        'scope': 'raw network forward; excludes decoding NMS, preprocessing and image transfer',
        'parameters': sum(p.numel() for p in network.parameters()),
        'profiler_supported_flops_lower_bound': flops,
        'profiler_supported_macs_lower_bound': flops / 2,
        'mean_latency_ms': statistics.mean(times),
        'median_latency_ms': statistics.median(times),
        'fps_batch1': 1000 / statistics.mean(times),
        'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(device)
        if device.type == 'cuda' else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
