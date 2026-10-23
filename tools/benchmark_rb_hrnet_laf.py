"""Measure parameters, supported profiler FLOPs and batch-one latency."""
import argparse
import json
from pathlib import Path
import time

import torch
from train_rb_hrnet_laf import CONFIGS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=CONFIGS)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--runs', type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output file already exists')
    if args.weights and not args.weights.is_file():
        parser.error('weights file does not exist')
    from ultralytics import RTDETR
    source = (args.weights if args.weights else
              ROOT / 'ultralytics/cfg/models/rt-detr' / CONFIGS[args.model])
    model = RTDETR(str(source)).model.to(args.device).eval()
    x = torch.randn(1, 3, 640, 640, device=args.device)
    def sync():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(5):
            model(x)
        sync()
        times = []
        for _ in range(args.runs):
            start = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - start) * 1000)
        with torch.profiler.profile(with_flops=True) as profile:
            model(x)
    flops = sum(item.flops for item in profile.key_averages())
    mean_ms = sum(times) / len(times)
    result = {'model': args.model, 'weights': str(args.weights) if args.weights else None,
              'parameters': sum(p.numel() for p in model.parameters()),
              'profiler_flops_lower_bound': flops,
              'profiler_macs_lower_bound': flops / 2,
              'mean_latency_ms': mean_ms, 'fps_batch1': 1000 / mean_ms,
              'peak_gpu_bytes': (torch.cuda.max_memory_allocated()
                                 if args.device.startswith('cuda') else None),
              'input': [1, 3, 640, 640],
              'scope': 'PyTorch profiler supported operations only'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
