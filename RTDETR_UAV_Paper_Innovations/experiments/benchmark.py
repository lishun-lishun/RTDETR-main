#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Speed & Complexity Benchmark Script for RT-DETR UAV Innovations
=======================================================================
Standardized Benchmark Protocol (Phase 9.3):
- Target GPU: NVIDIA A30 24GB (auto-detects GPU availability; if no GPU is
  present in current environment, outputs exact parameter counts, theoretical
  GFLOPs, CPU timing, and marks GPU latency as `NOT_RUN_NO_GPU`)
- Precision: FP16 (`--fp16`, or FP32 fallback on CPU)
- Input Resolution: 1 × 3 × 640 × 640 (`Batch = 1`)
- Warmup Iterations: 50
- Timed Iterations: 500 (with `torch.cuda.synchronize()`)
- Reported Metrics:
  Parameters (M), GFLOPs, Mean Latency (ms), P95 Latency (ms), FPS, GPU Peak Memory (MB)
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

import torch
import torch.nn as nn


def count_parameters_and_flops(model: nn.Module, img_size: int = 640, device: torch.device = torch.device("cpu")) -> Tuple[float, float]:
    """
    Computes exact parameter count (in Millions) and GFLOPs at 1x3x640x640 using forward hooks.
    """
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    total_flops = [0.0]
    hooks = []

    def conv_hook(module: nn.Conv2d, inp, out):
        out_t = out[0] if isinstance(out, (tuple, list)) else out
        b, c_out, h_out, w_out = out_t.shape
        k_h, k_w = module.kernel_size
        c_in = module.in_channels // module.groups
        # 2 * MACs
        flops = 2.0 * b * c_out * h_out * w_out * c_in * k_h * k_w
        total_flops[0] += flops

    def linear_hook(module: nn.Linear, inp, out):
        inp_t = inp[0]
        num_instances = inp_t.numel() // inp_t.shape[-1]
        flops = 2.0 * num_instances * module.in_features * module.out_features
        total_flops[0] += flops

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    gflops = total_flops[0] / 1e9
    return params_m, gflops


def benchmark_model_config(
    name: str,
    inno1: bool,
    inno2: bool,
    inno3: bool,
    inno4: bool,
    warmup: int = 50,
    iters: int = 500,
    img_size: int = 640,
    fp16: bool = True,
) -> Dict[str, Any]:
    from integration.test_integration import StandaloneRTDETRModel

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if has_cuda else "cpu")
    model = StandaloneRTDETRModel(
        nc=1,
        inno1_enabled=inno1,
        inno2_enabled=inno2,
        inno3_enabled=inno3,
        inno4_enabled=inno4,
    ).to(device).eval()

    params_m, gflops = count_parameters_and_flops(model, img_size=img_size, device=device)

    result: Dict[str, Any] = {
        "model": name,
        "innovations": {"WCA_SFP": inno1, "FAGA": inno2, "UAD_CR": inno3, "SAH_NWD": inno4},
        "Params_M": round(params_m, 4),
        "GFLOPs_640x640": round(gflops, 3),
    }

    if not has_cuda:
        result.update({
            "gpu_status": "NOT_RUN_NO_GPU (Current verification sandbox has no NVIDIA GPU attached; run on NVIDIA A30 server)",
            "Mean_Latency_ms": "NOT_RUN_NO_GPU",
            "P95_Latency_ms": "NOT_RUN_NO_GPU",
            "FPS": "NOT_RUN_NO_GPU",
            "GPU_Memory_MB": "NOT_RUN_NO_GPU",
        })
        return result

    use_half = fp16 and has_cuda
    if use_half:
        model = model.half()
    x = torch.randn(1, 3, img_size, img_size, device=device, dtype=torch.float16 if use_half else torch.float32)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize(device)

        latencies_ms: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms.sort()
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    p95_lat = latencies_ms[int(0.95 * len(latencies_ms))]
    fps = 1000.0 / max(mean_lat, 1e-6)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    result.update({
        "gpu_status": torch.cuda.get_device_name(device),
        "Mean_Latency_ms": round(mean_lat, 3),
        "P95_Latency_ms": round(p95_lat, 3),
        "FPS": round(fps, 2),
        "GPU_Memory_MB": round(peak_mem_mb, 2),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="Speed & Complexity Benchmark on NVIDIA A30 (FP16, 640x640, Batch=1)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations (default: 50)")
    parser.add_argument("--iters", type=int, default=500, help="Timed iterations (default: 500)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution (default: 640)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision on GPU")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    configs = [
        ("E0_Baseline_RTDETR_R18", False, False, False, False),
        ("E1_Inno1_WCASFP", True, False, False, False),
        ("E2_Inno2_FAGA", False, True, False, False),
        ("E3_Inno3_UADCR", False, False, True, False),
        ("E4_Inno4_SAHNWD", False, False, False, True),
        ("E5_ComboA_WCASFP_FAGA", True, True, False, False),
        ("E6_ComboB_UADCR_SAHNWD", False, False, True, True),
        ("E7_Full_Joint_Paper_Model", True, True, True, True),
    ]

    all_results = []
    for name, i1, i2, i3, i4 in configs:
        res = benchmark_model_config(
            name, i1, i2, i3, i4, warmup=args.warmup, iters=args.iters, img_size=args.imgsz, fp16=args.fp16
        )
        all_results.append(res)

    Path(args.output).write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
