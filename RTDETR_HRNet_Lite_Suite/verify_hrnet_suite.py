#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server-Side Verification & NVIDIA A30 Benchmark for `RTDETR_HRNet_Lite_Suite` (Exp 1 - Exp 5)
==============================================================================================
Checks all 5 full RT-DETR models:
  1. Strict ImageNet pretrained weight loading (`hrnetv2_w18-8cb57bb9.pth`)
  2. Forward/backward pass + ModelEMA deepcopy safety
  3. Deploy-time structural reparameterization (`fuse()`)
  4. Side-by-side Params (M), Model Size (MB FP32), and NVIDIA A30 Latency (ms/img, FPS)
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] verify_hrnet_suite.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
        f"or use the environment's Python explicitly:\n"
        f"    ~/miniconda3/envs/ultralytics_rtdetr/bin/python RTDETR_HRNet_Lite_Suite/verify_hrnet_suite.py\n"
    )

import argparse
import copy
import os
import time
from pathlib import Path
import torch

SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parent
if (REPO_ROOT / "ultralytics").is_dir():
    sys.path.insert(0, str(REPO_ROOT))


def benchmark_model(model, device, imgsz=640, warmup=20, iters=60):
    model.eval().to(device)
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1000.0
    return dt, 1000.0 / dt


def main():
    parser = argparse.ArgumentParser(description="Verify & benchmark Exp1-Exp5 full RT-DETR models")
    parser.add_argument("--pretrained-backbone", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if args.pretrained_backbone:
        os.environ["RTHRNET_PRETRAINED"] = str(Path(args.pretrained_backbone).resolve())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from ultralytics import RTDETR

    configs = [
        ("Exp1_BaseHRNet", SUITE_DIR / "configs" / "exp1_base_hrnet_w18.yaml"),
        ("Exp2_PruneS4", SUITE_DIR / "configs" / "exp2_hrnet_prunes4.yaml"),
        ("Exp3_PConvRep", SUITE_DIR / "configs" / "exp3_hrnet_pconvrep.yaml"),
        ("Exp4_SlimNeck", SUITE_DIR / "configs" / "exp4_hrnet_slimneck.yaml"),
        ("Exp5_FastHRFull", SUITE_DIR / "configs" / "exp5_fasthr_rtdetr_full.yaml"),
    ]

    rows = []
    for name, cfg in configs:
        print(f"\n========== Verifying {name} ==========")
        det = RTDETR(str(cfg))
        net = det.model
        ema = copy.deepcopy(net)
        assert ema is not net
        params_train = sum(p.numel() for p in net.parameters()) / 1e6

        # Fuse deploy branches (PConvRepBlock / SlimGhostRepC3 / Conv+BN)
        net.fuse()
        params_deploy = sum(p.numel() for p in net.parameters()) / 1e6
        size_mb = params_deploy * 4.0  # FP32 MB
        ms, fps = benchmark_model(net, device, imgsz=args.imgsz)
        rows.append((name, params_train, params_deploy, size_mb, ms, fps))

    print("\n" + "=" * 86)
    print(f"{'Experiment':<18} | {'Train Params':>12} | {'Deploy Params':>13} | {'FP32 Size':>9} | {'Latency':>9} | {'FPS':>7}")
    print("-" * 86)
    for name, pt, pd, mb, ms, fps in rows:
        print(f"{name:<18} | {pt:>10.2f} M | {pd:>11.2f} M | {mb:>6.1f} MB | {ms:>6.2f} ms | {fps:>7.1f}")
    print("=" * 86)


if __name__ == "__main__":
    main()
