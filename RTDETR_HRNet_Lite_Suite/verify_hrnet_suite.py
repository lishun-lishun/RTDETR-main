#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair Fused Benchmark on NVIDIA A30: RT-DETR-R18 vs Exp1, Exp6, Exp11, Exp12, Exp13, Exp14
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] verify_hrnet_suite.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import copy
import os
import time
from pathlib import Path
import torch

SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".yolo"))
if (REPO_ROOT / "ultralytics").is_dir():
    sys.path.insert(0, str(REPO_ROOT))
if (REPO_ROOT / ".pypkg").is_dir():
    sys.path.insert(0, str(REPO_ROOT / ".pypkg"))
for fallback_env in [
    r'E:\envs\my_env\Lib\site-packages',
    r'E:\envs\isaaclab\Lib\site-packages',
    r'E:\envs\RareCLIP\Lib\site-packages',
]:
    if os.path.isdir(fallback_env):
        sys.path.append(fallback_env)


def benchmark(model, device, batch, half, imgsz=640, warmup=15, iters=40):
    m = model.eval().to(device)
    if half and device.type == "cuda":
        m = m.half()
    x = torch.randn(batch, 3, imgsz, imgsz, device=device, dtype=torch.float16 if (half and device.type == "cuda") else torch.float32)
    with torch.no_grad():
        for _ in range(warmup):
            _ = m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms_per_img = (time.perf_counter() - t0) / (iters * batch) * 1000.0
    return ms_per_img


def main():
    parser = argparse.ArgumentParser(description="Fair fused A30 benchmark including RT-DETR-R18 & Exp13-14 Turbo")
    parser.add_argument("--pretrained-backbone", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if args.pretrained_backbone:
        os.environ["RTHRNET_PRETRAINED"] = str(Path(args.pretrained_backbone).resolve())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from ultralytics import RTDETR

    configs = [
        ("RT-DETR-R18 (Base)", REPO_ROOT / "ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml"),
        ("Exp1_BaseHRNet", SUITE_DIR / "configs" / "exp1_base_hrnet_w18.yaml"),
        ("Exp6_StarHRNet", SUITE_DIR / "configs" / "exp6_hrnet_starrep.yaml"),
        ("Exp11_StarHR_192Speed", SUITE_DIR / "configs" / "exp11_starhr_192speed.yaml"),
        ("Exp12_StarHR_Balanced", SUITE_DIR / "configs" / "exp12_starhr_balanced.yaml"),
        ("Exp13_StarHR_Turbo", SUITE_DIR / "configs" / "exp13_starhr_turbo.yaml"),
        ("Exp14_StarHR_Turbo_D2", SUITE_DIR / "configs" / "exp14_starhr_turbo_d2.yaml"),
    ]

    rows = []
    for name, cfg in configs:
        if not cfg.is_file():
            continue
        print(f"\n========== Verifying {name} ==========")
        det = RTDETR(str(cfg))
        net = det.model
        assert copy.deepcopy(net) is not net
        net.fuse()
        layers = len(list(net.modules()))
        deploy_params = sum(p.numel() for p in net.parameters()) / 1e6
        grads = sum(p.numel() for p in net.parameters() if p.requires_grad)
        ms_b16_fp16 = benchmark(copy.deepcopy(net), device, batch=16, half=True, imgsz=args.imgsz)
        ms_b1_fp32 = benchmark(copy.deepcopy(net), device, batch=1, half=False, imgsz=args.imgsz)
        rows.append((name, layers, deploy_params, grads, ms_b16_fp16, ms_b1_fp32))

    print("\n" + "=" * 102)
    print(f"{'Experiment':<22} | {'Layers':>6} | {'Deploy Params':>13} | {'Grads':>5} | {'FP16 b16 (ms/img)':>17} | {'FP32 b1 (ms/img)':>16}")
    print("-" * 102)
    for name, ly, pd, gr, m16, m1 in rows:
        print(f"{name:<22} | {ly:>6d} | {pd:>11.2f} M | {gr:>5d} | {m16:>14.2f} ms | {m1:>13.2f} ms")
    print("=" * 102)


if __name__ == "__main__":
    main()
