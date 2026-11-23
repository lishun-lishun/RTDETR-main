#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FP64 Fusion Equivalence Check & A30 Speed Benchmark for Neck Arena (Exp 20 - Exp 25)
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] verify_neck_arena.py requires Python 3.8+ (detected: {sys.version}).\n"
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
        return (time.perf_counter() - t0) / (iters * batch) * 1000.0


def main():
    parser = argparse.ArgumentParser(description="Verify & benchmark Exp20-Exp25 Neck Arena on A30")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from ultralytics import RTDETR

    configs = [
        ("Exp12_StarHR_Balanced", REPO_ROOT / "ultralytics/cfg/models/rt-detr/exp12_starhr_balanced.yaml"),
        ("Exp20_Neck_GatherDist", SUITE_DIR / "configs" / "exp20_neck_gather_dist.yaml"),
        ("Exp21_Neck_OrthFreq", SUITE_DIR / "configs" / "exp21_neck_orth_freq.yaml"),
        ("Exp22_Neck_HiLoAIFI", SUITE_DIR / "configs" / "exp22_neck_hilo_aifi.yaml"),
        ("Exp23_Neck_StarCCFF", SUITE_DIR / "configs" / "exp23_neck_star_ccff.yaml"),
        ("Exp24_Neck_LocalContrast", SUITE_DIR / "configs" / "exp24_neck_local_contrast.yaml"),
        ("Exp25_Neck_UltimateCombo", SUITE_DIR / "configs" / "exp25_neck_ultimate_combo.yaml"),
    ]

    rows = []
    for name, cfg in configs:
        if not cfg.is_file():
            continue
        print(f"\n========== Verifying {name} ==========")
        det = RTDETR(str(cfg))
        net = det.model
        assert copy.deepcopy(net) is not net

        # FP64 fusion equivalence check on target device
        m64 = copy.deepcopy(net).to(device).double().eval()
        with torch.no_grad():
            for mod in m64.modules():
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.weight.uniform_(0.5, 1.5)
                    mod.bias.uniform_(-0.2, 0.2)
                    mod.running_mean.uniform_(-0.2, 0.2)
                    mod.running_var.uniform_(0.5, 1.5)
            x64 = torch.randn(1, 3, 128, 128, device=device, dtype=torch.float64)
            y_before = m64(x64)
            m64.fuse()
            y_after = m64(x64)
            out_b = y_before[0] if isinstance(y_before, (tuple, list)) else y_before
            out_a = y_after[0] if isinstance(y_after, (tuple, list)) else y_after
            max_diff = (out_b - out_a).abs().max().item()
            print(f"  [FP64 Fuse Check] max_abs_diff = {max_diff:.2e}")
            assert max_diff < 1e-8, f"Fuse mismatch on {name}: {max_diff}"

        net.fuse()
        layers = len(list(net.modules()))
        params = sum(p.numel() for p in net.parameters()) / 1e6
        grads = sum(p.numel() for p in net.parameters() if p.requires_grad)
        m16 = benchmark(copy.deepcopy(net), device, batch=16, half=True, imgsz=args.imgsz)
        m1 = benchmark(copy.deepcopy(net), device, batch=1, half=False, imgsz=args.imgsz)
        rows.append((name, layers, params, grads, m16, m1))

    print("\n" + "=" * 104)
    print(f"{'Experiment':<24} | {'Layers':>6} | {'Deploy Params':>13} | {'Grads':>5} | {'FP16 b16 (ms/img)':>17} | {'FP32 b1 (ms/img)':>16}")
    print("-" * 104)
    for name, ly, pd, gr, m16, m1 in rows:
        print(f"{name:<24} | {ly:>6d} | {pd:>11.2f} M | {gr:>5d} | {m16:>14.2f} ms | {m1:>13.2f} ms")
    print("=" * 104)


if __name__ == "__main__":
    main()
