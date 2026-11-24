#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair Training Launcher & A30/4090 Speed Verifier for `RTDETR_Exp6_Speed` (Exp 26 - Exp 29)
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] run_exp6_speed.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import copy
import json
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

EXP_CONFIGS = {
    "Exp26_StarFast_SameNeck": str(SUITE_DIR / "configs" / "exp26_starfast_sameneck.yaml"),
    "Exp27_StarFast_NativeRepC3": str(SUITE_DIR / "configs" / "exp27_starfast_nativerepc3.yaml"),
    "Exp28_StarFast_HiLoHaar": str(SUITE_DIR / "configs" / "exp28_starfast_hilo_haar.yaml"),
    "Exp29_StarFast_ZeroConcat": str(SUITE_DIR / "configs" / "exp29_starfast_zeroconcat.yaml"),
}


def bench(model, device, batch, half, imgsz=640, warmup=15, iters=40):
    m = copy.deepcopy(model).eval().to(device)
    if half and device.type == "cuda":
        m = m.half()
    x = torch.randn(
        batch,
        3,
        imgsz,
        imgsz,
        device=device,
        dtype=torch.float16 if (half and device.type == "cuda") else torch.float32,
    )
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


def run_verify(device_str):
    from ultralytics import RTDETR

    device = torch.device(f"cuda:{device_str}" if torch.cuda.is_available() else "cpu")
    items = [("Exp6_StarHRNet", str(REPO_ROOT / "ultralytics/cfg/models/rt-detr/exp6_hrnet_starrep.yaml"))] + list(
        EXP_CONFIGS.items()
    )
    rows = []
    for name, cfg in items:
        if not Path(cfg).is_file():
            continue
        net = RTDETR(cfg).model
        assert copy.deepcopy(net) is not net
        net.fuse()
        ly = len(list(net.modules()))
        pm = sum(p.numel() for p in net.parameters()) / 1e6
        gr = sum(p.numel() for p in net.parameters() if p.requires_grad)
        m16 = bench(net, device, 16, True)
        m1_16 = bench(net, device, 1, True)
        m1_32 = bench(net, device, 1, False)
        rows.append((name, ly, pm, gr, m16, m1_16, m1_32))
    print("\n" + "=" * 104)
    print(f"{'Experiment':<26} | {'Layers':>6} | {'Params':>7} | {'Grads':>5} | {'FP16 b16':>9} | {'FP16 b1':>8} | {'FP32 b1':>8}")
    print("-" * 104)
    for n, ly, pm, gr, a, b, c in rows:
        print(f"{n:<26} | {ly:>6d} | {pm:>5.2f} M | {gr:>5d} | {a:>6.2f} ms | {b:>5.2f} ms | {c:>5.2f} ms")
    print("=" * 104)


def main():
    parser = argparse.ArgumentParser(description="Fair Trainer & Benchmark for Exp26-Exp29 (Exp6 Speed Upgrades)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "verify"])
    parser.add_argument("--exp", type=str, default="Exp28_StarFast_HiLoHaar", choices=list(EXP_CONFIGS.keys()))
    parser.add_argument("--data", type=str, default="dataset/DUT-Anti-UAV.yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=8, help="Compatibility parameter for batch size")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=str, default="runs/hrnet_lite_suite")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.mode == "verify":
        run_verify(args.device)
        return

    model_yaml = EXP_CONFIGS[args.exp]
    train_kwargs = {
        "model": model_yaml,
        "data": args.data,
        "epochs": int(args.epochs),
        "patience": 0,
        "batch": int(args.batch),
        "imgsz": int(args.imgsz),
        "device": args.device,
        "workers": 8,
        "optimizer": "AdamW",
        "lr0": float(args.lr0),
        "lrf": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "warmup_epochs": 5.0,
        "cos_lr": True,
        "amp": True,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "seed": int(args.seed),
        "deterministic": False,
        "project": args.project,
        "name": f"{args.exp}_seed{args.seed}",
        "exist_ok": True,
        "pretrained": False,
    }
    print(json.dumps(train_kwargs, indent=2, ensure_ascii=False))
    if not args.execute:
        print("[DRY-RUN] Add `--execute` to launch GPU training.")
        return

    from ultralytics import RTDETR

    model = RTDETR(model_yaml)
    out_dir = Path(args.project) / f"{args.exp}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_config.json").write_text(json.dumps(train_kwargs, indent=2, ensure_ascii=False), encoding="utf-8")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
