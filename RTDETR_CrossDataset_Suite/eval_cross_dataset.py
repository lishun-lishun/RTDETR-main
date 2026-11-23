#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Cross-Dataset Full-Metric Evaluator (Accuracy + Size + Fair Fused FP16/FP32 Speed)
Evaluates all trained models under `runs/cross_dataset/<Dataset>/<Model>_seed0/weights/best.pt`.
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] eval_cross_dataset.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
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

DATASET_YAMLS = {
    "Det-Fly": "dataset/Det-Fly.yaml",
    "NPS-Drones": "dataset/NPS-Drones.yaml",
    "TIB-Net": "dataset/TIB-Net.yaml",
}
MODELS = ["RT-DETR-R18", "Exp12_StarHR_Balanced"]


def bench_forward(model, device, batch, half, imgsz=640, warmup=15, iters=40):
    m = copy.deepcopy(model).eval().to(device)
    if half and device.type == "cuda":
        m = m.half()
    dtype = torch.float16 if (half and device.type == "cuda") else torch.float32
    x = torch.randn(batch, 3, imgsz, imgsz, device=device, dtype=dtype)
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
    del m, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ms_per_img, 1000.0 / ms_per_img


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained cross-dataset checkpoints")
    parser.add_argument("--project", type=str, default="runs/cross_dataset")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import get_flops

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    root = Path(args.project)
    rows = []

    for ds, data_yaml in DATASET_YAMLS.items():
        if not Path(data_yaml).is_file():
            print(f"[SKIP] Missing YAML: {data_yaml}")
            continue
        for mdl in MODELS:
            ckpt = root / ds / f"{mdl}_seed{args.seed}" / "weights" / "best.pt"
            if not ckpt.is_file():
                print(f"[SKIP] Missing checkpoint: {ckpt}")
                continue
            print(f"\n==================== Evaluating [{ds}] {mdl} ====================")
            disk_mb = ckpt.stat().st_size / (1024 * 1024)
            det = RTDETR(str(ckpt))
            split_used = args.split
            try:
                res = det.val(data=data_yaml, split=split_used, imgsz=args.imgsz, batch=16, device=str(args.device), verbose=False)
            except Exception:
                split_used = "val"
                res = det.val(data=data_yaml, split="val", imgsz=args.imgsz, batch=16, device=str(args.device), verbose=False)

            net = RTDETR(str(ckpt)).model
            net.fuse()
            layers = len(list(net.modules()))
            params_m = sum(p.numel() for p in net.parameters()) / 1e6
            grads = sum(p.numel() for p in net.parameters() if p.requires_grad)
            try:
                gflops = float(get_flops(copy.deepcopy(net), imgsz=args.imgsz))
            except Exception:
                gflops = -1.0

            fp16_b16_ms, fp16_b16_fps = bench_forward(net, device, batch=16, half=True, imgsz=args.imgsz)
            fp16_b1_ms, _ = bench_forward(net, device, batch=1, half=True, imgsz=args.imgsz)
            fp32_b1_ms, fp32_b1_fps = bench_forward(net, device, batch=1, half=False, imgsz=args.imgsz)

            rows.append({
                "Dataset": ds,
                "Split": split_used,
                "Model": mdl,
                "Layers": layers,
                "Params_M": round(params_m, 2),
                "Grads": grads,
                "GFLOPs": round(gflops, 1),
                "Pt_MB": round(disk_mb, 1),
                "P": round(float(res.box.mp), 4),
                "R": round(float(res.box.mr), 4),
                "mAP50": round(float(res.box.map50), 4),
                "mAP75": round(float(res.box.map75), 4),
                "mAP50_95": round(float(res.box.map), 4),
                "Val_ms": round(float(res.speed.get("inference", 0.0)), 2),
                "FP16_b16_ms": round(fp16_b16_ms, 2),
                "FP16_b16_FPS": round(fp16_b16_fps, 1),
                "FP16_b1_ms": round(fp16_b1_ms, 2),
                "FP32_b1_ms": round(fp32_b1_ms, 2),
                "FP32_b1_FPS": round(fp32_b1_fps, 1),
            })
            del det, net
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not rows:
        print("[WARN] No checkpoints found to evaluate.")
        return

    root.mkdir(parents=True, exist_ok=True)
    out_json = root / "cross_dataset_benchmark.json"
    out_csv = root / "cross_dataset_benchmark.csv"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n==================== CROSS-DATASET BENCHMARK SUMMARY ====================")
    print("| Dataset | Split | Model | Layers | Params(M) | GFLOPs | Pt(MB) | P | R | mAP50 | mAP75 | mAP50:95 | Val(ms) | FP16_b16(ms/FPS) | FP32_b1(ms/FPS) |")
    print("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in rows:
        print(
            f"| {r['Dataset']} | {r['Split']} | {r['Model']} | {r['Layers']} | {r['Params_M']:.2f} | {r['GFLOPs']:.1f} | {r['Pt_MB']:.1f} | "
            f"{r['P']:.4f} | {r['R']:.4f} | {r['mAP50']:.4f} | {r['mAP75']:.4f} | {r['mAP50_95']:.4f} | "
            f"{r['Val_ms']:.2f} | {r['FP16_b16_ms']:.2f} / {r['FP16_b16_FPS']:.1f} | {r['FP32_b1_ms']:.2f} / {r['FP32_b1_FPS']:.1f} |"
        )
    print(f"\nSaved to:\n  - {out_json}\n  - {out_csv}")


if __name__ == "__main__":
    main()
