#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subprocess-Isolated Single-Checkpoint Evaluator for Cross-Dataset 4090 Benchmark.
Avoids glibc/DataLoader heap corruption across multi-model runs and logs full metrics to JSONL.
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] eval_single_ckpt_4090.py requires Python 3.8+ (detected: {sys.version}).\n"
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


def bench(model, device, batch, half, imgsz=640, warmup=15, iters=40):
    m = copy.deepcopy(model).eval().to(device)
    if half and device.type == "cuda":
        m = m.half()
    dtype = torch.float16 if (half and device.type == "cuda") else torch.float32
    x = torch.randn(batch, 3, imgsz, imgsz, device=device, dtype=dtype)
    peak_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        ms = (time.perf_counter() - t0) / (iters * batch) * 1000.0
    return ms, 1000.0 / ms, peak_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-yaml", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-jsonl", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import get_flops

    ckpt = Path(args.ckpt)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    print(f"==================== Evaluating [{args.dataset}] {args.model_name} ====================")
    disk_mb = ckpt.stat().st_size / (1024 * 1024)
    det = RTDETR(str(ckpt))
    split_used = args.split
    try:
        res = det.val(data=args.data_yaml, split=split_used, imgsz=args.imgsz, batch=16, device=args.device, verbose=False)
    except Exception:
        split_used = "val"
        res = det.val(data=args.data_yaml, split="val", imgsz=args.imgsz, batch=16, device=args.device, verbose=False)

    P, R = float(res.box.mp), float(res.box.mr)
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0

    net = RTDETR(str(ckpt)).model
    net.fuse()
    layers = len(list(net.modules()))
    params_m = sum(p.numel() for p in net.parameters()) / 1e6
    grads = sum(p.numel() for p in net.parameters() if p.requires_grad)
    try:
        gflops = float(get_flops(copy.deepcopy(net), imgsz=args.imgsz))
    except Exception:
        gflops = -1.0

    fp16_b16_ms, fp16_b16_fps, vram16 = bench(net, device, 16, True, args.imgsz)
    fp16_b1_ms, _, _ = bench(net, device, 1, True, args.imgsz)
    fp32_b16_ms, _, _ = bench(net, device, 16, False, args.imgsz)
    fp32_b1_ms, fp32_b1_fps, _ = bench(net, device, 1, False, args.imgsz)

    row = {
        "Dataset": args.dataset,
        "Split": split_used,
        "Model": args.model_name,
        "GPU": gpu_name,
        "Layers": layers,
        "Params_M": round(params_m, 2),
        "Grads": grads,
        "GFLOPs": round(gflops, 1),
        "Pt_MB": round(disk_mb, 1),
        "VRAM16_MB": round(vram16, 0),
        "P": round(P, 4),
        "R": round(R, 4),
        "F1": round(F1, 4),
        "mAP50": round(float(res.box.map50), 4),
        "mAP75": round(float(res.box.map75), 4),
        "mAP50_95": round(float(res.box.map), 4),
        "Val_ms": round(float(res.speed.get("inference", 0.0)), 2),
        "FP16_b16_ms": round(fp16_b16_ms, 2),
        "FP16_b16_FPS": round(fp16_b16_fps, 1),
        "FP16_b1_ms": round(fp16_b1_ms, 2),
        "FP32_b16_ms": round(fp32_b16_ms, 2),
        "FP32_b1_ms": round(fp32_b1_ms, 2),
        "FP32_b1_FPS": round(fp32_b1_fps, 1),
    }
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[DONE] [{args.dataset}] {args.model_name} | mAP50={row['mAP50']} | mAP75={row['mAP75']} | mAP50:95={row['mAP50_95']} | FP16_b16={row['FP16_b16_ms']}ms | FP32_b1={row['FP32_b1_ms']}ms")


if __name__ == "__main__":
    main()
