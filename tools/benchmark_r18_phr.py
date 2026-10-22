#!/usr/bin/env python3
"""RT-DETR-R18 PHR 性能基准 (Linux 服务器端, GPU).

统计 PHR-1 / PHR-2 / PHR-3 的参数量、GFLOPs、GPU 吞吐 (FPS) 与延迟, 并与
baseline (rtdetr-r18.yaml) 对照, 输出 CSV。

用法示例
--------
    python tools/benchmark_r18_phr.py --device 0
    python tools/benchmark_r18_phr.py --model all --iters 200 --warmup 30 --device 0

说明
----
* FPS 通过固定 640x640 输入、``model.eval()`` + ``torch.no_grad()`` + CUDA 事件计时测得,
  不包含数据加载/后处理, 仅衡量网络前向。
* 参数量/GFLOPs 复用项目自带的 ``model_info`` (thop), 与 get_all_yaml_param_and_flops.py 一致。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RTDETR_MODEL_DIR = _PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"

_MODELS = {
    "baseline": "rtdetr-r18.yaml",
    "phr1": "rtdetr_r18_phr1.yaml",
    "phr2": "rtdetr_r18_phr2.yaml",
    "phr3": "rtdetr_r18_phr3.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", choices=("all", *_MODELS), default=("phr1",))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", type=Path, default=_PROJECT_ROOT / "runs" / "phr_benchmark.csv")
    return parser.parse_args()


def selected(names):
    if "all" in names:
        return dict(_MODELS)
    return {n: _MODELS[n] for n in names}


def measure_fps(model, device, imgsz, batch, warmup, iters):
    x = torch.randn(batch, 3, imgsz, imgsz, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start is not None:
            start.record()
        else:
            import time
            start = time.perf_counter()
        for _ in range(iters):
            model(x)
        if end is not None:
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
        else:
            elapsed = time.perf_counter() - start
    fps = (iters * batch) / elapsed
    return fps, elapsed / iters * 1000.0  # 每批平均延迟 (ms)


def main():
    args = parse_args()
    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import model_info

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    models = selected(args.model)
    rows = []
    for name, yaml_name in models.items():
        cfg = RTDETR_MODEL_DIR / yaml_name
        print(f"\n{'=' * 24} BENCH {name} ({yaml_name}) {'=' * 24}")
        model = RTDETR(str(cfg)).to(device)
        try:
            model.fuse()
        except Exception:
            pass
        n_l, n_p, n_g, flops = model_info(model.model, imgsz=args.imgsz)
        fps, latency_ms = measure_fps(model, device, args.imgsz, args.batch, args.warmup, args.iters)
        rows.append({
            "model": name,
            "yaml": yaml_name,
            "params": n_p,
            "gflops": round(flops, 2),
            "fps": round(fps, 2),
            "latency_ms": round(latency_ms, 3),
            "device": str(device),
        })
        print(f"params={n_p:,}  GFLOPs={flops:.2f}  FPS={fps:.2f}  latency={latency_ms:.3f} ms")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "yaml", "params", "gflops", "fps", "latency_ms", "device"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n基准已保存: {args.output}")


if __name__ == "__main__":
    main()
