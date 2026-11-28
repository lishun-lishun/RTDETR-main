#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8-GPU Subprocess-Isolated Master Evaluator for `final_model/<Dataset>/<Model>/weights/best.pt`.
Evaluates strictly on `split="test"` across all 4 datasets, saves CSV + prints full Markdown
tables and 4-Dataset Macro Average leaderboard.
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] eval_final_suite_8gpu.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import copy
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".yolo"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".yolo"))
if (REPO_ROOT / "ultralytics").is_dir():
    sys.path.insert(0, str(REPO_ROOT))
if (REPO_ROOT / ".pypkg").is_dir():
    sys.path.insert(0, str(REPO_ROOT / ".pypkg"))
for fallback_env in [
    r"E:\envs\my_env\Lib\site-packages",
    r"E:\envs\isaaclab\Lib\site-packages",
    r"E:\envs\RareCLIP\Lib\site-packages",
]:
    if os.path.isdir(fallback_env):
        sys.path.append(fallback_env)


def bench(model, device, batch: int, half: bool, imgsz: int = 640, warmup: int = 15, iters: int = 40):
    import torch
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
        ms = (time.perf_counter() - t0) / (iters * batch) * 1000.0
    return ms, 1000.0 / ms


def eval_single(dataset: str, model_name: str, ckpt: str, data_yaml: str, out_jsonl: str, imgsz: int = 640):
    import torch
    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import get_flops

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if has_cuda else "cpu")
    val_device = "0" if has_cuda else "cpu"
    det = RTDETR(ckpt)
    res = det.val(data=data_yaml, split="test", imgsz=imgsz, batch=16, device=val_device, verbose=False)
    P, R = float(res.box.mp), float(res.box.mr)
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0

    net = RTDETR(ckpt).model
    net.fuse()
    params_m = sum(p.numel() for p in net.parameters()) / 1e6
    grads = sum(p.numel() for p in net.parameters() if p.requires_grad)
    try:
        gflops = float(get_flops(copy.deepcopy(net), imgsz=imgsz))
    except Exception:
        gflops = -1.0

    fp16_b16_ms, fp16_b16_fps = bench(net, device, 16, True, imgsz)
    fp32_b1_ms, fp32_b1_fps = bench(net, device, 1, False, imgsz)

    row = {
        "Dataset": dataset,
        "Split": "test",
        "Model": model_name,
        "Params_M": round(params_m, 2),
        "Grads": int(grads),
        "GFLOPs": round(gflops, 1),
        "P": round(P, 4),
        "R": round(R, 4),
        "F1": round(F1, 4),
        "mAP50": round(float(res.box.map50), 4),
        "mAP75": round(float(res.box.map75), 4),
        "mAP50_95": round(float(res.box.map), 4),
        "FP16_b16_ms": round(fp16_b16_ms, 2),
        "FP16_b16_FPS": round(fp16_b16_fps, 1),
        "FP32_b1_ms": round(fp32_b1_ms, 2),
        "FP32_b1_FPS": round(fp32_b1_fps, 1),
    }
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[DONE] [{dataset}] {model_name} | mAP50={row['mAP50']} | mAP75={row['mAP75']} | mAP50:95={row['mAP50_95']} | FP16={row['FP16_b16_ms']}ms | FP32={row['FP32_b1_ms']}ms", flush=True)


def main():
    parser = argparse.ArgumentParser(description="8-GPU Parallel Evaluator for final_model/")
    parser.add_argument("--single", action="store_true", help="Internal flag for single-checkpoint subprocess")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--data-yaml", type=str, default="")
    parser.add_argument("--out-jsonl", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    if args.single:
        eval_single(args.dataset, args.model_name, args.ckpt, args.data_yaml, args.out_jsonl)
        return

    from RTDETR_Final_Paper_Suite.train_final_suite import resolve_dataset_yaml

    final_root = REPO_ROOT / "final_model"
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    datasets = ["DUT-Anti-UAV", "Det-Fly", "TIB-Net", "NPS-Drones"]

    tasks = []
    for ds in datasets:
        ds_dir = final_root / ds
        if not ds_dir.is_dir():
            continue
        data_yaml = str(resolve_dataset_yaml(ds))
        for ckpt in sorted(ds_dir.glob("*/weights/best.pt")):
            tasks.append((ds, ckpt.parents[1].name, str(ckpt), data_yaml))

    if not tasks:
        print(f"[INFO] No checkpoints found under {final_root}/<Dataset>/*/weights/best.pt")
        return

    print(f">>> Evaluating {len(tasks)} models across GPUs {gpus} (split=test)...", flush=True)
    for g in gpus:
        p = final_root / f"_eval_gpu{g}.jsonl"
        if p.exists():
            p.unlink()

    def run_gpu_queue(gpu_idx: int, gpu_id: str):
        out_jsonl = str(final_root / f"_eval_gpu{gpu_id}.jsonl")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["OMP_NUM_THREADS"] = "4"
        env["MKL_NUM_THREADS"] = "4"
        for i in range(gpu_idx, len(tasks), len(gpus)):
            ds, mname, ckpt, dyaml = tasks[i]
            cmd = [
                sys.executable, "-u", str(Path(__file__).resolve()),
                "--single", "--dataset", ds, "--model-name", mname,
                "--ckpt", ckpt, "--data-yaml", dyaml, "--out-jsonl", out_jsonl,
            ]
            subprocess.run(cmd, env=env, check=False)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = [ex.submit(run_gpu_queue, idx, g) for idx, g in enumerate(gpus)]
        for f in futs:
            f.result()

    rows = []
    for g in gpus:
        p = final_root / f"_eval_gpu{g}.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))

    ds_order = {d: i for i, d in enumerate(datasets)}
    m_order = {
        "Baseline_R18": 0, "Baseline_HRNet": 1,
        "Imp1_Backbone_HRNetStarFast": 2, "Imp2_Neck_HiLoHaar": 3,
        "Imp3_Loss_InnerNWD": 4, "Imp12_Backbone_Neck": 5, "Final_Ours_Full": 6,
    }
    rows.sort(key=lambda r: (ds_order.get(r["Dataset"], 99), m_order.get(r["Model"], 99), r["Model"]))

    csv_path = final_root / "final_4dataset_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"\n================ final_model 四数据集完整评测总表 ({len(rows)} records, split=test) ================\n")
        print("| Dataset | Model | Params(M) | GFLOPs | P | R | F1 | mAP50 | mAP75 | mAP50:95 | FP16_b16(ms/FPS) | FP32_b1(ms/FPS) |")
        print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in rows:
            print(f"| {r['Dataset']} | {r['Model']} | {r['Params_M']} | {r['GFLOPs']} | {r['P']} | {r['R']} | {r['F1']} | {r['mAP50']} | {r['mAP75']} | {r['mAP50_95']} | {r['FP16_b16_ms']} / {r['FP16_b16_FPS']} | {r['FP32_b1_ms']} / {r['FP32_b1_FPS']} |")

        # 4-Dataset Macro Average
        by_model = {}
        for r in rows:
            by_model.setdefault(r["Model"], []).append(r)
        print("\n================ 四数据集宏平均榜单 (Macro Average) ================\n")
        print("| Model | Datasets | Params(M) | GFLOPs | Avg F1 | Avg mAP50 | Avg mAP75 | Avg mAP50:95 |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for m, mrows in sorted(by_model.items(), key=lambda kv: m_order.get(kv[0], 99)):
            n = len(mrows)
            print(
                f"| {m} | {n} | {mrows[0]['Params_M']} | {mrows[0]['GFLOPs']} "
                f"| {sum(x['F1'] for x in mrows)/n:.4f} | {sum(x['mAP50'] for x in mrows)/n:.4f} "
                f"| {sum(x['mAP75'] for x in mrows)/n:.4f} | {sum(x['mAP50_95'] for x in mrows)/n:.4f} |"
            )
        print(f"\n[SAVED] {csv_path}")


if __name__ == "__main__":
    main()
