#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Fair Cross-Dataset Trainer (RT-DETR-R18 Baseline vs. Exp12_StarHR_Balanced)
Enforces 100% identical hyperparameters across Det-Fly, NPS-Drones, and TIB-Net.
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] train_cross_dataset.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import json
import os
from pathlib import Path

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

MODEL_YAMLS = {
    "RT-DETR-R18": "ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml",
    "Exp12_StarHR_Balanced": "ultralytics/cfg/models/rt-detr/exp12_starhr_balanced.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="Fair Cross-Dataset Trainer (RT-DETR-R18 vs Exp12_StarHR_Balanced)")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_YAMLS.keys()))
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_YAMLS.keys()))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=8, help="Compatibility parameter for batch size")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/cross_dataset")
    parser.add_argument("--r18-weights", type=str, default="", help="Optional RT-DETR-R18 weights (e.g. weights/rtdetr-r18.pt)")
    parser.add_argument("--pretrained-backbone", type=str, default="", help="Optional hrnetv2_w18-8cb57bb9.pth path")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    data_yaml = DATASET_YAMLS[args.dataset]
    model_yaml = MODEL_YAMLS[args.model]
    if not Path(data_yaml).is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}. Run prepare_cross_datasets.py first.")
    if not Path(model_yaml).is_file():
        raise FileNotFoundError(f"Model YAML not found: {model_yaml}")
    if args.pretrained_backbone:
        os.environ["RTHRNET_PRETRAINED"] = str(Path(args.pretrained_backbone).resolve())

    train_kwargs = {
        "model": model_yaml,
        "data": data_yaml,
        "epochs": int(args.epochs),
        "patience": 0,
        "batch": int(args.batch),
        "imgsz": int(args.imgsz),
        "device": str(args.device),
        "workers": int(args.workers),
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
        "project": f"{args.project}/{args.dataset}",
        "name": f"{args.model}_seed{args.seed}",
        "exist_ok": True,
        "pretrained": False,
    }

    print("=" * 84)
    print(f"[Cross-Dataset Fair Trainer] Dataset: {args.dataset} | Model: {args.model}")
    print(json.dumps(train_kwargs, indent=2, ensure_ascii=False))
    print("=" * 84)

    if not args.execute:
        print("[DRY-RUN] Verified config. Add `--execute` to launch GPU training.")
        return

    from ultralytics import RTDETR

    model = RTDETR(model_yaml)
    if args.model == "RT-DETR-R18" and args.r18_weights and os.path.isfile(args.r18_weights):
        print(f"[INFO] Loading matching RT-DETR-R18 weights from `{args.r18_weights}`")
        model.load(args.r18_weights)

    out_dir = Path(train_kwargs["project"]) / train_kwargs["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_config.json").write_text(json.dumps(train_kwargs, indent=2, ensure_ascii=False), encoding="utf-8")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
