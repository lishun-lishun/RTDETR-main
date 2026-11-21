#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair Training Launcher for RT-DETR HRNet-W18 Lightweight Suite (Exp 1 - Exp 5)
==============================================================================
Aligns key hyperparameters with original `RT-DETR-UAV` (`mosaic=0.0`, ImageNet HRNetV2-W18
pretrained backbone, AdamW + Cosine LR) so Exp 1 reproduces the +2% HRNet accuracy gain
and Exp 2-5 measure pure lightweighting / acceleration trade-offs fairly.
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] train_hrnet_suite.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
        f"or use the environment's Python explicitly:\n"
        f"    ~/miniconda3/envs/ultralytics_rtdetr/bin/python RTDETR_HRNet_Lite_Suite/train_hrnet_suite.py\n"
    )

import argparse
import json
import os
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SUITE_DIR.parent
if (REPO_ROOT / "ultralytics").is_dir():
    sys.path.insert(0, str(REPO_ROOT))

EXP_CONFIGS = {
    "Exp1_BaseHRNet": str(SUITE_DIR / "configs" / "exp1_base_hrnet_w18.yaml"),
    "Exp2_PruneS4": str(SUITE_DIR / "configs" / "exp2_hrnet_prunes4.yaml"),
    "Exp3_PConvRep": str(SUITE_DIR / "configs" / "exp3_hrnet_pconvrep.yaml"),
    "Exp4_SlimNeck": str(SUITE_DIR / "configs" / "exp4_hrnet_slimneck.yaml"),
    "Exp5_FastHRFull": str(SUITE_DIR / "configs" / "exp5_fasthr_rtdetr_full.yaml"),
}


def main():
    parser = argparse.ArgumentParser(description="Fair Training Launcher for HRNet-Lite Suite (Exp1-Exp5)")
    parser.add_argument("--exp", type=str, required=True, choices=list(EXP_CONFIGS.keys()))
    parser.add_argument("--data", type=str, default="dataset/DUT-Anti-UAV.yaml")
    parser.add_argument("--pretrained-backbone", type=str, default="", help="Optional path to hrnetv2_w18-8cb57bb9.pth")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=8, help="Compatibility parameter for batch size")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--mosaic", type=float, default=0.0, help="Default 0.0 (matches original RT-DETR-UAV, prevents shrinking tiny UAVs)")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/hrnet_lite_suite")
    parser.add_argument("--execute", action="store_true", help="Launch actual GPU training (omit for dry-run preview)")
    args = parser.parse_args()

    if args.pretrained_backbone:
        os.environ["RTHRNET_PRETRAINED"] = str(Path(args.pretrained_backbone).resolve())

    model_yaml = EXP_CONFIGS[args.exp]
    train_kwargs = {
        "model": model_yaml,
        "data": args.data,
        "epochs": int(args.epochs),
        "patience": 0,
        "batch": int(args.batch),
        "imgsz": int(args.imgsz),
        "device": args.device,
        "workers": int(args.workers),
        "optimizer": "AdamW",
        "lr0": float(args.lr0),
        "lrf": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "warmup_epochs": 5.0,
        "cos_lr": True,
        "amp": True,
        "mosaic": float(args.mosaic),
        "close_mosaic": 0 if args.mosaic == 0.0 else 15,
        "mixup": 0.0,
        "seed": int(args.seed),
        "deterministic": False,
        "project": args.project,
        "name": f"{args.exp}_seed{args.seed}",
        "exist_ok": False,
        "pretrained": False,  # Backbone loads ImageNet hrnetv2_w18-8cb57bb9.pth directly!
    }

    print("=" * 80)
    print(f"[HRNet-Lite Suite Launcher] Experiment: {args.exp} | mosaic={args.mosaic}")
    print(json.dumps(train_kwargs, indent=2, ensure_ascii=False))
    print("=" * 80)

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
