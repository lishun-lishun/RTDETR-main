#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair Training Launcher for RT-DETR HRNet-W18 Lightweight & Turbo Suite (Exp 1 - Exp 14)
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] train_hrnet_suite.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import json
import os
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
    r'E:\envs\my_env\Lib\site-packages',
    r'E:\envs\isaaclab\Lib\site-packages',
    r'E:\envs\RareCLIP\Lib\site-packages',
]:
    if os.path.isdir(fallback_env):
        sys.path.append(fallback_env)

EXP_CONFIGS = {
    "Exp1_BaseHRNet": str(SUITE_DIR / "configs" / "exp1_base_hrnet_w18.yaml"),
    "Exp2_PruneS4": str(SUITE_DIR / "configs" / "exp2_hrnet_prunes4.yaml"),
    "Exp3_PConvRep": str(SUITE_DIR / "configs" / "exp3_hrnet_pconvrep.yaml"),
    "Exp4_SlimNeck": str(SUITE_DIR / "configs" / "exp4_hrnet_slimneck.yaml"),
    "Exp5_FastHRFull": str(SUITE_DIR / "configs" / "exp5_fasthr_rtdetr_full.yaml"),
    "Exp6_StarHRNet": str(SUITE_DIR / "configs" / "exp6_hrnet_starrep.yaml"),
    "Exp7_Lite192Dec": str(SUITE_DIR / "configs" / "exp7_hrnet_lite192_decoder.yaml"),
    "Exp8_RepGhostHaar": str(SUITE_DIR / "configs" / "exp8_hrnet_repghost_haarneck.yaml"),
    "Exp9_UltraLite": str(SUITE_DIR / "configs" / "exp9_ultralite_pareto.yaml"),
    "Exp10_StarHR_Speed": str(SUITE_DIR / "configs" / "exp10_starhr_speed.yaml"),
    "Exp11_StarHR_192Speed": str(SUITE_DIR / "configs" / "exp11_starhr_192speed.yaml"),
    "Exp12_StarHR_Balanced": str(SUITE_DIR / "configs" / "exp12_starhr_balanced.yaml"),
    "Exp13_StarHR_Turbo": str(SUITE_DIR / "configs" / "exp13_starhr_turbo.yaml"),
    "Exp14_StarHR_Turbo_D2": str(SUITE_DIR / "configs" / "exp14_starhr_turbo_d2.yaml"),
}


def main():
    parser = argparse.ArgumentParser(description="Fair Training Launcher for HRNet-Lite Suite (Exp1-Exp14)")
    parser.add_argument("--exp", type=str, required=True, choices=list(EXP_CONFIGS.keys()))
    parser.add_argument("--data", type=str, default="dataset/DUT-Anti-UAV.yaml")
    parser.add_argument("--pretrained-backbone", type=str, default="")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=8, help="Compatibility parameter for batch size")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/hrnet_lite_suite")
    parser.add_argument("--execute", action="store_true")
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
        "pretrained": False,
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
