#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair Training Launcher for `RTDETR_Exp33_AntiCollapse` (Exp33 - Exp36).
Matches the exact Exp26-Exp32 fair training protocol (`epochs=150, batch=8, imgsz=640,
AdamW, lr0=1e-4, lrf=0.1, warmup_epochs=5.0, cos_lr=True, amp=True, mosaic=0.0, seed=0`).
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] train_exp33.py requires Python 3.8+ (detected: {sys.version}).\n"
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
    r"E:\envs\my_env\Lib\site-packages",
    r"E:\envs\isaaclab\Lib\site-packages",
    r"E:\envs\RareCLIP\Lib\site-packages",
]:
    if os.path.isdir(fallback_env):
        sys.path.append(fallback_env)

# exp_name -> (yaml_path, use_coupled_tight_nwd)
EXP_SPECS = {
    "Exp33_FCGHaar_Only": (
        str(REPO_ROOT / "ultralytics/cfg/models/rt-detr/exp33_fcghaar_only.yaml"),
        False,
    ),
    "Exp34_AntiCollapseDec_Only": (
        str(REPO_ROOT / "ultralytics/cfg/models/rt-detr/exp34_anticollapse_dec.yaml"),
        False,
    ),
    "Exp35_TightCoupledNWD_Only": (
        str(REPO_ROOT / "ultralytics/cfg/models/rt-detr/exp28_starfast_hilo_haar.yaml"),
        True,
    ),
    "Exp36_Full_FCG_AntiCollapse_Coupled": (
        str(REPO_ROOT / "ultralytics/cfg/models/rt-detr/exp36_full_anticollapse.yaml"),
        True,
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Train Exp33-Exp36 AntiCollapse & FCG-Haar Suite")
    parser.add_argument("--exp", type=str, required=True, choices=list(EXP_SPECS.keys()))
    parser.add_argument("--data", type=str, default="dataset/DUT-Anti-UAV.yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=8, help="Compatibility parameter for batch size")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/hrnet_lite_suite")
    parser.add_argument("--ratio", type=float, default=1.25)
    parser.add_argument("--nwd-c", type=float, default=0.05)
    parser.add_argument("--nwd-max-w", type=float, default=0.25)
    parser.add_argument("--area-tau", type=float, default=0.0015)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    model_yaml, use_coupled_nwd = EXP_SPECS[args.exp]
    if not Path(model_yaml).is_file():
        raise FileNotFoundError(f"Missing model YAML: {model_yaml}. Run install_exp33.py first!")

    run_name = f"{args.exp}_seed{args.seed}"
    train_kwargs = {
        "model": model_yaml,
        "data": args.data,
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
        "copy_paste": 0.0,
        "seed": int(args.seed),
        "deterministic": False,
        "project": args.project,
        "name": run_name,
        "exist_ok": True,
        "pretrained": False,
        "plots": True,
    }

    print("=" * 84)
    print(f"[Exp33+ Launcher] {run_name} | CoupledTightNWD={use_coupled_nwd} (area_tau={args.area_tau})")
    print(json.dumps(train_kwargs, indent=2, ensure_ascii=False))
    print("=" * 84)
    if not args.execute:
        print("[DRY-RUN] Pass --execute to start training.")
        return

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    from ultralytics import RTDETR

    if use_coupled_nwd:
        from ultralytics.nn.modules.exp33_anticollapse import patch_rtdetr_coupled_tight_nwd
        patch_rtdetr_coupled_tight_nwd(
            ratio=args.ratio,
            nwd_c=args.nwd_c,
            nwd_max_weight=args.nwd_max_w,
            area_tau=args.area_tau,
        )

    model = RTDETR(model_yaml)
    out_dir = Path(args.project) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(
        train_kwargs,
        use_coupled_tight_nwd=use_coupled_nwd,
        ratio=args.ratio,
        nwd_c=args.nwd_c,
        nwd_max_w=args.nwd_max_w,
        area_tau=args.area_tau,
    )
    (out_dir / "training_config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
