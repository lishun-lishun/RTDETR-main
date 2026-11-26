#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Dataset Fair Trainer for Extra Key Models on RTX 4090 Server:
  1. `Exp1_BaseHRNet`             (Original HRNetV2-W18 High-Accuracy Baseline)
  2. `Exp30_RepStarFast_HiLoHaar` (Optional: RepStarFastBlock + HiLoHaar, Standard GIoU)
  3. `Exp31_Exp28_InnerNWD`       (Exp28 + Scale-Aware Inner-GIoU + NWD Loss, 0.00 ms deploy overhead)
  4. `Exp32_Exp28Plus_Full`       (RepStarFastBlock + HiLoHaar + Inner-GIoU + NWD Full Champion)

Enforces 100% identical hyperparameters to `train_cross_4090.py` (`epochs=150, batch=8, imgsz=640,
AdamW, lr0=1e-4, lrf=0.1, warmup_epochs=5.0, cos_lr=True, amp=True, mosaic=0.0, seed=0`) and saves
to `runs/cross_dataset_4090/<Dataset>/<Model>_seed<seed>/` so `run_eval_cross_4090.sh` picks them
up automatically!
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] train_cross_extra_4090.py requires Python 3.8+ (detected: {sys.version}).\n"
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
    r"E:\envs\my_env\Lib\site-packages",
    r"E:\envs\isaaclab\Lib\site-packages",
    r"E:\envs\RareCLIP\Lib\site-packages",
]:
    if os.path.isdir(fallback_env):
        sys.path.append(fallback_env)

DATASET_YAMLS = {
    "Det-Fly": "dataset/Det-Fly.yaml",
    "NPS-Drones": "dataset/NPS-Drones.yaml",
    "TIB-Net": "dataset/TIB-Net.yaml",
}

# model_name -> (yaml_path, use_inner_nwd_loss)
MODEL_SPECS = {
    "Exp1_BaseHRNet": ("ultralytics/cfg/models/rt-detr/exp1_base_hrnet_w18.yaml", False),
    "Exp30_RepStarFast_HiLoHaar": ("ultralytics/cfg/models/rt-detr/exp30_repstarfast_hilo_haar.yaml", False),
    "Exp31_Exp28_InnerNWD": ("ultralytics/cfg/models/rt-detr/exp28_starfast_hilo_haar.yaml", True),
    "Exp32_Exp28Plus_Full": ("ultralytics/cfg/models/rt-detr/exp30_repstarfast_hilo_haar.yaml", True),
}


def main():
    parser = argparse.ArgumentParser(description="Fair Cross-Dataset Trainer for Exp1, Exp30, Exp31, Exp32 (4090)")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_YAMLS.keys()))
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=8, help="Compatibility parameter for batch size")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--ratio", type=float, default=1.25, help="Inner-GIoU scale ratio (for Exp31/Exp32)")
    parser.add_argument("--nwd-c", type=float, default=0.05, help="NWD scale constant (for Exp31/Exp32)")
    parser.add_argument("--nwd-weight", type=float, default=0.25, help="Max NWD weight (for Exp31/Exp32)")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/cross_dataset_4090")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    data_yaml = DATASET_YAMLS[args.dataset]
    model_yaml, use_inner_nwd = MODEL_SPECS[args.model]
    if not Path(data_yaml).is_file():
        raise FileNotFoundError(
            f"Missing dataset YAML: {data_yaml}. "
            f"Run `python RTDETR_CrossDataset_4090/prepare_datasets_4090.py --repo . --seed 0` first."
        )
    if not Path(model_yaml).is_file():
        raise FileNotFoundError(f"Missing model YAML: {model_yaml}")

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
    print(f"[Cross-Dataset Extra3 Trainer] Dataset={args.dataset} | Model={args.model} | InnerNWD={use_inner_nwd}")
    print(json.dumps(train_kwargs, indent=2, ensure_ascii=False))
    print("=" * 84)

    if not args.execute:
        print("[DRY-RUN] Verified config. Add `--execute` to launch GPU training.")
        return

    if use_inner_nwd:
        from ultralytics.nn.modules.exp28_plus import patch_rtdetr_inner_nwd_loss
        patch_rtdetr_inner_nwd_loss(ratio=args.ratio, nwd_c=args.nwd_c, nwd_max_weight=args.nwd_weight)

    from ultralytics import RTDETR

    model = RTDETR(model_yaml)
    out_dir = Path(train_kwargs["project"]) / train_kwargs["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(train_kwargs, use_inner_nwd=use_inner_nwd, ratio=args.ratio, nwd_c=args.nwd_c, nwd_weight=args.nwd_weight)
    (out_dir / "training_config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
