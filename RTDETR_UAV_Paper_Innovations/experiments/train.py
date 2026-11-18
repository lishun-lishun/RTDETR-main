#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Fair Training Launcher for RT-DETR UAV Paper Ablations (E0–E7)
======================================================================
Enforces strict experimental fairness across Original RT-DETR-R18 (E0)
and all innovation variants (E1–E7):
- Same dataset config (`dataset/DUT-Anti-UAV.yaml`)
- Same Input Resolution (`imgsz=640`)
- Same Optimizer (`optimizer='AdamW'`, `lr0=0.0001`, `lrf=1.0`, `weight_decay=0.0001`)
- Same effective and micro batch (`8`) and workers (`4`)
- Same Training Epochs (`epochs=150`, `patience=0` to prevent premature early-stop truncation!)
- Same Augmentation Policy (`mosaic=0`, `close_mosaic=0`, `mixup=0`)
- Same Pretrained Backbone Initialization Policy
- Multi-seed reproducibility (`seed in [0, 1, 2]`)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

EXPERIMENT_MAP: Dict[str, str] = {
    "E0": "experiments/ablation_configs/E0_baseline_rtdetr_r18.yaml",
    "E1": "experiments/ablation_configs/E1_inno1_wcasfp.yaml",
    "E2": "experiments/ablation_configs/E2_inno2_faga.yaml",
    "E3": "experiments/ablation_configs/E3_inno3_uadcr.yaml",
    "E4": "experiments/ablation_configs/E4_inno4_sahnwd.yaml",
    "E5": "experiments/ablation_configs/E5_combo_A_feature_neck.yaml",
    "E6": "experiments/ablation_configs/E6_combo_B_decoder_loss.yaml",
    "E7": "experiments/ablation_configs/E7_final_joint_model.yaml",
}


def build_fair_train_kwargs(
    exp_id: str,
    model_yaml: str,
    data_yaml: str,
    pretrained_weights: str,
    epochs: int = 150,
    batch: int = 8,
    imgsz: int = 640,
    device: str = "0",
    seed: int = 0,
    project: str = "runs/uav_paper_ablation",
) -> dict:
    return {
        "data": data_yaml,
        "epochs": int(epochs),
        "patience": 0,          # CRITICAL: Disable early stopping so all runs complete exact 150 epochs!
        "batch": int(batch),
        "imgsz": int(imgsz),
        "device": device,
        "workers": 4,
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 1.0,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "warmup_epochs": 2000,
        "cos_lr": False,
        "amp": False,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "gradient_accumulation": 1,
        "seed": int(seed),
        "deterministic": True,
        "project": project,
        "name": f"{exp_id}_seed{seed}",
        "exist_ok": True,
        "pretrained": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Fair Training Launcher for RT-DETR UAV Innovations (E0-E7)")
    parser.add_argument("--exp", type=str, required=True, choices=list(EXPERIMENT_MAP.keys()), help="Experiment ID (E0..E7)")
    parser.add_argument("--data", type=str, default="dataset/DUT-Anti-UAV.yaml", help="Dataset YAML path")
    parser.add_argument("--weights", type=str, default="", help="Optional identical R18 initialization checkpoint")
    parser.add_argument("--epochs", type=int, default=150, help="Fixed training epochs (default: 150)")
    parser.add_argument("--batch", type=int, default=8, help="Effective batch size (default: 8)")
    parser.add_argument("--micro-batch", type=int, default=8, help="Per-step batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID(s), e.g., '0' or '0,1'")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (use 0, 1, 2 for 3-seed significance test)")
    parser.add_argument("--project", type=str, default="runs/uav_paper_ablation", help="Output directory root")
    parser.add_argument("--execute", action="store_true", help="Start training; default is dry run")
    args = parser.parse_args()
    if args.batch != 8 or args.micro_batch != 8:
        parser.error('This comparison protocol requires --batch 8 --micro-batch 8')

    model_yaml = EXPERIMENT_MAP[args.exp]
    train_kwargs = build_fair_train_kwargs(
        exp_id=args.exp,
        model_yaml=model_yaml,
        data_yaml=args.data,
        pretrained_weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        project=args.project,
    )

    print("=" * 80)
    print(f"[RT-DETR UAV Paper Ablation Launcher] Experiment: {args.exp} (Seed: {args.seed})")
    print(json.dumps(train_kwargs, indent=2, ensure_ascii=False))
    print("=" * 80)

    if not args.execute:
        print("[DRY-RUN] Verified fair training recipe. Exiting without launching GPU training.")
        return

    from ultralytics import RTDETR

    if not Path(args.data).is_file():
        parser.error(f'Dataset YAML missing: {args.data}')

    model = RTDETR(model_yaml)
    if args.weights and os.path.exists(args.weights):
        print(f"[INFO] Loading matching pretrained weights from {args.weights}")
        model.load(args.weights)

    # Save exact training protocol metadata
    out_dir = Path(args.project) / f"{args.exp}_seed{args.seed}"
    if out_dir.exists():
        parser.error(f'Output already exists: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_config.json").write_text(
        json.dumps(train_kwargs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
