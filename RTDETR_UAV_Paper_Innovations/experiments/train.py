#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Fair Training Launcher for RT-DETR UAV Paper Ablations (E0–E15)
=======================================================================
Enforces strict experimental fairness across Original RT-DETR-R18 (E0) and
all standalone/joint innovations:
  - E0:  Baseline Original RT-DETR-R18
  - E1–E4:  Innovations 1–4 (E4 now includes the 300x `* nq` cls_loss fix!)
  - E5–E7:  Combinations of Innovations 1–4
  - E8–E10: Standalone Innovations 5 (DMBA), 6 (SSSC-Lite), 7 (GLRoI)
  - E11:    Standalone Innovation 8  (UAV-Dense-O2O Copy-Paste + Aux-O2M, 0 inference overhead)
  - E12:    Standalone Innovation 9  (SA-FDR Fine-Grained Distribution Refinement + AW-LSD)
  - E13:    Standalone Innovation 10 (TOLF-Lite Noise-Robust Uncertainty Localization)
  - E14:    Standalone Innovation 11 (DySample2x + HWDDownsample Alias-Free Sampling)
  - E15:    Standalone Innovation 12 (SM-AIFI Local-Global Spectrum-Modulated Encoder)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

ABLATION_DIR = REPO_ROOT / "RTDETR_UAV_Paper_Innovations" / "experiments" / "ablation_configs"

EXPERIMENT_MAP: Dict[str, str] = {
    "E0": str(ABLATION_DIR / "E0_baseline_rtdetr_r18.yaml"),
    "E1": str(ABLATION_DIR / "E1_inno1_wcasfp.yaml"),
    "E2": str(ABLATION_DIR / "E2_inno2_faga.yaml"),
    "E3": str(ABLATION_DIR / "E3_inno3_uadcr.yaml"),
    "E4": str(ABLATION_DIR / "E4_inno4_sahnwd.yaml"),
    "E5": str(ABLATION_DIR / "E5_combo_A_feature_neck.yaml"),
    "E6": str(ABLATION_DIR / "E6_combo_B_decoder_loss.yaml"),
    "E7": str(ABLATION_DIR / "E7_final_joint_model.yaml"),
    "E8": str(ABLATION_DIR / "E8_inno5_dmba.yaml"),
    "E9": str(ABLATION_DIR / "E9_inno6_sssc.yaml"),
    "E10": str(ABLATION_DIR / "E10_inno7_glroi.yaml"),
    "E11": str(ABLATION_DIR / "E11_inno8_denseo2o.yaml"),
    "E12": str(ABLATION_DIR / "E12_inno9_safdr.yaml"),
    "E13": str(ABLATION_DIR / "E13_inno10_tolf.yaml"),
    "E14": str(ABLATION_DIR / "E14_inno11_dysample_hwd.yaml"),
    "E15": str(ABLATION_DIR / "E15_inno12_smaifi.yaml"),
    "V2_A": str(ABLATION_DIR / "V2_A_wcasfpv2.yaml"),
    "V2_B": str(ABLATION_DIR / "V2_B_dmbasc.yaml"),
    "V2_C": str(ABLATION_DIR / "V2_C_cauad.yaml"),
    "V2_FULL": str(ABLATION_DIR / "V2_FULL_joint.yaml"),
    "HRNetV2_W18": str(REPO_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr_hrnet_w18.yaml"),
    "RTHRNet_Fast": str(REPO_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr_rthrnet_w18_fast.yaml"),
    "RTHRNet_Balanced": str(REPO_ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr_rthrnet_w18_balanced.yaml"),
}


def build_fair_train_kwargs(
    exp_id: str,
    model_yaml: str,
    data_yaml: str,
    pretrained_weights: str,
    epochs: int = 150,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    seed: int = 0,
    project: str = "runs/uav_paper_ablation",
) -> dict:
    return {
        "model": model_yaml,
        "data": data_yaml,
        "epochs": int(epochs),
        "patience": 0,          # Disable early stopping so all runs complete exact 150 epochs
        "batch": int(batch),
        "imgsz": int(imgsz),
        "device": device,
        "workers": 8,
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 0.1,             # Cosine decay to 1e-5 so AP75 / mAP50:95 converges cleanly
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "warmup_epochs": 3.0,
        "cos_lr": True,
        "amp": True,
        "mosaic": 1.0,
        "close_mosaic": 15,
        "mixup": 0.0,
        "seed": int(seed),
        "deterministic": False, # Required for CUDA F.grid_sample backward in RT-DETR/DySample/GLRoI
        "project": project,
        "name": f"{exp_id}_seed{seed}",
        "exist_ok": True,
        "pretrained": pretrained_weights if (pretrained_weights and os.path.exists(pretrained_weights)) else False,
    }


def main():
    parser = argparse.ArgumentParser(description="Fair Training Launcher for RT-DETR UAV Innovations (E0-E15, V2)")
    default_data = (
        "dataset/DUT-Anti-UAV.yaml"
        if (REPO_ROOT / "dataset" / "DUT-Anti-UAV.yaml").is_file()
        else "ultralytics/cfg/datasets/dut-anti-uav.yaml"
    )
    parser.add_argument("--exp", type=str, required=True, choices=list(EXPERIMENT_MAP.keys()), help="Experiment ID (E0..E15, V2_A..V2_FULL)")
    parser.add_argument("--data", type=str, default=default_data, help="Dataset YAML path")
    parser.add_argument("--weights", type=str, default="", help="Pretrained checkpoint path; leave empty to train from scratch (default: '')")
    parser.add_argument("--epochs", type=int, default=150, help="Fixed training epochs (default: 150)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--micro-batch", type=int, default=None, help="Micro-batch size (default: same as batch)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ID(s), e.g. '0' or '0,1'")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--project", type=str, default="runs/uav_paper_ablation", help="Output directory root")
    parser.add_argument("--execute", action="store_true", help="Start training; default is dry-run")
    parser.add_argument("--dry-run", action="store_true", help="Print fair training config without launching GPU run")
    args = parser.parse_args()

    actual_batch = args.batch
    if args.micro_batch is not None:
        actual_batch = args.micro_batch

    model_yaml = EXPERIMENT_MAP[args.exp]
    if not Path(model_yaml).is_file():
        parser.error(f"Model config file not found: {model_yaml}")

    train_kwargs = build_fair_train_kwargs(
        exp_id=args.exp,
        model_yaml=model_yaml,
        data_yaml=args.data,
        pretrained_weights=args.weights,
        epochs=args.epochs,
        batch=actual_batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        project=args.project,
    )

    print("=" * 80)
    print(f"[RT-DETR UAV Paper Ablation Launcher] Experiment: {args.exp} (Seed: {args.seed})")
    if args.weights and os.path.exists(args.weights):
        print(f"[INFO] Using pretrained weights: {args.weights}")
    else:
        print("[INFO] Scratch Training Mode: NO pretrained weights will be loaded (pretrained=False).")
    print(json.dumps(train_kwargs, indent=2, ensure_ascii=False))
    print("=" * 80)

    if args.dry_run or not args.execute:
        print("[DRY-RUN] Verified fair training recipe. Exiting without launching GPU training. Use --execute to start.")
        return

    import ultralytics
    expected_init = (REPO_ROOT / "ultralytics" / "__init__.py").resolve()
    actual_init = Path(ultralytics.__file__).resolve()
    if actual_init != expected_init:
        raise RuntimeError(f"Imported external ultralytics ({actual_init}) instead of repo ({expected_init})!")

    from ultralytics import RTDETR

    model = RTDETR(model_yaml)
    if args.weights and os.path.exists(args.weights):
        print(f"[INFO] Loading pretrained weights from {args.weights}")
        model.load(args.weights)
    else:
        print("[INFO] Model initialized from YAML architecture only (training from scratch).")

    out_dir = Path(args.project) / f"{args.exp}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_config.json").write_text(
        json.dumps(train_kwargs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
