#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Final Paper Suite Trainer (`final_model/<Dataset>/<Model>/`):
  - 4 Datasets: DUT-Anti-UAV, Det-Fly, TIB-Net, NPS-Drones (auto-uses `/dev/shm/NPS-Drones-720P` if mounted!)
  - 4 Core Ablation Models:
      1. `Imp1_Backbone_HRNetStarFast` : Improvement 1 Only (HRNetStarFast + Native Neck + GIoU)
      2. `Imp2_Neck_HiLoHaar`          : Improvement 2 Only (R18 Backbone + HiLoLinearAIFI/HaarPAN + GIoU)
      3. `Imp3_Loss_InnerNWD`          : Improvement 3 Only (R18 Backbone/Neck + Scale-Aware Inner-GIoU+NWD)
      4. `Final_Ours_Full`             : Imp1 + Imp2 + Imp3 Full Proposed Model (= Exp31_Exp28_InnerNWD)
  - Optional Extra Keys: `Baseline_R18`, `Baseline_HRNet`, `Imp12_Backbone_Neck`
  - Built-in speed & reliability fixes:
      * REPO_ROOT-anchored `final_model/<Dataset>/<Model>` absolute output path
      * `--resume` automatic checkpoint resume from `weights/last.pt`
      * `cudnn.benchmark=True`, `allow_tf32=True`, `deterministic=False`, `plots=False`
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] train_final_suite.py requires Python 3.8+ (detected: {sys.version}).\n"
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


def resolve_dataset_yaml(dataset: str) -> Path:
    """Auto-generates/returns a clean YAML under `final_model/_yamls/` for each dataset."""
    yaml_dir = REPO_ROOT / "final_model" / "_yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = yaml_dir / f"{dataset}.yaml"

    # Candidate dataset root directories (prioritizing RAM disk / 1080P accelerated copies)
    candidates = {
        "DUT-Anti-UAV": [
            Path("/home/aigc/disk1/LiShun/muchuan/DUT-Anti-UAV"),
            Path("/home/aigc/disk1/LiShun/muchuan/DUT-Anti-UAV/DUT-Anti-UAV"),
            Path("/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/DUT-Anti-UAV"),
            Path("/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/DUT-Anti-UAV/DUT-Anti-UAV"),
        ],
        "Det-Fly": [
            Path("/home/aigc/disk1/LiShun/muchuan/Det-Fly/Det-Fly-1080P"),
            Path("/home/aigc/disk1/LiShun/muchuan/Det-Fly/Det-Fly_yolo_std"),
            Path("/home/aigc/disk1/LiShun/muchuan/Det-Fly/Det-Fly"),
            Path("/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/Det-Fly/Det-Fly-1080P"),
            Path("/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/Det-Fly/Det-Fly_yolo_std"),
            Path("/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/Det-Fly/Det-Fly"),
        ],
        "TIB-Net": [
            Path("/home/aigc/disk1/LiShun/muchuan/TIB-Net"),
            Path("/home/aigc/disk1/LiShun/muchuan/TIB-Net_yolo_std"),
            Path("/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/TIB-Net"),
            Path("/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/TIB-Net_yolo_std"),
        ],
        "NPS-Drones": [
            Path("/dev/shm/NPS-Drones-720P"),
            Path("/home/aigc/disk1/LiShun/muchuan/NPS-Drones/NPS-Drones-1080P"),
            Path("/home/aigc/disk1/LiShun/muchuan/NPS-Drones/NPS-Drones_yolo_std"),
            Path("/home/aigc/disk1/LiShun/muchuan/NPS-Drones/NPS-Drones"),
            Path("/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/NPS-Drones/NPS-Drones-1080P"),
            Path("/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/NPS-Drones/NPS-Drones_yolo_std"),
            Path("/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/NPS-Drones/NPS-Drones"),
        ],
    }

    root = next((p for p in candidates[dataset] if (p / "images" / "train").is_dir()), None)
    if root is None:
        fallback_yaml = REPO_ROOT / "dataset" / f"{dataset}.yaml"
        if fallback_yaml.is_file():
            return fallback_yaml
        raise FileNotFoundError(
            f"Could not locate dataset '{dataset}'. Checked: {[str(p) for p in candidates[dataset]]} "
            f"and `{fallback_yaml}`. Pass `--data <path/to/dataset.yaml>` if stored elsewhere."
        )

    cls_name = "drone" if dataset == "DUT-Anti-UAV" else "UAV"
    out_yaml.write_text(
        f"path: {root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n"
        f"  0: {cls_name}\n",
        encoding="utf-8",
    )
    return out_yaml


# model_name -> (yaml_path, use_inner_nwd_loss)
MODEL_SPECS = {
    "Imp1_Backbone_HRNetStarFast": (SUITE_DIR / "configs" / "final_imp1_backbone_hrnetstarfast.yaml", False),
    "Imp2_Neck_HiLoHaar": (SUITE_DIR / "configs" / "final_imp2_neck_hilohaar.yaml", False),
    "Imp3_Loss_InnerNWD": (REPO_ROOT / "ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml", True),
    "Final_Ours_Full": (SUITE_DIR / "configs" / "final_ours_full.yaml", True),
    # Optional baselines / progressive ablation
    "Baseline_R18": (REPO_ROOT / "ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml", False),
    "Baseline_HRNet": (REPO_ROOT / "ultralytics/cfg/models/rt-detr/exp1_base_hrnet_w18.yaml", False),
    "Imp12_Backbone_Neck": (SUITE_DIR / "configs" / "final_ours_full.yaml", False),
}


def main():
    parser = argparse.ArgumentParser(description="Final Paper Suite Trainer -> final_model/<Dataset>/<Model>")
    parser.add_argument("--dataset", type=str, required=True, choices=["DUT-Anti-UAV", "Det-Fly", "TIB-Net", "NPS-Drones"])
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--data", type=str, default="", help="Optional custom data YAML override")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=16, help="Compatibility parameter for batch size")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from weights/last.pt if present")
    parser.add_argument("--execute", action="store_true", help="Launch actual training (otherwise dry-run)")
    args = parser.parse_args()

    model_yaml, use_inner_nwd = MODEL_SPECS[args.model]
    if not model_yaml.is_file():
        raise FileNotFoundError(f"Missing model YAML: {model_yaml}")

    data_yaml = Path(args.data) if args.data else resolve_dataset_yaml(args.dataset)
    project_dir = REPO_ROOT / "final_model" / args.dataset
    run_name = args.model
    out_dir = project_dir / run_name
    last_pt = out_dir / "weights" / "last.pt"

    train_kwargs = {
        "model": str(model_yaml),
        "data": str(data_yaml),
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
        "weight_decay": 1e-4,
        "warmup_epochs": 5.0,
        "cos_lr": True,
        "amp": True,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "seed": int(args.seed),
        "deterministic": False,
        "plots": False,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
        "pretrained": False,
    }

    print("=" * 88, flush=True)
    print(f"[Final Paper Suite] Dataset={args.dataset} | Model={args.model} | InnerNWD={use_inner_nwd}", flush=True)
    print(f"[Output Dir]        {out_dir}", flush=True)
    print(f"[Data YAML]         {data_yaml}", flush=True)
    print("=" * 88, flush=True)

    if not args.execute:
        print("[DRY-RUN] Verified config. Add `--execute` to start training.", flush=True)
        return

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if use_inner_nwd:
        from ultralytics.nn.modules.exp28_plus import patch_rtdetr_inner_nwd_loss
        patch_rtdetr_inner_nwd_loss(ratio=1.25, nwd_c=0.05, nwd_max_weight=0.25)

    from ultralytics import RTDETR

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(train_kwargs, dataset=args.dataset, model_key=args.model, use_inner_nwd=use_inner_nwd)
    (out_dir / "training_config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.resume and last_pt.is_file():
        print(f"[RESUME] Resuming training from {last_pt}", flush=True)
        model = RTDETR(str(last_pt))
        model.train(resume=True)
    else:
        model = RTDETR(str(model_yaml))
        model.train(**train_kwargs)


if __name__ == "__main__":
    main()
