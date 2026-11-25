#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair Trainer & Zero-Overhead Verifier for `Exp28+` (Exp 30 - Exp 32)
====================================================================
  - `Exp30_RepStarFast_HiLoHaar`: `HRNetRepStarFast` + Standard GIoU (isolates RepStarFastBlock)
  - `Exp31_Exp28_InnerNWD`:       Exact `Exp28` + Scale-Aware Inner-GIoU + NWD (isolates Loss!)
  - `Exp32_Exp28Plus_Full`:       `HRNetRepStarFast` + Scale-Aware Inner-GIoU + NWD (Full Champion!)

ALL THREE fuse to the EXACT SAME deploy layers & deploy params as `Exp28_StarFast_HiLoHaar`!
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] train_exp28_plus.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import copy
import json
import os
from pathlib import Path
import torch

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

# Map: exp_name -> (yaml_path, use_inner_nwd_loss)
EXP_SPECS = {
    "Exp30_RepStarFast_HiLoHaar": (str(SUITE_DIR / "configs" / "exp30_repstarfast_hilo_haar.yaml"), False),
    "Exp31_Exp28_InnerNWD": (str(REPO_ROOT / "ultralytics/cfg/models/rt-detr/exp28_starfast_hilo_haar.yaml"), True),
    "Exp32_Exp28Plus_Full": (str(SUITE_DIR / "configs" / "exp30_repstarfast_hilo_haar.yaml"), True),
}


def run_verify(device_str="0"):
    from ultralytics import RTDETR
    from ultralytics.nn.modules.exp28_plus import inner_giou_nwd_loss

    device = torch.device(f"cuda:{device_str}" if torch.cuda.is_available() else "cpu")
    exp28_yaml = REPO_ROOT / "ultralytics/cfg/models/rt-detr/exp28_starfast_hilo_haar.yaml"
    exp30_yaml = SUITE_DIR / "configs" / "exp30_repstarfast_hilo_haar.yaml"

    m28 = RTDETR(str(exp28_yaml)).model
    m28.fuse()
    l28, p28 = len(list(m28.modules())), sum(p.numel() for p in m28.parameters())

    m30 = RTDETR(str(exp30_yaml)).model
    # FP64 mathematical equivalence check before & after RepStarFastBlock fusion
    m64 = copy.deepcopy(m30).to(device).double().eval()
    with torch.no_grad():
        for mod in m64.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.2, 0.2)
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.running_var.uniform_(0.5, 1.5)
        x64 = torch.randn(1, 3, 128, 128, device=device, dtype=torch.float64)
        y_before = m64(x64)
        m64.fuse()
        y_after = m64(x64)
        out_b = y_before[0] if isinstance(y_before, (tuple, list)) else y_before
        out_a = y_after[0] if isinstance(y_after, (tuple, list)) else y_after
        max_diff = (out_b - out_a).abs().max().item()
        print(f"[PASS] FP64 RepStarFastBlock fusion max_abs_diff = {max_diff:.2e}")
        assert max_diff < 1e-8

    m30.fuse()
    l30, p30 = len(list(m30.modules())), sum(p.numel() for p in m30.parameters())
    g30 = sum(p.numel() for p in m30.parameters() if p.requires_grad)
    print(f"[CHECK] Exp28 fused:    {l28} layers, {p28/1e6:.2f}M params ({p28} exact)")
    print(f"[CHECK] Exp30/32 fused: {l30} layers, {p30/1e6:.2f}M params ({p30} exact), {g30} grads")
    assert l28 == l30 and p28 == p30 and g30 == 0, "Deploy structure must match Exp28 100% identically!"

    # Verify Inner-GIoU + NWD loss finite gradients
    pred = torch.tensor([[0.5, 0.5, 0.02, 0.02], [0.3, 0.4, 0.2, 0.2]], device=device, requires_grad=True)
    gt = torch.tensor([[0.505, 0.495, 0.018, 0.022], [0.31, 0.39, 0.19, 0.21]], device=device)
    loss = inner_giou_nwd_loss(pred, gt).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    print(f"[PASS] Inner-GIoU + NWD loss = {loss.item():.4f}, gradients finite!")
    print("\nALL EXP28+ ZERO-OVERHEAD VERIFICATIONS PASSED!")


def main():
    parser = argparse.ArgumentParser(description="Fair Trainer & Verifier for Exp28+ (Exp30-Exp32)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "verify"])
    parser.add_argument("--exp", type=str, default="Exp32_Exp28Plus_Full", choices=list(EXP_SPECS.keys()))
    parser.add_argument("--data", type=str, default="dataset/DUT-Anti-UAV.yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=8, help="Compatibility parameter for batch size")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--ratio", type=float, default=1.25, help="Inner-GIoU auxiliary box scale ratio")
    parser.add_argument("--nwd-c", type=float, default=0.05, help="NWD normalized scale constant")
    parser.add_argument("--nwd-weight", type=float, default=0.25, help="Max NWD weight for tiny boxes")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=str, default="runs/hrnet_lite_suite")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.mode == "verify":
        run_verify(args.device)
        return

    model_yaml, use_inner_nwd = EXP_SPECS[args.exp]
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
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "seed": int(args.seed),
        "deterministic": False,
        "project": args.project,
        "name": f"{args.exp}_seed{args.seed}",
        "exist_ok": True,
        "pretrained": False,
    }

    print("=" * 84)
    print(f"[Exp28+ Launcher] {args.exp} | Inner-GIoU+NWD={use_inner_nwd}")
    print(json.dumps(train_kwargs, indent=2, ensure_ascii=False))
    print("=" * 84)

    if not args.execute:
        print("[DRY-RUN] Add `--execute` to launch GPU training.")
        return

    if use_inner_nwd:
        from ultralytics.nn.modules.exp28_plus import patch_rtdetr_inner_nwd_loss

        patch_rtdetr_inner_nwd_loss(ratio=args.ratio, nwd_c=args.nwd_c, nwd_max_weight=args.nwd_weight)

    from ultralytics import RTDETR

    model = RTDETR(model_yaml)
    out_dir = Path(args.project) / f"{args.exp}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(train_kwargs, use_inner_nwd=use_inner_nwd, ratio=args.ratio, nwd_c=args.nwd_c, nwd_weight=args.nwd_weight)
    (out_dir / "training_config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
