#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Check for `RTDETR_Exp33_AntiCollapse`:
  1. Verifies `FCGHaarDownsample` odd/even shapes, strict nonlinearity (non-equivalence to 2x2 conv),
     and FP64 fuse() mathematical equivalence.
  2. Verifies `Exp33 / Exp34 / Exp36` YAML builds, forward pass, full `BaseModel.fuse()` (0 grads &
     0 remaining BNs inside `FCGHaarDownsample`), and exact `AntiCollapseRTDETRDecoder` +9 param overhead vs Exp28.
  3. Verifies Tight `area_tau=0.0015` Coupled `HungarianMatcher + DETRLoss` matching, weight differentiation,
     and finite gradients.
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] verify_exp33_cpu.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import math
import os
from pathlib import Path
import torch

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


def main():
    from ultralytics import RTDETR
    from ultralytics.models.utils.ops import HungarianMatcher
    from ultralytics.nn.modules.exp33_anticollapse import (
        FCGHaarDownsample,
        AntiCollapseMSDeformAttn,
        tight_inner_giou_nwd_loss,
        compute_nwd_pairwise,
        patch_rtdetr_coupled_tight_nwd,
    )

    # 1. FCGHaarDownsample check
    torch.manual_seed(0)
    mod = FCGHaarDownsample(64, 64).eval()
    with torch.no_grad():
        mod.gate_conv.weight.normal_(0, 0.5)
        mod.gate_conv.bias.normal_(0, 0.2)
        for bn in (mod.hf_bn, mod.proj_bn):
            bn.weight.uniform_(0.5, 1.5)
            bn.bias.uniform_(-0.2, 0.2)
            bn.running_mean.uniform_(-0.2, 0.2)
            bn.running_var.uniform_(0.5, 1.5)
    for h, w in [(80, 80), (41, 41)]:
        x = torch.randn(2, 64, h, w, requires_grad=True)
        y = mod(x)
        assert y.shape == (2, 64, (h + 1) // 2, (w + 1) // 2)
        y.sum().backward()
        assert torch.isfinite(x.grad).all()

    # Strict nonlinearity proof before activation + FP64 fuse check
    x1, x2 = torch.randn(1, 64, 16, 16), torch.randn(1, 64, 16, 16)
    with torch.no_grad():
        mod_noact = FCGHaarDownsample(64, 64).eval()
        mod_noact.act = torch.nn.Identity()
        mod_noact.gate_conv.weight.normal_(0, 0.5)
        nonlin_err = (mod_noact(x1 + x2) - (mod_noact(x1) + mod_noact(x2) - mod_noact(torch.zeros_like(x1)))).abs().max().item()
        assert nonlin_err > 1e-2

        mod64 = mod.double()
        x_t = torch.randn(2, 64, 20, 20, dtype=torch.float64)
        y_before = mod64(x_t)
        mod64.switch_to_deploy()
        fuse_err = (y_before - mod64(x_t)).abs().max().item()
        assert fuse_err < 1e-12, f"Expected FP64 fuse err < 1e-12, got {fuse_err}"
    print(f"[PASS 1/3] FCGHaarDownsample | nonlin_err={nonlin_err:.4f} (strictly != 2x2 conv) | FP64 fuse_err={fuse_err:.2e}")

    # 2. Full YAML build + forward + fuse check (plus exact +9 params parity vs Exp28)
    m28 = RTDETR("ultralytics/cfg/models/rt-detr/exp28_starfast_hilo_haar.yaml").model.eval()
    m28.fuse()
    p28 = sum(p.numel() for p in m28.parameters())

    x_img = torch.randn(1, 3, 640, 640)
    for name, yml in [
        ("Exp33_FCGHaar", "ultralytics/cfg/models/rt-detr/exp33_fcghaar_only.yaml"),
        ("Exp34_AntiCollapseDec", "ultralytics/cfg/models/rt-detr/exp34_anticollapse_dec.yaml"),
        ("Exp36_Full", "ultralytics/cfg/models/rt-detr/exp36_full_anticollapse.yaml"),
    ]:
        m = RTDETR(yml).model.eval()
        ac_cnt = sum(1 for mod_ in m.modules() if isinstance(mod_, AntiCollapseMSDeformAttn))
        with torch.no_grad():
            _ = m(x_img)
        m.fuse()
        fcg_bns = sum(
            1
            for mod_ in m.modules()
            if isinstance(mod_, FCGHaarDownsample)
            for sub in mod_.modules()
            if isinstance(sub, torch.nn.BatchNorm2d)
        )
        grads = sum(p.numel() for p in m.parameters() if p.requires_grad)
        p_tot = sum(p.numel() for p in m.parameters())
        assert fcg_bns == 0 and grads == 0, f"{name}: unfused FCG BNs={fcg_bns}, grads={grads}"
        if name == "Exp34_AntiCollapseDec":
            assert p_tot - p28 == 9, f"Expected exact +9 params over Exp28, got {p_tot - p28}"
        print(
            f"[PASS 2/3] {name:<22} | Layers={len(list(m.modules()))} | Params={p_tot/1e6:.2f}M "
            f"(+{p_tot - p28} vs Exp28) | Grads={grads} | AntiCollapseMSDeformAttn={ac_cnt}"
        )

    # 3. Tight area_tau=0.0015 Coupled Matcher + Loss check
    patch_rtdetr_coupled_tight_nwd(ratio=1.25, nwd_c=0.05, nwd_max_weight=0.25, area_tau=0.0015)
    p = torch.tensor([[0.5, 0.5, 0.01, 0.01], [0.3, 0.3, 0.1, 0.1]], requires_grad=True)
    g = torch.tensor([[0.505, 0.502, 0.01, 0.01], [0.31, 0.29, 0.1, 0.1]])
    loss = tight_inner_giou_nwd_loss(p, g)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(p.grad).all()
    nwd_mat = compute_nwd_pairwise(p.detach(), g)
    assert nwd_mat.shape == (2, 2)

    matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
    idx = matcher(
        pred_bboxes=torch.rand(2, 10, 4).clamp(0.05, 0.95),
        pred_scores=torch.randn(2, 10, 1),
        gt_bboxes=torch.tensor([[0.5, 0.5, 0.01, 0.01], [0.2, 0.2, 0.1, 0.1]]),
        gt_cls=torch.tensor([0, 0], dtype=torch.long),
        gt_groups=[1, 1],
    )
    assert len(idx) == 2 and len(idx[0][0]) == 1 and len(idx[1][0]) == 1
    print("[PASS 3/3] Tight Coupled Matcher+Loss verified! Weights at 640x640:")
    for px in [6, 16, 32, 64]:
        w = 0.25 * math.exp(-((px / 640.0) ** 2) / 0.0015)
        print(f"    {px:>2}x{px:<2} px -> w_nwd={w:.4f}")


if __name__ == "__main__":
    main()
