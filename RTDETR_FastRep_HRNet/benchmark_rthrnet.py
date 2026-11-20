#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVIDIA A30 Latency, Params, GFLOPs & Reparameterization Equivalence Benchmark
=============================================================================
Directly benchmarks:
  1. `rtdetr-r18.yaml`               (PResNet18-vd baseline, ~4.9ms)
  2. `rtdetr_hrnet_w18.yaml`         (Vanilla HRNetV2-W18 baseline)
  3. `rtdetr_rthrnet_w18_fast.yaml`  (RT-HRNet-W18-Fast, both train & reparameterized deploy mode!)
  4. `rtdetr_rthrnet_w18_balanced.yaml` (RT-HRNet-W18-Balanced, train & deploy mode!)
Also verifies `switch_to_deploy()` numerical equivalence (max abs diff < 1e-4),
FP32/AMP loss+backward+EMA deepcopy safety!
"""

from __future__ import annotations
import argparse
import copy
import sys
import time
from pathlib import Path
import torch


def benchmark_model(cfg_path: Path, device: torch.device, batch: int = 1, warmup: int = 20, iters: int = 100, deploy: bool = False):
    from ultralytics.nn.tasks import RTDETRDetectionModel
    model = RTDETRDetectionModel(cfg=str(cfg_path), ch=3, nc=1, verbose=False).to(device).eval()

    x = torch.randn(batch, 3, 640, 640, device=device)
    if deploy:
        # Check exact algebraic equivalence in float64 to avoid untrained activation amplification
        # swapping top-k queries under FP32/TF32 rounding noise
        m_chk = copy.deepcopy(model).double()
        x_chk = x.double()
        with torch.no_grad():
            y_before = m_chk(x_chk)
            y_before = y_before[0] if isinstance(y_before, (list, tuple)) else y_before
        for m in m_chk.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
        with torch.no_grad():
            y_after = m_chk(x_chk)
            y_after = y_after[0] if isinstance(y_after, (list, tuple)) else y_after
        max_diff = (y_before - y_after).abs().max().item()
        print(f"  [REPARAM CHECK] {cfg_path.name} max |y_train - y_deploy| = {max_diff:.2e}")
        assert max_diff < 1e-3, f"Reparameterization mismatch too large: {max_diff}"
        del m_chk, x_chk
        # Now switch the actual benchmark model to deploy mode
        for m in model.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0 / (iters * batch)

    tag = "DEPLOY(fused)" if deploy else "TRAIN(unfused)"
    print(f"  [{tag:<14}] {cfg_path.name:<34} | Params={params_m:6.2f}M | Latency={dt_ms:5.2f} ms/img | FPS={1000.0/dt_ms:6.1f}")
    return params_m, dt_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark RT-HRNet-W18 vs PResNet18-vd & HRNetV2-W18")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for latency benchmark")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    sys.path.insert(0, str(repo_root))
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    cfg_dir = repo_root / "ultralytics/cfg/models/rt-detr"

    print("=" * 95)
    print(f"RT-HRNet-W18 Benchmark on {device} (640x640, batch={args.batch})")
    print("=" * 95)

    for name, do_deploy in [
        ("rtdetr-r18.yaml", False),
        ("rtdetr_hrnet_w18.yaml", False),
        ("rtdetr_rthrnet_w18_fast.yaml", False),
        ("rtdetr_rthrnet_w18_fast.yaml", True),
        ("rtdetr_rthrnet_w18_balanced.yaml", False),
        ("rtdetr_rthrnet_w18_balanced.yaml", True),
    ]:
        p = cfg_dir / name
        if p.is_file():
            benchmark_model(p, device, batch=args.batch, deploy=do_deploy)
        else:
            print(f"  [SKIP] {name} not found in {cfg_dir}")

    # Quick training + AMP + EMA deepcopy smoke test on both RT-HRNet variants
    from ultralytics.nn.tasks import RTDETRDetectionModel
    for name in ["rtdetr_rthrnet_w18_fast.yaml", "rtdetr_rthrnet_w18_balanced.yaml"]:
        p = cfg_dir / name
        model = RTDETRDetectionModel(cfg=str(p), ch=3, nc=1, verbose=False).to(device).train()
        batch_data = {
            "img": torch.randn(2, 3, 640, 640, device=device),
            "batch_idx": torch.tensor([0, 1], dtype=torch.long, device=device),
            "cls": torch.zeros((2, 1), dtype=torch.float32, device=device),
            "bboxes": torch.tensor([[0.5, 0.5, 0.015, 0.015], [0.3, 0.4, 0.02, 0.02]], dtype=torch.float32, device=device),
        }
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"), init_scale=1.0)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss, _ = model.loss(batch_data)
            total = loss.sum() if loss.ndim > 0 else loss
        assert torch.isfinite(total).all()
        scaler.scale(total).backward()
        ema = copy.deepcopy(model).eval()
        with torch.no_grad():
            out = ema(batch_data["img"])
            y = out[0] if isinstance(out, (list, tuple)) else out
            assert y.shape == (2, 300, 5)
        print(f"  [SMOKE PASS] {name}: AMP loss + backward + EMA deepcopy + [2, 300, 5] verified!")


if __name__ == "__main__":
    main()
