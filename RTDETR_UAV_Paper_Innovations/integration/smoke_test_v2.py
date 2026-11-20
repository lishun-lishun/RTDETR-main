#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke Test for Empirically Reconstructed V2 Modules (V2-A, V2-B, V2-C, V2-FULL)
===============================================================================
Verifies Baseline (`rtdetr-r18.yaml`), `rtdetr-r18-v2a-wcasfpv2.yaml`,
`rtdetr-r18-v2b-dmbasc.yaml`, `rtdetr-r18-v2c-cauad.yaml`, and `rtdetr-r18-v2-full.yaml`
across `GT = 0, 1, 3`, `AMP = False/True` (`GradScaler`), and `copy.deepcopy(model).eval()`.
"""

from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path
import torch

CONFIGS_V2 = [
    "rtdetr-r18.yaml",
    "rtdetr-r18-v2a-wcasfpv2.yaml",
    "rtdetr-r18-v2b-dmbasc.yaml",
    "rtdetr-r18-v2c-cauad.yaml",
    "rtdetr-r18-v2-full.yaml",
]


def _make_batch(bs: int, num_gt: int, device: torch.device) -> dict:
    img = torch.randn(bs, 3, 640, 640, device=device)
    if num_gt == 0:
        return {
            "img": img,
            "batch_idx": torch.zeros((0,), dtype=torch.long, device=device),
            "cls": torch.zeros((0, 1), dtype=torch.float32, device=device),
            "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        }
    batch_idx = torch.tensor([i % bs for i in range(num_gt)], dtype=torch.long, device=device)
    cls = torch.zeros((num_gt, 1), dtype=torch.float32, device=device)
    bboxes = torch.tensor(
        [[0.50, 0.50, 0.012, 0.012], [0.32, 0.44, 0.018, 0.015], [0.71, 0.28, 0.025, 0.022]][:num_gt],
        dtype=torch.float32,
        device=device,
    )
    return {"img": img, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}


def run_smoke(repo_root: Path, device_str: str = "cuda") -> None:
    repo_root = repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    import ultralytics
    expected_init = (repo_root / "ultralytics" / "__init__.py").resolve()
    actual_init = Path(ultralytics.__file__).resolve()
    assert actual_init == expected_init, f"Imported wrong ultralytics: {actual_init} != {expected_init}"

    from ultralytics.nn.tasks import RTDETRDetectionModel

    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    cfg_dir = repo_root / "ultralytics" / "cfg" / "models" / "rt-detr"

    for cfg_name in CONFIGS_V2:
        cfg_path = cfg_dir / cfg_name
        assert cfg_path.is_file(), f"Missing config: {cfg_path}"
        for amp in ([False, True] if device.type == "cuda" else [False]):
            for num_gt in (0, 1, 3):
                print(f"[SMOKE-V2] cfg={cfg_name} | device={device.type} | amp={amp} | gt={num_gt}")
                model = RTDETRDetectionModel(cfg=str(cfg_path), ch=3, nc=1, verbose=False).to(device)
                model.train()
                batch = _make_batch(bs=2, num_gt=num_gt, device=device)
                model.zero_grad(set_to_none=True)

                scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"), init_scale=1.0)
                with torch.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
                    loss, loss_items = model.loss(batch)
                    total = loss.sum() if loss.ndim > 0 else loss

                assert torch.isfinite(total).all(), f"Non-finite loss in {cfg_name}: {loss} / {loss_items}"
                scaler.scale(total).backward()

                bad_grads = [
                    name for name, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()
                ]
                assert not bad_grads, f"Non-finite gradients in {cfg_name}: {bad_grads[:12]}"

                ema_copy = copy.deepcopy(model).eval()
                with torch.no_grad():
                    preds = ema_copy(batch["img"])
                    y = preds[0] if isinstance(preds, (list, tuple)) else preds
                    assert y.shape == (2, 300, 5), f"Unexpected eval shape for {cfg_name}: {y.shape}"

    print("[PASS] All V2 Reconstructed Models (V2-A, V2-B, V2-C, V2-FULL) passed smoke test!")


def main():
    parser = argparse.ArgumentParser(description="Smoke test for V2 Reconstructed RT-DETR UAV Models")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main repository root")
    parser.add_argument("--device", type=str, default="cuda", help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()
    run_smoke(Path(args.repo), args.device)


if __name__ == "__main__":
    main()
