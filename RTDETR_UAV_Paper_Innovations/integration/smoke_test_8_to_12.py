#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke Test for Fixed Innovation 4 + Standalone Innovations 8, 9, 10, 11, 12
===========================================================================
Verifies on the real `RTDETRDetectionModel`:
  - Baseline (`rtdetr-r18.yaml`)
  - Fixed Innovation 4 (`rtdetr-r18-inno4-sahnwd.yaml`, checking cls_loss > 0.05!)
  - Standalone Innovations 8..12 (`inno8-denseo2o`, `inno9-safdr`, `inno10-tolf`,
    `inno11-dysample-hwd`, `inno12-smaifi`)
across `GT = 0, 1, 3`, `AMP = False/True` (with `GradScaler`), non-finite gradient checks,
and `copy.deepcopy(model).eval()` (`ModelEMA` safety) + `[2, 300, 5]` output contract.
"""

from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "ultralytics" / "__init__.py").exists():
    sys.path.insert(0, str(REPO_ROOT))

CONFIGS_8_TO_12 = [
    "rtdetr-r18.yaml",
    "rtdetr-r18-inno4-sahnwd.yaml",
    "rtdetr-r18-inno8-denseo2o.yaml",
    "rtdetr-r18-inno9-safdr.yaml",
    "rtdetr-r18-inno10-tolf.yaml",
    "rtdetr-r18-inno11-dysample-hwd.yaml",
    "rtdetr-r18-inno12-smaifi.yaml",
]


def _make_batch(bs: int, num_gt: int, device: torch.device, size: int) -> dict:
    img = torch.rand(bs, 3, size, size, device=device)
    if num_gt == 0:
        return {
            "img": img,
            "batch_idx": torch.zeros((0,), dtype=torch.long, device=device),
            "cls": torch.zeros((0, 1), dtype=torch.float32, device=device),
            "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        }
    batch_idx = torch.tensor([i % bs for i in range(num_gt)], dtype=torch.long, device=device)
    cls = torch.zeros((num_gt, 1), dtype=torch.float32, device=device)
    tiny = max(0.012, 5.0 / size)
    bboxes = torch.tensor(
        [[0.50, 0.50, tiny, tiny], [0.32, 0.44, tiny * 1.3, tiny],
         [0.71, 0.28, tiny * 1.6, tiny * 1.2]][:num_gt],
        dtype=torch.float32,
        device=device,
    )
    return {"img": img, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}


def run_smoke(repo_root: Path, device_str: str = "cuda", size: int = 128, config: str | None = None) -> None:
    repo_root = repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    import ultralytics
    expected_init = (repo_root / "ultralytics" / "__init__.py").resolve()
    actual_init = Path(ultralytics.__file__).resolve()
    assert actual_init == expected_init, f"Imported wrong ultralytics: {actual_init} != {expected_init}"

    from ultralytics.nn.tasks import RTDETRDetectionModel

    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for AMP smoke test but unavailable")
    device = torch.device(device_str)
    cfg_dir = repo_root / "ultralytics" / "cfg" / "models" / "rt-detr"

    for cfg_name in ([config] if config else CONFIGS_8_TO_12):
        cfg_path = cfg_dir / cfg_name
        assert cfg_path.is_file(), f"Missing config: {cfg_path}"
        for amp in ([False, True] if device.type == "cuda" else [False]):
            for num_gt in (0, 1, 3):
                print(f"[SMOKE] cfg={cfg_name} | device={device.type} | amp={amp} | gt={num_gt}")
                model = RTDETRDetectionModel(cfg=str(cfg_path), ch=3, nc=1, verbose=False).to(device)
                model.train()
                batch = _make_batch(bs=2, num_gt=num_gt, device=device, size=size)
                model.zero_grad(set_to_none=True)

                scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"), init_scale=1.0)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(amp and device.type == "cuda")):
                    loss, loss_items = model.loss(batch)
                    total = loss.sum() if loss.ndim > 0 else loss

                assert torch.isfinite(total).all(), f"Non-finite loss in {cfg_name}: {loss} / {loss_items}"
                # Verify E4 cls_loss scaling fix (no longer 300x shrunk!)
                if cfg_name == "rtdetr-r18-inno4-sahnwd.yaml" and num_gt > 0 and loss_items.numel() >= 2:
                    assert loss_items[1].item() > 0.05, f"E4 cls_loss still too small: {loss_items[1].item()}"

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
                if num_gt == 3 and (amp or device.type != "cuda"):
                    with TemporaryDirectory() as tmp:
                        checkpoint = Path(tmp) / "checkpoint.pt"
                        torch.save({"model": model, "ema": ema_copy}, checkpoint)
                        restored = torch.load(checkpoint, map_location=device, weights_only=False)
                        restored["ema"].eval()
                        with torch.no_grad():
                            saved_pred = restored["ema"](batch["img"])
                            saved_y = saved_pred[0] if isinstance(saved_pred, (list, tuple)) else saved_pred
                        assert saved_y.shape == (2, 300, 5), f"Checkpoint output mismatch: {cfg_name}"

    print("[PASS] Baseline + Fixed Inno4 + Standalone Innovations 8–12 ALL passed smoke test!")


def main():
    parser = argparse.ArgumentParser(description="Smoke test for RT-DETR UAV Innovations 8–12 & Fixed Inno 4")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main repository root")
    parser.add_argument("--device", type=str, default="cuda", help="Device ('cuda' or 'cpu')")
    parser.add_argument("--size", type=int, default=128, help="Input size for smoke test (default: 128)")
    parser.add_argument("--config", choices=CONFIGS_8_TO_12, help="Test one model; default tests all seven")
    args = parser.parse_args()
    run_smoke(Path(args.repo), args.device, args.size, args.config)


if __name__ == "__main__":
    main()
