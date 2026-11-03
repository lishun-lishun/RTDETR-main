"""Shared paths, protocol and preflight helpers for LDW-R18 experiments."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
DATA = ROOT / "dataset" / "DUT-Anti-UAV.yaml"
PROFILE = ROOT / "ultralytics" / "cfg" / "csp_context_default.yaml"
RUNS = ROOT / "runs" / "ldw_r18"
CONFIGS = {
    "stage4": "rtdetr_r18_ldw_stage4.yaml",
    "stage5": "rtdetr_r18_ldw_stage5.yaml",
    "full": "rtdetr_r18_ldw_full.yaml",
    "early_stage4": "rtdetr_r18_early_stage4.yaml",
    "early_stage5": "rtdetr_r18_early_stage5.yaml",
    "dilation": "rtdetr_r18_ldw_dilation.yaml",
}


def local_package():
    """Ensure scripts use this checkout instead of an installed Ultralytics copy."""
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.find_spec("ultralytics")
    actual = Path(spec.origin).resolve() if spec and spec.origin else None
    expected = (ROOT / "ultralytics" / "__init__.py").resolve()
    if actual != expected:
        raise RuntimeError(f"Expected local Ultralytics {expected}, found {actual}")


def config_path(name: str) -> Path:
    if name not in CONFIGS:
        raise ValueError(f"unknown LDW model {name!r}; choose from {tuple(CONFIGS)}")
    return MODEL_DIR / CONFIGS[name]


def protocol_settings(args, parser):
    """Load the checked-in baseline protocol and make the run directory explicit."""
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    if args.baseline_args:
        baseline_file = Path(args.baseline_args)
        if not baseline_file.is_file():
            parser.error("baseline args.yaml does not exist")
        baseline = yaml.safe_load(baseline_file.read_text(encoding="utf-8"))
        keys = ("epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "weight_decay", "amp", "pretrained")
        mismatches = {key: (baseline.get(key), profile.get(key)) for key in keys
                      if baseline.get(key) != profile.get(key)}
        if mismatches:
            parser.error(f"baseline protocol differs from checked-in profile: {mismatches}")
        source = baseline
    else:
        print("WARNING: historical R18 args.yaml was not supplied; exact baseline parity is unverified.", flush=True)
        source = profile
    allowed = yaml.safe_load((ROOT / "ultralytics" / "cfg" / "default.yaml").read_text(encoding="utf-8"))
    settings = {key: value for key, value in source.items() if key in allowed}
    for key in ("model", "mode", "project", "name", "device", "resume", "save_dir", "cfg", "data"):
        settings.pop(key, None)
    settings.update(
        data=str(DATA), device=args.device, project=str(args.output_root / args.model),
        name=f"seed{args.seed}", exist_ok=True, seed=args.seed,
    )
    return settings


def preflight(model, input_size=128):
    """Return backbone and decoder feature shapes for a cheap construction check."""
    import torch

    found = {}
    handles = [model.model[index].register_forward_hook(
        lambda _m, _i, output, index=index: found.__setitem__(index, tuple(output.shape)))
               for index in (5, 6, 7)]
    with torch.inference_mode():
        model(torch.zeros(1, 3, input_size, input_size))
    for handle in handles:
        handle.remove()
    found["decoder_from"] = list(model.model[-1].f)
    return found


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
