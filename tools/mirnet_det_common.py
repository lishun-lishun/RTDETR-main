"""Shared configuration and protocol helpers for MIRNet-Det experiments."""

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
RUNS = ROOT / "runs" / "mirnet_det"
CONFIGS = {
    "mir_base": "rtdetr_mirnet_base.yaml",
    "mir_dual": "rtdetr_mirnet_dual.yaml",
    "mir_full": "rtdetr_mirnet_full.yaml",
    "mir_stride": "rtdetr_mirnet_stride.yaml",
    "mir_nofusion": "rtdetr_mirnet_nofusion.yaml",
    "mir_zero_fusion": "rtdetr_mirnet_zero_fusion.yaml",
}


def local_package():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.find_spec("ultralytics")
    actual = Path(spec.origin).resolve() if spec and spec.origin else None
    expected = (ROOT / "ultralytics" / "__init__.py").resolve()
    if actual != expected:
        raise RuntimeError(f"Expected local Ultralytics {expected}, found {actual}")


def config_path(name: str) -> Path:
    if name not in CONFIGS:
        raise ValueError(f"unknown MIRNet model {name!r}; choose from {tuple(CONFIGS)}")
    return MODEL_DIR / CONFIGS[name]


def protocol_settings(args, parser):
    profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    if args.baseline_args:
        baseline_path = Path(args.baseline_args)
        if not baseline_path.is_file():
            parser.error("baseline args.yaml does not exist")
        baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
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
    settings.update(data=str(DATA), device=args.device,
                    project=str(args.output_root / args.model),
                    name=f"seed{args.seed}", exist_ok=True, seed=args.seed)
    return settings


def preflight(model, input_size=128):
    import torch

    found = {}
    handles = [model.model[index].register_forward_hook(
        lambda _m, _i, output, index=index: found.__setitem__(index, tuple(output.shape)))
               for index in (1, 2, 3)]
    decoder_shapes = {}
    handles.append(model.model[-1].register_forward_pre_hook(
        lambda _m, values: decoder_shapes.__setitem__(
            "inputs", [tuple(x.shape) for x in values[0]])))
    with torch.inference_mode():
        model(torch.zeros(1, 3, input_size, input_size))
    for handle in handles:
        handle.remove()
    found["decoder"] = decoder_shapes.get("inputs")
    found["decoder_from"] = list(model.model[-1].f)
    return found


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
