"""Shared registry, protocol and preflight helpers for eight-backbone runs."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
DATA = ROOT / "dataset" / "DUT-Anti-UAV.yaml"
PROFILE = ROOT / "ultralytics" / "cfg" / "csp_context_default.yaml"
RUNS = ROOT / "runs" / "eight_backbones"
CONFIGS = {
    "hgnetv2_b2": "rtdetr_hgnetv2_b2.yaml",
    "convnextv2_pico": "rtdetr_convnextv2_pico.yaml",
    "fasternet_t1": "rtdetr_fasternet_t1.yaml",
    "repvit_m1_1": "rtdetr_repvit_m11.yaml",
    "efficientvit_b1": "rtdetr_efficientvit_b1.yaml",
    "pvt_v2_b1": "rtdetr_pvtv2_b1.yaml",
    "lsknet_t": "rtdetr_lsknet_t.yaml",
    "pkinetv2_t": "rtdetr_pkinetv2_t.yaml",
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
        raise ValueError(f"unknown backbone {name!r}; choose from {tuple(CONFIGS)}")
    return MODEL_DIR / CONFIGS[name]


def model_names(value: str) -> list[str]:
    if value == "all":
        return list(CONFIGS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in names if item not in CONFIGS]
    if unknown:
        raise ValueError(f"unknown models {unknown}; choose from {tuple(CONFIGS)} or all")
    if len(set(names)) != len(names):
        raise ValueError("model list contains duplicates")
    return names


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
    source = dict(source)
    if getattr(args, "batch", None) is not None:
        source["batch"] = int(args.batch)
        if getattr(args, "micro_batch", None) is None:
            source["micro_batch"] = int(args.batch)
    if getattr(args, "micro_batch", None) is not None:
        source["micro_batch"] = int(args.micro_batch)
    effective_batch = int(source.get("batch", 8))
    micro_batch = int(source.get("micro_batch", effective_batch))
    if effective_batch < 1 or micro_batch < 1 or effective_batch % micro_batch:
        parser.error(f"effective batch {effective_batch} must be a positive multiple of micro_batch {micro_batch}")
    allowed = yaml.safe_load((ROOT / "ultralytics" / "cfg" / "default.yaml").read_text(encoding="utf-8"))
    settings = {key: value for key, value in source.items() if key in allowed}
    for key in ("model", "mode", "project", "name", "device", "resume", "save_dir", "cfg", "data"):
        settings.pop(key, None)
    settings.update(
        data=str(DATA), device=args.device, project=str(args.output_root / args.model),
        name=f"seed{args.seed}", exist_ok=True, seed=args.seed,
        batch=micro_batch, gradient_accumulation=effective_batch // micro_batch,
    )
    return settings, {
        "effective_batch": effective_batch,
        "micro_batch": micro_batch,
        "gradient_accumulation": effective_batch // micro_batch,
        "warmup_epochs_value": source.get("warmup_epochs"),
        "warmup_unit": "micro-batch iterations in this trainer",
    }


def preflight(model, input_size=128):
    import torch
    from ultralytics.nn.modules import BackboneAdapter, FeatureSelect

    adapter = next(module for module in model.modules() if isinstance(module, BackboneAdapter))
    selectors = [module for module in model.model if isinstance(module, FeatureSelect)]
    if len(selectors) != 3 or [selector.index for selector in selectors] != [0, 1, 2]:
        raise RuntimeError("expected FeatureSelect(0/1/2) after BackboneAdapter")
    found, decoder = {}, {}
    handles = [adapter.register_forward_hook(
        lambda _m, _i, output: found.__setitem__("backbone", [tuple(item.shape) for item in output]))]
    handles.extend(selector.register_forward_hook(
        lambda _m, _i, output, key=key: found.__setitem__(key, tuple(output.shape)))
                   for key, selector in zip(("p3", "p4", "p5"), selectors))
    handles.append(model.model[-1].register_forward_pre_hook(
        lambda _m, values: decoder.__setitem__("inputs", [tuple(item.shape) for item in values[0]])))
    with torch.inference_mode():
        model(torch.zeros(1, 3, input_size, input_size))
    for handle in handles:
        handle.remove()
    found["decoder"] = decoder.get("inputs")
    found["decoder_from"] = list(model.model[-1].f)
    found["source"] = {
        "model_name": adapter.model_name,
        "resolved_model_name": adapter.resolved_model_name,
        "timm_version": adapter.timm_version,
        "source_channels": list(adapter.source_channels),
        "selected_indices": list(adapter._source_outputs),
    }
    return found


def git_state():
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--porcelain")}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

