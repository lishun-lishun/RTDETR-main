"""Shared registry and protocol helpers for the independent R18 variants."""

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
RUNS = ROOT / "runs" / "r18_eight_variants"

CONFIGS = {
    "baseline": "rtdetr-r18.yaml",
    "r18_stemconv": "rtdetr_r18_stemconv.yaml",
    "r18_aastem": "rtdetr_r18_aastem.yaml",
    "r18_resnetd": "rtdetr_r18_resnetd.yaml",
    "r18_p3deep": "rtdetr_r18_p3deep.yaml",
    "r18_p3wide": "rtdetr_r18_p3wide.yaml",
    "r18_p4dilated": "rtdetr_r18_p4dilated.yaml",
    "r18_p4dcn": "rtdetr_r18_p4dcn.yaml",
    "r18_p5pconv": "rtdetr_r18_p5pconv.yaml",
}

REFERENCE_CONFIGS = {
    "hrnetv2_w18": "rtdetr_hrnet_w18.yaml",
}

VARIANT_DESCRIPTIONS = {
    "baseline": "unchanged original RT-DETR-R18",
    "r18_stemconv": "replace stem MaxPool with Conv3x3 stride 2",
    "r18_aastem": "stride-1 MaxPool plus fixed BlurPool",
    "r18_resnetd": "explicit ResNet-D downsampling shortcuts",
    "r18_p3deep": "one additional Stage3 BasicBlock",
    "r18_p3wide": "Stage3 output width 160",
    "r18_p4dilated": "Stage4 second block branch2b dilation 2",
    "r18_p4dcn": "Stage4 second block branch2b DeformConv2d",
    "r18_p5pconv": "Stage5 second block branch2a PartialConv3",
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
        raise ValueError(f"unknown R18 variant {name!r}; choose from {tuple(CONFIGS)}")
    return MODEL_DIR / CONFIGS[name]


def model_names(value: str) -> list[str]:
    names = list(CONFIGS) if value == "all" else [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in names if item not in CONFIGS]
    if unknown:
        raise ValueError(f"unknown R18 variants {unknown}; choose from {tuple(CONFIGS)} or all")
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
    if args.batch is not None:
        source["batch"] = int(args.batch)
        if args.micro_batch is None:
            source["micro_batch"] = int(args.batch)
    if args.micro_batch is not None:
        source["micro_batch"] = int(args.micro_batch)
    effective_batch = int(source.get("batch", 8))
    micro_batch = int(source.get("micro_batch", effective_batch))
    if effective_batch < 1 or micro_batch < 1 or effective_batch % micro_batch:
        parser.error(f"effective batch {effective_batch} must be a positive multiple of micro-batch {micro_batch}")
    allowed = yaml.safe_load((ROOT / "ultralytics" / "cfg" / "default.yaml").read_text(encoding="utf-8"))
    settings = {key: value for key, value in source.items() if key in allowed}
    for key in ("model", "mode", "project", "name", "device", "resume", "save_dir", "cfg", "data"):
        settings.pop(key, None)
    settings.update(data=str(DATA), device=args.device, project=str(args.output_root / args.model),
                    name=f"seed{args.seed}", exist_ok=True, seed=args.seed, batch=micro_batch,
                    gradient_accumulation=effective_batch // micro_batch)
    return settings, {
        "effective_batch": effective_batch,
        "micro_batch": micro_batch,
        "gradient_accumulation": effective_batch // micro_batch,
        "warmup_epochs_value": source.get("warmup_epochs"),
        "warmup_unit": "micro-batch iterations in this trainer",
    }


def expected_shapes(name: str, input_size: int = 640):
    p3_channels = 160 if name == "r18_p3wide" else 128
    return {
        "backbone": [(1, p3_channels, input_size // 8, input_size // 8),
                     (1, 256, input_size // 16, input_size // 16),
                     (1, 512, input_size // 32, input_size // 32)],
        "decoder": [(1, 256, input_size // 8, input_size // 8),
                    (1, 256, input_size // 16, input_size // 16),
                    (1, 256, input_size // 32, input_size // 32)],
    }


def preflight(model, variant: str, input_size: int = 640):
    import torch

    found, decoder = {}, {}
    layers = model.model
    handles = []
    for index, key in ((5, "p3"), (6, "p4"), (7, "p5")):
        handles.append(layers[index].register_forward_hook(
            lambda _m, _i, output, key=key: found.__setitem__(key, tuple(output.shape))))
    handles.append(layers[-1].register_forward_pre_hook(
        lambda _m, values: decoder.__setitem__("inputs", [tuple(item.shape) for item in values[0]])))
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, input_size, input_size))
    finally:
        for handle in handles:
            handle.remove()
    result = {"backbone": [found[key] for key in ("p3", "p4", "p5")],
              "decoder": decoder.get("inputs"), "decoder_from": list(layers[-1].f)}
    expected = expected_shapes(variant, input_size)
    if result["backbone"] != expected["backbone"] or result["decoder"] != expected["decoder"]:
        raise RuntimeError(f"{variant} preflight mismatch: {result} != {expected}")
    if result["decoder_from"] != [19, 22, 25]:
        raise RuntimeError(f"{variant} changed the original decoder routes: {result['decoder_from']}")
    return result


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
