"""Shared registry and protocol helpers for the R09--R16 targeted variants.

The helpers in this module are deliberately side-effect free.  They validate
configuration files and build the command payload used by the train/evaluate
and benchmark entry points, but never start a training job implicitly.
"""
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
RUNS = ROOT / "runs" / "r18_targeted"

CONFIGS = {
    "r09_lipdown": "rtdetr_r18_lipdown.yaml",
    "r10_phasedown": "rtdetr_r18_phasedown.yaml",
    "r11_absr": "rtdetr_r18_absr.yaml",
    "r12_lcr": "rtdetr_r18_lcr.yaml",
    "r13_coordrefine": "rtdetr_r18_coordrefine.yaml",
    "r14_skresidual": "rtdetr_r18_skresidual.yaml",
    "r15_sdf": "rtdetr_r18_sdf.yaml",
    "r16_sfr": "rtdetr_r18_sfr.yaml",
}

DESCRIPTIONS = {
    "r09_lipdown": "local-importance weighted P2-to-P3 downsampling",
    "r10_phasedown": "phase-aware PixelUnshuffle downsampling",
    "r11_absr": "adaptive background smoothing residual at P3",
    "r12_lcr": "local contrast residual at P3",
    "r13_coordrefine": "coordinate-attention residual refinement at P3",
    "r14_skresidual": "selective 3x3/5x5 depthwise Stage4 residual",
    "r15_sdf": "semantic detail feedback from P4 into the P3 neck source",
    "r16_sfr": "saliency-gated feature residual at P3",
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
        raise ValueError(f"unknown targeted variant {name!r}; choose from {tuple(CONFIGS)}")
    return MODEL_DIR / CONFIGS[name]


def model_names(value: str) -> list[str]:
    names = list(CONFIGS) if value == "all" else [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in names if x not in CONFIGS]
    if unknown:
        raise ValueError(f"unknown targeted variants {unknown}; choose from {tuple(CONFIGS)} or all")
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
        mismatch = {k: (baseline.get(k), profile.get(k)) for k in keys if baseline.get(k) != profile.get(k)}
        if mismatch:
            parser.error(f"baseline protocol differs from checked-in profile: {mismatch}")
        source = baseline
    else:
        print("WARNING: historical R18 args.yaml was not supplied; exact baseline parity is unverified.", flush=True)
        source = profile
    source = dict(source)
    if args.epochs is not None:
        source["epochs"] = int(args.epochs)
    if args.batch is not None:
        source["batch"] = int(args.batch)
        if args.micro_batch is None:
            source["micro_batch"] = int(args.batch)
    if args.micro_batch is not None:
        source["micro_batch"] = int(args.micro_batch)
    effective = int(source.get("batch", 8))
    micro = int(source.get("micro_batch", effective))
    if effective < 1 or micro < 1 or effective % micro:
        parser.error(f"effective batch {effective} must be a positive multiple of micro-batch {micro}")
    allowed = yaml.safe_load((ROOT / "ultralytics" / "cfg" / "default.yaml").read_text(encoding="utf-8"))
    settings = {k: v for k, v in source.items() if k in allowed}
    for key in ("model", "mode", "project", "name", "device", "resume", "save_dir", "cfg", "data"):
        settings.pop(key, None)
    settings.update(data=str(DATA), device=args.device, project=str(args.output_root / args.model),
                    name=f"seed{args.seed}", exist_ok=True, seed=args.seed, batch=micro,
                    gradient_accumulation=effective // micro)
    return settings, {"effective_batch": effective, "micro_batch": micro,
                      "gradient_accumulation": effective // micro,
                      "warmup_epochs_value": source.get("warmup_epochs"),
                      "warmup_unit": "micro-batch iterations in this trainer"}


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


def validate_yaml(name: str) -> dict:
    """Validate immutable baseline head and route contracts for a variant."""
    cfg = yaml.safe_load(config_path(name).read_text(encoding="utf-8"))
    base = yaml.safe_load((MODEL_DIR / "rtdetr-r18.yaml").read_text(encoding="utf-8"))
    if name != "r15_sdf" and cfg.get("head") != base.get("head"):
        raise ValueError(f"{name} must preserve the original RT-DETR-R18 head")
    if name == "r15_sdf":
        # R15 is the only legal head-source edit: P3 comes from the SDF selector.
        dec = cfg.get("head", [])[-1]
        if dec[2] != "RTDETRDecoder" or dec[0] != [20, 23, 26]:
            raise ValueError(f"unexpected R15 decoder routes: {dec[0]}")
    return {"name": name, "yaml": str(config_path(name)), "description": DESCRIPTIONS[name]}


def expected_shapes(name: str, input_size: int = 640):
    """Return the immutable R18 feature contract for a variant."""
    return {"backbone": [(1, 128, input_size // 8, input_size // 8),
                          (1, 256, input_size // 16, input_size // 16),
                          (1, 512, input_size // 32, input_size // 32)],
            "decoder": [(1, 256, input_size // 8, input_size // 8),
                        (1, 256, input_size // 16, input_size // 16),
                        (1, 256, input_size // 32, input_size // 32)]}


def preflight(model, name: str, input_size: int = 640):
    """Run one inference pass and verify P3/P4/P5 and decoder routes."""
    found, decoder, sdf_inputs = {}, {}, {}
    indices = (5, 6, 8) if name == "r15_sdf" else (5, 6, 7)
    handles = []
    for index, key in zip(indices, ("p3", "p4", "p5")):
        handles.append(model.model[index].register_forward_hook(
            lambda _m, _i, output, key=key: found.__setitem__(key, tuple(output.shape))))
    handles.append(model.model[-1].register_forward_pre_hook(
        lambda _m, values: decoder.__setitem__("inputs", [tuple(item.shape) for item in values[0]])))
    if name == "r15_sdf":
        handles.append(model.model[7].register_forward_pre_hook(
            lambda _m, values: sdf_inputs.__setitem__("inputs", [tuple(item.shape) for item in values[0]])))
    try:
        import torch
        with torch.inference_mode(): model(torch.zeros(1, 3, input_size, input_size))
    finally:
        for handle in handles: handle.remove()
    result = {"backbone": [found[k] for k in ("p3", "p4", "p5")],
              "decoder": decoder.get("inputs"), "decoder_from": list(model.model[-1].f)}
    if sdf_inputs: result["sdf_inputs"] = sdf_inputs["inputs"]
    expected = expected_shapes(name, input_size)
    if result["backbone"] != expected["backbone"] or result["decoder"] != expected["decoder"]:
        raise RuntimeError(f"{name} preflight mismatch: {result} != {expected}")
    expected_routes = [20, 23, 26] if name == "r15_sdf" else [19, 22, 25]
    if result["decoder_from"] != expected_routes:
        raise RuntimeError(f"{name} changed decoder routes: {result['decoder_from']}")
    if name == "r15_sdf" and result.get("sdf_inputs") != [(1, 128, input_size // 8, input_size // 8), (1, 256, input_size // 16, input_size // 16)]:
        raise RuntimeError(f"{name} SDF route mismatch: {result.get('sdf_inputs')}")
    return result

