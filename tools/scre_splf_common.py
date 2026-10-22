"""Shared paths and checkpoint helpers for the SCRE/SPLF experiments."""

import json
import importlib.util
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
DATA_CONFIG = ROOT / "dataset" / "DUT-Anti-UAV.yaml"
PROFILE_CONFIG = ROOT / "ultralytics" / "cfg" / "anti_detr_default.yaml"
OUTPUT_ROOT = ROOT / "runs" / "r18_scre_splf"
VARIANTS = {
    "scre": "rtdetr_r18_scre.yaml",
    "splf": "rtdetr_r18_splf.yaml",
    "scre_splf": "rtdetr_r18_scre_splf.yaml",
    "no_offset": "rtdetr_r18_scre_splf_no_offset.yaml",
    "no_protect": "rtdetr_r18_scre_splf_no_protect.yaml",
}
BASELINE_YAML = yaml.safe_load((MODEL_DIR / "rtdetr-r18.yaml").read_text(encoding="utf-8"))
BACKBONE_COUNT = len(BASELINE_YAML["backbone"])
BASE_DECODER_INDEX = BACKBONE_COUNT + len(BASELINE_YAML["head"]) - 1


def ensure_local_ultralytics():
    """Use this repository's patched package, not the conda site-package.

    Running ``python tools/script.py`` puts ``tools/`` (not the repository
    root) at sys.path[0]. Without this guard, tests can pass against the local
    package while the actual training process silently imports a different
    Ultralytics installation.
    """
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.find_spec("ultralytics")
    expected = (ROOT / "ultralytics" / "__init__.py").resolve()
    actual = Path(spec.origin).resolve() if spec and spec.origin else None
    if actual != expected:
        raise RuntimeError(f"Expected local Ultralytics at {expected}, but Python resolved {actual}")
    return actual


def yaml_path(variant):
    return MODEL_DIR / VARIANTS[variant]


def read_baseline_args(path):
    if path is None:
        raise ValueError("A real baseline args.yaml is required for training parity")
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"--baseline-args must point to the original R18 args.yaml file, not a directory "
                         f"or missing path: {str(path)!r}. If using $BASE_ARGS, set it in this shell first.")
    args = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(args, dict):
        raise ValueError(f"Invalid baseline args.yaml: {path}")
    expected = {"epochs": 150, "imgsz": 640, "optimizer": "AdamW", "lr0": 0.0001,
                "lrf": 1.0,
                "weight_decay": 0.0001, "seed": 0, "amp": False, "pretrained": False,
                "freeze": None}
    mismatches = {k: (args.get(k), v) for k, v in expected.items() if args.get(k) != v}
    effective_batch = int(args.get("batch", 0)) * int(args.get("gradient_accumulation", 1))
    if mismatches or effective_batch != 8:
        raise ValueError(f"Baseline record differs from the specified protocol: {mismatches}; "
                         f"effective_batch={effective_batch}")
    return args


def main_layer_indices(variant):
    """Return original backbone+CCFM+decoder indices for protocol-A freezing."""
    if variant == "splf":
        return list(range(BASE_DECODER_INDEX)) + [BASE_DECODER_INDEX + 1]
    return (list(range(BACKBONE_COUNT)) +
            list(range(BACKBONE_COUNT + 1, BASE_DECODER_INDEX + 1)) +
            [BASE_DECODER_INDEX + 2])


def transfer_baseline(model, checkpoint, variant):
    """Map every baseline model state entry to its shifted target layer."""
    import torch

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_model = stored.get("ema") or stored.get("model") if isinstance(stored, dict) else stored
    if source_model is None or not hasattr(source_model, "state_dict"):
        raise ValueError(f"Checkpoint does not contain a model: {checkpoint}")
    source_yaml = getattr(source_model, "yaml", {})
    if (source_yaml.get("scre_splf") or source_yaml.get("backbone") != BASELINE_YAML["backbone"] or
            source_yaml.get("head") != BASELINE_YAML["head"] or source_yaml.get("nc") != 1):
        raise ValueError("Protocol-A initialization requires the one-class original R18 checkpoint")
    source = source_model.float().state_dict()
    target = model.state_dict()
    mapped = {}
    missed = []
    for key, tensor in source.items():
        match = re.match(r"^model\.(\d+)\.(.*)$", key)
        if not match:
            continue
        old_index = int(match.group(1))
        if variant == "splf":
            new_index = old_index if old_index < BASE_DECODER_INDEX else old_index + 1
        else:
            new_index = (old_index if old_index < BACKBONE_COUNT else old_index + 1
                         if old_index < BASE_DECODER_INDEX else old_index + 2)
        new_key = f"model.{new_index}.{match.group(2)}"
        if new_key in target and target[new_key].shape == tensor.shape:
            mapped[new_key] = tensor
        else:
            missed.append((key, new_key))
    if missed or not mapped:
        raise RuntimeError(f"Incomplete baseline weight transfer ({len(mapped)} mapped, "
                           f"{len(missed)} missed): {missed[:8]}")
    model.load_state_dict(mapped, strict=False)
    return {"baseline_checkpoint": str(checkpoint.resolve()), "mapped_tensors": len(mapped),
            "source_model_tensors": sum(k.startswith("model.") for k in source),
            "new_branch_tensors": len(target) - len(mapped)}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                          encoding="utf-8")
