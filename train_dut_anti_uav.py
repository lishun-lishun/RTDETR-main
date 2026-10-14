"""Train or test the RT-DETR-R18 baseline and all author-released anti-detr configs."""

import argparse
import csv
import warnings
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
ANTI_MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "anti-detr"
DATA_CONFIG = ROOT / "dataset" / "DUT-Anti-UAV.yaml"
PROFILE_CONFIG = ROOT / "ultralytics" / "cfg" / "anti_detr_default.yaml"

# Keep the baseline separate from anti-r18 even though their released topologies
# are identical. This makes the baseline/control experiment explicit in reports.
MODEL_CONFIGS = {
    "original-r18": ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr-r18.yaml",
    "anti-r18": ANTI_MODEL_DIR / "rtdetr-r18.yaml",
    "anti-wtconv": ANTI_MODEL_DIR / "rtdetr-WTConv.yaml",
    "anti-hifi": ANTI_MODEL_DIR / "rtdetr-HIFI.yaml",
    "anti-mrfpn": ANTI_MODEL_DIR / "rtdetr-MRFPN.yaml",
    "anti-p2": ANTI_MODEL_DIR / "rtdetr-p2.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("list", "train", "test"), default="list")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all", *MODEL_CONFIGS),
        default=("all",),
        help="Models to run; 'all' uses the baseline followed by every anti-detr config.",
    )
    parser.add_argument("--device", default=None, help="For example: 0, 0,1, or cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the paper/default profile.")
    parser.add_argument("--batch", type=int, default=None, help="Override the paper/default profile.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override the paper/default profile.")
    parser.add_argument("--workers", type=int, default=None, help="Override the paper/default profile.")
    parser.add_argument("--lr0", type=float, default=None, help="Override the initial learning rate.")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=None,
        help="Override warmup iterations (stored as warmup_epochs by this RTDETR-main fork).",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional checkpoint/pretrained weights; valid only when one model is selected.",
    )
    parser.add_argument(
        "--train-project",
        type=Path,
        default=ROOT / "runs" / "anti-detr" / "train",
    )
    parser.add_argument(
        "--test-project",
        type=Path,
        default=ROOT / "runs" / "anti-detr" / "test",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without loading PyTorch.")
    return parser.parse_args()


def selected_models(names):
    if "all" in names:
        return MODEL_CONFIGS
    return {name: MODEL_CONFIGS[name] for name in names}


def load_experiment_profile(args):
    profile = yaml.safe_load(PROFILE_CONFIG.read_text(encoding="utf-8"))
    for key in ("epochs", "batch", "imgsz", "workers", "lr0"):
        value = getattr(args, key)
        if value is not None:
            profile[key] = value
    if args.warmup_iters is not None:
        profile["warmup_epochs"] = args.warmup_iters
    if args.device is not None:
        profile["device"] = args.device
    return profile


def device_count(device):
    """Return the requested CUDA process count without importing PyTorch."""
    if device is None or not isinstance(device, str):
        return 1
    devices = [item.strip() for item in device.split(",") if item.strip()]
    return len(devices) if all(item.isdigit() for item in devices) else 1


def validate_profile(mode, profile):
    """Fail early when a DDP global batch cannot be divided across GPUs."""
    if mode != "train":
        return
    world_size = device_count(profile.get("device"))
    batch = profile["batch"]
    if world_size > 1 and batch % world_size:
        raise SystemExit(
            f"DDP requires batch to be divisible by GPU count: batch={batch}, GPUs={world_size}. "
            f"Use --batch {world_size * max(round(batch / world_size), 1)} or another multiple of {world_size}."
        )


def find_best_weights(project, model_name):
    candidates = list(project.glob(f"{model_name}*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No best.pt found for {model_name!r} below {project}. "
            "Train it first or pass --weights for a single model."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def print_plan(mode, models, profile, args):
    print(f"mode={mode}")
    print(f"data={DATA_CONFIG}")
    dataset_root = yaml.safe_load(DATA_CONFIG.read_text(encoding="utf-8"))["path"]
    print(f"dataset_root={dataset_root}")
    print(f"experiment_profile={PROFILE_CONFIG}")
    print("profile_provenance=RTDETR-main defaults; not a complete paper-released training config")
    print("settings=" + ", ".join(f"{key}={value}" for key, value in profile.items()))
    world_size = device_count(profile.get("device"))
    if world_size > 1:
        print(
            f"ddp_world_size={world_size}, global_batch={profile['batch']}, "
            f"per_gpu_batch={profile['batch'] // world_size}"
        )
    for name, config in models.items():
        if mode == "test":
            weights = args.weights or "<auto: latest best.pt>"
            print(f"{name}: config={config}, weights={weights}, split={args.split}")
        else:
            print(f"{name}: config={config}")


def train(models, profile, args):
    from ultralytics import RTDETR

    for name, config in models.items():
        print(f"\n{'=' * 24} TRAIN {name} {'=' * 24}")
        model = RTDETR(str(config))
        if args.weights is not None:
            model.load(str(args.weights))
        model.train(
            data=str(DATA_CONFIG),
            project=str(args.train_project),
            name=name,
            **profile,
        )


def test(models, profile, args):
    from ultralytics import RTDETR

    rows = []
    for name in models:
        weights = args.weights or find_best_weights(args.train_project, name)
        print(f"\n{'=' * 24} TEST {name} {'=' * 24}")
        model = RTDETR(str(weights))
        metrics = model.val(
            data=str(DATA_CONFIG),
            split=args.split,
            batch=profile["batch"],
            imgsz=profile["imgsz"],
            workers=profile["workers"],
            max_det=profile["max_det"],
            device=profile.get("device"),
            project=str(args.test_project),
            name=name,
        )
        row = {"model": name, "weights": str(weights), "split": args.split}
        row.update(getattr(metrics, "results_dict", {}))
        speed = getattr(metrics, "speed", {}) or {}
        row.update({f"speed/{key}": value for key, value in speed.items()})
        rows.append(row)

    args.test_project.mkdir(parents=True, exist_ok=True)
    summary = args.test_project / f"{args.split}_summary.csv"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with summary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary saved to {summary}")


def main():
    args = parse_args()
    models = selected_models(args.models)
    if args.weights is not None and len(models) != 1:
        raise SystemExit("--weights can only be used with one explicitly selected model")
    if "anti-mrfpn" in models:
        warnings.warn(
            "The author repository does not contain the MCAM/AGCAU implementation. "
            "anti-mrfpn currently uses the documented SBA interface-compatible fallback "
            "and is not an exact reproduction of the paper's internal AGCAU algorithm.",
            stacklevel=1,
        )

    profile = load_experiment_profile(args)
    validate_profile(args.mode, profile)
    if args.mode == "list" or args.dry_run:
        print_plan(args.mode, models, profile, args)
        return
    if args.mode == "train":
        train(models, profile, args)
    else:
        test(models, profile, args)


if __name__ == "__main__":
    main()
