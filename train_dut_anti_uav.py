"""Train one of the integrated anti-detr configurations on DUT-Anti-UAV."""

import argparse
from pathlib import Path

from ultralytics import RTDETR


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "anti-detr"
DATA_CONFIG = ROOT / "dataset" / "DUT-Anti-UAV.yaml"
MODEL_CONFIGS = {
    path.stem: path for path in sorted(MODEL_DIR.glob("*.yaml"))
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        default="rtdetr-HIFI",
        help="Integrated anti-detr model configuration.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="For example: 0, 0,1, or cpu.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional .pt weights to load.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "train")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model = RTDETR(str(MODEL_CONFIGS[args.model]))
    if args.weights:
        model.load(str(args.weights))

    train_args = {
        "data": str(DATA_CONFIG),
        "cache": False,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "patience": 0,
        "project": str(args.project),
        "name": args.name or args.model,
    }
    if args.device is not None:
        train_args["device"] = args.device
    model.train(**train_args)


if __name__ == "__main__":
    main()
