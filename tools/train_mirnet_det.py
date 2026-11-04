"""Train one MIRNet-Det variant with the repository Ultralytics trainer."""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

from mirnet_det_common import CONFIGS, ROOT, RUNS, config_path, local_package, preflight, protocol_settings, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=RUNS)
    parser.add_argument("--baseline-args", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    config = config_path(args.model)
    output = args.output_root / args.model / f"seed{args.seed}"
    if args.resume and args.resume.resolve() != (output / "weights" / "last.pt").resolve():
        parser.error("resume must point to this run's weights/last.pt")
    settings = protocol_settings(args, parser)
    print(json.dumps({"config": str(config), "output": str(output), "settings": settings}, indent=2), flush=True)
    if args.dry_run:
        return
    if output.exists() and not args.resume:
        parser.error(f"experiment output already exists: {output}")
    if args.resume and not args.resume.is_file():
        parser.error("resume checkpoint is missing")
    local_package()
    import torch
    from ultralytics import RTDETR
    from ultralytics.nn.tasks import RTDETRDetectionModel

    if args.device != "cpu" and (not torch.cuda.is_available() or not args.device.isdecimal() or int(args.device) >= torch.cuda.device_count()):
        parser.error("requested CUDA device is unavailable")
    if not args.resume:
        if not args.skip_tests:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_mirnet_det.py", "-v"], cwd=ROOT, check=True)
        network = RTDETRDetectionModel(str(config), nc=1, verbose=False).eval()
        shapes = preflight(network)
        expected = {"p3": (1, 128, 16, 16), "p4": (1, 256, 8, 8), "p5": (1, 512, 4, 4),
                    "decoder": [(1, 256, 16, 16), (1, 256, 8, 8), (1, 256, 4, 4)],
                    "decoder_from": [15, 18, 21]}
        if shapes != expected:
            raise RuntimeError(f"MIRNet backbone/head preflight mismatch: {shapes} != {expected}")
        summary = {"parameters": sum(p.numel() for p in network.parameters()),
                   "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
                   "layers": len(network.model), "shapes_at_128": shapes}
        del network
        model = RTDETR(str(config))
    else:
        model = RTDETR(str(args.resume))
        settings["resume"] = str(args.resume)
        summary = {"resumed_from": str(args.resume)}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "training_config.json", {
        "model": args.model, "yaml": str(config.resolve()), "data": str((ROOT / "dataset" / "DUT-Anti-UAV.yaml").resolve()),
        "settings": settings, "started_at": dt.datetime.now().astimezone().isoformat()})
    write_json(output / "model_summary.json", summary)
    log_file = output / "execution_log.txt"
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"[{dt.datetime.now().astimezone().isoformat()}] training started\n")
    try:
        model.train(**settings)
    except Exception as exc:
        with log_file.open("a", encoding="utf-8") as file:
            file.write(f"[{dt.datetime.now().astimezone().isoformat()}] failed: {type(exc).__name__}: {exc}\n")
        raise
    results_file = output / "results.csv"
    if not results_file.is_file():
        raise RuntimeError(f"trainer did not save per-epoch results: {results_file}")
    with results_file.open(newline="", encoding="utf-8") as file:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(file)]
    if not rows or not any(key.startswith("lr/pg") for key in rows[0]):
        raise RuntimeError("results.csv has no actual learning-rate columns")
    write_json(output / "epoch_history.json", rows)
    with log_file.open("a", encoding="utf-8") as file:
        file.write(f"[{dt.datetime.now().astimezone().isoformat()}] training complete\n")


if __name__ == "__main__":
    main()
