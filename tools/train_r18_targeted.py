"""Train one independent DUT-Anti-UAV targeted R09--R16 experiment.

The command is dry-run by default.  Pass ``--execute`` to start training.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from r18_targeted_common import (CONFIGS, DATA, ROOT, config_path, git_state,
                                 local_package, preflight, protocol_settings, validate_yaml, write_json)


def _local_cuda_device(value: str) -> str:
    """Map a physical GPU id to its logical index under CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or value == "cpu" or not value.isdecimal():
        return value
    ids = [item.strip() for item in visible.split(",") if item.strip()]
    if not ids:
        return value
    # With one visible GPU, CUDA always exposes it as cuda:0.  Accepting a
    # physical id or a stale logical id here avoids the common `VISIBLE=3,
    # --device 1` failure while preserving the selected physical card.
    if len(ids) == 1:
        return "0"
    if value in ids:
        return str(ids.index(value))
    if int(value) < len(ids):
        return value
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CONFIGS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "r18_targeted")
    parser.add_argument("--baseline-args", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the payload without training (default)")
    parser.add_argument("--execute", action="store_true", help="actually launch training (default is dry-run)")
    args = parser.parse_args()
    requested_device = args.device
    args.device = _local_cuda_device(args.device)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.epochs is not None and args.epochs < 1:
        parser.error("epochs must be positive")
    config = config_path(args.model)
    if not config.is_file():
        parser.error(f"missing model YAML: {config}")
    try:
        contract = validate_yaml(args.model)
    except Exception as exc:
        parser.error(str(exc))
    output = args.output_root / args.model / f"seed{args.seed}"
    if args.resume and args.resume.resolve() != (output / "weights" / "last.pt").resolve():
        parser.error("resume must point to this run's weights/last.pt")
    settings, protocol = protocol_settings(args, parser)
    payload = {**contract, "model": args.model, "config": str(config), "output": str(output),
               "requested_device": requested_device, "settings": settings,
               "protocol": protocol, "git": git_state()}
    print(json.dumps(payload, indent=2), flush=True)
    if not args.execute:
        return
    if output.exists() and not args.resume:
        parser.error(f"experiment output already exists: {output}")
    if args.resume and not args.resume.is_file():
        parser.error(f"resume checkpoint is missing: {args.resume}")
    local_package()
    import torch
    from ultralytics import RTDETR
    from ultralytics.nn.tasks import RTDETRDetectionModel
    if args.device != "cpu" and (not torch.cuda.is_available() or not args.device.isdecimal()
                                  or int(args.device) >= torch.cuda.device_count()):
        parser.error("requested CUDA device is unavailable")
    if not args.resume and not args.skip_tests:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                        "-p", "test_r18_targeted_variants.py", "-v"], cwd=ROOT, check=True)
    network = RTDETRDetectionModel(str(config), nc=1, verbose=False).eval() if not args.resume else None
    summary = {}
    if network is not None:
        shapes = preflight(network, args.model)
        summary = {"parameters": sum(p.numel() for p in network.parameters()),
                   "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
                   "layers": len(network.model), "shapes_at_640": shapes}
        del network
    model = RTDETR(str(args.resume) if args.resume else str(config))
    if args.resume:
        settings["resume"] = str(args.resume)
        summary = {"resumed_from": str(args.resume)}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "training_config.json", {
        "model": args.model, "yaml": str(config.resolve()), "data": str(DATA.resolve()),
        "settings": settings, "protocol": protocol, "git": git_state(),
        "started_at": dt.datetime.now().astimezone().isoformat(),
    })
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
