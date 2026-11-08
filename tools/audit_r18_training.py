"""Audit actual learning-rate, warm-up and effective-batch records of a run."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _num(value, default=None):
    try: return float(value)
    except (TypeError, ValueError): return default


def audit(run_dir: Path):
    config = {}
    for name in ("training_config.json", "args.yaml", "args.yml"):
        path = run_dir / name
        if path.is_file():
            if path.suffix == ".json": config = json.loads(path.read_text(encoding="utf-8"))
            else:
                import yaml
                config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            break
    settings = config.get("settings", config)
    rows = []
    results = run_dir / "results.csv"
    if results.is_file():
        with results.open(newline="", encoding="utf-8") as stream:
            rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(stream)]
    lr_columns = sorted(k for k in (rows[0].keys() if rows else []) if k.startswith("lr/"))
    history = []
    for index, row in enumerate(rows, 1):
        values = {key: _num(row.get(key)) for key in lr_columns}; values["epoch"] = _num(row.get("epoch"), index)
        observed = [v for k, v in values.items() if k.startswith("lr/") and v is not None]
        values["lr_min"] = min(observed) if observed else None; values["lr_max"] = max(observed) if observed else None
        history.append(values)
    micro = _num(settings.get("micro_batch", settings.get("batch", 8)), 8)
    accumulation = _num(settings.get("gradient_accumulation", 1), 1)
    warmup_key = "warmup_epochs" if "warmup_epochs" in settings else "warmup_steps"
    return {"run_dir": str(run_dir.resolve()), "config": config,
            "schedule": {"optimizer": settings.get("optimizer"), "lr0": settings.get("lr0"), "lrf": settings.get("lrf"),
                         "epochs": settings.get("epochs"), "warmup_value": settings.get(warmup_key),
                         "warmup_unit": "epochs" if warmup_key == "warmup_epochs" else "steps",
                         "micro_batch": micro, "gradient_accumulation": accumulation,
                         "effective_batch": micro * accumulation, "lr_columns": lr_columns,
                         "epoch_wise_lr": history,
                         "observed_first_lr": history[0]["lr_min"] if history else None,
                         "observed_last_lr": history[-1]["lr_min"] if history else None}}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); result = audit(args.run_dir); output = args.output or args.run_dir / "training_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    history = result["schedule"]["epoch_wise_lr"]
    if history:
        with output.with_name(output.stem + "_lr.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
