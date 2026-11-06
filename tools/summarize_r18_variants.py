"""Aggregate R18 variant training, evaluation and benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from r18_variants_common import CONFIGS, REFERENCE_CONFIGS, RUNS, VARIANT_DESCRIPTIONS, write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    rows = []
    for name in CONFIGS:
        run = args.runs / name / "seed0"
        summary = read_json(run / "model_summary.json")
        evaluation = read_json(run / "eval" / "evaluation_metrics.json")
        benchmark = read_json(run / "benchmark.json")
        results = run / "results.csv"
        status = "completed" if results.is_file() else "not_started"
        row = {"Model": name, "Seed": 0, "Description": VARIANT_DESCRIPTIONS[name],
               "Parameters": summary.get("parameters"), "GFLOPs": benchmark.get("GFLOPs"),
               "FPS": benchmark.get("FPS"), "mAP50": evaluation.get("mAP50"),
               "mAP75": evaluation.get("mAP75"), "mAP50:95": evaluation.get("mAP50:95"),
               "Tiny_AP": None, "Small_AP": None, "Medium_AP": None, "Large_AP": None,
               "Training_Status": status}
        size_file = run / "eval" / "size_group_metrics.csv"
        if size_file.is_file():
            with size_file.open(newline="", encoding="utf-8") as file:
                for item in csv.DictReader(file):
                    group = item.get("group", "")
                    if group in ("Tiny", "Small", "Medium", "Large"):
                        row[f"{group}_AP"] = item.get("AP50:95")
        rows.append(row)
    for name, filename in REFERENCE_CONFIGS.items():
        rows.append({"Model": name, "Seed": "reference", "Description": f"reference config: {filename}",
                     "Parameters": None, "GFLOPs": None, "FPS": None, "mAP50": None,
                     "mAP75": None, "mAP50:95": None, "Tiny_AP": None, "Small_AP": None,
                     "Medium_AP": None, "Large_AP": None, "Training_Status": "reference_only"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "r18_eight_variants_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.output_dir / "r18_eight_variants_summary.md"
    lines = ["# RT-DETR-R18 eight-variant summary", "", "| Model | mAP50:95 | Small AP | FPS | Params | Status |",
             "|---|---:|---:|---:|---:|---|"]
    for row in sorted(rows, key=lambda item: float(item["mAP50:95"] or -1), reverse=True):
        lines.append(f"| {row['Model']} | {row['mAP50:95'] or '-'} | {row['Small_AP'] or '-'} | "
                     f"{row['FPS'] or '-'} | {row['Parameters'] or '-'} | {row['Training_Status']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path), "rows": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
