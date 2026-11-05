"""Aggregate per-backbone evaluation and benchmark JSON files into CSV/Markdown."""

import argparse
import csv
import json
from pathlib import Path

from backbone_common import CONFIGS, RUNS


FIELDS = ["Model", "Params", "GFLOPs", "AP50", "AP75", "AP50:95",
          "Tiny AP", "Small AP", "Medium AP", "Large AP", "FPS", "Latency", "GPU Memory"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-csv", type=Path, default=Path("reports/eight_backbones_summary.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/eight_backbones_summary.md"))
    args = parser.parse_args()
    rows = []
    for name in CONFIGS:
        run = args.root / name / f"seed{args.seed}"
        metrics = read_json(run / f"eval_{args.split}" / "evaluation_metrics.json")
        benchmark = read_json(run / "benchmark.json")
        groups = {}
        group_file = run / f"eval_{args.split}" / "size_group_metrics.csv"
        if group_file.is_file():
            with group_file.open(newline="", encoding="utf-8") as file:
                groups = {item["group"]: item for item in csv.DictReader(file)}
        def value(item, key):
            value = item.get(key)
            return "" if value is None else value
        row = {
            "Model": name,
            "Params": value(benchmark, "parameters"),
            "GFLOPs": value(benchmark, "GFLOPs"),
            "AP50": value(metrics, "mAP50"),
            "AP75": value(metrics, "mAP75"),
            "AP50:95": value(metrics, "mAP50:95"),
            "Tiny AP": value(groups.get("Tiny", {}), "AP50:95"),
            "Small AP": value(groups.get("Small", {}), "AP50:95"),
            "Medium AP": value(groups.get("Medium", {}), "AP50:95"),
            "Large AP": value(groups.get("Large", {}), "AP50:95"),
            "FPS": value(benchmark, "fps_batch1"),
            "Latency": value(benchmark, "mean_latency_ms"),
            "GPU Memory": value(benchmark, "peak_gpu_memory_bytes"),
        }
        rows.append(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    def sort_key(row):
        try:
            return float(row["AP50:95"])
        except (TypeError, ValueError):
            return float("-inf")
    ranked = sorted(rows, key=sort_key, reverse=True)
    lines = [f"# Eight-backbone RT-DETR summary ({args.split}, seed {args.seed})", "",
             "Metrics are read from completed evaluation and benchmark artifacts. Missing values are left blank.", "",
             "| " + " | ".join(FIELDS) + " |", "|" + "|".join("---" for _ in FIELDS) + "|"]
    for row in ranked:
        lines.append("| " + " | ".join(str(row[field]) for field in FIELDS) + " |")
    lines.extend(["", "Ranking is ordered by AP50:95 when that metric is available; AP75, size groups, FPS and memory must be considered together."])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(args.output_csv), "markdown": str(args.output_md), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
