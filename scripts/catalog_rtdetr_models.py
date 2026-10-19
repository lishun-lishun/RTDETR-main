"""Classify the local RT-DETR YAML catalog by changed model section.

The YAML key ``head`` contains an encoder, a feature-fusion neck, and a decoder;
it must not be treated as synonymous with the neck. This is a static inventory,
not a guarantee that optional custom operators can be built on the server.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
DEFAULT_CSV = ROOT / "RTDETR_DUT_MODEL_CATALOG.csv"


def reference_name(filename):
    for family in ("r50", "r34", "r101", "l", "x"):
        if filename == f"rtdetr-{family}.yaml" or filename.startswith(f"rtdetr-{family}-"):
            return f"rtdetr-{family}.yaml"
    return "rtdetr-r18.yaml"


def changed_indices(current, reference):
    return ";".join(
        str(i) for i in range(max(len(current), len(reference)))
        if i >= len(current) or i >= len(reference) or current[i] != reference[i]
    )


def describe(filename, data, reference_filename, reference):
    backbone = data["backbone"]
    head = data["head"]
    ref_backbone = reference["backbone"]
    ref_head = reference["head"]
    backbone_changed = backbone != ref_backbone
    # The first head layer is an input projection; the second is normally AIFI.
    encoder_changed = len(head) > 1 and len(ref_head) > 1 and head[1] != ref_head[1]
    neck_changed = head[:1] != ref_head[:1] or head[2:-1] != ref_head[2:-1]
    decoder_changed = head[-1:] != ref_head[-1:]
    changed_parts = [
        name for name, changed in (
            ("backbone", backbone_changed),
            ("encoder", encoder_changed),
            ("neck", neck_changed),
            ("decoder", decoder_changed),
        ) if changed
    ]
    return {
        "model_yaml": filename,
        "reference_yaml": reference_filename,
        "original_nc": data.get("nc"),
        "dut_nc_at_training": 1,
        "classification": "+".join(changed_parts) if changed_parts else "reference",
        "backbone_changed": int(backbone_changed),
        "encoder_changed": int(encoder_changed),
        "neck_changed": int(neck_changed),
        "decoder_changed": int(decoder_changed),
        "backbone_layer_indices": changed_indices(backbone, ref_backbone),
        "head_layer_indices": changed_indices(head, ref_head),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    configs = {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in MODEL_DIR.glob("*.yaml")
    }
    rows = []
    for filename, data in sorted(configs.items()):
        reference_filename = reference_name(filename)
        reference = configs[reference_filename]
        if not isinstance(data, dict) or not isinstance(data.get("backbone"), list) or not isinstance(data.get("head"), list):
            raise ValueError(f"Invalid RT-DETR model structure: {filename}")
        rows.append(describe(filename, data, reference_filename, reference))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["classification"] for row in rows)
    print(f"Cataloged {len(rows)} YAML files -> {args.output}")
    print("All use nc=1 when trained with dataset/DUT-Anti-UAV.yaml.")
    print("Classification is structural only; optional modules are not build-tested.")
    for category, count in counts.most_common():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
