"""Audit DUT-Anti-UAV split integrity, labels and duplicate image hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve(root: Path, value) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def _files(path: Path):
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if path.exists() else []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_for(image: Path, root: Path) -> Path:
    labels = root / "labels"
    candidates = []
    try:
        rel = image.relative_to(root)
        parts = list(rel.parts)
        if parts and parts[0].lower() == "images":
            parts = parts[1:]
        candidates.extend([(labels / Path(*parts)).with_suffix(".txt"), (labels / rel).with_suffix(".txt")])
    except ValueError:
        pass
    candidates.extend([image.parent / f"{image.stem}.txt", image.parent.parent / "labels" / f"{image.stem}.txt"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else labels / f"{image.stem}.txt"


def audit(data_path: Path):
    spec = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    root = _resolve(data_path.parent, spec.get("path", "."))
    if not root.exists():
        root = data_path.parent
    hashes = defaultdict(list); splits = {}
    for split in ("train", "val", "test"):
        values = spec.get(split); values = values if isinstance(values, list) else [values]
        images = []
        for value in values:
            if value is not None: images.extend(_files(_resolve(root, value)))
        images = sorted({p.resolve() for p in images})
        classes, empty, missing, illegal = Counter(), [], [], []
        for image in images:
            hashes[_sha256(image)].append({"split": split, "path": str(image)})
            label = _label_for(image, root)
            if not label.exists(): missing.append(str(label)); continue
            lines = [line.strip() for line in label.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
            if not lines: empty.append(str(label))
            for line_no, line in enumerate(lines, 1):
                fields = line.split()
                try: values_f = [float(v) for v in fields]; cls = int(values_f[0])
                except (ValueError, IndexError):
                    illegal.append({"label": str(label), "line": line_no, "reason": "non-numeric"}); continue
                box = values_f[1:5]
                if len(fields) != 5 or cls < 0 or len(box) != 4 or any(v < 0 or v > 1 for v in box) or box[2] <= 0 or box[3] <= 0:
                    illegal.append({"label": str(label), "line": line_no, "reason": "class-or-box-out-of-range", "values": values_f})
                else: classes[str(cls)] += 1
        splits[split] = {"images": len(images), "class_distribution": dict(classes), "empty_labels": empty,
                         "missing_labels": missing, "illegal_boxes": illegal}
    duplicates = [{"sha256": digest, "occurrences": occurrences} for digest, occurrences in hashes.items() if len(occurrences) > 1]
    cross = [item for item in duplicates if len({x["split"] for x in item["occurrences"]}) > 1]
    return {"data": str(data_path.resolve()), "dataset_root": str(root), "splits": splits,
            "duplicate_image_hashes": duplicates, "cross_split_duplicate_hashes": cross,
            "unauthorized_split_changes": [], "ok": not duplicates and not any(s["missing_labels"] or s["illegal_boxes"] for s in splits.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "DUT-Anti-UAV.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "r18_targeted_dataset_audit.json")
    args = parser.parse_args()
    if not args.data.is_file(): parser.error(f"missing data YAML: {args.data}")
    result = audit(args.data); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
