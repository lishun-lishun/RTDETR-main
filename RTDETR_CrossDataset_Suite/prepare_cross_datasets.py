#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-Inspector & Standardizer for Cross-Dataset UAV Benchmarks:
  - Det-Fly    (/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/Det-Fly)
  - NPS-Drones (/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/NPS-Drones)
  - TIB-Net    (/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav/TIB-Net)

Features:
  1. Never modifies or deletes raw dataset files; creates clean symlinked YOLO directories
     under `<dataset_root>_yolo_std/` (single class `0: UAV`) when standardization is needed.
  2. Supports:
     - Existing Ultralytics `images/{train,val,test}` & Roboflow `{train,val,test}/images` layouts
     - Pascal VOC `.xml`, YOLO `.txt` (normalized or pixel coords), COCO `.json`
     - NPS-Drones per-clip frame-box `.txt` (`time_layer: ...` / `frame num_objs x1 y1 x2 y2 ...`)
       on extracted frames OR directly from raw `.mov`/`.mp4`/`.avi` videos via OpenCV!
  3. Anti-Leakage Splitting: groups images by parent folder / video-clip prefix (preventing
     adjacent video frame leakage) and splits deterministically (`seed=0`, 70% train / 15% val / 15% test).
  4. Generates `dataset/Det-Fly.yaml`, `dataset/NPS-Drones.yaml`, `dataset/TIB-Net.yaml`
     and prints full `(Images, Boxes, Empty_Labels)` statistics table.
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] prepare_cross_datasets.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import json
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mov", ".mp4", ".avi", ".mkv"}

DEFAULT_DATA_ROOT = "/home/sict/Documents/AILAB/Dataset/lishun/rtdetr_uav"
DATASET_NAMES = ["Det-Fly", "NPS-Drones", "TIB-Net"]


def _get_img_size(img_path):
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return float(im.size[0]), float(im.size[1])
    except Exception:
        pass
    try:
        import cv2
        im = cv2.imread(str(img_path))
        if im is not None:
            return float(im.shape[1]), float(im.shape[0])
    except Exception:
        pass
    return 0.0, 0.0


def _clip_box(cx, cy, w, h):
    x1 = max(0.0, min(1.0, cx - w / 2.0))
    y1 = max(0.0, min(1.0, cy - h / 2.0))
    x2 = max(0.0, min(1.0, cx + w / 2.0))
    y2 = max(0.0, min(1.0, cy + h / 2.0))
    nw, nh = x2 - x1, y2 - y1
    if nw <= 1e-6 or nh <= 1e-6:
        return None
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, nw, nh


def _pixel_xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    if img_w <= 0 or img_h <= 0:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return _clip_box(
        ((x1 + x2) / 2.0) / img_w,
        ((y1 + y2) / 2.0) / img_h,
        (x2 - x1) / img_w,
        (y2 - y1) / img_h,
    )


def _parse_voc_xml(xml_path, img_path=None):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    size = root.find("size")
    img_w = float(size.findtext("width", "0")) if size is not None else 0.0
    img_h = float(size.findtext("height", "0")) if size is not None else 0.0
    if (img_w <= 0 or img_h <= 0) and img_path is not None:
        img_w, img_h = _get_img_size(img_path)
    if img_w <= 0 or img_h <= 0:
        return []
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1 = float(bb.findtext("xmin", "0"))
        y1 = float(bb.findtext("ymin", "0"))
        x2 = float(bb.findtext("xmax", "0"))
        y2 = float(bb.findtext("ymax", "0"))
        b = _pixel_xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
        if b is not None:
            boxes.append(b)
    return boxes


def _parse_yolo_txt(txt_path, img_path=None):
    boxes = []
    img_w, img_h = 0.0, 0.0
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = re.split(r"[\s,]+", s)
        if len(parts) == 4:
            try:
                v0, v1, v2, v3 = map(float, parts)
            except ValueError:
                continue
            if max(abs(v0), abs(v1), abs(v2), abs(v3)) > 1.5:
                if img_w <= 0 and img_path is not None:
                    img_w, img_h = _get_img_size(img_path)
                b = _pixel_xyxy_to_yolo(v0, v1, v2, v3, img_w, img_h)
            else:
                b = _clip_box(v0, v1, v2, v3)
            if b is not None:
                boxes.append(b)
            continue
        if len(parts) < 5:
            continue
        try:
            v1, v2, v3, v4 = map(float, parts[1:5])
        except ValueError:
            continue
        if max(abs(v1), abs(v2), abs(v3), abs(v4)) > 1.5:
            if img_w <= 0 and img_path is not None:
                img_w, img_h = _get_img_size(img_path)
            # Treat pixel coords as x1 y1 x2 y2 if v3 > v1 and v4 > v2 else x y w h
            if v3 > v1 and v4 > v2:
                b = _pixel_xyxy_to_yolo(v1, v2, v3, v4, img_w, img_h)
            else:
                b = _pixel_xyxy_to_yolo(v1, v2, v1 + v3, v2 + v4, img_w, img_h)
        else:
            b = _clip_box(v1, v2, v3, v4)
        if b is not None:
            boxes.append(b)
    return boxes


def _parse_nps_clip_gt(gt_txt_path):
    """
    Parses NPS-Drones per-clip annotation files into `{frame_idx: [(x1, y1, x2, y2), ...]}` (pixel coords).
    Supports:
      1. `time_layer: 123, detections: [(x1, y1, x2, y2), ...]`
      2. `123 1 x1 y1 x2 y2` or `123 x1 y1 x2 y2` (space/comma separated)
    """
    frame_boxes = defaultdict(list)
    text = gt_txt_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m_tl = re.search(r"time_layer\s*:\s*(\d+)", s, flags=re.IGNORECASE)
        if m_tl:
            fidx = int(m_tl.group(1))
            tuples = re.findall(r"\(\s*([0-9.\-]+)\s*,\s*([0-9.\-]+)\s*,\s*([0-9.\-]+)\s*,\s*([0-9.\-]+)\s*\)", s)
            for t in tuples:
                x1, y1, x2, y2 = map(float, t)
                frame_boxes[fidx].append((x1, y1, x2, y2))
            continue
        nums = re.findall(r"-?\d+(?:\.\d+)?", s)
        if len(nums) >= 6:
            fidx = int(float(nums[0]))
            n_det = int(float(nums[1]))
            coords = list(map(float, nums[2:]))
            if n_det > 0 and len(coords) >= n_det * 4:
                for k in range(n_det):
                    x1, y1, x2, y2 = coords[4 * k : 4 * k + 4]
                    frame_boxes[fidx].append((x1, y1, x2, y2))
                continue
        if len(nums) == 5:
            fidx = int(float(nums[0]))
            x1, y1, x2, y2 = map(float, nums[1:5])
            frame_boxes[fidx].append((x1, y1, x2, y2))
    return frame_boxes


def _normalize_clip_id(name):
    m = re.search(r"(?:clip|video)[_\-\s]*0*(\d+)", name, flags=re.IGNORECASE)
    if m:
        return f"clip_{int(m.group(1)):03d}"
    return Path(name).stem.lower()


def _extract_group_key(img_path, root):
    """Groups frames by subfolder or video prefix to prevent adjacent-frame leakage."""
    rel = img_path.relative_to(root)
    if len(rel.parts) >= 2:
        parent_key = "/".join(rel.parts[:-1])
        if parent_key.lower() not in ("images", "imgs", "jpegimages", "train", "val", "valid", "test"):
            return parent_key
    m = re.match(r"^([A-Za-z0-9_\-]+?)[_\-]\d+$", img_path.stem)
    return m.group(1) if m else img_path.stem


def _link_or_copy(src, dst):
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _count_yolo_split(img_dir, lbl_dir):
    imgs = sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    n_boxes = 0
    n_empty = 0
    for im in imgs:
        rel = im.relative_to(img_dir)
        lb = (lbl_dir / rel).with_suffix(".txt")
        if not lb.is_file():
            n_empty += 1
            continue
        lines = [ln.strip() for ln in lb.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        if not lines:
            n_empty += 1
        else:
            n_boxes += len(lines)
    return len(imgs), n_boxes, n_empty


def _detect_existing_yolo_splits(root):
    """
    Checks if dataset already follows:
      Layout A: `<base>/images/{train,val,test}` + `<base>/labels/{train,val,test}`
      Layout B: `<base>/{train,val,valid,test}/images` + `<base>/{train,val,valid,test}/labels`
    """
    candidates = [root] + [p for p in sorted(root.iterdir()) if p.is_dir()]
    for base in candidates:
        img_dir = base / "images"
        lbl_dir = base / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            splits = {}
            for sp in ("train", "val", "valid", "test"):
                if (img_dir / sp).is_dir() and (lbl_dir / sp).is_dir():
                    key = "val" if sp == "valid" else sp
                    splits[key] = (img_dir / sp, lbl_dir / sp)
            if "train" in splits and ("val" in splits or "test" in splits):
                return base, splits

        # Layout B (Roboflow style: train/images, valid/images, test/images)
        splits_b = {}
        for sp in ("train", "val", "valid", "test"):
            if (base / sp / "images").is_dir() and (base / sp / "labels").is_dir():
                key = "val" if sp == "valid" else sp
                splits_b[key] = (base / sp / "images", base / sp / "labels")
        if "train" in splits_b and ("val" in splits_b or "test" in splits_b):
            return base, splits_b
    return None


def _resolve_sibling_ann(img_path, raw_dir, suffix, stem_map):
    # 1. Direct sibling file
    direct = img_path.with_suffix(suffix)
    if direct.is_file():
        return direct
    # 2. Replace images/JPEGImages -> labels/annotations/Annotations in path parts
    parts = list(img_path.relative_to(raw_dir).parts)
    for idx, part in enumerate(parts[:-1]):
        if part.lower() in ("images", "imgs", "jpegimages", "frames"):
            for alt in ("labels", "annotations", "Annotations", "Labels", "xml", "xmls", "txt"):
                cand_parts = parts[:idx] + [alt] + parts[idx + 1 :]
                cand = (raw_dir.joinpath(*cand_parts)).with_suffix(suffix)
                if cand.is_file():
                    return cand
    # 3. Unique stem fallback
    return stem_map.get(img_path.stem)


def _extract_nps_video_dataset(name, raw_dir, std_root, seed):
    """Handles raw NPS-Drones datasets stored as `.mov/.mp4` videos + per-clip `_gt.txt` files."""
    videos = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in VID_EXTS)
    gt_files = sorted(p for p in raw_dir.rglob("*.txt") if p.is_file() and p.name.lower() not in ("classes.txt", "readme.txt"))
    if not videos or not gt_files:
        return None

    import cv2

    gt_by_clip = {}
    for gf in gt_files:
        parsed = _parse_nps_clip_gt(gf)
        if parsed:
            gt_by_clip[_normalize_clip_id(gf.name)] = parsed
    if not gt_by_clip:
        return None

    matched_clips = []
    for vid in videos:
        cid = _normalize_clip_id(vid.name)
        if cid in gt_by_clip:
            matched_clips.append((cid, vid, gt_by_clip[cid]))
    if not matched_clips:
        return None

    print(f"[{name}] Detected {len(matched_clips)} raw video clips with per-clip GT annotations; extracting annotated frames...")
    rng = random.Random(seed)
    matched_clips.sort(key=lambda x: x[0])
    rng.shuffle(matched_clips)

    n_clips = len(matched_clips)
    n_tr = max(1, int(round(0.70 * n_clips)))
    n_va = max(1, int(round(0.15 * n_clips))) if n_clips >= 3 else 0
    clip_splits = {
        "train": matched_clips[:n_tr],
        "val": matched_clips[n_tr : n_tr + n_va] if n_va > 0 else matched_clips[n_tr:],
        "test": matched_clips[n_tr + n_va :] if (n_tr + n_va < n_clips) else (matched_clips[n_tr : n_tr + n_va] if n_va > 0 else matched_clips[:1]),
    }

    stats = {}
    for sp, clips in clip_splits.items():
        im_dir = std_root / "images" / sp
        lb_dir = std_root / "labels" / sp
        im_dir.mkdir(parents=True, exist_ok=True)
        lb_dir.mkdir(parents=True, exist_ok=True)
        n_imgs = 0
        n_boxes = 0
        n_empty = 0
        for cid, vid_path, fboxes in clips:
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                continue
            target_frames = sorted(fboxes.keys())
            target_set = set(target_frames)
            max_f = target_frames[-1] if target_frames else -1
            fidx = 0
            while fidx <= max_f:
                ok, frame = cap.read()
                if not ok:
                    break
                if fidx in target_set:
                    h, w = frame.shape[:2]
                    yolo_boxes = []
                    for (x1, y1, x2, y2) in fboxes[fidx]:
                        b = _pixel_xyxy_to_yolo(x1, y1, x2, y2, float(w), float(h))
                        if b is not None:
                            yolo_boxes.append(b)
                    if yolo_boxes:
                        stem = f"{cid}_f{fidx:06d}"
                        dst_img = im_dir / f"{stem}.jpg"
                        if not dst_img.exists():
                            cv2.imwrite(str(dst_img), frame)
                        lbl_lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for (cx, cy, bw, bh) in yolo_boxes]
                        (lb_dir / f"{stem}.txt").write_text("\n".join(lbl_lines), encoding="utf-8")
                        n_imgs += 1
                        n_boxes += len(yolo_boxes)
                fidx += 1
            cap.release()
        stats[sp] = (n_imgs, n_boxes, n_empty)
    return stats


def prepare_one_dataset(name, raw_dir, repo_root, seed=0):
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"[{name}] Directory not found: {raw_dir}")

    yaml_dir = repo_root / "dataset"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = yaml_dir / f"{name}.yaml"

    # Case 1: Already standard Ultralytics or Roboflow YOLO format
    existing = _detect_existing_yolo_splits(raw_dir)
    if existing is not None:
        base_dir, splits = existing
        train_im, train_lb = splits["train"]
        val_im, val_lb = splits.get("val", splits["test"])
        test_im, test_lb = splits.get("test", (val_im, val_lb))

        stats = {}
        for sp_name, (im_d, lb_d) in [("train", (train_im, train_lb)), ("val", (val_im, val_lb)), ("test", (test_im, test_lb))]:
            stats[sp_name] = _count_yolo_split(im_d, lb_d)

        yaml_text = (
            f"path: {base_dir.resolve().as_posix()}\n"
            f"train: {train_im.relative_to(base_dir).as_posix()}\n"
            f"val: {val_im.relative_to(base_dir).as_posix()}\n"
            f"test: {test_im.relative_to(base_dir).as_posix()}\n"
            f"nc: 1\n"
            f"names:\n  0: UAV\n"
        )
        yaml_path.write_text(yaml_text, encoding="utf-8")
        print(f"[{name}] Detected existing YOLO layout at {base_dir} -> wrote {yaml_path}")
        return yaml_path, stats

    # Case 2: Scan images + VOC XML / YOLO TXT / COCO JSON / NPS-Drones per-clip GT TXT
    all_imgs = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    std_root = raw_dir.parent / f"{name}_yolo_std"

    # Case 2a: Video-only dataset (e.g. raw NPS-Drones `.mov` + `_gt.txt`)
    if not all_imgs:
        vid_stats = _extract_nps_video_dataset(name, raw_dir, std_root, seed)
        if vid_stats is not None:
            yaml_text = (
                f"path: {std_root.resolve().as_posix()}\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"test: images/test\n"
                f"nc: 1\n"
                f"names:\n  0: UAV\n"
            )
            yaml_path.write_text(yaml_text, encoding="utf-8")
            print(f"[{name}] Extracted video dataset -> {std_root} | YAML: {yaml_path}")
            return yaml_path, vid_stats
        raise RuntimeError(f"[{name}] No images or supported video+GT pairs found under {raw_dir}")

    xml_map = {p.stem: p for p in raw_dir.rglob("*.xml")}
    all_txts = [p for p in raw_dir.rglob("*.txt") if p.name.lower() not in ("classes.txt", "readme.txt")]
    txt_map = {p.stem: p for p in all_txts}

    # Optional NPS-Drones per-clip GT index (when frames are already extracted on disk)
    nps_clip_gt = {}
    for tp in all_txts:
        if any(tok in tp.name.lower() for tok in ("gt", "clip", "video", "annot")):
            parsed = _parse_nps_clip_gt(tp)
            if parsed:
                nps_clip_gt[_normalize_clip_id(tp.name)] = parsed

    # Optional COCO JSON index
    coco_boxes = defaultdict(list)
    for jp in raw_dir.rglob("*.json"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "images" in data and "annotations" in data:
                id_to_meta = {
                    im["id"]: (Path(im["file_name"]).name, float(im["width"]), float(im["height"]))
                    for im in data["images"]
                }
                for ann in data["annotations"]:
                    if ann["image_id"] not in id_to_meta:
                        continue
                    fname, iw, ih = id_to_meta[ann["image_id"]]
                    x, y, w, h = ann["bbox"]
                    b = _clip_box((x + w / 2.0) / iw, (y + h / 2.0) / ih, w / iw, h / ih)
                    if b is not None:
                        coco_boxes[fname].append(b)
        except Exception:
            continue

    records = []
    for img in all_imgs:
        boxes = []
        if img.name in coco_boxes:
            boxes = coco_boxes[img.name]
        else:
            xml_p = _resolve_sibling_ann(img, raw_dir, ".xml", xml_map)
            txt_p = _resolve_sibling_ann(img, raw_dir, ".txt", txt_map)
            if xml_p is not None:
                boxes = _parse_voc_xml(xml_p, img_path=img)
            elif txt_p is not None:
                boxes = _parse_yolo_txt(txt_p, img_path=img)
            elif nps_clip_gt:
                cid = _normalize_clip_id(str(img.relative_to(raw_dir)))
                if cid in nps_clip_gt:
                    m_num = re.search(r"(\d+)$", img.stem)
                    if m_num:
                        fidx = int(m_num.group(1))
                        raw_b = nps_clip_gt[cid].get(fidx, [])
                        if raw_b:
                            iw, ih = _get_img_size(img)
                            for (x1, y1, x2, y2) in raw_b:
                                b = _pixel_xyxy_to_yolo(x1, y1, x2, y2, iw, ih)
                                if b is not None:
                                    boxes.append(b)
        group = _extract_group_key(img, raw_dir)
        records.append((img, boxes, group))

    # Group-aware deterministic 70/15/15 split
    groups = defaultdict(list)
    for rec in records:
        groups[rec[2]].append(rec)
    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_total = len(records)
    split_records = {"train": [], "val": [], "test": []}
    for idx_g, gk in enumerate(group_keys):
        rem_groups = len(group_keys) - idx_g
        if not split_records["test"] and rem_groups == 1 and len(group_keys) >= 3:
            split_records["test"].extend(groups[gk])
        elif not split_records["val"] and rem_groups == 2 and len(group_keys) >= 3:
            split_records["val"].extend(groups[gk])
        elif len(split_records["train"]) < 0.70 * n_total:
            split_records["train"].extend(groups[gk])
        elif len(split_records["val"]) < 0.15 * n_total:
            split_records["val"].extend(groups[gk])
        else:
            split_records["test"].extend(groups[gk])
    if not split_records["test"]:
        split_records["test"] = split_records["val"]

    stats = {}
    for sp, items in split_records.items():
        im_dir = std_root / "images" / sp
        lb_dir = std_root / "labels" / sp
        im_dir.mkdir(parents=True, exist_ok=True)
        lb_dir.mkdir(parents=True, exist_ok=True)
        n_boxes = 0
        n_empty = 0
        for idx, (img_p, boxes, _) in enumerate(items):
            stem = f"{sp}_{idx:06d}_{img_p.stem}"
            dst_img = im_dir / f"{stem}{img_p.suffix.lower()}"
            _link_or_copy(img_p, dst_img)
            lbl_lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (cx, cy, w, h) in boxes]
            (lb_dir / f"{stem}.txt").write_text("\n".join(lbl_lines), encoding="utf-8")
            n_boxes += len(boxes)
            if not boxes:
                n_empty += 1
        stats[sp] = (len(items), n_boxes, n_empty)

    yaml_text = (
        f"path: {std_root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names:\n  0: UAV\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"[{name}] Standardized -> {std_root} | YAML: {yaml_path}")
    return yaml_path, stats


def main():
    parser = argparse.ArgumentParser(description="Prepare Det-Fly, NPS-Drones, TIB-Net into standard YOLO format")
    parser.add_argument("--repo", type=str, default=".", help="Path to RTDETR-main")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="Parent folder containing Det-Fly, NPS-Drones, TIB-Net")
    parser.add_argument("--datasets", nargs="+", default=DATASET_NAMES, choices=DATASET_NAMES, help="Which datasets to prepare")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    data_root = Path(args.data_root)

    all_rows = []
    zero_box_datasets = []
    for name in args.datasets:
        _, stats = prepare_one_dataset(name, data_root / name, repo_root, seed=args.seed)
        total_boxes = sum(v[1] for v in stats.values())
        if total_boxes == 0:
            zero_box_datasets.append(name)
        for sp in ("train", "val", "test"):
            imgs, boxes, empty = stats.get(sp, (0, 0, 0))
            all_rows.append((name, sp, imgs, boxes, empty))

    print("\n" + "=" * 78)
    print(f"| {'Dataset':<12} | {'Split':<5} | {'Images':>8} | {'Boxes':>9} | {'Empty_Labels':>12} |")
    print("|" + "-" * 14 + "|" + "-" * 7 + "|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 14 + "|")
    for name, sp, imgs, boxes, empty in all_rows:
        print(f"| {name:<12} | {sp:<5} | {imgs:>8d} | {boxes:>9d} | {empty:>12d} |")
    print("=" * 78)

    if zero_box_datasets:
        raise RuntimeError(
            f"[ERROR] Zero bounding boxes parsed for dataset(s): {zero_box_datasets}. "
            f"Please inspect their annotation format!"
        )


if __name__ == "__main__":
    main()
