#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read-Only RAM-Disk Accelerator for NPS-Drones (`/dev/shm/NPS-Drones-720P`):
  1. Checks if `/dev/shm/NPS-Drones-720P` already exists with valid `images/{train,val,test}`
     and independent `labels/{train,val,test}`.
  2. If missing/incomplete, reads strictly READ-ONLY from the disk dataset
     (`/home/aigc/disk1/LiShun/muchuan/NPS-Drones/NPS-Drones` or `NPS-Drones_yolo_std`),
     compresses `images/train` and `images/val` to 1280x720 (JPEG q=92, `cv2.INTER_AREA`),
     symlinks `images/test` to full-resolution original test images (lossless evaluation!),
     and copies independent `labels/{train,val,test}`.
  3. NEVER modifies or deletes any file in the original disk dataset!
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] prepare_nps_720p_shm.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_files(d: Path, exts=None) -> int:
    if not d.exists():
        return 0
    if exts is None:
        return sum(1 for p in d.iterdir() if p.is_file() or p.is_symlink())
    return sum(1 for p in d.iterdir() if (p.is_file() or p.is_symlink()) and p.suffix.lower() in exts)


def check_ready(dst: Path) -> bool:
    for sp in ("train", "val", "test"):
        if count_files(dst / "images" / sp, IMG_EXTS) == 0:
            return False
        if not (dst / "labels" / sp).is_dir() or count_files(dst / "labels" / sp, {".txt"}) == 0:
            return False
    return True


def print_summary(dst: Path) -> None:
    print(f"\n[NPS-Drones-720P Status] {dst}")
    print("| Split | Images | IsSymlink | Labels (.txt) |")
    print("| :---: | :---: | :---: | :---: |")
    for sp in ("train", "val", "test"):
        img_dir = dst / "images" / sp
        lbl_dir = dst / "labels" / sp
        print(
            f"| {sp} | {count_files(img_dir, IMG_EXTS)} | {img_dir.is_symlink()} | "
            f"{count_files(lbl_dir, {'.txt'})} |"
        )


def resolve_source(src_arg: str) -> Path:
    candidates = [
        Path(src_arg),
        Path(src_arg).parent / f"{Path(src_arg).name}_yolo_std",
        Path("/home/aigc/disk1/LiShun/muchuan/NPS-Drones/NPS-Drones"),
        Path("/home/aigc/disk1/LiShun/muchuan/NPS-Drones/NPS-Drones_yolo_std"),
    ]
    for p in candidates:
        if (p / "images" / "train").is_dir() and (p / "labels" / "train").is_dir():
            return p
    raise FileNotFoundError(
        f"Could not find YOLO-formatted NPS-Drones (`images/train` + `labels/train`). Checked: {[str(c) for c in candidates]}"
    )


def main():
    parser = argparse.ArgumentParser(description="Check/Build /dev/shm/NPS-Drones-720P (read-only from disk)")
    parser.add_argument("--src", type=str, default="/home/aigc/disk1/LiShun/muchuan/NPS-Drones/NPS-Drones")
    parser.add_argument("--dst", type=str, default="/dev/shm/NPS-Drones-720P")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8) * 2))
    parser.add_argument("--force", action="store_true", help="Rebuild even if destination appears ready")
    args = parser.parse_args()

    dst = Path(args.dst)
    if not args.force and check_ready(dst):
        print(f"[READY] `{dst}` already contains valid train/val/test images and independent labels.")
        print_summary(dst)
        return

    try:
        import cv2
        use_cv2 = True
    except ImportError:
        from PIL import Image
        use_cv2 = False

    src = resolve_source(args.src)
    print(f"[BUILD] Reading READ-ONLY from `{src}` -> writing 720P RAM-disk copy to `{dst}` (backend={'cv2' if use_cv2 else 'PIL'}) ...")
    t0 = time.perf_counter()

    # 1. Compress train & val images to 1280x720 JPEG q=92 (INTER_AREA)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)] if use_cv2 else None
    for sp in ("train", "val"):
        src_img_dir = src / "images" / sp
        dst_img_dir = dst / "images" / sp
        if dst_img_dir.is_symlink():
            dst_img_dir.unlink()
        dst_img_dir.mkdir(parents=True, exist_ok=True)

        img_files = [p for p in sorted(src_img_dir.iterdir()) if (p.is_file() or p.is_symlink()) and p.suffix.lower() in IMG_EXTS]

        def _convert(p: Path):
            out_p = dst_img_dir / f"{p.stem}.jpg"
            if out_p.is_file() and not args.force:
                return 1
            if use_cv2:
                im = cv2.imread(str(p.resolve()), cv2.IMREAD_COLOR)
                if im is None:
                    return 0
                h, w = im.shape[:2]
                if w != args.width or h != args.height:
                    im = cv2.resize(im, (args.width, args.height), interpolation=cv2.INTER_AREA)
                return 1 if cv2.imwrite(str(out_p), im, encode_params) else 0
            with Image.open(p.resolve()) as im:
                im = im.convert("RGB")
                if im.size != (args.width, args.height):
                    im = im.resize((args.width, args.height), Image.BOX)
                im.save(out_p, format="JPEG", quality=int(args.quality))
            return 1

        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            ok_cnt = sum(ex.map(_convert, img_files))
        print(f"  [OK] images/{sp}: {ok_cnt}/{len(img_files)} compressed to {args.width}x{args.height} (JPEG q={args.quality})")

    # 2. Symlink images/test to full-resolution original test directory
    dst_test = dst / "images" / "test"
    src_test = (src / "images" / "test").resolve()
    if src_test.is_dir():
        if dst_test.is_symlink() or dst_test.is_file():
            dst_test.unlink()
        elif dst_test.is_dir():
            shutil.rmtree(dst_test)
        os.symlink(str(src_test), str(dst_test))
        print(f"  [OK] images/test -> symlinked to full-res `{src_test}`")

    # 3. Copy independent labels/{train,val,test} so Ultralytics .cache stays isolated in RAM disk
    for sp in ("train", "val", "test"):
        src_lbl = src / "labels" / sp
        dst_lbl = dst / "labels" / sp
        if not src_lbl.is_dir():
            continue
        if dst_lbl.is_symlink():
            dst_lbl.unlink()
        dst_lbl.mkdir(parents=True, exist_ok=True)
        lbl_files = [p for p in sorted(src_lbl.iterdir()) if (p.is_file() or p.is_symlink()) and p.suffix.lower() == ".txt"]

        def _copy_lbl(p: Path):
            out_p = dst_lbl / p.name
            if not out_p.is_file() or args.force:
                shutil.copy2(p.resolve(), out_p)
            return 1

        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            cnt = sum(ex.map(_copy_lbl, lbl_files))
        print(f"  [OK] labels/{sp}: {cnt} independent .txt files copied")

    # Remove stale Ultralytics *.cache inside dst if any
    for cache_file in (dst / "labels").glob("*.cache"):
        cache_file.unlink()

    print(f"[DONE] Prepared `{dst}` in {time.perf_counter() - t0:.1f}s (original disk dataset untouched).")
    print_summary(dst)


if __name__ == "__main__":
    main()
