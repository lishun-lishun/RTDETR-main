#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Idempotent Installer for `RTDETR_Neck_Arena` (Exp 20 - Exp 25) into `RTDETR-main`.
Integrates cleanly alongside `hrnet_lite_suite.py` and `sfgd_neck.py`.
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] install_neck_arena.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import re
import shutil
from pathlib import Path

SENTINEL_BEGIN = "# >>> [NECK_ARENA_BEGIN] >>>"
SENTINEL_END = "# <<< [NECK_ARENA_END] <<<"


def _strip(text):
    return re.compile(rf"[ \t]*{re.escape(SENTINEL_BEGIN)}.*?{re.escape(SENTINEL_END)}[ \t]*\n?", re.DOTALL).sub("", text)


def install(repo_path):
    pkg_root = Path(__file__).resolve().parent
    repo_root = Path(repo_path).resolve()
    if not (repo_root / "ultralytics").is_dir():
        print(f"[ERROR] Invalid RTDETR-main repository path: {repo_root}")
        return 1

    shutil.copy2(pkg_root / "neck_arena.py", repo_root / "ultralytics/nn/modules/neck_arena.py")

    cfg_dst = repo_root / "ultralytics/cfg/models/rt-detr"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    for y in sorted((pkg_root / "configs").glob("*.yaml")):
        shutil.copy2(y, cfg_dst / y.name)

    init_file = repo_root / "ultralytics/nn/modules/__init__.py"
    init_txt = _strip(init_file.read_text(encoding="utf-8")).rstrip() + "\n"
    init_block = f"""{SENTINEL_BEGIN}
from .neck_arena import (
    GatherDistributeGate,
    OrthogonalFreqFilter,
    HiLoLinearAIFI,
    StarC3,
    MultiScaleContrastModule,
)
{SENTINEL_END}
"""
    init_file.write_text(init_txt + init_block, encoding="utf-8")

    tasks_file = repo_root / "ultralytics/nn/tasks.py"
    tasks_txt = _strip(tasks_file.read_text(encoding="utf-8")).lstrip()

    top_import = f"""{SENTINEL_BEGIN}
from ultralytics.nn.modules.neck_arena import (
    GatherDistributeGate,
    OrthogonalFreqFilter,
    HiLoLinearAIFI,
    StarC3,
    MultiScaleContrastModule,
)
{SENTINEL_END}
"""
    parse_snippet = f"""        {SENTINEL_BEGIN}
        elif m is GatherDistributeGate:
            c1 = [ch[x] for x in f]
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2]
        elif m in (OrthogonalFreqFilter, MultiScaleContrastModule):
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2]
        elif m is HiLoLinearAIFI:
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
        elif m is StarC3:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, n, *args[1:]]
            n = 1
        {SENTINEL_END}"""

    anchor_candidates = [
        "        # >>> [SFGD_NECK_BEGIN] >>>",
        "        # >>> [HRNET_LITE_SUITE_BEGIN] >>>",
        "        elif m is RTDETRDecoder:",
    ]
    inserted = False
    for anchor in anchor_candidates:
        if anchor in tasks_txt:
            tasks_txt = tasks_txt.replace(anchor, parse_snippet + "\n" + anchor, 1)
            inserted = True
            break
    if not inserted:
        print("[WARN] Could not locate parse_model anchor automatically; please register Neck Arena modules in tasks.py.")

    tasks_file.write_text(top_import + tasks_txt, encoding="utf-8")
    print(f"[SUCCESS] Installed RTDETR_Neck_Arena (Exp20-Exp25) into {repo_root}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Install RTDETR_Neck_Arena (Exp20-Exp25) into RTDETR-main")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main")
    args = parser.parse_args()
    sys.exit(install(args.repo))


if __name__ == "__main__":
    main()
