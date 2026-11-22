#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Idempotent Installer for `RTDETR_SFGD_Neck` (Exp 15 - Exp 18) into `RTDETR-main`.
Integrates cleanly alongside `hrnet_lite_suite.py` without touching original baselines.
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] install_sfgd_neck.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import re
import shutil
from pathlib import Path

SENTINEL_BEGIN = "# >>> [SFGD_NECK_BEGIN] >>>"
SENTINEL_END = "# <<< [SFGD_NECK_END] <<<"


def _strip(text):
    return re.compile(rf"[ \t]*{re.escape(SENTINEL_BEGIN)}.*?{re.escape(SENTINEL_END)}[ \t]*\n?", re.DOTALL).sub("", text)


def install(repo_path):
    pkg_root = Path(__file__).resolve().parent
    repo_root = Path(repo_path).resolve()
    if not (repo_root / "ultralytics").is_dir():
        print(f"[ERROR] Invalid RTDETR-main repository path: {repo_root}")
        return 1

    shutil.copy2(pkg_root / "sfgd_neck.py", repo_root / "ultralytics/nn/modules/sfgd_neck.py")

    cfg_dst = repo_root / "ultralytics/cfg/models/rt-detr"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    for y in sorted((pkg_root / "configs").glob("*.yaml")):
        shutil.copy2(y, cfg_dst / y.name)

    init_file = repo_root / "ultralytics/nn/modules/__init__.py"
    init_txt = _strip(init_file.read_text(encoding="utf-8")).rstrip() + "\n"
    init_block = f"""{SENTINEL_BEGIN}
from .sfgd_neck import FreqDecoupledBridge, SoftplusZeroConcatFusion, DualFreqAIFI
{SENTINEL_END}
"""
    init_file.write_text(init_txt + init_block, encoding="utf-8")

    tasks_file = repo_root / "ultralytics/nn/tasks.py"
    tasks_txt = _strip(tasks_file.read_text(encoding="utf-8")).lstrip()

    top_import = f"""{SENTINEL_BEGIN}
from ultralytics.nn.modules.sfgd_neck import FreqDecoupledBridge, SoftplusZeroConcatFusion, DualFreqAIFI
{SENTINEL_END}
"""
    parse_snippet = f"""        {SENTINEL_BEGIN}
        elif m in (FreqDecoupledBridge, SoftplusZeroConcatFusion):
            c1 = [ch[x] for x in f]
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1[0], c1[1], c2]
        elif m is DualFreqAIFI:
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
        {SENTINEL_END}"""

    anchor_candidates = [
        "        # >>> [HRNET_LITE_SUITE_BEGIN] >>>",
        "        elif m in (RTHRNetW18, RTHRNetW18Fast, RTHRNetW18Balanced):",
        "        elif m is RTDETRDecoder:",
    ]
    inserted = False
    for anchor in anchor_candidates:
        if anchor in tasks_txt:
            tasks_txt = tasks_txt.replace(anchor, parse_snippet + "\n" + anchor, 1)
            inserted = True
            break
    if not inserted:
        print("[WARN] Could not locate parse_model anchor automatically; please register SFGD-Neck modules in tasks.py.")

    tasks_file.write_text(top_import + tasks_txt, encoding="utf-8")
    print(f"[SUCCESS] Installed RTDETR_SFGD_Neck (Exp15-Exp19) into {repo_root}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Install RTDETR_SFGD_Neck (Exp15-Exp19) into RTDETR-main")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main")
    args = parser.parse_args()
    sys.exit(install(args.repo))


if __name__ == "__main__":
    main()
