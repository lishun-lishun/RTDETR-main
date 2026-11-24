#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Idempotent Installer for `RTDETR_Exp6_Speed` (Exp 26 - Exp 29) into `RTDETR-main`.
Registers `HRNetStarFast` alongside existing `hrnet_lite_suite.py` without touching Exp1-Exp25.
"""

import sys
if sys.version_info < (3, 8):
    sys.exit(
        f"[ERROR] install_exp6_speed.py requires Python 3.8+ (detected: {sys.version}).\n"
        f"Please activate your Conda environment first:\n"
        f"    conda activate ultralytics_rtdetr\n"
    )

import argparse
import re
import shutil
from pathlib import Path

SENTINEL_BEGIN = "# >>> [EXP6_SPEED_BEGIN] >>>"
SENTINEL_END = "# <<< [EXP6_SPEED_END] <<<"


def _strip(text):
    return re.compile(rf"[ \t]*{re.escape(SENTINEL_BEGIN)}.*?{re.escape(SENTINEL_END)}[ \t]*\n?", re.DOTALL).sub("", text)


def install(repo_path):
    pkg_root = Path(__file__).resolve().parent
    repo_root = Path(repo_path).resolve()
    if not (repo_root / "ultralytics").is_dir():
        print(f"[ERROR] Invalid RTDETR-main repository path: {repo_root}")
        return 1

    shutil.copy2(pkg_root / "exp6_speed.py", repo_root / "ultralytics/nn/modules/exp6_speed.py")

    cfg_dst = repo_root / "ultralytics/cfg/models/rt-detr"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    for y in sorted((pkg_root / "configs").glob("*.yaml")):
        shutil.copy2(y, cfg_dst / y.name)

    init_file = repo_root / "ultralytics/nn/modules/__init__.py"
    init_txt = _strip(init_file.read_text(encoding="utf-8")).rstrip() + "\n"
    init_block = f"""{SENTINEL_BEGIN}
from .exp6_speed import HRNetStarFast
{SENTINEL_END}
"""
    init_file.write_text(init_txt + init_block, encoding="utf-8")

    tasks_file = repo_root / "ultralytics/nn/tasks.py"
    tasks_txt = _strip(tasks_file.read_text(encoding="utf-8")).lstrip()

    top_import = f"""{SENTINEL_BEGIN}
from ultralytics.nn.modules.exp6_speed import HRNetStarFast
{SENTINEL_END}
"""
    parse_snippet = f"""        {SENTINEL_BEGIN}
        elif m is HRNetStarFast:
            m_ = m(*args)
            c2 = list(m_.out_channels)
            t = str(m)[8:-2].replace("__main__.", "")
            m_.np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type = i, f, t
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
            continue
        {SENTINEL_END}"""

    anchor_candidates = [
        "        # >>> [NECK_ARENA_BEGIN] >>>",
        "        # >>> [SFGD_NECK_BEGIN] >>>",
        "        # >>> [HRNET_LITE_SUITE_BEGIN] >>>",
        "        elif m is RTDETRDecoder:",
    ]
    for anchor in anchor_candidates:
        if anchor in tasks_txt:
            tasks_txt = tasks_txt.replace(anchor, parse_snippet + "\n" + anchor, 1)
            break

    tasks_file.write_text(top_import + tasks_txt, encoding="utf-8")
    print(f"[SUCCESS] Installed RTDETR_Exp6_Speed (Exp26-Exp29) into {repo_root}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Install RTDETR_Exp6_Speed (Exp26-Exp29) into RTDETR-main")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main")
    args = parser.parse_args()
    sys.exit(install(args.repo))


if __name__ == "__main__":
    main()
