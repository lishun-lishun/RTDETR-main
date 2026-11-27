#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Idempotent Installer for `RTDETR_Exp33_AntiCollapse` (Exp33 - Exp36) into `RTDETR-main`.
Registers `FCGHaarDownsample` and `AntiCollapseRTDETRDecoder` cleanly without touching Exp1-Exp32.
"""

from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path

SENTINEL_BEGIN = "# >>> [EXP33_ANTICOLLAPSE_BEGIN] >>>"
SENTINEL_END = "# <<< [EXP33_ANTICOLLAPSE_END] <<<"


def _strip(text: str) -> str:
    return re.compile(rf"[ \t]*{re.escape(SENTINEL_BEGIN)}.*?{re.escape(SENTINEL_END)}[ \t]*\n?", re.DOTALL).sub("", text)


def install(repo_path: str) -> int:
    pkg_root = Path(__file__).resolve().parent
    repo_root = Path(repo_path).resolve()
    if not (repo_root / "ultralytics").is_dir():
        print(f"[ERROR] Invalid RTDETR-main repository path: {repo_root}")
        return 1

    # 1. Copy module file
    shutil.copy2(pkg_root / "exp33_anticollapse.py", repo_root / "ultralytics/nn/modules/exp33_anticollapse.py")

    # 2. Copy YAML configs
    cfg_dst = repo_root / "ultralytics/cfg/models/rt-detr"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    for y in sorted((pkg_root / "configs").glob("*.yaml")):
        shutil.copy2(y, cfg_dst / y.name)

    # 3. Export in ultralytics/nn/modules/__init__.py
    init_file = repo_root / "ultralytics/nn/modules/__init__.py"
    init_txt = _strip(init_file.read_text(encoding="utf-8")).rstrip() + "\n"
    init_block = f"""{SENTINEL_BEGIN}
from .exp33_anticollapse import FCGHaarDownsample, AntiCollapseMSDeformAttn, AntiCollapseRTDETRDecoder
{SENTINEL_END}
"""
    init_file.write_text(init_txt + init_block, encoding="utf-8")

    # 4. Register in ultralytics/nn/tasks.py
    tasks_file = repo_root / "ultralytics/nn/tasks.py"
    tasks_txt = _strip(tasks_file.read_text(encoding="utf-8")).lstrip()

    top_import = f"""{SENTINEL_BEGIN}
from ultralytics.nn.modules.exp33_anticollapse import FCGHaarDownsample, AntiCollapseRTDETRDecoder
{SENTINEL_END}
"""

    # Register FCGHaarDownsample (same channel contract as HaarDownsample: c1=ch[f], c2=args[0])
    # and AntiCollapseRTDETRDecoder (same contract as RTDETRDecoder: args.insert(1, [ch[x] for x in f]))
    parse_snippet = f"""        {SENTINEL_BEGIN}
        elif m is FCGHaarDownsample:
            c1, c2 = ch[f], args[0]
            args = [c1, c2, *args[1:]]
        elif m is AntiCollapseRTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        {SENTINEL_END}"""

    inserted = False
    for anchor in ("        elif m is RTDETRDecoder:", "        elif m in (RTDETRDecoder,"):
        if anchor in tasks_txt:
            tasks_txt = tasks_txt.replace(anchor, parse_snippet + "\n" + anchor, 1)
            inserted = True
            break

    if not inserted:
        print("[ERROR] Could not locate RTDETRDecoder anchor in ultralytics/nn/tasks.py")
        return 1

    # Also ensure RTDETRModel._predict_once / forward recognizes AntiCollapseRTDETRDecoder
    # (Already works via isinstance(m, RTDETRDecoder) because AntiCollapseRTDETRDecoder subclasses RTDETRDecoder!)
    tasks_file.write_text(top_import + tasks_txt, encoding="utf-8")
    print(f"[SUCCESS] Installed RTDETR_Exp33_AntiCollapse (Exp33-Exp36) into {repo_root}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Install RTDETR_Exp33_AntiCollapse into RTDETR-main")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main")
    args = parser.parse_args()
    sys.exit(install(args.repo))


if __name__ == "__main__":
    main()
