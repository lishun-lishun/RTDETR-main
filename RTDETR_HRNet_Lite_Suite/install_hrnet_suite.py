#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Idempotent Installer for `RTDETR_HRNet_Lite_Suite` (Exp 1 - Exp 5) into `RTDETR-main`.
Integrates cleanly without modifying original `rtdetr-r18.yaml`, `hrnet.py`, or `rtdetr_hrnet_w18.yaml`.
"""

from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path

SENTINEL_BEGIN = "# >>> [HRNET_LITE_SUITE_BEGIN] >>>"
SENTINEL_END = "# <<< [HRNET_LITE_SUITE_END] <<<"


def _strip(text: str) -> str:
    return re.compile(rf"[ \t]*{re.escape(SENTINEL_BEGIN)}.*?{re.escape(SENTINEL_END)}[ \t]*\n?", re.DOTALL).sub("", text)


def install(repo_path: str) -> int:
    pkg_root = Path(__file__).resolve().parent
    repo_root = Path(repo_path).resolve()
    if not (repo_root / "ultralytics").is_dir():
        print(f"[ERROR] Invalid RTDETR-main repository path: {repo_root}")
        return 1

    # 1. Copy hrnet_lite_suite.py into ultralytics/nn/modules/
    shutil.copy2(pkg_root / "hrnet_lite_suite.py", repo_root / "ultralytics/nn/modules/hrnet_lite_suite.py")

    # 2. Copy Exp1-Exp5 YAML configs into ultralytics/cfg/models/rt-detr/
    cfg_dst = repo_root / "ultralytics/cfg/models/rt-detr"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    for y in sorted((pkg_root / "configs").glob("*.yaml")):
        shutil.copy2(y, cfg_dst / y.name)

    # 3. Patch ultralytics/nn/modules/__init__.py
    init_file = repo_root / "ultralytics/nn/modules/__init__.py"
    init_txt = _strip(init_file.read_text(encoding="utf-8")).rstrip() + "\n"
    init_block = f"""{SENTINEL_BEGIN}
from .hrnet_lite_suite import (
    BaseHRNetV2W18,
    HRNetPruneS4,
    HRNetPConvRep,
    FastHRNetBackbone,
    SlimGhostRepC3,
    HaarP2FoldTransition,
    PConvRepBlock,
    ECALayer,
)
{SENTINEL_END}
"""
    init_file.write_text(init_txt + init_block, encoding="utf-8")

    # 4. Patch ultralytics/nn/tasks.py
    tasks_file = repo_root / "ultralytics/nn/tasks.py"
    tasks_txt = _strip(tasks_file.read_text(encoding="utf-8")).lstrip()

    top_import = f"""{SENTINEL_BEGIN}
from ultralytics.nn.modules.hrnet_lite_suite import (
    BaseHRNetV2W18,
    HRNetPruneS4,
    HRNetPConvRep,
    FastHRNetBackbone,
    SlimGhostRepC3,
)
{SENTINEL_END}
"""
    parse_snippet = f"""        {SENTINEL_BEGIN}
        elif m in (BaseHRNetV2W18, HRNetPruneS4, HRNetPConvRep, FastHRNetBackbone):
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
        elif m is SlimGhostRepC3:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, n, *args[1:]]
            n = 1
        {SENTINEL_END}"""

    anchor_candidates = [
        "        elif m in (RTHRNetW18, RTHRNetW18Fast, RTHRNetW18Balanced):",
        "        elif m in (HRNetV2W18, RBHRNetA, RBHRNetB):",
        "        elif m is HRNetV2W18:",
        "        elif m in (RTDETRDecoder, UAVResearchDecoder, TinyUAVDecoder):",
        "        elif m is RTDETRDecoder:",
    ]
    inserted = False
    for anchor in anchor_candidates:
        if anchor in tasks_txt:
            tasks_txt = tasks_txt.replace(anchor, parse_snippet + "\n" + anchor, 1)
            inserted = True
            break
    if not inserted:
        print("[WARN] Could not locate parse_model anchor automatically; please add BaseHRNetV2W18 & SlimGhostRepC3 in tasks.py.")

    tasks_file.write_text(top_import + tasks_txt, encoding="utf-8")
    print(f"[SUCCESS] Installed RTDETR_HRNet_Lite_Suite (Exp1-Exp5) into {repo_root}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Install RTDETR_HRNet_Lite_Suite into RTDETR-main")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main")
    args = parser.parse_args()
    sys.exit(install(args.repo))


if __name__ == "__main__":
    main()
