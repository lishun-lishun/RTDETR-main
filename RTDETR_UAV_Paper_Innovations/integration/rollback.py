#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rollback Script for RT-DETR UAV Paper Innovations
=================================================
Cleanly restores all modified files in `RTDETR-main` from `.uav_paper_backup/`
or strips sentinel blocks `# >>> [UAV_PAPER_INNOVATIONS_BEGIN] >>> ... # <<< [UAV_PAPER_INNOVATIONS_END] <<<`,
and removes installed innovation package files without touching baseline configs,
weights, or `runs/`.
"""

from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path

SENTINEL_PATTERN = re.compile(
    r"\n?# >>> \[UAV_PAPER_INNOVATIONS_BEGIN\] >>>.*?# <<< \[UAV_PAPER_INNOVATIONS_END\] <<<\n?",
    re.DOTALL,
)


def rollback(repo_path: str) -> int:
    repo_root = Path(repo_path).resolve()
    backup_dir = repo_root / ".uav_paper_backup"

    # 1. Restore from backup if available, otherwise strip sentinel blocks
    target_files = [
        repo_root / "ultralytics" / "nn" / "modules" / "__init__.py",
        repo_root / "ultralytics" / "nn" / "tasks.py",
    ]
    for fpath in target_files:
        if not fpath.exists():
            continue
        rel = fpath.relative_to(repo_root)
        bak = backup_dir / rel
        if bak.exists():
            shutil.copy2(bak, fpath)
            print(f"[RESTORED] {fpath} from backup")
        else:
            content = fpath.read_text(encoding="utf-8")
            cleaned = SENTINEL_PATTERN.sub("\n", content)
            fpath.write_text(cleaned, encoding="utf-8")
            print(f"[CLEANED] {fpath} via sentinel removal")

    # 2. Remove uav_innovations package directory
    uav_pkg = repo_root / "ultralytics" / "nn" / "uav_innovations"
    if uav_pkg.exists():
        shutil.rmtree(uav_pkg)
        print(f"[REMOVED] {uav_pkg}")

    # 3. Remove added paper YAML configs (strictly keeping rtdetr-r18.yaml intact)
    cfg_dir = repo_root / "ultralytics" / "cfg" / "models" / "rt-detr"
    if cfg_dir.exists():
        for name in [
            "rtdetr-r18-inno1-wcasfp.yaml",
            "rtdetr-r18-inno2-faga.yaml",
            "rtdetr-r18-inno3-uadcr.yaml",
            "rtdetr-r18-inno4-sahnwd.yaml",
            "rtdetr-r18-combo-a.yaml",
            "rtdetr-r18-combo-b.yaml",
            "rtdetr-r18-uav-paper-full.yaml",
        ]:
            p = cfg_dir / name
            if p.exists():
                p.unlink()
                print(f"[REMOVED] {p}")

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    print(f"[SUCCESS] Rollback completed cleanly on {repo_root}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Rollback RT-DETR UAV Paper Innovations")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main repository root")
    args = parser.parse_args()
    sys.exit(rollback(args.repo))


if __name__ == "__main__":
    main()
