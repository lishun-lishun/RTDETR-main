#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Idempotent Installer for RT-DETR UAV Paper Innovations (Innovations 1–4)
========================================================================
Safely integrates the 4 paper innovations into an existing `RTDETR-main`
(Ultralytics YOLOv8.0.201) repository without overwriting original baseline
configs (`rtdetr-r18.yaml`), pretrained weights, or `runs/` directories.

Key Safety Guarantees:
1. Idempotent: Running `python integration/install.py --repo /path/to/RTDETR-main`
   multiple times will NEVER duplicate code blocks.
2. Non-destructive backups: Saves original files to `.uav_paper_backup/` before
   applying minimal sentinel-guarded hooks.
3. Full Rollback: `python integration/rollback.py --repo /path/to/RTDETR-main`
   cleanly restores the exact original state.
"""

from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

SENTINEL_BEGIN = "# >>> [UAV_PAPER_INNOVATIONS_BEGIN] >>>"
SENTINEL_END = "# <<< [UAV_PAPER_INNOVATIONS_END] <<<"


def copy_innovation_packages(pkg_root: Path, repo_root: Path) -> None:
    """Copies innovation_1..4 into `ultralytics/nn/uav_innovations/` and copies YAML configs."""
    target_pkg = repo_root / "ultralytics" / "nn" / "uav_innovations"
    target_pkg.mkdir(parents=True, exist_ok=True)

    init_code = '''# -*- coding: utf-8 -*-
"""RT-DETR UAV Small-Object Detection Paper Innovations Package."""
from .innovation_1 import WCASFP, ShallowEllipseAuxLoss, HaarWaveletTransform2D, CenterSurroundContrast
from .innovation_2 import FAGAFusion
from .innovation_3 import UADCRDecoder, VarianceFlooredMSDeformAttn, UADDeformableDecoderLayer, UADDeformableDecoder
from .innovation_4 import SAHNWDRTDETRLoss, ScaleAdaptiveHungarianMatcher, normalized_wasserstein_distance

__all__ = [
    "WCASFP",
    "ShallowEllipseAuxLoss",
    "HaarWaveletTransform2D",
    "CenterSurroundContrast",
    "FAGAFusion",
    "UADCRDecoder",
    "VarianceFlooredMSDeformAttn",
    "UADDeformableDecoderLayer",
    "UADDeformableDecoder",
    "SAHNWDRTDETRLoss",
    "ScaleAdaptiveHungarianMatcher",
    "normalized_wasserstein_distance",
]
'''
    (target_pkg / "__init__.py").write_text(init_code, encoding="utf-8")

    for i in range(1, 5):
        src_mod = pkg_root / f"innovation_{i}" / "module.py"
        dst_mod = target_pkg / f"innovation_{i}.py"
        shutil.copy2(src_mod, dst_mod)

    # Copy new non-destructive model YAML configs into ultralytics/cfg/models/rt-detr/
    cfg_dir = repo_root / "ultralytics" / "cfg" / "models" / "rt-detr"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    model_cfgs_src = pkg_root / "integration" / "model_configs"
    if model_cfgs_src.exists():
        for yaml_file in model_cfgs_src.glob("*.yaml"):
            dst_yaml = cfg_dir / yaml_file.name
            # Never overwrite original rtdetr-r18.yaml
            if dst_yaml.name == "rtdetr-r18.yaml" and dst_yaml.exists():
                continue
            shutil.copy2(yaml_file, dst_yaml)


def patch_modules_init(repo_root: Path, backup_dir: Path) -> bool:
    """Registers WCASFP, FAGAFusion, UADCRDecoder in ultralytics/nn/modules/__init__.py."""
    init_file = repo_root / "ultralytics" / "nn" / "modules" / "__init__.py"
    if not init_file.exists():
        print(f"[WARN] {init_file} not found, skipping.")
        return False

    content = init_file.read_text(encoding="utf-8")
    if SENTINEL_BEGIN in content:
        print(f"[OK] {init_file} already patched (idempotent skip).")
        return True

    # Backup original file
    rel = init_file.relative_to(repo_root)
    bak = backup_dir / rel
    bak.parent.mkdir(parents=True, exist_ok=True)
    if not bak.exists():
        shutil.copy2(init_file, bak)

    patch_block = f"""
{SENTINEL_BEGIN}
from ultralytics.nn.uav_innovations import WCASFP, FAGAFusion, UADCRDecoder
{SENTINEL_END}
"""
    init_file.write_text(content + patch_block, encoding="utf-8")
    print(f"[PATCHED] {init_file}")
    return True


def patch_tasks_py(repo_root: Path, backup_dir: Path) -> bool:
    """
    Patches `ultralytics/nn/tasks.py` idempotently:
    1. Imports WCASFP, ShallowEllipseAuxLoss, FAGAFusion, UADCRDecoder, SAHNWDRTDETRLoss
    2. Registers WCASFP, FAGAFusion, UADCRDecoder in `parse_model`
    3. Adds optional SAHNWDRTDETRLoss & ShallowEllipseAuxLoss hook in `RTDETRDetectionModel`
    """
    tasks_file = repo_root / "ultralytics" / "nn" / "tasks.py"
    if not tasks_file.exists():
        print(f"[WARN] {tasks_file} not found, skipping.")
        return False

    content = tasks_file.read_text(encoding="utf-8")
    if SENTINEL_BEGIN in content:
        print(f"[OK] {tasks_file} already patched (idempotent skip).")
        return True

    rel = tasks_file.relative_to(repo_root)
    bak = backup_dir / rel
    bak.parent.mkdir(parents=True, exist_ok=True)
    if not bak.exists():
        shutil.copy2(tasks_file, bak)

    # 1. Add import at top or after existing module imports
    import_snippet = f"""{SENTINEL_BEGIN}
from ultralytics.nn.uav_innovations import (
    WCASFP,
    ShallowEllipseAuxLoss,
    FAGAFusion,
    UADCRDecoder,
    SAHNWDRTDETRLoss,
)
{SENTINEL_END}
"""

    # 2. Inject parse_model support for WCASFP, FAGAFusion, UADCRDecoder
    parse_target = "        elif m in (RTDETRDecoder, UAVResearchDecoder, TinyUAVDecoder):"
    parse_replacement = f"""        {SENTINEL_BEGIN}
        elif m is WCASFP:
            c1 = [ch[x] for x in f] if isinstance(f, list) else ch[f]
            c2 = args[0] if len(args) > 0 else 256
            args = [c1, c2, *args[1:]]
        elif m is FAGAFusion:
            c1 = [ch[x] for x in f]
            c2 = sum(c1)
            args = [c1, *args]
        elif m is UADCRDecoder:
            args.insert(1, [ch[x] for x in f])
        {SENTINEL_END}
        elif m in (RTDETRDecoder, UAVResearchDecoder, TinyUAVDecoder):"""

    if content.count(parse_target) != 1:
        raise RuntimeError('Current parse_model decoder anchor not found exactly once')
    content = content.replace(parse_target, parse_replacement, 1)

    # 3. Append runtime helper at end of tasks.py that wraps RTDETRDetectionModel.init_criterion & loss
    runtime_hook = f"""
{SENTINEL_BEGIN}
_ORIG_RTDETR_INIT_CRITERION = RTDETRDetectionModel.init_criterion
_ORIG_RTDETR_LOSS = RTDETRDetectionModel.loss


def _uav_init_criterion(self):
    yaml_cfg = getattr(self, "yaml", {{}}) or {{}}
    inno4_cfg = yaml_cfg.get("innovation_4", {{}})
    if isinstance(inno4_cfg, dict) and inno4_cfg.get("enabled", False):
        return SAHNWDRTDETRLoss(
            nc=self.nc,
            aux_loss=True,
            use_fl=True,
            use_vfl=True,
            tau_s=float(inno4_cfg.get("tau_s", 0.0025)),
            constant_c=float(inno4_cfg.get("constant_c", 12.8)),
            img_size=float(inno4_cfg.get("img_size", 640.0)),
            r_min=float(inno4_cfg.get("r_min", 1.00)),
            r_max=float(inno4_cfg.get("r_max", 1.30)),
            omega_nwd=float(inno4_cfg.get("omega_nwd", 0.50)),
            gamma_mal=float(inno4_cfg.get("gamma_mal", 1.5)),
            lambda_lsd=float(inno4_cfg.get("lambda_lsd", 0.50)),
            enabled=True,
        )
    return _ORIG_RTDETR_INIT_CRITERION(self)


def _uav_loss(self, batch, preds=None):
    total_loss, loss_items = _ORIG_RTDETR_LOSS(self, batch, preds=preds)
    if self.training:
        for m in self.model:
            if isinstance(m, WCASFP) and m.enabled and m.last_obj_logits is not None:
                if not hasattr(self, "_shallow_aux_criterion"):
                    yaml_cfg = getattr(self, "yaml", {{}}) or {{}}
                    inno1_cfg = yaml_cfg.get("innovation_1", {{}})
                    w = float(inno1_cfg.get("aux_shallow_loss_weight", 0.25)) if isinstance(inno1_cfg, dict) else 0.25
                    self._shallow_aux_criterion = ShallowEllipseAuxLoss(loss_weight=w)
                aux_s_loss = self._shallow_aux_criterion(m.last_obj_logits, batch)
                total_loss = total_loss + aux_s_loss
                m.last_obj_logits = None
                m.last_obj_map = None
    return total_loss, loss_items


RTDETRDetectionModel.init_criterion = _uav_init_criterion
RTDETRDetectionModel.loss = _uav_loss
{SENTINEL_END}
"""
    content = import_snippet + content + runtime_hook
    tasks_file.write_text(content, encoding="utf-8")
    print(f"[PATCHED] {tasks_file}")
    return True


def install(repo_path: str, apply: bool = False) -> int:
    pkg_root = Path(__file__).resolve().parent.parent
    repo_root = Path(repo_path).resolve()
    if not (repo_root / "ultralytics").exists():
        print(f"[ERROR] {repo_root}/ultralytics does not exist. Please pass valid RTDETR-main path.")
        return 1

    backup_dir = repo_root / ".uav_paper_backup"
    if not apply:
        print('[DRY RUN] Would copy four modules and seven new YAMLs, then patch modules/__init__.py and tasks.py')
        print('[DRY RUN] No files changed. Pass --apply after review.')
        return 0
    backup_dir.mkdir(parents=True, exist_ok=True)

    copy_innovation_packages(pkg_root, repo_root)
    patch_modules_init(repo_root, backup_dir)
    patch_tasks_py(repo_root, backup_dir)
    print(f"[SUCCESS] All 4 UAV Paper Innovations installed into {repo_root}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Install RT-DETR UAV Paper Innovations")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main repository root")
    parser.add_argument('--apply', action='store_true', help='write changes; default is dry run')
    args = parser.parse_args()
    sys.exit(install(args.repo, args.apply))


if __name__ == "__main__":
    main()
