#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Idempotent Installer for RT-DETR UAV Paper Innovations (Innovations 1–7)
========================================================================
Aligned with `https://github.com/lishun-lishun/RTDETR-main/tree/codex/uav-research-integration`
(including commits `6e9274d`, `7b472a9`, `4d7e695`, `de559ef`, `2415a23`, `0169ae7`).

Key Safety & Upgrade Guarantees:
1. In-Place Upgrade & Idempotent:
   Whether `RTDETR-main` already has Innovations 1–4 installed (with existing
   `# >>> [UAV_PAPER_INNOVATIONS_BEGIN] >>>` blocks) or is a fresh branch, running
   `python RTDETR_UAV_Paper_Innovations/integration/install.py --repo .` cleanly
   upgrades the sentinel blocks in-place to support all 7 innovations (1–7), and
   running it multiple times produces byte-for-byte identical files.
2. Strictly Independent Innovations 5, 6, 7 on Original `rtdetr-r18.yaml`:
   - `rtdetr-r18-inno5-dmba.yaml`: ONLY Innovation 5 (`DMBARepC3` on Layers 19 & 22)
   - `rtdetr-r18-inno6-sssc.yaml`: ONLY Innovation 6 (`SSSCRepC3` on Layers 19 & 22)
   - `rtdetr-r18-inno7-glroi.yaml`: ONLY Innovation 7 (`GLRoIDecoder` on Layer 26)
   Never overwrites original `rtdetr-r18.yaml`.
"""

from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path

SENTINEL_BEGIN = "# >>> [UAV_PAPER_INNOVATIONS_BEGIN] >>>"
SENTINEL_END = "# <<< [UAV_PAPER_INNOVATIONS_END] <<<"


def _strip_existing_sentinels(text: str) -> str:
    """Removes any existing sentinel-guarded blocks so we can cleanly re-inject the latest version."""
    pattern = re.compile(
        rf"[ \t]*{re.escape(SENTINEL_BEGIN)}.*?{re.escape(SENTINEL_END)}[ \t]*\n?",
        flags=re.DOTALL,
    )
    return pattern.sub("", text)


def copy_innovation_packages(pkg_root: Path, repo_root: Path) -> None:
    """Copies innovation_1..7 into `ultralytics/nn/uav_innovations/` and copies YAML configs."""
    target_pkg = repo_root / "ultralytics" / "nn" / "uav_innovations"
    target_pkg.mkdir(parents=True, exist_ok=True)

    init_code = '''# -*- coding: utf-8 -*-
"""RT-DETR UAV Small-Object Detection Paper Innovations Package (Innovations 1–7)."""
from .innovation_1 import WCASFP, ShallowEllipseAuxLoss, HaarWaveletTransform2D, CenterSurroundContrast
from .innovation_2 import FAGAFusion
from .innovation_3 import UADCRDecoder, VarianceFlooredMSDeformAttn, UADDeformableDecoderLayer, UADDeformableDecoder
from .innovation_4 import SAHNWDRTDETRLoss, ScaleAdaptiveHungarianMatcher, normalized_wasserstein_distance
from .innovation_5 import DMBARepC3, DMBABlock
from .innovation_6 import SSSCRepC3, SSSCLite
from .innovation_7 import GLRoIDecoder, P2RoIReSampler

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
    "DMBARepC3",
    "DMBABlock",
    "SSSCRepC3",
    "SSSCLite",
    "GLRoIDecoder",
    "P2RoIReSampler",
]
'''
    (target_pkg / "__init__.py").write_text(init_code, encoding="utf-8")

    for i in range(1, 8):
        src_mod = pkg_root / f"innovation_{i}" / "module.py"
        if src_mod.exists():
            dst_mod = target_pkg / f"innovation_{i}.py"
            shutil.copy2(src_mod, dst_mod)

    # Fix internal cross-imports inside installed `ultralytics/nn/uav_innovations/` so they work both
    # as `ultralytics.nn.uav_innovations.innovation_X` and as `RTDETR_UAV_Paper_Innovations.innovation_X.module`
    inno6_dst = target_pkg / "innovation_6.py"
    if inno6_dst.exists():
        txt6 = inno6_dst.read_text(encoding="utf-8")
        txt6 = txt6.replace(
            "from RTDETR_UAV_Paper_Innovations.innovation_5.module import _StdConvNormAct, _RepConvBlock",
            "from .innovation_5 import _StdConvNormAct, _RepConvBlock",
        )
        inno6_dst.write_text(txt6, encoding="utf-8")

    inno7_dst = target_pkg / "innovation_7.py"
    if inno7_dst.exists():
        txt7 = inno7_dst.read_text(encoding="utf-8")
        txt7 = txt7.replace(
            "from RTDETR_UAV_Paper_Innovations.innovation_3.module import",
            "from .innovation_3 import",
        )
        inno7_dst.write_text(txt7, encoding="utf-8")

    # Copy model YAML configs into ultralytics/cfg/models/rt-detr/
    cfg_dir = repo_root / "ultralytics" / "cfg" / "models" / "rt-detr"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    model_cfgs_src = pkg_root / "integration" / "model_configs"
    if model_cfgs_src.exists():
        for yaml_file in sorted(model_cfgs_src.glob("*.yaml")):
            dst_yaml = cfg_dir / yaml_file.name
            if dst_yaml.name == "rtdetr-r18.yaml" and dst_yaml.exists():
                continue
            shutil.copy2(yaml_file, dst_yaml)


def patch_modules_init(repo_root: Path, backup_dir: Path) -> bool:
    """Registers Innovations 1–7 in ultralytics/nn/modules/__init__.py."""
    init_file = repo_root / "ultralytics" / "nn" / "modules" / "__init__.py"
    if not init_file.exists():
        print(f"[WARN] {init_file} not found, skipping.")
        return False

    raw_content = init_file.read_text(encoding="utf-8")
    rel = init_file.relative_to(repo_root)
    bak = backup_dir / rel
    bak.parent.mkdir(parents=True, exist_ok=True)
    if not bak.exists() and SENTINEL_BEGIN not in raw_content:
        shutil.copy2(init_file, bak)

    clean_content = _strip_existing_sentinels(raw_content).rstrip() + "\n"
    patch_block = f"""{SENTINEL_BEGIN}
from ultralytics.nn.uav_innovations import (
    WCASFP,
    FAGAFusion,
    UADCRDecoder,
    DMBARepC3,
    DMBABlock,
    SSSCRepC3,
    SSSCLite,
    GLRoIDecoder,
)
{SENTINEL_END}
"""
    init_file.write_text(clean_content + patch_block, encoding="utf-8")
    print(f"[PATCHED] {init_file}")
    return True


def patch_tasks_py(repo_root: Path, backup_dir: Path) -> bool:
    """
    Patches `ultralytics/nn/tasks.py` idempotently (supports upgrading from Innovations 1–4 to 1–7):
    1. Imports WCASFP, ShallowEllipseAuxLoss, FAGAFusion, UADCRDecoder, SAHNWDRTDETRLoss,
       DMBARepC3, DMBABlock, SSSCRepC3, SSSCLite, GLRoIDecoder
    2. Ensures `self.nc = self.yaml['nc']` in `RTDETRDetectionModel.__init__` (commit 4d7e695)
    3. Registers WCASFP, FAGAFusion, UADCRDecoder, DMBARepC3, SSSCRepC3, SSSCLite, GLRoIDecoder in `parse_model`
    4. Appends `_uav_init_criterion` & `_uav_loss` (with `m.last_obj_map = None` from commit 0169ae7)
    """
    tasks_file = repo_root / "ultralytics" / "nn" / "tasks.py"
    if not tasks_file.exists():
        print(f"[WARN] {tasks_file} not found, skipping.")
        return False

    raw_content = tasks_file.read_text(encoding="utf-8")
    rel = tasks_file.relative_to(repo_root)
    bak = backup_dir / rel
    bak.parent.mkdir(parents=True, exist_ok=True)
    if not bak.exists() and SENTINEL_BEGIN not in raw_content:
        shutil.copy2(tasks_file, bak)

    content = _strip_existing_sentinels(raw_content).lstrip()

    # Ensure `self.nc = self.yaml['nc']` is present in RTDETRDetectionModel.__init__ (commit 4d7e695)
    rtdetr_super = "super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)"
    if rtdetr_super in content and "self.nc = self.yaml['nc']" not in content:
        content = content.replace(
            rtdetr_super,
            rtdetr_super + "\n        self.nc = self.yaml['nc']",
            1,
        )

    # 1. Top import block
    import_snippet = f"""{SENTINEL_BEGIN}
from ultralytics.nn.uav_innovations import (
    WCASFP,
    ShallowEllipseAuxLoss,
    FAGAFusion,
    UADCRDecoder,
    SAHNWDRTDETRLoss,
    DMBARepC3,
    DMBABlock,
    SSSCRepC3,
    SSSCLite,
    GLRoIDecoder,
)
{SENTINEL_END}
"""

    # 2. Inject parse_model support right before RTDETRDecoder branch
    parse_block = f"""        {SENTINEL_BEGIN}
        elif m is WCASFP:
            c1 = [ch[x] for x in f] if isinstance(f, list) else ch[f]
            c2 = args[0] if len(args) > 0 else 256
            args = [c1, c2, *args[1:]]
        elif m is FAGAFusion:
            c1 = [ch[x] for x in f]
            c2 = sum(c1)
            args = [c1, *args]
        elif m in (DMBARepC3, SSSCRepC3):
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, n, *args[1:]]
            n = 1
        elif m in (SSSCLite, DMBABlock):
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
        elif m in (UADCRDecoder, GLRoIDecoder):
            args.insert(1, [ch[x] for x in f])
        {SENTINEL_END}"""

    target_branch_a = "        elif m in (RTDETRDecoder, UAVResearchDecoder, TinyUAVDecoder):"
    target_branch_b = "        elif m is RTDETRDecoder:"
    if target_branch_a in content:
        content = content.replace(target_branch_a, parse_block + "\n" + target_branch_a, 1)
    elif target_branch_b in content:
        content = content.replace(target_branch_b, parse_block + "\n" + target_branch_b, 1)

    # 3. Runtime hook at end of tasks.py (preserving 0169ae7 cache cleanup)
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
    content = import_snippet + content.rstrip() + "\n" + runtime_hook
    tasks_file.write_text(content, encoding="utf-8")
    print(f"[PATCHED] {tasks_file}")
    return True


def install(repo_path: str) -> int:
    pkg_root = Path(__file__).resolve().parent.parent
    repo_root = Path(repo_path).resolve()
    if not (repo_root / "ultralytics").exists():
        print(f"[ERROR] {repo_root}/ultralytics does not exist. Please pass valid RTDETR-main path.")
        return 1

    backup_dir = repo_root / ".uav_paper_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    copy_innovation_packages(pkg_root, repo_root)
    patch_modules_init(repo_root, backup_dir)
    patch_tasks_py(repo_root, backup_dir)
    print(f"[SUCCESS] All 7 UAV Paper Innovations (1–7) installed into {repo_root}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Install RT-DETR UAV Paper Innovations (1–7)")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main repository root")
    args = parser.parse_args()
    sys.exit(install(args.repo))


if __name__ == "__main__":
    main()
