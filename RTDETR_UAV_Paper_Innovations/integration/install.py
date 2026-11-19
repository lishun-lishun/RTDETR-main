#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Idempotent Installer for RT-DETR UAV Paper Innovations (Innovations 1–12)
=========================================================================
Aligned with `https://github.com/lishun-lishun/RTDETR-main/tree/codex/uav-research-integration`
(commits `6e9274d`, `7b472a9`, `4d7e695`, `de559ef`, `2415a23`, `0169ae7`).

Supports seamless in-place upgrade from Innovations 1–4 or 1–7 to Innovations 1–12:
  - Innovation 1:  WCASFP + ShallowEllipseAuxLoss (synced with 2415a23 & 0169ae7)
  - Innovation 2:  FAGAFusion
  - Innovation 3:  UADCRDecoder
  - Innovation 4:  SAHNWDRTDETRLoss (FIXED 300x `* nq` cls_loss scaling bug!)
  - Innovation 5:  DMBARepC3 / DMBABlock (standalone on original rtdetr-r18)
  - Innovation 6:  SSSCRepC3 / SSSCLite (standalone on original rtdetr-r18)
  - Innovation 7:  GLRoIDecoder / P2RoIReSampler (standalone on original rtdetr-r18)
  - Innovation 8:  UAVDenseO2OAugmenter + AuxO2MRTDETRLoss (standalone train-time-only on original rtdetr-r18)
  - Innovation 9:  SAFDRDecoder + AWLSDDistillLoss (standalone on original rtdetr-r18)
  - Innovation 10: TOLFDecoder + TOLFUncertaintyLoss (standalone on original rtdetr-r18)
  - Innovation 11: DySample2x + HWDDownsample (standalone on original rtdetr-r18)
  - Innovation 12: SMAIFI (standalone on original rtdetr-r18)
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
    pattern = re.compile(
        rf"[ \t]*{re.escape(SENTINEL_BEGIN)}.*?{re.escape(SENTINEL_END)}[ \t]*\n?",
        flags=re.DOTALL,
    )
    return pattern.sub("", text)


def copy_innovation_packages(pkg_root: Path, repo_root: Path) -> None:
    target_pkg = repo_root / "ultralytics" / "nn" / "uav_innovations"
    target_pkg.mkdir(parents=True, exist_ok=True)

    init_code = '''# -*- coding: utf-8 -*-
"""RT-DETR UAV Small-Object Detection Paper Innovations Package (Innovations 1–12)."""
from .innovation_1 import WCASFP, ShallowEllipseAuxLoss, HaarWaveletTransform2D, CenterSurroundContrast
from .innovation_2 import FAGAFusion
from .innovation_3 import UADCRDecoder, VarianceFlooredMSDeformAttn, UADDeformableDecoderLayer, UADDeformableDecoder
from .innovation_4 import SAHNWDRTDETRLoss, ScaleAdaptiveHungarianMatcher, normalized_wasserstein_distance
from .innovation_5 import DMBARepC3, DMBABlock
from .innovation_6 import SSSCRepC3, SSSCLite
from .innovation_7 import GLRoIDecoder, P2RoIReSampler
from .innovation_8 import UAVDenseO2OAugmenter, AuxO2MRTDETRLoss
from .innovation_9 import SAFDRDecoder, AWLSDDistillLoss
from .innovation_10 import TOLFDecoder, TOLFUncertaintyLoss
from .innovation_11 import DySample2x, HWDDownsample
from .innovation_12 import SMAIFI

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
    "UAVDenseO2OAugmenter",
    "AuxO2MRTDETRLoss",
    "SAFDRDecoder",
    "AWLSDDistillLoss",
    "TOLFDecoder",
    "TOLFUncertaintyLoss",
    "DySample2x",
    "HWDDownsample",
    "SMAIFI",
]
'''
    (target_pkg / "__init__.py").write_text(init_code, encoding="utf-8")

    for i in range(1, 13):
        src_mod = pkg_root / f"innovation_{i}" / "module.py"
        if src_mod.exists():
            dst_mod = target_pkg / f"innovation_{i}.py"
            # An upgrade must preserve fixes already made to installed modules 1–7.
            if i <= 7 and dst_mod.exists():
                continue
            shutil.copy2(src_mod, dst_mod)

    # Convert cross-innovation imports to relative package imports inside `ultralytics/nn/uav_innovations/`
    for fname in ["innovation_6.py", "innovation_7.py", "innovation_9.py", "innovation_10.py"]:
        fpath = target_pkg / fname
        if fpath.exists():
            txt = fpath.read_text(encoding="utf-8")
            txt = txt.replace("from RTDETR_UAV_Paper_Innovations.innovation_5.module import", "from .innovation_5 import")
            txt = txt.replace("from RTDETR_UAV_Paper_Innovations.innovation_3.module import", "from .innovation_3 import")
            fpath.write_text(txt, encoding="utf-8")

    # Copy model YAML configs into ultralytics/cfg/models/rt-detr/ (never overwrite original rtdetr-r18.yaml)
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
    SAFDRDecoder,
    TOLFDecoder,
    DySample2x,
    HWDDownsample,
    SMAIFI,
)
{SENTINEL_END}
"""
    init_file.write_text(clean_content + patch_block, encoding="utf-8")
    print(f"[PATCHED] {init_file}")
    return True


def patch_tasks_py(repo_root: Path, backup_dir: Path) -> bool:
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

    # Ensure `self.nc = self.yaml['nc']` in RTDETRDetectionModel.__init__ (commit 4d7e695)
    rtdetr_super = "super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)"
    if rtdetr_super in content and "self.nc = self.yaml['nc']" not in content:
        content = content.replace(
            rtdetr_super,
            rtdetr_super + "\n        self.nc = self.yaml['nc']",
            1,
        )

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
    UAVDenseO2OAugmenter,
    AuxO2MRTDETRLoss,
    SAFDRDecoder,
    AWLSDDistillLoss,
    TOLFDecoder,
    TOLFUncertaintyLoss,
    DySample2x,
    HWDDownsample,
    SMAIFI,
)
{SENTINEL_END}
"""

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
        elif m in (SSSCLite, DMBABlock, DySample2x, SMAIFI):
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
        elif m is HWDDownsample:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m in (UADCRDecoder, GLRoIDecoder, SAFDRDecoder, TOLFDecoder):
            args.insert(1, [ch[x] for x in f])
        {SENTINEL_END}"""

    target_branch_a = "        elif m in (RTDETRDecoder, UAVResearchDecoder, TinyUAVDecoder):"
    target_branch_b = "        elif m is RTDETRDecoder:"
    if target_branch_a in content:
        content = content.replace(target_branch_a, parse_block + "\n" + target_branch_a, 1)
    elif target_branch_b in content:
        content = content.replace(target_branch_b, parse_block + "\n" + target_branch_b, 1)
    else:
        raise RuntimeError("Cannot find RT-DETR parse_model branch; refusing to write tasks.py")

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

    base_crit = _ORIG_RTDETR_INIT_CRITERION(self)
    inno8_cfg = yaml_cfg.get("innovation_8", {{}})
    if isinstance(inno8_cfg, dict) and inno8_cfg.get("enabled", False):
        return AuxO2MRTDETRLoss(
            base_criterion=base_crit,
            nc=self.nc,
            topk_o2m=int(inno8_cfg.get("topk_o2m", 3)),
            o2m_weight=float(inno8_cfg.get("o2m_weight", 0.25)),
            mal_gamma=float(inno8_cfg.get("mal_gamma", 1.5)),
            enabled=True,
        )
    return base_crit


def _uav_loss(self, batch, preds=None):
    yaml_cfg = getattr(self, "yaml", {{}}) or {{}}

    # Innovation 8: In-batch UAV-Dense-O2O soft Copy-Paste before forward when preds is None
    inno8_cfg = yaml_cfg.get("innovation_8", {{}})
    if self.training and preds is None and isinstance(inno8_cfg, dict) and inno8_cfg.get("enabled", False):
        if not hasattr(self, "_uav_dense_o2o_aug"):
            self._uav_dense_o2o_aug = UAVDenseO2OAugmenter(
                max_paste_per_img=int(inno8_cfg.get("max_paste_per_img", 3)),
                prob=float(inno8_cfg.get("paste_prob", 0.75)),
                max_obj_area=float(inno8_cfg.get("max_obj_area", 0.04)),
                enabled=True,
            )
        self._uav_dense_o2o_aug.train(self.training)
        batch = self._uav_dense_o2o_aug(batch)

    total_loss, loss_items = _ORIG_RTDETR_LOSS(self, batch, preds=preds)

    if self.training:
        for m in self.model:
            # Innovation 1: WCASFP shallow ellipse aux loss + graph cache cleanup (0169ae7)
            if isinstance(m, WCASFP) and m.enabled and m.last_obj_logits is not None:
                if not hasattr(self, "_shallow_aux_criterion"):
                    inno1_cfg = yaml_cfg.get("innovation_1", {{}})
                    w = float(inno1_cfg.get("aux_shallow_loss_weight", 0.25)) if isinstance(inno1_cfg, dict) else 0.25
                    self._shallow_aux_criterion = ShallowEllipseAuxLoss(loss_weight=w)
                total_loss = total_loss + self._shallow_aux_criterion(m.last_obj_logits, batch)
                m.last_obj_logits = None
                m.last_obj_map = None

            # Innovation 9: SAFDRDecoder Area-Weighted LSD aux loss + graph cache cleanup
            if isinstance(m, SAFDRDecoder) and m.enabled and m.last_fdr_logits is not None:
                if not hasattr(self, "_awlsd_criterion"):
                    inno9_cfg = yaml_cfg.get("innovation_9", {{}})
                    self._awlsd_criterion = AWLSDDistillLoss(
                        temperature=float(inno9_cfg.get("awlsd_temp", 2.0)) if isinstance(inno9_cfg, dict) else 2.0,
                        tau_area=float(inno9_cfg.get("tau_area", 0.015)) if isinstance(inno9_cfg, dict) else 0.015,
                        loss_weight=float(inno9_cfg.get("awlsd_weight", 0.35)) if isinstance(inno9_cfg, dict) else 0.35,
                        enabled=True,
                    )
                total_loss = total_loss + self._awlsd_criterion(
                    m.last_fdr_logits, m.last_fdr_final_boxes, m.last_fdr_final_scores
                )
                m.last_fdr_logits = None
                m.last_fdr_final_boxes = None
                m.last_fdr_final_scores = None

            # Innovation 10: TOLFDecoder Noise-Robust Uncertainty NLL aux loss + graph cache cleanup
            if isinstance(m, TOLFDecoder) and m.enabled and m.last_tolf_logvars is not None:
                if not hasattr(self, "_tolf_criterion"):
                    inno10_cfg = yaml_cfg.get("innovation_10", {{}})
                    self._tolf_criterion = TOLFUncertaintyLoss(
                        loss_weight=float(inno10_cfg.get("tolf_weight", 0.25)) if isinstance(inno10_cfg, dict) else 0.25,
                        tau_area=float(inno10_cfg.get("tau_area", 0.015)) if isinstance(inno10_cfg, dict) else 0.015,
                        enabled=True,
                    )
                total_loss = total_loss + self._tolf_criterion(
                    m.last_tolf_logvars, m.last_tolf_bboxes, batch, num_dn=m.last_tolf_num_dn
                )
                m.last_tolf_logvars = None
                m.last_tolf_bboxes = None
    return total_loss, loss_items


RTDETRDetectionModel.init_criterion = _uav_init_criterion
RTDETRDetectionModel.loss = _uav_loss
{SENTINEL_END}
"""
    content = import_snippet + content.rstrip() + "\n" + runtime_hook
    tasks_file.write_text(content, encoding="utf-8")
    print(f"[PATCHED] {tasks_file}")
    return True


def install(repo_path: str, dry_run: bool = False) -> int:
    pkg_root = Path(__file__).resolve().parent.parent
    repo_root = Path(repo_path).resolve()
    if not (repo_root / "ultralytics").exists():
        print(f"[ERROR] {repo_root}/ultralytics does not exist. Please pass valid RTDETR-main path.")
        return 1
    if dry_run:
        print(f"[DRY RUN] Would install new modules and YAMLs into {repo_root}, then patch two registration files")
        return 0

    backup_dir = repo_root / ".uav_paper_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    copy_innovation_packages(pkg_root, repo_root)
    patch_modules_init(repo_root, backup_dir)
    patch_tasks_py(repo_root, backup_dir)
    print(f"[SUCCESS] All 12 UAV Paper Innovations (1–12) installed into {repo_root}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Install RT-DETR UAV Paper Innovations (1–12)")
    parser.add_argument("--repo", "--repo-root", type=str, default=".", help="Path to RTDETR-main repository root")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changing files")
    args = parser.parse_args()
    sys.exit(install(args.repo, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
