#!/usr/bin/env python3
"""Idempotent, minimally invasive installer for lishun-lishun/RTDETR-main.

Defaults to dry-run. Apply only to a clean, matching local checkout.
Source modules copied as NEW files, original R18 YAML untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def patch_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if text.count(old) != 1:
        raise RuntimeError(f'{label}: expected exactly one original anchor, found {text.count(old)}')
    return text.replace(old, new, 1)


def build_plan(root: Path):
    init_file = root / 'ultralytics/nn/modules/__init__.py'
    tasks_file = root / 'ultralytics/nn/tasks.py'
    yaml_file = root / 'ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml'
    for f in (init_file, tasks_file, yaml_file):
        if not f.is_file():
            raise FileNotFoundError(f'Expected RTDETR-main file missing: {f}')
    init = init_file.read_text(encoding='utf8')
    tasks = tasks_file.read_text(encoding='utf8')
    init = patch_once(init, 'from .head import *',
                      'from .head import *\nfrom .tiny_uav_decoder import TinyUAVDecoder',
                      'module export')
    tasks = patch_once(tasks, 'elif m in (RTDETRDecoder, UAVResearchDecoder):  # special case',
                       'elif m in (RTDETRDecoder, UAVResearchDecoder, TinyUAVDecoder):  # special case',
                       'decoder parser')
    # Modify only RTDETRDetectionModel subclass: locate its own constructor.
    prefix = 'class RTDETRDetectionModel(DetectionModel):'
    start = tasks.find(prefix)
    if start < 0:
        raise RuntimeError('RTDETRDetectionModel class not found')
    end = tasks.find('\nclass ', start + len(prefix))
    if end < 0:
        end = len(tasks)
    cls = tasks[start:end]
    cls = patch_once(cls, 'super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)',
                     'super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)\n'
                     '        tiny = self.yaml.get("tiny_uav") or {}\n'
                     '        if tiny and isinstance(self.model[-1], TinyUAVDecoder):\n'
                     '            self.model[-1].configure(tiny)', 'RTDETR model ctor')
    orig = ('        cfg = self.yaml.get(\'uav_research\', {})')
    replace = ('        tiny = self.yaml.get("tiny_uav")\n'
               '        if tiny:\n'
               '            from ultralytics.models.utils.tiny_uav_loss import TinyUAVRTDETRLoss\n'
               '            return TinyUAVRTDETRLoss(nc=self.nc, settings=tiny, use_vfl=True, '
               'use_sl=False, use_emasl=False, use_svfl=False, '
               'use_emasvfl=False, use_mal=False)\n' + orig)
    cls = patch_once(cls, orig, replace, 'RTDETR loss constructor')
    tasks = tasks[:start] + cls + tasks[end:]
    # Original YAML contains an exactly matching final decoder declaration.
    base_yaml = yaml_file.read_text(encoding='utf8')
    # The upstream YAML has a whitespace-only line. Keep generated files
    # free of trailing whitespace without touching the original YAML.
    base_yaml = '\n'.join(line.rstrip() for line in base_yaml.splitlines()) + '\n'
    final_old = '[[19, 22, 25], 1, RTDETRDecoder, [nc, 256, 300, 4, 8, 3]]'
    final_new = '[[19, 22, 25], 1, TinyUAVDecoder, [nc, 256, 300, 4, 8, 3]]'
    if base_yaml.count(final_old) != 1:
        raise RuntimeError('original R18 YAML decoder signature changed; cannot install automatically')
    base_yaml = base_yaml.replace(final_old, final_new)
    candidates = {
        'm1_peak': {'peak_alpha': .6},
        'm2_dn': {'dn_pos_px': 1.25, 'dn_neg_px': 4.0},
        'm3_match': {'center_match_weight': .6},
        'm4_nwd': {'nwd_weight': 1.0},
        'm1_m3': {'peak_alpha': .6, 'center_match_weight': .6},
        'm2_m4': {'dn_pos_px': 1.25, 'dn_neg_px': 4.0, 'nwd_weight': 1.0},
        'all4': {'peak_alpha': .6, 'dn_pos_px': 1.25,
                 'dn_neg_px': 4.0, 'center_match_weight': .6, 'nwd_weight': 1.0},
        'off_control': {},
    }
    source = {
        root / 'ultralytics/nn/modules/tiny_uav_decoder.py': (HERE / 'src/tiny_uav_decoder.py').read_bytes(),
        root / 'ultralytics/models/utils/tiny_uav_loss.py': (HERE / 'src/tiny_uav_loss.py').read_bytes(),
        init_file: init.encode(), tasks_file: tasks.encode(),
    }
    for name, params in candidates.items():
        cfg = {'peak_alpha': 0, 'peak_kernel': 3,
               'dn_pos_px': 0, 'dn_neg_px': 0, 'center_match_weight': 0,
               'nwd_weight': 0, 'nwd_c_px': 12, 'input_pixels': 640,
               'tiny_threshold': 24, 'tiny_transition': 6}
        cfg.update(params)
        addon = '\n# Opt-in DUT-Anti-UAV research settings\ntiny_uav:\n'
        addon += ''.join(f'  {k}: {v}\n' for k, v in cfg.items())
        yaml_dest = yaml_file.parent / f'rtdetr_r18_tinyv2_{name}.yaml'
        source[yaml_dest] = (base_yaml + addon).encode()
    return source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    root = Path(args.repo).expanduser().resolve()
    changes = build_plan(root)
    preexisting = []
    for dst, data in changes.items():
        if dst.exists():
            if dst.read_bytes() == data:
                status = 'unchanged'
            elif dst.name in ('tasks.py', '__init__.py'):
                status = 'patch'
            else:
                raise FileExistsError(f'Will not overwrite nonmatching file: {dst}')
        else:
            status = 'new'
        preexisting.append((dst, data, status))
    for dst, data, status in preexisting:
        print(f'{status:9s} {dst.relative_to(root)}')
    if not args.apply:
        print('DRY RUN: no changes made. Pass --apply after reviewing the plan.')
        return
    # Preflight all writes before actually touching any file.
    for dst, data, status in preexisting:
        if status == 'unchanged':
            continue
        if status == 'patch':
            digest = hashlib.sha256(dst.read_bytes()).hexdigest()[:10]
            backup = dst.with_name(dst.name + '.tinyv2.' + digest + '.bak')
            if not backup.exists():
                shutil.copy2(dst, backup)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    print('Installed. Original rtdetr-r18.yaml and existing runs/checkpoints untouched.')


if __name__ == '__main__':
    main()
