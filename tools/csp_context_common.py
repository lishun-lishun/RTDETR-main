"""Shared paths and protocol handling for R18 CSP-stage experiments."""
import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'ultralytics/cfg/models/rt-detr'
PROFILE = ROOT / 'ultralytics/cfg/anti_detr_default.yaml'
DATA = ROOT / 'dataset/DUT-Anti-UAV.yaml'
RUNS = ROOT / 'runs/csp_context_r18'
CONFIGS = {
    'stage3': 'rtdetr_r18_csp_stage3.yaml',
    'stage4': 'rtdetr_r18_csp_stage4.yaml',
    'combined': 'rtdetr_r18_csp_context.yaml',
    'stage4_k3': 'rtdetr_r18_csp_stage4_k3.yaml',
    'stage3_ratio05': 'rtdetr_r18_csp_stage3_ratio05.yaml',
}


def local_package():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.find_spec('ultralytics')
    actual = Path(spec.origin).resolve() if spec and spec.origin else None
    expected = (ROOT / 'ultralytics/__init__.py').resolve()
    if actual != expected:
        raise RuntimeError(f'Expected local Ultralytics {expected}, found {actual}')


def settings_for_run(args, parser):
    profile = yaml.safe_load(PROFILE.read_text(encoding='utf-8'))
    if args.baseline_args:
        if not args.baseline_args.is_file():
            parser.error('baseline args.yaml does not exist')
        baseline = yaml.safe_load(args.baseline_args.read_text(encoding='utf-8'))
        keys = ('epochs', 'imgsz', 'optimizer', 'lr0', 'lrf', 'weight_decay',
                'amp', 'pretrained')
        mismatches = {key: (baseline.get(key), profile[key]) for key in keys
                      if baseline.get(key) != profile[key]}
        accumulation = baseline.get('gradient_accumulation', 1)
        batch = baseline.get('batch', 0) * accumulation
        if baseline.get('warmup_epochs', 0) / accumulation != profile['warmup_epochs']:
            mismatches['warmup_optimizer_iterations'] = (
                baseline.get('warmup_epochs', 0) / accumulation,
                profile['warmup_epochs'])
        if mismatches or batch != profile['batch']:
            parser.error(f'baseline protocol differs: {mismatches}; effective batch={batch}')
        allowed = yaml.safe_load((ROOT / 'ultralytics/cfg/default.yaml').read_text(encoding='utf-8'))
        settings = {key: value for key, value in baseline.items() if key in allowed}
    else:
        print('WARNING: historical R18 args.yaml unavailable; exact protocol parity is '
              'unverified. Using checked-in anti_detr_default.yaml.', flush=True)
        settings = {key: value for key, value in profile.items()
                    if key not in ('test_batch', 'micro_batch')}
        batch = profile['batch']
    settings['batch'] = profile['micro_batch']
    settings['gradient_accumulation'] = batch // profile['micro_batch']
    settings['warmup_epochs'] = profile['warmup_epochs'] * settings['gradient_accumulation']
    for key in ('model', 'mode', 'project', 'name', 'device', 'resume',
                'save_dir', 'cfg', 'data'):
        settings.pop(key, None)
    settings.update(data=str(DATA), device=args.device,
                    project=str(args.output_root / args.model),
                    name=f'seed{args.seed}', exist_ok=True, seed=args.seed)
    return settings, batch
