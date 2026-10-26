"""Shared paths and import guard for the R18 Res2/GD experiments."""
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'ultralytics/cfg/models/rt-detr'
PROFILE = ROOT / 'ultralytics/cfg/anti_detr_default.yaml'
DATA = ROOT / 'dataset/DUT-Anti-UAV.yaml'
RUNS = ROOT / 'runs/res2basic_litegd'
CONFIGS = {name: f'rtdetr_r18_{name}.yaml' for name in (
    'res2basic', 'res2basic_scale2', 'res2basic_both',
    'res2basic_stage34', 'res2basic_independent', 'litegd',
    'litegd_high_only', 'litegd_low_only', 'litegd_local_only',
    'litegd_p3_only', 'res2basic_litegd')}


def local_package():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.find_spec('ultralytics')
    actual = Path(spec.origin).resolve() if spec and spec.origin else None
    expected = (ROOT / 'ultralytics/__init__.py').resolve()
    if actual != expected:
        raise RuntimeError(f'Expected local Ultralytics {expected}, found {actual}')
    return actual


def config_path(name):
    return MODEL_DIR / CONFIGS[name]
