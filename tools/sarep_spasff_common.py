"""Paths and local-package guard shared by SA-Rep/SP-ASFF tools."""
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'ultralytics/cfg/models/rt-detr'
PROFILE = ROOT / 'ultralytics/cfg/anti_detr_default.yaml'
DATA = ROOT / 'dataset/DUT-Anti-UAV.yaml'
RUNS = ROOT / 'runs/sarep_spasff'
CONFIGS = {
    'sarep': 'rtdetr_r18_sarep.yaml',
    'sarep_both': 'rtdetr_r18_sarep_both.yaml',
    'sarep_stage34': 'rtdetr_r18_sarep_stage34.yaml',
    'spasff': 'rtdetr_r18_spasff.yaml',
    'sarep_spasff': 'rtdetr_r18_sarep_spasff.yaml',
    'asff_direct': 'rtdetr_r18_asff_direct.yaml',
    'spasff_no_difference': 'rtdetr_r18_spasff_no_difference.yaml',
}


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
