# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.201'

import sys
try:
    import timm
    if hasattr(timm, "models") and hasattr(timm.models, "layers"):
        if not hasattr(timm.models.layers, "use_fused_attn"):
            timm.models.layers.use_fused_attn = lambda experimental=False: True
    if not hasattr(timm, "layers") and hasattr(timm, "models") and hasattr(timm.models, "layers"):
        timm.layers = timm.models.layers
        sys.modules["timm.layers"] = timm.models.layers
    elif hasattr(timm, "layers") and not hasattr(timm.layers, "use_fused_attn"):
        timm.layers.use_fused_attn = lambda experimental=False: True
except ImportError:
    pass

from ultralytics.models import RTDETR, SAM, YOLO
from ultralytics.models.fastsam import FastSAM
from ultralytics.models.nas import NAS
from ultralytics.utils import SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

__all__ = '__version__', 'YOLO', 'NAS', 'SAM', 'FastSAM', 'RTDETR', 'checks', 'download', 'settings'
