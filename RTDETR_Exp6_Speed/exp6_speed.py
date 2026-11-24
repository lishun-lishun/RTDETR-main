# -*- coding: utf-8 -*-
"""
HRNetStarFast Backbone (Exp 26 - Exp 29):
Locks exact Exp6 (`HRNetStarRep`) golden depth (`Stage1=4, Stage3=4x3, Stage4=2x3`)
while swapping fragmented `StarRepBlock` (`split/cat/ECA`) -> hardware-aware `StarFastBlock`.
"""

from __future__ import annotations
from typing import Optional

try:
    from .hrnet_lite_suite import BaseHRNetV2W18
except ImportError:
    from hrnet_lite_suite import BaseHRNetV2W18

__all__ = ["HRNetStarFast"]


class HRNetStarFast(BaseHRNetV2W18):
    """Exact Exp6 depth (`Stage1=4, Stage3=4x3, Stage4=2x3`) with zero-fragment `StarFastBlock`."""

    def __init__(self, pretrained: bool = True, pretrained_path: Optional[str] = None):
        super().__init__(
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            prune_stage4_p2=True,
            sparse_fuse=True,
            lite_block_type="star_fast",
            lite_min_channels=72,
            stage1_blocks=4,
            stage3_modules=4,
            stage4_modules=2,
            blocks_per_branch=3,
        )
