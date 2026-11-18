# -*- coding: utf-8 -*-
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
