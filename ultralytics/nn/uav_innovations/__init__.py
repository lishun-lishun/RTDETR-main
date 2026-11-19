# -*- coding: utf-8 -*-
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
