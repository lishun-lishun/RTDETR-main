# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import *
from .conv import *
from .head import *
from .tiny_uav_decoder import TinyUAVDecoder
from .transformer import *
from .baps_rdcf import *
from .lsce_adcf import *
from .phr import *
from .scre_splf import *
from .hrnet import HRNetV2W18, HRNetFeature
from .rb_hrnet import RBHRNetA, RBHRNetB
from .laf import DynamicUpsample, IsolatedLAF
from .sarep_spasff import SARepConv, SARepBlocks, SPASFF, switch_sarep_to_deploy
from .res2basic import Res2Conv, Res2BasicBlock, Res2Blocks
from .lite_gd import GDGather, GDInject, LiteGDNeck, GDFeature
from .csp_context import ContextBlock, CSPResidualStage, CSPContextStage
from .ldw import DownsampleProjection, ResidualBlock, LateDownStage, WidthMatchedEarlyDownStage
from .mirnet_det import (MIRStem, SpatialRearrangeDownsample, StrideDownsample,
                          EfficientInvertedBlock, DualResolutionExchange,
                          TriResolutionExchange, MIRDetBackbone, FeatureSelect)
from .dla34_backbone import DLABasicBlock, DLARoot, DLATree, DLA34Backbone
from .backbone_adapter import BackboneAdapter, BACKBONE_SPECS, resolve_timm_model_name
from .pkinetv2_backbone import PKINetV2_T
from .r18_backbone_variants import (StemConvDown, BlurPool, AAStemPool, ResNetDDownsample,
                                    ResNetDBasicBlock, ResNetDBlocks, DilatedConvNorm,
                                    DilatedBasicBlock, P4DilatedBlocks, DCNConvNorm,
                                    DeformableBasicBlock, P4DeformableBlocks, PartialConv3,
                                    PConvBasicBlock, P5PConvBlocks)
from .r18_targeted_variants import (LIPDown, PhaseDown, UniformPool, UniformPhaseDown,
                                    ABSR, LCR, CoordRefine,
                                    SKResidual, SemanticDetailFeedback, SFR,
                                    LIPStage3Blocks, PhaseStage3Blocks,
                                    ABSRStage3Blocks, LCRStage3Blocks,
                                    CoordRefineStage3Blocks, SKStage4Blocks,
                                    SDFStage4Blocks, SFRStage3Blocks,
                                    AdaptiveBackgroundSmoothingResidual,
                                    LocalContrastResidual, CoordinateRefine,
                                    SparseForegroundResidual)
from .hrd_backbone import (HRStreamInit, HRUpdate3, HRInjectP3, HRUpdate4,
                           HRInjectP4, HRExchangeStage3, HRExchangeStage4,
                           HRDPResNet)

# __all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
#            'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
#            'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
#            'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
#            'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
#            'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP')

from .uav_research_decoder import UAVResearchDecoder  # Optional UAV research head
# >>> [UAV_PAPER_INNOVATIONS_BEGIN] >>>
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
    WCASFPv2,
    DMBASCRepC3,
    ZeroShiftSignedClutterGate,
    CAUADDecoder,
)
# <<< [UAV_PAPER_INNOVATIONS_END] <<<
# >>> [RTHRNET_W18_BEGIN] >>>
from .rt_hrnet import RTHRNetW18, RTHRNetW18Fast, RTHRNetW18Balanced, RepHRBlock, HaarP2FoldTransition
# <<< [RTHRNET_W18_END] <<<
# >>> [HRNET_LITE_SUITE_BEGIN] >>>
from .hrnet_lite_suite import (
    BaseHRNetV2W18,
    HRNetPruneS4,
    HRNetPConvRep,
    HRNetStarRep,
    FastHRNetBackbone,
    StarHRFastBackbone,
    StarHRBalancedBackbone,
    StarHRTurboBackbone,
    SlimGhostRepC3,
    RepGhostC3,
    HaarDownsample,
    HaarP2FoldTransition,
    PConvRepBlock,
    StarRepBlock,
    StarFastBlock,
    ECALayer,
)
# <<< [HRNET_LITE_SUITE_END] <<<
# >>> [SFGD_NECK_BEGIN] >>>
from .sfgd_neck import FreqDecoupledBridge, SoftplusZeroConcatFusion, DualFreqAIFI
# <<< [SFGD_NECK_END] <<<
# >>> [NECK_ARENA_BEGIN] >>>
from .neck_arena import (
    GatherDistributeGate,
    OrthogonalFreqFilter,
    HiLoLinearAIFI,
    StarC3,
    MultiScaleContrastModule,
)
# <<< [NECK_ARENA_END] <<<
# >>> [EXP6_SPEED_BEGIN] >>>
from .exp6_speed import HRNetStarFast
# <<< [EXP6_SPEED_END] <<<
