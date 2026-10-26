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

# __all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
#            'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
#            'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
#            'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
#            'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
#            'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP')
