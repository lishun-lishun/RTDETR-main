"""Resolution budget variants of the project's HRNetV2-W18 backbone."""
import torch
from torch import nn

from .hrnet import (HRNetV2W18, _BLOCKS, _HRNET_W18, _BN_MOMENTUM,
                    _PROJECT_WEIGHT_PATH, _PRETRAINED_URL)


class _RBHRNet(HRNetV2W18):
    channels = None

    def __init__(self, pretrained=False, pretrained_path=None):
        # Reuse the exact transition, multi-resolution and fusion builders.
        nn.Module.__init__(self)
        self.out_channels = list(self.channels[1:])
        self.out_strides = [8, 16, 32]
        self.pretrained_load_report = None
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=True)
        stage1 = _HRNET_W18['STAGE1']
        self.layer1 = self._make_layer(
            _BLOCKS['BOTTLENECK'], 64, 64, stage1['NUM_BLOCKS'][0])
        previous = [256]
        for number in (2, 3, 4):
            config = dict(_HRNET_W18[f'STAGE{number}'])
            config['NUM_CHANNELS'] = self.channels[:number]
            transition, stage, previous = self._build_stage(config, previous)
            setattr(self, f'transition{number-1}', transition)
            setattr(self, f'stage{number}', stage)
        self._init_weights()
        if pretrained or pretrained_path:
            if pretrained_path:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                weight_source = str(pretrained_path)
            elif _PROJECT_WEIGHT_PATH.is_file():
                checkpoint = torch.load(_PROJECT_WEIGHT_PATH, map_location='cpu')
                weight_source = str(_PROJECT_WEIGHT_PATH)
            else:
                checkpoint = self._download_pretrained_ddp_safe()
                weight_source = _PRETRAINED_URL
            source = self._unwrap_checkpoint(checkpoint)
            own = self.state_dict()
            matched = {k: v for k, v in source.items()
                       if k in own and v.shape == own[k].shape}
            # Names retain their branch/stage meaning; changed shapes remain
            # at the standard Kaiming/BatchNorm initialization above.
            self.load_state_dict(matched, strict=False)
            missing = sorted(set(own) - set(matched))
            coverage = len(matched) / len(own)
            self.pretrained_load_report = {
                'source': weight_source, 'matched_keys': sorted(matched),
                'missing_keys': missing, 'coverage': coverage}
            print(f'{type(self).__name__}: loaded {len(matched)}/{len(own)} tensors; '
                  f'missing={missing}')
            if coverage < 0.1:
                raise RuntimeError('RB-HRNet checkpoint coverage below 10%; check source')


class RBHRNetA(_RBHRNet):
    channels = (24, 36, 72, 144)


class RBHRNetB(_RBHRNet):
    channels = (24, 36, 64, 112)
