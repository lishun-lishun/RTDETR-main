"""Structural and gradient tests for MIRNet-Det."""

import tempfile
import unittest
from pathlib import Path

import torch

from ultralytics.nn.modules import (
    DualResolutionExchange,
    EfficientInvertedBlock,
    FeatureSelect,
    MIRDetBackbone,
    MIRStem,
    SpatialRearrangeDownsample,
    StrideDownsample,
    TriResolutionExchange,
)


ROOT = Path(__file__).parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
CONFIGS = ("rtdetr_mirnet_base.yaml", "rtdetr_mirnet_dual.yaml",
           "rtdetr_mirnet_full.yaml", "rtdetr_mirnet_stride.yaml")
DIAGNOSTIC_CONFIGS = ("rtdetr_mirnet_nofusion.yaml", "rtdetr_mirnet_zero_fusion.yaml")


class TestMIRNetDet(unittest.TestCase):
    def test_stem(self):
        self.assertEqual(tuple(MIRStem()(torch.randn(2, 3, 640, 640)).shape), (2, 48, 160, 160))

    def test_spatial_rearrange_and_stride_controls(self):
        x = torch.randn(2, 48, 160, 160)
        self.assertEqual(tuple(SpatialRearrangeDownsample(48, 128)(x).shape), (2, 128, 80, 80))
        self.assertEqual(tuple(StrideDownsample(48, 128)(x).shape), (2, 128, 80, 80))
        x.requires_grad_()
        SpatialRearrangeDownsample(48, 128)(x).mean().backward()
        self.assertIsNotNone(x.grad)

    def test_efficient_inverted_block_widths(self):
        for channels, size in ((128, 80), (224, 40), (384, 20)):
            block = EfficientInvertedBlock(channels)
            output = block(torch.randn(2, channels, size, size))
            self.assertEqual(tuple(output.shape), (2, channels, size, size))
            self.assertEqual(block.depthwise.conv.groups, block.hidden_channels)

    def test_dual_resolution_exchange_is_synchronous_and_trainable(self):
        module = DualResolutionExchange()
        s8 = torch.randn(2, 128, 80, 80, requires_grad=True)
        s16 = torch.randn(2, 224, 40, 40, requires_grad=True)
        out8, out16 = module(s8, s16)
        self.assertEqual(tuple(out8.shape), tuple(s8.shape))
        self.assertEqual(tuple(out16.shape), tuple(s16.shape))
        (out8.mean() + out16.mean()).backward()
        self.assertIsNotNone(module.s16_to_s8.conv.conv.weight.grad)
        self.assertIsNotNone(module.s8_to_s16.conv.conv.weight.grad)
        self.assertIsNotNone(module.block8.expand.conv.weight.grad)
        self.assertIsNotNone(module.block16.expand.conv.weight.grad)

    def test_tri_resolution_exchange_paths(self):
        module = TriResolutionExchange()
        inputs = (torch.randn(1, 128, 80, 80, requires_grad=True),
                  torch.randn(1, 224, 40, 40, requires_grad=True),
                  torch.randn(1, 384, 20, 20, requires_grad=True))
        outputs = module(*inputs)
        self.assertEqual([tuple(x.shape) for x in outputs], [tuple(x.shape) for x in inputs])
        sum(x.square().mean() for x in outputs).backward()
        for name in ("s16_to_s8", "s8_to_s16", "s32_to_s16", "s16_to_s32"):
            self.assertIsNotNone(getattr(module, name).conv.conv.weight.grad, name)
        for block in (module.block8, module.block16, module.block32):
            self.assertIsNotNone(block.expand.conv.weight.grad)

    def test_backbone_outputs(self):
        model = MIRDetBackbone()
        outputs = model(torch.randn(2, 3, 640, 640))
        self.assertEqual([tuple(x.shape) for x in outputs],
                         [(2, 128, 80, 80), (2, 256, 40, 40), (2, 512, 20, 20)])
        self.assertIsNotNone(model.last_states)

    def test_backbone_accepts_legal_nonstandard_size(self):
        outputs = MIRDetBackbone()(torch.randn(1, 3, 384, 512))
        self.assertEqual([tuple(x.shape[-2:]) for x in outputs], [(48, 64), (24, 32), (12, 16)])

    def test_full_rtdetr_routes_three_distinct_features(self):
        from ultralytics.nn.tasks import RTDETRDetectionModel

        model = RTDETRDetectionModel(str(MODEL_DIR / "rtdetr_mirnet_full.yaml"), nc=1, verbose=False).eval()
        seen = {}
        hooks = [model.model[index].register_forward_hook(
            lambda _m, inputs, output, index=index: seen.__setitem__(index, {
                "input": tuple(inputs[0].shape), "output": tuple(output.shape)}))
                 for index in (1, 2, 3)]
        decoder_inputs = {}
        hooks.append(model.model[-1].register_forward_pre_hook(
            lambda _m, values: decoder_inputs.__setitem__(
                "shapes", [tuple(x.shape) for x in values[0]])))
        try:
            with torch.inference_mode():
                model(torch.zeros(1, 3, 128, 128))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual([seen[i]["input"] for i in (1, 2, 3)],
                         [(1, 128, 16, 16), (1, 256, 8, 8), (1, 512, 4, 4)])
        self.assertEqual(decoder_inputs["shapes"], [(1, 256, 16, 16), (1, 256, 8, 8), (1, 256, 4, 4)])
        self.assertEqual(model.model[-1].f, [15, 18, 21])

    def test_all_configurations_construct(self):
        from ultralytics.nn.tasks import RTDETRDetectionModel

        for filename in CONFIGS + DIAGNOSTIC_CONFIGS:
            model = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False).eval()
            with torch.inference_mode():
                outputs = model(torch.zeros(1, 3, 128, 128))
            self.assertIsNotNone(outputs, filename)
            state = model.state_dict()
            clone = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False)
            clone.load_state_dict(state)

    def test_feature_select_rejects_bad_sources(self):
        with self.assertRaises(ValueError):
            FeatureSelect(0)(torch.randn(1, 4, 4, 4))


if __name__ == "__main__":
    unittest.main()
