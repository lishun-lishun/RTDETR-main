"""Structural, shape, gradient and checkpoint tests for E0/E1-R08 R18 models."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
import yaml

from ultralytics.nn.modules import (AAStemPool, BlurPool, DeformableBasicBlock,
                                    DilatedBasicBlock, PConvBasicBlock, PartialConv3,
                                    ResNetDBasicBlock, ResNetDDownsample, StemConvDown)

ROOT = Path(__file__).parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
CONFIGS = {
    "baseline": "rtdetr-r18.yaml",
    "r18_stemconv": "rtdetr_r18_stemconv.yaml",
    "r18_aastem": "rtdetr_r18_aastem.yaml",
    "r18_resnetd": "rtdetr_r18_resnetd.yaml",
    "r18_p3deep": "rtdetr_r18_p3deep.yaml",
    "r18_p3wide": "rtdetr_r18_p3wide.yaml",
    "r18_p4dilated": "rtdetr_r18_p4dilated.yaml",
    "r18_p4dcn": "rtdetr_r18_p4dcn.yaml",
    "r18_p5pconv": "rtdetr_r18_p5pconv.yaml",
}


def tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [item for v in value.values() for item in tensors(v)]
    if isinstance(value, (tuple, list)):
        return [item for v in value for item in tensors(v)]
    return []


class TestR18EightVariants(unittest.TestCase):
    def test_yaml_inventory_and_unchanged_head(self):
        baseline = yaml.safe_load((MODEL_DIR / CONFIGS["baseline"]).read_text(encoding="utf-8"))
        for name, filename in CONFIGS.items():
            path = MODEL_DIR / filename
            self.assertTrue(path.is_file(), filename)
            graph = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(graph["head"], baseline["head"], name)
            self.assertEqual(graph["head"][-1][0], [19, 22, 25], name)
            self.assertEqual(len(graph["backbone"]), 8, name)

    def test_specialized_modules(self):
        self.assertEqual(StemConvDown(64)(torch.zeros(1, 64, 320, 320)).shape[-2:], (160, 160))
        aa = AAStemPool(64)
        self.assertIn(aa.maxpool.stride, (1, (1, 1)))
        self.assertEqual(aa.blurpool.stride, 2)
        self.assertTrue(torch.equal(aa.blurpool.kernel[0, 0].sum(), torch.tensor(1.0)))
        self.assertIsInstance(ResNetDBasicBlock(128, 256, 2, False).short, ResNetDDownsample)
        self.assertEqual(DilatedBasicBlock(256, 256, 1, True).branch2b.conv.dilation, (2, 2))
        self.assertEqual(PartialConv3(512, 512).partial_channels, 128)
        self.assertIsInstance(PConvBasicBlock(512, 512, 1, True).branch2a, PartialConv3)
        if DeformableBasicBlock is not None:
            try:
                block = DeformableBasicBlock(256, 256, 1, True)
            except RuntimeError as exc:
                self.fail(f"R07 requires a working torchvision DeformConv2d operator: {exc}")
            else:
                self.assertEqual(block.branch2b.offset.out_channels, 18)

    def test_full_model_shapes_and_decoder_routes(self):
        from ultralytics.nn.tasks import RTDETRDetectionModel

        for name, filename in CONFIGS.items():
            if name == "r18_p4dcn":
                try:
                    model = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False).eval()
                except RuntimeError as exc:
                    self.fail(f"R07 requires a working torchvision DeformConv2d operator: {exc}")
            else:
                model = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False).eval()
            found, decoder = {}, {}
            handles = [model.model[i].register_forward_hook(
                lambda _m, _i, output, key=key: found.__setitem__(key, tuple(output.shape)))
                       for i, key in ((5, "p3"), (6, "p4"), (7, "p5"))]
            handles.append(model.model[-1].register_forward_pre_hook(
                lambda _m, values: decoder.__setitem__("shapes", [tuple(v.shape) for v in values[0]])))
            try:
                with torch.inference_mode():
                    model(torch.zeros(1, 3, 640, 640))
            finally:
                for handle in handles:
                    handle.remove()
            p3 = 160 if name == "r18_p3wide" else 128
            self.assertEqual([found[k] for k in ("p3", "p4", "p5")],
                             [(1, p3, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20)], name)
            self.assertEqual(decoder["shapes"], [(1, 256, 80, 80), (1, 256, 40, 40), (1, 256, 20, 20)], name)
            self.assertEqual(model.model[-1].f, [19, 22, 25], name)

    def test_batch_legal_size_and_backward(self):
        from ultralytics.nn.tasks import RTDETRDetectionModel

        for name, filename in CONFIGS.items():
            if name == "r18_p4dcn":
                try:
                    # Use eval-mode prediction with gradients enabled.  Train-mode
                    # RT-DETRDecoder requires a labeled batch for CDN loss inputs.
                    model = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False).eval()
                except RuntimeError as exc:
                    self.fail(f"R07 requires a working torchvision DeformConv2d operator: {exc}")
            else:
                model = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False).eval()
            outputs = model(torch.randn(2, 3, 128, 128))
            values = tensors(outputs)
            self.assertTrue(values, name)
            sum(item.float().square().mean() for item in values).backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()), name)

    def test_checkpoint_round_trip(self):
        from ultralytics.nn.tasks import RTDETRDetectionModel

        for name, filename in CONFIGS.items():
            if name == "r18_p4dcn":
                try:
                    model = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False).eval()
                except RuntimeError as exc:
                    self.fail(f"R07 requires a working torchvision DeformConv2d operator: {exc}")
            else:
                model = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False).eval()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"{name}.pt"
                torch.save(model.state_dict(), path)
                restored = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False).eval()
                restored.load_state_dict(torch.load(path, map_location="cpu"))
                self.assertEqual(set(model.state_dict()), set(restored.state_dict()), name)


if __name__ == "__main__":
    unittest.main()
