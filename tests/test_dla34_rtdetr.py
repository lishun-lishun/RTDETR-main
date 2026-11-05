"""Structural, routing and checkpoint tests for the DLA34 RT-DETR variant."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from ultralytics.nn.modules import DLABasicBlock, DLA34Backbone, DLARoot, DLATree, FeatureSelect


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / "rtdetr_dla34.yaml"


class TestDLA34RTDETR(unittest.TestCase):
    def test_dla34_outputs_and_strides(self):
        model = DLA34Backbone().eval()
        with torch.inference_mode():
            p3, p4, p5 = model(torch.randn(1, 3, 640, 640))
        self.assertEqual(tuple(p3.shape), (1, 128, 80, 80))
        self.assertEqual(tuple(p4.shape), (1, 256, 40, 40))
        self.assertEqual(tuple(p5.shape), (1, 512, 20, 20))
        self.assertEqual(model.out_channels, (128, 256, 512))
        self.assertEqual(model.out_strides, (8, 16, 32))

    def test_official_dla34_structure(self):
        model = DLA34Backbone()
        self.assertEqual(model.levels, (1, 1, 1, 2, 2, 1))
        self.assertEqual(model.channels, (16, 32, 64, 128, 256, 512))
        self.assertEqual(model.base_layer[0].kernel_size, (7, 7))
        self.assertEqual(model.base_layer[0].stride, (1, 1))
        for index, expected in ((2, 1), (3, 2), (4, 2), (5, 1)):
            tree = getattr(model, f"level{index}")
            self.assertIsInstance(tree, DLATree)
            self.assertEqual(tree.levels, expected)
            self.assertIsNotNone(tree.downsample)
        self.assertFalse(model.level2.level_root)
        self.assertTrue(model.level3.level_root)
        self.assertTrue(model.level4.level_root)
        self.assertTrue(model.level5.level_root)
        leaf = model.level3.tree1
        while leaf.levels > 1:
            leaf = leaf.tree1
        self.assertIsInstance(leaf.root, DLARoot)
        self.assertIsInstance(leaf.tree1, DLABasicBlock)
        self.assertIsInstance(leaf.tree2, DLABasicBlock)
        self.assertEqual(leaf.root.conv.in_channels, leaf.root_dim)

    def test_nonstandard_size_and_three_distinct_outputs(self):
        model = DLA34Backbone().eval()
        with torch.inference_mode():
            outputs = model(torch.randn(2, 3, 384, 512))
        self.assertEqual([tuple(item.shape[-2:]) for item in outputs], [(48, 64), (24, 32), (12, 16)])
        self.assertIsNot(outputs[0], outputs[1])
        self.assertIsNot(outputs[1], outputs[2])

    def test_tree_root_and_level_gradients(self):
        model = DLA34Backbone()
        outputs = model(torch.randn(1, 3, 128, 128))
        sum(item.square().mean() for item in outputs).backward()
        for name in ("level2", "level3", "level4", "level5"):
            tree = getattr(model, name)
            self.assertTrue(any(parameter.grad is not None for parameter in tree.parameters()), name)
        roots = [module for module in model.modules() if isinstance(module, DLARoot)]
        self.assertTrue(roots)
        self.assertTrue(all(module.conv.weight.grad is not None for module in roots))

    def test_deepcopy_after_forward(self):
        model = DLA34Backbone()
        model(torch.randn(1, 3, 128, 128))
        self.assertIsInstance(copy.deepcopy(model), DLA34Backbone)

    def test_full_rtdetr_routes_level3_level4_level5(self):
        from ultralytics.nn.tasks import RTDETRDetectionModel

        model = RTDETRDetectionModel(str(CONFIG), nc=1, verbose=False).eval()
        selectors = [module for module in model.model if isinstance(module, FeatureSelect)]
        self.assertEqual([selector.index for selector in selectors], [0, 1, 2])
        seen = {}
        handles = [selector.register_forward_hook(
            lambda _m, inputs, output, index=index: seen.__setitem__(index, {
                "input": [tuple(value.shape) for value in inputs[0]],
                "output": tuple(output.shape),
            })) for index, selector in enumerate(selectors)]
        decoder = {}
        handles.append(model.model[-1].register_forward_pre_hook(
            lambda _m, values: decoder.__setitem__("shapes", [tuple(value.shape) for value in values[0]])))
        try:
            with torch.inference_mode():
                model(torch.zeros(1, 3, 128, 128))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(seen[0]["input"], [(1, 128, 16, 16), (1, 256, 8, 8), (1, 512, 4, 4)])
        self.assertEqual([seen[index]["output"] for index in (0, 1, 2)],
                         [(1, 128, 16, 16), (1, 256, 8, 8), (1, 512, 4, 4)])
        self.assertEqual(decoder["shapes"], [(1, 256, 16, 16), (1, 256, 8, 8), (1, 256, 4, 4)])
        self.assertEqual(model.model[-1].f, [15, 18, 21])

    def test_checkpoint_round_trip(self):
        model = DLA34Backbone().eval()
        sample = torch.randn(1, 3, 128, 128)
        with torch.inference_mode():
            expected = model(sample)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dla34.pt"
            torch.save({"state_dict": model.state_dict()}, path)
            restored = DLA34Backbone(pretrained=True, pretrained_path=path).eval()
            self.assertIsNotNone(restored.pretrained_load_report)
            self.assertAlmostEqual(restored.pretrained_load_report["load_ratio"], 1.0)
            with torch.inference_mode():
                actual = restored(sample)
        for left, right in zip(expected, actual):
            self.assertTrue(torch.equal(left, right))


if __name__ == "__main__":
    unittest.main()
