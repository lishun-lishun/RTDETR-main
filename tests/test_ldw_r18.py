"""Unit tests for LDW-ResNet18 stage geometry, ordering and gradients."""

import unittest
from pathlib import Path

import torch

from ultralytics.nn.modules import LateDownStage, WidthMatchedEarlyDownStage


ROOT = Path(__file__).parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"


class TestLDWStages(unittest.TestCase):
    def test_stage4_output(self):
        stage = LateDownStage(128, 256, 128, 2, stage_num=4)
        self.assertEqual(tuple(stage(torch.randn(2, 128, 80, 80)).shape), (2, 256, 40, 40))

    def test_stage5_output(self):
        stage = LateDownStage(256, 512, 256, 2, stage_num=5)
        self.assertEqual(tuple(stage(torch.randn(2, 256, 40, 40)).shape), (2, 512, 20, 20))

    def test_early_control_has_same_geometry(self):
        stage = WidthMatchedEarlyDownStage(128, 256, 128, 2, stage_num=4)
        self.assertEqual(tuple(stage(torch.randn(2, 128, 80, 80)).shape), (2, 256, 40, 40))
        self.assertIsInstance(stage.output_projection, torch.nn.Module)

    def test_stride_order(self):
        late = LateDownStage(128, 256, 128, 2)
        early = WidthMatchedEarlyDownStage(128, 256, 128, 2)
        late_events, early_events = [], []
        handles = [
            late.blocks[0].register_forward_hook(lambda *_: late_events.append("block1")),
            late.blocks[1].register_forward_hook(lambda *_: late_events.append("block2")),
            late.late_downsample.register_forward_hook(lambda *_: late_events.append("down")),
            early.early_downsample.register_forward_hook(lambda *_: early_events.append("down")),
            early.blocks[0].register_forward_hook(lambda *_: early_events.append("block1")),
            early.blocks[1].register_forward_hook(lambda *_: early_events.append("block2")),
            early.output_projection.register_forward_hook(lambda *_: early_events.append("projection")),
        ]
        try:
            late(torch.randn(1, 128, 16, 16))
            early(torch.randn(1, 128, 16, 16))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(late_events, ["block1", "block2", "down"])
        self.assertEqual(early_events, ["down", "block1", "block2", "projection"])

    def test_full_model_backbone_and_decoder_shapes(self):
        from ultralytics.nn.tasks import RTDETRDetectionModel

        model = RTDETRDetectionModel(str(MODEL_DIR / "rtdetr_r18_ldw_full.yaml"), nc=3, verbose=False).eval()
        seen = {}
        handles = [model.model[i].register_forward_hook(
            lambda _m, _i, output, index=i: seen.__setitem__(index, tuple(output.shape)))
                   for i in (5, 6, 7)]
        with torch.inference_mode():
            model(torch.zeros(1, 3, 128, 128))
        for handle in handles:
            handle.remove()
        self.assertEqual(seen, {5: (1, 128, 16, 16), 6: (1, 256, 8, 8), 7: (1, 512, 4, 4)})
        self.assertEqual(model.model[-1].f, [19, 22, 25])

    def test_gradients_flow(self):
        stage = LateDownStage(128, 256, 128, 2)
        output = stage(torch.randn(1, 128, 16, 16)).square().mean()
        output.backward()
        self.assertIsNotNone(stage.blocks[0].branch2a.conv.weight.grad)
        self.assertIsNotNone(stage.late_downsample.conv.weight.grad)


if __name__ == "__main__":
    unittest.main()
