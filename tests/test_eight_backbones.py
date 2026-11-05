"""Preflight, routing, gradient and checkpoint tests for the eight candidates."""

import tempfile
import unittest
from pathlib import Path

import torch

from ultralytics.nn.modules import BACKBONE_SPECS, BackboneAdapter, FeatureSelect


ROOT = Path(__file__).parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
CONFIGS = {
    "hgnetv2_b2": "rtdetr_hgnetv2_b2.yaml",
    "convnextv2_pico": "rtdetr_convnextv2_pico.yaml",
    "fasternet_t1": "rtdetr_fasternet_t1.yaml",
    "repvit_m1_1": "rtdetr_repvit_m11.yaml",
    "efficientvit_b1": "rtdetr_efficientvit_b1.yaml",
    "pvt_v2_b1": "rtdetr_pvtv2_b1.yaml",
    "lsknet_t": "rtdetr_lsknet_t.yaml",
    "pkinetv2_t": "rtdetr_pkinetv2_t.yaml",
}


def tensor_values(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(tensor_values(item))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(tensor_values(item))
        return result
    return []


class TestEightBackbones(unittest.TestCase):
    def test_registry_and_yaml_inventory(self):
        self.assertEqual(set(BACKBONE_SPECS), set(CONFIGS))
        self.assertFalse(hasattr(BackboneAdapter, "backbone"))
        for filename in CONFIGS.values():
            self.assertTrue((MODEL_DIR / filename).is_file(), filename)

    def test_each_adapter_has_required_shapes_at_640_and_batch2(self):
        for name in CONFIGS:
            model = BackboneAdapter(name).eval()
            with torch.inference_mode():
                outputs = model(torch.zeros(1, 3, 640, 640))
                outputs_small = model(torch.zeros(2, 3, 128, 128))
            self.assertEqual([tuple(item.shape) for item in outputs],
                             [(1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20)], name)
            self.assertEqual([tuple(item.shape) for item in outputs_small],
                             [(2, 128, 16, 16), (2, 256, 8, 8), (2, 512, 4, 4)], name)

    def test_full_models_route_distinct_features_to_decoder(self):
        from ultralytics.nn.tasks import RTDETRDetectionModel

        for name, filename in CONFIGS.items():
            model = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False).eval()
            selectors = [module for module in model.model if isinstance(module, FeatureSelect)]
            self.assertEqual([selector.index for selector in selectors], [0, 1, 2], name)
            seen, decoder = {}, {}
            handles = [selector.register_forward_hook(
                lambda _m, inputs, output, index=index: seen.__setitem__(index, {
                    "input": [tuple(value.shape) for value in inputs[0]],
                    "output": tuple(output.shape),
                })) for index, selector in enumerate(selectors)]
            handles.append(model.model[-1].register_forward_pre_hook(
                lambda _m, values: decoder.__setitem__("shapes", [tuple(value.shape) for value in values[0]])))
            try:
                with torch.inference_mode():
                    model(torch.zeros(1, 3, 128, 128))
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(seen[0]["input"], [(1, 128, 16, 16), (1, 256, 8, 8), (1, 512, 4, 4)], name)
            self.assertEqual(decoder["shapes"], [(1, 256, 16, 16), (1, 256, 8, 8), (1, 256, 4, 4)], name)
            self.assertEqual(model.model[-1].f, [15, 18, 21], name)

    def test_full_models_decoder_shapes_at_640(self):
        from ultralytics.nn.tasks import RTDETRDetectionModel

        for name, filename in CONFIGS.items():
            model = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False).eval()
            decoder = {}
            handle = model.model[-1].register_forward_pre_hook(
                lambda _m, values: decoder.__setitem__("shapes", [tuple(value.shape) for value in values[0]]))
            try:
                with torch.inference_mode():
                    model(torch.zeros(1, 3, 640, 640))
            finally:
                handle.remove()
            self.assertEqual(decoder["shapes"], [(1, 256, 80, 80), (1, 256, 40, 40), (1, 256, 20, 20)], name)

    def test_full_model_prediction_backward_reaches_adapter(self):
        from ultralytics.nn.tasks import RTDETRDetectionModel

        for name, filename in CONFIGS.items():
            model = RTDETRDetectionModel(str(MODEL_DIR / filename), nc=1, verbose=False).eval()
            adapter = next(module for module in model.modules() if isinstance(module, BackboneAdapter))
            values = tensor_values(model.predict(torch.randn(2, 3, 128, 128)))
            self.assertTrue(values, name)
            sum(item.float().square().mean() for item in values).backward()
            self.assertTrue(any(parameter.grad is not None for parameter in adapter.parameters()), name)

    def test_each_adapter_backward_and_optimizer_step(self):
        for name in CONFIGS:
            model = BackboneAdapter(name).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            output = model(torch.randn(2, 3, 128, 128))
            loss = sum(item.square().mean() for item in output)
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            self.assertTrue(any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients), name)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            optimizer.step()
            self.assertTrue(any(not torch.equal(left, right) for left, right in zip(before, model.parameters())), name)

    def test_checkpoint_round_trip_for_each_adapter(self):
        for name in CONFIGS:
            model = BackboneAdapter(name).eval()
            sample = torch.randn(1, 3, 128, 128)
            with torch.inference_mode():
                expected = model(sample)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"{name}.pt"
                torch.save(model.state_dict(), path)
                restored = BackboneAdapter(name).eval()
                restored.load_state_dict(torch.load(path, map_location="cpu"))
                with torch.inference_mode():
                    actual = restored(sample)
            for left, right in zip(expected, actual):
                self.assertTrue(torch.equal(left, right), name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_and_amp_adapter_forward(self):
        for name in CONFIGS:
            model = BackboneAdapter(name).cuda().eval()
            sample = torch.randn(1, 3, 128, 128, device="cuda")
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(sample)
            self.assertEqual([tuple(item.shape) for item in outputs],
                             [(1, 128, 16, 16), (1, 256, 8, 8), (1, 512, 4, 4)], name)


if __name__ == "__main__":
    unittest.main()
