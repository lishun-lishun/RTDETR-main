"""Standalone module tests plus optional full Ultralytics integration checks."""
import ast
import importlib
import importlib.util
from collections import OrderedDict
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from torch import nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / 'ultralytics/nn/modules'
# Load the exact four original block definitions without importing optional zoo
# dependencies, which are absent in the local CPU development environment.
tree = ast.parse((MODULE_DIR / 'block.py').read_text(encoding='utf-8'))
selected = [node for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in
            {'get_activation', 'ConvNormLayer', 'BasicBlock', 'Blocks'}]
namespace = {'nn': nn, 'torch': torch, 'OrderedDict': OrderedDict}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(MODULE_DIR / 'block.py'), 'exec'), namespace)
package = types.ModuleType('_sarep_spasff_test')
package.__path__ = [str(MODULE_DIR)]
sys.modules[package.__name__] = package
block = types.ModuleType(package.__name__ + '.block')
for name in ('BasicBlock', 'Blocks'):
    setattr(block, name, namespace[name])
sys.modules[block.__name__] = block
modules = importlib.import_module(package.__name__ + '.sarep_spasff')


class SARepSPASFFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_sarep_forward_equivalence_groups_and_state_dict(self):
        for groups in (1, 4):
            layer = modules.SARepConv(8, 8, groups=groups).eval()
            # Nontrivial BN running statistics and affine parameters.
            for branch in (layer.branch_3x3, layer.branch_1x3, layer.branch_3x1):
                branch[1].running_mean.copy_(torch.randn(8))
                branch[1].running_var.copy_(torch.rand(8) + 0.3)
            x = torch.randn(2, 8, 17, 19)
            with torch.no_grad():
                before = layer(x)
                layer.switch_to_deploy()
                after = layer(x)
            self.assertEqual(len([m for m in layer.modules() if isinstance(m, nn.Conv2d)]), 1)
            self.assertLess((before - after).abs().max().item(), 1e-4)
            self.assertLess((before - after).abs().mean().item(), 1e-5)
            self.assertLess(((before - after).norm() / before.norm()).item(), 1e-5)
            loaded = modules.SARepConv(8, 8, groups=groups, deploy=True).eval()
            loaded.load_state_dict(layer.state_dict(), strict=True)
            torch.testing.assert_close(loaded(x), after)

    def test_sarep_replacement_shortcut_and_gradients(self):
        stage = modules.SARepBlocks(64, 128, 2, 3, replace_indices=(1,))
        self.assertIsInstance(stage.blocks[0].branch2b, namespace['ConvNormLayer'])
        self.assertIsInstance(stage.blocks[1].branch2b, modules.SARepConv)
        self.assertTrue(stage.blocks[1].shortcut)
        self.assertFalse(stage.blocks[0].shortcut)
        x = torch.randn(1, 64, 16, 16, requires_grad=True)
        y = stage(x)
        self.assertEqual(y.shape, (1, 128, 8, 8))
        second_input = stage.blocks[0](x)
        second = stage.blocks[1]
        manual = second.act(second.branch2b(second.branch2a(second_input)) + second_input)
        torch.testing.assert_close(second(second_input), manual)
        y.square().mean().backward()
        self.assertIsNotNone(stage.blocks[1].branch2b.branch_1x3[0].weight.grad)
        stage.eval()
        with torch.no_grad():
            before = stage(x.detach())
            self.assertEqual(modules.switch_sarep_to_deploy(stage), 1)
            after = stage(x.detach())
        self.assertLess((before - after).abs().max().item(), 1e-4)

    def test_spasff_shapes_identity_weights_and_two_step_gradients(self):
        for batch, size in ((1, 64), (2, 96)):
            layer = modules.SPASFF(256, 256, 256, 64)
            features = [torch.randn(batch, 256, size // d, size // d)
                        for d in (1, 2, 4)]
            layer.eval()
            with torch.no_grad():
                result = layer(features)
            torch.testing.assert_close(result, features[0], rtol=0, atol=0)
            self.assertEqual(result.shape, features[0].shape)
            self.assertTrue(torch.allclose(layer.last_weights.sum(1),
                                           torch.ones_like(layer.last_weights[:, 0]), atol=1e-6))
            layer.train()
            optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
            for _ in range(2):
                optimizer.zero_grad()
                layer(features).square().mean().backward()
                optimizer.step()
            self.assertIsNotNone(layer.project4.weight.grad)
            self.assertGreater(layer.project4.weight.grad.abs().sum().item(), 0)
            self.assertEqual(features[1].shape[-2:], (size // 2, size // 2))

    def test_yaml_contract(self):
        directory = ROOT / 'ultralytics/cfg/models/rt-detr'
        baseline = yaml.safe_load((directory / 'rtdetr-r18.yaml').read_text(encoding='utf-8'))
        for variant in ('sarep', 'sarep_both', 'sarep_stage34', 'spasff',
                        'sarep_spasff', 'asff_direct', 'spasff_no_difference'):
            graph = yaml.safe_load((directory / f'rtdetr_r18_{variant}.yaml').read_text(encoding='utf-8'))
            layers = graph['backbone'] + graph['head']
            changed = {index for index, (old, new) in enumerate(zip(
                baseline['backbone'], graph['backbone'])) if old != new}
            expected = ({5, 6} if variant == 'sarep_stage34' else {5}
                        if variant in ('sarep', 'sarep_both', 'sarep_spasff') else set())
            self.assertEqual(changed, expected)
            self.assertEqual(graph['head'][:18], baseline['head'][:18])
            self.assertEqual(layers[-1][2], 'RTDETRDecoder')
            self.assertEqual(layers[-1][3][2], 300)
            self.assertEqual(layers[-1][0], [26, 22, 25] if len(layers) == 28
                             else [19, 22, 25])
            if expected:
                self.assertEqual(layers[5][2], 'SARepBlocks')

    def test_size_group_letterbox_scale(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        from evaluate_r18_sarep_spasff import group_name
        image = {'width': 1280, 'height': 720}
        self.assertEqual(group_name({'bbox': [0, 0, 16, 4]}, image), 'Tiny')
        self.assertEqual(group_name({'bbox': [0, 0, 20, 4]}, image), 'Small')

    def test_size_group_letterbox_scale(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        from evaluate_r18_sarep_spasff import group_name
        image = {'width': 1280, 'height': 720}
        self.assertEqual(group_name({'bbox': [0, 0, 16, 4]}, image), 'Tiny')
        self.assertEqual(group_name({'bbox': [0, 0, 20, 4]}, image), 'Small')

    def test_full_models_and_decoder_inputs(self):
        try:
            import ultralytics
            from ultralytics import RTDETR
        except (ImportError, OSError) as exc:
            self.skipTest(f'full zoo dependencies unavailable: {exc}')
        self.assertEqual(Path(ultralytics.__file__).resolve().parent,
                         ROOT / 'ultralytics')
        for variant in ('baseline', 'sarep', 'spasff', 'sarep_spasff'):
            filename = 'rtdetr-r18.yaml' if variant == 'baseline' else f'rtdetr_r18_{variant}.yaml'
            config = ROOT / 'ultralytics/cfg/models/rt-detr' / filename
            model = RTDETR(str(config)).model.eval()
            incoming = []
            handle = model.model[-1].register_forward_pre_hook(
                lambda _module, args: incoming.extend(tuple(x.shape) for x in args[0]))
            with torch.inference_mode():
                model(torch.zeros(1, 3, 640, 640))
            handle.remove()
            self.assertEqual(incoming, [(1, 256, 80, 80), (1, 256, 40, 40),
                                        (1, 256, 20, 20)])

    def test_full_deploy_checkpoint_roundtrip(self):
        try:
            from ultralytics import RTDETR
            from ultralytics.nn.modules.sarep_spasff import switch_sarep_to_deploy
        except (ImportError, OSError) as exc:
            self.skipTest(f'full zoo dependencies unavailable: {exc}')
        config = ROOT / 'ultralytics/cfg/models/rt-detr/rtdetr_r18_sarep.yaml'
        model = RTDETR(str(config)).model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.inference_mode():
            before = model(x)
            self.assertEqual(switch_sarep_to_deploy(model), 1)
            after = model(x)
        self.assertLess(((before[0].float() - after[0].float()).norm() /
                         before[0].float().norm().clamp_min(1e-8)).item(), 1e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'deploy.pt'
            torch.save({'model': model, 'train_args': {'task': 'detect'}}, path)
            restored = RTDETR(str(path)).model.eval()
            with torch.inference_mode():
                loaded = restored(x)
            self.assertLess(((before[0].float() - loaded[0].float()).norm() /
                             before[0].float().norm().clamp_min(1e-8)).item(), 1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp(self):
        layer = modules.SPASFF(256, 256, 256).cuda().train()
        rep = modules.SARepConv(32, 32).cuda().train()
        features = [torch.randn(1, 256, size, size, device='cuda')
                    for size in (80, 40, 20)]
        with torch.autocast('cuda', dtype=torch.float16):
            output = layer(features)
            rep_output = rep(torch.randn(1, 32, 80, 80, device='cuda'))
            (output.square().mean() + rep_output.square().mean()).backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(rep_output).all())


if __name__ == '__main__':
    unittest.main()
