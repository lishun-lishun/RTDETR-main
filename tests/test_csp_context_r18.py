"""CSP R18 unit checks; complete-model and CUDA checks run on the server."""
import ast
from collections import OrderedDict
import importlib
import io
from pathlib import Path
import sys
import types
import unittest

import torch
from torch import nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / 'ultralytics/nn/modules'
tree = ast.parse((MODULE_DIR / 'block.py').read_text(encoding='utf-8'))
selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in {'get_activation', 'ConvNormLayer', 'BasicBlock'}]
namespace = {'nn': nn, 'torch': torch, 'OrderedDict': OrderedDict}
exec(compile(ast.Module(body=selected, type_ignores=[]),
             str(MODULE_DIR / 'block.py'), 'exec'), namespace)
package = types.ModuleType('_csp_context_test')
package.__path__ = [str(MODULE_DIR)]
sys.modules[package.__name__] = package
block = types.ModuleType(package.__name__ + '.block')
for name in ('ConvNormLayer', 'BasicBlock'):
    setattr(block, name, namespace[name])
sys.modules[block.__name__] = block
csp = importlib.import_module(package.__name__ + '.csp_context')


class CSPContextR18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_stage_shapes_split_and_original_downsample(self):
        for stage, c1, c2, size in ((csp.CSPResidualStage, 64, 128, 160),
                                    (csp.CSPContextStage, 128, 256, 80)):
            module = stage(c1, c2).eval()
            self.assertEqual((module.process_channels, module.bypass_channels),
                             (round(c2 * .75), c2 - round(c2 * .75)))
            original = namespace['BasicBlock'](c1, c2, 2, False, 'relu', 'd').eval()
            original.load_state_dict(module.downsample.state_dict())
            x = torch.randn(2, c1, size, size)
            with torch.inference_mode():
                out = module(x)
                z = original(x)
                torch.testing.assert_close(out, z + module.gamma * module.last_features['fusion'])
            self.assertEqual(tuple(out.shape), (2, c2, size // 2, size // 2))
            torch.testing.assert_close(module.last_features['downsample'], z)
            self.assertEqual(module.last_features['processed'].shape[1],
                             module.process_channels)
            self.assertEqual(module.last_features['bypass'].shape[1],
                             module.bypass_channels)
            self.assertAlmostEqual(module.gamma.item(), .05, places=6)
            self.assertTrue(module.downsample.shortcut is False)
            self.assertIsInstance(module.downsample.short[0], nn.AvgPool2d)

    def test_context_kernels_ratios_and_validation(self):
        for kernel in (3, 5, 7):
            block = csp.ContextBlock(24, kernel)
            self.assertEqual(block.dwconv.groups, 24)
            self.assertEqual(block.dwconv.kernel_size, (kernel, kernel))
            self.assertEqual(block.expand.out_channels, 48)
            self.assertEqual(block(torch.randn(2, 24, 9, 11)).shape,
                             (2, 24, 9, 11))
        stage = csp.CSPResidualStage(32, 65, process_ratio=.5)
        self.assertEqual(stage.process_channels + stage.bypass_channels, 65)
        self.assertGreater(stage.process_channels, 0)
        self.assertGreater(stage.bypass_channels, 0)
        with self.assertRaises(ValueError):
            csp.CSPContextStage(128, 256, kernel_size=9)
        with self.assertRaises(ValueError):
            csp.CSPResidualStage(64, 128, process_ratio=1)

    def test_gradient_train_eval_and_state_roundtrip(self):
        for stage, c1, c2 in ((csp.CSPResidualStage, 64, 128),
                              (csp.CSPContextStage, 128, 256)):
            module = stage(c1, c2).train()
            optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)
            x = torch.randn(2, c1, 20, 20)
            for _ in range(2):
                optimizer.zero_grad()
                output = module(x)
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertIsNotNone(module.raw_gamma.grad)
                self.assertGreater(module.raw_gamma.grad.abs().sum().item(), 0)
                self.assertGreater(module.split_projection.conv.weight.grad.abs().sum().item(), 0)
                self.assertGreater(module.split_projection.conv.weight.grad[
                    module.process_channels:].abs().sum().item(), 0)
                self.assertGreater(module.fusion.conv.weight.grad.abs().sum().item(), 0)
                self.assertGreater(sum(p.grad.abs().sum().item() for p in
                                       module.process.parameters() if p.grad is not None), 0)
                optimizer.step()
            state = {key: value.clone() for key, value in module.state_dict().items()}
            restored = stage(c1, c2).eval()
            restored.load_state_dict(state)
            module.eval()
            with torch.inference_mode():
                torch.testing.assert_close(restored(x), module(x))
            self.assertTrue(any('running_mean' in key for key in state))

    def test_yaml_head_and_topology(self):
        directory = ROOT / 'ultralytics/cfg/models/rt-detr'
        base = yaml.safe_load((directory / 'rtdetr-r18.yaml').read_text(encoding='utf-8'))
        from_map = {'stage3': (True, False), 'stage4': (False, True),
                    'combined': (True, True), 'stage4_k3': (False, True),
                    'stage3_ratio05': (True, False)}
        filenames = {'stage3': 'rtdetr_r18_csp_stage3.yaml',
                     'stage4': 'rtdetr_r18_csp_stage4.yaml',
                     'combined': 'rtdetr_r18_csp_context.yaml',
                     'stage4_k3': 'rtdetr_r18_csp_stage4_k3.yaml',
                     'stage3_ratio05': 'rtdetr_r18_csp_stage3_ratio05.yaml'}
        for name, (has_stage3, has_stage4) in from_map.items():
            model = yaml.safe_load((directory / filenames[name]).read_text(encoding='utf-8'))
            self.assertEqual(model['head'], base['head'])
            self.assertEqual(len(model['backbone']), 8)
            self.assertEqual(model['backbone'][:5], base['backbone'][:5])
            self.assertEqual(model['backbone'][7], base['backbone'][7])
            self.assertEqual(model['backbone'][5][2] == 'CSPResidualStage', has_stage3)
            self.assertEqual(model['backbone'][6][2] == 'CSPContextStage', has_stage4)
            self.assertEqual(model['head'][-1][0], [19, 22, 25])
            self.assertEqual(model['head'][-1][3][2], 300)

    def test_full_models_and_optimizer(self):
        try:
            from ultralytics.nn.tasks import RTDETRDetectionModel
            from ultralytics.nn.modules import CSPResidualStage, CSPContextStage
        except (ImportError, OSError) as exc:
            self.skipTest(f'complete Ultralytics dependencies unavailable: {exc}')
        directory = ROOT / 'ultralytics/cfg/models/rt-detr'
        filenames = {'stage3': 'rtdetr_r18_csp_stage3.yaml',
                     'stage4': 'rtdetr_r18_csp_stage4.yaml',
                     'combined': 'rtdetr_r18_csp_context.yaml'}
        for name, filename in filenames.items():
            model = RTDETRDetectionModel(str(directory / filename), nc=1, verbose=False).eval()
            expected3, expected4 = name in ('stage3', 'combined'), name in ('stage4', 'combined')
            self.assertEqual(isinstance(model.model[5], CSPResidualStage), expected3)
            self.assertEqual(isinstance(model.model[6], CSPContextStage), expected4)
            self.assertEqual(model.model[-1].f, [19, 22, 25])
            for batch_size, size in ((1, 640), (2, 128), (1, 160)):
                observed = {}
                hooks = []
                for index in (5, 6, 7):
                    hooks.append(model.model[index].register_forward_hook(
                        lambda _m, _i, o, key=index: observed.__setitem__(key, tuple(o.shape))))
                hooks.append(model.model[-1].register_forward_pre_hook(
                    lambda _m, values: observed.__setitem__(
                        'decoder', [tuple(x.shape) for x in values[0]])))
                with torch.inference_mode():
                    model(torch.zeros(batch_size, 3, size, size))
                for hook in hooks:
                    hook.remove()
                self.assertEqual(observed[5], (batch_size, 128, size // 8, size // 8))
                self.assertEqual(observed[6], (batch_size, 256, size // 16, size // 16))
                self.assertEqual(observed[7], (batch_size, 512, size // 32, size // 32))
                self.assertEqual(observed['decoder'], [
                    (batch_size, 256, size // 8, size // 8),
                    (batch_size, 256, size // 16, size // 16),
                    (batch_size, 256, size // 32, size // 32)])
            # DetectionTrainer sets nc before training; direct construction does not.
            model.nc = model.yaml['nc']
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            optimizer.zero_grad()
            train_batch = {'img': torch.rand(2, 3, 128, 128),
                           'batch_idx': torch.tensor([0, 1]),
                           'cls': torch.zeros(2, 1, dtype=torch.long),
                           'bboxes': torch.tensor([[.5, .5, .15, .15],
                                                   [.35, .4, .1, .12]])}
            loss, _ = model(train_batch)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            for index in (5, 6, 7, 19, 22, 25, 26):
                self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.model[index].parameters()))
            for index in ((5, 6) if name == 'combined' else ((5,) if expected3 else (6,))):
                stage = model.model[index]
                self.assertIsNotNone(stage.raw_gamma.grad)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in stage.process.parameters()))
                self.assertGreater(stage.split_projection.conv.weight.grad[
                    stage.process_channels:].abs().sum().item(), 0)
                if isinstance(stage, CSPContextStage):
                    self.assertGreater(stage.process[0].dwconv.weight.grad.abs().sum().item(), 0)
            optimizer.step()
            if name == 'combined':
                checkpoint = io.BytesIO()
                torch.save(model.state_dict(), checkpoint)
                checkpoint.seek(0)
                restored = RTDETRDetectionModel(str(directory / filename), nc=1,
                                                verbose=False).eval()
                restored.load_state_dict(torch.load(checkpoint, map_location='cpu',
                                                    weights_only=True))
                model.eval()
                with torch.inference_mode():
                    original_pred = model(torch.zeros(1, 3, 128, 128))[0]
                    restored_pred = restored(torch.zeros(1, 3, 128, 128))[0]
                torch.testing.assert_close(original_pred, restored_pred)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp(self):
        for stage, c1, c2 in ((csp.CSPResidualStage, 64, 128),
                              (csp.CSPContextStage, 128, 256)):
            module = stage(c1, c2).cuda().train()
            x = torch.randn(2, c1, 32, 32, device='cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                output = module(x)
                loss = output.float().square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(output).all())
            self.assertIsNotNone(module.raw_gamma.grad)


if __name__ == '__main__':
    unittest.main()
