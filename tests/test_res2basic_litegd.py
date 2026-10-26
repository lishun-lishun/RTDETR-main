"""Standalone Res2/GD tests, with optional complete-model server tests."""
import ast
from collections import OrderedDict
import importlib
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
selected = [node for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in
            {'get_activation', 'ConvNormLayer', 'BasicBlock', 'Blocks'}]
namespace = {'nn': nn, 'torch': torch, 'OrderedDict': OrderedDict}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(MODULE_DIR / 'block.py'), 'exec'), namespace)
package = types.ModuleType('_res2gd_test')
package.__path__ = [str(MODULE_DIR)]
sys.modules[package.__name__] = package
block = types.ModuleType(package.__name__ + '.block')
for name in ('ConvNormLayer', 'BasicBlock', 'Blocks'):
    setattr(block, name, namespace[name])
sys.modules[block.__name__] = block
res2 = importlib.import_module(package.__name__ + '.res2basic')
gd = importlib.import_module(package.__name__ + '.lite_gd')


class Res2GDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_res2_scales_hierarchy_and_gradients(self):
        for scale in (2, 4):
            module = res2.Res2Conv(128, 128, scale=scale).eval()
            x = torch.randn(2, 128, 12, 14, requires_grad=True)
            output = module(x)
            self.assertEqual(output.shape, x.shape)
            self.assertEqual(len(module.last_group_outputs), scale)
            self.assertTrue(all(part.shape[1] == 128 // scale
                                for part in module.last_group_outputs))
            split = module.pre(x).split(module.width, dim=1)
            for i, value in enumerate(module.last_hierarchical_inputs, start=1):
                expected = split[i] + module.last_group_outputs[i - 1] if i > 1 else split[i]
                torch.testing.assert_close(value, expected)
            output.mean().backward()
            self.assertTrue(all(layer.conv.weight.grad is not None
                                for layer in module.groups))
            self.assertIsNotNone(module.pre.conv.weight.grad)
            self.assertIsNotNone(module.post.conv.weight.grad)
        plain = res2.Res2Conv(128, 128, hierarchical=False).eval()
        plain(torch.randn(1, 128, 8, 8))
        self.assertEqual(len(plain.last_hierarchical_inputs), 3)

    def test_basicblock_position_residual_and_stride(self):
        stage = res2.Res2Blocks(64, 128, 2, 3, replace_indices=(1,))
        self.assertIsInstance(stage.blocks[0], namespace['BasicBlock'])
        self.assertNotIsInstance(stage.blocks[0], res2.Res2BasicBlock)
        self.assertIsInstance(stage.blocks[1], res2.Res2BasicBlock)
        self.assertIsInstance(stage.blocks[1].branch2b, namespace['ConvNormLayer'])
        x = torch.randn(2, 64, 16, 16)
        self.assertEqual(stage(x).shape, (2, 128, 8, 8))
        b = stage.blocks[1]
        incoming = stage.blocks[0](x)
        manual = b.act(b.branch2b(b.branch2a(incoming)) + incoming)
        torch.testing.assert_close(b(incoming), manual)
        both = res2.Res2Blocks(64, 128, 2, 3, replace_indices=(0, 1))
        self.assertEqual(both(x).shape, (2, 128, 8, 8))

    def test_gather_injection_and_ablation_paths(self):
        x = [torch.randn(2, c, size, size, requires_grad=True)
             for c, size in ((128, 32), (256, 16), (256, 8))]
        neck = gd.LiteGDNeck(128, 256, 256).eval()
        outputs = neck(x)
        self.assertEqual([tuple(o.shape) for o in outputs],
                         [(2, 256, 32, 32), (2, 256, 16, 16), (2, 256, 8, 8)])
        self.assertEqual(neck.last_high.shape[-2:], (8, 8))
        self.assertEqual(neck.last_low.shape[-2:], (16, 16))
        sum(o.square().mean() for o in outputs).backward()
        self.assertIsNotNone(neck.high_gather.fuse[0][0].weight.grad)
        self.assertIsNotNone(neck.low_gather.fuse[0][0].weight.grad)
        self.assertIsNotNone(neck.low_inject[0].output.weight.grad)
        self.assertIsNotNone(neck.high_inject5.raw_gain.grad)
        local = gd.LiteGDNeck(128, 256, 256, mode='local_only').eval()
        local_output = local(x)
        expected = [refine(project(part)) for part, project, refine in
                    zip(x, local.base, local.local_refine)]
        for actual, reference in zip(local_output, expected):
            torch.testing.assert_close(actual, reference)
        high = gd.LiteGDNeck(128, 256, 256, mode='high_only').eval()
        self.assertEqual(high(x)[0].shape[-2:], (32, 32))
        low = gd.LiteGDNeck(128, 256, 256, mode='low_only').eval()
        self.assertEqual(low(x)[2].shape[-2:], (8, 8))
        p3_only = gd.LiteGDNeck(128, 256, 256, mode='p3_only').eval()
        p3_outputs = p3_only(x)
        local_features = [refine(project(part)) for part, project, refine in
                          zip(x, p3_only.base, p3_only.local_refine)]
        self.assertFalse(torch.allclose(p3_outputs[0], local_features[0]))
        for actual, local_feature in zip(p3_outputs[1:], local_features[1:]):
            torch.testing.assert_close(actual, local_feature)
        disabled = gd.LiteGDNeck(128, 256, 256, distribute=(False, False, False)).eval()
        disabled_outputs = disabled(x)
        expected = [refine(project(part)) for part, project, refine in
                    zip(x, disabled.base, disabled.local_refine)]
        for actual, reference in zip(disabled_outputs, expected):
            torch.testing.assert_close(actual, reference)

    def test_yaml_topology(self):
        directory = ROOT / 'ultralytics/cfg/models/rt-detr'
        base = yaml.safe_load((directory / 'rtdetr-r18.yaml').read_text(encoding='utf-8'))
        for name in ('res2basic', 'res2basic_independent', 'litegd',
                     'litegd_high_only', 'litegd_low_only', 'litegd_local_only',
                     'litegd_p3_only', 'res2basic_litegd'):
            model = yaml.safe_load((directory / f'rtdetr_r18_{name}.yaml').read_text(encoding='utf-8'))
            self.assertEqual(model['backbone'][:5], base['backbone'][:5])
            layers = model['backbone'] + model['head']
            self.assertEqual(layers[-1][2], 'RTDETRDecoder')
            self.assertEqual(layers[-1][3][2], 300)
            if 'litegd' in name:
                self.assertEqual(model['head'][:2], base['head'][:2])
                self.assertEqual(layers[10][0], [5, 6, 9])
                self.assertEqual(layers[-1][0], [11, 12, 13])
            else:
                self.assertEqual(model['head'], base['head'])

    def test_full_model_shapes_and_update(self):
        try:
            from ultralytics.nn.tasks import RTDETRDetectionModel
        except (ImportError, OSError) as exc:
            self.skipTest(f'full zoo dependencies unavailable: {exc}')
        directory = ROOT / 'ultralytics/cfg/models/rt-detr'
        for name in ('res2basic', 'litegd', 'res2basic_litegd'):
            model = RTDETRDetectionModel(str(directory / f'rtdetr_r18_{name}.yaml'),
                                         nc=1, verbose=False).eval()
            for batch, size in ((1, 640), (2, 640), (1, 160), (2, 128)):
                incoming = []
                hook = model.model[-1].register_forward_pre_hook(
                    lambda _m, args: incoming.extend(tuple(f.shape) for f in args[0]))
                with torch.inference_mode():
                    model(torch.zeros(batch, 3, size, size))
                hook.remove()
                self.assertEqual(incoming, [(batch, 256, size // 8, size // 8),
                                            (batch, 256, size // 16, size // 16),
                                            (batch, 256, size // 32, size // 32)])
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            optimizer.zero_grad()
            output = model(torch.randn(2, 3, 128, 128))
            def tensors(value):
                if torch.is_tensor(value):
                    return [value]
                if isinstance(value, (list, tuple)):
                    return [tensor for part in value for tensor in tensors(part)]
                return []
            sum(t.float().square().mean() for t in tensors(output)).backward()
            from ultralytics.nn.modules import Res2Conv, LiteGDNeck
            added = [m for m in model.modules() if isinstance(m, (Res2Conv, LiteGDNeck))]
            self.assertTrue(added)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for m in added for p in m.parameters()))
            optimizer.step()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_amp(self):
        conv = res2.Res2Conv(128, 128).cuda()
        neck = gd.LiteGDNeck(128, 256, 256).cuda()
        features = [torch.randn(2, c, size, size, device='cuda')
                    for c, size in ((128, 80), (256, 40), (256, 20))]
        with torch.autocast('cuda', dtype=torch.float16):
            output = conv(features[0])
            fused = neck(features)
            (output.square().mean() + sum(x.square().mean() for x in fused)).backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(torch.isfinite(x).all() for x in fused))


if __name__ == '__main__':
    unittest.main()
