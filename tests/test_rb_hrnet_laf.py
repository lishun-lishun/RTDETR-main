"""Standalone checks; import only the new modules to avoid optional zoo deps."""
import importlib
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / 'ultralytics/nn/modules'
package = types.ModuleType('_rb_hrnet_port_test')
package.__path__ = [str(MODULES)]
sys.modules[package.__name__] = package
hrnet = importlib.import_module(package.__name__ + '.hrnet')
rb = importlib.import_module(package.__name__ + '.rb_hrnet')
laf = importlib.import_module(package.__name__ + '.laf')


class RBHRNetLAFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_backbone_branch_shapes_and_gradients(self):
        for constructor, channels in ((hrnet.HRNetV2W18, [18, 36, 72, 144]),
                                      (rb.RBHRNetA, [24, 36, 72, 144]),
                                      (rb.RBHRNetB, [24, 36, 64, 112])):
            model = constructor(pretrained=False).train()
            branches = []
            hook = model.stage4[-1].register_forward_hook(
                lambda _m, _i, out: branches.extend(tuple(x.shape) for x in out))
            outputs = model(torch.randn(2, 3, 128, 128))
            hook.remove()
            self.assertEqual([shape[1] for shape in branches], channels)
            self.assertEqual([out.shape[-2:] for out in outputs],
                             [(16, 16), (8, 8), (4, 4)])
            sum(out.square().mean() for out in outputs).backward()
            self.assertIsNotNone(model.transition3[-1][0][0].weight.grad)

    def test_laf_geometry_identity_and_two_step_gradient(self):
        sampler = laf.DynamicUpsample(4)
        f4 = torch.randn(2, 4, 4, 4)
        f3 = torch.randn(2, 4, 8, 8)
        sampled = sampler([f4, f3])
        expected = torch.nn.functional.interpolate(
            f4, size=(8, 8), mode='bilinear', align_corners=False)
        torch.testing.assert_close(sampled, expected)
        block = laf.IsolatedLAF(4)
        torch.testing.assert_close(block([f3, f4]), f3, rtol=0, atol=0)
        optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
        for _ in range(2):
            optimizer.zero_grad()
            block([f3, f4]).square().mean().backward()
            optimizer.step()
        self.assertGreater(block.sampler.offset_predictor.weight.grad.abs().sum(), 0)

    def test_weight_loading_reports_partial_coverage(self):
        reference = hrnet.HRNetV2W18(pretrained=False)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'hrnet.pth'
            torch.save(reference.state_dict(), checkpoint)
            with contextlib.redirect_stdout(io.StringIO()):
                candidate = rb.RBHRNetB(pretrained_path=str(checkpoint))
        report = candidate.pretrained_load_report
        self.assertGreater(report['coverage'], 0.1)
        self.assertLess(report['coverage'], 1.0)
        self.assertIn('stage4.0.branches.0.0.conv1.weight', report['missing_keys'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA AMP unavailable')
    def test_cuda_amp_forward_backward(self):
        model = rb.RBHRNetA(pretrained=False).cuda().train()
        image = torch.randn(1, 3, 256, 256, device='cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            outputs = model(image)
            loss = sum(x.square().mean() for x in outputs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.conv1.weight.grad)

    def test_yaml_decoder_contract(self):
        directory = ROOT / 'ultralytics/cfg/models/rt-detr'
        cases = {
            'rtdetr_hrnet_w18.yaml': ('HRNetV2W18', 22),
            'rtdetr_rb_hrnet_a.yaml': ('RBHRNetA', 22),
            'rtdetr_rb_hrnet_b.yaml': ('RBHRNetB', 22),
            'rtdetr_hrnet_laf0.yaml': ('HRNetV2W18', 22),
            'rtdetr_hrnet_laf1.yaml': ('HRNetV2W18', 23),
            'rtdetr_rb_hrnet_laf1.yaml': ('RBHRNetA', 23),
        }
        for name, (backbone, decoder_index) in cases.items():
            config = yaml.safe_load((directory / name).read_text(encoding='utf-8'))
            self.assertEqual(config['backbone'][0][2], backbone)
            self.assertEqual([row[2] for row in config['backbone'][1:]],
                             ['HRNetFeature'] * 3)
            decoder = config['head'][-1]
            self.assertEqual(decoder[2], 'RTDETRDecoder')
            self.assertEqual(decoder[3][2], 300)
            self.assertEqual(len(config['backbone']) + len(config['head']) - 1,
                             decoder_index)
            self.assertEqual(len(decoder[0]), 3)

    def test_full_model_decoder_inputs_when_zoo_dependencies_available(self):
        try:
            from ultralytics.nn.tasks import RTDETRDetectionModel
        except ImportError as error:
            self.skipTest(f'Full Ultralytics dependencies unavailable: {error}')
        directory = ROOT / 'ultralytics/cfg/models/rt-detr'
        for filename in ('rtdetr_rb_hrnet_a.yaml', 'rtdetr_rb_hrnet_b.yaml',
                         'rtdetr_hrnet_laf0.yaml', 'rtdetr_hrnet_laf1.yaml',
                         'rtdetr_rb_hrnet_laf1.yaml'):
            model = RTDETRDetectionModel(str(directory / filename), nc=1,
                                         verbose=False).eval()
            seen = []
            hook = model.model[-1].register_forward_pre_hook(
                lambda _m, args: seen.extend(tuple(x.shape) for x in args[0]))
            with torch.inference_mode():
                output = model(torch.randn(1, 3, 128, 128))
            hook.remove()
            self.assertEqual(seen, [(1, 256, 16, 16),
                                    (1, 256, 8, 8), (1, 256, 4, 4)])
            self.assertEqual(output[0].shape[1], 300)


if __name__ == '__main__':
    unittest.main()
