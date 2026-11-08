"""Contract tests for the independent DUT-Anti-UAV R09--R16 variants."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr"
NAMES = ("lipdown", "phasedown", "absr", "lcr", "coordrefine", "skresidual", "sdf", "sfr")


class TestR18TargetedContracts(unittest.TestCase):
    def test_yaml_registry_and_immutable_baseline(self):
        base = (MODEL_DIR / "rtdetr-r18.yaml").read_bytes()
        baseline = yaml.safe_load(base)
        for name in NAMES:
            path = MODEL_DIR / f"rtdetr_r18_{name}.yaml"
            self.assertTrue(path.is_file(), path)
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["scales"], baseline["scales"])
        self.assertEqual(base, (MODEL_DIR / "rtdetr-r18.yaml").read_bytes())

    def test_head_and_decoder_routes(self):
        baseline = yaml.safe_load((MODEL_DIR / "rtdetr-r18.yaml").read_text(encoding="utf-8"))
        for name in NAMES:
            cfg = yaml.safe_load((MODEL_DIR / f"rtdetr_r18_{name}.yaml").read_text(encoding="utf-8"))
            if name != "sdf":
                self.assertEqual(cfg["head"], baseline["head"], name)
                self.assertEqual(cfg["head"][-1][0], [19, 22, 25])
            else:
                self.assertEqual(cfg["head"][-1][0], [20, 23, 26])
                # P4 lateral, refined P3 lateral, then P5 are explicit.
                self.assertEqual(cfg["head"][4][0], 6)
                self.assertEqual(cfg["head"][9][0], 7)

    def test_targeted_module_symbols_and_gamma(self):
        try:
            import torch
            from ultralytics.nn.modules.r18_targeted_variants import (ABSR, CoordRefine, LCR, LIPDown,
                PhaseDown, SFR, SKResidual, SemanticDetailFeedback)
        except (ImportError, ModuleNotFoundError):
            self.skipTest("PyTorch is not installed")
        x = torch.randn(2, 64, 80, 80)
        self.assertEqual(LIPDown(64)(x).shape, (2, 64, 40, 40))
        self.assertEqual(PhaseDown(64)(x).shape, (2, 64, 40, 40))
        for cls in (ABSR, LCR, CoordRefine, SFR):
            mod = cls(64)
            y = mod(x)
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(hasattr(mod, "gamma"))
            with torch.no_grad():
                mod.gamma.zero_()
            self.assertTrue(torch.equal(mod(x), x))
        self.assertEqual(SKResidual(64)(x).shape, x.shape)
        self.assertEqual(SemanticDetailFeedback(64, 128)(x, torch.randn(2, 128, 40, 40)).shape, x.shape)

    def test_full_model_contracts(self):
        try:
            import torch
            from ultralytics.nn.tasks import RTDETRDetectionModel
        except (ImportError, ModuleNotFoundError):
            self.skipTest("PyTorch is not installed")
        for name in NAMES:
            with self.subTest(name=name):
                model = RTDETRDetectionModel(str(MODEL_DIR / f"rtdetr_r18_{name}.yaml"), nc=1, verbose=False).eval()
                with torch.inference_mode():
                    out = model(torch.zeros(1, 3, 128, 128))
                self.assertIsNotNone(out)

    def test_amp_or_fp32_forward_backward_finite(self):
        try:
            import torch
            from ultralytics.nn.modules.r18_targeted_variants import SFR
        except (ImportError, ModuleNotFoundError):
            self.skipTest("PyTorch is not installed")
        mod = SFR(16).train()
        x = torch.randn(2, 16, 20, 20, requires_grad=True)
        y = mod(x).square().mean(); y.backward()
        self.assertTrue(torch.isfinite(y))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(mod.gamma.grad)


if __name__ == "__main__":
    unittest.main()
