# -*- coding: utf-8 -*-
"""
Unit Tests for Innovation 2: FAGAFusion
Covers: Forward, Backward, Gradient flow, Output Shape, Zero-Init Identity,
Disabled Fallback equivalence, AMP (FP16/BF16), and Multi-scale Auto-alignment.
"""

import unittest
import torch
from innovation_2.module import FAGAFusion


class TestInnovation2FAGA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_zero_init_exact_identity_at_step0(self):
        """
        Because offset_head, beta_low, and beta_high are zero-initialized,
        at training step 0 (when f_high and f_low have same resolution),
        FAGAFusion must match torch.cat([f_high, f_low], dim=1) up to bilinear grid identity (~1e-5).
        """
        faga = FAGAFusion(c1=[256, 256], dimension=1, r_max=1.5, enabled=True).to(self.device).eval()
        f_high = torch.randn(2, 256, 80, 80, device=self.device)
        f_low = torch.randn(2, 256, 80, 80, device=self.device)
        with torch.no_grad():
            out = faga([f_high, f_low])
            ref = torch.cat([f_high, f_low], dim=1)
        max_diff = (out - ref).abs().max().item()
        self.assertLessEqual(max_diff, 1e-5, f"Zero-init identity drift too large: {max_diff}")

    def test_disabled_fallback_exact_equivalence(self):
        faga = FAGAFusion(c1=[256, 256], dimension=1, r_max=1.5, enabled=False).to(self.device).eval()
        # Non-zero parameters to verify `enabled=False` truly bypasses computation
        with torch.no_grad():
            faga.beta_low.fill_(0.5)
            faga.beta_high.fill_(-0.5)
        f_high = torch.randn(1, 256, 40, 40, device=self.device)
        f_low = torch.randn(1, 256, 40, 40, device=self.device)
        with torch.no_grad():
            out_disabled = faga([f_high, f_low])
            out_concat = torch.cat([f_high, f_low], dim=1)
        self.assertTrue(torch.equal(out_disabled, out_concat))

    def test_forward_backward_and_gradients(self):
        for bs in [1, 2]:
            faga = FAGAFusion(c1=[256, 256], dimension=1, r_max=1.5, enabled=True).to(self.device)
            # Nudge beta_low/beta_high slightly so gradient flows to all gates immediately
            with torch.no_grad():
                faga.beta_low.fill_(0.1)
                faga.beta_high.fill_(0.1)
                faga.offset_head.weight.normal_(0.0, 0.01)

            f_high = torch.randn(bs, 256, 80, 80, device=self.device, requires_grad=True)
            f_low = torch.randn(bs, 256, 80, 80, device=self.device, requires_grad=True)
            out = faga([f_high, f_low])
            self.assertEqual(out.shape, (bs, 512, 80, 80))
            self.assertTrue(torch.isfinite(out).all())

            loss = (out ** 2).mean()
            loss.backward()
            self.assertIsNotNone(f_high.grad)
            self.assertIsNotNone(f_low.grad)
            self.assertGreater(f_high.grad.abs().sum().item(), 0.0)
            self.assertGreater(f_low.grad.abs().sum().item(), 0.0)
            self.assertGreater(faga.offset_head.weight.grad.abs().sum().item(), 0.0)
            self.assertGreater(faga.beta_low.grad.abs().sum().item(), 0.0)
            self.assertGreater(faga.beta_high.grad.abs().sum().item(), 0.0)

    def test_auto_upsample_and_amp(self):
        faga = FAGAFusion(c1=[256, 256], dimension=1, r_max=1.5, enabled=True).to(self.device)
        f_high_half = torch.randn(2, 256, 40, 40, device=self.device)
        f_low_full = torch.randn(2, 256, 80, 80, device=self.device)
        amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
            out = faga([f_high_half, f_low_full])
            self.assertEqual(out.shape, (2, 512, 80, 80))
            self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
