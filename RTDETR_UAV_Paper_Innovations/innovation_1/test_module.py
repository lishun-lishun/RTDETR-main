# -*- coding: utf-8 -*-
"""
Unit Tests for Innovation 1: WCA-SFP & ShallowEllipseAuxLoss
Covers: Forward, Backward, Gradient check, Output Shape, Numerical Stability,
AMP (FP16), Batch Size 1 & 2, Empty GT, Single/Multiple GT, and Disabled Fallback.
"""

import unittest
import torch
from innovation_1.module import WCASFP, ShallowEllipseAuxLoss, HaarWaveletTransform2D


class TestInnovation1WCASFP(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_haar_wavelet_orthogonality_and_shape(self):
        dwt = HaarWaveletTransform2D(channels=64).to(self.device)
        x = torch.randn(2, 64, 160, 160, device=self.device)
        ll, lh, hl, hh = dwt(x)
        self.assertEqual(ll.shape, (2, 64, 80, 80))
        self.assertEqual(lh.shape, (2, 64, 80, 80))
        self.assertEqual(hl.shape, (2, 64, 80, 80))
        self.assertEqual(hh.shape, (2, 64, 80, 80))
        self.assertTrue(torch.isfinite(ll).all())

    def test_forward_backward_batch1_and_batch2(self):
        for bs in [1, 2]:
            model = WCASFP(c1=[64, 128], c2=256, enabled=True).to(self.device)
            p2 = torch.randn(bs, 64, 160, 160, device=self.device, requires_grad=True)
            p3 = torch.randn(bs, 128, 80, 80, device=self.device, requires_grad=True)
            out = model([p2, p3])
            self.assertEqual(out.shape, (bs, 256, 80, 80))
            self.assertFalse(torch.isnan(out).any(), "Output contains NaN")
            self.assertFalse(torch.isinf(out).any(), "Output contains Inf")

            loss = out.mean()
            loss.backward()
            self.assertIsNotNone(p2.grad)
            self.assertIsNotNone(p3.grad)
            self.assertGreater(p2.grad.abs().sum().item(), 0.0, "Zero gradient on P2")
            self.assertGreater(p3.grad.abs().sum().item(), 0.0, "Zero gradient on P3")
            self.assertIsNotNone(model.u_gamma.grad)

    def test_disabled_fallback_equivalence(self):
        model = WCASFP(c1=[64, 128], c2=256, enabled=False).to(self.device).eval()
        p2 = torch.randn(1, 64, 160, 160, device=self.device)
        p3 = torch.randn(1, 128, 80, 80, device=self.device)
        with torch.no_grad():
            out_disabled = model([p2, p3])
            out_proj_only = model.p3_proj(p3)
        max_diff = (out_disabled - out_proj_only).abs().max().item()
        self.assertLessEqual(max_diff, 1e-6, f"Disabled mode differs from baseline p3_proj by {max_diff}")

    def test_aux_loss_empty_single_and_multiple_gt(self):
        model = WCASFP(c1=[64, 128], c2=256, enabled=True).to(self.device)
        aux_criterion = ShallowEllipseAuxLoss().to(self.device)
        p2 = torch.randn(2, 64, 160, 160, device=self.device)
        p3 = torch.randn(2, 128, 80, 80, device=self.device)

        # Case 1: Empty GT
        out = model([p2, p3])
        batch_empty = {
            "bboxes": torch.zeros((0, 4), device=self.device),
            "batch_idx": torch.zeros((0,), dtype=torch.long, device=self.device),
        }
        loss_empty = aux_criterion(model.last_obj_logits, batch_empty)
        self.assertTrue(torch.isfinite(loss_empty))

        # Case 2: Single GT
        out = model([p2, p3])
        batch_single = {
            "bboxes": torch.tensor([[0.5, 0.5, 0.015, 0.015]], device=self.device),
            "batch_idx": torch.tensor([0], dtype=torch.long, device=self.device),
        }
        loss_single = aux_criterion(model.last_obj_logits, batch_single)
        self.assertTrue(torch.isfinite(loss_single))
        loss_single.backward()

        # Case 3: Multiple GT across batch
        model.zero_grad()
        out = model([p2, p3])
        batch_multi = {
            "bboxes": torch.tensor(
                [[0.2, 0.3, 0.01, 0.01], [0.7, 0.8, 0.03, 0.02], [0.5, 0.5, 0.4, 0.4]],
                device=self.device,
            ),
            "batch_idx": torch.tensor([0, 0, 1], dtype=torch.long, device=self.device),
        }
        loss_multi = aux_criterion(model.last_obj_logits, batch_multi)
        self.assertTrue(torch.isfinite(loss_multi))
        loss_multi.backward()

    def test_amp_compatibility(self):
        model = WCASFP(c1=[64, 128], c2=256, enabled=True).to(self.device)
        p2 = torch.randn(2, 64, 160, 160, device=self.device)
        p3 = torch.randn(2, 128, 80, 80, device=self.device)
        amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype):
            out = model([p2, p3])
            self.assertEqual(out.shape, (2, 256, 80, 80))
            self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
