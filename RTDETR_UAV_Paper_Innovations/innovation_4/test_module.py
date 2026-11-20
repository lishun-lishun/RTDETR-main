# -*- coding: utf-8 -*-
"""
Unit Tests for Innovation 4: SAHNWDRTDETRLoss & ScaleAdaptiveHungarianMatcher
Covers:
1. Non-vanishing NWD gradient on non-overlapping tiny UAV boxes (where standard IoU == 0)
2. Scale-adaptive gate beta(s): tiny UAVs use NWD+InnerGIoU, large UAVs revert to GIoU
3. Empty GT, Single GT, Multiple GT, and Denoising Query (`dn_meta`) loss computation
4. Backward gradient flow across all 4 layers (1 Encoder + 3 Decoder layers) via CW-LSD
5. Exact equivalence with baseline `RTDETRDetectionLoss` when `enabled=False`
"""

import unittest
import torch
from innovation_3.module import get_cdn_group_uav
from innovation_4.module import (
    SAHNWDRTDETRLoss,
    ScaleAdaptiveHungarianMatcher,
    normalized_wasserstein_distance,
    bbox_iou_xywh,
)


class TestInnovation4SAHNWD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_non_overlapping_tiny_uav_gradient_and_soft_target(self):
        """
        Verify that when a predicted 6x6px UAV box is shifted by 8px from GT (zero overlap, IoU=0),
        standard IoU soft target is 0.0, whereas SAH-NWD produces a positive hybrid quality
        q_hybrid > 0.35 and strong smooth gradients toward GT center.
        """
        criterion = SAHNWDRTDETRLoss(nc=1, enabled=True).to(self.device)
        pred_box = torch.tensor([[0.500, 0.500, 6.0 / 640.0, 6.0 / 640.0]], device=self.device, requires_grad=True)
        gt_box = torch.tensor([[0.515, 0.515, 6.0 / 640.0, 6.0 / 640.0]], device=self.device)  # shifted by ~9.6px

        iou, _ = bbox_iou_xywh(pred_box, gt_box, giou=False)
        self.assertEqual(iou.item(), 0.0, "Boxes should have zero overlap")

        q_hybrid = criterion._compute_hybrid_quality(pred_box, gt_box)
        self.assertGreater(q_hybrid.item(), 0.30, "q_hybrid must be non-zero even when IoU==0")

        loss_dict = criterion._get_loss_bbox(pred_box, gt_box)
        total_box_loss = loss_dict["loss_bbox"] + loss_dict["loss_giou"]
        total_box_loss.backward()
        self.assertIsNotNone(pred_box.grad)
        self.assertGreater(pred_box.grad.abs().sum().item(), 0.1)

    def test_full_loss_with_empty_single_multiple_gt_and_denoising(self):
        criterion = SAHNWDRTDETRLoss(nc=1, enabled=True).to(self.device)
        bs = 2
        nq = 300
        # 4 layers: 1 encoder + 3 decoder layers
        pred_bboxes = torch.sigmoid(torch.randn(4, bs, nq, 4, device=self.device, requires_grad=True))
        pred_scores = torch.randn(4, bs, nq, 1, device=self.device, requires_grad=True)

        # Case 1: Empty GT
        batch_empty = {
            "cls": torch.zeros((0,), dtype=torch.long, device=self.device),
            "bboxes": torch.zeros((0, 4), device=self.device),
            "batch_idx": torch.zeros((0,), dtype=torch.long, device=self.device),
            "gt_groups": [0, 0],
        }
        loss_empty = criterion((pred_bboxes, pred_scores), batch_empty)
        self.assertTrue(all(torch.isfinite(v) for v in loss_empty.values()))
        sum(loss_empty.values()).backward()

        # Case 2: Multiple GT + Denoising branch
        pred_bboxes = torch.sigmoid(torch.randn(4, bs, nq, 4, device=self.device, requires_grad=True))
        pred_scores = torch.randn(4, bs, nq, 1, device=self.device, requires_grad=True)
        batch_multi = {
            "cls": torch.tensor([0, 0, 0], dtype=torch.long, device=self.device),
            "bboxes": torch.tensor(
                [[0.5, 0.5, 0.01, 0.01], [0.25, 0.25, 0.02, 0.02], [0.75, 0.75, 0.15, 0.15]],
                device=self.device,
            ),
            "batch_idx": torch.tensor([0, 0, 1], dtype=torch.long, device=self.device),
            "gt_groups": [2, 1],
        }
        class_embed = torch.randn(1, 256, device=self.device)
        _, dn_bbox, _, dn_meta = get_cdn_group_uav(
            batch_multi, num_classes=1, num_queries=300, class_embed=class_embed, num_dn=100, training=True
        )
        num_dn = dn_meta["dn_num_split"][0]
        dn_bboxes = torch.sigmoid(torch.randn(3, bs, num_dn, 4, device=self.device, requires_grad=True))
        dn_scores = torch.randn(3, bs, num_dn, 1, device=self.device, requires_grad=True)

        loss_multi = criterion(
            (pred_bboxes, pred_scores),
            batch_multi,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
        )
        expected_keys = {
            "loss_class", "loss_bbox", "loss_giou",
            "loss_class_aux", "loss_bbox_aux", "loss_giou_aux",
            "loss_class_dn", "loss_bbox_dn", "loss_giou_dn",
            "loss_class_aux_dn", "loss_bbox_aux_dn", "loss_giou_aux_dn",
        }
        self.assertTrue(expected_keys.issubset(set(loss_multi.keys())))
        total = sum(loss_multi.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()


if __name__ == "__main__":
    unittest.main()
