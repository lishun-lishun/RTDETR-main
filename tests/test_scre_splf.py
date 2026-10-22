"""Preflight tests for SCRE/SPLF; run on the Linux A30 before training."""

import gc
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml


class YAMLTopologyTests(unittest.TestCase):
    def test_all_five_variants_keep_original_main_path(self):
        from tools.generate_scre_splf_yaml import BASELINE, make, shift_from

        original_neck = BASELINE["head"][:-1]
        for variant in ("scre", "splf", "scre_splf", "scre_splf_no_offset",
                        "scre_splf_no_protect"):
            config = make(variant)
            self.assertEqual(config["backbone"], BASELINE["backbone"])
            self.assertEqual(len(config["head"][-1][0]), 3)
            if variant == "splf":
                self.assertEqual(config["head"][:len(original_neck)], original_neck)
                self.assertEqual(config["head"][-1][0], [26, 22, 25])
            else:
                self.assertEqual(config["head"][0][0], [4, 5])
                self.assertEqual(config["head"][1][0], 7)
                expected = [[7 if i == 0 else shift_from(row[0]), *row[1:]]
                            for i, row in enumerate(original_neck)]
                self.assertEqual(config["head"][1:1 + len(original_neck)], expected)
                self.assertEqual(config["head"][-1][0], [27, 23, 26])
            disk = ROOT / "ultralytics" / "cfg" / "models" / "rt-detr" / f"rtdetr_r18_{variant}.yaml"
            self.assertEqual(yaml.safe_load(disk.read_text(encoding="utf-8")), config)

    def test_size_group_ap_confidence_order_and_ignore(self):
        from tools.scre_splf_metrics import group_summary

        records = [{"gt_boxes": [[0, 0, 6, 6], [40, 40, 80, 80]],
                    "gt_groups": ["Tiny", "Large"],
                    "pred_boxes": [[40, 40, 80, 80], [0, 0, 6, 6], [90, 90, 95, 95]],
                    "scores": [0.95, 0.9, 0.2]}]
        rows = {row["group"]: row for row in group_summary(records)}
        self.assertAlmostEqual(rows["Tiny"]["ap50"], 1.0)
        self.assertAlmostEqual(rows["Large"]["ap50"], 1.0)
        self.assertEqual(rows["Tiny"]["false_positives_at_conf"], 0)
        self.assertIsNone(rows["Small"]["ap50"])

    def test_frozen_indices_exclude_only_new_branches(self):
        from tools.scre_splf_common import main_layer_indices

        self.assertEqual(main_layer_indices("splf"), list(range(26)) + [27])
        self.assertEqual(main_layer_indices("scre_splf"),
                         list(range(8)) + list(range(9, 27)) + [28])
        for variant in ("scre", "no_offset", "no_protect"):
            self.assertEqual(main_layer_indices(variant), main_layer_indices("scre_splf"))

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from ultralytics.nn.modules.scre_splf import SCRE, SPLF, SCREFixedFusion
    from ultralytics.nn.modules.scre_splf_loss import (auxiliary_losses, center_focal_loss,
                                                        make_aux_targets, matched_offset_loss,
                                                        protection_loss)


@unittest.skipIf(torch is None, "PyTorch/custom Ultralytics imports unavailable on this machine")
class SCRESPLFTests(unittest.TestCase):
    @staticmethod
    def fake_batch(batch_size=2, side=128, boxes=None):
        boxes = boxes or []
        return {
            "img": torch.zeros(batch_size, 3, side, side),
            "bboxes": torch.tensor([b[1:] for b in boxes], dtype=torch.float32).reshape(-1, 4),
            "batch_idx": torch.tensor([b[0] for b in boxes], dtype=torch.long),
            "cls": torch.zeros(len(boxes), 1),
        }

    def test_scre_forward_batch_topk_and_bounds(self):
        model = SCRE(64, 128, 64, topk=32, patch_size=5)
        for b, h, w in ((1, 8, 8), (2, 12, 16), (1, 4, 4)):
            p2 = torch.randn(b, 64, h * 2, w * 2)
            p3 = torch.randn(b, 128, h, w)
            out = model([p2, p3])
            self.assertEqual(tuple(out.shape), (b, 65, h, w))
            selected = model.last["indices"]
            self.assertEqual(tuple(selected.shape), (b, min(32, h * w)))
            for row in selected:
                self.assertEqual(row.unique().numel(), row.numel())
                self.assertTrue(((row >= 0) & (row < h * w)).all())

    def test_grid_coordinates_align_corners_false(self):
        model = SCRE(64, 128, 64, 1, 5)
        p2 = torch.arange(16, dtype=torch.float32).view(1, 1, 1, 16).expand(1, 64, 16, 16) + 0.5
        patch = model._sample_p2(p2, torch.tensor([[5.0]]), torch.tensor([[7.0]]))
        self.assertAlmostEqual(float(patch[0, 0, 0, 2, 2]), 5.0, places=5)
        edge = model._sample_p2(p2, torch.tensor([[0.2]]), torch.tensor([[0.2]]))
        self.assertTrue(torch.isfinite(edge).all())

    def test_scatter_overlap_and_gradient(self):
        descriptors = torch.tensor([[[2.0], [4.0]]], requires_grad=True)
        offsets = torch.tensor([[[0.2, 0.0], [0.0, 0.0]]], requires_grad=True)
        out = SCRE._scatter(descriptors, torch.tensor([[5, 5]]), offsets, 4, 4)
        self.assertGreater(float(out[0, 0, 1, 1]), 0.0)
        self.assertLessEqual(float(out[0, 0, 1, 1]), 4.0)
        out[0, 0, 1, 2].backward()
        self.assertIsNotNone(descriptors.grad)
        self.assertGreater(float(descriptors.grad.abs().sum()), 0)
        self.assertGreater(float(offsets.grad.abs().sum()), 0)

    def test_splf_identity_and_f4_f5_untouched(self):
        f3 = torch.randn(2, 256, 8, 8)
        f4 = torch.randn(2, 256, 4, 4)
        f5 = torch.randn(2, 256, 2, 2)
        aux = torch.randn(2, 65, 8, 8)
        module = SPLF(256, 65, 64, 32, 0.1)
        out = module([f3, aux])
        self.assertTrue(torch.equal(out, f3))
        self.assertTrue(torch.equal(f4, f4.clone()))
        self.assertTrue(torch.equal(f5, f5.clone()))
        with torch.no_grad():
            module.alpha_raw.fill_(0.5)
        changed = module([f3, aux])
        self.assertFalse(torch.equal(changed, f3))
        self.assertLessEqual(float((changed - f3).abs().max()), 0.1 + 1e-6)
        standalone = SPLF(256, None, 64, 16, 0.1)
        self.assertEqual(tuple(standalone(f3).shape), tuple(f3.shape))
        self.assertIsNotNone(standalone.last["logits"])
        fixed = SCREFixedFusion(256, 65, 64, 0.05)
        self.assertEqual(tuple(fixed([f3, aux]).shape), tuple(f3.shape))

    def test_aux_targets_empty_large_multiple_and_boundary(self):
        batch = self.fake_batch(2, 128, [
            (0, 0.2, 0.2, 0.04, 0.04),
            (0, 0.95, 0.95, 0.025, 0.025),
            (1, 0.5, 0.5, 0.25, 0.25),
        ])
        heat, valid, protect, centers = make_aux_targets(batch, 16, 16)
        self.assertEqual(len(centers[0]), 2)
        self.assertEqual(len(centers[1]), 0)
        self.assertEqual(int(heat.sum() > 0), 1)
        self.assertEqual(float(valid[1, 0, 8, 8]), 0)
        self.assertEqual(float(protect[1, 0, 8, 8]), 1)
        empty = self.fake_batch(1)
        heat2, valid2, _, centers2 = make_aux_targets(empty, 16, 16)
        self.assertEqual(len(centers2[0]), 0)
        self.assertTrue(torch.isfinite(center_focal_loss(torch.zeros_like(heat2), heat2, valid2)))

    def test_offset_one_to_one(self):
        indices = torch.tensor([[0, 1]])
        offsets = torch.zeros(1, 2, 2, requires_grad=True)
        centers = [torch.tensor([[0.6, 0.6], [0.7, 0.7]])]
        loss = matched_offset_loss(indices, offsets, centers, 4, max_distance=1.0)
        self.assertGreater(float(loss), 0)
        loss.backward()
        self.assertGreater(float(offsets.grad.abs().sum()), 0)

    def test_two_steps_all_aux_gradients(self):
        scre = SCRE(64, 128, 64, topk=16)
        splf = SPLF(256, 65)
        optimizer = torch.optim.SGD(list(scre.parameters()) + list(splf.parameters()), lr=0.1)
        p2 = torch.randn(1, 64, 16, 16)
        p3 = torch.randn(1, 128, 8, 8)
        f3 = torch.randn(1, 256, 8, 8)
        for step in range(2):
            optimizer.zero_grad()
            side = scre([p2, p3])
            chosen = int(scre.last["indices"][0, 0])
            box = ((chosen % 8 + 0.75) / 8, (chosen // 8 + 0.75) / 8, 0.04, 0.04)
            batch = self.fake_batch(1, 64, [(0, *box)])
            output = splf([f3, side])
            center, offset, protect = auxiliary_losses(
                batch, {"lambda_center": 0.05, "lambda_offset": 0.1,
                        "lambda_protect": 0.01}, scre, splf)
            self.assertTrue(all(torch.isfinite(x) for x in (center, offset, protect)))
            (output.square().mean() + center + offset + protect).backward()
            self.assertIsNotNone(scre.center_head.weight.grad)
            self.assertGreater(float(scre.center_head.weight.grad.abs().sum()), 0)
            self.assertIsNotNone(scre.offset_head.weight.grad)
            self.assertGreater(float(scre.offset_head.weight.grad.abs().sum()), 0)
            self.assertIsNotNone(splf.alpha_raw.grad)
            optimizer.step()
            scre.last = splf.last = None
        with torch.no_grad():
            splf.alpha_raw.fill_(0.5)
        splf.zero_grad()
        side = scre([p2, p3])
        splf([f3, side])
        protection_loss(splf.last["base"], splf.last["enhanced"],
                        torch.ones(1, 1, 8, 8)).backward()
        self.assertIsNotNone(splf.project.weight.grad)
        self.assertGreater(float(splf.project.weight.grad.abs().sum()), 0)

    def test_yaml_structure_and_model_loading(self):
        import yaml
        from ultralytics.nn.tasks import RTDETRDetectionModel
        from tools.scre_splf_common import MODEL_DIR, VARIANTS

        for name in ("scre", "splf", "scre_splf"):
            cfg = yaml.safe_load((MODEL_DIR / VARIANTS[name]).read_text(encoding="utf-8"))
            layers = cfg["backbone"] + cfg["head"]
            self.assertEqual(layers[-1][2], "RTDETRDecoder")
            self.assertEqual(layers[-1][3][2], 300)
            self.assertEqual(len(layers[-1][0]), 3)
            if name != "splf":
                self.assertEqual(layers[9][0], 7)  # original P5, not SCRE
                self.assertEqual(layers[8][0], [4, 5])
                self.assertEqual(layers[-1][0][1:], [23, 26])
            model = RTDETRDetectionModel(str(MODEL_DIR / VARIANTS[name]), nc=1, verbose=False)
            self.assertEqual(len(model.model[-1].f), 3)
            model.eval()
            with torch.no_grad():
                for batch_size, side in ((1, 128), (2, 160)):
                    self.assertIsNotNone(model.predict(torch.zeros(batch_size, 3, side, side)))
            del model
            gc.collect()

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA/A30 required")
    def test_cuda_amp_and_full_decoder_hooks(self):
        from tools.scre_splf_common import yaml_path
        from ultralytics.nn.tasks import RTDETRDetectionModel

        model = RTDETRDetectionModel(str(yaml_path("scre_splf")), nc=1, verbose=False).cuda().eval()
        outputs = {}
        handles = []
        for index in (4, 5, 6, 7, 8, 20, 23, 26, 27):
            handles.append(model.model[index].register_forward_hook(
                lambda module, incoming, result, index=index: outputs.__setitem__(index, result)))
        routed_inputs = {}
        for index in (6, 7, 8, 9, 13, 18):
            handles.append(model.model[index].register_forward_pre_hook(
                lambda module, incoming, index=index: routed_inputs.__setitem__(index, incoming[0])))
        decoder_inputs = []
        handles.append(model.model[-1].register_forward_pre_hook(
            lambda module, incoming: decoder_inputs.extend(incoming[0])))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            prediction = model.predict(torch.zeros(1, 3, 640, 640, device="cuda"))
        self.assertIsNotNone(prediction)
        self.assertEqual(tuple(outputs[4].shape), (1, 64, 160, 160))
        self.assertEqual(tuple(outputs[5].shape), (1, 128, 80, 80))
        self.assertEqual(tuple(outputs[6].shape[-2:]), (40, 40))
        self.assertEqual(tuple(outputs[7].shape[-2:]), (20, 20))
        self.assertEqual(tuple(outputs[8].shape), (1, 65, 80, 80))
        self.assertIs(routed_inputs[6], outputs[5])   # original P3 -> Stage4
        self.assertIs(routed_inputs[7], outputs[6])   # original P4 -> Stage5
        self.assertIs(routed_inputs[8][0], outputs[4])  # P2 -> SCRE
        self.assertIs(routed_inputs[8][1], outputs[5])  # P3 -> SCRE
        self.assertIs(routed_inputs[9], outputs[7])   # original P5 -> CCFM
        self.assertIs(routed_inputs[13], outputs[6])  # original P4 -> CCFM
        self.assertIs(routed_inputs[18], outputs[5])  # original P3 -> CCFM
        self.assertEqual([tuple(x.shape) for x in decoder_inputs],
                         [(1, 256, 80, 80), (1, 256, 40, 40), (1, 256, 20, 20)])
        self.assertIs(decoder_inputs[0], outputs[27])
        self.assertIs(decoder_inputs[1], outputs[23])
        self.assertIs(decoder_inputs[2], outputs[26])
        self.assertTrue(torch.equal(outputs[27], outputs[20]))  # alpha=0
        for handle in handles:
            handle.remove()

        scre = SCRE(64, 128).cuda().train()
        splf = SPLF(256, 65).cuda().train()
        with torch.autocast("cuda", dtype=torch.float16):
            side = scre([torch.randn(2, 64, 32, 32, device="cuda"),
                         torch.randn(2, 128, 16, 16, device="cuda")])
            result = splf([torch.randn(2, 256, 16, 16, device="cuda"), side])
            loss = result.float().square().mean() + side.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(scre.p2_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
