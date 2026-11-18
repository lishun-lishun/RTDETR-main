# -*- coding: utf-8 -*-
"""Unit tests for Innovation 7: GLRoIDecoder & P2RoIReSampler."""

import copy
import torch
from RTDETR_UAV_Paper_Innovations.innovation_7.module import GLRoIDecoder, P2RoIReSampler


def test_glroi_decoder_shapes_amp_grad_and_deepcopy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for bs in (1, 2):
        p2 = torch.randn(bs, 64, 160, 160, device=device, requires_grad=True)
        p3 = torch.randn(bs, 256, 80, 80, device=device, requires_grad=True)
        p4 = torch.randn(bs, 256, 40, 40, device=device, requires_grad=True)
        p5 = torch.randn(bs, 256, 20, 20, device=device, requires_grad=True)

        dec = GLRoIDecoder(
            nc=1,
            ch=(64, 256, 256, 256),
            hd=256,
            nq=300,
            ndp=4,
            nh=8,
            ndl=3,
            top_m=16,
            roi_grid_size=5,
            init_gain=0.1,
            enabled=True,
        ).to(device)

        # 1. Training mode with CDN batch
        dec.train()
        batch = {
            "batch_idx": torch.tensor([0, 0], dtype=torch.long, device=device) if bs == 1 else torch.tensor([0, 1], dtype=torch.long, device=device),
            "cls": torch.zeros((2, 1), dtype=torch.long, device=device),
            "bboxes": torch.tensor([[0.5, 0.5, 0.012, 0.012], [0.3, 0.3, 0.015, 0.015]], dtype=torch.float32, device=device),
            "gt_groups": [2] if bs == 1 else [1, 1],
        }

        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=use_amp):
            dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = dec([p2, p3, p4, p5], batch=batch)
            loss = dec_bboxes.square().mean() + dec_scores.square().mean() + enc_bboxes.square().mean() + enc_scores.square().mean()

        loss.backward()
        assert p2.grad is not None and torch.isfinite(p2.grad).all()
        for name, p in dec.roi_resampler.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), f"Bad grad on {name}"

        # 2. Eval mode -> exact [B, 300, 5] contract
        dec.eval()
        with torch.no_grad():
            y, _ = dec([p2, p3, p4, p5])
            assert y.shape == (bs, 300, 5), f"Expected {(bs, 300, 5)}, got {y.shape}"

        # 3. ModelEMA deepcopy safety
        cloned = copy.deepcopy(dec)
        assert isinstance(cloned, GLRoIDecoder)


if __name__ == "__main__":
    test_glroi_decoder_shapes_amp_grad_and_deepcopy()
    print("[PASS] Innovation 7 (GLRoIDecoder) unit tests passed.")
