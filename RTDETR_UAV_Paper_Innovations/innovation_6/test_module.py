# -*- coding: utf-8 -*-
"""Unit tests for Innovation 6: SSSCLite & SSSCRepC3."""

import copy
import torch
from RTDETR_UAV_Paper_Innovations.innovation_6.module import SSSCLite, SSSCRepC3


def test_sssc_shapes_amp_grad_and_deepcopy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for bs in (1, 2):
        x = torch.randn(bs, 512, 80, 80, device=device, requires_grad=True)
        mod = SSSCRepC3(
            c1=512,
            c2=256,
            n=3,
            e=0.5,
            top_k=128,
            max_offset=0.25,
            norm_tau=2.0,
            beta_max=0.35,
            init_gain=0.1,
            enabled=True,
        ).to(device)
        mod.train()

        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = mod(x)
            assert out.shape == (bs, 256, 80, 80)
            loss = out.square().mean()

        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for name, p in mod.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), f"Bad grad on {name}"

        # Verify ModelEMA deepcopy safety
        cloned = copy.deepcopy(mod)
        assert isinstance(cloned, SSSCRepC3)

        # Verify enabled=False fallback
        mod.enabled = False
        mod.eval()
        with torch.no_grad():
            out_off = mod(x)
            assert out_off.shape == (bs, 256, 80, 80)


if __name__ == "__main__":
    test_sssc_shapes_amp_grad_and_deepcopy()
    print("[PASS] Innovation 6 (SSSCRepC3) unit tests passed.")
