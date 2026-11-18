# -*- coding: utf-8 -*-
"""Unit tests for Innovation 5: DMBARepC3 & DMBABlock."""

import copy
import torch
from RTDETR_UAV_Paper_Innovations.innovation_5.module import DMBABlock, DMBARepC3


def test_dmba_shapes_amp_grad_and_deepcopy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for bs in (1, 2):
        x = torch.randn(bs, 512, 80, 80, device=device, requires_grad=True)
        mod = DMBARepC3(c1=512, c2=256, n=3, e=0.5, strip_k=7, gamma_max=0.35, init_gain=0.1, enabled=True).to(device)
        mod.train()

        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = mod(x)
            assert out.shape == (bs, 256, 80, 80)
            loss = out.square().mean()

        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert mod.dmba.u_gamma.grad is not None and torch.isfinite(mod.dmba.u_gamma.grad).all()

        # Verify ModelEMA deepcopy safety
        cloned = copy.deepcopy(mod)
        assert isinstance(cloned, DMBARepC3)

        # Verify enabled=False fallback
        mod.enabled = False
        mod.eval()
        with torch.no_grad():
            out_off = mod(x)
            assert out_off.shape == (bs, 256, 80, 80)


if __name__ == "__main__":
    test_dmba_shapes_amp_grad_and_deepcopy()
    print("[PASS] Innovation 5 (DMBARepC3) unit tests passed.")
