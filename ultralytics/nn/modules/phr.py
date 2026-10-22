"""PHR (Partial High-Resolution Residual Backbone) pluggable module for RT-DETR-R18.

PHR adds a lightweight, single high-resolution (1/4 scale) computational path
beside the original ResNet-18 staged backbone. It reads the already-computed
P2/P3/P4 features and produces an *enhanced P3* that is compensated back into the
Neck P3 lateral input only. The original P3/P4/P5 backbone path and the Neck /
Decoder are left untouched.

The module is intentionally a *delayed, additive compensation*:

    P3_enhanced = P3 + gamma * Delta_P3

where ``gamma`` is a bounded, zero-initialized learnable scale. At initialization
the module is therefore an exact identity (``P3_enhanced == P3``), and only the
residual *projection* is normally initialized so that ``gamma`` (and, after the
first step, the whole high-resolution path) receives a non-zero gradient — the
ReZero convention: normal-init the projection, zero-init the scale.

Design notes:
    * pluggable: registered by name in ``parse_model`` via ``globals()[m]``;
    * multi-input: declared with a YAML ``from`` list ``[p2_idx, p3_idx, p4_idx]``;
    * DAG-safe: placed *after* the original Stage5 in the YAML, it only reads
      features that have already been produced (P2/P3/P4), so no cycle is formed;
    * AMP/FP16-safe: pure conv/BN/interpolate ops, no fixed kernels, no ``.detach()``;
    * free of heavy dependencies (torch only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import ConvNormLayer

__all__ = ('PHR', 'HRBlock')


class HRBlock(nn.Module):
    """Lightweight residual block for the high-resolution path.

    Depthwise 3x3 -> Pointwise 1x1 -> BatchNorm -> ReLU -> residual add.
    Keeps the spatial size and channel count unchanged, and uses a single
    normalization + activation (no global self-attention). The internal layers are
    normally initialized; identity at the *module* level is guaranteed by the
    zero-initialized ``gamma`` scale in :class:`PHR`, not by zeroing this block.
    """

    def __init__(self, channels):
        super().__init__()
        c = int(channels)
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, 1, bias=False)
        self.norm = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        out = self.norm(out)
        out = self.act(out)
        return out + x


class PHR(nn.Module):
    """Partial High-Resolution Residual Backbone for the P3 Neck input.

    Inputs (from a YAML ``from`` list ``[p2_idx, p3_idx, p4_idx]``)::

        P2: [B, c_p2, H/4,  W/4 ]
        P3: [B, c_p3, H/8,  W/8 ]
        P4: [B, c_p4, H/16, W/16]

    Output::

        P3_enhanced: [B, c_p3, H/8, W/8]

    Pipeline::

        H0 = hr_init(P2)                                   # 1x1 conv c_p2 -> h
        for i in range(num_hr_blocks):
            H = hr_blocks[i](H)
            if P3 feedback pending:  H = H + beta3 * up(P3_proj)   # injected after block
            if P4 feedback pending:  H = H + beta4 * up(P4_proj)
        Delta_P3 = delta(ReLU(BN(downsample(H))))          # h@H/4 -> h@H/8 -> c_p3
        gamma = alpha_max * tanh(raw_gamma)                # bounded, zero-initialized
        return P3 + gamma * Delta_P3

    Feedback positions follow the three progressive experiment layouts:
    * PHR-1 (no feedback):  H0 -> block0 -> block1 -> compensate (num_hr_blocks=2)
    * PHR-2 (P3 feedback):  H0 -> block0 -> +P3 -> block1 -> compensate (2 blocks)
    * PHR-3 (P3+P4):        H0 -> block0 -> +P3 -> block1 -> +P4 -> block2 (3 blocks)
    """

    def __init__(self, c_p2, c_p3, c_p4, hr_channels=32, num_hr_blocks=1,
                 use_p3_feedback=False, use_p4_feedback=False, alpha_max=1.0, beta_max=1.0):
        super().__init__()
        c2, c3, c4 = int(c_p2), int(c_p3), int(c_p4)
        h = int(hr_channels)
        self.num_hr_blocks = int(num_hr_blocks)
        self.use_p3_feedback = bool(use_p3_feedback)
        self.use_p4_feedback = bool(use_p4_feedback)
        self.alpha_max = float(alpha_max)
        self.beta_max = float(beta_max)

        # High-resolution init: P2 -> H0 (1/4 scale).
        self.hr_init = ConvNormLayer(c2, h, 1, 1, act='relu')

        # Lightweight residual blocks (keep 1/4 scale throughout).
        self.hr_blocks = nn.ModuleList([HRBlock(h) for _ in range(self.num_hr_blocks)])

        # Optional P3 semantic feedback (P3 is H/8 -> upsample to H/4).
        if self.use_p3_feedback:
            self.p3_proj = nn.Conv2d(c3, h, 1, 1, bias=False)
            self.p3_norm = nn.BatchNorm2d(h)
            self.raw_beta3 = nn.Parameter(torch.zeros(1))  # bounded, zero-initialized

        # Optional P4 semantic feedback (P4 is H/16 -> upsample to H/4).
        if self.use_p4_feedback:
            self.p4_proj = nn.Conv2d(c4, h, 1, 1, bias=False)
            self.p4_norm = nn.BatchNorm2d(h)
            self.raw_beta4 = nn.Parameter(torch.zeros(1))  # bounded, zero-initialized

        # Delayed compensation: H/4 -> H/8 (downsample), then h -> c_p3 (project).
        # The downsample and projection are normally initialized; only ``gamma`` is
        # zero-initialized, so ``gamma`` receives a non-zero gradient on the first
        # backward while the final output remains exactly P3 at init.
        self.downsample = nn.Conv2d(h, h, 3, 2, 1, bias=False)
        self.down_norm = nn.BatchNorm2d(h)
        self.down_act = nn.ReLU(inplace=False)
        self.delta = nn.Conv2d(h, c3, 1, 1, bias=True)

        # Bounded, learnable residual scale, initialized to zero -> identity at init.
        self.raw_gamma = nn.Parameter(torch.zeros(1))

    def _feedback(self, p3, p4, target_size):
        """Return the list of feedback tensors in the fixed order [P3, P4]."""
        feedbacks = []
        if self.use_p3_feedback:
            fb = F.interpolate(self.p3_norm(self.p3_proj(p3)),
                               size=target_size, mode='bilinear', align_corners=False)
            beta3 = self.beta_max * torch.tanh(self.raw_beta3)
            feedbacks.append(beta3 * fb)
        if self.use_p4_feedback:
            fb = F.interpolate(self.p4_norm(self.p4_proj(p4)),
                               size=target_size, mode='bilinear', align_corners=False)
            beta4 = self.beta_max * torch.tanh(self.raw_beta4)
            feedbacks.append(beta4 * fb)
        return feedbacks

    def forward(self, x):
        p2, p3, p4 = x

        h = self.hr_init(p2)                                    # H0 @ 1/4
        feedbacks = self._feedback(p3, p4, target_size=h.shape[2:])

        fi = 0
        for i in range(self.num_hr_blocks):
            h = self.hr_blocks[i](h)
            if fi < len(feedbacks):
                h = h + feedbacks[fi]
                fi += 1

        d = self.down_act(self.down_norm(self.downsample(h)))   # 1/4 -> 1/8
        delta = self.delta(d)                                   # -> c_p3
        gamma = self.alpha_max * torch.tanh(self.raw_gamma)     # bounded, zero-init
        return p3 + gamma * delta
