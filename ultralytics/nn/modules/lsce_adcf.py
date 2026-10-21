"""LSCE and ADCF pluggable modules for the RT-DETR-R18 backbone/neck improvements.

LSCE (Local Structural Contrast Encoding) enhances the P2 backbone feature with a
multi-scale ring-background contrast statistic. The core idea is to express every
pixel by how its *central* appearance differs from its *surrounding* background,
so that small, low-contrast targets (e.g. tiny UAVs against sky) are amplified
relative to their local context. A learned background-complexity gate (gradient +
local variance) then suppresses the contrast response in textureless regions where
contrast is uninformative, and a bounded, zero-initialized residual *scale* makes
the module an exact identity mapping at initialization (output == input).

ADCF (Asymmetric Disagreement-aware Cross-scale Fusion) fuses the P3 neck feature
(F3) with the P4 lateral feature (F4). A single shared evidence head reads both
scales and produces two calibrated evidence maps S3/S4; their *agreement* A=S3*S4
routes the coarse-scale semantic cue through one path, while their shallow-only
*disagreement* relu(S3-S4) routes the fine-scale local cue through a second,
independent path. Two independent, bounded, zero-initialized scale factors gate the
two paths, so at initialization the module is an exact identity (F3_enhanced == F3)
and the two paths are free to learn complementary contributions.

Both modules are:
    * pluggable: registered by name in parse_model via ``globals()[m]``;
    * multi-input / single-input: declared with a ``from`` list in the YAML;
    * AMP/FP16-safe: fixed kernels are cast to the input dtype on the fly and all
      normalizations use a numerically stable epsilon;
    * gradient-preserving: no ``.detach()`` is ever used on the residual path;
    * free of heavy dependencies (torch only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import ConvNormLayer
from .baps_rdcf import FixedDepthwiseConv, _sobel_x, _sobel_y

__all__ = ('LSCE', 'ADCF')


def _avg_pool_replicate(x, kernel_size):
    """Replicate-padded average pooling that preserves the spatial size.

    Zero-padding would bias the border means toward zero and produce spurious ring
    contrast at the image boundary; replicate padding keeps every border window
    "full", so ``25*avg5 - 9*avg3`` and ``81*avg9 - 9*avg3`` are exact ring sums
    with no boundary artifact.
    """
    p = kernel_size // 2
    x = F.pad(x, (p, p, p, p), mode='replicate')
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=0)


def _ring_mean(avg_big, avg_center, big_area, center_area):
    """Mean of the ring = (area_big*mean_big - area_center*mean_center) / (area_big - area_center)."""
    return (big_area * avg_big - center_area * avg_center) / (big_area - center_area)


class LSCE(nn.Module):
    """Local Structural Contrast Encoding for the P2 backbone feature.

    Input (from a YAML ``from`` list ``[-1]``):
        P2: [B, channels, H/4, W/4]   (64 channels for R18)
    Output:
        P2_enhanced: [B, channels, H/4, W/4]

    Pipeline:
        1. F = 1x1 conv(P2)                    -> [B, h, H/4, W/4]
        2. F_center = PW(ReLU(BN(DW(F))))      -> [B, h, H/4, W/4]  (central structure)
        3. ring5 / ring9 = multi-scale ring background means of F
        4. C5 = (F_center-ring5)/(|F_center|+|ring5|+eps); C9 likewise
        5. contrast_feature = 1x1 conv(concat(C5, C9))
        6. gate = sigmoid(1x1(ReLU(1x1(concat(C5, C9, grad, var)))))
        7. delta = projection(gate * contrast_feature)
        8. out = x + alpha * tanh(delta), alpha = alpha_max * tanh(raw_alpha)
    """

    def __init__(self, channels, hidden_channels=32, alpha_max=1.0, multi_scale=True, use_complexity=True):
        super().__init__()
        c = int(channels)
        h = int(hidden_channels)
        self.eps = 1e-4
        self.alpha_max = float(alpha_max)
        self.multi_scale = bool(multi_scale)
        self.use_complexity = bool(use_complexity)
        n_scale = 2 if self.multi_scale else 1

        # 1. Center-structure branch.
        self.reduce = ConvNormLayer(c, h, 1, 1, act='relu')          # F
        self.center_dw = nn.Conv2d(h, h, 3, 1, 1, groups=h, bias=False)
        self.center_bn = nn.BatchNorm2d(h)
        self.center_pw = ConvNormLayer(h, h, 1, 1, act='relu')       # F_center

        # 5. Contrast fusion: concat(C5[, C9]) -> h.
        self.contrast_fuse = ConvNormLayer(h * n_scale, h, 1, 1, act='relu')

        # 6. Background-complexity gate (optional).
        if self.use_complexity:
            self.grad_x = FixedDepthwiseConv(h, _sobel_x())
            self.grad_y = FixedDepthwiseConv(h, _sobel_y())
        # gate input = contrast(s) [+ grad + var].
        gate_in = h * n_scale + (2 * h if self.use_complexity else 0)
        self.gate = nn.Sequential(
            ConvNormLayer(gate_in, h, 1, 1, act='relu'),
            nn.Conv2d(h, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # 7. Residual projection. Normal init so that raw_alpha receives a nonzero
        #    gradient at the first backward (ReZero: zero-init the *scale*, not the
        #    projection). Identity at init is guaranteed by raw_alpha == 0 below.
        self.delta = nn.Conv2d(h, c, 1, bias=True)

        # 8. Bounded, learnable scale factor, initialized to zero -> identity at init.
        self.raw_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f = self.reduce(x)                                            # [B, h, H, W]

        # Center structure.
        f_center = self.center_pw(F.relu(self.center_bn(self.center_dw(f))))

        # Multi-scale ring background means (replicate-padded, boundary-safe).
        avg3 = _avg_pool_replicate(f, 3)
        ring5 = _ring_mean(_avg_pool_replicate(f, 5), avg3, 25.0, 9.0)

        # Local structural contrast.
        c5 = (f_center - ring5) / (f_center.abs() + ring5.abs() + self.eps)
        if self.multi_scale:
            ring9 = _ring_mean(_avg_pool_replicate(f, 9), avg3, 81.0, 9.0)
            c9 = (f_center - ring9) / (f_center.abs() + ring9.abs() + self.eps)
            contrast_feature = self.contrast_fuse(torch.cat([c5, c9], dim=1))
            gate_in = [c5, c9]
        else:
            contrast_feature = self.contrast_fuse(c5)
            gate_in = [c5]

        # Background complexity (gradient magnitude + local variance), optional.
        if self.use_complexity:
            grad = self.grad_x(f).abs() + self.grad_y(f).abs()
            var = _avg_pool_replicate(f * f, 3) - avg3 * avg3
            gate_in.extend([grad, var])

        gate = self.gate(torch.cat(gate_in, dim=1))                   # [B, 1, H, W]

        delta = self.delta(gate * contrast_feature)                   # [B, c, H, W]
        alpha = self.alpha_max * torch.tanh(self.raw_alpha)
        return x + alpha * torch.tanh(delta)


class ADCF(nn.Module):
    """Asymmetric Disagreement-aware Cross-scale Fusion for the P3 neck feature.

    Input (from a YAML ``from`` list ``[f3_idx, f4_idx]``):
        F3: [B, c_p3, H/8, W/8]   (P3 neck feature, 256 channels for R18)
        F4: [B, c_p4, H/16, W/16] (P4 lateral feature, 256 channels for R18)
    Output:
        F3_enhanced: [B, c_p3, H/8, W/8]

    Pipeline:
        1. D3 = 1x1 conv(F3), D4 = upsample(1x1 conv(F4)) -> [B, h, H/8, W/8]
        2. A single shared evidence head maps D3/D4 -> S3/S4 in (0, 1). Sharing the
           readout keeps the two cross-scale evidence maps *calibrated*: the only
           differences entering S3-S4 are real feature differences, not per-head
           output bias (addressing the cross-scale calibration concern).
        3. agreement A = S3*S4            -> semantic path over D4
        4. disagreement relu(S3-S4)       -> local path over D3
        5. F3_enhanced = F3 + a_s*proj_sem(A*phi_sem(D4)) + a_l*proj_loc(D*phi_loc(D3))
           with a_s, a_l bounded and zero-initialized (identity at init).
    """

    def __init__(self, c_p3, c_p4, hidden_channels=64, alpha_max=1.0, use_center_supervision=False,
                 use_evidence=True):
        super().__init__()
        c3 = int(c_p3)
        c4 = int(c_p4)
        h = int(hidden_channels)
        self.alpha_max = float(alpha_max)
        # Reserved for the optional E6 center-supervision ablation. When False the
        # module adds no auxiliary loss and behaves as a pure feature fusion block.
        self.use_center_supervision = bool(use_center_supervision)
        # use_evidence=False selects the E5 plain learned cross-scale gated fusion
        # (a single gate over concat(D3, D4)) instead of the disagreement-aware
        # asymmetric evidence path.
        self.use_evidence = bool(use_evidence)

        # 1. Cross-scale projections into a shared latent space.
        self.d3_proj = ConvNormLayer(c3, h, 1, 1, act='relu')
        self.d4_proj = ConvNormLayer(c4, h, 1, 1, act='relu')

        if self.use_evidence:
            # 2. Shared evidence head (calibrated S3/S4).
            self.evidence = nn.Sequential(
                ConvNormLayer(h, h, 3, 1, act='relu'),
                nn.Conv2d(h, 1, 1, bias=True),
            )

            # 3-4. Two asymmetric paths.
            self.phi_semantic = ConvNormLayer(h, h, 3, 1, act='relu')
            self.phi_local = ConvNormLayer(h, h, 3, 1, act='relu')
        else:
            # E5: plain learned gate over concat(D3, D4) -> fused feature.
            self.gate = nn.Sequential(
                ConvNormLayer(h * 2, h, 1, 1, act='relu'),
                nn.Conv2d(h, 1, 1, bias=True),
                nn.Sigmoid(),
            )
            self.phi_fused = ConvNormLayer(h, h, 3, 1, act='relu')

        # 5. Residual projections. Normal init so the scale factors receive a
        #    nonzero gradient at the first backward (ReZero-style); identity at init
        #    is guaranteed by the zero-initialized scale factors below.
        self.delta_semantic = nn.Conv2d(h, c3, 1, bias=True)
        self.delta_local = nn.Conv2d(h, c3, 1, bias=True)

        # Independent bounded scale factors, initialized to zero -> identity at init.
        self.raw_alpha_semantic = nn.Parameter(torch.zeros(1))
        self.raw_alpha_local = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f3, f4 = x

        d3 = self.d3_proj(f3)                                          # [B, h, H/8, W/8]
        d4 = self.d4_proj(f4)                                          # [B, h, H/16, W/16]
        d4 = F.interpolate(d4, size=d3.shape[2:], mode='bilinear', align_corners=False)

        a_s = self.alpha_max * torch.tanh(self.raw_alpha_semantic)
        a_l = self.alpha_max * torch.tanh(self.raw_alpha_local)

        if self.use_evidence:
            # Calibrated evidence maps (shared readout).
            s3 = torch.sigmoid(self.evidence(d3))                      # [B, 1, H, W]
            s4 = torch.sigmoid(self.evidence(d4))                      # [B, 1, H, W]

            agreement = s3 * s4                                        # [B, 1, H, W]
            disagreement = torch.relu(s3 - s4)                         # [B, 1, H, W]

            sem = agreement * self.phi_semantic(d4)                    # [B, h, H, W]
            loc = disagreement * self.phi_local(d3)                    # [B, h, H, W]

            delta_semantic = self.delta_semantic(sem)                  # [B, c3, H, W]
            delta_local = self.delta_local(loc)                        # [B, c3, H, W]
            return f3 + a_s * delta_semantic + a_l * delta_local

        # E5: plain learned gated fusion of the two cross-scale projections.
        g = self.gate(torch.cat([d3, d4], dim=1))                      # [B, 1, H, W]
        fused = self.phi_fused(g * d3 + (1.0 - g) * d4)                # [B, h, H, W]
        delta = self.delta_semantic(fused)                             # [B, c3, H, W]
        return f3 + a_s * delta
