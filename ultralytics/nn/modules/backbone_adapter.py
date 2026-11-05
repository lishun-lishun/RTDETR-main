"""Uniform multi-scale adapter for the eight RT-DETR backbone candidates.

The adapter deliberately performs no feature fusion.  It selects the source
backbone tensors at reductions 8, 16 and 32, then applies independent 1x1
projections to the channels expected by the original RT-DETR neck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ("BackboneAdapter", "BACKBONE_SPECS", "resolve_timm_model_name")


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    family: str
    candidates: tuple[str, ...]
    source: str
    expected_reductions: tuple[int, int, int] = (8, 16, 32)


BACKBONE_SPECS = {
    "hgnetv2_b2": BackboneSpec("hgnetv2_b2", "timm", ("hgnetv2_b2",), "timm"),
    "convnextv2_pico": BackboneSpec("convnextv2_pico", "timm", ("convnextv2_pico",), "timm"),
    "fasternet_t1": BackboneSpec("fasternet_t1", "timm", ("fasternet_t1",), "timm"),
    "repvit_m1_1": BackboneSpec("repvit_m1_1", "timm", ("repvit_m1_1",), "timm"),
    "efficientvit_b1": BackboneSpec(
        "efficientvit_b1", "timm", ("efficientvit_b1", "efficientvit_b1.r224_in1k"), "timm"),
    "pvt_v2_b1": BackboneSpec("pvt_v2_b1", "timm", ("pvt_v2_b1",), "timm"),
    "lsknet_t": BackboneSpec("lsknet_t", "local", (), "official LSKNet repository adapter"),
    "pkinetv2_t": BackboneSpec("pkinetv2_t", "local", (), "official PKINet repository adapter"),
}


def resolve_timm_model_name(name: str) -> tuple[str, tuple[str, ...]]:
    """Resolve a base candidate against the installed timm registry.

    timm frequently appends checkpoint suffixes such as ``.in1k``.  We first
    prefer exact names, then names whose stem equals the requested candidate.
    The returned tuple contains the selected name and the full registry list,
    which makes missing-model errors reproducible in diagnostics.
    """
    import timm

    spec = BACKBONE_SPECS.get(name)
    candidates = spec.candidates if spec else (name,)
    available = tuple(timm.list_models())
    available_set = set(available)
    for candidate in candidates:
        if candidate in available_set:
            return candidate, available
    for candidate in candidates:
        stem_matches = [item for item in available if item.split(".", 1)[0] == candidate]
        if stem_matches:
            return sorted(stem_matches)[0], available
    similar = [item for item in available if any(token in item for token in candidates)]
    raise RuntimeError(
        f"timm model {name!r} is unavailable. Tried {candidates}; "
        f"similar installed names are {similar[:20]}. Install a compatible timm version "
        "or update the explicit alias mapping."
    )


class BackboneAdapter(nn.Module):
    """Return P3/P4/P5 with fixed channels and checked spatial reductions."""

    def __init__(self, model_name: str, pretrained: bool = False,
                 in_channels: int = 3, output_channels: Sequence[int] = (128, 256, 512)):
        super().__init__()
        if model_name not in BACKBONE_SPECS:
            raise ValueError(f"unknown backbone {model_name!r}; choose from {tuple(BACKBONE_SPECS)}")
        if in_channels != 3:
            raise ValueError(f"BackboneAdapter expects RGB input (3 channels), got {in_channels}")
        if tuple(output_channels) != (128, 256, 512):
            raise ValueError("RT-DETR adapter output channels must be (128, 256, 512)")
        self.model_name = model_name
        self.source = BACKBONE_SPECS[model_name].source
        self.output_channels = (128, 256, 512)
        self.out_strides = (8, 16, 32)
        self.resolved_model_name = None
        self.registry_models = ()
        self.timm_version = None

        if self.source == "timm":
            resolved, registry = resolve_timm_model_name(model_name)
            import timm
            self.resolved_model_name = resolved
            self.registry_models = registry
            self.timm_version = getattr(timm, "__version__", "unknown")
            self.backbone = timm.create_model(resolved, pretrained=pretrained, features_only=True)
            reductions = tuple(int(value) for value in self.backbone.feature_info.reduction())
            channels = tuple(int(value) for value in self.backbone.feature_info.channels())
            self._source_outputs = self._select_indices(reductions, model_name)
            source_channels = tuple(channels[index] for index in self._source_outputs)
        elif model_name == "lsknet_t":
            from .lsknet_backbone import create_lsknet_t
            self.backbone = create_lsknet_t()
            # Official LSKNet-T emits reductions 4/8/16/32 and channels
            # 32/64/160/256.
            self._source_outputs = (1, 2, 3)
            source_channels = (64, 160, 256)
        else:
            from .pkinetv2_backbone import create_pkinetv2_t
            self.backbone = create_pkinetv2_t()
            # PKINet-v2-T emits stages at reductions 4/8/16/32 with channels
            # 32/64/160/256; select the last three stages.
            self._source_outputs = (1, 2, 3)
            source_channels = (64, 160, 256)

        self.source_channels = tuple(source_channels)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            for in_channels, out_channels in zip(self.source_channels, self.output_channels)
        ])

    @staticmethod
    def _select_indices(reductions: tuple[int, ...], model_name: str) -> tuple[int, int, int]:
        selected = []
        for target in (8, 16, 32):
            matches = [index for index, reduction in enumerate(reductions) if reduction == target]
            if not matches:
                raise RuntimeError(
                    f"timm backbone {model_name!r} reductions are {reductions}; "
                    f"no unique reduction={target} feature exists")
            selected.append(matches[-1])
        if len(set(selected)) != 3:
            raise RuntimeError(f"timm backbone {model_name!r} mapped duplicate feature indices {selected}")
        return tuple(selected)

    @staticmethod
    def _as_feature_list(outputs):
        if isinstance(outputs, dict):
            outputs = tuple(outputs.values())
        if not isinstance(outputs, (tuple, list)):
            raise RuntimeError(f"backbone returned {type(outputs).__name__}; expected a feature sequence")
        return tuple(outputs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self._as_feature_list(self.backbone(x))
        if max(self._source_outputs) >= len(outputs):
            raise RuntimeError(
                f"{self.model_name} returned {len(outputs)} features, "
                f"but selected indices are {self._source_outputs}")
        selected = [outputs[index] for index in self._source_outputs]
        projected = []
        for index, (feature, projection, expected_stride) in enumerate(
                zip(selected, self.projections, self.out_strides)):
            if feature.ndim != 4:
                raise RuntimeError(f"{self.model_name} feature {index} has shape {tuple(feature.shape)}, expected NCHW")
            expected_size = (x.shape[-2] // expected_stride, x.shape[-1] // expected_stride)
            if feature.shape[-2:] != expected_size:
                raise RuntimeError(
                    f"{self.model_name} feature {index} has spatial size {tuple(feature.shape[-2:])}; "
                    f"expected stride {expected_stride} size {expected_size}")
            projected.append(projection(feature))
        return tuple(projected)

