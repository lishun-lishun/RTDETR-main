"""Official LSKNet-T feature extractor wrapper used by BackboneAdapter."""

from __future__ import annotations

from ultralytics.nn.backbone.lsknet import lsknet_t


def create_lsknet_t():
    """Build the repository's official LSKNet-T implementation."""
    return lsknet_t()

