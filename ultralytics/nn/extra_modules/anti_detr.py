"""Compatibility layers for model definitions imported from the anti-detr project."""

from .block import SBA
from .transformer import TransformerEncoderLayer_DHSA

__all__ = ("HIFI", "MCAM")


class HIFI(TransformerEncoderLayer_DHSA):
    """anti-detr name for the histogram-attention encoder used by this project."""


class MCAM(SBA):
    """Batch-preserving two-scale fusion compatible with anti-detr's MCAM YAML contract.

    The supplied anti-detr project references ``MCAM`` in ``rtdetr-MRFPN.yaml`` but
    does not include an MCAM implementation.  RTDETR-main's SBA layer has the same
    required interface: it consumes two BCHW feature maps and emits one BCHW map,
    preserving the batch dimension and using the second input's spatial scale and
    channel count.
    """
