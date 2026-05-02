from __future__ import annotations

from torch import nn


class EEGConformer(nn.Module):
    """Placeholder for the gated Conformer milestone.

    The implementation is intentionally deferred until the EEGNet path is stable.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("EEG Conformer is gated on stable EEGNet fine-tuning.")
