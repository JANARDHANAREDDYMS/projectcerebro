"""Model registry."""
from .shallowconv import ShallowConvNet
from .eegnet import EEGNet
from .eegconformer import EEGConformer

__all__ = ["ShallowConvNet", "EEGNet", "EEGConformer", "build_model"]


def build_model(name: str, **kwargs):
    name = name.lower()
    if name == "shallowconvnet":
        return ShallowConvNet(**kwargs)
    if name == "eegnet":
        return EEGNet(**kwargs)
    if name in {"eegconformer", "conformer", "eeg_conformer"}:
        return EEGConformer(**kwargs)
    raise ValueError(f"Unknown model: {name}")
