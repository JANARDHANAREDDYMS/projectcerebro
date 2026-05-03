"""Model registry."""
from .shallowconv import ShallowConvNet
from .eegnet import EEGNet

__all__ = ["ShallowConvNet", "EEGNet", "build_model"]


def build_model(name: str, **kwargs):
    name = name.lower()
    if name == "shallowconvnet":
        return ShallowConvNet(**kwargs)
    if name == "eegnet":
        return EEGNet(**kwargs)
    raise ValueError(f"Unknown model: {name}")
