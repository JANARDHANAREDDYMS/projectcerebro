"""Model and preprocessing artifact loading for the serving layer."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ml_core.data.normalize import EuclideanAligner, NormStats
from ml_core.models.eegnet import EEGNet
from ml_core.models.shallowconv import ShallowConvNet

from . import config


def get_device() -> torch.device:
    """Return the best available inference device."""
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count all model parameters."""
    return int(sum(param.numel() for param in model.parameters()))


def _load_state_dict(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load a repo checkpoint into a model."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_eegnet(device: torch.device) -> EEGNet:
    """Load the pretrained EEGNet checkpoint."""
    model = EEGNet(n_classes=config.N_CLASSES, n_channels=config.N_CHANNELS, n_samples=config.N_SAMPLES)
    return _load_state_dict(model, config.EEGNET_CHECKPOINT, device)  # type: ignore[return-value]


def load_shallowconv(device: torch.device) -> ShallowConvNet:
    """Load the pretrained ShallowConvNet checkpoint."""
    model = ShallowConvNet(n_classes=config.N_CLASSES, n_channels=config.N_CHANNELS, n_samples=config.N_SAMPLES)
    return _load_state_dict(model, config.SHALLOW_CHECKPOINT, device)  # type: ignore[return-value]


def load_eegnet_artifacts() -> tuple[EuclideanAligner, NormStats]:
    """Load EEGNet preprocessing artifacts."""
    return EuclideanAligner.load(config.EEGNET_ALIGNER), NormStats.load(config.EEGNET_NORM)


def load_shallow_artifacts() -> tuple[EuclideanAligner, NormStats]:
    """Load ShallowConvNet preprocessing artifacts."""
    return EuclideanAligner.load(config.SHALLOW_ALIGNER), NormStats.load(config.SHALLOW_NORM)


def freeze_backbone_for_classifier_adaptation(model: nn.Module) -> nn.Module:
    """Freeze all model parameters except the classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        raise AttributeError("Model has no classifier attribute to unfreeze.")
    for param in classifier.parameters():
        param.requires_grad = True
    return model
