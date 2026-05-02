from __future__ import annotations

import torch
from torch import nn


class Conv2dSameTemporal(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = False) -> None:
        super().__init__()
        pad_left = kernel_size // 2
        pad_right = kernel_size - 1 - pad_left
        self.net = nn.Sequential(
            nn.ZeroPad2d((pad_left, pad_right, 0, 0)),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGNet(nn.Module):
    """EEGNet-8,2 style classifier with a 128-dimensional embedding head."""

    def __init__(
        self,
        n_channels: int = 5,
        n_samples: int = 512,
        n_classes: int = 3,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            Conv2dSameTemporal(1, f1, kernel_length, bias=False),
            nn.BatchNorm2d(f1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                f1,
                f1 * depth_multiplier,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat_dim = self._features(dummy).flatten(1).shape[1]
        self.embedding = nn.Sequential(
            nn.Linear(flat_dim, embedding_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.separable(self.depthwise(self.temporal(x)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x).flatten(1)
        return self.embedding(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


def load_encoder_weights(model: EEGNet, checkpoint_state: dict) -> None:
    compatible = {
        key: value
        for key, value in checkpoint_state.items()
        if not key.startswith("classifier.")
        and key in model.state_dict()
        and model.state_dict()[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
