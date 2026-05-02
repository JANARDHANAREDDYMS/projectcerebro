from __future__ import annotations

import torch
from torch import nn


class Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(x)


class SafeLog(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    """Compact EEG baseline for inputs shaped (batch, 1, 5, 512)."""

    def __init__(
        self,
        n_channels: int = 5,
        n_samples: int = 512,
        n_classes: int = 3,
        temporal_filters: int = 20,
        spatial_filters: int = 20,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 25), bias=False),
            nn.Conv2d(temporal_filters, spatial_filters, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(spatial_filters),
            Square(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 8)),
            SafeLog(),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat_dim = self.features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(1))
