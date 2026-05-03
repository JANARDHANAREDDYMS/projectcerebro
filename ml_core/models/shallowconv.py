"""ShallowConvNet (Schirrmeister et al., 2017) - frequency-domain baseline.

Input shape:  (B, 1, C, T) with C=5, T=512.
Output shape: (B, n_classes).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _square(x: torch.Tensor) -> torch.Tensor:
    return x * x


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    """Reference shallow CNN for motor-imagery EEG decoding."""

    def __init__(
        self,
        n_classes: int = 3,
        n_channels: int = 5,
        n_samples: int = 512,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        n_filters_spat: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_samples = n_samples

        self.conv_time = nn.Conv2d(1, n_filters_time, (1, filter_time_length), bias=False)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_spat, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop = nn.Dropout(dropout)

        # Compute classifier in_features from a dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat = self._features(dummy)
            self._flat = feat.numel()
        self.classifier = nn.Linear(self._flat, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = _square(x)
        x = self.pool(x)
        x = _safe_log(x)
        x = self.drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        x = x.flatten(1)
        return self.classifier(x)
