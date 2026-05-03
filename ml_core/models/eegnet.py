"""EEGNet-8,2 (Lawhern et al., 2018) with a 128-d embedding head.

Input shape:  (B, 1, C, T)
Outputs:      logits of shape (B, n_classes); embedding of shape (B, embed_dim).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _DepthwiseConv2d(nn.Conv2d):
    """Depthwise conv with a depth multiplier (groups == in_channels)."""

    def __init__(self, in_channels: int, depth_multiplier: int, kernel_size, **kwargs):
        super().__init__(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size,
            groups=in_channels,
            bias=False,
            **kwargs,
        )


class _SeparableConv2d(nn.Module):
    """Depthwise + pointwise (separable) conv block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding="same", groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, (1, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class EEGNet(nn.Module):
    """EEGNet-8,2 with classifier and 128-dim embedding head."""

    def __init__(
        self,
        n_classes: int = 3,
        n_channels: int = 5,
        n_samples: int = 512,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.25,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim

        # Block 1: temporal conv -> depthwise spatial conv -> BN -> ELU -> avgpool -> dropout
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            _DepthwiseConv2d(F1, depth_multiplier=D, kernel_size=(n_channels, 1)),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: separable conv -> BN -> ELU -> avgpool -> dropout
        self.block2 = nn.Sequential(
            _SeparableConv2d(F1 * D, F2, (1, 16)),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Compute backbone output dim with a dummy pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            backbone = self.block2(self.block1(dummy))
            self._flat = backbone.numel()

        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.features(x))

    def forward(self, x: torch.Tensor, *, return_embedding: bool = False):
        emb = self.embedding(x)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits

    def replace_classifier(self, n_classes: int) -> None:
        """Drop and replace the head for transfer learning."""
        self.classifier = nn.Linear(self.embed_dim, n_classes)
        self.n_classes = n_classes
