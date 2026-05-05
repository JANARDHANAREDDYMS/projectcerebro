"""Compact EEG Conformer for motor-imagery epoch classification.

Input shape:  (B, 1, C, T), with C=5 and T=512 by default.
Outputs:      logits of shape (B, n_classes); optional embedding (B, embed_dim).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGConformer(nn.Module):
    """Convolutional tokenizer plus Transformer encoder for EEG decoding.

    The convolutional stem learns local temporal filters and a spatial projection
    across EEG channels. The Transformer then models longer-range dependencies
    across the resulting sequence of temporal tokens.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_channels: int = 5,
        n_samples: int = 512,
        embed_dim: int = 64,
        conv_filters: int = 40,
        kernel_length: int = 25,
        pool_length: int = 8,
        n_heads: int = 4,
        depth: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads.")

        self.n_classes = n_classes
        self.embed_dim = embed_dim

        # Replace stem in eegconformer.py with:
        self.stem = nn.Sequential(
            # Temporal filter per channel
            nn.Conv2d(1, 40, 
                    kernel_size=(1, 25),
                    padding=(0, 12),
                    bias=False),
            nn.BatchNorm2d(40),
            nn.ELU(),
            # Depthwise spatial - keep all channels
            nn.Conv2d(40, 40,
                    kernel_size=(n_channels, 1),
                    groups=40,
                    bias=False),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        # After stem: (B, 40, 1, 64) -> squeeze -> (B, 40, 64)
        # 64 tokens each with 40 features
        self.token_projection = nn.Linear(conv_filters, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

        # Validate shape at construction time and catches invalid pooling choices.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            _ = self.embedding(dummy)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return temporal tokens with shape `(B, T_tokens, embed_dim)`."""
        x = self.stem(x)          # (B, conv_filters, 1, T_tokens)
        x = x.squeeze(2)          # (B, conv_filters, T_tokens)
        x = x.transpose(1, 2)     # (B, T_tokens, conv_filters)
        return self.token_projection(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled 128-d EEG representation."""
        x = self.tokens(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.embedding_head(x)

    def forward(self, x: torch.Tensor, *, return_embedding: bool = False):
        """Return logits, and optionally the latent embedding."""
        emb = self.embedding(x)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits

    def replace_classifier(self, n_classes: int) -> None:
        """Replace the classifier head for transfer learning or class-count changes."""
        self.classifier = nn.Linear(self.embed_dim, n_classes)
        self.n_classes = n_classes
