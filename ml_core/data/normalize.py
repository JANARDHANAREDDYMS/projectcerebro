"""Per-channel z-score statistics computed on TRAIN ONLY."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import EXPECTED_FEATURE_LEN, N_CHANNELS, N_SAMPLES


@dataclass
class NormStats:
    """Per-channel mean and std (length N_CHANNELS)."""

    mean: list[float]
    std: list[float]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.mean, dtype=np.float32), np.asarray(self.std, dtype=np.float32)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "NormStats":
        return cls(**json.loads(Path(path).read_text()))


def compute_norm_stats(df: pd.DataFrame, *, eps: float = 1e-6) -> NormStats:
    """Aggregate per-channel mean/std across every epoch in `df`."""
    feats = np.stack([
        np.asarray(f, dtype=np.float32).reshape(N_CHANNELS, N_SAMPLES) for f in df["features"]
    ])  # shape (N, C, T)
    # Flatten over (N, T) per channel.
    mean = feats.mean(axis=(0, 2))
    std = feats.std(axis=(0, 2))
    std = np.where(std < eps, 1.0, std)
    return NormStats(mean=mean.tolist(), std=std.tolist())


def apply_norm_stats(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Apply z-score per channel to a single epoch shaped (C, T) or (1, C, T)."""
    mean, std = stats.as_arrays()
    if x.ndim == 3 and x.shape[0] == 1:
        return ((x[0] - mean[:, None]) / std[:, None])[None, ...].astype(np.float32)
    if x.ndim == 2 and x.shape == (N_CHANNELS, N_SAMPLES):
        return ((x - mean[:, None]) / std[:, None]).astype(np.float32)
    if x.ndim == 1 and x.shape[0] == EXPECTED_FEATURE_LEN:
        x2 = x.reshape(N_CHANNELS, N_SAMPLES)
        return ((x2 - mean[:, None]) / std[:, None]).astype(np.float32)
    raise ValueError(f"Unsupported shape for normalization: {x.shape}")
