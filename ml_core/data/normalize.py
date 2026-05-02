from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml_core.data.schema import FEATURE_LENGTH, N_CHANNELS, N_SAMPLES


@dataclass(frozen=True)
class NormalizationStats:
    mean: list[float]
    std: list[float]

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(self.mean, dtype=np.float32).reshape(N_CHANNELS, 1)
        std = np.asarray(self.std, dtype=np.float32).reshape(N_CHANNELS, 1)
        return mean, std


def features_to_array(features: object) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.size != FEATURE_LENGTH:
        raise ValueError(f"Expected {FEATURE_LENGTH} features, got {arr.size}")
    return arr.reshape(N_CHANNELS, N_SAMPLES)


def compute_train_stats(df: pd.DataFrame) -> NormalizationStats:
    if df.empty:
        raise ValueError("Cannot compute normalization stats from an empty dataframe")
    stacked = np.stack([features_to_array(value) for value in df["features"]], axis=0)
    mean = stacked.mean(axis=(0, 2))
    std = stacked.std(axis=(0, 2))
    std = np.where(std < 1e-6, 1.0, std)
    return NormalizationStats(mean=mean.astype(float).tolist(), std=std.astype(float).tolist())


def normalize_epoch(epoch: np.ndarray, stats: NormalizationStats | None) -> np.ndarray:
    if stats is None:
        return epoch.astype(np.float32)
    mean, std = stats.to_arrays()
    return ((epoch - mean) / std).astype(np.float32)


def save_stats(stats: NormalizationStats, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")


def load_stats(path: str | Path) -> NormalizationStats:
    return NormalizationStats(**json.loads(Path(path).read_text(encoding="utf-8")))
