"""Euclidean Alignment and train-fitted channel z-score normalization."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import EXPECTED_FEATURE_LEN, N_CHANNELS, N_SAMPLES


def epochs_to_array(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert a DataFrame or array into shape `(n_epochs, 5, 512)`."""
    if isinstance(data, pd.DataFrame):
        features = np.asarray(data["features"].tolist(), dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != EXPECTED_FEATURE_LEN:
            raise ValueError(f"Expected features shape (N, {EXPECTED_FEATURE_LEN}), got {features.shape}")
        return features.reshape(-1, N_CHANNELS, N_SAMPLES)

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == EXPECTED_FEATURE_LEN:
        return arr.reshape(-1, N_CHANNELS, N_SAMPLES)
    if arr.ndim == 3 and arr.shape[1:] == (N_CHANNELS, N_SAMPLES):
        return arr
    if arr.ndim == 4 and arr.shape[1:] == (1, N_CHANNELS, N_SAMPLES):
        return arr[:, 0, :, :]
    raise ValueError(f"Expected array shape (N, 5, 512), (N, 1, 5, 512), or (N, 2560); got {arr.shape}")


class EuclideanAligner:
    """Euclidean Alignment transform fitted on training epochs only.

    For each epoch `X` with shape `(5, 512)`, covariance is `X @ X.T / 512`.
    The aligner stores `R_bar^{-1/2}` computed from the mean covariance matrix
    and applies `R_bar^{-1/2} @ X` to any split.
    """

    def __init__(self, r_bar_inv_sqrt: np.ndarray | None = None, epsilon: float = 1e-6) -> None:
        self.r_bar_inv_sqrt = r_bar_inv_sqrt.astype(np.float32) if r_bar_inv_sqrt is not None else None
        self.epsilon = float(epsilon)
        self.fitted_on_n_epochs = 0

    def fit(self, train_epochs_array: pd.DataFrame | np.ndarray) -> "EuclideanAligner":
        """Fit alignment statistics from training epochs."""
        epochs = epochs_to_array(train_epochs_array).astype(np.float64)
        if epochs.shape[0] == 0:
            raise ValueError("Cannot fit EuclideanAligner on zero epochs.")

        covariances = np.matmul(epochs, np.transpose(epochs, (0, 2, 1))) / float(N_SAMPLES)
        r_bar = covariances.mean(axis=0)
        r_bar = r_bar + self.epsilon * np.eye(N_CHANNELS, dtype=np.float64)
        chol = np.linalg.cholesky(r_bar)
        self.r_bar_inv_sqrt = np.linalg.inv(chol).astype(np.float32)
        self.fitted_on_n_epochs = int(epochs.shape[0])
        return self

    def transform(self, epochs_array: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply the fitted alignment transform to epochs."""
        if self.r_bar_inv_sqrt is None:
            raise ValueError("EuclideanAligner must be fitted before transform().")
        epochs = epochs_to_array(epochs_array)
        return np.einsum("ij,njt->nit", self.r_bar_inv_sqrt, epochs).astype(np.float32)

    def fit_transform(self, train_epochs_array: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit on training epochs and return transformed training epochs."""
        return self.fit(train_epochs_array).transform(train_epochs_array)

    def save(self, path: str | Path) -> None:
        """Serialize the aligner to a NumPy `.npz` file."""
        if self.r_bar_inv_sqrt is None:
            raise ValueError("Cannot save an unfitted EuclideanAligner.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            r_bar_inv_sqrt=self.r_bar_inv_sqrt,
            epsilon=np.asarray(self.epsilon, dtype=np.float32),
            fitted_on_n_epochs=np.asarray(self.fitted_on_n_epochs, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EuclideanAligner":
        """Load a serialized aligner from a NumPy `.npz` file."""
        blob = np.load(Path(path), allow_pickle=False)
        obj = cls(blob["r_bar_inv_sqrt"], epsilon=float(blob["epsilon"]))
        obj.fitted_on_n_epochs = int(blob["fitted_on_n_epochs"])
        return obj


@dataclass(frozen=True)
class NormStats:
    """Per-channel z-score statistics fitted on training epochs only."""

    mean: np.ndarray
    std: np.ndarray
    fitted_on_n_epochs: int

    def to_json(self, path: str | Path) -> None:
        """Write normalization statistics to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "fitted_on_n_epochs": int(self.fitted_on_n_epochs),
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def save(self, path: str | Path) -> None:
        """Alias for `to_json` for callers that prefer save/load naming."""
        self.to_json(path)

    @classmethod
    def from_json(cls, path: str | Path) -> "NormStats":
        """Load normalization statistics from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            fitted_on_n_epochs=int(payload["fitted_on_n_epochs"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        """Alias for `from_json` for callers that prefer save/load naming."""
        return cls.from_json(path)


def compute_norm_stats(train_epochs_array: pd.DataFrame | np.ndarray) -> NormStats:
    """Compute per-channel z-score stats from training epochs only."""
    epochs = epochs_to_array(train_epochs_array)
    mean = epochs.mean(axis=(0, 2)).astype(np.float32)
    std = epochs.std(axis=(0, 2)).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return NormStats(mean=mean, std=std, fitted_on_n_epochs=int(epochs.shape[0]))


def apply_norm_stats(epochs_array: pd.DataFrame | np.ndarray, norm_stats: NormStats) -> np.ndarray:
    """Apply train-derived per-channel z-score statistics to epochs."""
    epochs = epochs_to_array(epochs_array)
    mean = norm_stats.mean.reshape(1, N_CHANNELS, 1)
    std = norm_stats.std.reshape(1, N_CHANNELS, 1)
    return ((epochs - mean) / std).astype(np.float32)
