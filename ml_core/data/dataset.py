"""PyTorch Dataset wrapper around a Delta-loaded epoch DataFrame."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .normalize import NormStats, apply_norm_stats
from .schema import EXPECTED_FEATURE_LEN, N_CHANNELS, N_SAMPLES

ShapeMode = Literal["bcnt", "cnt"]
"""Output shape:
    "bcnt": (1, C, T)  - convolutional EEGNet/ShallowConvNet input.
    "cnt":  (C, T)     - Conformer-style input (no extra channel dim).
"""


class EpochDataset(Dataset):
    """Wraps a DataFrame of epochs into ``(x, y, meta)`` torch samples.

    Parameters
    ----------
    df: pandas DataFrame from ``read_epochs``.
    norm_stats: optional per-channel z-score stats to apply.
    shape_mode: "bcnt" -> (1, C, T) for CNN models; "cnt" -> (C, T).
    return_meta: also yield a metadata dict (epoch_id, subject_id, dataset).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        norm_stats: NormStats | None = None,
        shape_mode: ShapeMode = "bcnt",
        return_meta: bool = False,
    ) -> None:
        if shape_mode not in ("bcnt", "cnt"):
            raise ValueError(shape_mode)
        self._features: list[np.ndarray] = [
            np.asarray(f, dtype=np.float32) for f in df["features"].tolist()
        ]
        self._labels: np.ndarray = df["label_code"].to_numpy(dtype=np.int64)
        self._epoch_ids: list[str] = df["epoch_id"].astype(str).tolist()
        self._subject_ids: list[str] = df["subject_id"].astype(str).tolist()
        self._datasets: list[str] = df["dataset"].astype(str).tolist()
        self.norm_stats = norm_stats
        self.shape_mode = shape_mode
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        feat = self._features[idx]
        if feat.shape != (EXPECTED_FEATURE_LEN,):
            raise ValueError(
                f"Bad feature shape at idx={idx}: {feat.shape} != ({EXPECTED_FEATURE_LEN},)"
            )
        x = feat.reshape(N_CHANNELS, N_SAMPLES)
        if self.norm_stats is not None:
            x = apply_norm_stats(x, self.norm_stats)
        if self.shape_mode == "bcnt":
            x = x[None, ...]  # (1, C, T)
        x_t = torch.from_numpy(np.ascontiguousarray(x))
        y_t = torch.tensor(int(self._labels[idx]), dtype=torch.long)
        if self.return_meta:
            meta = {
                "epoch_id": self._epoch_ids[idx],
                "subject_id": self._subject_ids[idx],
                "dataset": self._datasets[idx],
            }
            return x_t, y_t, meta
        return x_t, y_t

    def class_counts(self) -> dict[int, int]:
        unique, counts = np.unique(self._labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def subject_ids(self) -> list[str]:
        return list(self._subject_ids)
