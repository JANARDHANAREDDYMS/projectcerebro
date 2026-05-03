"""PyTorch Dataset for Stage 2 ProjectCerebro epochs."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .normalize import EuclideanAligner, NormStats, apply_norm_stats, epochs_to_array
from .schema import LABEL_MAP, validate_schema


class EpochDataset(Dataset):
    """Dataset that returns EEG tensors and labels from an epoch DataFrame.

    By default the dataset returns `(X, y)` for trainer compatibility. With
    `return_meta=True`, it returns `(X, y, metadata)`. `X` is shaped
    `(1, 5, 512)` for EEGNet/ShallowConvNet.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        aligner: EuclideanAligner | None = None,
        norm_stats: NormStats | None = None,
        augment: bool = False,
        return_meta: bool = False,
        shape_mode: str = "bcnt",
        noise_std: float = 0.01,
    ) -> None:
        validate_schema(df)
        if shape_mode != "bcnt":
            raise ValueError("Only shape_mode='bcnt' is supported.")

        self.df = df.reset_index(drop=True).copy()
        epochs = epochs_to_array(self.df)
        if aligner is not None:
            epochs = aligner.transform(epochs)
        if norm_stats is not None:
            epochs = apply_norm_stats(epochs, norm_stats)

        self._x = epochs[:, None, :, :].astype(np.float32)
        self._labels = self.df["label_code"].astype(int).to_numpy()
        self.augment = bool(augment)
        self.return_meta = bool(return_meta)
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        """Return the number of epochs."""
        return int(len(self.df))

    def __getitem__(self, idx: int):
        """Return one epoch tensor, label, and optionally metadata."""
        x = self._x[idx]
        if self.augment:
            x = x + np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(int(self._labels[idx]), dtype=torch.long)

        if not self.return_meta:
            return x_tensor, y_tensor

        row = self.df.iloc[idx]
        meta: dict[str, Any] = {
            "epoch_id": str(row.get("epoch_id", "")),
            "subject_id": str(row.get("subject_id", "")),
            "dataset": str(row.get("dataset", "")),
            "is_rest_synthetic": bool(row.get("is_rest_synthetic", False)),
        }
        return x_tensor, y_tensor, meta

    def class_counts(self) -> dict[str, int]:
        """Return label counts keyed by class name."""
        counts = np.bincount(self._labels, minlength=len(LABEL_MAP))
        return {LABEL_MAP[i]: int(counts[i]) for i in sorted(LABEL_MAP)}
