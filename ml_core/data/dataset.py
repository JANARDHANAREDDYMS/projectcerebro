from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset

from ml_core.data.normalize import NormalizationStats, features_to_array, normalize_epoch
from ml_core.data.schema import validate_epoch_dataframe


@dataclass(frozen=True)
class EpochMetadata:
    epoch_id: str
    dataset: str
    subject_id: str
    session_id: str | None
    run_id: str | None
    filter_version: str
    preprocessing_version: str
    is_rest_synthetic: bool | None


class EpochDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        stats: NormalizationStats | None = None,
        validate: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.stats = stats
        if validate and not self.df.empty:
            validate_epoch_dataframe(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        epoch = normalize_epoch(features_to_array(row["features"]), self.stats)
        x = torch.from_numpy(epoch).unsqueeze(0)
        y = torch.tensor(int(row["label_code"]), dtype=torch.long)
        meta = EpochMetadata(
            epoch_id=str(row["epoch_id"]),
            dataset=str(row["dataset"]),
            subject_id=str(row["subject_id"]),
            session_id=None if pd.isna(row.get("session_id")) else str(row.get("session_id")),
            run_id=None if pd.isna(row.get("run_id")) else str(row.get("run_id")),
            filter_version=str(row["filter_version"]),
            preprocessing_version=str(row["preprocessing_version"]),
            is_rest_synthetic=None
            if pd.isna(row.get("is_rest_synthetic"))
            else bool(row.get("is_rest_synthetic")),
        )
        return x, y, meta


def collate_epoch_batch(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)
