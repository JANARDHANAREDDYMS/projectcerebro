from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


N_CHANNELS = 5
N_SAMPLES = 512
FEATURE_LENGTH = N_CHANNELS * N_SAMPLES
CHANNEL_NAMES = ["FZ", "C3", "CZ", "C4", "PZ"]
SAMPLING_RATE_HZ = 128.0

LABEL_MAP = {0: "left", 1: "right", 2: "rest"}
LABEL_NAME_TO_CODE = {name: code for code, name in LABEL_MAP.items()}

PATH_FILTER_TO_COLUMN = {
    "bp8_30": "bp_8_30_v1",
    "bp4_38": "bp_4_38_v1",
}

PRIMARY_DELTA_PATH = "delta_lake/epochs_mi_v1_ch5_sr128_bp8_30"
ABLATION_DELTA_PATH = "delta_lake/epochs_mi_v1_ch5_sr128_bp4_38"

REQUIRED_COLUMNS = [
    "epoch_id",
    "dataset",
    "subject_id",
    "session_id",
    "run_id",
    "label_code",
    "label_name",
    "features",
    "n_channels",
    "n_samples",
    "channel_names",
    "sampling_rate_hz",
    "filter_version",
    "preprocessing_version",
    "is_rest_synthetic",
]


@dataclass(frozen=True)
class SchemaReport:
    n_rows: int
    n_bad_features: int
    n_bad_channels: int
    n_bad_samples: int
    n_bad_labels: int

    @property
    def is_valid(self) -> bool:
        return (
            self.n_bad_features == 0
            and self.n_bad_channels == 0
            and self.n_bad_samples == 0
            and self.n_bad_labels == 0
        )


def resolve_filter_version(filter_key_or_version: str | None) -> str | None:
    if filter_key_or_version is None:
        return None
    return PATH_FILTER_TO_COLUMN.get(filter_key_or_version, filter_key_or_version)


def require_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Delta columns: {missing}")


def feature_length(value: object) -> int:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return -1


def validate_epoch_dataframe(df: pd.DataFrame) -> SchemaReport:
    require_columns(df)
    lengths = df["features"].map(feature_length)
    labels = set(df["label_code"].dropna().astype(int).tolist())
    report = SchemaReport(
        n_rows=len(df),
        n_bad_features=int((lengths != FEATURE_LENGTH).sum()),
        n_bad_channels=int((df["n_channels"].astype(int) != N_CHANNELS).sum()),
        n_bad_samples=int((df["n_samples"].astype(int) != N_SAMPLES).sum()),
        n_bad_labels=len(labels - set(LABEL_MAP)),
    )
    if not report.is_valid:
        raise ValueError(f"Invalid epoch dataframe: {report}")
    return report


def drop_bad_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df)
    mask = df["features"].map(feature_length) == FEATURE_LENGTH
    return df.loc[mask].reset_index(drop=True)
