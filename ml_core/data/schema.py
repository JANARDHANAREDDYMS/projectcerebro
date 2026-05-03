"""Schema constants and validation for Stage 2 EEG epoch tables."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


EXPECTED_FEATURE_LEN = 2560
N_CHANNELS = 5
N_SAMPLES = 512
LABEL_MAP = {0: "left", 1: "right", 2: "rest"}
COMMON_CHANNELS = ["FZ", "C3", "CZ", "C4", "PZ"]

REQUIRED_COLUMNS = [
    "epoch_id",
    "dataset",
    "subject_id",
    "label_code",
    "features",
    "filter_version",
    "preprocessing_version",
]

STAGE2_COLUMNS = [
    "epoch_id",
    "dataset",
    "subject_id",
    "session_id",
    "run_id",
    "source_file",
    "label_code",
    "label_name",
    "features",
    "n_channels",
    "n_samples",
    "channel_names",
    "sampling_rate_hz",
    "epoch_start_sec",
    "epoch_end_sec",
    "filter_version",
    "preprocessing_version",
    "ingested_at",
    "is_rest_synthetic",
]


class SchemaError(ValueError):
    """Raised when an epoch DataFrame does not satisfy the ML data contract."""


def _feature_lengths(features: Iterable[object]) -> list[int]:
    """Return feature lengths, treating non-sized values as invalid length -1."""
    lengths: list[int] = []
    for value in features:
        try:
            lengths.append(len(value))  # type: ignore[arg-type]
        except TypeError:
            lengths.append(-1)
    return lengths


def validate_schema(df: pd.DataFrame) -> None:
    """Validate required epoch columns, label codes, and feature vector length.

    The real Stage 2 output has the full `STAGE2_COLUMNS` schema. Unit tests and
    some synthetic fixtures use a minimal subset, so this validator requires the
    columns needed for ML training and validates optional Stage 2 columns when
    they are present.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    bad_labels = sorted(set(df["label_code"].dropna().astype(int)) - set(LABEL_MAP))
    if bad_labels:
        raise SchemaError(f"Unexpected label_code values: {bad_labels}")

    lengths = _feature_lengths(df["features"])
    bad_count = sum(length != EXPECTED_FEATURE_LEN for length in lengths)
    if bad_count:
        raise SchemaError(
            f"{bad_count} rows have wrong feature length; expected {EXPECTED_FEATURE_LEN}"
        )

    if "n_channels" in df.columns:
        bad_channels = df["n_channels"].dropna().astype(int).ne(N_CHANNELS).sum()
        if bad_channels:
            raise SchemaError(f"{bad_channels} rows have n_channels != {N_CHANNELS}")

    if "n_samples" in df.columns:
        bad_samples = df["n_samples"].dropna().astype(int).ne(N_SAMPLES).sum()
        if bad_samples:
            raise SchemaError(f"{bad_samples} rows have n_samples != {N_SAMPLES}")


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with invalid feature-length rows removed."""
    if "features" not in df.columns:
        raise SchemaError("Missing required columns: ['features']")

    lengths = _feature_lengths(df["features"])
    mask = [length == EXPECTED_FEATURE_LEN for length in lengths]
    return df.loc[mask].reset_index(drop=True).copy()
