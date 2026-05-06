"""Delta Lake epoch schema constants and validation.

Mirrors the schema written by `scripts/stage2_spark_preprocess.py` (lines 863-883).
Update both if the upstream schema changes.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

# Stage-2 fixed acquisition parameters
N_CHANNELS: int = 5
N_SAMPLES: int = 512
SAMPLING_RATE_HZ: float = 128.0
EXPECTED_FEATURE_LEN: int = N_CHANNELS * N_SAMPLES  # 2560
CHANNEL_NAMES: tuple[str, ...] = ("FZ", "C3", "CZ", "C4", "PZ")

LABEL_MAP: dict[str, int] = {"left": 0, "right": 1, "rest": 2}
INV_LABEL_MAP: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

# Columns the loader and trainer rely on. Any of these missing -> hard fail.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "epoch_id",
    "dataset",
    "subject_id",
    "run_id",
    "label_code",
    "features",
    "filter_version",
    "preprocessing_version",
)

# Optional columns surfaced for filtering/metrics if present.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "session_id",
    "label_name",
    "channel_names",
    "n_channels",
    "n_samples",
    "sampling_rate_hz",
    "is_rest_synthetic",
    "epoch_start_sec",
    "epoch_end_sec",
    "source_file",
    "ingested_at",
)


class SchemaError(ValueError):
    """Raised when a Delta DataFrame does not match the expected schema."""


def validate_schema(df: pd.DataFrame, *, sample_check: int = 32) -> None:
    """Hard-fail if `df` is missing required columns or features have wrong shape.

    Parameters
    ----------
    df: DataFrame loaded from Delta.
    sample_check: how many feature rows to sample-check for length.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    if df.empty:
        raise SchemaError("DataFrame has zero rows.")

    # Spot-check feature length on a sample so we fail fast on shape mismatches.
    sample = df["features"].head(sample_check)
    for idx, feat in sample.items():
        if feat is None:
            raise SchemaError(f"Row {idx}: features is None.")
        if len(feat) != EXPECTED_FEATURE_LEN:
            raise SchemaError(
                f"Row {idx}: features len {len(feat)} != {EXPECTED_FEATURE_LEN}"
            )

    bad_labels = set(df["label_code"].unique()) - set(INV_LABEL_MAP)
    if bad_labels:
        raise SchemaError(f"Unexpected label_code values: {bad_labels}")


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with feature length != EXPECTED_FEATURE_LEN. Pure function."""
    mask = df["features"].apply(lambda f: f is not None and len(f) == EXPECTED_FEATURE_LEN)
    return df.loc[mask].reset_index(drop=True)


def assert_unique_epoch_ids(df: pd.DataFrame) -> None:
    if df["epoch_id"].duplicated().any():
        dupes: Iterable = df.loc[df["epoch_id"].duplicated(), "epoch_id"].unique()
        raise SchemaError(f"Duplicate epoch_id values: {list(dupes)[:5]}...")
