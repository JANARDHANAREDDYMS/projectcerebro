"""Shared fixtures for ml_core tests.

Many tests run without a real Delta table by synthesizing a tiny in-memory
DataFrame with the same schema. The synthetic fixture decouples loader unit
tests from the Drive download requirement.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_core.data.schema import EXPECTED_FEATURE_LEN, N_CHANNELS, N_SAMPLES


def _make_features(n: int, *, seed: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal(EXPECTED_FEATURE_LEN).astype(np.float32).tolist()
        for _ in range(n)
    ]


@pytest.fixture
def synthetic_epochs_df() -> pd.DataFrame:
    """120 epochs across 12 subjects, 3 classes, 2 datasets."""
    n_subj = 12
    per_subj = 10
    rows = []
    seed = 0
    for s in range(n_subj):
        ds = "physionet" if s < 8 else "bci_iv_2a"
        for i in range(per_subj):
            rows.append(
                {
                    "epoch_id": f"sub{s:03d}-ep{i:04d}",
                    "dataset": ds,
                    "subject_id": f"S{s:03d}",
                    "session_id": "T",
                    "run_id": f"R{i % 3}",
                    "label_code": i % 3,
                    "label_name": ["left", "right", "rest"][i % 3],
                    "filter_version": "bp_8_30_v1",
                    "preprocessing_version": "v1",
                    "n_channels": N_CHANNELS,
                    "n_samples": N_SAMPLES,
                    "sampling_rate_hz": 128.0,
                    "is_rest_synthetic": False,
                }
            )
            seed += 1
    df = pd.DataFrame(rows)
    df["features"] = _make_features(len(df), seed=42)
    return df


@pytest.fixture
def real_delta_path() -> Path | None:
    """Return a real Delta table path if it exists locally; else None."""
    p = Path("delta_lake/epochs_mi_v1_ch5_sr128_bp8_30")
    return p if p.exists() else None


@pytest.fixture
def run_db_tests() -> bool:
    """Gate Postgres-backed tests behind RUN_DB_TESTS=1."""
    return os.environ.get("RUN_DB_TESTS") == "1"
