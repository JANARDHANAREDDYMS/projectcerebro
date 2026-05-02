from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_core.data.schema import CHANNEL_NAMES, FEATURE_LENGTH


@pytest.fixture
def synthetic_epoch_df() -> pd.DataFrame:
    rows = []
    subjects = [f"S{i:03d}" for i in range(1, 13)]
    rng = np.random.default_rng(42)
    for subject in subjects:
        for idx in range(3):
            label = idx % 3
            rows.append(
                {
                    "epoch_id": f"physionet|{subject}|R04|{label}|{idx}",
                    "dataset": "physionet",
                    "subject_id": subject,
                    "session_id": None,
                    "run_id": "R04",
                    "source_file": "synthetic",
                    "label_code": label,
                    "label_name": ["left", "right", "rest"][label],
                    "features": rng.normal(size=FEATURE_LENGTH).astype(np.float32).tolist(),
                    "n_channels": 5,
                    "n_samples": 512,
                    "channel_names": CHANNEL_NAMES,
                    "sampling_rate_hz": 128.0,
                    "epoch_start_sec": 0.0,
                    "epoch_end_sec": 4.0,
                    "filter_version": "bp_8_30_v1",
                    "preprocessing_version": "v1.1.0",
                    "ingested_at": "2026-05-02T00:00:00Z",
                    "is_rest_synthetic": False,
                }
            )
    return pd.DataFrame(rows)
