"""Loader and schema tests."""
from __future__ import annotations

import numpy as np
import pytest

from ml_core.data.schema import (
    EXPECTED_FEATURE_LEN,
    SchemaError,
    filter_valid_rows,
    validate_schema,
)


def test_validate_schema_passes(synthetic_epochs_df):
    validate_schema(synthetic_epochs_df)


def test_validate_schema_missing_column(synthetic_epochs_df):
    df = synthetic_epochs_df.drop(columns=["filter_version"])
    with pytest.raises(SchemaError, match="Missing required columns"):
        validate_schema(df)


def test_filter_valid_rows_drops_bad(synthetic_epochs_df):
    df = synthetic_epochs_df.copy()
    bad = df.iloc[0]["features"][:-1]  # length 2559
    df.at[0, "features"] = bad
    cleaned = filter_valid_rows(df)
    assert len(cleaned) == len(df) - 1
    for f in cleaned["features"]:
        assert len(f) == EXPECTED_FEATURE_LEN


def test_validate_schema_rejects_unknown_label(synthetic_epochs_df):
    df = synthetic_epochs_df.copy()
    df.at[0, "label_code"] = 99
    with pytest.raises(SchemaError, match="Unexpected label_code"):
        validate_schema(df)


def test_real_delta_smoke(real_delta_path):
    if real_delta_path is None:
        pytest.skip("delta_lake/ not present locally")
    from ml_core.data import read_epochs

    df = read_epochs(real_delta_path)
    assert len(df) > 0
    sample = np.asarray(df.iloc[0]["features"], dtype=np.float32)
    assert sample.shape == (EXPECTED_FEATURE_LEN,)
