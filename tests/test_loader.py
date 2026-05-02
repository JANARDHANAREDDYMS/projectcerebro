from __future__ import annotations

import pytest

from ml_core.data.dataset import EpochDataset
from ml_core.data.normalize import compute_train_stats
from ml_core.data.schema import (
    FEATURE_LENGTH,
    drop_bad_feature_rows,
    resolve_filter_version,
    validate_epoch_dataframe,
)


def test_schema_validation_accepts_stage2_contract(synthetic_epoch_df):
    report = validate_epoch_dataframe(synthetic_epoch_df)
    assert report.n_rows == len(synthetic_epoch_df)
    assert report.is_valid
    assert resolve_filter_version("bp8_30") == "bp_8_30_v1"


def test_bad_feature_rows_can_be_rejected(synthetic_epoch_df):
    bad = synthetic_epoch_df.copy()
    bad.at[0, "features"] = [0.0] * (FEATURE_LENGTH - 1)
    with pytest.raises(ValueError):
        validate_epoch_dataframe(bad)
    cleaned = drop_bad_feature_rows(bad)
    assert len(cleaned) == len(bad) - 1


def test_epoch_dataset_shape_dtype_and_metadata(synthetic_epoch_df):
    stats = compute_train_stats(synthetic_epoch_df)
    dataset = EpochDataset(synthetic_epoch_df, stats=stats)
    x, y, meta = dataset[0]
    assert tuple(x.shape) == (1, 5, 512)
    assert str(x.dtype) == "torch.float32"
    assert int(y) in {0, 1, 2}
    assert meta.subject_id.startswith("S")
