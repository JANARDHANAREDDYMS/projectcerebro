"""Subject-level split tests."""
from __future__ import annotations

import pytest

from ml_core.data.splits import SplitManifest, subject_split


def test_split_is_deterministic(synthetic_epochs_df):
    a = subject_split(synthetic_epochs_df, seed=42)
    b = subject_split(synthetic_epochs_df, seed=42)
    assert a[3].train_subjects == b[3].train_subjects
    assert a[3].val_subjects == b[3].val_subjects
    assert a[3].test_subjects == b[3].test_subjects


def test_no_subject_overlap(synthetic_epochs_df):
    _train, _val, _test, manifest = subject_split(synthetic_epochs_df, seed=42)
    manifest.assert_disjoint()


def test_split_ratios_close(synthetic_epochs_df):
    _train, _val, _test, manifest = subject_split(synthetic_epochs_df, seed=7, ratios=(0.7, 0.15, 0.15))
    n = len(manifest.train_subjects) + len(manifest.val_subjects) + len(manifest.test_subjects)
    assert n == 12  # synthetic fixture
    assert len(manifest.train_subjects) >= len(manifest.val_subjects)
    assert len(manifest.test_subjects) >= 1


def test_split_too_few_subjects():
    import pandas as pd
    df = pd.DataFrame({"subject_id": ["a", "b"]})
    with pytest.raises(ValueError, match=">= 3"):
        subject_split(df)


def test_manifest_round_trip(tmp_path, synthetic_epochs_df):
    _train, _val, _test, manifest = subject_split(synthetic_epochs_df, seed=42)
    p = tmp_path / "manifest.json"
    manifest.to_json(p)
    loaded = SplitManifest.from_json(p)
    assert loaded.train_subjects == manifest.train_subjects


def test_bci_like_9_subjects():
    """BCI IV-2a edge case: only 9 subjects -> still gets >=1 in val and test."""
    import numpy as np
    import pandas as pd

    rows = [{"subject_id": f"S{i}", "features": [0.0] * 2560, "epoch_id": str(i),
             "dataset": "bci_iv_2a", "run_id": "R0", "label_code": i % 3,
             "filter_version": "bp_8_30_v1", "preprocessing_version": "v1"} for i in range(9)]
    df = pd.DataFrame(rows)
    _train, _val, _test, m = subject_split(df, seed=42)
    assert len(m.val_subjects) >= 1
    assert len(m.test_subjects) >= 1
    assert len(m.train_subjects) >= 1
    m.assert_disjoint()
    _ = np  # silence
