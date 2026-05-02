from __future__ import annotations

from ml_core.data.splits import assert_no_overlap, make_pretrain_split, make_subject_split, split_dataframe


def test_subject_split_is_deterministic_and_disjoint(synthetic_epoch_df):
    subjects = synthetic_epoch_df["subject_id"].tolist()
    a = make_subject_split(subjects, seed=42)
    b = make_subject_split(subjects, seed=42)
    assert a == b
    assert_no_overlap(a)
    parts = split_dataframe(synthetic_epoch_df, a)
    assert set(parts) == {"train", "val", "test"}
    assert len(parts["train"]) > len(parts["val"]) >= 1
    assert len(parts["test"]) >= 1


def test_bci_nine_subject_floor():
    split = make_subject_split([f"A{i:02d}" for i in range(1, 10)], seed=42)
    assert len(split.train_subjects) == 6
    assert len(split.val_subjects) == 2
    assert len(split.test_subjects) == 1


def test_pretrain_split_has_no_test_subjects(synthetic_epoch_df):
    split = make_pretrain_split(synthetic_epoch_df["subject_id"].tolist(), seed=42)
    assert split.test_subjects == []
    assert len(split.train_subjects) > 0
    assert len(split.val_subjects) > 0
