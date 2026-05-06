"""LOSO split + aggregator tests."""
from __future__ import annotations

from ml_core.data.splits import loso_iter
from ml_core.evaluation.subject_eval import aggregate_loso


def test_loso_one_fold_per_subject(synthetic_epochs_df):
    folds = list(loso_iter(synthetic_epochs_df, seed=42))
    n_subjects = synthetic_epochs_df["subject_id"].nunique()
    assert len(folds) == n_subjects
    held_seen: set[str] = set()
    for _train, _val, test, manifest in folds:
        held = manifest.test_subjects[0]
        held_seen.add(held)
        # test_df contains exactly the held-out subject
        assert set(test["subject_id"].astype(str).unique().tolist()) == {held}
        # No subject overlap across the three sets
        manifest.assert_disjoint()
    assert held_seen == set(synthetic_epochs_df["subject_id"].astype(str).unique().tolist())


def test_loso_holdout_dataset(synthetic_epochs_df):
    bci_subj = sorted(
        synthetic_epochs_df.loc[synthetic_epochs_df["dataset"] == "bci_iv_2a", "subject_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    folds = list(loso_iter(synthetic_epochs_df, seed=0, holdout_dataset="bci_iv_2a"))
    assert [m.test_subjects[0] for *_x, m in folds] == bci_subj
    # PhysioNet subjects must remain available as training data on every fold.
    for train, _val, _test, _m in folds:
        assert (train["dataset"] == "physionet").any()


def test_loso_deterministic(synthetic_epochs_df):
    a = list(loso_iter(synthetic_epochs_df, seed=7))
    b = list(loso_iter(synthetic_epochs_df, seed=7))
    for fa, fb in zip(a, b):
        assert fa[3].val_subjects == fb[3].val_subjects
        assert fa[3].test_subjects == fb[3].test_subjects


def test_aggregate_loso_means_and_std():
    fake = {
        "S1": {"accuracy": 0.8, "macro_f1": 0.7, "balanced_accuracy": 0.75},
        "S2": {"accuracy": 0.6, "macro_f1": 0.5, "balanced_accuracy": 0.55},
    }
    out = aggregate_loso(fake)
    assert out["n_folds"] == 2
    assert abs(out["mean"]["accuracy"] - 0.7) < 1e-9
    assert abs(out["mean"]["macro_f1"] - 0.6) < 1e-9
    assert out["std"]["accuracy"] > 0.0
