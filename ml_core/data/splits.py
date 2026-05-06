"""Subject-level train/val/test split with deterministic seed.

Critical: never split by epoch. EEG models trivially memorize subject-specific
artifacts; subject leakage inflates test metrics.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class SplitManifest:
    """Records exactly which subject_ids ended up in each fold."""

    seed: int
    ratios: tuple[float, float, float]
    train_subjects: list[str] = field(default_factory=list)
    val_subjects: list[str] = field(default_factory=list)
    test_subjects: list[str] = field(default_factory=list)
    dataset: str | None = None
    filter_version: str | None = None

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitManifest":
        data = json.loads(Path(path).read_text())
        data["ratios"] = tuple(data["ratios"])
        return cls(**data)

    def assert_disjoint(self) -> None:
        s_train, s_val, s_test = (
            set(self.train_subjects),
            set(self.val_subjects),
            set(self.test_subjects),
        )
        if s_train & s_val or s_train & s_test or s_val & s_test:
            raise ValueError("Subjects overlap across splits.")


def subject_split(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    dataset: str | None = None,
    filter_version: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest]:
    """Split `df` into train/val/test by `subject_id`.

    For very small datasets (e.g. BCI IV-2a with 9 subjects) we guarantee at
    least one validation and one test subject.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0 (got {sum(ratios)}).")
    if len(ratios) != 3:
        raise ValueError("Need exactly three ratios (train/val/test).")

    subjects = sorted(df["subject_id"].astype(str).unique().tolist())
    n = len(subjects)
    if n < 3:
        raise ValueError(f"Need >= 3 subjects to split; got {n}.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    shuffled = [subjects[i] for i in perm]

    n_train = max(1, int(round(ratios[0] * n)))
    n_val = max(1, int(round(ratios[1] * n)))
    # Ensure at least 1 subject in each of val and test.
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)
    n_test = n - n_train - n_val
    assert n_train >= 1 and n_val >= 1 and n_test >= 1, (n_train, n_val, n_test)

    train_subj = shuffled[:n_train]
    val_subj = shuffled[n_train : n_train + n_val]
    test_subj = shuffled[n_train + n_val :]

    manifest = SplitManifest(
        seed=seed,
        ratios=tuple(ratios),
        train_subjects=sorted(train_subj),
        val_subjects=sorted(val_subj),
        test_subjects=sorted(test_subj),
        dataset=dataset,
        filter_version=filter_version,
    )
    manifest.assert_disjoint()

    sid = df["subject_id"].astype(str)
    train_df = df[sid.isin(train_subj)].reset_index(drop=True)
    val_df = df[sid.isin(val_subj)].reset_index(drop=True)
    test_df = df[sid.isin(test_subj)].reset_index(drop=True)
    return train_df, val_df, test_df, manifest


def loso_iter(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    holdout_dataset: str | None = None,
    val_fraction: float = 0.15,
    dataset_for_manifest: str | None = None,
    filter_version: str | None = None,
):
    """Yield ``(train_df, val_df, test_df, manifest)`` for every held-out subject.

    For each subject ``s`` in the eligible pool:
        * test_df = epochs from ``s`` only.
        * val_df  = epochs from a deterministically chosen subset of the remaining
                    subjects (``val_fraction`` of them, at least one).
        * train_df = remaining epochs.

    ``holdout_dataset`` restricts the LOSO sweep to that dataset (e.g.
    ``"bci_iv_2a"``). All other dataset rows still go into ``train_df``, so
    cross-dataset transfer is preserved when desired.
    """
    if "subject_id" not in df.columns:
        raise ValueError("df missing 'subject_id'.")

    sid = df["subject_id"].astype(str)
    if holdout_dataset is not None:
        if "dataset" not in df.columns:
            raise ValueError("df missing 'dataset' but holdout_dataset given.")
        eligible = sorted(df.loc[df["dataset"] == holdout_dataset, "subject_id"].astype(str).unique().tolist())
    else:
        eligible = sorted(sid.unique().tolist())
    if len(eligible) < 2:
        raise ValueError(
            f"Need >= 2 eligible subjects for LOSO; got {len(eligible)} "
            f"(holdout_dataset={holdout_dataset})."
        )

    rng = np.random.default_rng(seed)

    for held in eligible:
        # Pool of "other" subjects (anything not held out from THIS holdout pool).
        # When holdout_dataset is set, val is sampled from the same dataset only,
        # so the test/val/train partition stays interpretable.
        if holdout_dataset is not None:
            other_pool = [s for s in eligible if s != held]
        else:
            other_pool = [s for s in sid.unique().tolist() if s != held]

        n_val = max(1, int(round(val_fraction * len(other_pool))))
        n_val = min(n_val, max(1, len(other_pool) - 1))
        perm = rng.permutation(len(other_pool))
        val_subj = sorted(other_pool[i] for i in perm[:n_val])
        val_set = set(val_subj)
        held_set = {held}

        is_test = sid.isin(held_set)
        is_val = sid.isin(val_set)
        is_train = ~(is_test | is_val)

        train_df = df.loc[is_train].reset_index(drop=True)
        val_df = df.loc[is_val].reset_index(drop=True)
        test_df = df.loc[is_test].reset_index(drop=True)

        manifest = SplitManifest(
            seed=seed,
            ratios=(0.0, val_fraction, 0.0),  # encoded shape: LOSO uses single test subject
            train_subjects=sorted(set(train_df["subject_id"].astype(str).tolist())),
            val_subjects=val_subj,
            test_subjects=[held],
            dataset=dataset_for_manifest,
            filter_version=filter_version,
        )
        manifest.assert_disjoint()
        yield train_df, val_df, test_df, manifest
