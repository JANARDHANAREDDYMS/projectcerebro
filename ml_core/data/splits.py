"""Subject-level train/validation/test splitting without epoch leakage."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitManifest:
    """Serializable record of subject IDs assigned to each split."""

    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]
    seed: int = 42
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)

    def assert_disjoint(self) -> None:
        """Raise if a subject appears in more than one split."""
        train = set(self.train_subjects)
        val = set(self.val_subjects)
        test = set(self.test_subjects)
        if train & val or train & test or val & test:
            raise ValueError("Subject split leakage: split subject sets overlap.")

    def to_json(self, path: str | Path) -> None:
        """Write this manifest to a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    def save(self, path: str | Path) -> None:
        """Alias for `to_json` for callers that prefer save/load naming."""
        self.to_json(path)

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitManifest":
        """Load a split manifest from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["ratios"] = tuple(data.get("ratios", (0.7, 0.15, 0.15)))
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        """Alias for `from_json` for callers that prefer save/load naming."""
        return cls.from_json(path)


def _counts(n_subjects: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Compute train/val/test subject counts with non-empty val/test for n>=3."""
    if n_subjects < 3:
        raise ValueError("subject_split requires >= 3 unique subjects.")

    train_ratio, val_ratio, test_ratio = ratios
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    n_test = max(1, int(round(n_subjects * test_ratio)))
    n_val = max(1, int(round(n_subjects * val_ratio)))
    if n_val + n_test >= n_subjects:
        n_val = 1
        n_test = 1
    n_train = n_subjects - n_val - n_test
    return n_train, n_val, n_test


def _split_subjects(subjects: list[str], seed: int, ratios: tuple[float, float, float]) -> tuple[list[str], list[str], list[str]]:
    """Shuffle and split one dataset's subject IDs."""
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(subjects), dtype=object)
    rng.shuffle(shuffled)
    n_train, n_val, n_test = _counts(len(shuffled), ratios)
    train = sorted(shuffled[:n_train].tolist())
    val = sorted(shuffled[n_train : n_train + n_val].tolist())
    test = sorted(shuffled[n_train + n_val : n_train + n_val + n_test].tolist())
    return train, val, test


def subject_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    ratios: tuple[float, float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitManifest]:
    """Split epochs by `subject_id`, stratifying subject assignment by dataset.

    No subject ID can appear in more than one split. If the DataFrame contains
    multiple datasets, subjects are split independently inside each dataset and
    then concatenated, which keeps small BCI cohorts represented in val/test.
    """
    if "subject_id" not in df.columns:
        raise ValueError("subject_split requires a subject_id column.")

    split_ratios = ratios or (train_ratio, val_ratio, test_ratio)
    dataset_col = "dataset" if "dataset" in df.columns else None
    groups = sorted(df[dataset_col].dropna().unique().tolist()) if dataset_col else ["__all__"]

    train_subjects: list[str] = []
    val_subjects: list[str] = []
    test_subjects: list[str] = []

    for i, dataset_name in enumerate(groups):
        part = df if dataset_col is None else df[df[dataset_col] == dataset_name]
        subjects = sorted(part["subject_id"].astype(str).unique().tolist())
        tr, va, te = _split_subjects(subjects, seed + i, split_ratios)
        train_subjects.extend(tr)
        val_subjects.extend(va)
        test_subjects.extend(te)

    manifest = SplitManifest(
        train_subjects=sorted(train_subjects),
        val_subjects=sorted(val_subjects),
        test_subjects=sorted(test_subjects),
        seed=seed,
        ratios=split_ratios,
    )
    manifest.assert_disjoint()

    train_df = df[df["subject_id"].astype(str).isin(manifest.train_subjects)].reset_index(drop=True)
    val_df = df[df["subject_id"].astype(str).isin(manifest.val_subjects)].reset_index(drop=True)
    test_df = df[df["subject_id"].astype(str).isin(manifest.test_subjects)].reset_index(drop=True)
    return train_df, val_df, test_df, manifest
