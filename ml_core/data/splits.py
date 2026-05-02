from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SubjectSplit:
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def _split_counts(n_subjects: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n_subjects < 3:
        raise ValueError("At least 3 subjects are required for train/val/test splits")
    n_train = max(1, int(n_subjects * train_ratio))
    n_val = max(1, math.ceil(n_subjects * val_ratio))
    if n_train + n_val >= n_subjects:
        n_train = max(1, n_subjects - 2)
        n_val = 1
    n_test = n_subjects - n_train - n_val
    return n_train, n_val, n_test


def make_subject_split(
    subjects: list[str],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SubjectSplit:
    unique_subjects = sorted(set(subjects))
    rng = random.Random(seed)
    rng.shuffle(unique_subjects)
    n_train, n_val, _ = _split_counts(len(unique_subjects), train_ratio, val_ratio)
    train_subjects = sorted(unique_subjects[:n_train])
    val_subjects = sorted(unique_subjects[n_train : n_train + n_val])
    test_subjects = sorted(unique_subjects[n_train + n_val :])
    split = SubjectSplit(train_subjects, val_subjects, test_subjects, seed)
    assert_no_overlap(split)
    return split


def make_pretrain_split(
    subjects: list[str],
    seed: int = 42,
    train_ratio: float = 0.85,
) -> SubjectSplit:
    split = make_subject_split(subjects, seed=seed, train_ratio=train_ratio, val_ratio=0.15)
    return SubjectSplit(split.train_subjects, sorted(split.val_subjects + split.test_subjects), [], seed)


def assert_no_overlap(split: SubjectSplit) -> None:
    train = set(split.train_subjects)
    val = set(split.val_subjects)
    test = set(split.test_subjects)
    if train & val or train & test or val & test:
        raise ValueError(f"Subject split overlap detected: {split}")


def split_dataframe(df: pd.DataFrame, split: SubjectSplit) -> dict[str, pd.DataFrame]:
    groups = {
        "train": set(split.train_subjects),
        "val": set(split.val_subjects),
        "test": set(split.test_subjects),
    }
    return {
        name: df[df["subject_id"].isin(subjects)].reset_index(drop=True)
        for name, subjects in groups.items()
    }


def save_split_manifest(split: SubjectSplit, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")


def load_split_manifest(path: str | Path) -> SubjectSplit:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SubjectSplit(**payload)
