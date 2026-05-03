"""Metric correctness vs sklearn reference."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from ml_core.evaluation.metrics import compute_classification_metrics
from ml_core.evaluation.subject_eval import per_subject_metrics


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=200)
    y_pred = rng.integers(0, 3, size=200)
    metrics = compute_classification_metrics(y_true, y_pred, n_classes=3)
    expected = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    assert abs(metrics["macro_f1"] - expected) < 1e-9


def test_confusion_matrix_shape():
    y_true = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    metrics = compute_classification_metrics(y_true, y_pred, n_classes=3)
    cm = np.asarray(metrics["confusion_matrix"])
    assert cm.shape == (3, 3)
    assert int(cm.sum()) == len(y_true)


def test_per_subject_split_consistent():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    sids = ["A", "A", "A", "B", "B", "B"]
    out = per_subject_metrics(y_true, y_pred, sids, n_classes=3)
    assert set(out.keys()) == {"A", "B"}
    assert out["A"]["n_samples"] == 3
    assert out["B"]["n_samples"] == 3
