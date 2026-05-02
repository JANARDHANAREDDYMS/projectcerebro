from __future__ import annotations

from sklearn.metrics import f1_score

from ml_core.evaluation.metrics import compute_metrics


def test_macro_f1_matches_sklearn_reference():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["macro_f1"] == f1_score(y_true, y_pred, average="macro")
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 1, 1], [0, 1, 1]]
