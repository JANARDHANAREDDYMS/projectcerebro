"""Classification metrics: accuracy, macro F1, balanced accuracy, per-class P/R, confmat."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return a flat metrics dict suitable for JSON serialization."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_precision": {class_names[i]: float(p[i]) for i in range(n_classes)},
        "per_class_recall": {class_names[i]: float(r[i]) for i in range(n_classes)},
        "per_class_f1": {class_names[i]: float(f[i]) for i in range(n_classes)},
        "confusion_matrix": cm,
        "class_names": class_names,
        "n_samples": int(len(y_true)),
    }


def save_classification_report(metrics: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
