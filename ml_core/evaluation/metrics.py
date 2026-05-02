from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ml_core.data.schema import LABEL_MAP


def compute_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float | list[list[int]] | dict]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(LABEL_MAP)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics: dict[str, float | list[list[int]] | dict] = {
        f"{prefix}acc": float(accuracy_score(y_true, y_pred)),
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        f"{prefix}balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }
    for idx, label in enumerate(labels):
        name = LABEL_MAP[label]
        metrics[f"{prefix}{name}_precision"] = float(precision[idx])
        metrics[f"{prefix}{name}_recall"] = float(recall[idx])
    return metrics


def classification_report_dict(y_true, y_pred) -> dict:
    target_names = [LABEL_MAP[label] for label in sorted(LABEL_MAP)]
    return classification_report(
        y_true,
        y_pred,
        labels=sorted(LABEL_MAP),
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )


def save_metrics_json(metrics: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
