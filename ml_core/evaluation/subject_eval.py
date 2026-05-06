"""Per-subject evaluation aggregator."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .metrics import compute_classification_metrics


def per_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: Sequence[str],
    *,
    n_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute classification metrics independently for each subject."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sids = np.asarray(subject_ids)
    if not (len(y_true) == len(y_pred) == len(sids)):
        raise ValueError("Length mismatch between predictions and subjects.")

    out: dict[str, dict[str, Any]] = {}
    for sid in sorted(set(sids.tolist())):
        mask = sids == sid
        out[sid] = compute_classification_metrics(
            y_true[mask], y_pred[mask], n_classes=n_classes, class_names=class_names
        )
    return out
