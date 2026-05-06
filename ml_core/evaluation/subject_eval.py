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


def aggregate_loso(per_fold_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-fold (per-held-out-subject) classification metrics.

    Parameters
    ----------
    per_fold_metrics: ``{held_out_subject: classification_metrics_dict}``.

    Returns
    -------
    A dict with mean and std for the scalar metrics across folds, plus the
    raw per-subject dict.
    """
    if not per_fold_metrics:
        return {"mean": {}, "std": {}, "n_folds": 0, "per_subject": {}}

    keys = ["accuracy", "macro_f1", "balanced_accuracy"]
    arrays: dict[str, list[float]] = {k: [] for k in keys}
    for sid, metrics in per_fold_metrics.items():
        for k in keys:
            v = metrics.get(k)
            if v is not None:
                arrays[k].append(float(v))

    means = {k: float(np.mean(v)) if v else float("nan") for k, v in arrays.items()}
    stds = {k: float(np.std(v, ddof=0)) if v else float("nan") for k, v in arrays.items()}
    return {
        "mean": means,
        "std": stds,
        "n_folds": len(per_fold_metrics),
        "per_subject": per_fold_metrics,
    }
