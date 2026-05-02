from __future__ import annotations

import pandas as pd

from ml_core.evaluation.metrics import compute_metrics


def per_subject_metrics(subject_ids, y_true, y_pred) -> dict[str, dict]:
    df = pd.DataFrame({"subject_id": subject_ids, "y_true": y_true, "y_pred": y_pred})
    results: dict[str, dict] = {}
    for subject_id, group in df.groupby("subject_id"):
        results[str(subject_id)] = compute_metrics(group["y_true"], group["y_pred"])
    return results
