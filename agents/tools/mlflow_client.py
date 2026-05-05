"""MLflow helper functions for the HPO advisor."""
from __future__ import annotations

import os
import socket
from typing import Any
from urllib.parse import urlparse

MLFLOW_URI = os.getenv("CEREBRO_MLFLOW_URI", "http://localhost:5000")


def best_runs_by_metric(metric_name: str = "val_macro_f1", *, max_results: int = 50) -> list[Any]:
    """Return MLflow runs sorted by a metric across all experiments."""
    try:
        parsed = urlparse(MLFLOW_URI)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=1.0):
            pass

        import mlflow

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        runs = []
        for experiment in client.search_experiments():
            runs.extend(
                client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=[f"metrics.{metric_name} DESC"],
                    max_results=max_results,
                )
            )
        return runs
    except Exception as exc:
        print(f"[MLflow] query failed: {exc}")
        return []
