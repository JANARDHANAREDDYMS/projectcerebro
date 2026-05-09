"""REST-only MLflow helpers for the HPO advisor."""
from __future__ import annotations

import os
from typing import Any

import requests

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", os.getenv("CEREBRO_MLFLOW_URI", "http://127.0.0.1:5001"))


def _request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """Call MLflow REST API and return JSON."""
    response = requests.request(method, f"{MLFLOW_URI}{path}", timeout=10, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {path} failed: {response.status_code} {response.text}")
    if not response.text:
        return {}
    return response.json()


def mlflow_available() -> bool:
    """Return whether the MLflow tracking server is reachable."""
    try:
        _request("POST", "/api/2.0/mlflow/experiments/search", json={"max_results": 1})
        return True
    except Exception as exc:
        print(f"[MLflow] not reachable: {exc}")
        return False


def best_runs_by_metric(metric_name: str = "val_macro_f1", *, max_results: int = 50) -> list[dict[str, Any]]:
    """Return runs sorted by metric across all experiments without importing mlflow."""
    try:
        experiments = _request(
            "POST",
            "/api/2.0/mlflow/experiments/search",
            json={"max_results": 1000},
        ).get("experiments", [])
        runs: list[dict[str, Any]] = []
        for experiment in experiments:
            experiment_id = experiment["experiment_id"]
            body = {
                "experiment_ids": [experiment_id],
                "max_results": max_results,
                "order_by": [f"metrics.{metric_name} DESC"],
            }
            runs.extend(_request("POST", "/api/2.0/mlflow/runs/search", json=body).get("runs", []))

        def metric_value(run: dict[str, Any]) -> float:
            for metric in run.get("data", {}).get("metrics", []):
                if metric.get("key") == metric_name:
                    return float(metric.get("value", 0.0))
            return 0.0

        return sorted(runs, key=metric_value, reverse=True)[:max_results]
    except Exception as exc:
        print(f"[MLflow] query failed: {exc}")
        return []
