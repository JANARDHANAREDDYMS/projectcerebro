from __future__ import annotations

from pathlib import Path
from typing import Any


class MLflowLogger:
    def __init__(self, tracking_uri: str = "file:./artifacts/mlruns", experiment_name: str = "projectcerebro"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.mlflow = None
        self.active = False

    def __enter__(self):
        try:
            import mlflow

            self.mlflow = mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run()
            self.active = True
        except Exception as exc:  # MLflow must never block local smoke runs.
            print(f"MLflow disabled: {exc}")
            self.active = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.active and self.mlflow:
            self.mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.active or not self.mlflow:
            return
        flat = _flatten(params)
        for key, value in flat.items():
            self.mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.active or not self.mlflow:
            return
        numeric = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        self.mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if self.active and self.mlflow and Path(path).exists():
            self.mlflow.log_artifact(str(path))


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, full_key))
        else:
            result[full_key] = value
    return result
