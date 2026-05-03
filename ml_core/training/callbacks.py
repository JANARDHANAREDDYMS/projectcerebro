"""Training callbacks: MLflow logger + a no-op default."""
from __future__ import annotations

from typing import Any, Mapping


class NoOpCallback:
    """Used when MLflow is unavailable or disabled."""

    def start(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def end(self) -> None: ...


class MLflowCallback:
    """Thin wrapper around MLflow's run lifecycle for the trainer."""

    def __init__(
        self,
        experiment_name: str,
        *,
        tracking_uri: str | None = None,
        run_name: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        import mlflow  # local import keeps mlflow optional at module load.

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run_name = run_name
        self._tags = dict(tags or {})
        self._active = None

    def start(self, params: Mapping[str, Any]) -> None:
        self._active = self._mlflow.start_run(run_name=self._run_name)
        if self._tags:
            self._mlflow.set_tags(self._tags)
        # Cast non-primitive values to str so MLflow accepts them.
        flat = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in params.items()}
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)

    def end(self) -> None:
        if self._active is not None:
            self._mlflow.end_run()
            self._active = None
