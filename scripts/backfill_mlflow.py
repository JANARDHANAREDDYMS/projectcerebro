"""Backfill existing ProjectCerebro experiment results into MLflow via REST.

This script intentionally avoids ``import mlflow`` because the local SDK import
can hang in some development environments. It talks directly to the MLflow REST
API exposed by the Docker service.

Usage:
    cerebro_env/bin/python scripts/backfill_mlflow.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

MLFLOW_URI = "http://localhost:5001"
CHECKPOINTS = Path("artifacts/checkpoints")
EXPERIMENT = "projectcerebro_eeg_bci"


def _request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """Call the MLflow REST API and return JSON."""
    url = f"{MLFLOW_URI}{path}"
    response = requests.request(method, url, timeout=10, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{method} {path} failed: {response.status_code} {response.text}")
    if not response.text:
        return {}
    return response.json()


def _get_or_create_experiment(name: str) -> str:
    """Return an existing MLflow experiment id or create the experiment."""
    response = requests.get(
        f"{MLFLOW_URI}/api/2.0/mlflow/experiments/get-by-name",
        params={"experiment_name": name},
        timeout=10,
    )
    if response.status_code == 200:
        return str(response.json()["experiment"]["experiment_id"])
    if response.status_code != 404:
        raise RuntimeError(f"get-by-name failed: {response.status_code} {response.text}")

    created = _request("POST", "/api/2.0/mlflow/experiments/create", json={"name": name})
    return str(created["experiment_id"])


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _params_from_name(name: str) -> list[dict[str, str]]:
    """Extract common run parameters from a checkpoint directory name."""
    params: dict[str, Any] = {"run_name": name}

    lr_patterns = {
        "lr5e5": 5e-5,
        "lr5e4": 5e-4,
        "lr1e5": 1e-5,
        "lr1e4": 1e-4,
        "lr2e4": 2e-4,
        "lr1e3": 1e-3,
    }
    for token, value in lr_patterns.items():
        if token in name:
            params["lr"] = value

    for batch_size in (16, 32, 64):
        if f"bs{batch_size}" in name:
            params["batch_size"] = batch_size

    for seed in (0, 42, 123):
        if f"seed{seed}" in name:
            params["seed"] = seed

    dropout_patterns = {
        "dropout04": 0.4,
        "dropout05": 0.5,
        "dropout06": 0.6,
        "dropout07": 0.7,
    }
    for token, value in dropout_patterns.items():
        if token in name:
            params["dropout"] = value

    if "noea" in name:
        params["use_ea"] = False
    elif "ea" in name:
        params["use_ea"] = True

    return [{"key": key, "value": str(value)} for key, value in params.items()]


def _metrics_from_results(results: dict[str, Any], timestamp_ms: int) -> list[dict[str, Any]]:
    """Extract MLflow REST metric payloads from a result JSON payload."""
    overall = results.get("test_overall", results)
    summary = results.get("summary", {})
    metrics: dict[str, float] = {}

    if "accuracy" in overall:
        metrics["test_accuracy"] = float(overall["accuracy"])
    if "macro_f1" in overall:
        metrics["test_macro_f1"] = float(overall["macro_f1"])
    if "balanced_accuracy" in overall:
        metrics["test_bal_acc"] = float(overall["balanced_accuracy"])
    if "loss" in overall:
        metrics["test_loss"] = float(overall["loss"])
    if "best_val_macro_f1" in summary:
        metrics["val_macro_f1"] = float(summary["best_val_macro_f1"])
    elif "test_macro_f1" in metrics:
        metrics["val_macro_f1"] = metrics["test_macro_f1"]
    if "best_epoch" in summary:
        metrics["best_epoch"] = float(summary["best_epoch"])

    return [
        {"key": key, "value": value, "timestamp": timestamp_ms, "step": 0}
        for key, value in metrics.items()
    ]


def main() -> None:
    """Backfill all checkpoint result files into MLflow."""
    print(f"Connecting to MLflow at {MLFLOW_URI}...")
    experiment_id = _get_or_create_experiment(EXPERIMENT)
    print(f"Experiment: {EXPERIMENT} (id={experiment_id})")

    logged = 0
    for ckpt_dir in sorted(CHECKPOINTS.iterdir()):
        if not ckpt_dir.is_dir():
            continue

        test_json = ckpt_dir / "test_overall.json"
        if not test_json.exists():
            continue

        name = ckpt_dir.name
        timestamp_ms = int(time.time() * 1000)
        results = _load_json(test_json)
        metrics = _metrics_from_results(results, timestamp_ms)
        if not metrics:
            print(f"  Skipping {name} - no metrics found")
            continue

        run = _request(
            "POST",
            "/api/2.0/mlflow/runs/create",
            json={"experiment_id": experiment_id, "run_name": name, "start_time": timestamp_ms},
        )
        run_id = run["run"]["info"]["run_id"]
        _request(
            "POST",
            "/api/2.0/mlflow/runs/log-batch",
            json={
                "run_id": run_id,
                "params": _params_from_name(name),
                "metrics": metrics,
                "tags": [{"key": "source", "value": "backfill"}],
            },
        )
        _request(
            "POST",
            "/api/2.0/mlflow/runs/update",
            json={"run_id": run_id, "status": "FINISHED", "end_time": timestamp_ms},
        )
        logged += 1
        metric_preview = {metric["key"]: metric["value"] for metric in metrics}
        print(f"  Logged: {name} -> {metric_preview}")

    print(f"Backfill complete. Logged {logged} runs.")
    print(f"View: {MLFLOW_URI}")


if __name__ == "__main__":
    main()

