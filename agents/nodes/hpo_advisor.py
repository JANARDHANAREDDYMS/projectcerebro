"""HPO Advisor Agent that reads MLflow and records next-run suggestions."""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

from agents.state import BrainState
from agents.tools.mlflow_client import best_runs_by_metric, mlflow_available
from agents.tools.mongodb_client import insert_one


def hpo_advisor_node(state: BrainState) -> BrainState:
    """Analyze MLflow history and suggest a new hyperparameter configuration."""
    print("[HPO Advisor] Checking MLflow availability...", flush=True)
    if not mlflow_available():
        print("[HPO Advisor] MLflow not reachable. Skipping.")
        return state

    print("[HPO Advisor] Querying MLflow runs...", flush=True)
    runs = best_runs_by_metric("val_macro_f1", max_results=50)
    if not runs:
        print("[HPO Advisor] No MLflow runs available.")
        return state

    def run_metric(run: dict, key: str) -> float:
        for metric in run.get("data", {}).get("metrics", []):
            if metric.get("key") == key:
                return float(metric.get("value", 0.0))
        return 0.0

    def run_params(run: dict) -> dict[str, str]:
        return {
            param.get("key"): param.get("value", "")
            for param in run.get("data", {}).get("params", [])
        }

    def run_metrics(run: dict) -> dict[str, float]:
        return {
            metric.get("key"): float(metric.get("value", 0.0))
            for metric in run.get("data", {}).get("metrics", [])
        }

    best_run = max(runs, key=lambda run: run_metric(run, "val_macro_f1"))
    best_params = run_params(best_run)
    best_metrics = run_metrics(best_run)

    current_lr = float(best_params.get("lr", 1e-4))
    current_bs = int(best_params.get("batch_size", 32))
    current_dropout = float(best_params.get("dropout", 0.25))
    tried_lrs = {
        float(run_params(run).get("lr", 0.0))
        for run in runs
        if run_params(run).get("lr")
    }
    lr_candidates = [current_lr * 0.5, current_lr, current_lr * 2.0]
    next_lr = next((lr for lr in lr_candidates if lr not in tried_lrs), current_lr * 0.5)

    suggested_config = {
        "lr": next_lr,
        "batch_size": random.choice([16, current_bs, 64]),
        "dropout": random.choice([0.25, current_dropout, 0.5]),
        "weight_decay": float(best_params.get("weight_decay", 1e-4)),
        "epochs": 100,
        "patience": 25,
    }
    best_f1 = float(best_metrics.get("val_macro_f1", 0.0))
    rationale = (
        f"Best run achieved val_macro_f1={best_f1:.4f} with lr={current_lr:.2e}. "
        f"Suggesting lr={next_lr:.2e} as next trial."
    )

    insert_one(
        "hpo_recommendations",
        {
            "recommendation_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc),
            "mlflow_run_id": best_run["info"]["run_id"],
            "current_metrics": best_metrics,
            "best_params": best_params,
            "suggested_config": suggested_config,
            "rationale": rationale,
            "expected_improvement": 0.02,
            "status": "pending",
        },
    )
    print(f"[HPO Advisor] {rationale}")
    print(f"[HPO Advisor] Suggested: {suggested_config}")
    return state
