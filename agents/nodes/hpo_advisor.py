"""HPO Advisor Agent that reads MLflow and records next-run suggestions."""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

from agents.state import BrainState
from agents.tools.mlflow_client import best_runs_by_metric
from agents.tools.mongodb_client import insert_one


def hpo_advisor_node(state: BrainState) -> BrainState:
    """Analyze MLflow history and suggest a new hyperparameter configuration."""
    runs = best_runs_by_metric("val_macro_f1", max_results=50)
    if not runs:
        print("[HPO Advisor] No MLflow runs available.")
        return state

    best_run = max(runs, key=lambda run: run.data.metrics.get("val_macro_f1", 0.0))
    best_params = best_run.data.params
    best_metrics = best_run.data.metrics

    current_lr = float(best_params.get("lr", 1e-4))
    current_bs = int(best_params.get("batch_size", 32))
    current_dropout = float(best_params.get("dropout", 0.25))
    tried_lrs = {float(run.data.params.get("lr", 0.0)) for run in runs}
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
            "mlflow_run_id": best_run.info.run_id,
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

