"""Report Generator Agent for session summaries."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from agents.state import BrainState
from agents.tools.mongodb_client import find_many, insert_one

REPORTS_DIR = Path("artifacts/reports")


def report_generator_node(state: BrainState) -> BrainState:
    """Generate a text session report and persist its metadata."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    session_id = state["session_id"]
    subject_id = state["subject_id"]

    predictions = find_many("predictions", {"session_id": session_id}, {"_id": 0})
    flags = find_many("trial_quality_flags", {"session_id": session_id}, {"_id": 0})
    alerts = find_many("alerts", {"session_id": session_id}, {"_id": 0})

    n_total = len(predictions)
    n_left = sum(1 for pred in predictions if pred.get("label_code") == 0)
    n_right = sum(1 for pred in predictions if pred.get("label_code") == 1)
    n_rest = sum(1 for pred in predictions if pred.get("label_code") == 2)
    mean_conf = sum(float(pred.get("confidence", 0.0) or 0.0) for pred in predictions) / max(n_total, 1)
    n_flagged = sum(1 for flag in flags if flag.get("trial_flagged"))
    n_retraining = sum(1 for flag in flags if flag.get("retraining_candidate"))

    summary = {
        "session_id": session_id,
        "subject_id": subject_id,
        "n_predictions": n_total,
        "n_left": n_left,
        "n_right": n_right,
        "n_rest": n_rest,
        "mean_confidence": round(mean_conf, 4),
        "n_flagged_trials": n_flagged,
        "n_retraining_candidates": n_retraining,
        "n_alerts": len(alerts),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    report_id = str(uuid.uuid4())[:8]
    report_path = REPORTS_DIR / f"session_{session_id}_{report_id}.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("ProjectCerebro Session Report\n")
        handle.write("=" * 40 + "\n")
        handle.write(f"Session ID:  {session_id}\n")
        handle.write(f"Subject ID:  {subject_id}\n")
        handle.write(f"Generated:   {summary['generated_at']}\n\n")
        handle.write("Predictions\n" + "-" * 20 + "\n")
        handle.write(f"Total:  {n_total}\nLeft:   {n_left}\nRight:  {n_right}\nRest:   {n_rest}\n")
        handle.write(f"Mean confidence: {mean_conf:.2%}\n\n")
        handle.write("Data Quality\n" + "-" * 20 + "\n")
        handle.write(f"Flagged trials: {n_flagged}\nRetraining candidates: {n_retraining}\n\n")
        handle.write(f"Alerts ({len(alerts)})\n" + "-" * 20 + "\n")
        for alert in alerts:
            handle.write(f"[{alert.get('severity', 'info').upper()}] {alert.get('message', '')}\n")

    insert_one(
        "session_reports",
        {
            "report_id": report_id,
            "session_id": session_id,
            "subject_id": subject_id,
            "generated_at": datetime.now(timezone.utc),
            "report_path": str(report_path),
            "summary": summary,
        },
    )
    print(f"[ReportGenerator] Report saved: {report_path}")
    return state

