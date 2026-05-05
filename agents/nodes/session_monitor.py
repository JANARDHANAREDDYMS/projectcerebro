"""Session Monitor Agent for prediction persistence and rolling stats."""
from __future__ import annotations

from datetime import datetime, timezone

from agents.state import BrainState
from agents.tools.mongodb_client import insert_one
from agents.tools.pgvector_client import insert_embedding

BREAK_THRESHOLD_N = 50
CONFIDENCE_DROP_WINDOW = 20
CONFIDENCE_DROP_THRESH = 0.55


def session_monitor_node(state: BrainState) -> BrainState:
    """Store predictions and embeddings while updating rolling session counters."""
    session_alerts = list(state.get("session_alerts", []))
    suggest_break = False

    if state.get("label_code") is not None:
        pred_doc = {
            "epoch_id": state["epoch_id"],
            "subject_id": state["subject_id"],
            "session_id": state["session_id"],
            "timestamp": datetime.now(timezone.utc),
            "label_code": state["label_code"],
            "label_name": state["label_name"],
            "confidence": state["confidence"],
            "probabilities": state["probabilities"],
            "model_used": state.get("model_used"),
            "signal_quality": state["signal_quality"],
            "quality_score": state["quality_score"],
        }
        insert_one("predictions", pred_doc)

        n_predictions = state.get("n_predictions", 0) + 1
        n_left = state.get("n_left", 0) + int(state["label_code"] == 0)
        n_right = state.get("n_right", 0) + int(state["label_code"] == 1)
        n_rest = state.get("n_rest", 0) + int(state["label_code"] == 2)
        old_mean = state.get("mean_confidence", 0.0)
        confidence = float(state.get("confidence") or 0.0)
        mean_confidence = ((old_mean * (n_predictions - 1)) + confidence) / n_predictions

        if n_predictions % BREAK_THRESHOLD_N == 0:
            suggest_break = True
            session_alerts.append(f"Suggestion: consider a short break after {n_predictions} trials")

        if mean_confidence < CONFIDENCE_DROP_THRESH and n_predictions > CONFIDENCE_DROP_WINDOW:
            session_alerts.append(
                f"Warning: mean confidence {mean_confidence:.2f} below threshold. Consider recalibration."
            )

        if n_predictions > 30 and (n_rest / n_predictions) > 0.75:
            session_alerts.append("Warning: subject predicting rest >75% of trials. May not be engaging.")

        if state.get("embedding") is not None:
            insert_embedding(
                epoch_id=state["epoch_id"],
                subject_id=state["subject_id"],
                session_id=state["session_id"],
                label_code=state.get("label_code"),
                label_name=state.get("label_name"),
                embedding=state["embedding"] or [],
                confidence=state.get("confidence"),
            )
    else:
        n_predictions = state.get("n_predictions", 0)
        n_left = state.get("n_left", 0)
        n_right = state.get("n_right", 0)
        n_rest = state.get("n_rest", 0)
        mean_confidence = state.get("mean_confidence", 0.0)

    print(f"[SessionMonitor] session={state['session_id']} n={n_predictions} mean_conf={mean_confidence:.3f}")
    return {
        **state,
        "n_predictions": n_predictions,
        "n_left": n_left,
        "n_right": n_right,
        "n_rest": n_rest,
        "mean_confidence": mean_confidence,
        "session_alerts": session_alerts,
        "suggest_break": suggest_break,
    }

