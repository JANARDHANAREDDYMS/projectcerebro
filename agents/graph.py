"""LangGraph agent graph for ProjectCerebro real-time EEG epochs."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from langgraph.graph import END, StateGraph

from agents.nodes.data_curator import data_curator_node
from agents.nodes.hpo_advisor import hpo_advisor_node
from agents.nodes.prediction import prediction_node
from agents.nodes.report_generator import report_generator_node
from agents.nodes.session_monitor import session_monitor_node
from agents.nodes.signal_quality import signal_quality_node
from agents.state import BrainState
from agents.tools.mongodb_client import insert_one
from agents.tools.pgvector_client import nearest_neighbors


def retrieval_node(state: BrainState) -> BrainState:
    """Query pgvector for nearest historical trials and produce a short explanation."""
    embedding = state.get("embedding")
    if embedding is None:
        return {**state, "similar_trials": [], "is_anomaly": False, "explanation": None}

    similar_trials = nearest_neighbors(embedding, limit=5)
    is_anomaly = bool(similar_trials and similar_trials[0]["similarity"] < 0.5)
    if similar_trials:
        top = similar_trials[0]
        explanation = (
            f"This epoch is most similar to a {top['label_name']} trial from subject "
            f"{top['subject_id']} (similarity={top['similarity']:.2f}). "
            f"Prediction: {state.get('label_name')}."
        )
    else:
        explanation = "No similar trials found in database."

    print(f"[Retrieval] epoch={state['epoch_id']} neighbors={len(similar_trials)} anomaly={is_anomaly}")
    return {
        **state,
        "similar_trials": similar_trials,
        "is_anomaly": is_anomaly,
        "explanation": explanation,
    }


def alert_node(state: BrainState) -> BrainState:
    """Collect alerts from all prior nodes and write them to MongoDB."""
    alerts: list[dict] = []

    if state["signal_quality"] == "bad":
        alerts.append(
            {
                "severity": "critical",
                "message": (
                    f"Bad signal quality (score={state['quality_score']:.2f}). "
                    f"Artifacts: {state['artifact_types']}"
                ),
                "agent": "signal_quality",
            }
        )
    elif state["signal_quality"] == "noisy":
        alerts.append(
            {
                "severity": "warning",
                "message": f"Noisy signal (score={state['quality_score']:.2f})",
                "agent": "signal_quality",
            }
        )

    if state.get("is_uncertain"):
        alerts.append(
            {
                "severity": "warning",
                "message": f"Low confidence prediction ({state.get('confidence', 0):.2f}). Consider recalibration.",
                "agent": "prediction",
            }
        )

    if state.get("is_anomaly"):
        alerts.append(
            {
                "severity": "warning",
                "message": "Anomalous epoch: far from all previously seen trials.",
                "agent": "retrieval",
            }
        )

    for message in state.get("session_alerts", []):
        alerts.append({"severity": "info", "message": message, "agent": "session_monitor"})

    if state.get("suggest_break"):
        alerts.append({"severity": "info", "message": "Suggested: take a short break.", "agent": "session_monitor"})

    severities = [alert["severity"] for alert in alerts]
    if "critical" in severities:
        final_severity = "critical"
    elif "warning" in severities:
        final_severity = "warning"
    else:
        final_severity = "info"

    for alert in alerts:
        insert_one(
            "alerts",
            {
                **alert,
                "session_id": state["session_id"],
                "epoch_id": state["epoch_id"],
                "timestamp": datetime.now(timezone.utc),
            },
        )

    print(f"[Alert] epoch={state['epoch_id']} severity={final_severity} n_alerts={len(alerts)}")
    return {**state, "alerts": alerts, "final_severity": final_severity}


def route_after_quality(state: BrainState) -> str:
    """Skip prediction when signal quality is bad."""
    return "skip" if state.get("skip_inference", False) else "predict"


def build_graph():
    """Build and compile the ProjectCerebro LangGraph."""
    graph = StateGraph(BrainState)
    graph.add_node("signal_quality", signal_quality_node)
    graph.add_node("prediction", prediction_node)
    graph.add_node("session_monitor", session_monitor_node)
    graph.add_node("data_curator", data_curator_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("alert", alert_node)

    graph.set_entry_point("signal_quality")
    graph.add_conditional_edges(
        "signal_quality",
        route_after_quality,
        {"predict": "prediction", "skip": "data_curator"},
    )
    graph.add_edge("prediction", "session_monitor")
    graph.add_edge("session_monitor", "data_curator")
    graph.add_edge("data_curator", "retrieval")
    graph.add_edge("retrieval", "alert")
    graph.add_edge("alert", END)
    return graph.compile()


def _initial_state(
    features: list[float],
    subject_id: str,
    session_id: str,
    epoch_id: str | None = None,
    true_label_code: int | None = None,
    true_label_name: str | None = None,
) -> BrainState:
    """Create a fully populated BrainState for a new epoch."""
    return {
        "features": features,
        "subject_id": subject_id,
        "session_id": session_id,
        "epoch_id": epoch_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "true_label_code": true_label_code,
        "true_label_name": true_label_name,
        "signal_quality": "good",
        "quality_score": 1.0,
        "bad_channels": [],
        "artifact_types": [],
        "skip_inference": False,
        "label_code": None,
        "label_name": None,
        "confidence": None,
        "probabilities": None,
        "embedding": None,
        "model_used": None,
        "is_uncertain": False,
        "n_predictions": 0,
        "n_left": 0,
        "n_right": 0,
        "n_rest": 0,
        "mean_confidence": 0.0,
        "session_alerts": [],
        "suggest_break": False,
        "calibration_status": "not_started",
        "trial_flagged": False,
        "flag_reason": None,
        "retraining_candidate": False,
        "similar_trials": [],
        "is_anomaly": False,
        "explanation": None,
        "alerts": [],
        "final_severity": "info",
    }


def run_epoch(
    features: list[float],
    subject_id: str,
    session_id: str,
    *,
    true_label_code: int | None = None,
    true_label_name: str | None = None,
) -> BrainState:
    """Run one EEG epoch through the full real-time agent graph."""
    graph = build_graph()
    result = graph.invoke(
        _initial_state(
            features,
            subject_id,
            session_id,
            true_label_code=true_label_code,
            true_label_name=true_label_name,
        )
    )
    return result


def run_hpo(session_id: str, subject_id: str) -> None:
    """Run the HPO advisor independently."""
    state = _initial_state([], subject_id, session_id, epoch_id="hpo_trigger")
    hpo_advisor_node(state)


def run_report(session_id: str, subject_id: str) -> None:
    """Generate a session report independently."""
    state = _initial_state([], subject_id, session_id, epoch_id="report_trigger")
    report_generator_node(state)
