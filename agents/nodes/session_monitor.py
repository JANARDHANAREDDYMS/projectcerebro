"""Session Monitor Agent for prediction persistence and rolling stats."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from agents.state import BrainState
from agents.tools.mongodb_client import insert_one
from agents.tools.pgvector_client import insert_embedding

FASTAPI_URL = "http://127.0.0.1:8001"
SHOTS_NEEDED = 50
CALIBRATION_STORE: dict[str, dict[int, list[dict]]] = {}
CALIBRATED_SUBJECTS: set[str] = set()

BREAK_THRESHOLD_N = 50
CONFIDENCE_DROP_WINDOW = 20
CONFIDENCE_DROP_THRESH = 0.55


def session_monitor_node(state: BrainState) -> BrainState:
    """Store predictions and embeddings while updating rolling session counters."""
    session_alerts = list(state.get("session_alerts", []))
    suggest_break = False
    subject_id = state["subject_id"]
    label_code = state.get("label_code")
    features = state.get("features", [])
    calibration_status = state.get("calibration_status", "not_started")

    # Accumulate balanced subject-specific calibration epochs and trigger
    # FastAPI adaptation once all classes have enough trials.
    if subject_id not in CALIBRATED_SUBJECTS:
        if subject_id not in CALIBRATION_STORE:
            CALIBRATION_STORE[subject_id] = {0: [], 1: [], 2: []}

        if label_code in [0, 1, 2] and len(features) == 2560:
            store = CALIBRATION_STORE[subject_id]
            if len(store[label_code]) < SHOTS_NEEDED:
                store[label_code].append({"features": features, "label_code": label_code})

        store = CALIBRATION_STORE[subject_id]
        n_cal_left = len(store[0])
        n_cal_right = len(store[1])
        n_cal_rest = len(store[2])
        calibration_status = "collecting"

        print(
            f"[SessionMonitor] Calibration: left={n_cal_left} "
            f"right={n_cal_right} rest={n_cal_rest} / {SHOTS_NEEDED} needed"
        )

        if n_cal_left >= SHOTS_NEEDED and n_cal_right >= SHOTS_NEEDED and n_cal_rest >= SHOTS_NEEDED:
            print(f"[SessionMonitor] Calibration ready for {subject_id}. Calling /calibrate...")
            calibration_status = "calibrating"
            calibration_epochs = []
            for class_id in [0, 1, 2]:
                calibration_epochs.extend(store[class_id][:SHOTS_NEEDED])

            try:
                response = httpx.post(
                    f"{FASTAPI_URL}/calibrate",
                    json={
                        "subject_id": subject_id,
                        "calibration_epochs": calibration_epochs,
                        "adapt_epochs": 20,
                        "adapt_lr": 1e-3,
                    },
                    timeout=60.0,
                )
                if response.status_code == 200:
                    CALIBRATED_SUBJECTS.add(subject_id)
                    calibration_status = "calibrated"
                    print(
                        f"[SessionMonitor] Subject {subject_id} calibrated! "
                        "Future predictions use personalized model."
                    )
                else:
                    calibration_status = "failed"
                    print(f"[SessionMonitor] Calibration failed: {response.status_code} {response.text}")
            except Exception as exc:
                calibration_status = "failed"
                print(f"[SessionMonitor] Calibration error: {exc}")
    else:
        calibration_status = "calibrated"

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
        "calibration_status": calibration_status,
    }
