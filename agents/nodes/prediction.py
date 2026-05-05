"""Prediction Agent that calls the local FastAPI serving layer."""
from __future__ import annotations

from agents.state import BrainState
from agents.tools.fastapi_client import post_json

CONFIDENCE_THRESHOLD = 0.60


def prediction_node(state: BrainState) -> BrainState:
    """Run personalized or ensemble inference and fetch the EEGNet embedding."""
    if state.get("skip_inference", False):
        print(f"[Prediction] epoch={state['epoch_id']} skipped due to signal quality")
        return {
            **state,
            "label_code": None,
            "label_name": None,
            "confidence": None,
            "probabilities": None,
            "embedding": None,
            "model_used": None,
            "is_uncertain": False,
        }

    payload = {"features": state["features"], "subject_id": state["subject_id"]}
    status, prediction = post_json("/predict/personalized", payload, timeout=10.0)
    model_used = "personalized"
    if status != 200:
        status, prediction = post_json("/predict/ensemble", payload, timeout=10.0)
        model_used = "ensemble"
    if status != 200:
        print(f"[Prediction] FastAPI prediction failed: {prediction}")
        return {**state, "is_uncertain": False}

    emb_status, embedding_body = post_json("/embed", payload, timeout=10.0)
    embedding = embedding_body.get("embedding") if emb_status == 200 else None
    confidence = float(prediction["confidence"])

    print(
        f"[Prediction] epoch={state['epoch_id']} label={prediction['label_name']} "
        f"conf={confidence:.3f} model={model_used}"
    )
    return {
        **state,
        "label_code": int(prediction["label_code"]),
        "label_name": prediction["label_name"],
        "confidence": confidence,
        "probabilities": prediction["probabilities"],
        "embedding": embedding,
        "model_used": model_used,
        "is_uncertain": confidence < CONFIDENCE_THRESHOLD,
    }

