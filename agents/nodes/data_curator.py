"""Data Curator Agent for trial quality flags and retraining candidates."""
from __future__ import annotations

from datetime import datetime, timezone

from agents.state import BrainState
from agents.tools.mongodb_client import insert_one

RETRAINING_QUALITY_THRESHOLD = 0.75
FLAG_QUALITY_THRESHOLD = 0.40


def data_curator_node(state: BrainState) -> BrainState:
    """Flag bad trials and mark high-quality confident trials for retraining."""
    quality_score = float(state.get("quality_score", 1.0))
    artifact_types = list(state.get("artifact_types", []))
    trial_flagged = quality_score < FLAG_QUALITY_THRESHOLD
    retraining_candidate = (
        quality_score >= RETRAINING_QUALITY_THRESHOLD
        and not artifact_types
        and state.get("label_code") is not None
        and not state.get("is_uncertain", False)
    )
    flag_reason = None
    if trial_flagged:
        flag_reason = f"Quality score {quality_score:.2f} below threshold. Artifacts: {artifact_types}"

    insert_one(
        "trial_quality_flags",
        {
            "epoch_id": state["epoch_id"],
            "subject_id": state["subject_id"],
            "session_id": state["session_id"],
            "timestamp": datetime.now(timezone.utc),
            "quality_score": quality_score,
            "artifact_types": artifact_types,
            "trial_flagged": trial_flagged,
            "flag_reason": flag_reason,
            "retraining_candidate": retraining_candidate,
            "label_code": state.get("label_code"),
            "label_name": state.get("label_name"),
            "recommendation": "skip" if quality_score < 0.40 else "noisy" if quality_score < 0.70 else "use",
        },
    )

    print(
        f"[DataCurator] epoch={state['epoch_id']} flagged={trial_flagged} "
        f"retraining_candidate={retraining_candidate}"
    )
    return {
        **state,
        "trial_flagged": trial_flagged,
        "flag_reason": flag_reason,
        "retraining_candidate": retraining_candidate,
    }

