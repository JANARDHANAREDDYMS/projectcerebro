"""Shared state schema for the ProjectCerebro agent graph."""
from __future__ import annotations

from typing import Optional, TypedDict


class BrainState(TypedDict):
    """State passed between ProjectCerebro graph nodes."""

    features: list[float]
    subject_id: str
    session_id: str
    epoch_id: str
    timestamp: str

    signal_quality: str
    quality_score: float
    bad_channels: list[str]
    artifact_types: list[str]
    skip_inference: bool

    label_code: Optional[int]
    label_name: Optional[str]
    confidence: Optional[float]
    probabilities: Optional[dict]
    embedding: Optional[list[float]]
    model_used: Optional[str]
    is_uncertain: bool

    n_predictions: int
    n_left: int
    n_right: int
    n_rest: int
    mean_confidence: float
    session_alerts: list[str]
    suggest_break: bool

    trial_flagged: bool
    flag_reason: Optional[str]
    retraining_candidate: bool

    similar_trials: list[dict]
    is_anomaly: bool
    explanation: Optional[str]

    alerts: list[dict]
    final_severity: str

