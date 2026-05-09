"""Pydantic schemas for serving requests and responses."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .config import FEATURE_LEN


class PredictRequest(BaseModel):
    """Request body for single-epoch prediction."""

    features: list[float]
    subject_id: str | None = None
    return_embedding: bool = False

    @field_validator("features")
    @classmethod
    def check_features_length(cls, value: list[float]) -> list[float]:
        """Validate flattened EEG epoch length."""
        if len(value) != FEATURE_LEN:
            raise ValueError(f"features must have length {FEATURE_LEN}, got {len(value)}")
        return value


class CalibrationEpoch(BaseModel):
    """One labeled calibration epoch."""

    features: list[float]
    label_code: int = Field(ge=0, le=2)

    @field_validator("features")
    @classmethod
    def check_features_length(cls, value: list[float]) -> list[float]:
        """Validate flattened EEG epoch length."""
        if len(value) != FEATURE_LEN:
            raise ValueError(f"features must have length {FEATURE_LEN}, got {len(value)}")
        return value


class CalibrateRequest(BaseModel):
    """Request body for subject-specific calibration."""

    subject_id: str
    calibration_epochs: list[CalibrationEpoch]
    adapt_epochs: int = Field(default=20, ge=1, le=500)
    adapt_lr: float = Field(default=1e-3, gt=0.0)

    @field_validator("calibration_epochs")
    @classmethod
    def check_calibration_epochs(cls, value: list[CalibrationEpoch]) -> list[CalibrationEpoch]:
        """Require at least one calibration sample."""
        if not value:
            raise ValueError("calibration_epochs must contain at least one epoch")
        return value


class PredictionResponse(BaseModel):
    """Single-model prediction response."""

    label_code: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    inference_time_ms: float


class EnsemblePredictionResponse(PredictionResponse):
    """Ensemble prediction response with per-model summaries."""

    individual_predictions: dict[str, dict[str, float | int]]


class EmbeddingResponse(BaseModel):
    """Embedding endpoint response."""

    embedding: list[float]
    subject_id: str | None
    model: str

