"""Inference, normalization, and calibration helpers for serving."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from ml_core.data.normalize import EuclideanAligner, NormStats, apply_norm_stats, compute_norm_stats

from .config import CLASS_NAMES, FEATURE_LEN, LABEL_MAP, N_CHANNELS, N_SAMPLES
from .models import freeze_backbone_for_classifier_adaptation


@dataclass
class PersonalizedModel:
    """Subject-specific adapted model with calibration-fitted preprocessing."""

    model: nn.Module
    aligner: EuclideanAligner
    norm_stats: NormStats


def features_to_epochs(features: list[float] | np.ndarray) -> np.ndarray:
    """Convert flat feature vector(s) to `(N, 5, 512)` numpy epochs."""
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 1:
        if array.shape[0] != FEATURE_LEN:
            raise ValueError(f"Expected {FEATURE_LEN} features, got {array.shape[0]}")
        return array.reshape(1, N_CHANNELS, N_SAMPLES)
    if array.ndim == 2 and array.shape[1] == FEATURE_LEN:
        return array.reshape(-1, N_CHANNELS, N_SAMPLES)
    if array.ndim == 3 and array.shape[1:] == (N_CHANNELS, N_SAMPLES):
        return array
    raise ValueError(f"Unsupported feature shape: {array.shape}")


def preprocess_epochs(
    epochs: np.ndarray,
    *,
    aligner: EuclideanAligner,
    norm_stats: NormStats,
) -> np.ndarray:
    """Apply EA and z-score normalization and add the model channel dimension."""
    aligned = aligner.transform(epochs)
    normalized = apply_norm_stats(aligned, norm_stats)
    return normalized[:, None, :, :].astype(np.float32)


def tensor_from_features(
    features: list[float],
    *,
    aligner: EuclideanAligner,
    norm_stats: NormStats,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a flat epoch and convert it to a model tensor."""
    x = preprocess_epochs(features_to_epochs(features), aligner=aligner, norm_stats=norm_stats)
    return torch.from_numpy(x).to(device)


def probabilities_from_tensor(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Run model inference and return softmax probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.detach().cpu().numpy()


def predict_from_probabilities(probabilities: np.ndarray, *, model_name: str, elapsed_ms: float) -> dict:
    """Build an API prediction payload from probability scores."""
    probs = probabilities.reshape(-1)
    label_code = int(np.argmax(probs))
    return {
        "label_code": label_code,
        "label_name": LABEL_MAP[label_code],
        "confidence": float(probs[label_code]),
        "probabilities": {name: float(probs[idx]) for idx, name in enumerate(CLASS_NAMES)},
        "model": model_name,
        "inference_time_ms": float(elapsed_ms),
    }


def embedding_from_tensor(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Return the EEGNet embedding for a normalized epoch tensor."""
    model.eval()
    with torch.no_grad():
        _, embedding = model(x, return_embedding=True)
    return [float(value) for value in embedding.detach().cpu().numpy().reshape(-1)]


def calibration_arrays(calibration_epochs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert calibration request items to epoch and label arrays."""
    features = np.asarray([item["features"] for item in calibration_epochs], dtype=np.float32)
    labels = np.asarray([item["label_code"] for item in calibration_epochs], dtype=np.int64)
    return features_to_epochs(features), labels


def adapt_classifier(
    model: nn.Module,
    calibration_epochs: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    adapt_epochs: int,
    adapt_lr: float,
) -> tuple[PersonalizedModel, float]:
    """Fit calibration preprocessing and fine-tune only the classifier head."""
    if calibration_epochs.shape[0] != labels.shape[0]:
        raise ValueError("Calibration epoch and label counts do not match.")

    aligner = EuclideanAligner().fit(calibration_epochs)
    aligned = aligner.transform(calibration_epochs)
    norm_stats = compute_norm_stats(aligned)
    x = apply_norm_stats(aligned, norm_stats)[:, None, :, :].astype(np.float32)

    freeze_backbone_for_classifier_adaptation(model)
    model.to(device)
    model.train()

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(labels.astype(np.int64)))
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=adapt_lr)
    criterion = nn.CrossEntropyLoss()
    final_loss = 0.0

    for _ in range(adapt_epochs):
        losses: list[float] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else 0.0

    model.eval()
    return PersonalizedModel(model=model, aligner=aligner, norm_stats=norm_stats), final_loss

