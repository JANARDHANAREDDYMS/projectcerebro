"""FastAPI app for ProjectCerebro EEG BCI inference."""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from . import config
from .inference import (
    adapt_classifier,
    calibration_arrays,
    embedding_from_tensor,
    predict_from_probabilities,
    probabilities_from_tensor,
    tensor_from_features,
)
from .models import count_parameters, get_device, load_eegnet, load_eegnet_artifacts, load_shallow_artifacts, load_shallowconv
from .schemas import CalibrateRequest, EmbeddingResponse, EnsemblePredictionResponse, PredictRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and preprocessing artifacts once at API startup."""
    device = get_device()
    app.state.device = device
    app.state.shallowconv = load_shallowconv(device)
    app.state.eegnet = load_eegnet(device)
    app.state.shallow_aligner, app.state.shallow_norm = load_shallow_artifacts()
    app.state.eegnet_aligner, app.state.eegnet_norm = load_eegnet_artifacts()
    app.state.calibrated_subjects = {}
    app.state.lock = threading.Lock()
    yield
    app.state.calibrated_subjects.clear()


app = FastAPI(title="ProjectCerebro Serving API", version=config.VERSION, lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Return service health and loaded model state."""
    calibrated = sorted(app.state.calibrated_subjects.keys())
    return {
        "status": "healthy",
        "models_loaded": {
            "shallowconv": getattr(app.state, "shallowconv", None) is not None,
            "eegnet": getattr(app.state, "eegnet", None) is not None,
        },
        "calibrated_subjects": calibrated,
        "device": str(app.state.device),
        "version": config.VERSION,
    }


@app.get("/models/info")
async def models_info() -> dict:
    """Return model metadata used by the serving layer."""
    return {
        "shallowconv": {
            "checkpoint": str(config.SHALLOW_CHECKPOINT.relative_to(config.PROJECT_ROOT)),
            "n_parameters": count_parameters(app.state.shallowconv),
            "pretrained_on": "physionet",
            "n_pretrain_subjects": 104,
        },
        "eegnet": {
            "checkpoint": str(config.EEGNET_CHECKPOINT.relative_to(config.PROJECT_ROOT)),
            "n_parameters": count_parameters(app.state.eegnet),
            "pretrained_on": "physionet",
            "n_pretrain_subjects": 104,
        },
        "ensemble_weights": {
            "shallowconv": config.ENSEMBLE_WEIGHT_SHALLOW,
            "eegnet": config.ENSEMBLE_WEIGHT_EEGNET,
        },
        "input_shape": [1, config.N_CHANNELS, config.N_SAMPLES],
        "n_classes": config.N_CLASSES,
        "classes": config.CLASS_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest) -> dict:
    """Predict a motor-imagery class with ShallowConvNet."""
    start = time.time()
    x = tensor_from_features(
        request.features,
        aligner=app.state.shallow_aligner,
        norm_stats=app.state.shallow_norm,
        device=app.state.device,
    )
    probs = probabilities_from_tensor(app.state.shallowconv, x)
    elapsed_ms = (time.time() - start) * 1000.0
    return predict_from_probabilities(probs, model_name="shallowconv", elapsed_ms=elapsed_ms)


@app.post("/predict/ensemble", response_model=EnsemblePredictionResponse)
async def predict_ensemble(request: PredictRequest) -> dict:
    """Predict with a weighted ShallowConvNet plus EEGNet ensemble."""
    start = time.time()
    shallow_x = tensor_from_features(
        request.features,
        aligner=app.state.shallow_aligner,
        norm_stats=app.state.shallow_norm,
        device=app.state.device,
    )
    eegnet_x = tensor_from_features(
        request.features,
        aligner=app.state.eegnet_aligner,
        norm_stats=app.state.eegnet_norm,
        device=app.state.device,
    )
    shallow_probs = probabilities_from_tensor(app.state.shallowconv, shallow_x)
    eegnet_probs = probabilities_from_tensor(app.state.eegnet, eegnet_x)
    probs = config.ENSEMBLE_WEIGHT_SHALLOW * shallow_probs + config.ENSEMBLE_WEIGHT_EEGNET * eegnet_probs
    elapsed_ms = (time.time() - start) * 1000.0

    payload = predict_from_probabilities(probs, model_name="ensemble", elapsed_ms=elapsed_ms)
    shallow_top = int(np.argmax(shallow_probs.reshape(-1)))
    eegnet_top = int(np.argmax(eegnet_probs.reshape(-1)))
    payload["individual_predictions"] = {
        "shallowconv": {"label_code": shallow_top, "confidence": float(shallow_probs.reshape(-1)[shallow_top])},
        "eegnet": {"label_code": eegnet_top, "confidence": float(eegnet_probs.reshape(-1)[eegnet_top])},
    }
    return payload


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: PredictRequest) -> dict:
    """Return the 128-dimensional EEGNet embedding for an epoch."""
    x = tensor_from_features(
        request.features,
        aligner=app.state.eegnet_aligner,
        norm_stats=app.state.eegnet_norm,
        device=app.state.device,
    )
    return {
        "embedding": embedding_from_tensor(app.state.eegnet, x),
        "subject_id": request.subject_id,
        "model": "eegnet",
    }


@app.post("/calibrate")
async def calibrate(request: CalibrateRequest) -> dict:
    """Fine-tune a subject-specific ShallowConvNet classifier in memory."""
    epochs, labels = calibration_arrays([item.model_dump() for item in request.calibration_epochs])
    model = load_shallowconv(app.state.device)
    personalized, final_loss = adapt_classifier(
        model,
        epochs,
        labels,
        device=app.state.device,
        adapt_epochs=request.adapt_epochs,
        adapt_lr=request.adapt_lr,
    )
    with app.state.lock:
        app.state.calibrated_subjects[request.subject_id] = personalized

    return {
        "subject_id": request.subject_id,
        "n_calibration_trials": int(len(request.calibration_epochs)),
        "adapt_epochs": int(request.adapt_epochs),
        "train_loss_final": float(final_loss),
        "status": "calibrated",
        "message": f"Model calibrated for subject {request.subject_id}",
    }


@app.post("/predict/personalized", response_model=PredictionResponse)
async def predict_personalized(request: PredictRequest) -> dict:
    """Predict with a subject-specific calibrated ShallowConvNet model."""
    if not request.subject_id:
        raise HTTPException(status_code=422, detail="subject_id is required for personalized prediction.")

    with app.state.lock:
        personalized = app.state.calibrated_subjects.get(request.subject_id)
    if personalized is None:
        raise HTTPException(
            status_code=404,
            detail=f"Subject {request.subject_id} not calibrated. POST /calibrate first.",
        )

    start = time.time()
    x = tensor_from_features(
        request.features,
        aligner=personalized.aligner,
        norm_stats=personalized.norm_stats,
        device=app.state.device,
    )
    probs = probabilities_from_tensor(personalized.model, x)
    elapsed_ms = (time.time() - start) * 1000.0
    return predict_from_probabilities(probs, model_name="personalized_shallowconv", elapsed_ms=elapsed_ms)

