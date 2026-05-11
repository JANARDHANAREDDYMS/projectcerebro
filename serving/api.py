"""FastAPI app for ProjectCerebro EEG BCI inference."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

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
from .schemas import (
    CalibrateRequest,
    EmbeddingResponse,
    EnsemblePredictionResponse,
    PredictRequest,
    PredictionResponse,
    StartPredictionRequest,
)


FIF_TARGET_SFREQ = 128
STREAM_CACHE_DIR = config.PROJECT_ROOT / "artifacts" / "stream_cache"
EPOCH_DELTA_PATH = config.PROJECT_ROOT / "delta_lake" / "epochs_mi_v1_ch5_sr128_bp8_30"
EPOCH_STREAM_COLUMNS = [
    "epoch_id",
    "dataset",
    "subject_id",
    "label_code",
    "label_name",
    "features",
    "epoch_start_sec",
    "epoch_end_sec",
    "filter_version",
]


def _load_fif_stream_state(app: FastAPI) -> None:
    """Initialize cached Stream 1 playback state without reading FIF via MNE."""
    STREAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    app.state.fif_cache = {}
    available = sorted(path.name.split("_", 1)[0] for path in STREAM_CACHE_DIR.glob("*_continuous.npy"))
    print(f"STARTUP: Stream cache subjects available: {available or 'none'}", flush=True)


def _load_subject_fif_cache(subject: str) -> dict:
    """Load one subject's prebuilt continuous EEG stream cache."""
    subject = subject.upper()
    data_path = STREAM_CACHE_DIR / f"{subject}_continuous.npy"
    times_path = STREAM_CACHE_DIR / f"{subject}_times.npy"
    meta_path = STREAM_CACHE_DIR / f"{subject}_continuous_meta.json"

    missing = [str(path) for path in (data_path, times_path, meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"No stream cache found for {subject}. Missing: {missing}. "
            f"Run: cerebro_env/bin/python scripts/read_fif_fast.py --subject {subject}"
        )

    data = np.load(data_path, mmap_mode="r")
    times = np.load(times_path, mmap_mode="r")
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    channels = meta.get("channels") or ["FZ", "C3", "CZ", "C4", "PZ"]
    sfreq = int(meta.get("sfreq", FIF_TARGET_SFREQ))
    return {
        "subject": subject,
        "data": data,
        "times": times,
        "channels": channels,
        "sfreq": sfreq,
        "duration": float(meta.get("duration", float(times[-1]) if len(times) else 0.0)),
        "source": meta.get("source"),
    }


def _has_parquet_magic(path: Path) -> bool:
    """Return true if a file has Parquet magic bytes at header and footer."""
    stat = path.stat()
    if stat.st_size < 8 or getattr(stat, "st_blocks", 1) == 0:
        return False
    with path.open("rb") as handle:
        head = handle.read(4)
        handle.seek(-4, 2)
        tail = handle.read(4)
    return head == b"PAR1" and tail == b"PAR1"


def _load_subject_epoch_cache(subject: str, filter_version: str = "bp_8_30_v1") -> dict:
    """Load BCI IV-2a epoch rows for one subject from the Delta parquet directory."""
    subject = subject.upper()
    cache_key = f"{subject}:{filter_version}"
    cached = app.state.epoch_cache.get(cache_key)
    if cached is not None:
        return cached

    import pandas as pd

    start = time.time()
    parquet_files = [
        path
        for path in sorted(EPOCH_DELTA_PATH.glob("*.parquet"))
        if not path.name.startswith(".") and _has_parquet_magic(path)
    ]
    if not parquet_files:
        raise FileNotFoundError(f"No valid parquet files found under {EPOCH_DELTA_PATH}")

    frames = []
    for parquet_path in parquet_files:
        frame = pd.read_parquet(parquet_path, columns=EPOCH_STREAM_COLUMNS)
        frame = frame[
            (frame["dataset"] == "bci_iv_2a")
            & (frame["filter_version"] == filter_version)
            & (frame["subject_id"] == subject)
        ]
        if len(frame):
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No BCI IV-2a epochs found for {subject} ({filter_version})")

    df = pd.concat(frames, ignore_index=True)
    if "epoch_id" in df.columns:
        df = df.drop_duplicates(subset=["epoch_id"], keep="first")
    else:
        df = df.drop_duplicates(subset=["epoch_start_sec", "label_code"], keep="first")
    if "epoch_start_sec" in df.columns:
        df = df.sort_values(["epoch_start_sec", "epoch_id"], kind="stable")

    rows = df.to_dict("records")
    elapsed = time.time() - start
    payload = {
        "subject": subject,
        "filter_version": filter_version,
        "rows": rows,
        "n_epochs": len(rows),
        "load_time_sec": elapsed,
        "valid_parquet_files": len(parquet_files),
    }
    app.state.epoch_cache[cache_key] = payload
    print(
        f"[stream/epochs] Loaded {len(rows)} epochs for {subject} "
        f"from {len(parquet_files)} parquet files in {elapsed:.2f}s",
        flush=True,
    )
    return payload


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
    app.state.prediction_runs = {}
    app.state.lock = threading.Lock()
    app.state.epoch_cache = {}
    _load_fif_stream_state(app)
    yield
    app.state.calibrated_subjects.clear()
    for run in app.state.prediction_runs.values():
        for proc in (run.get("consumer"), run.get("producer")):
            if proc is not None and proc.poll() is None:
                proc.terminate()
    app.state.prediction_runs.clear()
    app.state.epoch_cache.clear()


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


@app.get("/stream/fif")
async def stream_fif(request: Request, subject: str = "A09", start_sec: float = 0.0):
    """
    Stream raw continuous EEG from a subject's prebuilt FIF cache.

    Sends one second of 128 Hz samples per SSE event and loops at file end.
    """
    subject = subject.upper()

    async def generate():
        load_error = None
        with app.state.lock:
            cache = app.state.fif_cache.get(subject)
            if cache is None:
                try:
                    cache = _load_subject_fif_cache(subject)
                    app.state.fif_cache[subject] = cache
                    print(
                        f"[stream/fif] Loaded cache for {subject}: "
                        f"shape={cache['data'].shape} duration={cache['duration']:.1f}s",
                        flush=True,
                    )
                except Exception as exc:
                    load_error = str(exc)

        if load_error:
            yield f"data: {json.dumps({'type': 'error', 'subject': subject, 'error': load_error})}\n\n"
            return

        if cache is None:
            yield f"data: {json.dumps({'type': 'error', 'subject': subject, 'error': 'No FIF cache loaded'})}\n\n"
            return

        data = cache["data"]
        sfreq = int(cache["sfreq"])
        channels = cache["channels"]
        chunk_size = sfreq
        n_samples = int(data.shape[1])
        start_idx = max(0, min(int(float(start_sec) * sfreq), n_samples - 1))
        start_t = float(start_idx / sfreq)

        meta = {
            "type": "meta",
            "subject": subject,
            "channels": channels,
            "sfreq": sfreq,
            "duration": float(n_samples / sfreq),
            "start_sec": start_t,
            "source": cache.get("source"),
        }
        yield f"data: {json.dumps(meta)}\n\n"

        t = start_t
        idx = start_idx
        while True:
            if await request.is_disconnected():
                print("[stream/fif] Client disconnected", flush=True)
                break

            end_idx = min(idx + chunk_size, n_samples)
            chunk = data[:, idx:end_idx]
            payload = {
                "type": "chunk",
                "t": round(t, 3),
                "channels": {
                    channel: chunk[channel_idx].tolist()
                    for channel_idx, channel in enumerate(channels)
                },
                "n_samples": int(chunk.shape[1]),
            }
            yield f"data: {json.dumps(payload)}\n\n"

            idx += chunk_size
            t += 1.0
            if idx >= n_samples:
                idx = start_idx
                t = start_t
                print(f"[stream/fif] Looping back to {start_t:.3f}s", flush=True)

            await asyncio.sleep(1.0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Connection": "keep-alive",
        },
    )


@app.get("/stream/epochs")
async def stream_epochs(
    request: Request,
    subject: str = "A09",
    filter_version: str = "bp_8_30_v1",
    interval_sec: float = 1.0,
):
    """
    Stream preprocessed 4-second BCI epoch windows from Delta parquet.

    This is the Stream 2 source. Each SSE event contains one 5x512 epoch plus
    its label and original recording timeline offsets.
    """
    subject = subject.upper()
    interval_sec = max(0.05, float(interval_sec))

    async def generate():
        try:
            cache = await asyncio.to_thread(_load_subject_epoch_cache, subject, filter_version)
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'subject': subject, 'error': str(exc)})}\n\n"
            return

        rows = cache["rows"]
        meta = {
            "type": "meta",
            "subject": subject,
            "filter_version": filter_version,
            "n_epochs": cache["n_epochs"],
            "load_time_sec": round(cache["load_time_sec"], 3),
            "valid_parquet_files": cache["valid_parquet_files"],
            "channels": ["FZ", "C3", "CZ", "C4", "PZ"],
            "sfreq": FIF_TARGET_SFREQ,
            "n_samples": 512,
        }
        yield f"data: {json.dumps(meta)}\n\n"

        idx = 0
        while True:
            if await request.is_disconnected():
                print("[stream/epochs] Client disconnected", flush=True)
                break

            row = rows[idx]
            features = row["features"]
            if hasattr(features, "tolist"):
                features = features.tolist()

            payload = {
                "type": "epoch",
                "subject": subject,
                "index": idx,
                "epoch_id": str(row.get("epoch_id", "")),
                "label_code": int(row["label_code"]),
                "label_name": str(row.get("label_name", row["label_code"])),
                "epoch_start_sec": float(row.get("epoch_start_sec", 0.0)),
                "epoch_end_sec": float(row.get("epoch_end_sec", 0.0)),
                "features": features,
                "n_channels": 5,
                "n_samples": 512,
            }
            yield f"data: {json.dumps(payload)}\n\n"

            idx = (idx + 1) % len(rows)
            await asyncio.sleep(interval_sec)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Connection": "keep-alive",
        },
    )


@app.get("/stream/agents")
async def stream_agents(
    request: Request,
    session_id: str | None = None,
    poll_interval: float = 1.0,
):
    """
    Stream live LangGraph agent outputs from MongoDB for dashboard Stream 3.

    The Kafka consumer persists predictions, quality flags, and alerts to
    MongoDB. This endpoint polls for new prediction documents and emits compact
    SSE events that the React dashboard can render as a live agent log.
    """

    async def generate():
        try:
            from pymongo import MongoClient
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': f'pymongo unavailable: {exc}'})}\n\n"
            return

        mongo_uri = os.getenv(
            "CEREBRO_MONGO_URI",
            "mongodb://cerebro:cerebro123@localhost:27017/?authSource=admin",
        )
        mongo_db = os.getenv("CEREBRO_MONGO_DB", "projectcerebro")

        try:
            mongo = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
            mongo.admin.command("ping")
            db = mongo[mongo_db]
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        initial_query = {"session_id": session_id} if session_id else {}
        try:
            initial_count = db.predictions.count_documents(initial_query)
        except Exception:
            initial_count = 0

        yield f"data: {json.dumps({'type': 'connected', 'database': mongo_db, 'session_id': session_id, 'initial_count': initial_count})}\n\n"

        last_id = None
        n_predictions = 0
        n_left = 0
        n_right = 0
        n_rest = 0
        n_alerts = 0
        confidence_sum = 0.0
        poll_interval_sec = max(0.25, float(poll_interval))

        while True:
            if await request.is_disconnected():
                mongo.close()
                break

            try:
                query = {}
                if session_id:
                    query["session_id"] = session_id
                if last_id is not None:
                    query["_id"] = {"$gt": last_id}

                new_predictions = list(
                    db.predictions.find(query).sort("_id", 1).limit(10)
                )

                for pred in new_predictions:
                    last_id = pred["_id"]
                    n_predictions += 1

                    label = str(pred.get("label_name", "unknown"))
                    confidence = float(pred.get("confidence") or 0.0)
                    confidence_sum += confidence

                    if label == "left":
                        n_left += 1
                    elif label == "right":
                        n_right += 1
                    elif label == "rest":
                        n_rest += 1

                    epoch_alerts = list(
                        db.alerts.find(
                            {"epoch_id": pred.get("epoch_id")},
                            {"_id": 0, "severity": 1, "message": 1, "agent": 1},
                        )
                    )
                    n_alerts += len(epoch_alerts)

                    timestamp = pred.get("timestamp", "")
                    if hasattr(timestamp, "isoformat"):
                        timestamp = timestamp.isoformat()
                    else:
                        timestamp = str(timestamp)

                    model_used = str(pred.get("model_used", "ensemble"))
                    explicit_calibration = pred.get("calibration_status")
                    calibration_status = explicit_calibration or (
                        "calibrated" if "personalized" in model_used else "collecting"
                    )

                    event = {
                        "type": "prediction",
                        "epoch_id": str(pred.get("epoch_id", "")),
                        "subject_id": str(pred.get("subject_id", "")),
                        "session_id": str(pred.get("session_id", "")),
                        "label_name": label,
                        "label_code": int(pred.get("label_code", -1)),
                        "confidence": round(confidence, 4),
                        "model_used": model_used,
                        "signal_quality": str(pred.get("signal_quality", "unknown")),
                        "quality_score": round(float(pred.get("quality_score") or 0.0), 3),
                        "calibration_status": calibration_status,
                        "alerts": epoch_alerts,
                        "timestamp": timestamp,
                        "n_predictions": n_predictions,
                        "n_left": n_left,
                        "n_right": n_right,
                        "n_rest": n_rest,
                        "mean_confidence": round(confidence_sum / n_predictions, 4),
                        "n_alerts": n_alerts,
                    }

                    yield f"data: {json.dumps(event)}\n\n"

                if not new_predictions and n_predictions > 0:
                    stats = {
                        "type": "stats",
                        "n_predictions": n_predictions,
                        "n_left": n_left,
                        "n_right": n_right,
                        "n_rest": n_rest,
                        "mean_confidence": round(confidence_sum / n_predictions, 4),
                        "n_alerts": n_alerts,
                    }
                    yield f"data: {json.dumps(stats)}\n\n"
            except Exception as exc:
                print(f"[stream/agents] Error: {exc}", flush=True)

            await asyncio.sleep(poll_interval_sec)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Connection": "keep-alive",
        },
    )


@app.post("/stream/start-prediction")
async def start_prediction_stream(request: StartPredictionRequest) -> dict:
    """
    Start Kafka producer and LangGraph consumer processes for one dashboard run.

    The dashboard creates a session_id when Run Stream is clicked. This endpoint
    uses that same session id for the producer and consumer so Stream 3 can read
    only MongoDB outputs for the active dashboard run.
    """
    subject = request.subject.upper()
    session_id = request.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id is required.")

    with app.state.lock:
        existing = app.state.prediction_runs.get(session_id)
        if existing:
            consumer = existing.get("consumer")
            producer = existing.get("producer")
            consumer_running = consumer is not None and consumer.poll() is None
            producer_running = producer is not None and producer.poll() is None
            if consumer_running or producer_running:
                return {
                    "status": "already_running",
                    "subject": subject,
                    "session_id": session_id,
                    "consumer_pid": consumer.pid if consumer is not None else None,
                    "producer_pid": producer.pid if producer is not None else None,
                }

    logs_dir = config.PROJECT_ROOT / "artifacts" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_session = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in session_id)
    consumer_log_path = logs_dir / f"dashboard_consumer_{safe_session}.log"
    producer_log_path = logs_dir / f"dashboard_producer_{safe_session}.log"

    consumer_log = consumer_log_path.open("ab")
    producer_log = producer_log_path.open("ab")

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    env.setdefault("MPLBACKEND", "Agg")

    consumer_cmd = [
        sys.executable,
        "-u",
        "-m",
        "agents.kafka_consumer",
        "--session-id",
        session_id,
        "--filter-session-id",
        session_id,
        "--topic",
        "raw-eeg",
        "--group-id",
        f"cerebro-dashboard-{safe_session}",
        "--auto-offset-reset",
        "earliest",
        "--timeout-ms",
        str(int(request.timeout_ms)),
    ]
    producer_cmd = [
        sys.executable,
        "-u",
        "scripts/kafka_eeg_producer.py",
        "--subject",
        subject,
        "--session-id",
        session_id,
        "--interval",
        str(float(request.interval)),
    ]

    try:
        consumer = subprocess.Popen(
            consumer_cmd,
            cwd=str(config.PROJECT_ROOT),
            env=env,
            stdout=consumer_log,
            stderr=subprocess.STDOUT,
        )
        await asyncio.sleep(2.0)
        producer = subprocess.Popen(
            producer_cmd,
            cwd=str(config.PROJECT_ROOT),
            env=env,
            stdout=producer_log,
            stderr=subprocess.STDOUT,
        )
        consumer_log.close()
        producer_log.close()
    except Exception as exc:
        consumer_log.close()
        producer_log.close()
        raise HTTPException(status_code=500, detail=f"Failed to start prediction run: {exc}") from exc

    with app.state.lock:
        app.state.prediction_runs[session_id] = {
            "subject": subject,
            "consumer": consumer,
            "producer": producer,
            "consumer_log": str(consumer_log_path),
            "producer_log": str(producer_log_path),
            "started_at": time.time(),
        }

    return {
        "status": "started",
        "subject": subject,
        "session_id": session_id,
        "consumer_pid": consumer.pid,
        "producer_pid": producer.pid,
        "consumer_log": str(consumer_log_path.relative_to(config.PROJECT_ROOT)),
        "producer_log": str(producer_log_path.relative_to(config.PROJECT_ROOT)),
    }


dashboard_dist = os.path.join(os.path.dirname(__file__), "..", "cerebro-dashboard", "dist")
if os.path.exists(dashboard_dist):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dist, html=True), name="dashboard")
    print("STARTUP: Dashboard served at /dashboard", flush=True)
else:
    print("STARTUP: Dashboard not built yet. Run: cd cerebro-dashboard && npm run build", flush=True)
