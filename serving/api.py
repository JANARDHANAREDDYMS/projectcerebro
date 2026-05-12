"""FastAPI app for ProjectCerebro EEG BCI inference."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid
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


def _worker_running(worker: object | None) -> bool:
    """Return true for a running subprocess or thread-like worker."""
    if worker is None:
        return False
    if hasattr(worker, "poll"):
        return worker.poll() is None
    if hasattr(worker, "is_alive"):
        return bool(worker.is_alive())
    return False


def _worker_pid(worker: object | None) -> int | None:
    """Return a PID for subprocess workers; threads do not have one."""
    return getattr(worker, "pid", None)


def _build_fif_cache_from_binary(subject: str) -> dict:
    """Build and load a subject FIF stream cache with the direct binary reader."""
    subject = subject.upper()
    from scripts.read_fif_fast import read_fif_binary, resolve_fif_path, save_cache

    fif_path = resolve_fif_path(subject)
    start = time.time()
    data, times, channels = read_fif_binary(fif_path)
    elapsed = time.time() - start
    save_cache(subject, fif_path, data, times, channels, elapsed)
    print(
        f"[stream/fif] Built binary FIF cache for {subject}: "
        f"shape={data.shape} duration={float(times[-1]):.1f}s "
        f"in {elapsed:.2f}s",
        flush=True,
    )
    return _load_subject_fif_cache(subject)


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


def _build_pseudo_fif_cache_from_epochs(subject: str, filter_version: str = "bp_8_30_v1") -> dict:
    """
    Build a Stream 1 compatible continuous signal from 4-second epoch rows.

    Some local FIF files may be Finder/iCloud placeholders and cannot be read
    until downloaded. This fallback keeps dashboard demos available by placing
    each preprocessed epoch at its original recording time and filling gaps with
    low-amplitude noise.
    """
    cache = _load_subject_epoch_cache(subject, filter_version)
    rows = cache["rows"]
    if not rows:
        raise FileNotFoundError(f"No epochs available to synthesize Stream 1 for {subject}.")

    sfreq = FIF_TARGET_SFREQ
    channels = ["FZ", "C3", "CZ", "C4", "PZ"]
    max_end = max(float(row.get("epoch_end_sec", 0.0)) for row in rows)
    n_samples = max(int(max_end * sfreq) + 1, sfreq)

    rng = np.random.default_rng(abs(hash((subject, filter_version))) % (2**32))
    data = rng.normal(0.0, 0.05e-6, size=(len(channels), n_samples)).astype(np.float32)

    for row in rows:
        features = row["features"]
        if hasattr(features, "tolist"):
            features = features.tolist()
        if len(features) != 5 * 512:
            continue
        epoch = np.asarray(features, dtype=np.float32).reshape(5, 512)
        start_idx = int(float(row.get("epoch_start_sec", 0.0)) * sfreq)
        end_idx = min(start_idx + 512, n_samples)
        width = max(0, end_idx - start_idx)
        if width:
            data[:, start_idx:end_idx] = epoch[:, :width]

    return {
        "subject": subject.upper(),
        "data": data,
        "times": np.arange(n_samples, dtype=np.float32) / sfreq,
        "channels": channels,
        "sfreq": sfreq,
        "duration": float(n_samples / sfreq),
        "source": "pseudo_continuous_from_epochs",
    }


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
        stop_event = run.get("producer_stop")
        if stop_event is not None:
            stop_event.set()
        for worker in (run.get("consumer"), run.get("producer")):
            if worker is None:
                continue
            if hasattr(worker, "poll") and worker.poll() is None:
                worker.terminate()
    app.state.prediction_runs.clear()
    app.state.epoch_cache.clear()


app = FastAPI(title="ProjectCerebro Serving API", version=config.VERSION, lifespan=lifespan)


def _dashboard_kafka_producer(
    subject: str,
    session_id: str,
    start_sec: float | None,
    time_scale: float,
    stop_event: threading.Event,
    log_path: Path,
) -> None:
    """Publish dashboard epochs from the already-loaded FastAPI epoch cache."""
    from kafka import KafkaProducer

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
            handle.flush()

    try:
        cache = _load_subject_epoch_cache(subject)
        rows = sorted(cache["rows"], key=lambda item: float(item.get("epoch_start_sec", 0.0)))
        if start_sec is not None:
            rows = [row for row in rows if float(row.get("epoch_start_sec", 0.0)) >= start_sec]
        if not rows:
            log(f"ERROR: no rows available for {subject} at start_sec={start_sec}")
            return

        base_sec = float(start_sec) if start_sec is not None else float(rows[0].get("epoch_start_sec", 0.0))
        time_scale = max(float(time_scale), 1e-6)
        label_counts: dict[str, int] = {}
        for row in rows:
            label = str(row.get("label_name", row.get("label_code", "unknown")))
            label_counts[label] = label_counts.get(label, 0) + 1

        log("Loading epochs from FastAPI Stream 2 cache")
        log(f"Subject:     {subject}")
        log(f"Epochs:      {len(rows)}  {label_counts}")
        log(f"Timeline:    True")
        log(f"Start sec:   {base_sec:.3f}s")
        log(f"Time scale:  {time_scale}x")
        log(f"Session:     {session_id}")
        log("Connecting to Kafka...")

        producer = KafkaProducer(
            bootstrap_servers="localhost:9092",
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            acks="all",
            retries=3,
        )
        log("Connected.")

        loop_started_at = time.monotonic()
        total_sent = 0
        for row in rows:
            if stop_event.is_set():
                break

            epoch_start_sec = float(row.get("epoch_start_sec", 0.0))
            target_elapsed = max(0.0, epoch_start_sec - base_sec) / time_scale
            while True:
                sleep_for = loop_started_at + target_elapsed - time.monotonic()
                if sleep_for <= 0 or stop_event.is_set():
                    break
                time.sleep(min(0.1, sleep_for))
            if stop_event.is_set():
                break

            features = row["features"]
            if hasattr(features, "tolist"):
                features = features.tolist()

            message = {
                "epoch_id": str(uuid.uuid4()),
                "subject_id": subject,
                "session_id": session_id,
                "features": features,
                "label_code": int(row["label_code"]),
                "label_name": str(row.get("label_name", row["label_code"])),
                "epoch_start_sec": epoch_start_sec,
                "epoch_end_sec": float(row.get("epoch_end_sec", 0.0)),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "iteration": 1,
            }
            producer.send("raw-eeg", value=message)
            producer.flush()
            total_sent += 1

            if total_sent == 1 or total_sent % 10 == 0:
                log(
                    f"  [{total_sent:>4}] label={message['label_name']:<6} "
                    f"epoch={message['epoch_id'][:8]} "
                    f"start={epoch_start_sec:.1f}s session={session_id}"
                )

        producer.flush()
        producer.close()
        log(f"Producer closed cleanly. Total sent: {total_sent}")
    except Exception as exc:
        log(f"ERROR: {exc}")


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
                    try:
                        cache = _build_fif_cache_from_binary(subject)
                        app.state.fif_cache[subject] = cache
                        print(
                            f"[stream/fif] Loaded on-demand binary cache for {subject}: "
                            f"shape={cache['data'].shape} duration={cache['duration']:.1f}s "
                            f"because FIF cache was unavailable: {exc}",
                            flush=True,
                        )
                    except Exception as binary_exc:
                        try:
                            cache = _build_pseudo_fif_cache_from_epochs(subject)
                            app.state.fif_cache[subject] = cache
                            print(
                                f"[stream/fif] Built pseudo cache for {subject}: "
                                f"shape={cache['data'].shape} duration={cache['duration']:.1f}s "
                                f"because binary FIF cache build failed: {binary_exc}",
                                flush=True,
                            )
                        except Exception as fallback_exc:
                            load_error = (
                                f"{exc}; binary build failed: {binary_exc}; "
                                f"fallback failed: {fallback_exc}"
                            )

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
                        "stream_time_sec": pred.get("stream_time_sec", pred.get("epoch_start_sec")),
                        "epoch_start_sec": pred.get("epoch_start_sec"),
                        "epoch_end_sec": pred.get("epoch_end_sec"),
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


def _serialize_agent_prediction(pred: dict) -> dict:
    """Convert a MongoDB prediction document to a dashboard-safe payload."""
    timestamp = pred.get("timestamp", "")
    if hasattr(timestamp, "isoformat"):
        timestamp = timestamp.isoformat()
    else:
        timestamp = str(timestamp)

    return {
        "epoch_id": str(pred.get("epoch_id", "")),
        "subject_id": str(pred.get("subject_id", "")),
        "session_id": str(pred.get("session_id", "")),
        "label_name": str(pred.get("label_name", "unknown")),
        "label_code": int(pred.get("label_code", -1)),
        "confidence": round(float(pred.get("confidence") or 0.0), 4),
        "model_used": str(pred.get("model_used", "ensemble")),
        "signal_quality": str(pred.get("signal_quality", "unknown")),
        "quality_score": round(float(pred.get("quality_score") or 0.0), 3),
        "calibration_status": str(pred.get("calibration_status", "")),
        "stream_time_sec": pred.get("stream_time_sec", pred.get("epoch_start_sec")),
        "epoch_start_sec": pred.get("epoch_start_sec"),
        "epoch_end_sec": pred.get("epoch_end_sec"),
        "timestamp": timestamp,
    }


@app.get("/stream/agents/snapshot")
async def stream_agents_snapshot(session_id: str | None = None, limit: int = 50) -> dict:
    """
    Return the latest Stream 3 state from MongoDB.

    This is a non-streaming fallback for browsers/dev proxies that keep the SSE
    connection open but miss incremental events.
    """
    try:
        from pymongo import MongoClient
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pymongo unavailable: {exc}") from exc

    mongo_uri = os.getenv(
        "CEREBRO_MONGO_URI",
        "mongodb://cerebro:cerebro123@localhost:27017/?authSource=admin",
    )
    mongo_db = os.getenv("CEREBRO_MONGO_DB", "projectcerebro")
    query = {"session_id": session_id} if session_id else {}
    limit = max(1, min(int(limit), 200))

    try:
        mongo = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        mongo.admin.command("ping")
        db = mongo[mongo_db]
        docs = list(db.predictions.find(query).sort("_id", -1).limit(limit))
        total = db.predictions.count_documents(query)
        alert_total = db.alerts.count_documents(query)
        epoch_ids = [doc.get("epoch_id") for doc in docs if doc.get("epoch_id")]
        alerts_by_epoch: dict[str, list[dict]] = {str(epoch_id): [] for epoch_id in epoch_ids}
        if epoch_ids:
            for alert in db.alerts.find(
                {"epoch_id": {"$in": epoch_ids}},
                {"_id": 0, "epoch_id": 1, "severity": 1, "message": 1, "agent": 1},
            ):
                alerts_by_epoch.setdefault(str(alert.get("epoch_id", "")), []).append(
                    {
                        "severity": alert.get("severity", "info"),
                        "message": alert.get("message", ""),
                        "agent": alert.get("agent", ""),
                    }
                )
        mongo.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ordered_docs = list(reversed(docs))
    # Counts should use the full session, not just the latest limited slice.
    label_counts = {"left": 0, "right": 0, "rest": 0}
    confidence_values = []
    try:
        mongo = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        db = mongo[mongo_db]
        for doc in db.predictions.find(query, {"label_name": 1, "confidence": 1}):
            label = str(doc.get("label_name", "unknown"))
            if label in label_counts:
                label_counts[label] += 1
            confidence_values.append(float(doc.get("confidence") or 0.0))
        mongo.close()
    except Exception:
        for doc in ordered_docs:
            label = str(doc.get("label_name", "unknown"))
            if label in label_counts:
                label_counts[label] += 1
            confidence_values.append(float(doc.get("confidence") or 0.0))

    predictions = []
    for index, pred in enumerate(ordered_docs, start=max(1, total - len(ordered_docs) + 1)):
        payload = _serialize_agent_prediction(pred)
        payload.update(
            {
                "n_predictions": index,
                "alerts": alerts_by_epoch.get(payload["epoch_id"], []),
            }
        )
        predictions.append(payload)

    return {
        "type": "snapshot",
        "database": mongo_db,
        "session_id": session_id,
        "n_predictions": total,
        "n_left": label_counts["left"],
        "n_right": label_counts["right"],
        "n_rest": label_counts["rest"],
        "mean_confidence": round(sum(confidence_values) / len(confidence_values), 4)
        if confidence_values
        else 0.0,
        "n_alerts": alert_total,
        "predictions": predictions,
    }


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
            consumer_running = _worker_running(consumer)
            producer_running = _worker_running(producer)
            if consumer_running or producer_running:
                return {
                    "status": "already_running",
                    "subject": subject,
                    "session_id": session_id,
                    "consumer_pid": _worker_pid(consumer),
                    "producer_pid": _worker_pid(producer),
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
    try:
        consumer = subprocess.Popen(
            consumer_cmd,
            cwd=str(config.PROJECT_ROOT),
            env=env,
            stdout=consumer_log,
            stderr=subprocess.STDOUT,
        )
        await asyncio.sleep(2.0)
        producer_log.close()
        producer_stop = threading.Event()
        producer = threading.Thread(
            target=_dashboard_kafka_producer,
            args=(subject, session_id, request.start_sec, 1.0, producer_stop, producer_log_path),
            daemon=True,
            name=f"dashboard-producer-{safe_session}",
        )
        producer.start()
        consumer_log.close()
    except Exception as exc:
        consumer_log.close()
        producer_log.close()
        raise HTTPException(status_code=500, detail=f"Failed to start prediction run: {exc}") from exc

    with app.state.lock:
        app.state.prediction_runs[session_id] = {
            "subject": subject,
            "consumer": consumer,
            "producer": producer,
            "producer_stop": producer_stop,
            "consumer_log": str(consumer_log_path),
            "producer_log": str(producer_log_path),
            "started_at": time.time(),
        }

    return {
        "status": "started",
        "subject": subject,
        "session_id": session_id,
        "consumer_pid": consumer.pid,
        "producer_pid": None,
        "consumer_log": str(consumer_log_path.relative_to(config.PROJECT_ROOT)),
        "producer_log": str(producer_log_path.relative_to(config.PROJECT_ROOT)),
    }


dashboard_dist = os.path.join(os.path.dirname(__file__), "..", "cerebro-dashboard", "dist")
if os.path.exists(dashboard_dist):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dist, html=True), name="dashboard")
    print("STARTUP: Dashboard served at /dashboard", flush=True)
else:
    print("STARTUP: Dashboard not built yet. Run: cd cerebro-dashboard && npm run build", flush=True)
