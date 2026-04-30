"""
ProjectCerebro — Stage 2: Spark Preprocessing Pipeline
=======================================================
Reads cleaned .fif files from Stage 1.
Applies task-specific filtering, epoch extraction,
resampling, baseline correction, and channel alignment.
Writes standardized (5, 512) epochs to Delta Lake.

Parallelism: multiprocessing.Pool(4) for file processing
             Spark used for Delta Lake write only

Changes from v1:
- PhysioNet rest epochs now use explicit T0 events
  instead of mining inter-trial gaps (better quality)
- BCI IV-2a rest still uses inter-trial gap mining
- is_rest_synthetic=False for T0 epochs (explicit rest)

Usage:
    # Test mode (one subject per dataset, sequential):
    python scripts/stage2_spark_preprocess.py --test

    # Full pipeline, primary filter (8-30Hz):
    python scripts/stage2_spark_preprocess.py --filter bp8_30

    # Full pipeline, ablation filter (4-38Hz):
    python scripts/stage2_spark_preprocess.py --filter bp4_38

    # Both filters sequentially:
    python scripts/stage2_spark_preprocess.py --filter both
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("WARNING")


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
CLEANED_ROOT      = PROJECT_ROOT / "data_cleaned"
CLEANED_PHYSIONET = CLEANED_ROOT / "physionet"
CLEANED_BCI       = CLEANED_ROOT / "bci_iv_2a"
DELTA_ROOT        = PROJECT_ROOT / "delta_lake"
MANIFEST_PATH     = PROJECT_ROOT / "data_cleaned" / "stage2_manifest.jsonl"


# =========================================================
# CONFIG
# =========================================================

PREPROCESSING_VERSION = "v1.1.0"  # bumped for T0 fix
POOL_SIZE             = 4

FILTER_CONFIGS = {
    "bp8_30": {
        "l_freq":         8.0,
        "h_freq":         30.0,
        "filter_version": "bp_8_30_v1",
        "delta_path":     "epochs_mi_v1_ch5_sr128_bp8_30",
    },
    "bp4_38": {
        "l_freq":         4.0,
        "h_freq":         38.0,
        "filter_version": "bp_4_38_v1",
        "delta_path":     "epochs_mi_v1_ch5_sr128_bp4_38",
    },
}

TARGET_SFREQ    = 128.0
TMIN            = -1.0
TMAX            =  2.999
REST_BUFFER_SEC =  1.0

LABEL_LEFT  = 0
LABEL_RIGHT = 1
LABEL_REST  = 2

N_CHANNELS  = 5
N_SAMPLES   = 512   # 4s x 128Hz

COMMON_CHANNELS = ["FZ", "C3", "CZ", "C4", "PZ"]

PHYSIONET_UPPERCASE_MAP = {
    "Fz": "FZ", "Cz": "CZ", "Pz": "PZ", "Oz": "OZ", "Iz": "IZ",
    "Fp1": "FP1", "Fp2": "FP2", "Fpz": "FPZ",
    "FCz": "FCZ", "CPz": "CPZ", "POz": "POZ",
    "AFz": "AFZ", "FTz": "FTZ", "TPz": "TPZ",
}

BCI_EOG_CHANNELS      = ["EOG-LEFT", "EOG-CENTRAL", "EOG-RIGHT"]
PRIVATE_DROP_CHANNELS = ["A1", "A2", "EMG1", "EMG2", "ECG1", "ECG2"]

BCI_LEFT_EVENT     = 769
BCI_RIGHT_EVENT    = 770
BCI_ARTIFACT_EVENT = 1023


# =========================================================
# DATA MODELS
# =========================================================

@dataclass
class EpochRecord:
    epoch_id:               str
    dataset:                str
    subject_id:             str
    session_id:             str | None
    run_id:                 str | None
    source_file:            str
    label_code:             int
    label_name:             str
    features:               list[float]
    n_channels:             int
    n_samples:              int
    channel_names:          list[str]
    sampling_rate_hz:       float
    epoch_start_sec:        float
    epoch_end_sec:          float
    filter_version:         str
    preprocessing_version:  str
    ingested_at:            str
    is_rest_synthetic:      bool


@dataclass
class Stage2Record:
    dataset:        str
    subject_id:     str
    run_id:         str
    source_file:    str
    filter_version: str
    n_epochs:       int
    n_left:         int
    n_right:        int
    n_rest:         int
    status:         str
    error:          str | None
    processed_at:   str


# =========================================================
# HELPERS
# =========================================================

def ensure_dirs(filter_key: str) -> None:
    cfg = FILTER_CONFIGS[filter_key]
    (DELTA_ROOT / cfg["delta_path"]).mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_manifest(record: Stage2Record) -> None:
    with MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def build_epoch_id(
    dataset: str,
    subject_id: str,
    run_id: str,
    label_code: int,
    start_sec: float,
    end_sec: float,
) -> str:
    start_ms = int(round(start_sec * 1000))
    end_ms   = int(round(end_sec   * 1000))
    return f"{dataset}|{subject_id}|{run_id}|{label_code}|{start_ms}|{end_ms}"


# =========================================================
# CHANNEL ALIGNMENT
# =========================================================

def standardize_physionet_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    rename = {}
    for ch in raw.ch_names:
        new = PHYSIONET_UPPERCASE_MAP.get(ch, ch)
        for prefix in ["FC", "CP", "PO", "AF", "FT", "TP"]:
            if ch.startswith(prefix[0].upper() + prefix[1].lower()) and len(ch) > 2:
                new = prefix + ch[2:]
                break
        if new != ch:
            rename[ch] = new
    if rename:
        raw = raw.copy().rename_channels(rename)
    return raw


def drop_non_eeg_channels(raw: mne.io.BaseRaw, dataset: str) -> mne.io.BaseRaw:
    to_drop = []
    if dataset == "bci_iv_2a":
        to_drop = [ch for ch in BCI_EOG_CHANNELS if ch in raw.ch_names]
    elif dataset == "private":
        to_drop = [ch for ch in PRIVATE_DROP_CHANNELS if ch in raw.ch_names]
    if to_drop:
        raw = raw.copy().drop_channels(to_drop)
    return raw


def select_common_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    ch_upper = {ch.upper(): ch for ch in raw.ch_names}
    available, missing = [], []
    for target in COMMON_CHANNELS:
        if target in ch_upper:
            available.append(ch_upper[target])
        else:
            missing.append(target)
    if missing:
        raise ValueError(
            f"Missing common channels: {missing}. Available: {raw.ch_names}"
        )
    raw = raw.copy().pick_channels(available, ordered=True)
    rename = {ch: ch.upper() for ch in raw.ch_names if ch != ch.upper()}
    if rename:
        raw = raw.rename_channels(rename)
    return raw


def align_channels(raw: mne.io.BaseRaw, dataset: str) -> mne.io.BaseRaw:
    if dataset == "physionet":
        raw = standardize_physionet_channels(raw)
    raw = drop_non_eeg_channels(raw, dataset)
    raw = select_common_channels(raw)
    return raw


# =========================================================
# TASK FILTER
# =========================================================

def apply_task_filter(
    raw: mne.io.BaseRaw,
    l_freq: float,
    h_freq: float,
) -> mne.io.BaseRaw:
    safe_h = min(h_freq, raw.info["sfreq"] / 2.0 - 1.0)
    return raw.copy().filter(
        l_freq=l_freq,
        h_freq=safe_h,
        method="fir",
        phase="zero",
        verbose=False,
    )


# =========================================================
# RESAMPLE
# =========================================================

def resample_epoch(epoch_data: np.ndarray, orig_sfreq: float) -> np.ndarray:
    if orig_sfreq == TARGET_SFREQ:
        result = epoch_data
    else:
        n_ch = epoch_data.shape[0]
        info = mne.create_info(
            ch_names=[f"ch{i}" for i in range(n_ch)],
            sfreq=orig_sfreq,
            ch_types="eeg",
        )
        tmp = mne.io.RawArray(epoch_data, info, verbose=False)
        tmp.resample(TARGET_SFREQ, npad="auto", verbose=False)
        result = tmp.get_data()

    # Trim to exactly N_SAMPLES (handles MNE off-by-one endpoint inclusion)
    if result.shape[1] > N_SAMPLES:
        result = result[:, :N_SAMPLES]

    return result


# =========================================================
# BASELINE CORRECTION
# =========================================================

def apply_baseline_correction(epoch_data: np.ndarray) -> np.ndarray:
    n_baseline = int(round(abs(TMIN) * TARGET_SFREQ))  # 128 samples
    baseline_mean = epoch_data[:, :n_baseline].mean(axis=1, keepdims=True)
    return epoch_data - baseline_mean


# =========================================================
# REST EPOCH EXTRACTION — PHYSIONET T0 EVENTS
# =========================================================

def extract_physionet_t0_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: dict,
    sfreq: float,
    filter_cfg: dict,
    subject_id: str,
    run_id: str,
    source_file: str,
    now: str,
) -> list[EpochRecord]:
    """
    Extract rest epochs anchored to PhysioNet T0 events.
    T0 = explicit rest/baseline period onset marked by experiment.
    Much better quality than mining inter-trial gaps.
    is_rest_synthetic=False because T0 is an explicit protocol event.

    PhysioNet structure per run:
    T0 = rest cue (15 per run)
    T1 = left fist imagery cue (7-8 per run)
    T2 = right fist imagery cue (7-8 per run)
    """
    records = []
    t0_code = event_id.get("T0")

    if t0_code is None:
        print(f"  Warning: T0 not found in {run_id}, skipping rest extraction")
        return records

    try:
        epochs_mne = mne.Epochs(
            raw,
            events,
            event_id={"rest": t0_code},
            tmin=TMIN,
            tmax=TMAX,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            verbose=False,
        )
    except Exception as e:
        print(f"  Warning: T0 epoch extraction failed {run_id}: {e}")
        return records

    for i, epoch in enumerate(epochs_mne):
        event_sample = epochs_mne.events[i, 0]

        resampled = resample_epoch(epoch, sfreq)
        if resampled.shape[1] > N_SAMPLES:
            resampled = resampled[:, :N_SAMPLES]
        if resampled.shape[0] != N_CHANNELS or resampled.shape[1] < N_SAMPLES:
            continue

        corrected = apply_baseline_correction(resampled)
        start_sec = (event_sample / sfreq) + TMIN
        end_sec   = (event_sample / sfreq) + TMAX

        eid = build_epoch_id(
            "physionet", subject_id, run_id,
            LABEL_REST, start_sec, end_sec
        )

        records.append(EpochRecord(
            epoch_id=eid,
            dataset="physionet",
            subject_id=subject_id,
            session_id=None,
            run_id=run_id,
            source_file=source_file,
            label_code=LABEL_REST,
            label_name="rest",
            features=corrected.flatten().astype(np.float32).tolist(),
            n_channels=N_CHANNELS,
            n_samples=N_SAMPLES,
            channel_names=COMMON_CHANNELS,
            sampling_rate_hz=TARGET_SFREQ,
            epoch_start_sec=float(start_sec),
            epoch_end_sec=float(end_sec),
            filter_version=filter_cfg["filter_version"],
            preprocessing_version=PREPROCESSING_VERSION,
            ingested_at=now,
            is_rest_synthetic=False,  # T0 is explicit protocol rest
        ))

    return records


# =========================================================
# REST EPOCH EXTRACTION — BCI IV-2a GAP MINING
# =========================================================

def extract_rest_epochs(
    raw: mne.io.BaseRaw,
    imagery_event_samples: list[int],
    sfreq: float,
    n_rest_target: int,
    filter_cfg: dict,
    dataset: str,
    subject_id: str,
    run_id: str,
    source_file: str,
    now: str,
) -> list[EpochRecord]:
    """
    Extract synthetic rest epochs from inter-trial gaps.
    Used for BCI IV-2a which has no explicit T0 rest events.
    1.0s buffer on each side of imagery trials.
    Capped at n_rest_target to match imagery epoch count.
    """
    records        = []
    total_samples  = raw.n_times
    epoch_samples  = int(round(4.0 * sfreq))
    buffer_samples = int(round(REST_BUFFER_SEC * sfreq))
    tmin_samples   = int(round(abs(TMIN) * sfreq))
    tmax_samples   = int(round(TMAX * sfreq))

    sorted_events = sorted(imagery_event_samples)
    if not sorted_events:
        return records

    gaps = []

    # Gap before first trial
    first_start = sorted_events[0] + int(round(TMIN * sfreq))
    gap_s = buffer_samples
    gap_e = first_start - buffer_samples
    if gap_e - gap_s >= epoch_samples:
        gaps.append((gap_s, gap_e))

    # Gaps between consecutive trials
    for i in range(len(sorted_events) - 1):
        prev_end   = sorted_events[i]   + tmax_samples
        next_start = sorted_events[i+1] + int(round(TMIN * sfreq))
        gap_s = prev_end   + buffer_samples
        gap_e = next_start - buffer_samples
        if gap_e - gap_s >= epoch_samples:
            gaps.append((gap_s, gap_e))

    # Gap after last trial
    last_end = sorted_events[-1] + tmax_samples
    gap_s = last_end + buffer_samples
    gap_e = total_samples - buffer_samples
    if gap_e - gap_s >= epoch_samples:
        gaps.append((gap_s, gap_e))

    rest_count = 0
    for gap_s, gap_e in gaps:
        if rest_count >= n_rest_target:
            break

        mid     = (gap_s + gap_e) // 2
        e_start = mid - tmin_samples
        e_end   = mid + tmax_samples

        if e_start < 0 or e_end > total_samples:
            continue

        epoch_data = raw.get_data(start=e_start, stop=e_end)
        resampled  = resample_epoch(epoch_data, sfreq)

        if resampled.shape[1] > N_SAMPLES:
            resampled = resampled[:, :N_SAMPLES]
        if resampled.shape[0] != N_CHANNELS or resampled.shape[1] < N_SAMPLES:
            continue

        corrected = apply_baseline_correction(resampled)
        start_sec = e_start / sfreq
        end_sec   = e_end   / sfreq
        eid       = build_epoch_id(dataset, subject_id, run_id,
                                   LABEL_REST, start_sec, end_sec)

        records.append(EpochRecord(
            epoch_id=eid,
            dataset=dataset,
            subject_id=subject_id,
            session_id=None,
            run_id=run_id,
            source_file=source_file,
            label_code=LABEL_REST,
            label_name="rest",
            features=corrected.flatten().astype(np.float32).tolist(),
            n_channels=N_CHANNELS,
            n_samples=N_SAMPLES,
            channel_names=COMMON_CHANNELS,
            sampling_rate_hz=TARGET_SFREQ,
            epoch_start_sec=float(start_sec),
            epoch_end_sec=float(end_sec),
            filter_version=filter_cfg["filter_version"],
            preprocessing_version=PREPROCESSING_VERSION,
            ingested_at=now,
            is_rest_synthetic=True,
        ))
        rest_count += 1

    return records


# =========================================================
# EPOCH EXTRACTION — PHYSIONET
# =========================================================

def extract_physionet_epochs(
    raw: mne.io.BaseRaw,
    filter_cfg: dict,
    subject_id: str,
    run_id: str,
    source_file: str,
) -> list[EpochRecord]:
    records = []
    sfreq   = raw.info["sfreq"]
    now     = datetime.now(timezone.utc).isoformat()

    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    except Exception as e:
        print(f"  Warning: events failed {run_id}: {e}")
        return records

    left_code  = event_id.get("T1")
    right_code = event_id.get("T2")

    if left_code is None or right_code is None:
        print(f"  Warning: T1/T2 not found in {run_id}")
        return records

    imagery_event_id = {"left": left_code, "right": right_code}

    try:
        epochs_mne = mne.Epochs(
            raw, events,
            event_id=imagery_event_id,
            tmin=TMIN, tmax=TMAX,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            verbose=False,
        )
    except Exception as e:
        print(f"  Warning: MNE Epochs failed {run_id}: {e}")
        return records

    for i, epoch in enumerate(epochs_mne):
        event_sample = epochs_mne.events[i, 0]
        event_code   = epochs_mne.events[i, 2]

        lcode = LABEL_LEFT  if event_code == left_code else LABEL_RIGHT
        lname = "left"      if event_code == left_code else "right"

        resampled = resample_epoch(epoch, sfreq)
        if resampled.shape[1] > N_SAMPLES:
            resampled = resampled[:, :N_SAMPLES]
        if resampled.shape[0] != N_CHANNELS or resampled.shape[1] < N_SAMPLES:
            continue

        corrected = apply_baseline_correction(resampled)
        start_sec = (event_sample / sfreq) + TMIN
        end_sec   = (event_sample / sfreq) + TMAX
        eid       = build_epoch_id("physionet", subject_id, run_id,
                                   lcode, start_sec, end_sec)

        records.append(EpochRecord(
            epoch_id=eid,
            dataset="physionet",
            subject_id=subject_id,
            session_id=None,
            run_id=run_id,
            source_file=source_file,
            label_code=lcode,
            label_name=lname,
            features=corrected.flatten().astype(np.float32).tolist(),
            n_channels=N_CHANNELS,
            n_samples=N_SAMPLES,
            channel_names=COMMON_CHANNELS,
            sampling_rate_hz=TARGET_SFREQ,
            epoch_start_sec=float(start_sec),
            epoch_end_sec=float(end_sec),
            filter_version=filter_cfg["filter_version"],
            preprocessing_version=PREPROCESSING_VERSION,
            ingested_at=now,
            is_rest_synthetic=False,
        ))

    # Use T0 events for rest (explicit protocol rest, better than gap mining)
    rest_records = extract_physionet_t0_epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        sfreq=sfreq,
        filter_cfg=filter_cfg,
        subject_id=subject_id,
        run_id=run_id,
        source_file=source_file,
        now=now,
    )
    records.extend(rest_records)
    return records


# =========================================================
# EPOCH EXTRACTION — BCI IV-2a
# =========================================================

def extract_bci_epochs(
    raw: mne.io.BaseRaw,
    filter_cfg: dict,
    subject_id: str,
    run_id: str,
    source_file: str,
) -> list[EpochRecord]:
    records = []
    sfreq   = raw.info["sfreq"]
    now     = datetime.now(timezone.utc).isoformat()

    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    except Exception as e:
        print(f"  Warning: events failed {run_id}: {e}")
        return records

    left_code = right_code = artifact_code = None
    for key, val in event_id.items():
        if str(key) == str(BCI_LEFT_EVENT):
            left_code = val
        elif str(key) == str(BCI_RIGHT_EVENT):
            right_code = val
        elif str(key) == str(BCI_ARTIFACT_EVENT):
            artifact_code = val

    if left_code is None or right_code is None:
        print(f"  Warning: 769/770 not found in {run_id}. Found: {event_id}")
        return records

    artifact_samples = set()
    if artifact_code is not None:
        for ev in events:
            if ev[2] == artifact_code:
                artifact_samples.add(ev[0])

    imagery_event_id = {
        str(BCI_LEFT_EVENT):  left_code,
        str(BCI_RIGHT_EVENT): right_code,
    }

    try:
        epochs_mne = mne.Epochs(
            raw, events,
            event_id=imagery_event_id,
            tmin=TMIN, tmax=TMAX,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            verbose=False,
        )
    except Exception as e:
        print(f"  Warning: MNE Epochs failed {run_id}: {e}")
        return records

    imagery_event_samples = []

    for i, epoch in enumerate(epochs_mne):
        event_sample = epochs_mne.events[i, 0]
        event_code   = epochs_mne.events[i, 2]

        if event_sample in artifact_samples:
            continue

        imagery_event_samples.append(event_sample)

        lcode = LABEL_LEFT  if event_code == left_code else LABEL_RIGHT
        lname = "left"      if event_code == left_code else "right"

        resampled = resample_epoch(epoch, sfreq)
        if resampled.shape[1] > N_SAMPLES:
            resampled = resampled[:, :N_SAMPLES]
        if resampled.shape[0] != N_CHANNELS or resampled.shape[1] < N_SAMPLES:
            continue

        corrected = apply_baseline_correction(resampled)
        start_sec = (event_sample / sfreq) + TMIN
        end_sec   = (event_sample / sfreq) + TMAX
        eid       = build_epoch_id("bci_iv_2a", subject_id, run_id,
                                   lcode, start_sec, end_sec)

        records.append(EpochRecord(
            epoch_id=eid,
            dataset="bci_iv_2a",
            subject_id=subject_id,
            session_id=None,
            run_id=run_id,
            source_file=source_file,
            label_code=lcode,
            label_name=lname,
            features=corrected.flatten().astype(np.float32).tolist(),
            n_channels=N_CHANNELS,
            n_samples=N_SAMPLES,
            channel_names=COMMON_CHANNELS,
            sampling_rate_hz=TARGET_SFREQ,
            epoch_start_sec=float(start_sec),
            epoch_end_sec=float(end_sec),
            filter_version=filter_cfg["filter_version"],
            preprocessing_version=PREPROCESSING_VERSION,
            ingested_at=now,
            is_rest_synthetic=False,
        ))

    # BCI IV-2a has no T0 events so use gap mining
    rest_records = extract_rest_epochs(
        raw=raw,
        imagery_event_samples=imagery_event_samples,
        sfreq=sfreq,
        n_rest_target=len(records),
        filter_cfg=filter_cfg,
        dataset="bci_iv_2a",
        subject_id=subject_id,
        run_id=run_id,
        source_file=source_file,
        now=now,
    )
    records.extend(rest_records)
    return records


# =========================================================
# WORKER FUNCTION — called by Pool
# =========================================================

def process_one_file(args: tuple) -> tuple[list[EpochRecord], Stage2Record]:
    dataset, subject_id, run_id, fif_path, filter_key = args
    filter_cfg = FILTER_CONFIGS[filter_key]

    print(f"  [{mp.current_process().name}] "
          f"[{dataset}] {subject_id}/{run_id}")

    try:
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        raw = align_channels(raw, dataset)

        if len(raw.ch_names) != N_CHANNELS:
            raise ValueError(
                f"Expected {N_CHANNELS} channels after alignment, "
                f"got {len(raw.ch_names)}: {raw.ch_names}"
            )

        raw = apply_task_filter(
            raw,
            l_freq=filter_cfg["l_freq"],
            h_freq=filter_cfg["h_freq"],
        )

        if dataset == "physionet":
            records = extract_physionet_epochs(
                raw=raw,
                filter_cfg=filter_cfg,
                subject_id=subject_id,
                run_id=run_id,
                source_file=str(fif_path),
            )
        elif dataset == "bci_iv_2a":
            records = extract_bci_epochs(
                raw=raw,
                filter_cfg=filter_cfg,
                subject_id=subject_id,
                run_id=run_id,
                source_file=str(fif_path),
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        n_left  = sum(1 for r in records if r.label_code == LABEL_LEFT)
        n_right = sum(1 for r in records if r.label_code == LABEL_RIGHT)
        n_rest  = sum(1 for r in records if r.label_code == LABEL_REST)

        print(f"  [{mp.current_process().name}] "
              f"{subject_id}/{run_id} -> "
              f"L={n_left} R={n_right} rest={n_rest} total={len(records)}")

        manifest = Stage2Record(
            dataset=dataset,
            subject_id=subject_id,
            run_id=run_id,
            source_file=str(fif_path),
            filter_version=filter_cfg["filter_version"],
            n_epochs=len(records),
            n_left=n_left,
            n_right=n_right,
            n_rest=n_rest,
            status="success",
            error=None,
            processed_at=datetime.now(timezone.utc).isoformat(),
        )
        return records, manifest

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"  [{mp.current_process().name}] "
              f"FAILED {subject_id}/{run_id}: {e}")

        manifest = Stage2Record(
            dataset=dataset,
            subject_id=subject_id,
            run_id=run_id,
            source_file=str(fif_path),
            filter_version=filter_cfg["filter_version"],
            n_epochs=0,
            n_left=0,
            n_right=0,
            n_rest=0,
            status="failed",
            error=err_msg,
            processed_at=datetime.now(timezone.utc).isoformat(),
        )
        return [], manifest


# =========================================================
# FILE ITERATORS
# =========================================================

def iter_physionet_jobs(filter_key: str, test_mode: bool) -> list[tuple]:
    jobs = []
    for subject_dir in sorted(CLEANED_PHYSIONET.glob("S*")):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name
        for fif in sorted(subject_dir.glob("*_cleaned_raw.fif")):
            run_id = fif.stem.replace("_cleaned_raw", "")
            jobs.append(("physionet", subject_id, run_id, fif, filter_key))
        if test_mode:
            break
    return jobs


def iter_bci_jobs(filter_key: str, test_mode: bool) -> list[tuple]:
    jobs = []
    for fif in sorted(CLEANED_BCI.glob("**/*_cleaned_raw.fif")):
        run_id     = fif.stem.replace("_cleaned_raw", "")
        subject_id = run_id[:3]
        jobs.append(("bci_iv_2a", subject_id, run_id, fif, filter_key))
        if test_mode:
            break
    return jobs


# =========================================================
# DELTA LAKE WRITER
# =========================================================

def write_to_delta_lake(
    records: list[EpochRecord],
    filter_key: str,
    spark,
) -> None:
    from pyspark.sql.types import (
        StructType, StructField,
        StringType, IntegerType, FloatType,
        DoubleType, ArrayType, BooleanType,
    )

    if not records:
        print("  No records to write.")
        return

    cfg        = FILTER_CONFIGS[filter_key]
    delta_path = str(DELTA_ROOT / cfg["delta_path"])

    schema = StructType([
        StructField("epoch_id",              StringType(),                               False),
        StructField("dataset",               StringType(),                               False),
        StructField("subject_id",            StringType(),                               False),
        StructField("session_id",            StringType(),                               True),
        StructField("run_id",                StringType(),                               True),
        StructField("source_file",           StringType(),                               True),
        StructField("label_code",            IntegerType(),                              False),
        StructField("label_name",            StringType(),                               False),
        StructField("features",              ArrayType(FloatType(), containsNull=False), False),
        StructField("n_channels",            IntegerType(),                              False),
        StructField("n_samples",             IntegerType(),                              False),
        StructField("channel_names",         ArrayType(StringType(), containsNull=False),False),
        StructField("sampling_rate_hz",      FloatType(),                                False),
        StructField("epoch_start_sec",       DoubleType(),                               True),
        StructField("epoch_end_sec",         DoubleType(),                               True),
        StructField("filter_version",        StringType(),                               False),
        StructField("preprocessing_version", StringType(),                               False),
        StructField("ingested_at",           StringType(),                               False),
        StructField("is_rest_synthetic",     BooleanType(),                              True),
    ])

    rows = [asdict(r) for r in records]
    df   = spark.createDataFrame(rows, schema=schema)

    print(f"\n  Writing {len(records):,} epochs to Delta Lake")
    print(f"  Path: {delta_path}")

    (df
     .repartition(8)
     .write
     .format("delta")
     .mode("overwrite")
     .save(delta_path))

    print(f"  Delta Lake write complete.")


# =========================================================
# SPARK SESSION
# =========================================================

def create_spark_session():
    from pyspark.sql import SparkSession
    from delta import configure_spark_with_delta_pip

    builder = (
        SparkSession.builder
        .appName("ProjectCerebro-Stage2")
        .master("local[*]")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.maxResultSize", "2g")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_pipeline(filter_key: str, test_mode: bool) -> None:
    cfg = FILTER_CONFIGS[filter_key]

    print(f"\n{'='*60}")
    print(f"ProjectCerebro — Stage 2 Spark Preprocessing")
    print(f"Filter:    {filter_key} "
          f"({cfg['l_freq']}-{cfg['h_freq']} Hz)")
    print(f"Pool size: {1 if test_mode else POOL_SIZE} workers")
    print(f"Test mode: {test_mode}")
    print(f"{'='*60}\n")

    ensure_dirs(filter_key)

    physionet_jobs = iter_physionet_jobs(filter_key, test_mode)
    bci_jobs       = iter_bci_jobs(filter_key, test_mode)
    all_jobs       = physionet_jobs + bci_jobs

    print(f"PhysioNet jobs: {len(physionet_jobs)}")
    print(f"BCI IV-2a jobs: {len(bci_jobs)}")
    print(f"Total jobs:     {len(all_jobs)}\n")

    all_records  = []
    total_left   = 0
    total_right  = 0
    total_rest   = 0
    failed       = 0

    if test_mode:
        print("Running sequentially (test mode)\n")
        for job in all_jobs:
            records, manifest = process_one_file(job)
            all_records.extend(records)
            append_manifest(manifest)
            if manifest.status == "failed":
                failed += 1
            else:
                total_left  += manifest.n_left
                total_right += manifest.n_right
                total_rest  += manifest.n_rest
    else:
        print(f"Running with Pool({POOL_SIZE}) workers\n")
        with mp.Pool(processes=POOL_SIZE, maxtasksperchild=20) as pool:
            results = pool.map(process_one_file, all_jobs)

        for records, manifest in results:
            all_records.extend(records)
            append_manifest(manifest)
            if manifest.status == "failed":
                failed += 1
            else:
                total_left  += manifest.n_left
                total_right += manifest.n_right
                total_rest  += manifest.n_rest

    print(f"\n{'='*60}")
    print(f"Preprocessing complete")
    print(f"Total epochs: {len(all_records):,}")
    print(f"  Left:        {total_left:,}")
    print(f"  Right:       {total_right:,}")
    print(f"  Rest:        {total_rest:,}")
    print(f"  Failed jobs: {failed}")
    print(f"{'='*60}\n")

    if not all_records:
        print("No records to write. Check errors above.")
        return

    expected_len = N_CHANNELS * N_SAMPLES
    bad = [r for r in all_records if len(r.features) != expected_len]
    if bad:
        print(f"WARNING: {len(bad)} epochs have wrong feature length, removing.")
        all_records = [r for r in all_records if len(r.features) == expected_len]
        print(f"Remaining: {len(all_records):,}")

    print("Starting Spark session...")
    spark = create_spark_session()
    print("Spark ready\n")

    write_to_delta_lake(all_records, filter_key, spark)
    spark.stop()

    print(f"\nOutput: {DELTA_ROOT / cfg['delta_path']}")
    print("Stage 2 complete.")


# =========================================================
# ENTRYPOINT
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProjectCerebro Stage 2 Spark Preprocessing"
    )
    parser.add_argument(
        "--filter",
        choices=["bp8_30", "bp4_38", "both"],
        default="bp8_30",
        help="Filter version (default: bp8_30)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: one subject per dataset, sequential",
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    if args.filter == "both":
        for fk in ["bp8_30", "bp4_38"]:
            run_pipeline(filter_key=fk, test_mode=args.test)
    else:
        run_pipeline(filter_key=args.filter, test_mode=args.test)


if __name__ == "__main__":
    main()