"""
ProjectCerebro — Stage 2 (Parquet export, no Spark): Cho 2017
=============================================================
Reads cleaned .fif files from Stage 1 (cho2017).
Applies task-specific filtering, epoch extraction,
resampling, baseline correction, and channel alignment.
Writes standardized (5, 512) epochs to Parquet files.

Same preprocessing logic as stage2_spark_preprocess.py
but outputs to Parquet directly via pyarrow (no Spark needed).

Usage:
    python scripts/stage2_parquet_cho2017.py
    python scripts/stage2_parquet_cho2017.py --test
    python scripts/stage2_parquet_cho2017.py --filter bp4_38
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

mne.set_log_level("WARNING")


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
CLEANED_ROOT    = PROJECT_ROOT / "data_cleaned"
CLEANED_CHO2017 = CLEANED_ROOT / "cho2017"
PARQUET_ROOT    = PROJECT_ROOT / "parquet_export"
MANIFEST_PATH   = PROJECT_ROOT / "data_cleaned" / "stage2_cho2017_manifest.jsonl"


# =========================================================
# CONFIG
# =========================================================

PREPROCESSING_VERSION = "v1.1.0"

FILTER_CONFIGS = {
    "bp8_30": {
        "l_freq":         8.0,
        "h_freq":         30.0,
        "filter_version": "bp_8_30_v1",
        "parquet_path":   "cho2017_epochs_ch5_sr128_bp8_30",
    },
    "bp4_38": {
        "l_freq":         4.0,
        "h_freq":         38.0,
        "filter_version": "bp_4_38_v1",
        "parquet_path":   "cho2017_epochs_ch5_sr128_bp4_38",
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


# =========================================================
# DATA MODEL
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


# =========================================================
# HELPERS
# =========================================================

def build_epoch_id(
    dataset: str, subject_id: str, run_id: str,
    label_code: int, start_sec: float, end_sec: float,
) -> str:
    start_ms = int(round(start_sec * 1000))
    end_ms   = int(round(end_sec   * 1000))
    return f"{dataset}|{subject_id}|{run_id}|{label_code}|{start_ms}|{end_ms}"


# =========================================================
# CHANNEL ALIGNMENT
# =========================================================

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


# =========================================================
# TASK FILTER
# =========================================================

def apply_task_filter(
    raw: mne.io.BaseRaw, l_freq: float, h_freq: float,
) -> mne.io.BaseRaw:
    safe_h = min(h_freq, raw.info["sfreq"] / 2.0 - 1.0)
    return raw.copy().filter(
        l_freq=l_freq, h_freq=safe_h,
        method="fir", phase="zero", verbose=False,
    )


# =========================================================
# RESAMPLE / BASELINE
# =========================================================

def resample_epoch(epoch_data: np.ndarray, orig_sfreq: float) -> np.ndarray:
    if orig_sfreq == TARGET_SFREQ:
        result = epoch_data
    else:
        n_ch = epoch_data.shape[0]
        info = mne.create_info(
            ch_names=[f"ch{i}" for i in range(n_ch)],
            sfreq=orig_sfreq, ch_types="eeg",
        )
        tmp = mne.io.RawArray(epoch_data, info, verbose=False)
        tmp.resample(TARGET_SFREQ, npad="auto", verbose=False)
        result = tmp.get_data()
    if result.shape[1] > N_SAMPLES:
        result = result[:, :N_SAMPLES]
    return result


def apply_baseline_correction(epoch_data: np.ndarray) -> np.ndarray:
    n_baseline = int(round(abs(TMIN) * TARGET_SFREQ))
    baseline_mean = epoch_data[:, :n_baseline].mean(axis=1, keepdims=True)
    return epoch_data - baseline_mean


# =========================================================
# REST MINING
# =========================================================

def extract_rest_epochs(
    raw, imagery_event_samples, sfreq, n_rest_target,
    filter_cfg, dataset, subject_id, run_id, source_file, now,
) -> list[EpochRecord]:
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
    first_start = sorted_events[0] + int(round(TMIN * sfreq))
    gap_s = buffer_samples
    gap_e = first_start - buffer_samples
    if gap_e - gap_s >= epoch_samples:
        gaps.append((gap_s, gap_e))

    for i in range(len(sorted_events) - 1):
        prev_end   = sorted_events[i]   + tmax_samples
        next_start = sorted_events[i+1] + int(round(TMIN * sfreq))
        gap_s = prev_end   + buffer_samples
        gap_e = next_start - buffer_samples
        if gap_e - gap_s >= epoch_samples:
            gaps.append((gap_s, gap_e))

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
        eid = build_epoch_id(dataset, subject_id, run_id, LABEL_REST, start_sec, end_sec)

        records.append(EpochRecord(
            epoch_id=eid, dataset=dataset,
            subject_id=subject_id, session_id=None,
            run_id=run_id, source_file=source_file,
            label_code=LABEL_REST, label_name="rest",
            features=corrected.flatten().astype(np.float32).tolist(),
            n_channels=N_CHANNELS, n_samples=N_SAMPLES,
            channel_names=COMMON_CHANNELS, sampling_rate_hz=TARGET_SFREQ,
            epoch_start_sec=float(start_sec), epoch_end_sec=float(end_sec),
            filter_version=filter_cfg["filter_version"],
            preprocessing_version=PREPROCESSING_VERSION,
            ingested_at=now, is_rest_synthetic=True,
        ))
        rest_count += 1

    return records


# =========================================================
# EPOCH EXTRACTION — CHO 2017
# =========================================================

def extract_cho2017_epochs(
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

    left_code  = event_id.get("left")
    right_code = event_id.get("right")

    if left_code is None or right_code is None:
        print(f"  Warning: left/right not found in {run_id}. Found: {event_id}")
        return records

    imagery_event_id = {"left": left_code, "right": right_code}

    try:
        epochs_mne = mne.Epochs(
            raw, events, event_id=imagery_event_id,
            tmin=TMIN, tmax=TMAX, baseline=None,
            preload=True, reject_by_annotation=True, verbose=False,
        )
    except Exception as e:
        print(f"  Warning: Epochs failed {run_id}: {e}")
        return records

    imagery_event_samples = []

    for i, epoch in enumerate(epochs_mne):
        event_sample = epochs_mne.events[i, 0]
        event_code   = epochs_mne.events[i, 2]
        imagery_event_samples.append(event_sample)

        lcode = LABEL_LEFT if event_code == left_code else LABEL_RIGHT
        lname = "left"     if event_code == left_code else "right"

        resampled = resample_epoch(epoch, sfreq)
        if resampled.shape[1] > N_SAMPLES:
            resampled = resampled[:, :N_SAMPLES]
        if resampled.shape[0] != N_CHANNELS or resampled.shape[1] < N_SAMPLES:
            continue

        corrected = apply_baseline_correction(resampled)
        start_sec = (event_sample / sfreq) + TMIN
        end_sec   = (event_sample / sfreq) + TMAX
        eid = build_epoch_id("cho2017", subject_id, run_id, lcode, start_sec, end_sec)

        records.append(EpochRecord(
            epoch_id=eid, dataset="cho2017",
            subject_id=subject_id, session_id=None,
            run_id=run_id, source_file=source_file,
            label_code=lcode, label_name=lname,
            features=corrected.flatten().astype(np.float32).tolist(),
            n_channels=N_CHANNELS, n_samples=N_SAMPLES,
            channel_names=COMMON_CHANNELS, sampling_rate_hz=TARGET_SFREQ,
            epoch_start_sec=float(start_sec), epoch_end_sec=float(end_sec),
            filter_version=filter_cfg["filter_version"],
            preprocessing_version=PREPROCESSING_VERSION,
            ingested_at=now, is_rest_synthetic=False,
        ))

    # Mine rest from inter-trial gaps
    rest_records = extract_rest_epochs(
        raw=raw, imagery_event_samples=imagery_event_samples,
        sfreq=sfreq, n_rest_target=len(records) // 2,
        filter_cfg=filter_cfg, dataset="cho2017",
        subject_id=subject_id, run_id=run_id,
        source_file=source_file, now=now,
    )
    records.extend(rest_records)
    return records


# =========================================================
# PROCESS ONE FILE
# =========================================================

def process_one_file(
    fif_path: Path, subject_id: str, run_id: str, filter_key: str,
) -> tuple[list[EpochRecord], dict]:
    filter_cfg = FILTER_CONFIGS[filter_key]
    print(f"  [cho2017] {subject_id}/{run_id}")

    try:
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        raw = select_common_channels(raw)

        if len(raw.ch_names) != N_CHANNELS:
            raise ValueError(
                f"Expected {N_CHANNELS} channels, got {len(raw.ch_names)}: {raw.ch_names}"
            )

        raw = apply_task_filter(raw, filter_cfg["l_freq"], filter_cfg["h_freq"])
        records = extract_cho2017_epochs(
            raw=raw, filter_cfg=filter_cfg,
            subject_id=subject_id, run_id=run_id, source_file=str(fif_path),
        )

        n_left  = sum(1 for r in records if r.label_code == LABEL_LEFT)
        n_right = sum(1 for r in records if r.label_code == LABEL_RIGHT)
        n_rest  = sum(1 for r in records if r.label_code == LABEL_REST)

        print(f"  -> L={n_left} R={n_right} rest={n_rest} total={len(records)}")

        manifest = {
            "dataset": "cho2017", "subject_id": subject_id,
            "run_id": run_id, "source_file": str(fif_path),
            "filter_version": filter_cfg["filter_version"],
            "n_epochs": len(records), "n_left": n_left,
            "n_right": n_right, "n_rest": n_rest,
            "status": "success", "error": None,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        return records, manifest

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"  FAILED {subject_id}/{run_id}: {e}")
        manifest = {
            "dataset": "cho2017", "subject_id": subject_id,
            "run_id": run_id, "source_file": str(fif_path),
            "filter_version": filter_cfg["filter_version"],
            "n_epochs": 0, "n_left": 0, "n_right": 0, "n_rest": 0,
            "status": "failed", "error": err_msg,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        return [], manifest


# =========================================================
# PYARROW SCHEMA
# =========================================================

PARQUET_SCHEMA = pa.schema([
    pa.field("epoch_id",              pa.string(),              False),
    pa.field("dataset",               pa.string(),              False),
    pa.field("subject_id",            pa.string(),              False),
    pa.field("session_id",            pa.string(),              True),
    pa.field("run_id",                pa.string(),              True),
    pa.field("source_file",           pa.string(),              True),
    pa.field("label_code",            pa.int32(),               False),
    pa.field("label_name",            pa.string(),              False),
    pa.field("features",              pa.list_(pa.float32()),   False),
    pa.field("n_channels",            pa.int32(),               False),
    pa.field("n_samples",             pa.int32(),               False),
    pa.field("channel_names",         pa.list_(pa.string()),    False),
    pa.field("sampling_rate_hz",      pa.float32(),             False),
    pa.field("epoch_start_sec",       pa.float64(),             True),
    pa.field("epoch_end_sec",         pa.float64(),             True),
    pa.field("filter_version",        pa.string(),              False),
    pa.field("preprocessing_version", pa.string(),              False),
    pa.field("ingested_at",           pa.string(),              False),
    pa.field("is_rest_synthetic",     pa.bool_(),               True),
])


def write_parquet(records: list[EpochRecord], out_path: Path) -> None:
    if not records:
        print("  No records to write.")
        return

    rows = [asdict(r) for r in records]
    df   = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)

    out_path.mkdir(parents=True, exist_ok=True)
    pq.write_to_dataset(
        table,
        root_path=str(out_path),
        partition_cols=["subject_id"],
        existing_data_behavior="overwrite_or_ignore",
    )
    print(f"  Parquet written to: {out_path}")
    print(f"  Total rows: {len(records):,}")


# =========================================================
# MAIN
# =========================================================

def run_pipeline(filter_key: str, test_mode: bool) -> None:
    cfg = FILTER_CONFIGS[filter_key]
    out_path = PARQUET_ROOT / cfg["parquet_path"]

    print(f"\n{'='*60}")
    print(f"ProjectCerebro — Stage 2 Parquet (Cho 2017)")
    print(f"Filter:    {filter_key} ({cfg['l_freq']}-{cfg['h_freq']} Hz)")
    print(f"Test mode: {test_mode}")
    print(f"{'='*60}\n")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(CLEANED_CHO2017.glob("s*"))
    if test_mode:
        subject_dirs = subject_dirs[:1]
        print("Test mode: first subject only")

    if not subject_dirs:
        print(f"No subjects found in {CLEANED_CHO2017}")
        return

    print(f"Subjects to process: {len(subject_dirs)}\n")

    all_records = []
    total_left = total_right = total_rest = failed = 0

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        fif_files  = sorted(subject_dir.glob("*_cleaned_raw.fif"))
        for fif in fif_files:
            run_id  = fif.stem.replace("_cleaned_raw", "")
            records, manifest = process_one_file(fif, subject_id, run_id, filter_key)
            all_records.extend(records)

            with MANIFEST_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(manifest) + "\n")

            if manifest["status"] == "failed":
                failed += 1
            else:
                total_left  += manifest["n_left"]
                total_right += manifest["n_right"]
                total_rest  += manifest["n_rest"]

    print(f"\n{'='*60}")
    print(f"Preprocessing complete")
    print(f"Total epochs: {len(all_records):,}")
    print(f"  Left:  {total_left:,}")
    print(f"  Right: {total_right:,}")
    print(f"  Rest:  {total_rest:,}")
    print(f"  Failed jobs: {failed}")
    print(f"{'='*60}\n")

    if not all_records:
        print("No records to write.")
        return

    expected_len = N_CHANNELS * N_SAMPLES
    bad = [r for r in all_records if len(r.features) != expected_len]
    if bad:
        print(f"WARNING: {len(bad)} bad-length epochs removed.")
        all_records = [r for r in all_records if len(r.features) == expected_len]

    print("Writing Parquet...")
    write_parquet(all_records, out_path)
    print(f"\nOutput: {out_path}")
    print("Stage 2 complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ProjectCerebro Stage 2 Parquet — Cho 2017"
    )
    parser.add_argument(
        "--filter", choices=["bp8_30", "bp4_38", "both"], default="bp8_30",
    )
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.filter == "both":
        for fk in ["bp8_30", "bp4_38"]:
            run_pipeline(filter_key=fk, test_mode=args.test)
    else:
        run_pipeline(filter_key=args.filter, test_mode=args.test)


if __name__ == "__main__":
    main()
