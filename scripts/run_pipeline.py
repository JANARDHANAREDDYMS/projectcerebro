"""
ProjectCerebro — Full Pipeline Runner
======================================
Stage 1: ICA cleaning for PhysioNet (EDF) and BCI IV-2a (via MOABB)
Stage 2: Epoch extraction and Parquet output (no Delta Lake/Spark needed)

Usage:
    # Test mode — 1 subject each, sequential:
    python scripts/run_pipeline.py --test

    # Full pipeline (PhysioNet only):
    python scripts/run_pipeline.py --physionet-only

    # Full pipeline with BCI IV-2a:
    python scripts/run_pipeline.py

    # Specific number of PhysioNet subjects:
    python scripts/run_pipeline.py --n-subjects 10
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

PROJECT_ROOT          = Path(__file__).resolve().parent.parent
PHYSIONET_ROOT        = (
    PROJECT_ROOT / "data" / "physionet_mne" / "MNE-eegbci-data"
    / "files" / "eegmmidb" / "1.0.0"
)
BCI_ROOT              = PROJECT_ROOT / "data" / "bci_iv_2a" / "BCICIV_2a_gdf"
CLEANED_ROOT          = PROJECT_ROOT / "data_cleaned"
CLEANED_PHYSIONET     = CLEANED_ROOT / "physionet"
CLEANED_BCI           = CLEANED_ROOT / "bci_iv_2a"
PARQUET_ROOT          = PROJECT_ROOT / "parquet_output"
STAGE1_MANIFEST       = CLEANED_ROOT / "stage1_cleaning_manifest.jsonl"
STAGE2_MANIFEST       = CLEANED_ROOT / "stage2_manifest.jsonl"


# =========================================================
# STAGE 1 CONFIG
# =========================================================

PHYSIONET_RUNS               = [4, 6, 8, 10, 12, 14]
CORRUPTED_PHYSIONET_SUBJECTS = {88, 92, 100, 104, 106}

L_FREQ = 1.0
H_FREQ = 100.0

ICA_N_COMPONENTS = 0.95
ICA_METHOD       = "infomax"
ICA_FIT_PARAMS   = {"extended": True}
ICA_RANDOM_STATE = 97
ICA_MAX_ITER     = 500

REMOVE_NON_BRAIN_LABELS = {
    "muscle artifact", "eye blink", "heart beat",
    "line noise", "channel noise", "other",
}

BCI_RENAME_MAP = {
    "EEG-Fz": "Fz", "EEG-C3": "C3", "EEG-Cz": "Cz", "EEG-C4": "C4",
    "EEG-Pz": "Pz", "EEG-0": "FC3", "EEG-1": "FC1", "EEG-2": "FCz",
    "EEG-3": "FC2", "EEG-4": "FC4", "EEG-5": "C5", "EEG-6": "C1",
    "EEG-7": "C2", "EEG-8": "C6", "EEG-9": "CP3", "EEG-10": "CP1",
    "EEG-11": "CPz", "EEG-12": "CP2", "EEG-13": "CP4", "EEG-14": "P1",
    "EEG-15": "P2", "EEG-16": "POz",
    "EOG-left": "EOG-LEFT", "EOG-central": "EOG-CENTRAL", "EOG-right": "EOG-RIGHT",
}


# =========================================================
# STAGE 2 CONFIG
# =========================================================

PREPROCESSING_VERSION = "v1.1.0"
POOL_SIZE             = 4

FILTER_CONFIGS = {
    "bp8_30": {
        "l_freq":         8.0,
        "h_freq":         30.0,
        "filter_version": "bp_8_30_v1",
        "parquet_name":   "epochs_mi_bp8_30.parquet",
    },
}

TARGET_SFREQ    = 128.0
TMIN            = -1.0
TMAX            =  2.999

LABEL_LEFT  = 0
LABEL_RIGHT = 1
LABEL_REST  = 2

N_CHANNELS  = 5
N_SAMPLES   = 512

COMMON_CHANNELS = ["FZ", "C3", "CZ", "C4", "PZ"]

PHYSIONET_UPPERCASE_MAP = {
    "Fz": "FZ", "Cz": "CZ", "Pz": "PZ", "Oz": "OZ", "Iz": "IZ",
    "Fp1": "FP1", "Fp2": "FP2", "Fpz": "FPZ",
    "FCz": "FCZ", "CPz": "CPZ", "POz": "POZ",
    "AFz": "AFZ", "FTz": "FTZ", "TPz": "TPZ",
}

BCI_EOG_CHANNELS = ["EOG-LEFT", "EOG-CENTRAL", "EOG-RIGHT"]

BCI_LEFT_EVENT     = 769
BCI_RIGHT_EVENT    = 770
BCI_ARTIFACT_EVENT = 1023


# =========================================================
# DATA MODELS
# =========================================================

@dataclass
class CleaningRecord:
    dataset:             str
    subject_id:          str
    run_id:              str
    input_file:          str
    output_file:         str
    sfreq:               float
    n_channels:          int
    ica_n_components:    int
    excluded_components: list
    excluded_labels:     list
    cleaned_at_utc:      str


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
    features:               list
    n_channels:             int
    n_samples:              int
    channel_names:          list
    sampling_rate_hz:       float
    epoch_start_sec:        float
    epoch_end_sec:          float
    filter_version:         str
    preprocessing_version:  str
    ingested_at:            str
    is_rest_synthetic:      bool


# =========================================================
# STAGE 1 HELPERS
# =========================================================

def ensure_dirs() -> None:
    CLEANED_PHYSIONET.mkdir(parents=True, exist_ok=True)
    CLEANED_BCI.mkdir(parents=True, exist_ok=True)
    STAGE1_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    PARQUET_ROOT.mkdir(parents=True, exist_ok=True)


def append_stage1_manifest(record: CleaningRecord) -> None:
    with STAGE1_MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def attach_standard_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw2 = raw.copy()
    montage = mne.channels.make_standard_montage("standard_1005")
    raw2.set_montage(montage, on_missing="warn")
    return raw2


def normalize_physionet_name(ch: str) -> str:
    clean = ch.strip().replace(".", "")
    if len(clean) > 1:
        clean = clean[:1].upper() + clean[1:].lower()
    else:
        clean = clean.upper()

    replacements = {
        "Fpz": "Fpz", "Afz": "AFz", "Fcz": "FCz", "Cpz": "CPz",
        "Poz": "POz", "Cz": "Cz", "Fz": "Fz", "Pz": "Pz", "Oz": "Oz", "Iz": "Iz",
    }
    clean = replacements.get(clean, clean)

    for pfx in [("Fc", "FC"), ("Cp", "CP"), ("Po", "PO"), ("Af", "AF"), ("Ft", "FT"), ("Tp", "TP")]:
        if clean.startswith(pfx[0]) and len(clean) > 2:
            clean = pfx[1] + clean[2:]
    return clean


def standardize_physionet(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    rename_map = {ch: normalize_physionet_name(ch) for ch in raw.ch_names}
    raw2 = raw.copy().rename_channels(rename_map)
    return attach_standard_montage(raw2)


def standardize_bci(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw2 = raw.copy().rename_channels(BCI_RENAME_MAP)
    eog_map = {ch: "eog" for ch in ["EOG-LEFT", "EOG-CENTRAL", "EOG-RIGHT"] if ch in raw2.ch_names}
    if eog_map:
        raw2.set_channel_types(eog_map)
    return attach_standard_montage(raw2)


def prepare_cleaning_branch(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    branch = raw.copy()
    safe_h = min(H_FREQ, raw.info["sfreq"] / 2.0 - 1.0)
    if safe_h <= L_FREQ:
        raise ValueError(f"Invalid filter bounds for sfreq={raw.info['sfreq']}")
    branch.filter(l_freq=L_FREQ, h_freq=safe_h, method="fir", phase="zero", verbose=False)
    branch.set_eeg_reference(ref_channels="average", verbose=False)
    return branch


def fit_ica(raw_branch: mne.io.BaseRaw) -> mne.preprocessing.ICA:
    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        fit_params=ICA_FIT_PARAMS,
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
    )
    ica.fit(raw_branch, picks="eeg", reject_by_annotation=True, verbose=False)
    return ica


def choose_components(raw_branch, ica):
    from mne_icalabel import label_components
    iclabel = label_components(raw_branch, ica, method="iclabel")
    labels  = iclabel["labels"]
    exclude_idx    = [i for i, l in enumerate(labels) if l in REMOVE_NON_BRAIN_LABELS]
    exclude_labels = [labels[i] for i in exclude_idx]
    return exclude_idx, exclude_labels


def clean_one_file(dataset: str, subject_id: str, run_id: str, fpath: Path) -> None:
    print(f"  [{dataset}] {run_id} loading {fpath.name}")

    if dataset == "physionet":
        raw = mne.io.read_raw_edf(str(fpath), preload=True, verbose=False)
        raw = standardize_physionet(raw)
    elif dataset == "bci_iv_2a":
        raw = mne.io.read_raw_gdf(str(fpath), preload=True, verbose=False)
        raw = standardize_bci(raw)
    else:
        raise ValueError(f"Unsupported: {dataset}")

    raw_branch = prepare_cleaning_branch(raw)
    ica        = fit_ica(raw_branch)
    excl_idx, excl_labels = choose_components(raw_branch, ica)
    cleaned = raw.copy()
    ica.apply(cleaned, exclude=excl_idx)

    out_dir = (CLEANED_PHYSIONET / subject_id) if dataset == "physionet" else (CLEANED_BCI / subject_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}_cleaned_raw.fif"
    cleaned.save(str(out_path), overwrite=True, verbose=False)

    n_comp = getattr(ica, "n_components_", len(excl_idx))
    append_stage1_manifest(CleaningRecord(
        dataset=dataset, subject_id=subject_id, run_id=run_id,
        input_file=str(fpath), output_file=str(out_path),
        sfreq=float(cleaned.info["sfreq"]), n_channels=len(cleaned.ch_names),
        ica_n_components=int(n_comp),
        excluded_components=excl_idx, excluded_labels=excl_labels,
        cleaned_at_utc=datetime.now(timezone.utc).isoformat(),
    ))
    print(f"  [{dataset}] {run_id} -> {len(excl_idx)} ICs removed -> {out_path.name}")


def clean_one_file_safe(args):
    dataset, subject_id, run_id, fpath = args
    try:
        clean_one_file(dataset, subject_id, run_id, fpath)
        return True
    except Exception as e:
        print(f"  FAILED [{dataset}] {run_id}: {e}")
        return False


def iter_physionet_jobs(n_subjects: int | None = None):
    subjects = sorted(PHYSIONET_ROOT.glob("S*"))
    if n_subjects:
        subjects = subjects[:n_subjects]
    jobs = []
    for subject_dir in subjects:
        if not subject_dir.is_dir():
            continue
        snum = int(subject_dir.name[1:])
        if snum in CORRUPTED_PHYSIONET_SUBJECTS:
            continue
        for run in PHYSIONET_RUNS:
            fpath = subject_dir / f"{subject_dir.name}R{run:02d}.edf"
            if fpath.exists():
                jobs.append(("physionet", subject_dir.name, f"{subject_dir.name}R{run:02d}", fpath))
    return jobs


def iter_bci_jobs():
    """Yields GDF-based jobs if GDF files exist."""
    if not BCI_ROOT.exists():
        return []
    jobs = []
    for snum in range(1, 10):
        run_id = f"A{snum:02d}T"
        fpath  = BCI_ROOT / f"{run_id}.gdf"
        if fpath.exists():
            jobs.append(("bci_iv_2a", f"A{snum:02d}", run_id, fpath))
    return jobs


def clean_bci_subject_moabb(subject_num: int) -> None:
    """Clean one BCI IV-2a subject using MOABB data (no GDF files needed)."""
    from moabb.datasets import BNCI2014_001
    import warnings
    warnings.filterwarnings("ignore")

    subject_id = f"A{subject_num:02d}"
    ds   = BNCI2014_001()
    data = ds.get_data(subjects=[subject_num])
    sess = data[subject_num]

    for session_key, runs in sess.items():
        if session_key != "0train":
            continue
        for run_key, raw in runs.items():
            run_id = f"{subject_id}T_sess{session_key}_run{run_key}"
            print(f"  [bci_iv_2a MOABB] {subject_id} {session_key}/{run_key}")

            # Drop stim channel before ICA
            raw2 = raw.copy()
            if "STI" in raw2.ch_names:
                raw2.drop_channels(["STI"])

            # Set montage (MOABB usually does this but verify)
            montage = mne.channels.make_standard_montage("standard_1005")
            try:
                raw2.set_montage(montage, on_missing="warn")
            except Exception:
                pass

            raw_branch = prepare_cleaning_branch(raw2)
            ica        = fit_ica(raw_branch)
            excl_idx, excl_labels = choose_components(raw_branch, ica)
            cleaned = raw2.copy()
            ica.apply(cleaned, exclude=excl_idx)

            out_dir = CLEANED_BCI / subject_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{run_id}_cleaned_raw.fif"
            cleaned.save(str(out_path), overwrite=True, verbose=False)

            n_comp = getattr(ica, "n_components_", len(excl_idx))
            append_stage1_manifest(CleaningRecord(
                dataset="bci_iv_2a", subject_id=subject_id, run_id=run_id,
                input_file=f"moabb:{subject_num}/{session_key}/{run_key}",
                output_file=str(out_path),
                sfreq=float(cleaned.info["sfreq"]),
                n_channels=len(cleaned.ch_names),
                ica_n_components=int(n_comp),
                excluded_components=excl_idx, excluded_labels=excl_labels,
                cleaned_at_utc=datetime.now(timezone.utc).isoformat(),
            ))
            print(f"  [bci_iv_2a MOABB] {run_id} -> {len(excl_idx)} ICs removed")


def clean_bci_subject_moabb_safe(subject_num: int) -> bool:
    try:
        clean_bci_subject_moabb(subject_num)
        return True
    except Exception as e:
        print(f"  FAILED BCI subject A{subject_num:02d}: {e}")
        return False


# =========================================================
# STAGE 2 HELPERS
# =========================================================

def standardize_physionet_channels_s2(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
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


def drop_non_eeg(raw: mne.io.BaseRaw, dataset: str) -> mne.io.BaseRaw:
    to_drop = [ch for ch in BCI_EOG_CHANNELS if ch in raw.ch_names] if dataset == "bci_iv_2a" else []
    return raw.copy().drop_channels(to_drop) if to_drop else raw


def select_common_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    ch_upper = {ch.upper(): ch for ch in raw.ch_names}
    missing = [t for t in COMMON_CHANNELS if t not in ch_upper]
    if missing:
        raise ValueError(f"Missing channels: {missing}. Available: {raw.ch_names}")
    available = [ch_upper[t] for t in COMMON_CHANNELS]
    raw = raw.copy().pick_channels(available, ordered=True)
    rename = {ch: ch.upper() for ch in raw.ch_names if ch != ch.upper()}
    return raw.rename_channels(rename) if rename else raw


def align_channels(raw: mne.io.BaseRaw, dataset: str) -> mne.io.BaseRaw:
    if dataset == "physionet":
        raw = standardize_physionet_channels_s2(raw)
    raw = drop_non_eeg(raw, dataset)
    return select_common_channels(raw)


def apply_task_filter(raw: mne.io.BaseRaw, l_freq: float, h_freq: float) -> mne.io.BaseRaw:
    safe_h = min(h_freq, raw.info["sfreq"] / 2.0 - 1.0)
    return raw.copy().filter(l_freq=l_freq, h_freq=safe_h, method="fir", phase="zero", verbose=False)


def resample_epoch(data: np.ndarray, orig_sfreq: float) -> np.ndarray:
    if orig_sfreq == TARGET_SFREQ:
        result = data
    else:
        info = mne.create_info([f"ch{i}" for i in range(data.shape[0])], orig_sfreq, "eeg")
        tmp  = mne.io.RawArray(data, info, verbose=False)
        tmp.resample(TARGET_SFREQ, npad="auto", verbose=False)
        result = tmp.get_data()
    return result[:, :N_SAMPLES] if result.shape[1] > N_SAMPLES else result


def apply_baseline(data: np.ndarray) -> np.ndarray:
    n_base = int(round(abs(TMIN) * TARGET_SFREQ))
    return data - data[:, :n_base].mean(axis=1, keepdims=True)


def build_epoch_id(dataset, subject_id, run_id, label_code, start_sec, end_sec):
    return f"{dataset}|{subject_id}|{run_id}|{label_code}|{int(round(start_sec*1000))}|{int(round(end_sec*1000))}"


def extract_physionet_epochs(raw, filter_cfg, subject_id, run_id, source_file):
    records = []
    sfreq   = raw.info["sfreq"]
    now     = datetime.now(timezone.utc).isoformat()

    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    except Exception:
        return records

    left_code  = event_id.get("T1")
    right_code = event_id.get("T2")
    if not left_code or not right_code:
        return records

    try:
        epo = mne.Epochs(raw, events, event_id={"left": left_code, "right": right_code},
                         tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
                         reject_by_annotation=True, verbose=False)
    except Exception:
        return records

    for i, epoch in enumerate(epo):
        ev_sample  = epo.events[i, 0]
        ev_code    = epo.events[i, 2]
        lcode      = LABEL_LEFT if ev_code == left_code else LABEL_RIGHT
        lname      = "left"     if ev_code == left_code else "right"
        resampled  = resample_epoch(epoch, sfreq)
        if resampled.shape[0] != N_CHANNELS or resampled.shape[1] < N_SAMPLES:
            continue
        corrected  = apply_baseline(resampled)
        start_sec  = (ev_sample / sfreq) + TMIN
        end_sec    = (ev_sample / sfreq) + TMAX
        records.append(EpochRecord(
            epoch_id=build_epoch_id("physionet", subject_id, run_id, lcode, start_sec, end_sec),
            dataset="physionet", subject_id=subject_id, session_id=None, run_id=run_id,
            source_file=source_file, label_code=lcode, label_name=lname,
            features=corrected.flatten().astype(np.float32).tolist(),
            n_channels=N_CHANNELS, n_samples=N_SAMPLES, channel_names=COMMON_CHANNELS,
            sampling_rate_hz=TARGET_SFREQ, epoch_start_sec=float(start_sec), epoch_end_sec=float(end_sec),
            filter_version=filter_cfg["filter_version"], preprocessing_version=PREPROCESSING_VERSION,
            ingested_at=now, is_rest_synthetic=False,
        ))

    # T0 rest epochs
    t0_code = event_id.get("T0")
    if t0_code:
        try:
            epo_r = mne.Epochs(raw, events, event_id={"rest": t0_code},
                               tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
                               reject_by_annotation=True, verbose=False)
            for i, epoch in enumerate(epo_r):
                ev_sample = epo_r.events[i, 0]
                resampled = resample_epoch(epoch, sfreq)
                if resampled.shape[0] != N_CHANNELS or resampled.shape[1] < N_SAMPLES:
                    continue
                corrected = apply_baseline(resampled)
                start_sec = (ev_sample / sfreq) + TMIN
                end_sec   = (ev_sample / sfreq) + TMAX
                records.append(EpochRecord(
                    epoch_id=build_epoch_id("physionet", subject_id, run_id, LABEL_REST, start_sec, end_sec),
                    dataset="physionet", subject_id=subject_id, session_id=None, run_id=run_id,
                    source_file=source_file, label_code=LABEL_REST, label_name="rest",
                    features=corrected.flatten().astype(np.float32).tolist(),
                    n_channels=N_CHANNELS, n_samples=N_SAMPLES, channel_names=COMMON_CHANNELS,
                    sampling_rate_hz=TARGET_SFREQ, epoch_start_sec=float(start_sec), epoch_end_sec=float(end_sec),
                    filter_version=filter_cfg["filter_version"], preprocessing_version=PREPROCESSING_VERSION,
                    ingested_at=now, is_rest_synthetic=False,
                ))
        except Exception:
            pass

    return records


def extract_bci_epochs(raw, filter_cfg, subject_id, run_id, source_file):
    records = []
    sfreq   = raw.info["sfreq"]
    now     = datetime.now(timezone.utc).isoformat()

    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    except Exception:
        return records

    left_code = right_code = None
    for k, v in event_id.items():
        ks = str(k)
        # GDF format uses numeric codes; MOABB uses label strings
        if ks in (str(BCI_LEFT_EVENT), "left_hand", "769"):
            left_code  = v
        if ks in (str(BCI_RIGHT_EVENT), "right_hand", "770"):
            right_code = v

    if not left_code or not right_code:
        print(f"  Warning: left/right not found in {run_id}. Keys: {list(event_id.keys())}")
        return records

    try:
        epo = mne.Epochs(raw, events,
                         event_id={str(BCI_LEFT_EVENT): left_code, str(BCI_RIGHT_EVENT): right_code},
                         tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
                         reject_by_annotation=True, verbose=False)
    except Exception:
        return records

    for i, epoch in enumerate(epo):
        ev_sample = epo.events[i, 0]
        ev_code   = epo.events[i, 2]
        lcode     = LABEL_LEFT if ev_code == left_code else LABEL_RIGHT
        lname     = "left"     if ev_code == left_code else "right"
        resampled = resample_epoch(epoch, sfreq)
        if resampled.shape[0] != N_CHANNELS or resampled.shape[1] < N_SAMPLES:
            continue
        corrected = apply_baseline(resampled)
        start_sec = (ev_sample / sfreq) + TMIN
        end_sec   = (ev_sample / sfreq) + TMAX
        records.append(EpochRecord(
            epoch_id=build_epoch_id("bci_iv_2a", subject_id, run_id, lcode, start_sec, end_sec),
            dataset="bci_iv_2a", subject_id=subject_id, session_id=None, run_id=run_id,
            source_file=source_file, label_code=lcode, label_name=lname,
            features=corrected.flatten().astype(np.float32).tolist(),
            n_channels=N_CHANNELS, n_samples=N_SAMPLES, channel_names=COMMON_CHANNELS,
            sampling_rate_hz=TARGET_SFREQ, epoch_start_sec=float(start_sec), epoch_end_sec=float(end_sec),
            filter_version=filter_cfg["filter_version"], preprocessing_version=PREPROCESSING_VERSION,
            ingested_at=now, is_rest_synthetic=True,
        ))

    return records


def process_one_fif(args):
    dataset, subject_id, run_id, fif_path, filter_key = args
    filter_cfg = FILTER_CONFIGS[filter_key]
    try:
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        raw = align_channels(raw, dataset)
        if len(raw.ch_names) != N_CHANNELS:
            raise ValueError(f"Got {len(raw.ch_names)} channels: {raw.ch_names}")
        raw = apply_task_filter(raw, filter_cfg["l_freq"], filter_cfg["h_freq"])
        if dataset == "physionet":
            records = extract_physionet_epochs(raw, filter_cfg, subject_id, run_id, str(fif_path))
        else:
            records = extract_bci_epochs(raw, filter_cfg, subject_id, run_id, str(fif_path))
        n_l = sum(1 for r in records if r.label_code == LABEL_LEFT)
        n_r = sum(1 for r in records if r.label_code == LABEL_RIGHT)
        n_s = sum(1 for r in records if r.label_code == LABEL_REST)
        print(f"  {subject_id}/{run_id} L={n_l} R={n_r} rest={n_s} total={len(records)}")
        return records
    except Exception as e:
        print(f"  FAILED {subject_id}/{run_id}: {e}")
        return []


# =========================================================
# STAGE 2 — PARQUET WRITER
# =========================================================

def write_parquet(records: list[EpochRecord], filter_key: str) -> Path:
    import pandas as pd

    if not records:
        print("  No records to write.")
        return None

    cfg = FILTER_CONFIGS[filter_key]
    out_path = PARQUET_ROOT / cfg["parquet_name"]

    rows = []
    for r in records:
        d = asdict(r)
        d["channel_names"] = ",".join(d["channel_names"])
        # Store features as numpy array bytes — write flat columns instead
        rows.append(d)

    df = pd.DataFrame(rows)
    # Keep features as list → convert to numpy serializable format
    # Store features as separate columns f_0 .. f_{N_CHANNELS*N_SAMPLES-1}?
    # Better: store as bytes using numpy
    import io

    def features_to_bytes(feat_list):
        arr = np.array(feat_list, dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    df["features_bytes"] = df["features"].apply(features_to_bytes)
    df = df.drop(columns=["features"])

    df.to_parquet(str(out_path), index=False, engine="pyarrow", compression="snappy")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Written {len(records):,} epochs to {out_path} ({size_mb:.1f} MB)")
    return out_path


# =========================================================
# PIPELINE
# =========================================================

def run_stage1(physionet_jobs, bci_jobs, n_workers: int, test_mode: bool) -> None:
    all_jobs = physionet_jobs + bci_jobs
    print(f"\n{'='*60}")
    print(f"Stage 1: ICA Cleaning")
    print(f"PhysioNet: {len(physionet_jobs)}  BCI IV-2a: {len(bci_jobs)}  Total: {len(all_jobs)}")
    print(f"Workers: {n_workers}  Test: {test_mode}")
    print(f"{'='*60}\n")

    job_args = [(d, s, r, p) for d, s, r, p in all_jobs]

    if n_workers == 1 or test_mode:
        for args in job_args:
            clean_one_file_safe(args)
    else:
        with mp.Pool(processes=n_workers, maxtasksperchild=10) as pool:
            pool.map(clean_one_file_safe, job_args)

    print("\nStage 1 complete.")


def run_stage2(filter_key: str, test_mode: bool, n_workers: int) -> Path | None:
    filter_cfg = FILTER_CONFIGS[filter_key]

    physionet_jobs = []
    for subject_dir in sorted(CLEANED_PHYSIONET.glob("S*")):
        for fif in sorted(subject_dir.glob("*_cleaned_raw.fif")):
            run_id = fif.stem.replace("_cleaned_raw", "")
            physionet_jobs.append(("physionet", subject_dir.name, run_id, fif, filter_key))
        if test_mode:
            break

    bci_jobs = []
    for fif in sorted(CLEANED_BCI.glob("**/*_cleaned_raw.fif")):
        run_id     = fif.stem.replace("_cleaned_raw", "")
        subject_id = run_id[:3]
        bci_jobs.append(("bci_iv_2a", subject_id, run_id, fif, filter_key))
        if test_mode:
            break

    all_jobs = physionet_jobs + bci_jobs

    print(f"\n{'='*60}")
    print(f"Stage 2: Epoch Extraction -> Parquet")
    print(f"Filter: {filter_key}  Jobs: {len(all_jobs)}  Workers: {n_workers}")
    print(f"{'='*60}\n")

    if n_workers == 1 or test_mode:
        all_records = []
        for job in all_jobs:
            all_records.extend(process_one_fif(job))
    else:
        with mp.Pool(processes=n_workers, maxtasksperchild=20) as pool:
            results = pool.map(process_one_fif, all_jobs)
        all_records = [r for batch in results for r in batch]

    print(f"\nTotal epochs extracted: {len(all_records):,}")
    expected_len = N_CHANNELS * N_SAMPLES
    all_records  = [r for r in all_records if len(r.features) == expected_len]
    print(f"After validation: {len(all_records):,}")

    if not all_records:
        return None

    return write_parquet(all_records, filter_key)


# =========================================================
# ENTRYPOINT
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ProjectCerebro Full Pipeline")
    parser.add_argument("--test",           action="store_true", help="Test mode: 1 subject each")
    parser.add_argument("--physionet-only", action="store_true", help="Skip BCI IV-2a")
    parser.add_argument("--n-subjects",     type=int, default=None, help="Limit PhysioNet subjects")
    parser.add_argument("--n-workers",      type=int, default=4,    help="Parallel workers (default: 4)")
    parser.add_argument("--skip-stage1",    action="store_true",    help="Skip ICA cleaning (use existing FIFs)")
    parser.add_argument("--filter",         default="bp8_30", choices=list(FILTER_CONFIGS), help="Bandpass filter")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    ensure_dirs()

    n_workers = 1 if args.test else args.n_workers

    if not args.skip_stage1:
        n_subj = 1 if args.test else args.n_subjects
        physionet_jobs = iter_physionet_jobs(n_subj)
        bci_gdf_jobs   = iter_bci_jobs() if not args.physionet_only else []

        if args.test and physionet_jobs:
            physionet_jobs = physionet_jobs[:1]

        if not physionet_jobs and not bci_gdf_jobs and args.physionet_only:
            print(f"ERROR: No input files found.")
            print(f"  PhysioNet root: {PHYSIONET_ROOT}")
            return

        run_stage1(physionet_jobs, bci_gdf_jobs, n_workers, args.test)

        # BCI IV-2a via MOABB (when no GDF files available)
        if not args.physionet_only and not bci_gdf_jobs:
            print(f"\n--- Running BCI IV-2a via MOABB ---")
            bci_subjects = list(range(1, 10))
            if args.test:
                bci_subjects = [1]
            for snum in bci_subjects:
                clean_bci_subject_moabb_safe(snum)

    out_path = run_stage2(args.filter, args.test, n_workers)

    if out_path:
        print(f"\n{'='*60}")
        print(f"Pipeline complete!")
        print(f"Parquet output: {out_path}")
        print(f"Size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
