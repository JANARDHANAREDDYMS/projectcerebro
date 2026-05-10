"""
ProjectCerebro — Stage 1: Cho 2017 Dataset Ingestion
=====================================================
Reads .mat files from Cho 2017 dataset.
Applies ICA cleaning same as PhysioNet pipeline.
Saves cleaned .fif files to data_cleaned/cho2017/

Cho 2017 dataset:
  52 subjects (s1.mat - s52.mat)
  64 EEG channels
  512Hz sampling rate
  240 trials per subject (120 left, 120 right)
  Left hand = label 1
  Right hand = label 2

Usage:
  python scripts/stage1_cho2017_ingest.py
  python scripts/stage1_cho2017_ingest.py --test
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import scipy.io as sio

mne.set_log_level("WARNING")

# ── Paths ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "cho2017"
CLEAN_DIR    = PROJECT_ROOT / "data_cleaned" / "cho2017"
MANIFEST     = PROJECT_ROOT / "data_cleaned" / \
               "cho2017_stage1_manifest.jsonl"

# ── Config ─────────────────────────────────────────
SFREQ_TARGET  = 512.0   # Cho 2017 native rate
ICA_METHOD    = "infomax"
ICA_EXTENDED  = True
ICA_VARIANCE  = 0.95

COMMON_CHANNELS = ["FZ", "C3", "CZ", "C4", "PZ"]

# Cho 2017 actual channel names (64 EEG + 4 EMG) — from MOABB gigadb source
CHO2017_EEG_CHANNELS = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
    "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
    "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
    "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]
CHO2017_EMG_CHANNELS = ["EMG1", "EMG2", "EMG3", "EMG4"]
CHO2017_ALL_CHANNELS = CHO2017_EEG_CHANNELS + CHO2017_EMG_CHANNELS  # 68 total

# Cho 2017 has explicit rest recordings
# in separate variable - use if available
# Otherwise use inter-trial gaps


def load_cho_subject(mat_path: Path) -> dict:
    """Load one Cho 2017 .mat file.

    Actual mat structure:
      eeg.imagery_left   (68, n_trials * epoch_samples) — continuous left trials
      eeg.imagery_right  (68, n_trials * epoch_samples) — continuous right trials
      eeg.srate          512
      eeg.n_imagery_trials  100
      eeg.frame          [-2000, 5000] ms (tmin/tmax relative to onset)
    """
    mat = sio.loadmat(
        str(mat_path),
        struct_as_record=False,
        squeeze_me=True
    )
    eeg = mat['eeg']

    sfreq     = float(eeg.srate)
    n_trials  = int(eeg.n_imagery_trials)
    frame_ms  = eeg.frame  # e.g. [-2000, 5000]

    epoch_len_ms      = int(frame_ms[1]) - int(frame_ms[0])
    epoch_len_samples = int(epoch_len_ms / 1000.0 * sfreq)

    n_channels = eeg.imagery_left.shape[0]  # 68
    chan_names  = CHO2017_ALL_CHANNELS[:n_channels]

    # Subtract per-channel mean (DC removal, as done in MOABB)
    img_left  = eeg.imagery_left  - eeg.imagery_left.mean(axis=1, keepdims=True)
    img_right = eeg.imagery_right - eeg.imagery_right.mean(axis=1, keepdims=True)

    # Convert µV → V (MNE expects Volts)
    img_left  = img_left  * 1e-6
    img_right = img_right * 1e-6

    # Reshape continuous blocks → (trials, channels, samples)
    x_left = (img_left
               .reshape(n_channels, n_trials, epoch_len_samples)
               .transpose(1, 0, 2))   # (n_trials, 68, epoch_len_samples)
    x_right = (img_right
                .reshape(n_channels, n_trials, epoch_len_samples)
                .transpose(1, 0, 2))

    # Labels: 1 = left, 2 = right
    y_left  = np.ones(n_trials, dtype=int)
    y_right = np.full(n_trials, 2, dtype=int)

    x_all = np.concatenate([x_left, x_right], axis=0)   # (2*n_trials, 68, epoch_len_samples)
    y_all = np.concatenate([y_left,  y_right], axis=0)

    return {
        "x": x_all,        # (n_trials, n_channels, n_samples)
        "y": y_all,        # (n_trials,)
        "sfreq": sfreq,
        "chan_names": chan_names,
    }


def trials_to_raw(
    x: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    chan_names: list[str],
) -> mne.io.RawArray:
    """
    Convert trial array to continuous MNE Raw object.
    Adds annotations for left (1) and right (2) trials.
    Inserts 2s gap between trials for rest mining.
    """
    n_trials, n_channels, n_samples = x.shape
    gap_samples = int(sfreq * 2.0)  # 2s gap between trials
    gap_data    = np.zeros((n_channels, gap_samples))

    # Build continuous signal
    segments = []
    onsets   = []
    labels   = []
    current_sample = 0

    for i in range(n_trials):
        # Add gap before each trial
        segments.append(gap_data)
        current_sample += gap_samples

        # Record onset of this trial
        onsets.append(current_sample / sfreq)
        labels.append(int(y[i]))

        # Add trial data
        segments.append(x[i])
        current_sample += n_samples

    # Final gap
    segments.append(gap_data)

    # Concatenate
    continuous = np.concatenate(segments, axis=1)

    # Determine channel types: EEG for first 64, EMG for remaining
    n_eeg = min(len(chan_names), len(CHO2017_EEG_CHANNELS))
    ch_types = ['eeg'] * n_eeg + ['emg'] * (len(chan_names) - n_eeg)

    # Create MNE info
    info = mne.create_info(
        ch_names=chan_names,
        sfreq=sfreq,
        ch_types=ch_types,
    )

    raw = mne.io.RawArray(continuous, info, verbose=False)

    # Set standard montage so ICLabel can use electrode positions
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing='ignore', verbose=False)

    # Average reference for proper ICA decomposition
    raw.set_eeg_reference('average', projection=True, verbose=False)

    # Add annotations
    # Label 1 = left hand, Label 2 = right hand
    descriptions = []
    for label in labels:
        if label == 1:
            descriptions.append('left')
        elif label == 2:
            descriptions.append('right')
        else:
            descriptions.append('unknown')

    trial_duration = n_samples / sfreq
    annotations = mne.Annotations(
        onset=onsets,
        duration=[trial_duration] * len(onsets),
        description=descriptions,
    )
    raw.set_annotations(annotations)

    return raw


def apply_ica_cleaning(raw: mne.io.RawArray) -> mne.io.RawArray:
    """
    Apply ICA artifact removal.
    Same pipeline as PhysioNet Stage 1.
    Uses ICLabel for artifact classification.
    """
    from mne_icalabel import label_components

    # Bandpass for ICA (broadband) — apply average ref projection first
    raw_filt = raw.copy()
    raw_filt.apply_proj(verbose=False)
    raw_filt = raw_filt.filter(
        l_freq=1.0, h_freq=100.0,
        method='fir', phase='zero',
        verbose=False,
    )

    # Fit ICA — use fixed 20 components for 64-ch EEG stability
    n_components = 20
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=ICA_METHOD,
        fit_params={"extended": ICA_EXTENDED},
        random_state=42,
        verbose=False,
    )
    ica.fit(raw_filt, picks='eeg', verbose=False)

    # Label components with ICLabel
    ic_labels = label_components(raw_filt, ica,
                                  method='iclabel')
    labels     = ic_labels['labels']
    confidence = ic_labels['y_pred_proba']

    # Exclude artifact components
    artifact_types = [
        'eye blink', 'muscle artifact',
        'line noise', 'channel noise', 'other'
    ]
    exclude_idx = []
    excluded_labels = []

    for idx, (label, conf) in enumerate(
        zip(labels, confidence)
    ):
        max_conf = conf.max()
        if label in artifact_types and max_conf > 0.8:
            exclude_idx.append(idx)
            excluded_labels.append(label)

    # Apply ICA to original (unfiltered) raw
    ica.exclude = exclude_idx
    raw_clean   = ica.apply(raw.copy(), verbose=False)

    return raw_clean, len(exclude_idx), excluded_labels


def process_subject(
    mat_path: Path,
    subject_id: str,
    test_mode: bool = False,
) -> dict:
    """
    Full Stage 1 pipeline for one Cho 2017 subject.
    Returns manifest record.
    """
    out_dir = CLEAN_DIR / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject_id}_cleaned_raw.fif"

    print(f"\n[{subject_id}] Loading {mat_path.name}")

    try:
        # Load .mat file
        data = load_cho_subject(mat_path)
        print(f"  Shape: {data['x'].shape}")
        print(f"  sfreq: {data['sfreq']} Hz")
        print(f"  Channels: {len(data['chan_names'])}")
        print(f"  Trials: {len(data['y'])}")
        print(f"  Labels unique: {np.unique(data['y'])}")

        # Convert to continuous MNE Raw
        raw = trials_to_raw(
            x=data['x'],
            y=data['y'],
            sfreq=data['sfreq'],
            chan_names=data['chan_names'],
        )
        print(f"  Raw duration: {raw.times[-1]:.1f}s")

        # Apply ICA cleaning
        print(f"  Running ICA...")
        raw_clean, n_excluded, excluded_labels = \
            apply_ica_cleaning(raw)
        print(f"  Excluded {n_excluded} components: "
              f"{excluded_labels}")

        # Save cleaned .fif
        raw_clean.save(str(out_path),
                       overwrite=True, verbose=False)
        print(f"  Saved: {out_path}")

        return {
            "dataset": "cho2017",
            "subject_id": subject_id,
            "source_file": str(mat_path),
            "output_file": str(out_path),
            "n_trials": len(data['y']),
            "n_left": int((data['y'] == 1).sum()),
            "n_right": int((data['y'] == 2).sum()),
            "sfreq": data['sfreq'],
            "n_channels": len(data['chan_names']),
            "ica_n_components": "0.95var",
            "n_excluded": n_excluded,
            "excluded_labels": excluded_labels,
            "status": "success",
            "error": None,
            "processed_at":
                datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"  FAILED: {e}")
        return {
            "dataset": "cho2017",
            "subject_id": subject_id,
            "source_file": str(mat_path),
            "status": "failed",
            "error": err,
            "processed_at":
                datetime.now(timezone.utc).isoformat(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", action="store_true",
        help="Process first subject only"
    )
    args = parser.parse_args()

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # Find all .mat files
    mat_files = sorted(DATA_DIR.glob("s*.mat"))
    if not mat_files:
        print(f"No .mat files found in {DATA_DIR}")
        print("Download from: http://gigadb.org/dataset/100542")
        return

    print(f"Found {len(mat_files)} subjects")

    if args.test:
        mat_files = mat_files[:1]
        print("Test mode: processing first subject only")

    manifests = []
    for mat_path in mat_files:
        subject_id = mat_path.stem  # s1, s2, ...
        record = process_subject(
            mat_path, subject_id, args.test
        )
        manifests.append(record)

        # Write manifest entry
        with MANIFEST.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # Summary
    success = sum(
        1 for m in manifests if m["status"] == "success"
    )
    failed  = sum(
        1 for m in manifests if m["status"] == "failed"
    )
    print(f"\n{'='*50}")
    print(f"Stage 1 Cho 2017 Complete")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Output:  {CLEAN_DIR}")


if __name__ == "__main__":
    main()
