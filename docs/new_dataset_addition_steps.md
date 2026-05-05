## What You Need To Do

```
Goal: Add Cho 2017 dataset to the pretrain pool
      Clean it through the same pipeline as
      PhysioNet and BCI IV-2a
      Store in Delta Lake for EEGNet retraining

Time estimate: 4-6 hours total
```

---

## Step 1 — Clone and Setup

```bash
# Clone the repo
git clone <repo_url>
cd projectcerebro

# Create virtual environment
python3.11 -m venv cerebro_env
source cerebro_env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install mne mne-icalabel scipy pymatreader

# Create Docker volumes and start databases
mkdir -p volumes/mongodb volumes/cassandra \
         volumes/redis volumes/postgres
docker-compose up -d

# Wait 90 seconds for Cassandra
sleep 90
docker-compose ps
# All four should show healthy
```

---

## Step 2 — Download Cho 2017 Dataset

```bash
# Create data directory
mkdir -p data/cho2017

# Download from GigaDB
# Go to: http://gigadb.org/dataset/100542
# Download all .mat files
# There are 52 subjects
# Files named: s1.mat, s2.mat, ... s52.mat
# Total size: approximately 5-7GB

# Place all .mat files here:
# projectcerebro/data/cho2017/
# Verify:
ls data/cho2017/
# Should show s1.mat s2.mat ... s52.mat
```

---

## Step 3 — Understand the Cho 2017 Structure

```python
# Run this in Jupyter to inspect one file
import scipy.io as sio
import numpy as np

mat = sio.loadmat('data/cho2017/s1.mat',
                  struct_as_record=False,
                  squeeze_me=True)

print("Keys:", mat.keys())
print("\nData structure:")

# Cho 2017 structure:
# mat['eeg'] contains the EEG data
# Fields inside eeg:
#   .x_train    EEG data shape (trials, channels, samples)
#   .y_train    labels (1=left, 2=right)
#   .x_test     test EEG data
#   .y_test     test labels
#   .smt        sampling rate
#   .chan_list  channel names

eeg = mat['eeg']
print(f"\nSampling rate: {eeg.smt}")
print(f"Train shape: {eeg.x_train.shape}")
print(f"Test shape:  {eeg.x_test.shape}")
print(f"Train labels unique: {np.unique(eeg.y_train)}")
print(f"Channel names: {eeg.chan_list}")
print(f"N channels: {len(eeg.chan_list)}")
```

---

## Step 4 — Verify Channel Intersection

```python
# Check that our 5 common channels exist in Cho 2017
import scipy.io as sio
import numpy as np

mat = sio.loadmat('data/cho2017/s1.mat',
                  struct_as_record=False,
                  squeeze_me=True)
eeg = mat['eeg']

# Clean channel names
chan_names = [str(c).strip().upper()
              for c in eeg.chan_list]
print("All channels:", chan_names)

# Check common channels
COMMON = ['FZ', 'C3', 'CZ', 'C4', 'PZ']
for ch in COMMON:
    found = ch in chan_names
    print(f"  {ch}: {'YES' % found}")

# Find indices of common channels
common_idx = [chan_names.index(ch)
              for ch in COMMON
              if ch in chan_names]
print(f"\nCommon channel indices: {common_idx}")
```

---

## Step 5 — Write Stage 1 Ingestion Script

```
Create this file:
projectcerebro/scripts/stage1_cho2017_ingest.py
```

```python
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

# Cho 2017 has explicit rest recordings
# in separate variable - use if available
# Otherwise use inter-trial gaps


def load_cho_subject(mat_path: Path) -> dict:
    """Load one Cho 2017 .mat file."""
    mat = sio.loadmat(
        str(mat_path),
        struct_as_record=False,
        squeeze_me=True
    )
    eeg = mat['eeg']

    # Extract fields
    x_train = eeg.x_train  # (trials, channels, samples)
    y_train = eeg.y_train  # labels: 1=left, 2=right
    x_test  = eeg.x_test
    y_test  = eeg.y_test
    sfreq   = float(eeg.smt)

    # Clean channel names
    chan_names = [str(c).strip().upper()
                  for c in eeg.chan_list]

    # Combine train and test
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

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

    # Create MNE info
    info = mne.create_info(
        ch_names=chan_names,
        sfreq=sfreq,
        ch_types=['eeg'] * len(chan_names),
    )

    raw = mne.io.RawArray(continuous, info, verbose=False)

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

    # Bandpass for ICA (broadband)
    raw_filt = raw.copy().filter(
        l_freq=1.0, h_freq=100.0,
        method='fir', phase='zero',
        verbose=False,
    )

    # Fit ICA
    n_components = ICA_VARIANCE
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=ICA_METHOD,
        fit_params={"extended": ICA_EXTENDED},
        random_state=42,
        verbose=False,
    )
    ica.fit(raw_filt, verbose=False)

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
```

---

## Step 6 — Run Stage 1 Test First

```bash
# Test on one subject first
python scripts/stage1_cho2017_ingest.py --test

# Verify output
ls -la data_cleaned/cho2017/s1/
# Should show: s1_cleaned_raw.fif

# Check the file
python -c "
import mne
raw = mne.io.read_raw_fif(
    'data_cleaned/cho2017/s1/s1_cleaned_raw.fif',
    preload=False, verbose=False
)
print('Channels:', raw.ch_names[:10])
print('Duration:', raw.times[-1])
print('Annotations:', raw.annotations)
"
```

---

## Step 7 — Run Full Stage 1

```bash
# If test passes run all 52 subjects
python scripts/stage1_cho2017_ingest.py

# Takes approximately 2-3 hours
# Monitor progress in terminal
```

---

## Step 8 — Update Stage 2 to Include Cho 2017

```
The existing stage2_spark_preprocess.py
needs one small update to handle Cho 2017.

In the script add a new iterator function:
```

```python
def iter_cho2017_jobs(
    filter_key: str,
    test_mode: bool
) -> list[tuple]:
    """Iterate Cho 2017 cleaned .fif files."""
    cho_dir = CLEANED_ROOT / "cho2017"
    jobs = []
    for subject_dir in sorted(cho_dir.glob("s*")):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name
        for fif in sorted(
            subject_dir.glob("*_cleaned_raw.fif")
        ):
            run_id = fif.stem.replace("_cleaned_raw", "")
            jobs.append((
                "cho2017",
                subject_id,
                run_id,
                fif,
                filter_key
            ))
        if test_mode:
            break
    return jobs
```

Also add event extraction for Cho 2017 in the main process function:

```python
# In process_one_file() add cho2017 case:
elif dataset == "cho2017":
    records = extract_cho2017_epochs(
        raw=raw,
        filter_cfg=filter_cfg,
        subject_id=subject_id,
        run_id=run_id,
        source_file=str(fif_path),
    )
```

And the epoch extraction function:

```python
def extract_cho2017_epochs(
    raw: mne.io.BaseRaw,
    filter_cfg: dict,
    subject_id: str,
    run_id: str,
    source_file: str,
) -> list[EpochRecord]:
    """
    Extract left/right epochs from Cho 2017.
    Annotations: 'left' and 'right'
    Rest: mined from inter-trial gaps (2s gaps)
    """
    records = []
    sfreq   = raw.info["sfreq"]
    now     = datetime.now(timezone.utc).isoformat()

    try:
        events, event_id = mne.events_from_annotations(
            raw, verbose=False
        )
    except Exception as e:
        print(f"  Warning: events failed {run_id}: {e}")
        return records

    left_code  = event_id.get("left")
    right_code = event_id.get("right")

    if left_code is None or right_code is None:
        print(f"  Warning: left/right not found "
              f"in {run_id}. Found: {event_id}")
        return records

    imagery_event_id = {
        "left":  left_code,
        "right": right_code,
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
        print(f"  Warning: Epochs failed {run_id}: {e}")
        return records

    imagery_event_samples = []

    for i, epoch in enumerate(epochs_mne):
        event_sample = epochs_mne.events[i, 0]
        event_code   = epochs_mne.events[i, 2]
        imagery_event_samples.append(event_sample)

        lcode = LABEL_LEFT if event_code == left_code \
                else LABEL_RIGHT
        lname = "left" if event_code == left_code \
                else "right"

        resampled = resample_epoch(epoch, sfreq)
        if resampled.shape[1] > N_SAMPLES:
            resampled = resampled[:, :N_SAMPLES]
        if resampled.shape[0] != N_CHANNELS or \
           resampled.shape[1] < N_SAMPLES:
            continue

        corrected = apply_baseline_correction(resampled)
        start_sec = (event_sample / sfreq) + TMIN
        end_sec   = (event_sample / sfreq) + TMAX
        eid       = build_epoch_id(
            "cho2017", subject_id, run_id,
            lcode, start_sec, end_sec
        )

        records.append(EpochRecord(
            epoch_id=eid,
            dataset="cho2017",
            subject_id=subject_id,
            session_id=None,
            run_id=run_id,
            source_file=source_file,
            label_code=lcode,
            label_name=lname,
            features=corrected.flatten()
                     .astype(np.float32).tolist(),
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

    # Mine rest from inter-trial gaps
    rest_records = extract_rest_epochs(
        raw=raw,
        imagery_event_samples=imagery_event_samples,
        sfreq=sfreq,
        n_rest_target=len(records) // 2,
        filter_cfg=filter_cfg,
        dataset="cho2017",
        subject_id=subject_id,
        run_id=run_id,
        source_file=source_file,
        now=now,
    )
    records.extend(rest_records)
    return records
```

---

## Step 9 — Run Stage 2 with Cho 2017

```bash
# Test mode first
python scripts/stage2_spark_preprocess.py \
  --test --filter bp8_30

# Check output includes cho2017:
# Should see:
# [cho2017] s1/s1

# Full pipeline
python scripts/stage2_spark_preprocess.py \
  --filter both
```

---

## Step 10 — Verify Delta Lake

```python
# Run in Jupyter to verify
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

builder = (
    SparkSession.builder
    .appName("verify_cho")
    .master("local[*]")
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)
spark = configure_spark_with_delta_pip(
    builder
).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.format("delta").load(
    "delta_lake/epochs_mi_v1_ch5_sr128_bp8_30"
)

print(f"Total epochs: {df.count():,}")
df.groupBy("dataset").count().show()
df.groupBy("dataset", "label_name").count().show()

# Expected:
# physionet: ~14,000 epochs
# cho2017:   ~12,480 epochs (52 x 240)
# bci_iv_2a: ~1,296 epochs
```

---

## Step 11 — Retrain EEGNet on Expanded Dataset

```bash
# Retrain PhysioNet+Cho pretrain
cerebro_env/bin/python -u \
  -m ml_core.experiments.pretrain_eegnet_physionet \
  --delta-path delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 \
  --filter-version bp_8_30_v1 \
  --datasets physionet cho2017 \
  --out-dir \
    artifacts/checkpoints/eegnet_physionet_cho_seed42 \
  --epochs 80 \
  --batch-size 64 \
  --lr 1e-3 \
  --patience 20 \
  --seed 42 \
  --use-ea \
  2>&1 | tee \
    artifacts/logs/eegnet_physionet_cho_seed42.log
```

---

## What To Share Back

After running everything share these files:

```
1. data_cleaned/cho2017_stage1_manifest.jsonl
   Shows how many subjects cleaned successfully

2. Terminal output of Stage 2 test mode
   Confirms cho2017 epochs extracted correctly

3. Delta Lake verification output
   Shows epoch counts per dataset

4. First 10 epochs of EEGNet pretrain log
   Shows if training is progressing
```