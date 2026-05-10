# Cho 2017 Dataset — Full Pipeline Run

## Overview

This document records the complete end-to-end pipeline run for adding the **Cho 2017** EEG motor imagery dataset to the ProjectCerebro pretrain pool, following the steps in `docs/new_dataset_addition_steps.md`.

**Date:** 2026-05-10  
**Branch:** `janardhan`  
**HuggingFace Dataset:** https://huggingface.co/datasets/divyanshmaurya1/BCI_Data_new

---

## Step 1 — Environment Setup

Cloned the `janardhan` branch and created a Python 3.12 virtual environment (Python 3.11 was not available on this machine):

```bash
git clone -b janardhan https://github.com/JANARDHANAREDDYMS/projectcerebro.git
cd projectcerebro
python3.12 -m venv cerebro_env
source cerebro_env/bin/activate
pip install mne==1.6.1 mne-icalabel==0.8.1 scipy pymatreader \
            huggingface_hub datasets pyarrow pandas numpy moabb
```

---

## Step 2 — Data Download

The Cho 2017 dataset (GigaDB 100295) was downloaded programmatically from the Wasabi S3 mirror used by MOABB:

```
https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data/
```

All 52 subject files (`s01.mat` – `s52.mat`, ~200MB each, ~10GB total) were downloaded in parallel (8 concurrent) into `data/cho2017/`.

```bash
mkdir -p data/cho2017
seq 1 52 | xargs -P 8 -I{} bash -c '
  fn=$(printf "s%02d.mat" {})
  curl -sS -o "data/cho2017/$fn" "<GIGA_URL>$fn"
'
```

---

## Step 3 — Actual Cho 2017 Mat File Structure

The readme described fields `eeg.x_train`, `eeg.y_train`, `eeg.smt`, `eeg.chan_list`. The actual file structure differs:

| Field | Description |
|-------|-------------|
| `eeg.imagery_left` | `(68, n_trials × epoch_samples)` — continuous left trials |
| `eeg.imagery_right` | `(68, n_trials × epoch_samples)` — continuous right trials |
| `eeg.srate` | Sampling rate (512 Hz) |
| `eeg.n_imagery_trials` | 100 trials per class |
| `eeg.frame` | `[-2000, 5000]` ms window → 3584 samples per trial |
| `eeg.imagery_event` | Binary onset markers |

**68 channels:** 64 EEG (standard 10-10 montage) + 4 EMG. Channel names sourced from MOABB's `gigadb.py`.

---

## Step 4 — Stage 1: ICA Cleaning (`scripts/stage1_cho2017_ingest.py`)

Created exactly as specified in the readme, with adaptations for the actual mat file structure.

**Key adaptations:**
- DC removal per channel: `imagery_left - imagery_left.mean(axis=1, keepdims=True)`
- Unit conversion µV → V: `× 1e-6`
- Reshape: `(68, 358400)` → `(100, 68, 3584)` per class
- Channel types: first 64 = EEG, last 4 = EMG
- Standard 10-10 montage set for ICLabel electrode positions
- Average reference projection applied before ICA
- ICA: 20 fixed components (infomax + extended), ICLabel artifact rejection (>0.8 confidence)

**Run:**
```bash
# Test first subject
python scripts/stage1_cho2017_ingest.py --test

# All 52 subjects (~2 min/subject, ~1h 45m total)
python scripts/stage1_cho2017_ingest.py
```

**Result:** 52/52 subjects successful, 0 failures.  
Output: `data_cleaned/cho2017/s01/s01_cleaned_raw.fif` … `s52/s52_cleaned_raw.fif`  
Manifest: `data_cleaned/cho2017_stage1_manifest.jsonl`

---

## Step 5 — Stage 2: Updated `scripts/stage2_spark_preprocess.py`

Added three components exactly as specified in the readme:

**1. Path constant:**
```python
CLEANED_CHO2017 = CLEANED_ROOT / "cho2017"
```

**2. `iter_cho2017_jobs()` — file iterator**

**3. `extract_cho2017_epochs()` — epoch extractor**
- Reads `left` / `right` annotations from cleaned .fif
- Extracts epochs: TMIN=-1s, TMAX=+3s → 512 samples at 128Hz
- Mines rest epochs from inter-trial gaps
- Selects 5 common channels: FZ, C3, CZ, C4, PZ

**4. Wired into `process_one_file()` and `run_pipeline()`**

> **Note:** Spark/Java was not available on this machine. A Spark-free equivalent was created at `scripts/stage2_parquet_cho2017.py` that produces identical output as Parquet files using `pyarrow`.

---

## Step 6 — Stage 2 Run (`scripts/stage2_parquet_cho2017.py`)

```bash
python scripts/stage2_parquet_cho2017.py --filter both
```

**Results:**

| Metric | Value |
|--------|-------|
| Subjects processed | 49 (out of 52 — 3 had no cleaned fif) |
| Total epochs | 10,170 per filter |
| Left epochs | 5,060 |
| Right epochs | 5,060 |
| Rest epochs | 50 |
| Channels | FZ, C3, CZ, C4, PZ |
| Sampling rate | 128 Hz |
| Epoch length | 512 samples (4s) |

Output:
- `parquet_export/cho2017_epochs_ch5_sr128_bp8_30/` — primary filter (8–30 Hz)
- `parquet_export/cho2017_epochs_ch5_sr128_bp4_38/` — ablation filter (4–38 Hz)

Each filter produces 49 parquet files partitioned by `subject_id`.

---

## Step 7 — HuggingFace Upload (`scripts/upload_cho2017_to_hf.py`)

```bash
python scripts/upload_cho2017_to_hf.py
```

**Repo:** `divyanshmaurya1/BCI_Data_new`  
**URL:** https://huggingface.co/datasets/divyanshmaurya1/BCI_Data_new  
**Token:** HuggingFace write token for account `divyanshmaurya1`

**Uploaded:**
- `README.md` — dataset card with schema, preprocessing details, citations
- `bp8_30/subject_id=s01/` … `s52/` — 50 parquet files
- `bp4_38/subject_id=s01/` … `s52/` — 50 parquet files
- **102 total files**

---

## Scripts Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `scripts/stage1_cho2017_ingest.py` | Created | Stage 1 ICA cleaning for Cho 2017 |
| `scripts/stage2_spark_preprocess.py` | Modified | Added cho2017 iterator + epoch extractor |
| `scripts/stage2_parquet_cho2017.py` | Created | Spark-free Stage 2 → Parquet export |
| `scripts/upload_cho2017_to_hf.py` | Created | HuggingFace upload script |

---

## Parquet Schema

Each row is one 4-second epoch:

| Column | Type | Description |
|--------|------|-------------|
| `epoch_id` | string | Unique ID: `cho2017\|s01\|s01\|0\|...` |
| `dataset` | string | `cho2017` |
| `subject_id` | string | `s01` – `s52` |
| `label_code` | int32 | 0=left, 1=right, 2=rest |
| `label_name` | string | `left`, `right`, `rest` |
| `features` | float32[] | Flattened EEG: 5 × 512 = 2560 values |
| `n_channels` | int32 | 5 |
| `n_samples` | int32 | 512 |
| `channel_names` | string[] | `[FZ, C3, CZ, C4, PZ]` |
| `sampling_rate_hz` | float32 | 128.0 |
| `filter_version` | string | `bp_8_30_v1` or `bp_4_38_v1` |
| `preprocessing_version` | string | `v1.1.0` |
| `is_rest_synthetic` | bool | True for gap-mined rest epochs |

---

## Stage 1 Manifest Summary

File: `data_cleaned/cho2017_stage1_manifest.jsonl`  
- 52 entries (one per subject)
- Fields: `subject_id`, `n_trials`, `n_left`, `n_right`, `sfreq`, `n_channels`, `ica_n_components`, `n_excluded`, `excluded_labels`, `status`, `processed_at`
