# ProjectCerebro — Data Pipeline Work (Divyansh)

## Overview

This document covers the full data ingestion and preprocessing pipeline implemented for ProjectCerebro. The goal was to download the EEG datasets, run Stage 1 (ICA artifact cleaning) and Stage 2 (epoch extraction), and produce standardized Parquet files ready for ML training — bypassing the Delta Lake / Spark dependency that was blocking the team.

---

## What Was Done

### 1. Repository Setup

- Cloned the `janardhan` branch of [JANARDHANAREDDYMS/projectcerebro](https://github.com/JANARDHANAREDDYMS/projectcerebro/tree/janardhan)
- Working directory: `/teamspace/studios/this_studio/projectcerebro/`

---

### 2. Dataset Downloads

#### PhysioNet EEG Motor Imagery Database (EEGMMIDB)

- **Source:** PhysioNet — `eegmmidb` v1.0.0
- **Download method:** `mne.datasets.eegbci.load_data()` (MNE's built-in downloader)
- **Subjects:** 104 non-corrupted subjects (1–109, excluding 88, 92, 100, 104, 106)
- **Runs per subject:** 6 motor imagery runs — R04, R06, R08, R10, R12, R14
- **Total files:** 624 EDF files
- **Local path:**
  ```
  projectcerebro/data/physionet_mne/MNE-eegbci-data/files/eegmmidb/1.0.0/<SXXX>/<SXXXRXX>.edf
  ```

#### BCI Competition IV Dataset 2a (BNCI2014-001)

- **Source:** Graz University BCI Lab via [MOABB](https://github.com/NeuroTechX/moabb)
- **Download method:** `moabb.datasets.BNCI2014_001().get_data(subjects=[...])`
- **Subjects:** All 9 subjects (A01–A09)
- **Sessions used:** Training session (`0train`) — 6 runs per subject
- **Total:** 54 runs
- **Local cache:** `~/mne_data/MNE-bnci-data/~bci/database/001-2014/`

---

### 3. Dependency Installation

The following packages were installed into the active environment:

```
mne==1.6.1 (upgraded to 1.12.1 by moabb)
mne-icalabel==0.8.1
wfdb==4.3.1
moabb==1.5.0
pyarrow==24.0.0
pyspark==3.5.4          # kept for reference; pipeline now uses pandas instead
delta-spark==3.2.1      # kept for reference; pipeline now uses pandas instead
huggingface_hub==1.14.0
tqdm
scipy==1.12.0 (pinned for MNE compatibility)
```

---

### 4. Path Fix — Stage 1 Script

The existing `scripts/run_ica_cleaning.py` expected PhysioNet data at:
```
data/physionet/physionet.org/files/eegmmidb/1.0.0/
```

MNE's downloader places files at:
```
data/physionet_mne/MNE-eegbci-data/files/eegmmidb/1.0.0/
```

**Fix:** Updated `PHYSIONET_ROOT` in `scripts/run_ica_cleaning.py` to match the MNE download path.

---

### 5. New File: `scripts/run_pipeline.py`

A unified pipeline script was written that replaces the separate `run_ica_cleaning.py` + `stage2_spark_preprocess.py` combination. Key improvements:

#### Stage 1 — ICA Cleaning
- Reads PhysioNet EDF files from the MNE download path
- Reads BCI IV-2a data **directly from MOABB** (no GDF files required)
- Applies broadband filter (1–100 Hz) + average reference
- Runs ICA using `infomax` (extended) with `mne-icalabel` to automatically label and remove non-brain components (muscle artifact, eye blink, heart beat, line noise, channel noise)
- Saves cleaned data as `.fif` files under `data_cleaned/`
- Writes a JSONL manifest at `data_cleaned/stage1_cleaning_manifest.jsonl`

#### Stage 2 — Epoch Extraction → Parquet
- Reads cleaned `.fif` files from Stage 1 output
- Aligns channels to the 5-channel common set: **FZ, C3, CZ, C4, PZ**
- Applies bandpass filter: **8–30 Hz** (primary motor imagery band)
- Extracts motor imagery epochs: `tmin=-1.0 s`, `tmax=+3.0 s` at 128 Hz → shape `(5, 512)`
- Labels: `0 = left hand`, `1 = right hand`, `2 = rest`
- **Writes directly to Parquet** (no Spark / Delta Lake needed) using `pandas` + `pyarrow` with Snappy compression

#### Key design decisions
| Original | New |
|----------|-----|
| Spark + Delta Lake for writing | Plain Parquet via pandas/pyarrow |
| GDF files for BCI IV-2a | MOABB API (downloads automatically) |
| Sequential Stage 1 | `multiprocessing.Pool(4)` for Stage 1 & 2 |
| Single monolithic pipeline | Modular with `--test`, `--n-subjects`, `--skip-stage1` flags |

#### Usage
```bash
# Test mode — 1 subject, verify end-to-end works
python scripts/run_pipeline.py --test

# PhysioNet only, 4 parallel workers
python scripts/run_pipeline.py --physionet-only --n-workers 4

# Full pipeline (PhysioNet + BCI IV-2a via MOABB)
python scripts/run_pipeline.py --n-workers 4

# Skip Stage 1 if FIFs already exist, jump straight to Stage 2
python scripts/run_pipeline.py --skip-stage1
```

---

### 6. New File: `scripts/run_full_pipeline.sh`

Shell wrapper that:
1. Polls until all 104 PhysioNet subjects are downloaded
2. Then launches `run_pipeline.py` with 4 parallel workers

```bash
nohup bash scripts/run_full_pipeline.sh > pipeline.log 2>&1 &
```

Monitor progress:
```bash
tail -f pipeline.log
```

---

### 7. New File: `scripts/share_output.py`

Utility to upload the Parquet output for team sharing:

```bash
# Upload to catbox.moe (anonymous, permanent URL)
python scripts/share_output.py --transfer

# Upload to Hugging Face dataset
python scripts/share_output.py --hf --token <HF_TOKEN> --repo yourname/projectcerebro-eeg
```

---

## Pipeline Test Run — Verified Working

A full end-to-end test was run for subject S001, run R04:

| Step | Result |
|------|--------|
| Stage 1 — ICA cleaning | 4 ICs removed (non-brain) |
| Stage 2 — Epoch extraction | 29 epochs: 8 left, 7 right, 14 rest |
| Parquet output | 0.3 MB, 19 columns, valid schema |

Test Parquet preview (S001 only):
- **Download:** https://files.catbox.moe/n8b83c.parquet

---

## Parquet Output Location

### HuggingFace Dataset (Team Shared)

The parquet is hosted on HuggingFace Datasets for team access:

**https://huggingface.co/datasets/divyanshmaurya1/projectcerebro-eeg**

Teammates can load it directly:
```python
from huggingface_hub import hf_hub_download
import pandas as pd, numpy as np, io

path = hf_hub_download(repo_id="divyanshmaurya1/projectcerebro-eeg",
                       filename="epochs_mi_bp8_30.parquet", repo_type="dataset")
df = pd.read_parquet(path)

def load_features(row):
    return np.load(io.BytesIO(row["features_bytes"]))  # shape: (5, 512)

X = np.stack([load_features(r) for _, r in df.iterrows()])
y = df["label_code"].values
```

### Local Studio Path (Full Pipeline Output)
```
projectcerebro/parquet_output/epochs_mi_bp8_30.parquet
```

**Full path on this studio:**
```
/teamspace/studios/this_studio/projectcerebro/parquet_output/epochs_mi_bp8_30.parquet
```

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `epoch_id` | string | Unique ID: `dataset\|subject\|run\|label\|start_ms\|end_ms` |
| `dataset` | string | `physionet` or `bci_iv_2a` |
| `subject_id` | string | e.g. `S001`, `A01` |
| `run_id` | string | e.g. `S001R04` |
| `label_code` | int | 0=left, 1=right, 2=rest |
| `label_name` | string | `left`, `right`, `rest` |
| `features_bytes` | bytes | numpy `.npy` bytes of shape `(5, 512)` float32 — 5 channels × 512 samples |
| `n_channels` | int | Always 5 |
| `n_samples` | int | Always 512 |
| `channel_names` | string | `FZ,C3,CZ,C4,PZ` |
| `sampling_rate_hz` | float | 128.0 |
| `filter_version` | string | `bp_8_30_v1` |
| `preprocessing_version` | string | `v1.1.0` |
| `is_rest_synthetic` | bool | False for PhysioNet T0 rest, True for BCI gap-mined rest |

**To load the features from the parquet:**
```python
import pandas as pd
import numpy as np
import io

df = pd.read_parquet("parquet_output/epochs_mi_bp8_30.parquet")

def load_features(row):
    buf = io.BytesIO(row["features_bytes"])
    return np.load(buf)  # shape: (5, 512)

# Example: get all left-hand epochs as numpy array
left = df[df["label_code"] == 0]
X = np.stack([load_features(r) for _, r in left.iterrows()])  # (N, 5, 512)
```

### Expected Scale (Full Run)
- ~18,000+ epochs from PhysioNet (104 subjects × 6 runs × ~29 epochs/run)
- ~1,300+ epochs from BCI IV-2a (9 subjects × 6 runs/session × ~24 epochs/run)
- Estimated file size: **50–150 MB** (Snappy-compressed Parquet)

---

## Files Added / Modified

| File | Status | Description |
|------|--------|-------------|
| `scripts/run_ica_cleaning.py` | Modified | Fixed `PHYSIONET_ROOT` path to match MNE download location |
| `scripts/run_pipeline.py` | **New** | Unified Stage 1 + Stage 2 pipeline (Parquet output, no Spark) |
| `scripts/run_full_pipeline.sh` | **New** | Shell runner: waits for download, then runs full pipeline |
| `scripts/share_output.py` | **New** | Upload parquet to catbox.moe or HuggingFace for team sharing |
| `docs/divyansh_pipeline_progress.md` | **New** | This document |

---

## Current Status (as of push)

The full pipeline is **running in the background** on the Lightning AI Studio. Progress can be monitored with:

```bash
tail -f /teamspace/studios/this_studio/projectcerebro/pipeline.log
```

Expected completion: **~7–10 hours** from pipeline start (download ~1.5h + Stage 1 ICA ~6h + Stage 2 ~30min).

Once complete, upload the parquet for team access:
```bash
cd /teamspace/studios/this_studio/projectcerebro
python scripts/share_output.py --transfer
```
