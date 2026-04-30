# ProjectCerebro — Week 1 Setup Guide

**Team:** Janardhana Reddy (jr6922), Divyansh Maurya (dm6022), Chiranjeev Kumar (ck4137)  
**Supervisor:** Prof. Sambit Sahu, NYU  
**Goal:** Replicate the Week 1 preprocessing pipeline on your local machine

---

## Overview

By the end of this guide you will have:
- All Docker databases running and healthy
- Raw EEG datasets downloaded
- Cleaned `.fif` files (from Stage 1 ICA cleaning)
- Preprocessed epochs in Delta Lake (from Stage 2 Spark)

The pipeline has two stages:

```
Stage 1: ICA Cleaning (standalone Python, ~2-3 hours)
  Raw .edf/.gdf files -> Cleaned .fif files

Stage 2: Spark Preprocessing (multiprocessing Pool(4), ~45-60 mins)
  Cleaned .fif files -> Delta Lake Parquet epochs
```

---

## Prerequisites

- macOS (M1/M2/Intel) or Linux
- Python 3.11
- Docker Desktop installed and running
- ~100GB free disk space
- Git

---

## Step 1 — Clone the Repository

```bash
git clone <repo_url>
cd projectcerebro
```

---

## Step 2 — Environment Setup

### 2a. Create Python virtual environment

```bash
python3.11 -m venv cerebro_env
source cerebro_env/bin/activate
pip install -r requirements.txt
```

### 2b. Install additional packages

```bash
pip install mne mne-icalabel pyspark delta-spark pyriemann
```

### 2c. Create Docker volume folders

```bash
mkdir -p volumes/mongodb
mkdir -p volumes/cassandra
mkdir -p volumes/redis
mkdir -p volumes/postgres
```

### 2d. Start Docker containers

```bash
docker-compose up -d
```

Wait 90 seconds for Cassandra to fully initialize, then verify all containers are healthy:

```bash
docker-compose ps
```

Expected output:
```
NAME                STATUS
cerebro_cassandra   Up (healthy)
cerebro_mongodb     Up (healthy)
cerebro_postgres    Up (healthy)
cerebro_redis       Up (healthy)
```

**Credentials for all services:** username `cerebro`, password `cerebro123`

**Ports:**
- MongoDB:    27017
- Cassandra:  9042
- PostgreSQL: 5433
- Redis:      6379

---

## Step 3 — Download Raw Datasets

### 3a. PhysioNet EEGMMIDB (3.4 GB)

```bash
mkdir -p data/physionet
wget -r -N -c -np \
  https://physionet.org/files/eegmmidb/1.0.0/ \
  -P data/physionet/
```

This downloads 109 subjects (S001-S109), each with 14 runs in `.edf` format at 160Hz, 64 channels.

### 3b. BCI Competition IV Dataset 2a (420 MB)

Download manually from:
```
https://bnci-horizon-2020.eu/database/data-sets
```

Select **Dataset 2a** from BCI Competition IV. Download all `.gdf` files and place them here:

```
data/bci_iv_2a/BCICIV_2a_gdf/
  A01T.gdf  A01E.gdf
  A02T.gdf  A02E.gdf
  ...
  A09T.gdf  A09E.gdf
```

9 subjects, training (T) and evaluation (E) sessions, 22 EEG + 3 EOG channels at 250Hz.

### 3c. Verify downloads

```bash
ls data/physionet/physionet.org/files/eegmmidb/1.0.0/ | head -5
ls data/bci_iv_2a/BCICIV_2a_gdf/
```

---

## Step 4 — Reproduce Preprocessing Pipeline

You have two options. **Option A** is fastest if someone shares the preprocessed files. **Option B** reproduces everything from scratch.

---

### Option A — Use Shared Preprocessed Files (Recommended)

Download from the shared Google Drive folder (ask Janardhana for link):

```
data_cleaned/          <- 3.6 GB  (Stage 1 output)
delta_lake/            <- ~400 MB (Stage 2 output)
```

Place both folders at the project root:

```
projectcerebro/
  data_cleaned/
  delta_lake/
  data/
  scripts/
  ...
```

Skip to **Verify Setup** below.

---

### Option B — Reproduce From Scratch

#### Stage 1 — ICA Cleaning (~2-3 hours)

This reads raw `.edf`/`.gdf` files, runs ICA artifact removal using ICLabel, and saves cleaned signals as `.fif` files.

```bash
source cerebro_env/bin/activate
python scripts/stage1_ica_cleaning.py
```

What it does:
- **PhysioNet:** applies 1-100Hz broadband filter, Common Average Reference, extended Infomax ICA, and ICLabel to remove eye blinks, muscle artifacts, and line noise
- **BCI IV-2a:** applies EOG regression first (uses dedicated EOG channels), then ICLabel for remaining artifacts
- Saves cleaned `.fif` files to `data_cleaned/physionet/` and `data_cleaned/bci_iv_2a/`
- Logs every run to `data_cleaned/stage1_cleaning_manifest.jsonl`

Expected output:
```
624 PhysioNet .fif files (~4.9 MB each)
9 BCI IV-2a .fif files (~63 MB each)
Total: ~3.6 GB
```

To run on a single subject first to verify:
```bash
# Edit stage1_ica_cleaning.py and set:
TEST_MODE = True
# Then run
python scripts/stage1_ica_cleaning.py
# Verify output looks correct, then set TEST_MODE = False and run full pipeline
```

#### Stage 2 — Spark Preprocessing (~45-60 minutes)

This reads cleaned `.fif` files, applies task-specific bandpass filtering, extracts epochs, resamples to 128Hz, applies baseline correction, and aligns channels. Writes to Delta Lake.

```bash
# Test on one subject first
python scripts/stage2_spark_preprocess.py --test --filter bp8_30

# If test passes, run full pipeline (both filter versions)
python scripts/stage2_spark_preprocess.py --filter both
```

What it does per epoch:
1. Channel alignment -> 5 common channels: `FZ, C3, CZ, C4, PZ`
2. Task filter: 8-30Hz (primary) or 4-38Hz (ablation)
3. Epoch extraction: -1s baseline + 3s post-cue = 4s total
4. Resample to 128Hz -> shape `(5, 512)` per epoch
5. Baseline correction: subtract mean of pre-event 1s window
6. Rest epochs: anchored to PhysioNet T0 events; BCI gap mining

Expected output:
```
Total epochs: ~20,086
  Left:   ~5,351
  Right:  ~5,305
  Rest:   ~9,430

Delta Lake tables:
  delta_lake/epochs_mi_v1_ch5_sr128_bp8_30/  <- primary
  delta_lake/epochs_mi_v1_ch5_sr128_bp4_38/  <- ablation
```

---

## Verify Setup

Run this in a Jupyter notebook to confirm everything is working:

```python
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import numpy as np

builder = (
    SparkSession.builder
    .appName("verify")
    .master("local[*]")
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.format("delta").load(
    "delta_lake/epochs_mi_v1_ch5_sr128_bp8_30"
)

print(f"Total epochs: {df.count()}")
df.groupBy("label_name").count().show()
df.groupBy("dataset").count().show()

row = df.first()
arr = np.array(row.features).reshape(row.n_channels, row.n_samples)
print(f"Epoch shape: {arr.shape}")       # should be (5, 512)
print(f"Channels: {row.channel_names}")  # should be [FZ, C3, CZ, C4, PZ]
```

Expected output:
```
Total epochs: 20086
+----------+-----+
|label_name|count|
+----------+-----+
|      left| 5351|
|      rest| 9430|
|     right| 5305|
+----------+-----+

Epoch shape: (5, 512)
Channels: ['FZ', 'C3', 'CZ', 'C4', 'PZ']
```

---

## Project Structure

```
projectcerebro/
├── data/                          # Raw datasets (not in git)
│   ├── physionet/                 # PhysioNet EEGMMIDB
│   └── bci_iv_2a/BCICIV_2a_gdf/  # BCI Competition IV-2a
├── data_cleaned/                  # Stage 1 output (not in git)
│   ├── physionet/                 # Cleaned .fif files per subject
│   └── bci_iv_2a/                 # Cleaned .fif files per subject
├── delta_lake/                    # Stage 2 output (not in git)
│   ├── epochs_mi_v1_ch5_sr128_bp8_30/   # Primary filter
│   └── epochs_mi_v1_ch5_sr128_bp4_38/   # Ablation filter
├── scripts/
│   ├── stage1_ica_cleaning.py     # Stage 1 ICA pipeline
│   ├── stage2_spark_preprocess.py # Stage 2 Spark pipeline
│   ├── ingest_mongodb.py          # MongoDB ingestion
│   └── ingest_cassandra_v2.py     # Cassandra ingestion
├── notebooks/                     # Jupyter notebooks
├── volumes/                       # Docker persistent storage (not in git)
├── docker-compose.yml
├── requirements.txt
└── docs/
    └── week1_setup.md             # This file
```

---

## Common Issues

**Cassandra takes too long to start:**
```bash
# Wait 90 seconds after docker-compose up -d
# Then check health
docker-compose ps
```

**PhysioNet wget is slow:**
```bash
# Add parallel downloads
wget -r -N -c -np --limit-rate=10m \
  https://physionet.org/files/eegmmidb/1.0.0/ \
  -P data/physionet/
```

**Stage 2 produces 0 epochs:**
```
This was a known bug (MNE off-by-one endpoint).
Already fixed in current stage2_spark_preprocess.py.
Make sure you have the latest version from git.
```

**Out of memory during Stage 2:**
```python
# In stage2_spark_preprocess.py reduce pool size:
POOL_SIZE = 2  # instead of 4
```

---

## What Comes Next (Week 2)

- Train/val/test split (subject-independent)
- Euclidean Alignment (fit on train only)
- Z-score normalization (fit on train only)
- EEGNet, EEG Conformer, ShallowConvNet training
- MLflow experiment tracking
- Ray Tune hyperparameter optimization
- Ensemble + Platt scaling calibration