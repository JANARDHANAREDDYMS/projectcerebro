# ProjectCerebro — MongoDB Subject Metadata Ingestion
# Reads EDF/GDF headers and stores subject metadata in MongoDB
# Run: python3 scripts/ingest_mongodb.py

import os
import mne
import pymongo
from datetime import datetime
from tqdm import tqdm

mne.set_log_level('WARNING')

# ── Config ────────────────────────────────────────────────
PHYSIONET_BASE = "data/physionet/physionet.org/files/eegmmidb/1.0.0"
BCI_BASE       = "data/bci_iv_2a/BCICIV_2a_gdf"

MONGO_URI      = "mongodb://cerebro:cerebro123@localhost:27017"
DB_NAME        = "projectcerebro"
COLLECTION     = "subjects"

# Known corrupted PhysioNet subjects
CORRUPTED      = {88, 92, 100, 104, 106}

# Motor imagery runs we care about
PHYSIONET_RUNS = [4, 6, 8, 10, 12, 14]

# ── MongoDB Connection ─────────────────────────────────────
client     = pymongo.MongoClient(MONGO_URI)
db         = client[DB_NAME]
collection = db[COLLECTION]

# Clear existing data
collection.drop()
print("✅ MongoDB collection cleared")

# ── PhysioNet Ingestion ────────────────────────────────────
def ingest_physionet():
    print("\n📥 Ingesting PhysioNet subjects...")
    subjects = sorted([
        d for d in os.listdir(PHYSIONET_BASE)
        if d.startswith("S") and os.path.isdir(
            os.path.join(PHYSIONET_BASE, d)
        )
    ])

    success = 0
    failed  = 0

    for subject_folder in tqdm(subjects):
        subject_id = subject_folder  # e.g. S001
        subject_num = int(subject_id[1:])
        subject_path = os.path.join(PHYSIONET_BASE, subject_folder)

        is_corrupted = subject_num in CORRUPTED

        # Read first available motor imagery run for metadata
        metadata = None
        available_runs = []

        for run in PHYSIONET_RUNS:
            run_str  = f"{subject_id}R{run:02d}"
            edf_path = os.path.join(subject_path, f"{run_str}.edf")

            if not os.path.exists(edf_path):
                continue

            available_runs.append(run)

            # Read metadata from first run only
            if metadata is None and not is_corrupted:
                try:
                    raw = mne.io.read_raw_edf(edf_path, preload=False)
                    metadata = {
                        "sampling_rate": raw.info['sfreq'],
                        "n_channels":    len(raw.ch_names),
                        "channel_names": raw.ch_names,
                        "duration_sec":  raw.times[-1],
                    }
                    raw.close()
                except Exception as e:
                    print(f"⚠️  Could not read {edf_path}: {e}")

        # Build MongoDB document
        doc = {
            "subject_id":     subject_id,
            "dataset":        "physionet",
            "is_corrupted":   is_corrupted,
            "available_runs": available_runs,
            "sampling_rate":  metadata["sampling_rate"] if metadata else None,
            "n_channels":     metadata["n_channels"]    if metadata else None,
            "channel_names":  metadata["channel_names"] if metadata else None,
            "duration_sec":   metadata["duration_sec"]  if metadata else None,
            "labels": {
                "T0": "rest",
                "T1": "left_fist",
                "T2": "right_fist"
            },
            "ingested_at": datetime.utcnow()
        }

        try:
            collection.insert_one(doc)
            success += 1
        except Exception as e:
            print(f"❌ Failed to insert {subject_id}: {e}")
            failed += 1

    print(f"✅ PhysioNet done: {success} inserted, {failed} failed")

# ── BCI IV-2a Ingestion ────────────────────────────────────
def ingest_bci():
    print("\n📥 Ingesting BCI IV-2a subjects...")

    success = 0
    failed  = 0

    for subject_num in tqdm(range(1, 10)):  # subjects 1-9
        subject_id = f"A{subject_num:02d}"

        # Training and evaluation sessions
        sessions = {}
        for session in ["T", "E"]:
            gdf_path = os.path.join(
                BCI_BASE, f"{subject_id}{session}.gdf"
            )
            if not os.path.exists(gdf_path):
                continue
            try:
                raw = mne.io.read_raw_gdf(gdf_path, preload=False)
                sessions[session] = {
                    "sampling_rate": raw.info['sfreq'],
                    "n_channels":    len(raw.ch_names),
                    "channel_names": raw.ch_names,
                    "duration_sec":  raw.times[-1],
                }
                raw.close()
            except Exception as e:
                print(f"⚠️  Could not read {gdf_path}: {e}")

        # Build MongoDB document
        doc = {
            "subject_id":   subject_id,
            "dataset":      "bci_iv_2a",
            "is_corrupted": False,
            "sessions":     sessions,
            "sampling_rate": sessions.get("T", {}).get("sampling_rate"),
            "n_channels":    sessions.get("T", {}).get("n_channels"),
            "channel_names": sessions.get("T", {}).get("channel_names"),
            "duration_sec":  sessions.get("T", {}).get("duration_sec"),
            "labels": {
                "769": "left_hand",
                "770": "right_hand",
                "771": "feet",
                "772": "tongue"
            },
            "ingested_at": datetime.utcnow()
        }

        try:
            collection.insert_one(doc)
            success += 1
        except Exception as e:
            print(f" Failed to insert {subject_id}: {e}")
            failed += 1

    print(f" BCI IV-2a done: {success} inserted, {failed} failed")

# ── Verify ─────────────────────────────────────────────────
def verify():
    print("\n📊 MongoDB Verification:")
    total      = collection.count_documents({})
    physionet  = collection.count_documents({"dataset": "physionet"})
    bci        = collection.count_documents({"dataset": "bci_iv_2a"})
    corrupted  = collection.count_documents({"is_corrupted": True})

    print(f"  Total subjects:      {total}")
    print(f"  PhysioNet subjects:  {physionet}")
    print(f"  BCI IV-2a subjects:  {bci}")
    print(f"  Corrupted (flagged): {corrupted}")

    print("\n📄 Sample PhysioNet document:")
    sample = collection.find_one({"dataset": "physionet"})
    print(f"  subject_id:     {sample['subject_id']}")
    print(f"  sampling_rate:  {sample['sampling_rate']} Hz")
    print(f"  n_channels:     {sample['n_channels']}")
    print(f"  available_runs: {sample['available_runs']}")
    print(f"  is_corrupted:   {sample['is_corrupted']}")

    print("\n📄 Sample BCI document:")
    sample_bci = collection.find_one({"dataset": "bci_iv_2a"})
    print(f"  subject_id:    {sample_bci['subject_id']}")
    print(f"  sampling_rate: {sample_bci['sampling_rate']} Hz")
    print(f"  n_channels:    {sample_bci['n_channels']}")
    print(f"  sessions:      {list(sample_bci['sessions'].keys())}")

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    print(" ProjectCerebro — MongoDB Ingestion")
    print("=" * 45)
    ingest_physionet()
    ingest_bci()
    verify()
    client.close()
    print("\n MongoDB ingestion complete!")