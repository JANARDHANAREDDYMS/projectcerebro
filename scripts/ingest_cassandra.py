# ProjectCerebro -- Cassandra EEG Epoch Ingestion v2
# Schema: list<float> per channel (64 rows per epoch vs 40,960)
# 3-class classification: left(0), right(1), rest(2)
# Epoch: 1.0s baseline + 3.0s post-event = 4.0s total
# Run from project root:
#   python3 scripts/ingest_cassandra_v2.py

import os
from pathlib import Path
from datetime import datetime

import mne
import numpy as np
from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent_with_args
from tqdm import tqdm

mne.set_log_level("WARNING")

# ── Project Paths ─────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
PHYSIONET_BASE = PROJECT_ROOT / "data" / "physionet" / "physionet.org" / "files" / "eegmmidb" / "1.0.0"
BCI_BASE       = PROJECT_ROOT / "data" / "bci_iv_2a" / "BCICIV_2a_gdf"
PROGRESS_FILE  = PROJECT_ROOT / "scripts" / ".ingest_progress.txt"

# ── Config ────────────────────────────────────────────────
CASSANDRA_HOST      = "localhost"
CASSANDRA_PORT      = 9042
KEYSPACE            = "projectcerebro"

# Toggle these before running
DEBUG               = False       # set True for verbose per-epoch logging
TEST_ONE_SUBJECT    = False        # set True to test on S001 + A01 only

# Epoch parameters
BASELINE_SEC        = 1.0         # seconds before event onset
POST_EVENT_SEC      = 3.0         # seconds after event onset
EPOCH_LENGTH_SEC    = BASELINE_SEC + POST_EVENT_SEC  # 4.0s total

# Cassandra tuning
CASSANDRA_TIMEOUT   = 120         # seconds
INSERT_CONCURRENCY  = 50          # concurrent row inserts per epoch

# PhysioNet config
PHYSIONET_RUNS      = [4, 6, 8, 10, 12, 14]
CORRUPTED           = {88, 92, 100, 104, 106}

# Label mappings
PHYSIONET_LABELS    = {
    1: ("rest",  2),
    2: ("left",  0),
    3: ("right", 1),
}


# ── Helpers ───────────────────────────────────────────────
def debug(message):
    if DEBUG:
        print(f"  [DEBUG] {message}")


def ensure_paths_exist():
    print("\nChecking dataset paths...")
    print(f"  PhysioNet: {PHYSIONET_BASE}")
    print(f"  BCI IV-2a: {BCI_BASE}")
    if not PHYSIONET_BASE.exists():
        raise FileNotFoundError(
            f"PhysioNet path not found: {PHYSIONET_BASE}"
        )
    if not BCI_BASE.exists():
        print(f"  Warning: BCI path not found: {BCI_BASE}")
    print("  Path check passed")


def build_epoch_id(dataset, subject_id, run_id, label_code,
                   epoch_start_sec, epoch_end_sec):
    # Deterministic ID -- prevents duplicates on restart
    start_ms = int(round(epoch_start_sec * 1000))
    end_ms   = int(round(epoch_end_sec   * 1000))
    return f"{dataset}|{subject_id}|{run_id}|{label_code}|{start_ms}|{end_ms}"


def save_progress(subject_id):
    # Append completed subject to progress file
    # Allows safe resume after crash
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{subject_id}\n")


def load_completed_subjects():
    # Return set of already ingested subjects
    if not PROGRESS_FILE.exists():
        return set()
    with open(PROGRESS_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def clear_progress():
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("  Progress file cleared")


# ── Cassandra Connection ──────────────────────────────────
def connect_cassandra():
    print("\nConnecting to Cassandra...")
    cluster = Cluster(
        [CASSANDRA_HOST],
        port=CASSANDRA_PORT
    )
    session = cluster.connect()
    session.default_timeout = CASSANDRA_TIMEOUT
    print(f"  Connected")
    print(f"  Timeout: {CASSANDRA_TIMEOUT}s")
    return cluster, session


# ── Schema Setup ──────────────────────────────────────────
def setup_schema(session):
    print("\nSetting up Cassandra schema...")

    session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}
        WITH replication = {{
            'class': 'SimpleStrategy',
            'replication_factor': 1
        }}
    """)

    session.set_keyspace(KEYSPACE)

    # New schema: one row per channel per epoch
    # amplitudes stored as list<float>
    # 64 rows per PhysioNet epoch (vs 40,960 in v1)
    # 25 rows per BCI epoch (vs 25,000 in v1)
    session.execute("""
        CREATE TABLE IF NOT EXISTS eeg_epochs (
            epoch_id        TEXT,
            subject_id      TEXT,
            dataset         TEXT,
            run_id          TEXT,
            label           TEXT,
            label_code      INT,
            channel         TEXT,
            channel_idx     INT,
            amplitudes      LIST<FLOAT>,
            n_samples       INT,
            sampling_rate   FLOAT,
            epoch_start_sec FLOAT,
            epoch_end_sec   FLOAT,
            ingested_at     TIMESTAMP,
            PRIMARY KEY ((epoch_id), channel_idx)
        )
    """)

    session.execute("""
        CREATE INDEX IF NOT EXISTS ON eeg_epochs (subject_id)
    """)
    session.execute("""
        CREATE INDEX IF NOT EXISTS ON eeg_epochs (label_code)
    """)
    session.execute("""
        CREATE INDEX IF NOT EXISTS ON eeg_epochs (dataset)
    """)

    print("  Schema ready")
    print("  Storage: list<float> per channel")
    print("  Rows per PhysioNet epoch: 64")
    print("  Rows per BCI epoch:       25")


# ── Prepared Statement ────────────────────────────────────
def get_prepared_statement(session):
    return session.prepare("""
        INSERT INTO eeg_epochs (
            epoch_id,
            subject_id,
            dataset,
            run_id,
            label,
            label_code,
            channel,
            channel_idx,
            amplitudes,
            n_samples,
            sampling_rate,
            epoch_start_sec,
            epoch_end_sec,
            ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)


# ── Epoch Insertion ───────────────────────────────────────
def insert_epoch(
    session,
    prepared,
    epoch_data,
    ch_names,
    subject_id,
    dataset,
    run_id,
    label,
    label_code,
    sfreq,
    epoch_start_sec,
    epoch_end_sec,
    ingested_at
):
    epoch_id  = build_epoch_id(
        dataset, subject_id, run_id,
        label_code, epoch_start_sec, epoch_end_sec
    )
    n_samples = epoch_data.shape[1]

    debug(
        f"Inserting {epoch_id} | "
        f"label={label} | "
        f"shape={epoch_data.shape} | "
        f"rows={len(ch_names)}"
    )

    # One row per channel, all samples as a list
    rows = []
    for ch_idx, channel in enumerate(ch_names):
        amplitudes = epoch_data[ch_idx, :].tolist()
        rows.append((
            epoch_id,
            subject_id,
            dataset,
            run_id,
            label,
            label_code,
            channel,
            ch_idx,
            amplitudes,
            n_samples,
            float(sfreq),
            float(epoch_start_sec),
            float(epoch_end_sec),
            ingested_at
        ))

    # Concurrent insert of all channel rows
    results = execute_concurrent_with_args(
        session,
        prepared,
        rows,
        concurrency=INSERT_CONCURRENCY,
        raise_on_first_error=False
    )

    failed = [r for success, r in results if not success]
    if failed:
        print(f"  Warning: {len(failed)} rows failed for {epoch_id}")

    return epoch_id


# ── Rest Epoch Extraction ─────────────────────────────────
def extract_rest_epochs(raw, imagery_events, sfreq):
    rest_epochs   = []
    epoch_samples = int(EPOCH_LENGTH_SEC * sfreq)
    total_samples = raw.get_data().shape[1]

    for i in range(len(imagery_events) - 1):
        current_end = imagery_events[i][0] + int(POST_EVENT_SEC * sfreq)
        next_start  = imagery_events[i + 1][0]
        rest_start  = current_end
        rest_end    = rest_start + epoch_samples

        # Rest window must fit between trials and within recording
        if rest_end < next_start and rest_end < total_samples:
            epoch_data = raw.get_data(
                start=rest_start,
                stop=rest_end
            )
            rest_epochs.append((
                epoch_data,
                rest_start / sfreq,
                rest_end   / sfreq
            ))

    return rest_epochs


# ── PhysioNet Ingestion ───────────────────────────────────
def ingest_physionet(session, prepared):
    print("\nIngesting PhysioNet subjects...")

    subjects = sorted([
        d for d in os.listdir(PHYSIONET_BASE)
        if d.startswith("S") and
        os.path.isdir(PHYSIONET_BASE / d)
    ])

    if TEST_ONE_SUBJECT:
        subjects = subjects[:1]
        print(f"  Test mode: {subjects}")

    # Resume support
    completed = load_completed_subjects()
    if completed:
        print(f"  Resuming: {len(completed)} subjects already done")
        subjects = [s for s in subjects if s not in completed]
        print(f"  Remaining: {len(subjects)} subjects")

    total_epochs = 0
    failed_runs  = 0
    ingested_at  = datetime.utcnow()

    for subject_folder in tqdm(subjects, desc="PhysioNet"):
        subject_num = int(subject_folder[1:])

        if subject_num in CORRUPTED:
            debug(f"Skipping corrupted: {subject_folder}")
            save_progress(subject_folder)
            continue

        subject_id   = subject_folder
        subject_path = PHYSIONET_BASE / subject_folder

        for run in PHYSIONET_RUNS:
            run_str  = f"{subject_id}R{run:02d}"
            edf_path = subject_path / f"{run_str}.edf"

            if not edf_path.exists():
                debug(f"Missing: {edf_path}")
                continue

            try:
                raw   = mne.io.read_raw_edf(str(edf_path), preload=True)
                sfreq = raw.info["sfreq"]
                events, _ = mne.events_from_annotations(raw)

                debug(f"{run_str}: {len(events)} events")

                imagery_events = []

                for event in events:
                    event_sample = event[0]
                    event_code   = event[2]

                    if event_code not in PHYSIONET_LABELS:
                        continue

                    label, label_code = PHYSIONET_LABELS[event_code]

                    # Rest handled separately below
                    if label_code == 2:
                        continue

                    start_sample = int(event_sample - BASELINE_SEC * sfreq)
                    end_sample   = int(event_sample + POST_EVENT_SEC * sfreq)

                    if start_sample < 0:
                        continue
                    if end_sample > raw.get_data().shape[1]:
                        continue

                    epoch_data = raw.get_data(
                        start=start_sample,
                        stop=end_sample
                    )

                    insert_epoch(
                        session, prepared,
                        epoch_data, raw.ch_names,
                        subject_id, "physionet", run_str,
                        label, label_code, sfreq,
                        start_sample / sfreq,
                        end_sample   / sfreq,
                        ingested_at
                    )
                    total_epochs  += 1
                    imagery_events.append(event)

                # Rest epochs between imagery events
                if len(imagery_events) > 0:
                    rest_epochs = extract_rest_epochs(
                        raw, np.array(imagery_events), sfreq
                    )
                    for epoch_data, start_sec, end_sec in rest_epochs:
                        insert_epoch(
                            session, prepared,
                            epoch_data, raw.ch_names,
                            subject_id, "physionet", run_str,
                            "rest", 2, sfreq,
                            start_sec, end_sec,
                            ingested_at
                        )
                        total_epochs += 1

                raw.close()

            except Exception as e:
                print(f"  Failed {run_str}: {e}")
                failed_runs += 1

        save_progress(subject_folder)

    print(f"  PhysioNet done: {total_epochs} epochs, {failed_runs} failed runs")
    return total_epochs


# ── BCI IV-2a Ingestion ───────────────────────────────────
def ingest_bci(session, prepared):
    print("\nIngesting BCI IV-2a subjects...")

    if not BCI_BASE.exists():
        print("  BCI path not found. Skipping.")
        return 0

    completed    = load_completed_subjects()
    total_epochs = 0
    failed_files = 0
    ingested_at  = datetime.utcnow()

    for subject_num in tqdm(range(1, 10), desc="BCI IV-2a"):
        subject_id = f"A{subject_num:02d}"

        if subject_id in completed:
            debug(f"Skipping completed: {subject_id}")
            continue

        for session_type in ["T"]:
            gdf_path = BCI_BASE / f"{subject_id}{session_type}.gdf"

            if not gdf_path.exists():
                debug(f"Missing: {gdf_path}")
                continue

            try:
                raw   = mne.io.read_raw_gdf(str(gdf_path), preload=True)
                sfreq = raw.info["sfreq"]
                events, event_id = mne.events_from_annotations(raw)

                debug(f"{subject_id}{session_type}: {len(events)} events")

                # Map GDF event codes to labels
                label_map = {}
                for k, v in event_id.items():
                    if k == "769":
                        label_map[v] = ("left",  0)
                    elif k == "770":
                        label_map[v] = ("right", 1)
                    elif k == "1023":
                        label_map[v] = ("rejected", -1)

                imagery_events = []

                for event in events:
                    event_sample = event[0]
                    event_code   = event[2]

                    if event_code not in label_map:
                        continue

                    label, label_code = label_map[event_code]

                    if label_code == -1:
                        continue

                    start_sample = int(event_sample - BASELINE_SEC * sfreq)
                    end_sample   = int(event_sample + POST_EVENT_SEC * sfreq)

                    if start_sample < 0:
                        continue
                    if end_sample > raw.get_data().shape[1]:
                        continue

                    epoch_data = raw.get_data(
                        start=start_sample,
                        stop=end_sample
                    )

                    insert_epoch(
                        session, prepared,
                        epoch_data, raw.ch_names,
                        subject_id, "bci_iv_2a",
                        f"{subject_id}{session_type}",
                        label, label_code, sfreq,
                        start_sample / sfreq,
                        end_sample   / sfreq,
                        ingested_at
                    )
                    total_epochs  += 1
                    imagery_events.append(event)

                # Rest epochs between imagery events
                if len(imagery_events) > 0:
                    rest_epochs = extract_rest_epochs(
                        raw, np.array(imagery_events), sfreq
                    )
                    for epoch_data, start_sec, end_sec in rest_epochs:
                        insert_epoch(
                            session, prepared,
                            epoch_data, raw.ch_names,
                            subject_id, "bci_iv_2a",
                            f"{subject_id}{session_type}",
                            "rest", 2, sfreq,
                            start_sec, end_sec,
                            ingested_at
                        )
                        total_epochs += 1

                raw.close()

            except Exception as e:
                print(f"  Failed {subject_id}{session_type}: {e}")
                failed_files += 1

        save_progress(subject_id)

    print(f"  BCI IV-2a done: {total_epochs} epochs, {failed_files} failed files")
    return total_epochs


# ── Verify ────────────────────────────────────────────────
def verify(session):
    print("\nVerification:")
    session.default_timeout = 300

    sample = session.execute(
        "SELECT * FROM eeg_epochs LIMIT 1"
    ).one()

    if sample:
        print(f"  epoch_id:        {sample.epoch_id}")
        print(f"  subject_id:      {sample.subject_id}")
        print(f"  dataset:         {sample.dataset}")
        print(f"  label:           {sample.label}")
        print(f"  label_code:      {sample.label_code}")
        print(f"  channel:         {sample.channel}")
        print(f"  channel_idx:     {sample.channel_idx}")
        print(f"  n_samples:       {sample.n_samples}")
        print(f"  sampling_rate:   {sample.sampling_rate}")
        print(f"  amplitudes[:5]:  {sample.amplitudes[:5]}")
        print(f"  len(amplitudes): {len(sample.amplitudes)}")
    else:
        print("  No rows found")

    try:
        result = session.execute(
            "SELECT COUNT(*) FROM eeg_epochs"
        ).one()
        print(f"\n  Total rows: {result[0]:,}")
    except Exception:
        print("\n  COUNT(*) timed out — data is present, table is large")
        print("  Run later: SELECT COUNT(*) FROM projectcerebro.eeg_epochs;")


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("ProjectCerebro -- Cassandra EEG Ingestion v2")
    print("=" * 50)
    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Epoch length:       {EPOCH_LENGTH_SEC}s")
    print(f"Baseline:           {BASELINE_SEC}s")
    print(f"Post-event:         {POST_EVENT_SEC}s")
    print(f"Classes:            left(0), right(1), rest(2)")
    print(f"Storage:            list<float> per channel")
    print(f"Rows per epoch:     64 (PhysioNet) / 25 (BCI)")
    print(f"Insert concurrency: {INSERT_CONCURRENCY}")
    print(f"Cassandra timeout:  {CASSANDRA_TIMEOUT}s")
    print(f"Test one subject:   {TEST_ONE_SUBJECT}")
    print(f"Debug mode:         {DEBUG}")
    print("=" * 50)

    ensure_paths_exist()

    cluster, session = connect_cassandra()

    try:
        setup_schema(session)
        prepared = get_prepared_statement(session)

        # physionet_count = ingest_physionet(session, prepared)
        bci_count       = ingest_bci(session, prepared)

        verify(session)

        print(f"\nTotal epochs ingested: {physionet_count + bci_count:,}")
        print("Cassandra ingestion complete")

    finally:
        cluster.shutdown()