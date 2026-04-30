# ProjectCerebro -- Cassandra Failed Row Retry Script
# Retries specific failed channel rows from .failed_rows.txt
# Run from project root:
#   python3 scripts/retry_failed_rows.py

from pathlib import Path
from datetime import datetime

import mne
from cassandra.cluster import Cluster
from cassandra.concurrent import execute_concurrent_with_args

mne.set_log_level("WARNING")

# ── Project Paths ─────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
PHYSIONET_BASE = PROJECT_ROOT / "data" / "physionet" / "physionet.org" / "files" / "eegmmidb" / "1.0.0"
BCI_BASE       = PROJECT_ROOT / "data" / "bci_iv_2a" / "BCICIV_2a_gdf"
FAILED_LOG     = PROJECT_ROOT / "scripts" / ".failed_rows.txt"

# ── Config ────────────────────────────────────────────────
CASSANDRA_HOST     = "localhost"
CASSANDRA_PORT     = 9042
KEYSPACE           = "projectcerebro"
CASSANDRA_TIMEOUT  = 120
INSERT_CONCURRENCY = 50
MAX_RETRIES        = 3

BASELINE_SEC   = 1.0
POST_EVENT_SEC = 3.0

LABEL_MAP = {0: "left", 1: "right", 2: "rest"}

# ── Known failed epochs (from ingestion output) ───────────
# These were reported as having failed rows
# Format: epoch_id as printed during ingestion
KNOWN_FAILED_EPOCHS = [
    "physionet|S030|S030R06|2|7200|11200",
    "physionet|S061|S061R14|2|65300|69300",
    "physionet|S067|S067R12|1|93300|97300",
    "physionet|S081|S081R08|1|11300|15300",
]


# ── Cassandra Connection ──────────────────────────────────
def connect_cassandra():
    print("Connecting to Cassandra...")
    cluster = Cluster(
        [CASSANDRA_HOST],
        port=CASSANDRA_PORT
    )
    session = cluster.connect()
    session.set_keyspace(KEYSPACE)
    session.default_timeout = CASSANDRA_TIMEOUT
    print(f"  Connected — timeout: {CASSANDRA_TIMEOUT}s")
    return cluster, session


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


# ── Parse epoch_id ────────────────────────────────────────
def parse_epoch_id(epoch_id):
    # Format: dataset|subject_id|run_id|label_code|start_ms|end_ms
    parts      = epoch_id.split("|")
    dataset    = parts[0]
    subject_id = parts[1]
    run_id     = parts[2]
    label_code = int(parts[3])
    start_sec  = int(parts[4]) / 1000.0
    end_sec    = int(parts[5]) / 1000.0
    return dataset, subject_id, run_id, label_code, start_sec, end_sec


# ── Load Source File ──────────────────────────────────────
def load_raw(dataset, subject_id, run_id):
    if dataset == "physionet":
        file_path = PHYSIONET_BASE / subject_id / f"{run_id}.edf"
        return mne.io.read_raw_edf(str(file_path), preload=True)
    else:
        file_path = BCI_BASE / f"{run_id}.gdf"
        return mne.io.read_raw_gdf(str(file_path), preload=True)


# ── Check Existing Channels ───────────────────────────────
def get_existing_channel_indices(session, epoch_id):
    # Find which channel_idx rows already exist
    rows = session.execute(
        "SELECT channel_idx FROM eeg_epochs WHERE epoch_id = %s",
        [epoch_id]
    )
    return set(row.channel_idx for row in rows)


# ── Retry One Epoch ───────────────────────────────────────
def retry_epoch(session, prepared, epoch_id):
    print(f"\n  Retrying: {epoch_id}")

    dataset, subject_id, run_id, label_code, start_sec, end_sec = \
        parse_epoch_id(epoch_id)

    label       = LABEL_MAP[label_code]
    ingested_at = datetime.utcnow()

    # Load source file
    try:
        raw   = load_raw(dataset, subject_id, run_id)
        sfreq = raw.info["sfreq"]
    except Exception as e:
        print(f"  Could not load source file: {e}")
        return False

    # Extract epoch data
    start_sample = int(start_sec * sfreq)
    end_sample   = int(end_sec   * sfreq)
    epoch_data   = raw.get_data(start=start_sample, stop=end_sample)
    n_samples    = epoch_data.shape[1]
    ch_names     = raw.ch_names
    raw.close()

    # Find which channels are missing
    existing_idx = get_existing_channel_indices(session, epoch_id)
    all_idx      = set(range(len(ch_names)))
    missing_idx  = all_idx - existing_idx

    if not missing_idx:
        print(f"  All channels already present — nothing to retry")
        return True

    print(f"  Missing {len(missing_idx)} channels: {sorted(missing_idx)}")
    print(f"  Existing {len(existing_idx)} channels already present")

    # Build rows for missing channels only
    rows = []
    for ch_idx in sorted(missing_idx):
        channel    = ch_names[ch_idx]
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
            float(start_sec),
            float(end_sec),
            ingested_at
        ))

    # Retry with multiple attempts
    attempt = 0
    while rows and attempt < MAX_RETRIES:
        attempt += 1
        print(f"  Attempt {attempt}/{MAX_RETRIES} — inserting {len(rows)} rows")

        results = execute_concurrent_with_args(
            session,
            prepared,
            rows,
            concurrency=INSERT_CONCURRENCY,
            raise_on_first_error=False
        )

        failed_rows = [
            rows[i] for i, (success, _) in enumerate(results)
            if not success
        ]

        if not failed_rows:
            print(f"  All rows inserted successfully on attempt {attempt}")
            return True

        print(f"  {len(failed_rows)} rows still failing")
        rows = failed_rows

    print(f"  Failed after {MAX_RETRIES} attempts")
    return False


# ── Verify Epoch Completeness ─────────────────────────────
def verify_epoch(session, epoch_id, expected_channels):
    rows = session.execute(
        "SELECT channel_idx FROM eeg_epochs WHERE epoch_id = %s",
        [epoch_id]
    )
    present = set(row.channel_idx for row in rows)
    missing = set(range(expected_channels)) - present

    if missing:
        print(f"  Still missing channels: {sorted(missing)}")
        return False
    else:
        print(f"  All {expected_channels} channels present")
        return True


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("ProjectCerebro -- Failed Row Retry")
    print("=" * 45)

    # Load additional failed epochs from log file if it exists
    failed_epochs = list(KNOWN_FAILED_EPOCHS)

    if FAILED_LOG.exists():
        print(f"Loading additional failures from {FAILED_LOG}")
        with open(FAILED_LOG) as f:
            for line in f:
                line = line.strip()
                if line and line not in failed_epochs:
                    epoch_id = line.split("|channel_idx")[0]
                    if epoch_id not in failed_epochs:
                        failed_epochs.append(epoch_id)

    print(f"Epochs to retry: {len(failed_epochs)}")
    for e in failed_epochs:
        print(f"  {e}")
    print("=" * 45)

    cluster, session = connect_cassandra()
    prepared = get_prepared_statement(session)

    success_count = 0
    fail_count    = 0

    try:
        for epoch_id in failed_epochs:
            success = retry_epoch(session, prepared, epoch_id)

            if success:
                success_count += 1
                # Verify completeness
                dataset, subject_id, run_id, label_code, \
                    start_sec, end_sec = parse_epoch_id(epoch_id)

                if dataset == "physionet":
                    expected_channels = 64
                else:
                    expected_channels = 25

                verify_epoch(session, epoch_id, expected_channels)
            else:
                fail_count += 1

        print(f"\nRetry complete:")
        print(f"  Succeeded: {success_count}/{len(failed_epochs)}")
        print(f"  Failed:    {fail_count}/{len(failed_epochs)}")

        if fail_count == 0 and FAILED_LOG.exists():
            FAILED_LOG.unlink()
            print("  Failed rows log cleared")

    finally:
        cluster.shutdown()