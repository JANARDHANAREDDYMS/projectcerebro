"""ProjectCerebro Kafka EEG Producer.

Reads preprocessed BCI IV-2a epochs from Delta Lake and publishes them to the
Kafka topic ``raw-eeg``. This simulates real-time EEG hardware streaming.

Usage:
    cerebro_env/bin/python scripts/kafka_eeg_producer.py
    cerebro_env/bin/python scripts/kafka_eeg_producer.py --subject A09 --interval 1.0
    cerebro_env/bin/python scripts/kafka_eeg_producer.py --subject A09 --interval 0.5 --loop
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from ml_core.data import read_epochs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ProjectCerebro EEG Kafka Producer")
    parser.add_argument("--delta-path", default="delta_lake/epochs_mi_v1_ch5_sr128_bp8_30")
    parser.add_argument("--filter-version", default="bp_8_30_v1")
    parser.add_argument("--subject", default="A09", help="BCI subject ID to stream (A01-A09)")
    parser.add_argument("--session-id", default=None, help="Session ID (auto-generated if not provided)")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between epochs. 4.0=real-time, 1.0=4x faster, 0.5=8x faster",
    )
    parser.add_argument("--kafka-host", default="localhost:9092")
    parser.add_argument("--topic", default="raw-eeg")
    parser.add_argument("--loop", action="store_true", help="Loop through epochs repeatedly until Ctrl+C")
    parser.add_argument(
        "--timeline-sync",
        action="store_true",
        help="Publish epochs according to their epoch_start_sec timeline instead of a fixed interval.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="Recording time in seconds where timeline-synchronized streaming starts.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Timeline speed multiplier for --timeline-sync. 1.0 means real time.",
    )
    return parser.parse_args()


def main() -> None:
    """Stream EEG epochs to Kafka."""
    args = parse_args()
    session_id = args.session_id or f"session_{str(uuid.uuid4())[:8]}"

    print(f"Loading epochs for subject {args.subject}...")
    df = read_epochs(
        args.delta_path,
        filter_version=args.filter_version,
        datasets=["bci_iv_2a"],
    )
    subj_df = df[df.subject_id == args.subject].reset_index(drop=True)
    if "epoch_start_sec" in subj_df.columns:
        subj_df = subj_df.sort_values("epoch_start_sec", kind="stable").reset_index(drop=True)
    if args.timeline_sync and args.start_sec is not None and "epoch_start_sec" in subj_df.columns:
        subj_df = subj_df[subj_df.epoch_start_sec >= args.start_sec].reset_index(drop=True)

    if len(subj_df) == 0:
        print(f"ERROR: No epochs found for subject {args.subject}")
        print(f"Available subjects: {sorted(df.subject_id.unique())}")
        sys.exit(1)

    label_counts = subj_df.label_name.value_counts().to_dict()
    if args.timeline_sync and "epoch_start_sec" in subj_df.columns and len(subj_df) > 0:
        timeline_start = float(args.start_sec) if args.start_sec is not None else float(subj_df.epoch_start_sec.iloc[0])
        timeline_end = float(subj_df.epoch_start_sec.iloc[-1])
        est_seconds = max(0.0, timeline_end - timeline_start) / max(args.time_scale, 1e-6)
    else:
        timeline_start = None
        est_seconds = len(subj_df) * args.interval

    print(f"\n{'=' * 55}")
    print("  ProjectCerebro EEG Stream Producer")
    print(f"{'=' * 55}")
    print(f"  Subject:     {args.subject}")
    print(f"  Epochs:      {len(subj_df)}  {label_counts}")
    print(f"  Interval:    {args.interval}s per epoch")
    print(f"  Timeline:    {args.timeline_sync}")
    if args.timeline_sync:
        print(f"  Start sec:   {timeline_start:.3f}s")
        print(f"  Time scale:  {args.time_scale}x")
    print(f"  Duration:    {est_seconds:.0f}s ({est_seconds / 60:.1f} min)")
    print(f"  Session:     {session_id}")
    print(f"  Topic:       {args.topic}")
    print(f"  Kafka:       {args.kafka_host}")
    print(f"  Loop:        {args.loop}")
    print(f"{'=' * 55}\n")

    print("Connecting to Kafka...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=args.kafka_host,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            acks="all",
            retries=3,
        )
        print("Connected.\n")
    except NoBrokersAvailable:
        print(f"ERROR: Cannot connect to Kafka at {args.kafka_host}")
        print("Is Kafka running? Check: docker compose ps")
        sys.exit(1)

    iteration = 0
    total_sent = 0

    try:
        while True:
            iteration += 1
            if args.loop:
                print(f"--- Loop {iteration} ---")

            loop_started_at = time.monotonic()
            for _, row in subj_df.iterrows():
                if args.timeline_sync:
                    epoch_start_sec = float(row.get("epoch_start_sec", 0.0))
                    base_sec = timeline_start if timeline_start is not None else epoch_start_sec
                    target_elapsed = max(0.0, epoch_start_sec - base_sec) / max(args.time_scale, 1e-6)
                    sleep_for = loop_started_at + target_elapsed - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

                features = row["features"]
                if hasattr(features, "tolist"):
                    features = features.tolist()

                message = {
                    "epoch_id": str(uuid.uuid4()),
                    "subject_id": str(row["subject_id"]),
                    "session_id": session_id,
                    "features": features,
                    "label_code": int(row["label_code"]),
                    "label_name": str(row.get("label_name", row["label_code"])),
                    "epoch_start_sec": float(row.get("epoch_start_sec", 0.0)),
                    "epoch_end_sec": float(row.get("epoch_end_sec", 0.0)),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "iteration": iteration,
                }

                producer.send(args.topic, value=message)
                total_sent += 1

                if total_sent % 10 == 0 or total_sent == 1:
                    print(
                        f"  [{total_sent:>4}] "
                        f"label={message['label_name']:<6} "
                        f"epoch={message['epoch_id'][:8]} "
                        f"session={session_id}"
                    )

                if not args.timeline_sync:
                    time.sleep(args.interval)

            producer.flush()
            print(f"\n  Completed loop {iteration}. Total sent: {total_sent}\n")

            if not args.loop:
                break

    except KeyboardInterrupt:
        print(f"\nProducer stopped. Total sent: {total_sent}")
    finally:
        producer.flush()
        producer.close()
        print("Producer closed cleanly.")


if __name__ == "__main__":
    main()
