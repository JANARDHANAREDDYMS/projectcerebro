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

    if len(subj_df) == 0:
        print(f"ERROR: No epochs found for subject {args.subject}")
        print(f"Available subjects: {sorted(df.subject_id.unique())}")
        sys.exit(1)

    label_counts = subj_df.label_name.value_counts().to_dict()
    est_seconds = len(subj_df) * args.interval

    print(f"\n{'=' * 55}")
    print("  ProjectCerebro EEG Stream Producer")
    print(f"{'=' * 55}")
    print(f"  Subject:     {args.subject}")
    print(f"  Epochs:      {len(subj_df)}  {label_counts}")
    print(f"  Interval:    {args.interval}s per epoch")
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

            for _, row in subj_df.iterrows():
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

