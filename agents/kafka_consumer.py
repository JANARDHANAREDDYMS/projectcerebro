"""ProjectCerebro Kafka EEG Agent Consumer.

Consumes EEG epochs from Kafka topic ``raw-eeg`` and runs each through the full
LangGraph agent pipeline.

Usage:
    cerebro_env/bin/python -m agents.kafka_consumer
    cerebro_env/bin/python -m agents.kafka_consumer --session-id demo_001
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from agents.graph import run_epoch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ProjectCerebro EEG Agent Consumer")
    parser.add_argument("--kafka-host", default="localhost:9092")
    parser.add_argument("--topic", default="raw-eeg")
    parser.add_argument("--group-id", default="cerebro-agents")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--filter-session-id", default=None)
    parser.add_argument("--auto-offset-reset", choices=["earliest", "latest"], default="latest")
    parser.add_argument("--timeout-ms", type=int, default=120000, help="Stop after N ms of no messages")
    return parser.parse_args()


def main() -> None:
    """Consume EEG epochs and run the agent graph."""
    args = parse_args()
    session_id = args.session_id or f"session_{str(uuid.uuid4())[:8]}"

    print(f"\n{'=' * 55}")
    print("  ProjectCerebro EEG Agent Consumer")
    print(f"{'=' * 55}")
    print(f"  Topic:    {args.topic}")
    print(f"  Kafka:    {args.kafka_host}")
    print(f"  Group:    {args.group_id}")
    print(f"  Session:  {session_id}")
    print("  Waiting for EEG epochs from Kafka...")
    print(f"{'=' * 55}\n")

    try:
        consumer = KafkaConsumer(
            args.topic,
            bootstrap_servers=args.kafka_host,
            group_id=args.group_id,
            auto_offset_reset=args.auto_offset_reset,
            value_deserializer=lambda message: json.loads(message.decode("utf-8")),
            consumer_timeout_ms=args.timeout_ms,
        )
    except NoBrokersAvailable:
        print(f"ERROR: Cannot connect to Kafka at {args.kafka_host}")
        print("Is Kafka running? Check: docker compose ps")
        sys.exit(1)

    n_processed = 0
    n_errors = 0
    n_calibrated_switch = False

    try:
        for message in consumer:
            try:
                epoch = message.value
                epoch_id = epoch.get("epoch_id", str(uuid.uuid4()))
                subject_id = epoch.get("subject_id", "unknown")
                label_name = epoch.get("label_name", "unknown")
                label_code = epoch.get("label_code")
                features = epoch.get("features", [])
                msg_session = epoch.get("session_id", session_id)

                if args.filter_session_id and msg_session != args.filter_session_id:
                    continue

                print(f"\n[{n_processed + 1:>4}] epoch={epoch_id[:8]} subject={subject_id} label={label_name}")

                if len(features) != 2560:
                    print(f"  WARNING: expected 2560 features got {len(features)}. Skipping.")
                    n_errors += 1
                    continue

                result = run_epoch(
                    features=features,
                    subject_id=subject_id,
                    session_id=msg_session,
                    true_label_code=int(label_code) if label_code is not None else None,
                    true_label_name=label_name if label_name != "unknown" else None,
                )
                n_processed += 1

                # Print full LangGraph state for each epoch
                import json as _json

                printable = {
                    k: v for k, v in result.items()
                    if k not in ("features", "embedding", "similar_trials")
                }

                print("\n=== AGENT STATE ===")
                print(_json.dumps(printable, indent=2, default=str))

                # Also print similar trials summary separately
                similar = result.get("similar_trials", [])
                if similar:
                    print(f"  similar_trials: {len(similar)} neighbors found")
                    for t in similar[:3]:
                        print(f"    subject={t.get('subject_id')} "
                              f"label={t.get('label_name')} "
                              f"similarity={t.get('similarity', 0):.3f}")
                print("===================\n")

                cal_status = result.get("calibration_status", "")
                if cal_status == "calibrated" and not n_calibrated_switch:
                    n_calibrated_switch = True
                    print(f"\n  *** CALIBRATION COMPLETE for {subject_id} ***")
                    print("  *** Future predictions use personalized model ***\n")

                print(
                    f"  quality={result['signal_quality']} "
                    f"score={result['quality_score']:.2f} | "
                    f"pred={result.get('label_name', '?')} "
                    f"conf={result.get('confidence') or 0:.3f} | "
                    f"model={result.get('model_used', '?')} | "
                    f"cal={cal_status} | "
                    f"severity={result['final_severity']}"
                )

            except Exception as exc:
                n_errors += 1
                import traceback

                print(f"  ERROR processing message: {exc}")
                print(traceback.format_exc())
                continue

    except KeyboardInterrupt:
        print("\nShutting down consumer...")
    finally:
        consumer.close()
        print("\nConsumer closed.")
        print(f"Processed: {n_processed}  Errors: {n_errors}")


if __name__ == "__main__":
    main()
