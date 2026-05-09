"""Create all required Kafka topics for ProjectCerebro.

Run once before starting the streaming pipeline.

Usage:
    cerebro_env/bin/python scripts/setup_kafka_topics.py
"""
from __future__ import annotations

from kafka import KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

KAFKA_HOST = "localhost:9092"

TOPICS = [
    NewTopic(name="raw-eeg", num_partitions=1, replication_factor=1),
    NewTopic(name="processed-epochs", num_partitions=1, replication_factor=1),
    NewTopic(name="predictions", num_partitions=1, replication_factor=1),
]


def main() -> None:
    """Create Kafka topics and print broker topic list."""
    print(f"Connecting to Kafka at {KAFKA_HOST}...")
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_HOST, client_id="cerebro-admin")

    for topic in TOPICS:
        try:
            admin.create_topics([topic])
            print(f"  Created topic: {topic.name}")
        except TopicAlreadyExistsError:
            print(f"  Topic already exists: {topic.name}")
        except Exception as exc:
            print(f"  Error creating {topic.name}: {exc}")

    admin.close()
    print("\nAll topics ready.")

    consumer = KafkaConsumer(bootstrap_servers=KAFKA_HOST)
    existing = consumer.topics()
    consumer.close()
    print(f"Topics on broker: {sorted(existing)}")


if __name__ == "__main__":
    main()

