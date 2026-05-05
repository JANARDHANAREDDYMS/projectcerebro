"""Kafka consumer helper for future streaming integration."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

KAFKA_BOOTSTRAP = os.getenv("CEREBRO_KAFKA_BOOTSTRAP", "localhost:9092")


def iter_json_messages(topic: str) -> Iterator[dict[str, Any]]:
    """Yield JSON messages from Kafka when kafka-python is installed."""
    try:
        from kafka import KafkaConsumer
    except Exception as exc:
        raise RuntimeError("kafka-python is required for Kafka streaming.") from exc

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda data: json.loads(data.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )
    for message in consumer:
        yield message.value

