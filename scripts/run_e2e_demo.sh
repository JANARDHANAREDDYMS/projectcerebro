#!/bin/bash
set -e

echo ""
echo "========================================"
echo "  ProjectCerebro End-to-End Demo"
echo "========================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Checking FastAPI at localhost:8001..."
if ! curl -s --max-time 3 http://localhost:8001/health > /dev/null; then
  echo ""
  echo "ERROR: FastAPI is not running."
  echo "Start it first in another terminal:"
  echo ""
  echo "  cerebro_env/bin/uvicorn serving.api:app --port 8001 --reload"
  echo ""
  exit 1
fi
echo "FastAPI: OK"

echo "Checking Kafka at localhost:9092..."
if ! cerebro_env/bin/python - <<'PY' > /dev/null 2>&1
from kafka import KafkaConsumer

consumer = KafkaConsumer(bootstrap_servers="localhost:9092")
consumer.topics()
consumer.close()
PY
then
  echo "ERROR: Kafka not reachable."
  echo "Run: docker compose up -d kafka"
  exit 1
fi
echo "Kafka: OK"

echo "Setting up Kafka topics..."
cerebro_env/bin/python scripts/setup_kafka_topics.py

SESSION="demo_$(date +%Y%m%d_%H%M%S)"
SUBJECT="${1:-A09}"
INTERVAL="${2:-1.0}"

echo ""
echo "Session ID: $SESSION"
echo "Subject:    $SUBJECT"
echo "Interval:   ${INTERVAL}s per epoch"
echo ""

echo "Starting LangGraph Agent Consumer..."
cerebro_env/bin/python -m agents.kafka_consumer \
  --session-id "$SESSION" \
  --topic raw-eeg &
CONSUMER_PID=$!
echo "Consumer PID: $CONSUMER_PID"
sleep 3

echo ""
echo "Starting EEG Stream Producer..."
echo "Streaming subject $SUBJECT -> Kafka -> Agents"
echo "Press Ctrl+C to stop"
echo ""

cerebro_env/bin/python scripts/kafka_eeg_producer.py \
  --subject "$SUBJECT" \
  --session-id "$SESSION" \
  --interval "$INTERVAL"

echo ""
echo "Producer finished. Letting consumer process queued messages..."
sleep "${DRAIN_SECONDS:-30}"

echo ""
echo "Stopping consumer..."
kill "$CONSUMER_PID" 2>/dev/null || true
wait "$CONSUMER_PID" 2>/dev/null || true

echo ""
echo "========================================"
echo "  Demo complete."
echo "  Session: $SESSION"
echo "  MLflow:  http://localhost:5001"
echo "  MongoDB: check db.predictions"
echo "========================================"
