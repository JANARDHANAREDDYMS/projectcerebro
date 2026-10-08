# ProjectCerebro — Local Demo Setup

This is the runbook for starting the ProjectCerebro EEG/BCI dashboard on a new machine. It describes the current local demo architecture and the services that are required.

The demo replays recorded EEG and preprocessed EEG epochs using a shared playback clock. This makes the dashboard look and behave like a real-time system while remaining deterministic and repeatable. It is not a live EEG headset connection.

## Project demo video

Watch the ProjectCerebro dashboard demonstration on YouTube:

[ProjectCerebro demo](https://youtu.be/WWUBNFFCZw0)

## 1. What the demo shows

ProjectCerebro takes a five-channel EEG signal and predicts a motor-imagery class:

- `LEFT` — imagined left-hand movement
- `RIGHT` — imagined right-hand movement
- `REST` — rest/no movement

The dashboard has three coordinated views:

```text
Recorded FIF/cache ──> Stream 1: raw continuous EEG
                         │
                         └── shared playback time t

Delta/Parquet epochs ──> Stream 2: preprocessed 4-second epoch window

Kafka raw-eeg ──> agent consumer ──> LangGraph ──> Stream 3: predictions/events
                                      │
                                      └── FastAPI SSE endpoint
```

## 2. Prerequisites

Install or start the following before running the demo:

- Python 3.11
- Node.js/npm
- Docker Desktop with Docker Compose
- The repository data/artifacts checked out locally

Run commands from the repository root:

```bash
cd /Users/janardhanareddyms/Documents/Tandon/courses/bigdata/projectcerebro
```

### Use the correct Python environment

Use `cerebro_env_clean`. The older `cerebro_env` and `.venv` environments have previously contained incomplete/corrupted Uvicorn or pip installations. Calling `python3` or `uvicorn` without an explicit environment can silently use the wrong installation.

The safest form is to use the environment's interpreter directly:

```bash
cerebro_env_clean/bin/python --version
cerebro_env_clean/bin/python -c "import fastapi, uvicorn, torch, numpy, pandas, kafka, langgraph; print('Python dependencies OK')"
```

Activation is also supported:

```bash
source cerebro_env_clean/bin/activate
which python
python --version
```

After activation, `python -m uvicorn` should refer to `cerebro_env_clean`. If it does not, use the explicit `cerebro_env_clean/bin/python` commands below.

## 3. Start the services

### Terminal 1 — start the Docker services needed for the demo

The prediction path needs Kafka and ZooKeeper. MLflow is useful for the model-tracking portion of the architecture but is not required to draw Streams 1 and 2.

```bash
docker compose up -d zookeeper kafka mlflow
docker compose ps
```

Expected important ports:

| Service | Port | Purpose |
|---|---:|---|
| Kafka | `9092` | EEG epoch messages on `raw-eeg` |
| ZooKeeper | `2181` | Kafka coordination |
| MLflow | `5001` | Experiment/model tracking UI |
| FastAPI | `8001` | Dashboard API, SSE, inference endpoints |
| Vite dashboard | `14173` | Browser UI |

The Compose file also defines MongoDB, Cassandra, Redis, and PostgreSQL/pgvector. They support the broader architecture, but they are not all required to replay the local dashboard. Some local code paths fall back to files when optional databases are unavailable. If the optional containers are restarting, do not treat that alone as a dashboard failure; check the FastAPI health endpoint and the Kafka container first.

If you want the complete development stack instead:

```bash
docker compose up -d
```

Then inspect failures with:

```bash
docker compose ps
docker compose logs --tail=100 kafka
```

### Terminal 2 — start FastAPI

```bash
cerebro_env_clean/bin/python -m uvicorn serving.api:app \
  --host 127.0.0.1 --port 8001
```

A healthy startup includes messages similar to:

```text
STARTUP: Dashboard served at /dashboard
STARTUP: Stream cache subjects available: [...]
Uvicorn running on http://127.0.0.1:8001
```

Verify it from another terminal:

```bash
curl -sS http://127.0.0.1:8001/health
```

The API also serves the built dashboard at `http://127.0.0.1:8001/dashboard` when the frontend build is available, but the development workflow below uses Vite.

### Terminal 3 — start the React/Vite dashboard

```bash
cd cerebro-dashboard
npm install                 # first run only, or after package-lock changes
npm run dev -- --host 127.0.0.1 --port 14173
```

Open:

```text
http://127.0.0.1:14173
```

The Vite proxy forwards `/stream`, `/health`, `/predict`, and `/calibrate` requests to FastAPI on port `8001`. Therefore, the backend must be running before testing the browser UI.

## 4. Run the demo

1. Open `http://127.0.0.1:14173`.
2. Select the subject with an available stream cache, usually `A01` or `A08` depending on the local data.
3. Click `RESTART STREAM`/ `RUN STREAM`.
4. Wait for the epoch snapshot to load. The app loads Stream 2's epoch metadata before starting Stream 1 so both views can use the same timeline origin.
5. Confirm that Stream 1's time and Stream 2's `timeline` time advance together.
6. Click `START PREDICTION` after the playback clock is live.
7. Watch Stream 3 for prediction events, confidence, quality, calibration, and alerts.

For a clean demo, start Kafka before clicking `START PREDICTION`. If Kafka is unavailable, Streams 1 and 2 can still replay, but Stream 3 cannot receive the normal producer/consumer events.

## 5. Important timing behavior

There are two different clocks to keep straight:

1. **Stream/playback time** — the EEG position in the recording, for example `t = 735.3s`.
2. **Processing/arrival time** — when the browser, Kafka consumer, or model finishes handling a payload.

The dashboard uses Stream 1 as the live playback clock. Stream 2 maps the same clock to the epoch's original `epoch_start_sec` and `epoch_end_sec`. Stream 3 reports the source epoch time while the result is delivered asynchronously.

That means a prediction row may arrive later than the epoch's visible interval if the processing path is slow, but its displayed stream timestamp should still identify the correct epoch. This distinction is important when demonstrating the difference between synchronized visualization and end-to-end inference latency.

## 6. Data and service map

| Component | Role in ProjectCerebro | Required for the local replay demo? |
|---|---|---|
| FIF-derived cache | Stream 1 raw replay | Yes |
| Delta/Parquet epoch dataset | Stream 2 preprocessed epochs | Yes |
| FastAPI | Loads caches, serves dashboard/API/SSE, inference endpoints | Yes |
| React/Vite | Browser dashboard | Yes |
| Kafka + ZooKeeper | Producer/consumer transport for Stream 3 | Yes for normal prediction flow |
| LangGraph agent | Quality, prediction, retrieval/alerts pipeline | Yes for agent output |
| MLflow | Tracks experiments and model artifacts | Optional during replay |
| Cassandra | Ingestion/storage path for EEG epoch records; its commit log is Cassandra's recovery log | Optional during replay |
| MongoDB | Session/metadata/prediction persistence path | Optional; local fallback exists |
| Redis | Real-time cache/state path | Optional for the local demo |
| PostgreSQL/pgvector | Embedding similarity-search path | Optional for the local replay |

Cassandra's `commitlog` is not the source Stream 2 reads from during this dashboard replay. It is an internal write-ahead/recovery log used by Cassandra. Do not delete it while Cassandra is running. If disk usage becomes a concern, stop Cassandra first and inspect the mounted `volumes/cassandra` directory before cleaning anything.

## 7. Troubleshooting

### `address already in use` on port 8001

Another FastAPI process is already listening. Find it:

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
```

Either reuse that healthy server or stop it with its terminal's `Ctrl+C`, then start the clean environment command again. Do not start a second API on 8002 unless you also update the Vite proxy.

### Uvicorn errors such as `uvicorn._compat` or `uvicorn.middleware.proxy_headers`

The wrong/broken environment is being used. Check:

```bash
which python
python -c "import sys, uvicorn; print(sys.executable); print(uvicorn.__file__)"
```

Then bypass shell activation and use:

```bash
cerebro_env_clean/bin/python -m uvicorn serving.api:app --host 127.0.0.1 --port 8001
```

### Stream 2 says no valid Parquet files or never loads

Confirm that the epoch dataset exists:

```bash
find delta_lake/epochs_mi_v1_ch5_sr128_bp8_30 -type f | head
```

Then check the FastAPI terminal for the exact subject/path error. Select a subject that exists in the local dataset instead of assuming every subject is available.

### Stream 1 says no stream cache

The backend startup line lists cached subjects. If the requested subject is absent, choose an available subject or rebuild the FIF-derived cache using the data-preparation tooling documented in `docs/week1_setup.md`. Do not point the dashboard at a random FIF file without matching the expected five channels and 128 Hz sampling rate.

### Stream 3 is disconnected or `START PREDICTION` stays on `STARTING...`

Check Kafka and the API logs:

```bash
docker compose ps kafka zookeeper
docker compose logs --tail=100 kafka
```

Also confirm that port `9092` is reachable and that the FastAPI process is still running. MongoDB/PostgreSQL/Redis restart loops are separate from the Kafka transport; they can affect persistence but should not be mistaken for a healthy Kafka check.

If the dashboard says prediction started but the prediction count remains zero, inspect the per-session consumer log:

```bash
ls -t artifacts/logs/dashboard_consumer_*.log | head -1
tail -50 artifacts/logs/dashboard_consumer_*.log
```

The consumer must import LangGraph. If the log contains:

```text
ModuleNotFoundError: No module named 'langgraph'
```

the Kafka producer may still be sending epochs, but the consumer has already exited and no inference can happen. Install the missing package into the same environment used by FastAPI:

```bash
cerebro_env_clean/bin/python -m pip install langgraph
```

Then stop and restart FastAPI, reload the dashboard, and start a fresh prediction session. Verify the package before starting the demo:

```bash
cerebro_env_clean/bin/python -c "import langgraph; print(langgraph.__file__)"
```

The screenshot may show `mongo=local-log`; that is an expected fallback when MongoDB is unavailable and does not, by itself, prevent Stream 3 events from appearing.

### The dashboard page loads but API calls fail

The frontend is probably running while FastAPI is not, or the API is on a different port. The expected pair is:

```text
FastAPI  http://127.0.0.1:8001
Vite     http://127.0.0.1:14173
```

Run `curl -sS http://127.0.0.1:8001/health` and restart the backend if it fails.

## 8. Stop everything after the demo

Stop the frontend and backend with `Ctrl+C` in their terminals. Stop Docker services when they are no longer needed:

```bash
docker compose stop
```

Use `docker compose down` only when you intentionally want Compose to remove the containers from this stack. The bind-mounted data under `volumes/` and `mlruns/` is separate, but do not delete those directories as part of routine shutdown.

## 9. Quick start checklist

```bash
# Terminal 1
docker compose up -d zookeeper kafka mlflow

# Terminal 2, from the repository root
cerebro_env_clean/bin/python -m uvicorn serving.api:app --host 127.0.0.1 --port 8001

# Terminal 3
cd cerebro-dashboard
npm run dev -- --host 127.0.0.1 --port 14173

# Browser
open http://127.0.0.1:14173
```

Then run the stream, wait for the shared playback clock, start prediction, and explain the three streams using the sections above.
