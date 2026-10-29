# Standalone Real-Time CCTV Activity Intelligence Backend (IURIS-Compatible)

A modular, production-oriented computer vision and activity intelligence backend for live RTSP cameras, USB/integrated webcams, and prerecorded video files. The backend performs single-pass 17-keypoint COCO human pose estimation, per-camera ByteTrack multi-person tracking, temporal activity recognition (`STANDING`, `WALKING`, `RUNNING`, `SITTING`, `LYING`, `FALLING`, `UNKNOWN`), deduplicated camera-level activity summaries, independent configurable risk assessment, versioned `/v1/*` REST API exposure, and optional authenticated outbound event delivery compatible with **IURIS**.

## Key Capabilities & Architectural Boundaries

- **Standalone Execution**: Runs independently of IURIS and requires zero changes to IURIS source code.
- **Decoupled Pipeline Stages**: Explicit protocol interfaces (`src/activity_system/interfaces.py`) separate input stream adapters (`WebcamStreamAdapter`, `RtspStreamAdapter`, `VideoFileStreamAdapter`, `StreamManager`), person detection, pose estimation (`OnnxPoseBackend`, `UltralyticsPoseBackend`, `SyntheticPoseBackend`), ByteTrack tracking (`PersonTracker` + `TrackStateManager`), temporal activity classification (`ActivityClassifier`), camera summary aggregation (`CameraSummaryAggregator`), risk assessment (`RiskAssessmentEngine`), bounded event storage (`BoundedEventStore`), and IURIS outbound delivery (`src/activity_system/iuris/`).
- **Responsible Use & Privacy**: Does not perform facial recognition, biometric identification, cross-camera re-identification, or intent inference. RTSP credentials (`rtsp://***:***@host`) are redacted across all logs, exceptions, and API payloads.

## Versioned FastAPI Endpoints (`src/activity_system/api.py`)

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/health` | `GET` | Process liveness check |
| `/ready` | `GET` | Readiness of required models and camera pipeline (`200` or `503`) |
| `/v1/cameras` | `GET` | Camera list, connection status, frame timestamps, dimensions, and processing state |
| `/v1/tracks` | `GET` | Currently active tracked people (paginated & filterable) |
| `/v1/activities` | `GET` | Latest activity observations (`source=live`) or bounded history (`source=history`) |
| `/v1/camera-summaries` | `GET` | Deduplicated camera-level activity distributions and percentages |
| `/v1/risk-events` | `GET` | Risk-event history (`UNUSUAL_ACTIVITY_CONCENTRATION`, `POSSIBLE_FALL_DETECTED`, `PROLONGED_IMMOBILITY`, `ZONE_RESTRICTED_ACTIVITY`) |
| `/metrics` | `GET` | Operational JSON telemetry and IURIS delivery counters |
| `/metrics/prometheus` | `GET` | Prometheus text exposition format |

*(Legacy endpoints `/cameras`, `/tracks`, `/activities`, `/frame.jpg`, `/stream.mjpg`, and `/` are also preserved.)*

## Installation & Quick Start

### 1. Install Dependencies (`requirements.txt` or `pyproject.toml`)
```bash
pip install -r requirements.txt
pip install --no-deps -e .
```
*(Or via editable extras: `pip install -e ".[api]"`)*

### 2. Run Complete Unit & Integration Test Suite (`55` Tests)
```bash
PYTHONPATH=src:. python3 -B -m unittest discover -s tests -v
```

### 3. Run Standalone FastAPI Server
```bash
PYTHONPATH=src:. AI_BACKEND_PORT=8081 python3 scripts/run_server.py --config config/default.yaml
```

### 4. Run End-to-End Video CLI & Hardware Benchmarks
```bash
PYTHONPATH=src:. python3 -B scripts/run_video.py --config config/default.yaml --max-frames 45
PYTHONPATH=src:. python3 -B scripts/benchmark.py
```

## Documentation Index

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System architecture, stage protocols, state separation, and summary calculation formulas.
- [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) — Versioned REST endpoints, query parameters, pagination, error formats, and JSON event schemas.
- [`docs/IURIS_COMPATIBILITY.md`](docs/IURIS_COMPATIBILITY.md) — IURIS outbound delivery, authentication, retries, idempotency, and future IURIS integration guide.
- [`docs/MODEL_STRATEGY.md`](docs/MODEL_STRATEGY.md) — Pose estimation backends, ByteTrack association, temporal kinematics, and `FALLING` vs. `LYING` distinction.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — CPU/GPU execution, model weights setup, Docker networking, observability, and shutdown procedures.
- [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) — Phase-by-phase implementation log, test verification results, and known limitations.
