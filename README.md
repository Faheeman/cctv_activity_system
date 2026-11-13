# Pose Estimation Activity System
A modular, production-oriented computer vision and activity intelligence backend for live RTSP cameras, USB/integrated webcams, and prerecorded video files. 
<img width="2591" height="1439" alt="Gemini_Generated_Image_c75orec75orec75o" src="https://github.com/user-attachments/assets/f542fdfc-1c40-42d4-bbc8-ac2d84ef690a" />

## The backend performs 
Single-pass 17-keypoint COCO human pose estimation, per-camera ByteTrack multi-person tracking, temporal activity recognition, deduplicated camera-level activity summaries, independent configurable risk assessment, versioned `/v1/*` REST API exposure.

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
