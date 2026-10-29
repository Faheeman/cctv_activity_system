# Operations, Deployment, and Observability Guide

## 1. Local Execution Instructions

### 1.1 Installation
```bash
# Install runtime dependencies from requirements.txt
pip install -r requirements.txt
pip install --no-deps -e .

# Alternative: Install via pyproject.toml extras
pip install -e ".[api]"

# Optional: Install Ultralytics PyTorch runtime if using .pt weights directly
pip install -e ".[api,yolo]"
```

### 1.2 Running the Standalone FastAPI Backend
```bash
# Run with default configuration on port 8081
AI_BACKEND_HOST=0.0.0.0 AI_BACKEND_PORT=8081 python3 scripts/run_server.py --config config/default.yaml

# Run with ONNX pose model on a live RTSP stream
PYTHONPATH=src:. python3 scripts/run_server.py \
  --config config/production_rtsp.yaml \
  --pose-backend onnx \
  --host 0.0.0.0 \
  --port 8081

# Run with local USB/integrated webcam (/dev/video0)
PYTHONPATH=src:. python3 scripts/run_server.py \
  --config config/live_webcam.yaml \
  --source-type webcam \
  --source-uri 0 \
  --pose-backend onnx
```

### 1.3 Running Batch Video Processing & Benchmarks
```bash
# Process a video file and emit annotated .avi + JSONL events
PYTHONPATH=src:. python3 -B scripts/run_video.py --config config/default.yaml --max-frames 45

# Run sequence-split evaluation and multi-profile hardware benchmarks
PYTHONPATH=src:. python3 -B scripts/benchmark.py
```

### 1.4 Running the Automated Test Suite
```bash
PYTHONPATH=src:. python3 -B -m unittest discover -s tests -v
```

## 2. CPU vs. Optional GPU Execution

- **CPU Execution (Default)**:
  - Set `AI_DEVICE=cpu` (or `pose.device: "cpu"` in YAML) and `pose.backend: "onnx"`.
  - Tune `pose.cpu_threads` and `OMP_NUM_THREADS=2` to match available physical cores.
  - For multi-camera CPU deployments, set `sample_every_n_frames: 2` and `resize_width: 640`, `resize_height: 480`.
- **Optional GPU Execution (CUDA / TensorRT)**:
  - Set `AI_DEVICE=auto` or `AI_DEVICE=cuda:0`.
  - When using `OnnxPoseBackend`, install `onnxruntime-gpu` so `CUDAExecutionProvider` is available; `OnnxPoseBackend` automatically detects and prioritizes `CUDAExecutionProvider` with fallback to `CPUExecutionProvider`.
  - When running in Docker with NVIDIA GPUs, pass `--gpus all` to `docker run` or add the `deploy.resources.reservations.devices` block in `docker-compose.yaml`.

## 3. Model Weights Setup (`models/`)

- Pre-exported `models/yolov8n-pose.onnx` and `models/yolov8n-pose.pt` are supported out of the box.
- To export a new ONNX model and generate a SHA-256 provenance manifest:
  ```bash
  PYTHONPATH=src:. python3 scripts/export_model.py --weights models/yolov8n-pose.pt --imgsz 640
  ```

## 4. Docker & Docker Compose Networking

### 4.1 Build and Run with Docker Compose
```bash
cp .env.example .env
docker compose up --build -d
```

### 4.2 Container Networking with IURIS
- **Never use `localhost` or `127.0.0.1` for `IURIS_BASE_URL` inside a container** unless both processes share the exact same network namespace (`network_mode: host`). Inside a standard Docker bridge network (`iuris-bridge`), `localhost` resolves to the `cctv-activity-engine` container itself.
- Always configure `IURIS_BASE_URL` using the target service's Docker DNS name (e.g., `IURIS_BASE_URL=http://iuris-backend:8080`) or host gateway (`http://host.docker.internal:8080`).

## 5. Observability, Health Checks, and Graceful Shutdown

- **Liveness (`GET /health`)**: Returns `200 OK` with `status: "ok"` (or `"degraded"` if a camera failed) and process uptime.
- **Readiness (`GET /ready`)**: Verifies `model_ready` and camera availability. Returns `200 OK` when ready or `503 Service Unavailable` when not ready.
- **Operational Metrics (`GET /metrics` and `GET /metrics/prometheus`)**:
  - Tracks `active_cameras`, `connected_cameras`, `disconnected_cameras`, `active_tracks`, `frames_received`, `frames_processed`, `frames_dropped`, `mean_inference_ms`, `mean_classification_ms`, `mean_end_to_end_ms`, `risk_events_generated`, `event_queue_depth`, `event_delivery_failures`, `model_ready`, `cpu_utilization_pct`, and `memory_rss_mb`.
- **Graceful Shutdown**:
  - Sending `SIGINT` or `SIGTERM` sets the global `stop_event`, releases all OpenCV `VideoCapture` handles (`StreamManager.close_all()`), drains any remaining queued outbound IURIS events to the durable dead-letter JSONL file (`outputs/iuris_dead_letter.jsonl`), and closes the `BoundedEventStore`.
