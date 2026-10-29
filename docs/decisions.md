# Architectural Decision Records (ADR)

## ADR-001: Minimal Modular Monolith with Single Inference Worker

### Context
The system must ingest live RTSP/webcam/file video streams, estimate human poses, track multiple individuals, classify physical activities over time, and optionally expose state via HTTP on modest hardware without operational bloat.

### Alternatives Considered
1. **Microservice Architecture with Kafka/Redis**: Separate containers for ingestion, pose inference, tracking, and classification communicating over message queues.
2. **Unbounded Multi-Threaded Per-Camera Inference**: Spawning a separate model instance and inference thread per camera stream.
3. **Minimal Modular Monolith with Shared Single Inference Worker and Per-Camera State Isolation** (Selected).

### Decision
Adopt a single-process modular monolith (`src/activity_system/`) where each camera maintains isolated ingestion and tracking state (`VideoSource`, `PersonTracker`, `TrackStateManager`), while sharing a single stateless `PoseEstimator` worker scheduled via bounded latest-frame queues.

### Implications
- **Performance**: Eliminates inter-process serialization overhead for raw video frames and prevents GPU/CPU memory exhaustion from loading duplicate model weights.
- **Maintainability**: Strongly typed Python interfaces allow unit-testing each stage in isolation while keeping deployment to a single binary/container.

## ADR-002: Pluggable Pose Estimation Backend (`YOLO-Pose Nano` Default + `ONNX Runtime` / Deterministic Synthetic Backend)

### Context
We need a fast, accurate pretrained multi-person 2D pose estimator that outputs bounding boxes, person confidence, and 17 COCO keypoints in a single forward pass, while supporting both production weights and offline deterministic verification.

### Decision
Select single-pass [Ultralytics YOLO Pose](https://docs.ultralytics.com/tasks/pose) (`yolov8n-pose` / `yolo11n-pose`, 17 COCO keypoints) wrapped in a `PoseBackend` protocol supporting:
- `UltralyticsPoseBackend` for native PyTorch/Ultralytics execution (`pip install .[yolo]`).
- `OnnxPoseBackend` using pre-installed `onnxruntime==1.31.0` for lightweight CPU inference.
- `SyntheticPoseBackend` for deterministic offline unit, integration, and video pipeline verification.

## ADR-003: Standalone Per-Camera ByteTrack Implementation with Session Namespacing

### Context
Multi-person activity recognition requires persistent track IDs across frames, resilience against brief occlusions, and strict isolation between cameras and across RTSP reconnections.

### Decision
Implement a standalone ByteTrack algorithm per camera using NumPy Kalman filtering and [SciPy `linear_sum_assignment`](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.optimize.linear_sum_assignment.html), modeled on [Ultralytics Track Mode](https://docs.ultralytics.com/modes/track). Whenever `VideoSource` reconnects a live stream, `session_id` increments so track IDs (`{camera_id}_s{session_id}_track_{id}`) never collide with pre-disconnection tracks.

## ADR-004: CCTV Viewpoint Scale-Normalized Kinematic + Gait Baseline

### Context
Elevated CCTV mounting angles introduce perspective foreshortening along the ground plane and depth-dependent bounding-box scale changes.

### Decision
Normalize displacement over real elapsed time (`dt`) by anatomical torso/bounding-box height and apply a configurable `cctv_foreshortening_factor` (`1.25` default for elevated corridor/perimeter CCTV views) to vertical image-plane movement component when computing ground-plane speed proxy in body-heights per second ($\text{BH/s}$), combined with lower-body ankle/knee oscillation and hysteresis smoothing.

## ADR-005: Copy-on-Write Immutable State Snapshots for Optional FastAPI Layer (Phase 7)

### Context
In Phase 7, the optional [FastAPI](https://fastapi.tiangolo.com/) server must serve `GET /health`, `/cameras`, `/tracks`, `/activities`, and `/metrics` without introducing Redis/databases and without blocking the real-time video ingestion or pose inference loop.

### Decision
Package FastAPI (`fastapi`, `uvicorn`, `httpx`) strictly under the optional `[project.optional-dependencies] api` group in `pyproject.toml`. Use an in-memory `StateSnapshotStore` where the pipeline thread atomically swaps an immutable snapshot reference after each frame step, and FastAPI HTTP handlers read the latest snapshot reference without holding locks over camera or inference objects.
