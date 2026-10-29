# Model, Tracking, and Activity Classification Strategy

## 1. Person Detection & Human Pose Estimation (`src/activity_system/pose_estimator.py`)

### 1.1 Pluggable Backends
The pipeline separates detection/pose estimation from tracking and activity recognition via `PersonDetector`, `HumanPoseEstimator`, and `PoseBackend` protocols, supporting three runtime backends:
1. **`OnnxPoseBackend` (`pose.backend: "onnx"`)**:
   - Executes single-pass person detection and 17-keypoint COCO pose estimation using `onnxruntime` and `models/yolov8n-pose.onnx`.
   - Supports `AI_DEVICE=auto`, `cpu`, `cuda`, and `tensorrt`, automatically selecting `CUDAExecutionProvider` when available and falling back cleanly to `CPUExecutionProvider`.
2. **`UltralyticsPoseBackend` (`pose.backend: "ultralytics"`)**:
   - Uses native PyTorch/Ultralytics (`models/yolov8n-pose.pt`) when the optional `pip install -e ".[yolo]"` extra is installed.
3. **`SyntheticPoseBackend` (`pose.backend: "synthetic"`)**:
   - Deterministic pose generator for CI testing, offline evaluation sequences, and synthetic video fixtures.

### 1.2 Validation & Non-Fabrication Guarantee
- Bounding boxes are clamped to frame dimensions `(width, height)` and rejected if smaller than `6x12` pixels or below `min_person_confidence`.
- Individual keypoints below `min_keypoint_confidence` (`0.35`) or outside image bounds are marked `visible=False`. The estimator never fabricates missing keypoints when a person is partially occluded.
- Each detection is graded into `ObservationQuality.GOOD`, `PARTIAL`, or `POOR` based on torso (`5, 6, 11, 12`) and lower-body (`13, 14, 15, 16`) keypoint visibility.

## 2. Multi-Object Tracking (`src/activity_system/person_tracker.py`)

- Uses a per-camera two-stage **ByteTrack** association algorithm backed by an 8-state constant-velocity Kalman filter (`KalmanBoxState`) and Hungarian IoU assignment (`scipy.optimize.linear_sum_assignment`).
- Stage 1 associates confirmed/active tracks against high-confidence detections (`>= high_thresh`).
- Stage 2 associates remaining unmatched tracks against low-confidence detections (`low_thresh <= conf < high_thresh`) to bridge partial occlusions.
- Track IDs (`{camera_id}_s{session_id}_track_{id}`) are scoped to a single camera and stream session. When an RTSP stream reconnects, `session_id` increments so post-reconnection tracks never collide with stale pre-disconnection tracks.

## 3. Temporal Activity Classification (`src/activity_system/activity_classifier.py`)

### 3.1 Scale-Normalized Temporal Features
Over a sliding window of timestamped `TrackObservation` records (`max_window_duration_seconds: 1.5s`), `ActivityClassifier.extract_features()` computes:
- **Anatomical Scale (`body_height_px`)**: Estimated from median visible torso length ($\times 3.12$) or bounding-box maximum dimension.
- **Viewpoint-Compensated Speed (`speed_bh_per_sec`)**: Net smoothed displacement in body-heights per second ($\text{BH/s}$), applying `cctv_foreshortening_factor` (`1.25`) to vertical image-plane motion for elevated CCTV viewpoints.
- **Lower-Limb Gait Features**: `ankle_std_bh`, `knee_std_bh`, `stride_amp_bh` (10th–90th percentile inter-ankle separation), and `cadence_hz` (zero-crossing stride frequency).
- **Posture & Collapse Features**: `aspect_ratio_start`, `aspect_ratio_median`, `aspect_ratio_latest` ($w/h$), `hip_knee_vertical_ratio`, `torso_horizontal_ratio`, `downward_velocity_bh_per_sec`, and `height_drop_ratio`.

### 3.2 Distinction Between `FALLING`, `LYING`, and `SITTING`
- **`FALLING`**: Requires multi-frame transition evidence: the track must start the window upright (`aspect_ratio_start < 0.95`), end collapsed/horizontal (`aspect_ratio_latest >= 0.95` or `height_drop_ratio >= 0.25`), and exhibit rapid downward vertical velocity (`downward_velocity_bh_per_sec >= 0.80 BH/s`) within `<= 1.50s`, confirmed across consecutive frames via hysteresis. A single horizontal frame never triggers `FALLING`.
- **`LYING`**: Requires sustained horizontal posture (`aspect_ratio_median >= 1.15` or `torso_horizontal_ratio >= 1.35`) for `>= 0.60s` with low translation speed (`<= 0.45 BH/s`) and no active rapid vertical descent.
- **`SITTING`**: Requires stationary compact posture (`speed <= 0.35 BH/s`), upright torso (`torso_horizontal_ratio < 0.85`), and bent/foreshortened hip-to-knee vertical ratio (`<= 0.22`).

### 3.3 Transparency of Confidence vs. Evidence Score
- When `activity.classifier_type: "heuristic"` (the default kinematic rule baseline), `ActivityEvent.confidence` is explicitly set to `null` (`None`), while rule match strength is reported in `evidence_score` (`[0.0, 1.0]`). Heuristic rules are never misrepresented as calibrated neural network probabilities.
- When `activity.classifier_type: "supervised"`, `SupervisedActivityClassifier` computes calibrated multinomial logistic probabilities reported in `confidence`.
