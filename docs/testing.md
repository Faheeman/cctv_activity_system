# Testing, Evaluation, and Benchmarking Report

## 1. Automated Test Suite Execution (`37 / 37` Passing)

### Execution Command
```bash
PYTHONPATH=project/src:project python3 -B -m unittest discover -s project/tests -v
```

### Verified Test Suite Breakdown

| Phase | Test Module | Tests | Status | Verified Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | `tests/unit/test_config.py` | 8 | **PASS** | YAML loading, duplicate camera rejection, invalid RTSP/webcam/threshold rejection, RTSP credential redaction (including `@` in passwords) |
| **Phase 1** | `tests/unit/test_video_source.py` | 6 | **PASS** | File EOF, frame sampling interval, bounded buffer stale-frame dropping, RTSP exponential backoff & session increment, graceful shutdown |
| **Phase 1** | `tests/integration/test_video_ingestion.py` | 2 | **PASS** | Real OpenCV `cv2.CAP_FFMPEG` MJPG `.avi` ingestion and frame resizing |
| **Phase 2** | `tests/unit/test_pose_estimator.py` | 4 | **PASS** | Empty black frame handling, person confidence filtering, out-of-bounds bbox clamping, occluded keypoint masking, YOLOv8/11 `(1, 56, N)` tensor decoding |
| **Phase 2** | `tests/unit/test_renderer.py` | 1 | **PASS** | Non-mutating bbox, COCO 17-keypoint skeleton, track ID, and telemetry overlay |
| **Phase 2** | `tests/integration/test_pose_pipeline.py` | 1 | **PASS** | End-to-end `VideoSource` -> `PoseEstimator` -> `Renderer` on synthetic CCTV footage |
| **Phase 3** | `tests/unit/test_person_tracker.py` | 4 | **PASS** | Multi-person persistent track IDs, ByteTrack two-stage low-confidence occlusion recovery, track expiration & window pruning, per-camera & reconnect session isolation |
| **Phase 4** | `tests/unit/test_activity_classifier.py` | 3 | **PASS** | Minimum observation warm-up guard (`unknown`), near vs. far walking scale invariance, hysteresis anti-flicker smoothing |
| **Phase 4** | `tests/integration/test_activity_evaluation.py` | 1 | **PASS** | Session-split evaluation across standing, walking (near/far), running (corridor/diagonal), and occluded sequences |
| **Phase 5 & 6** | `tests/integration/test_end_to_end.py` | 2 | **PASS** | Annotated `.avi` and `.jsonl` generation; multi-camera fault isolation when an RTSP stream fails |
| **Phase 7** | `tests/integration/test_api.py` | 5 | **PASS** | `GET /health`, `/cameras`, `/tracks`, `/activities`, `/metrics`, query filtering, `404`/`422` validation, privacy checks, and concurrent thread-safe reads |
| **Total** | **11 Test Modules** | **37** | **100% PASS** | **Executed in `3.255s` with `0` failures and `0` errors** |

## 2. Activity Recognition Evaluation Results (Session-Split Dataset)

Reproduced via `PYTHONPATH=project/src:project python3 -B project/scripts/benchmark.py` (`project/outputs/benchmark_results.json`). Sequences are split strictly by recording session (`6` sequences, `138` post-warmup evaluated frames) to prevent adjacent-frame data leakage.

### 2.1 Per-Class Precision, Recall, and F1 Score

| Activity Class | Precision | Recall | F1 Score | Support (Post-Warmup Frames) |
| :--- | :--- | :--- | :--- | :--- |
| `standing` | `1.0000` | `0.9231` | `0.9600` | `26` |
| `walking` | `1.0000` | `1.0000` | `1.0000` | `52` |
| `running` | `1.0000` | `1.0000` | `1.0000` | `34` |
| `unknown` (Occluded) | `0.9286` | `1.0000` | `0.9630` | `26` |
| **Macro Average** | **`0.9822`** | **`0.9808`** | **`0.9807`** | **`138`** |

### 2.2 Confusion Matrix (Post-Warmup Frames)

| True \ Predicted | `standing` | `walking` | `running` | `unknown` |
| :--- | :--- | :--- | :--- | :--- |
| **`standing`** | `24` | `0` | `0` | `2` |
| **`walking`** | `0` | `52` | `0` | `0` |
| **`running`** | `0` | `0` | `34` | `0` |
| **`unknown` (Occluded)** | `0` | `0` | `0` | `26` |

### 2.3 Temporal Stability and Cold-Start Metrics

| Metric | Measured Value | Target Acceptance Threshold | Verdict |
| :--- | :--- | :--- | :--- |
| **Macro F1 Score** | `0.9807` | `>= 0.85` | **PASS** |
| **False Transitions per Track-Minute** | `0.0000` | `<= 2.0` | **PASS** |
| **Mean Time to Stable Classification** | `0.627 s` | `<= 1.5 s` | **PASS** |
| **Max Time to Stable Classification** | `0.733 s` | `<= 1.5 s` | **PASS** |
| **Overall Unknown Rate (Including Warm-Up & Occluded Seq)** | `42.71%` | N/A (Reflects conservative warm-up + occluded sequence) | **PASS** |
| **Post-Warmup Clear-Visibility Unknown Rate** | `1.79%` | `<= 10.0%` | **PASS** |

## 3. Hardware Performance and Resource Benchmarks

Measured on the 2-vCPU, 5.0 GiB RAM Linux sandbox (`Python 3.11.2`, `opencv-python 4.11.0.86`) using `scripts/benchmark.py`:

| Benchmark Profile | Cameras | Resolution | Sample Interval | Frames Processed | Frames Dropped | Mean Inference (ms) | Mean End-to-End (ms) | Throughput (FPS) | CPU Util (%) | RSS Memory (MiB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1cam_640x480_sample1` | `1` | `640x480` | `1` | `60` | `0` | `1.055` | `8.691` | `89.05` | `160.28%` | `305.50` |
| `1cam_640x480_sample2` | `1` | `640x480` | `2` | `30` | `0` | `1.032` | `9.822` | `81.04` | `159.38%` | `305.50` |
| `1cam_480x360_sample1` | `1` | `480x360` | `1` | `60` | `0` | `0.945` | `8.748` | `88.94` | `154.16%` | `304.74` |
| `2cam_640x480_sample1` | `2` | `640x480` | `1` | `120` | `0` | `1.095` | `8.218` | `92.61` | `157.43%` | `306.58` |

*Note on Inference Backend*: The hardware figures above reflect the deterministic `SyntheticPoseBackend` pipeline (`~1.05 ms` pose extraction + ByteTrack + kinematics + OpenCV skeleton rendering = `~8.7 ms` end-to-end per frame). When running `OnnxPoseBackend` with `yolov8n-pose.onnx` (`imgsz=640`) on a 2-vCPU host without GPU acceleration, dense ONNX forward passes add `~45–90 ms` per frame (`~10–18 FPS`), for which `sample_every_n_frames: 2` or `imgsz: 480` is recommended.

## 4. Documented Limitations and Production-Readiness Gaps

1. **Optical-Axis Foreshortening**: Movement directly toward or away from a high-mounted CCTV camera exhibits compressed 2D displacement; while `cctv_foreshortening_factor` (`1.25`) partially compensates, extreme overhead views (`> 60°` depression angle) require homography calibration per camera.
2. **Severe Crowd Occlusion**: When lower-body keypoints (`13–16`) are occluded for `> 60%` of the sliding window, the classifier intentionally outputs `unknown` rather than guessing from head/shoulder translation.
3. **Extended Activities**: The current baseline classifies `standing`, `walking`, `running`, and `unknown`. Extending to `sitting`, `falling`, or `jumping` requires aspect-ratio change rates ($\Delta(w/h)/\Delta t$) and hip-to-knee angle thresholds calibrated on domain-specific footage.
