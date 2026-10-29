# System Architecture: Standalone Real-Time CCTV Activity Intelligence Backend

## 1. Architectural Overview & Pipeline Topology

The backend is implemented as a modular, production-oriented Python 3.11 service (`src/activity_system/`) that processes live camera streams, detects and tracks people, estimates temporal activities, computes deduplicated camera-level statistics, evaluates configurable risk rules independently from activity classification, and exposes structured events via both a versioned REST API and an optional authenticated outbound HTTP delivery client for future IURIS consumption.

```text
Webcam / RTSP / Video File
            |
            v
      Stream Manager (StreamManager + InputStreamAdapter)
            |
            v
     Person Detection (PoseEstimator.detect_persons / PersonDetector)
            |
            v
    Multi-Object Tracking (PersonTracker + TrackStateManager)
            |
            v
    Human Pose Estimation (PoseEstimator.estimate / HumanPoseEstimator)
            |
            v
 Temporal Activity Recognition (ActivityClassifier)
            |
            +-------------------+
            |                   |
            v                   v
 Activity Events         Camera Summaries (CameraSummaryAggregator)
            |                   |
            +---------+---------+
                      |
                      v
         Independent Risk Engine (RiskAssessmentEngine)
                      |
                      v
         Validated Event Schemas (ActivityObservationV1, CameraActivitySummaryV1, RiskEventV1)
                      |
            +---------+---------+
            |                   |
            v                   v
     FastAPI Backend     Authenticated Event Delivery (IurisDeliveryClient)
   (/v1/* + /metrics)           |
            |                   v
            +----------> Future IURIS Consumer
```

## 2. Decoupled Stage Interfaces (`src/activity_system/interfaces.py`)

Each pipeline stage adheres to an explicit `Protocol` interface so models, tracking algorithms, and risk rules can be swapped independently without rewriting upstream or downstream stages:

| Stage | Protocol Interface | Default Implementation | File Path |
| :--- | :--- | :--- | :--- |
| **Input Sources** | `InputStreamAdapter` | `WebcamStreamAdapter`, `RtspStreamAdapter`, `VideoFileStreamAdapter` | `src/activity_system/video_source.py` |
| **Stream Orchestration** | `StreamManager` | `StreamManager` (camera-scoped fault isolation) | `src/activity_system/video_source.py` |
| **Person Detection** | `PersonDetector` | `PoseEstimator.detect_persons` | `src/activity_system/pose_estimator.py` |
| **Human Pose Estimation** | `HumanPoseEstimator` | `PoseEstimator` (`OnnxPoseBackend`, `UltralyticsPoseBackend`, `SyntheticPoseBackend`) | `src/activity_system/pose_estimator.py` |
| **Multi-Object Tracking** | `MultiObjectTracker` | `PersonTracker` (2-stage ByteTrack + 8-state Kalman filter + `TrackStateManager`) | `src/activity_system/person_tracker.py`, `src/activity_system/track_state.py` |
| **Feature Extraction** | `ActivityFeatureExtractor` | `ActivityClassifier.extract_features` | `src/activity_system/activity_classifier.py` |
| **Temporal Activity Recognition** | `TemporalActivityClassifier` | `ActivityClassifier` (+ optional `SupervisedActivityClassifier`) | `src/activity_system/activity_classifier.py` |
| **Camera Summaries** | `CameraSummaryEvaluatorProtocol` | `CameraSummaryAggregator` | `src/activity_system/camera_summary.py` |
| **Risk Assessment** | `RiskEngineProtocol`, `RiskRuleProtocol` | `RiskAssessmentEngine` (`UnusualActivityConcentrationRule`, `PossibleFallRule`, `ProlongedImmobilityRule`, `ZoneActivityRule`) | `src/activity_system/risk_engine.py` |
| **Event Persistence** | `BoundedEventStore` | `BoundedEventStore` (bounded ring buffers + optional SQLite) | `src/activity_system/event_store.py` |
| **Outbound IURIS Delivery** | `IurisDeliveryClient` | `IurisDeliveryClient` (`mapper.py`, `retry.py`, `schemas.py`, `client.py`) | `src/activity_system/iuris/` |

## 3. State Separation: Live In-Memory Snapshot vs. Bounded Event History

1. **Live In-Memory State (`StateSnapshotStore`)**:
   - Holds an immutable `ApplicationSnapshot` reference swapped atomically in $O(1)$ time after each processed frame.
   - Powers non-blocking reads for `/health`, `/ready`, `/v1/cameras`, `/v1/tracks`, `/v1/activities?source=live`, `/v1/camera-summaries?source=live`, `/metrics`, and `/stream.mjpg`.
2. **Bounded Event History (`BoundedEventStore`)**:
   - Stores deduplicated historical events (`activity.detected`, `camera.activity_summary`, `risk.detected`) in fixed-capacity `collections.deque(maxlen=...)` ring buffers (with optional SQLite persistence via `storage.sqlite_path`).
   - Prevents unbounded Python list memory leaks while serving paginated historical queries (`/v1/activities?source=history`, `/v1/camera-summaries?source=history`, `/v1/risk-events`).

## 4. Camera-Level Activity Summary Calculation & Deduplication

- Over a configurable sliding window (`summary.window_seconds`, default `10.0`s), `CameraSummaryAggregator` buffers every emitted `ActivityEvent` on that camera.
- **Per-Track Deduplication**: To avoid double-counting the same tracked person across multiple frames in the window, observations are grouped by unique `track_id`. Each distinct `track_id` observed in the window counts as **1 person** (`observed_people = len(unique_tracks_in_window)`), attributed to its most recent committed activity label in the window.
- **`sample_count`**: Exposes the total number of frame-level track observations accumulated across the window.
- **`UNKNOWN` Treatment**:
  - When `summary.exclude_unknown_from_percentages: true`, `eligible_people = observed_people - unknown_count`, and percentages for known activities are calculated over `eligible_people` (`unknown_treatment = "exclude_from_percentage_denominator"`).
  - When `summary.exclude_unknown_from_percentages: false` (default), `eligible_people = observed_people`, and percentages across all 7 labels sum to `100.0%` (`unknown_treatment = "include_in_percentage_denominator"`).
- **Responsible Interpretation**: Activity percentages describe the proportion of observed tracked individuals performing an activity during the window; they must never be interpreted as probabilities of danger or hostile intent.

## 5. Security, Privacy, and Fault Isolation

- **Credential Redaction**: `logging_config.redact_uri()` strips `rtsp://user:pass@host` credentials before any log record, exception string, or API response is emitted.
- **No Biometric Profiling**: Tracking IDs (`{camera_id}_s{session_id}_track_{id}`) are temporary, camera-scoped, and session-scoped. No facial recognition or cross-camera re-identification is performed.
- **Fault Isolation**: A failed camera stream transitions through exponential backoff + jitter to `failed` without interrupting other cameras or the FastAPI server. An unavailable IURIS destination triggers bounded retries and writes undelivered envelopes to the JSONL dead-letter outbox (`outputs/iuris_dead_letter.jsonl`) without stalling computer-vision inference.
