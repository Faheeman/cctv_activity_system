# REST API & Event Schema Contract (`v1.0`)

## 1. Overview & Conventions

- **Base URL**: `http://<host>:8081` (configurable via `AI_BACKEND_HOST` and `AI_BACKEND_PORT`)
- **Timestamps**: All timestamps are timezone-aware UTC in ISO 8601 format (`YYYY-MM-DDTHH:MM:SSZ`).
- **Pagination**: Versioned collection endpoints (`/v1/*`) accept `limit` (`1..500`, default `50`) and `offset` (`>= 0`, default `0`) and return a `pagination` object:
  ```json
  {
    "total": 42,
    "count": 10,
    "limit": 10,
    "offset": 0
  }
  ```
- **State Source Separation**: Endpoints that expose both current frame state and historical event logs include `"state_source": "live_in_memory"` or `"state_source": "event_history"`.

## 2. Endpoints Summary

| Method | Path | Purpose | Query Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Process liveness check | None |
| `GET` | `/ready` | Readiness of models, pipeline, and services (`200` ready, `503` not ready) | None |
| `GET` | `/v1/cameras` | Camera list, connection status, frame timestamps, dimensions, processing state | `camera_id`, `state`, `limit`, `offset` |
| `GET` | `/v1/tracks` | Currently active tracked people across cameras | `camera_id`, `lifecycle`, `limit`, `offset` |
| `GET` | `/v1/activities` | Latest activity observations (`source=live`) or bounded history (`source=history`) | `camera_id`, `activity`, `track_id`, `since`, `source`, `limit`, `offset` |
| `GET` | `/v1/camera-summaries` | Camera-level activity distributions and percentages | `camera_id`, `since`, `source`, `limit`, `offset` |
| `GET` | `/v1/risk-events` | Risk-event history from the independent risk engine | `camera_id`, `severity`, `risk_type`, `status`, `since`, `limit`, `offset` |
| `GET` | `/metrics` | Operational JSON telemetry and IURIS delivery counters | None |
| `GET` | `/metrics/prometheus` | Prometheus text exposition format (`text/plain; version=0.0.4`) | None |

*(Legacy read-only endpoints `/cameras`, `/tracks`, `/activities`, `/frame.jpg`, `/stream.mjpg`, and `/` are also preserved for backward compatibility.)*

## 3. Canonical Event Schemas (`schema_version: "1.0"`)

### 3.1 Activity Observation Event (`activity.detected` — `ActivityObservationV1`)

| Field | Type | Required | Constraints / Semantics |
| :--- | :--- | :--- | :--- |
| `schema_version` | `string` | Yes | `"1.0"` |
| `event_id` | `string` | Yes | Unique UUIDv4 identifier for this event |
| `event_type` | `string` | Yes | `"activity.detected"` |
| `camera_id` | `string` | Yes | Configured camera identifier |
| `track_id` | `string` | Yes | Temporary camera- and session-scoped tracking ID (`{camera_id}_s{session}_track_{n}`) |
| `timestamp` | `string` | Yes | UTC ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`) |
| `activity` | `string` | Yes | Enum: `STANDING`, `WALKING`, `RUNNING`, `SITTING`, `LYING`, `FALLING`, `UNKNOWN` |
| `confidence` | `float \| null` | No | Calibrated model confidence `[0.0, 1.0]` when using a supervised classifier; `null` for heuristic rule baseline |
| `observation_quality` | `float` | Yes | Pose/keypoint visibility quality score `[0.0, 1.0]` |
| `duration_seconds` | `float` | Yes | Elapsed duration (`>= 0.0`) of the current committed activity on this track |
| `bbox` | `list[int] \| null` | No | `[x1, y1, x2, y2]` in image pixel coordinates (`x2 > x1`, `y2 > y1`) |
| `pose_available` | `bool` | Yes | Whether at least one validated COCO keypoint is visible |
| `model_version` | `string` | Yes | Configured pose/activity model version string |
| `metadata` | `object` | Yes | Diagnostic kinematic features (`speed_bh_per_sec`, `ankle_std_bh`, `evidence_score`, etc.) |

**Example Payload**:
```json
{
  "schema_version": "1.0",
  "event_id": "7f8c2b10-4e1a-4c3d-9a21-8b7e6f5d4c3b",
  "event_type": "activity.detected",
  "camera_id": "camera_01",
  "track_id": "camera_01_s1_track_42",
  "timestamp": "2026-10-26T10:30:00Z",
  "activity": "RUNNING",
  "confidence": null,
  "observation_quality": 0.88,
  "duration_seconds": 3.2,
  "bbox": [120, 80, 250, 420],
  "pose_available": true,
  "model_version": "heuristic-kinematic-v1.0",
  "metadata": {
    "evidence_score": 0.91,
    "quality_grade": "good",
    "speed_bh_per_sec": 2.35
  }
}
```

### 3.2 Camera-Level Activity Summary (`camera.activity_summary` — `CameraActivitySummaryV1`)

| Field | Type | Required | Constraints / Semantics |
| :--- | :--- | :--- | :--- |
| `schema_version` | `string` | Yes | `"1.0"` |
| `event_id` | `string` | Yes | Unique UUIDv4 identifier |
| `event_type` | `string` | Yes | `"camera.activity_summary"` |
| `camera_id` | `string` | Yes | Configured camera identifier |
| `timestamp` | `string` | Yes | UTC ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`) |
| `window_seconds` | `float` | Yes | Sliding aggregation window in seconds (`> 0.0`) |
| `observed_people` | `int` | Yes | Count of distinct tracked individuals (`track_id`) observed in the window |
| `activity_distribution` | `object` | Yes | Map of all 7 activity labels to integer person counts |
| `activity_percentages` | `object` | Yes | Map of activity labels to percentage (`0.0..100.0`) |
| `observation_quality` | `float` | Yes | Mean observation quality `[0.0, 1.0]` across unique tracks in the window |
| `sample_count` | `int` | Yes | Total frame-track observations evaluated in the window |
| `eligible_people` | `int` | Yes | Denominator used for percentage computation |
| `unknown_count` | `int` | Yes | Number of unique tracks currently classified as `UNKNOWN` |
| `unknown_treatment` | `string` | Yes | `"include_in_percentage_denominator"` or `"exclude_from_percentage_denominator"` |
| `aggregation_method` | `string` | Yes | `"unique_track_latest_committed_activity"` |

**Example Payload**:
```json
{
  "schema_version": "1.0",
  "event_id": "3a1f9e40-1b2c-4d5e-8f90-123456789abc",
  "event_type": "camera.activity_summary",
  "camera_id": "camera_01",
  "timestamp": "2026-10-26T10:30:00Z",
  "window_seconds": 10.0,
  "observed_people": 20,
  "activity_distribution": {
    "STANDING": 8,
    "WALKING": 5,
    "RUNNING": 7,
    "SITTING": 0,
    "LYING": 0,
    "FALLING": 0,
    "UNKNOWN": 0
  },
  "activity_percentages": {
    "STANDING": 40.0,
    "WALKING": 25.0,
    "RUNNING": 35.0,
    "SITTING": 0.0,
    "LYING": 0.0,
    "FALLING": 0.0,
    "UNKNOWN": 0.0
  },
  "observation_quality": 0.92,
  "sample_count": 150,
  "eligible_people": 20,
  "unknown_count": 0,
  "unknown_treatment": "include_in_percentage_denominator",
  "aggregation_method": "unique_track_latest_committed_activity"
}
```

### 3.3 Risk Assessment Event (`risk.detected` — `RiskEventV1`)

| Field | Type | Required | Constraints / Semantics |
| :--- | :--- | :--- | :--- |
| `schema_version` | `string` | Yes | `"1.0"` |
| `event_id` | `string` | Yes | Unique UUIDv4 identifier |
| `event_type` | `string` | Yes | `"risk.detected"` |
| `camera_id` | `string` | Yes | Configured camera identifier |
| `timestamp` | `string` | Yes | UTC ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`) |
| `risk_type` | `string` | Yes | e.g., `UNUSUAL_ACTIVITY_CONCENTRATION`, `POSSIBLE_FALL_DETECTED`, `PROLONGED_IMMOBILITY`, `ZONE_RESTRICTED_ACTIVITY` |
| `rule_name` | `string` | Yes | Configured rule identifier that triggered the event |
| `severity` | `string` | Yes | Operational category: `INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` |
| `status` | `string` | Yes | `OPEN`, `ACKNOWLEDGED`, `RESOLVED`, `EXPIRED` |
| `activity` | `string \| null` | No | Associated `ActivityCode` |
| `measured_percentage` | `float \| null` | No | Measured activity percentage `[0.0, 100.0]` for concentration rules |
| `threshold_percentage` | `float \| null` | No | Configured threshold percentage `[0.0, 100.0]` |
| `observed_people` | `int` | Yes | Number of observed people in the window |
| `duration_seconds` | `float` | Yes | Sustained duration of the rule condition in seconds |
| `window_seconds` | `float` | Yes | Observation window duration in seconds |
| `observation_quality` | `float` | Yes | Observation quality `[0.0, 1.0]` |
| `track_id` | `string \| null` | No | Track ID for single-person rules (fall, immobility, zone) |
| `zone_id` | `string \| null` | No | Zone identifier for polygon zone rules |
| `evidence` | `object` | Yes | Structured rule thresholds, counts, and bounding-box evidence |
| `explanation` | `string` | Yes | Neutral, observable description of the rule condition met |
| `model_version` | `string` | Yes | Model version string |

**Example Payload**:
```json
{
  "schema_version": "1.0",
  "event_id": "9d8e7f60-5c4b-3a21-9087-654321fedcba",
  "event_type": "risk.detected",
  "camera_id": "camera_01",
  "timestamp": "2026-10-26T10:30:00Z",
  "risk_type": "UNUSUAL_ACTIVITY_CONCENTRATION",
  "rule_name": "unusual_running_concentration",
  "severity": "MEDIUM",
  "status": "OPEN",
  "activity": "RUNNING",
  "measured_percentage": 70.0,
  "threshold_percentage": 70.0,
  "observed_people": 20,
  "duration_seconds": 5.0,
  "window_seconds": 10.0,
  "observation_quality": 0.92,
  "confidence": null,
  "track_id": null,
  "zone_id": null,
  "evidence": {
    "rule": "unusual_running_concentration",
    "matching_people": 14,
    "observed_people": 20,
    "minimum_people_required": 5
  },
  "explanation": "The configured running-concentration threshold was met.",
  "model_version": "heuristic-kinematic-v1.0"
}
```

## 4. Error Response Format

- **`404 Not Found`**: Returned when filtering by an unregistered `camera_id`:
  ```json
  {
    "detail": "Camera 'unknown_cam' is not registered. Known cameras: ['cam_corridor_01']"
  }
  ```
- **`422 Unprocessable Entity`**: Returned by FastAPI/Pydantic when query parameters fail validation (e.g., invalid `activity` enum, invalid `severity`, or out-of-range `limit`/`offset`).
- **`503 Service Unavailable`**: Returned by `/ready` when required models or all configured camera streams have failed, or if the snapshot store is uninitialized.
