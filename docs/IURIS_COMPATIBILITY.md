# IURIS Integration & Compatibility Specification

## 1. Architectural Boundary

This AI backend runs as an independent standalone service. **Zero files in the `sofikulislam727/iuris` repository were modified.** Instead, this backend provides:
1. **Pull / Polling Integration**: Versioned REST endpoints (`GET /v1/cameras`, `GET /v1/tracks`, `GET /v1/activities`, `GET /v1/camera-summaries`, `GET /v1/risk-events`) that IURIS can poll or query on demand.
2. **Push / Webhook Integration (`src/activity_system/iuris/`)**: An optional, authenticated, idempotent outbound HTTP delivery client (`IurisDeliveryClient`) that pushes validated `v1.0` events to a configurable IURIS ingestion endpoint when enabled.

## 2. Identifier Separation Semantics

To prevent schema confusion when mapping into IURIS domain models, three identifiers are strictly separated on all emitted events:
- `event_id` (`string`, UUIDv4): Globally unique identifier for the specific emitted event instance. Used for deduplication and idempotency.
- `camera_id` (`string`): Configured logical camera identifier (e.g., `"cam_north_gate"`).
- `track_id` (`string | null`): Temporary, camera-scoped, session-scoped tracking identifier (`"{camera_id}_s{session_id}_track_{numeric_id}"`). Must **never** be treated as a persistent human identity or user ID.

## 3. Outbound HTTP Delivery Behavior (`src/activity_system/iuris/`)

### 3.1 Configuration & Environment Variables

| Environment Variable | Config Path | Default | Description |
| :--- | :--- | :--- | :--- |
| `IURIS_DELIVERY_ENABLED` | `iuris.delivery_enabled` | `false` | Enable/disable outbound HTTP push delivery |
| `IURIS_BASE_URL` | `iuris.base_url` | `""` | Base URL of the IURIS service (e.g., `http://iuris-backend:8080`) |
| `IURIS_EVENTS_PATH` | `iuris.events_path` | `"/api/v1/cctv/events"` | Configurable target ingestion endpoint path on IURIS |
| `IURIS_API_TOKEN` | `iuris.api_token` | `""` | Secret token injected via environment variable |
| `EVENT_QUEUE_MAX_SIZE` | `iuris.max_queue_size` | `1000` | Maximum in-memory outbound queue capacity |
| `EVENT_RETRY_MAX_ATTEMPTS` | `iuris.max_retry_attempts` | `8` | Maximum retry attempts per event on transient failure |

### 3.2 Authentication Options
- **Bearer Token (`iuris.auth_scheme: "bearer"`, default)**: Sends `Authorization: Bearer <IURIS_API_TOKEN>`.
- **API Key Header (`iuris.auth_scheme: "api_key_header"`)**: Sends `X-API-Key: <IURIS_API_TOKEN>` (or custom header configured via `iuris.auth_header_name`).

### 3.3 HTTP Request Headers & Idempotency Semantics
Every outbound `POST` request to `{IURIS_BASE_URL}{IURIS_EVENTS_PATH}` includes:
```http
POST /api/v1/cctv/events HTTP/1.1
Content-Type: application/json
Accept: application/json
Authorization: Bearer <IURIS_API_TOKEN>
Idempotency-Key: iuris-<event_id>-<sha256_prefix>
X-Event-Id: <event_id>
X-Event-Type: risk.detected
X-Schema-Version: 1.0
```
- **Idempotency Guarantee**: `Idempotency-Key` is derived deterministically from `(schema_version, event_type, event_id)` via `build_idempotency_key()`.
- If a network timeout causes a retry after IURIS already persisted the event, IURIS can return either `200 OK`, `201 Created`, `202 Accepted`, or `409 Conflict` (duplicate `event_id` / `Idempotency-Key`); `IurisDeliveryClient` treats `2xx` and `409` as successful idempotent completion.

### 3.4 Retry Policy, Backpressure, and Dead-Letter Outbox
- **Retryable Conditions**: Network timeouts, connection errors, and HTTP status codes `408`, `425`, `429`, `500`, `502`, `503`, `504` trigger exponential backoff with symmetric jitter (`compute_backoff_delay`).
- **Non-Retryable Conditions**: HTTP `400`, `401`, `403`, `404`, `422` immediately stop retrying and route the envelope to the dead-letter outbox.
- **Queue Saturation & Graceful Shutdown**: When the bounded queue reaches `EVENT_QUEUE_MAX_SIZE` or when the backend shuts down with undelivered events, envelopes are persisted to `outputs/iuris_dead_letter.jsonl` (`iuris.dead_letter_path`) so events are **never** silently discarded.

## 4. Activity & Operational Risk Label Definitions

### 4.1 Activity Labels (`activity`)
- `STANDING`: Upright person with near-zero translation speed (`<= 0.35 BH/s`) and minimal lower-limb oscillation over `>= 0.60s`.
- `WALKING`: Upright person moving at moderate normalized speed (`0.35..1.80 BH/s`) with periodic lower-limb gait oscillation.
- `RUNNING`: Upright person moving at high normalized speed (`>= 1.80 BH/s`) with elevated ankle/stride amplitude or cadence.
- `SITTING`: Stationary person in a compact seated geometry (upright torso with foreshortened vertical hip-to-knee span).
- `LYING`: Sustained horizontal body posture (`w/h >= 1.15` or horizontal torso orientation) for `>= 0.60s` without active high-velocity vertical descent.
- `FALLING`: Multi-frame rapid vertical descent (`>= 0.80 BH/s` downward) transitioning from an upright posture to a collapsed/horizontal posture within `<= 1.50s`.
- `UNKNOWN`: Emitted during initial track warm-up (`< 6` observations or `< 0.50s`), lower-body occlusion (`< 40%` keypoint visibility), or ambiguous motion.

### 4.2 Risk Types (`risk_type`) & Severities (`severity`)
- **Severities**: `INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` (operational triage categories for human review, not statistical probabilities).
- **Risk Types**:
  - `UNUSUAL_ACTIVITY_CONCENTRATION`: Configured percentage and minimum headcount of observed people performing a target activity (e.g., `RUNNING >= 70%` with `>= 5` people) sustained for `minimum_duration_seconds`.
  - `POSSIBLE_FALL_DETECTED`: Tracked person exhibiting sustained `FALLING` kinematics above minimum quality threshold.
  - `PROLONGED_IMMOBILITY`: Tracked person remaining in `LYING` posture continuously beyond `immobility_duration_seconds`.
  - `ZONE_RESTRICTED_ACTIVITY`: Tracked person performing a configured trigger activity inside a configured polygon zone for at least `minimum_duration_seconds`.

## 5. Required Future Implementation Steps on the IURIS Side

Because IURIS does not yet expose a dedicated CCTV activity/risk ingestion endpoint, when the IURIS team integrates with this backend later, IURIS should implement either (or both) of the following:

1. **Option A — Webhook Ingestion Endpoint in IURIS (Push Mode)**:
   - Implement `POST /api/v1/cctv/events` (or configure `IURIS_EVENTS_PATH` to match IURIS's route).
   - Validate the `Authorization: Bearer <IURIS_API_TOKEN>` header.
   - Deduplicate incoming payloads by `event_id` or the `Idempotency-Key` header (returning `200 OK` or `409 Conflict` on repeats).
   - Route payloads by `event_type`:
     - `"activity.detected"` -> real-time camera track overlay / telemetry store.
     - `"camera.activity_summary"` -> camera crowd analytics time-series table.
     - `"risk.detected"` -> operator alert triage queue (mapping `severity`, `risk_type`, `camera_id`, `evidence`, and `explanation`).
2. **Option B — Scheduled Polling Worker in IURIS (Pull Mode)**:
   - Poll `GET /v1/risk-events?since=<last_poll_utc_iso>&limit=100` and `GET /v1/camera-summaries?source=live` against this backend (`http://cctv-activity-engine:8081`).
   - Deduplicate records using `event_id`.
