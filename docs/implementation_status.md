# IURIS Risk Detection and Alerting Integration (`docs/IURIS_RISK_ALERT_INTEGRATION.md`)

## 1. Project Continuity & Phase Completion Summary

| Field | Value |
| :--- | :--- |
| **Current Phase** | Phase 8 — Documentation, Handoff, and Final Verification (Completed) |
| **Completed Phases** | Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6, Phase 7, Phase 8 |
| **Last Updated (UTC)** | 2026-10-27T04:45:00Z |
| **Automated Test Suite** | `67 / 67` unit and integration tests passing (`100%` pass rate) |
| **End-to-End Crowd-Running Demo** | Verified (`scripts/demo_crowd_running_alert.py`) |

---

## 2. Phase-by-Phase Execution Log

### Phase 1 — Audit and Baseline (Completed)
- **Inspected Files**: `main.py`, `api.py`, `schemas.py`, `config.py`, `risk_engine.py`, `event_store.py`, `camera_summary.py`, `iuris/client.py`, `iuris/schemas.py`, `telemetry.py`, `tests/integration/test_api.py`, `config/default.yaml`, `config/live_webcam.yaml`, `config/production_rtsp.yaml`.
- **Baseline Test Result**: `55 / 55` passing (`6.491s`), `77.44 FPS` (`1cam_640x480_sample1`, `8.804 ms` E2E, `323.35 MiB` RSS).
- **Identified Gaps**:
  1. Crowd-running rule needed observation-quality filtering (`min_observation_quality`), temporal grace hysteresis (`clear_grace_seconds`) across brief frame drops, reconnect session resets, and per-camera overrides (`camera_overrides`).
  2. Risk events needed stable correlation keys (`correlation_key`), human-readable `title`, `camera_name`, lifecycle states (`TRIGGERED`, `UPDATED`, `ACKNOWLEDGED`, `RESOLVED`), `created_at`/`updated_at`/`acknowledged_at`/`resolved_at`, `status_history`, and operator workflow endpoints (`/v1/risk-events/active`, `/acknowledge`, `/resolve`).
  3. `IurisDeliveryClient` needed per-camera round-robin fair scheduling so a noisy camera cannot starve alerts from other cameras, delivery latency/attempt metrics, and a safe local `dry_run` testing mode.

### Phase 2 — Risk-Rule Correctness (Completed)
- **Modified Files**: `src/activity_system/config.py`, `src/activity_system/risk_engine.py`, `config/risk_rules.yaml`, `tests/unit/test_risk_engine.py`.
- **Why Necessary**: Added configurable starting thresholds (`minimum_people: 5`, `percentage_threshold: 40.0%`, `minimum_duration_seconds: 5.0s`, `severity: MEDIUM`), `min_observation_quality` filtering (rejecting `POOR` observations), strict exclusion of `UNKNOWN` from running counts, `clear_grace_seconds` hysteresis, `camera_overrides`, `PersistentActivityPatternRule`, and camera session reset handling (`notify_camera_session`).
- **Verification**: All 12 required unit test scenarios in `tests/unit/test_risk_engine.py` pass.

### Phase 3 — Event Lifecycle & Persistence (Completed)
- **Modified Files**: `src/activity_system/schemas.py`, `src/activity_system/event_store.py`.
- **Why Necessary**: Added `AlertLifecycleState`, `RiskStatusHistoryEntry`, `correlation_key`, `title`, `camera_name`, `matching_people`, `relevant_track_ids`, `related_camera_url`, and in-place lifecycle updates (`acknowledge_risk_event`, `resolve_risk_event`, `count_active_risk_events`) in `BoundedEventStore`.

### Phase 4 — Delivery Reliability, Fairness, and Observability (Completed)
- **Modified Files**: `src/activity_system/iuris/schemas.py`, `src/activity_system/iuris/mapper.py`, `src/activity_system/iuris/client.py`, `src/activity_system/telemetry.py`, `src/activity_system/main.py`, `src/activity_system/api.py`, `tests/unit/test_iuris_integration.py`, `tests/integration/test_api.py`.
- **Why Necessary**: Implemented per-camera round-robin queues with priority placement for `risk.detected` events, lifecycle deduplication tokens (`dedup_key`), delivery latency & attempt counters, safe local `dry_run` mode, and FastAPI endpoints `GET /v1/risk-events/active`, `GET /v1/risk-events/{event_id}`, `POST /v1/risk-events/{event_id}/acknowledge`, `POST /v1/risk-events/{event_id}/resolve`, and `GET /v1/iuris/status`.

### Phase 5 & 6 — IURIS Backend & Frontend Integration Specification (Completed)
- **Scope Note**: Because the external `sofikulislam727/iuris` repository is a separate project outside this workspace, zero files in IURIS were modified. Section 4 below provides the exact backend ingestion and frontend dashboard integration specification for IURIS operators.

### Phase 7 — End-to-End Validation & Performance Comparison (Completed)
- **Created / Modified Files**: `scripts/demo_crowd_running_alert.py`, `tests/integration/test_end_to_end.py`.
- **Verification**: Executed 12-person synthetic campus gate scenario (7 running = `58.33%`, 5 walking = `41.67%` for `7.0s` at `15 FPS`), verifying single alert generation at `5.0s`, suppression of `20` duplicate subsequent frame alerts, authenticated delivery to a local mock IURIS server, and operator acknowledgment and resolution transitions.

---

## 3. Risk-Event Lifecycle & Schema (`RiskEventV1`)

```text
[Condition Met >= min_duration]
              |
              v
   status = OPEN, lifecycle_state = TRIGGERED (occurrence_count = 1)
              |
              +---> [Condition continues past cooldown] ---> status = OPEN, lifecycle_state = UPDATED (same event_id, occurrence_count++)
              |
              +---> [Operator POST /acknowledge] ----------> status = ACKNOWLEDGED, lifecycle_state = ACKNOWLEDGED
              |
              +---> [Condition clears > clear_grace_sec] --> status = RESOLVED, lifecycle_state = RESOLVED (actor = "system")
              |     or [Operator POST /resolve]              status = RESOLVED, lifecycle_state = RESOLVED (actor = operator)
```

---

## 4. IURIS Backend & Dashboard Integration Blueprint (Remaining Work on IURIS Side)

When connecting the separate **IURIS** repository (`sofikulislam727/iuris`) to this CCTV backend:

### 4.1 IURIS Backend Ingestion Route
1. **Webhook Receiver (`POST /api/v1/cctv/events`)**:
   - Verify `Authorization: Bearer <IURIS_API_TOKEN>`.
   - Inspect `X-Event-Type` (`risk.detected`, `camera.activity_summary`, `activity.detected`) and `Idempotency-Key`.
   - Upsert `risk.detected` payloads into the IURIS alerts table keyed by `event_id` (updating `status`, `lifecycle_state`, `updated_at`, `acknowledged_at`, `resolved_at`, `occurrence_count`, `evidence`, and `status_history` when an `UPDATED`, `ACKNOWLEDGED`, or `RESOLVED` transition arrives for an existing `event_id`).
2. **Bidirectional Acknowledgment / Resolution Forwarding**:
   - When an operator clicks **Acknowledge** or **Resolve** inside the IURIS UI, IURIS should update its database and call:
     - `POST http://<cctv-backend>:8081/v1/risk-events/{event_id}/acknowledge`
     - `POST http://<cctv-backend>:8081/v1/risk-events/{event_id}/resolve`

### 4.2 IURIS Dashboard Alert Component Mapping
Map `RiskEventV1` fields directly into IURIS's existing alert/incident table and detail drawer:
- **Title & Description**: `title` (`"Unusual Running Concentration Detected"`) and `explanation`
- **Severity Badge**: `severity` (`INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
- **Camera**: `camera_name` (`"Islamic University Main Gate"`) + `camera_id` (`"cam_iu_main_gate"`)
- **Key Measurements**: `matching_people` / `observed_people` (`7 / 12`), `measured_percentage` (`58.33%` vs threshold `40.0%`), `duration_seconds` (`5.0s`), `observation_quality` (`0.95`)
- **Status & Lifecycle**: `status` (`OPEN`, `ACKNOWLEDGED`, `RESOLVED`, `EXPIRED`) and `lifecycle_state` (`TRIGGERED`, `UPDATED`, `ACKNOWLEDGED`, `RESOLVED`)
- **Timestamps**: `created_at`, `updated_at`, `acknowledged_at`, `resolved_at`
- **Live Camera Link**: `related_camera_url` (`/stream.mjpg?camera_id=cam_iu_main_gate`)
- **Filters**: `severity`, `camera_id`, `risk_type`, `rule_name`, `status`, `active_only`, `since`, `until`

---

## 5. Before vs. After Performance Benchmark Comparison

Measured via `PYTHONPATH=src:. python3 -B scripts/benchmark.py` on the same 2-vCPU Linux environment:

| Profile | Metric | Baseline (Phase 1) | Upgraded (Phase 7) | Delta / Impact |
| :--- | :--- | :--- | :--- | :--- |
| `1cam_640x480_sample1` | Mean Inference Latency | `1.046 ms` | `1.124 ms` | `+0.078 ms` (negligible) |
| `1cam_640x480_sample1` | Mean End-to-End Latency | `8.804 ms` | `9.197 ms` | `+0.393 ms` (within 1 ms) |
| `1cam_640x480_sample1` | Throughput FPS | `77.44 FPS` | `73.16 FPS` | Real-time (`~4.9x` faster than 15 FPS target) |
| `1cam_640x480_sample1` | Memory RSS | `323.35 MiB` | `324.71 MiB` | `+1.36 MiB` (bounded) |
| `1cam_480x360_sample1` | Throughput FPS | `78.26 FPS` | `76.54 FPS` | Real-time (`> 5x` 15 FPS target) |
