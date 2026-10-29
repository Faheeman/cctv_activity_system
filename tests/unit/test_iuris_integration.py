"""Unit and mocked HTTP server tests for IURIS delivery reliability, fairness, timeouts, auth, dry_run, and BoundedEventStore."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import tempfile
import threading
from typing import Dict, List, Tuple
import unittest

from activity_system.config import EventStorageConfig, IurisConfig
from activity_system.event_store import BoundedEventStore
from activity_system.iuris import (
    IurisDeliveryClient,
    IurisEventMapper,
    build_idempotency_key,
    compute_backoff_delay,
    is_retryable_status_code,
)
from activity_system.schemas import (
    ActivityCode,
    ActivityObservationV1,
    AlertLifecycleState,
    CameraActivitySummaryV1,
    RiskEventV1,
    RiskSeverity,
    RiskStatus,
)


class _MockIurisHttpHandler(BaseHTTPRequestHandler):
    received_requests: List[Dict] = []
    status_sequence: List[int] = [200]

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8")
        parsed = json.loads(raw_body) if raw_body else {}
        self.__class__.received_requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "body": parsed,
            }
        )
        code = (
            self.__class__.status_sequence.pop(0)
            if len(self.__class__.status_sequence) > 1
            else self.__class__.status_sequence[0]
        )
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "accepted", "code": code}).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        pass


class TestIurisIntegrationAndEventStore(unittest.TestCase):
    def test_mapper_and_idempotency_key_semantics(self) -> None:
        obs = ActivityObservationV1(
            event_id="evt-act-001",
            camera_id="camera_01",
            track_id="track_00042",
            timestamp="2026-10-26T10:30:00Z",
            activity=ActivityCode.RUNNING,
            confidence=0.91,
            observation_quality=0.88,
            duration_seconds=3.2,
            bbox=[120, 80, 250, 420],
            pose_available=True,
            model_version="yolov8n-pose-1.0",
            metadata={"frame": "MUST_BE_STRIPPED", "keypoints": "MUST_BE_STRIPPED", "speed_bh_per_sec": 2.4},
        )
        env = IurisEventMapper.map_activity(obs)
        self.assertEqual(env.schema_version, "1.0")
        self.assertEqual(env.event_id, "evt-act-001")
        self.assertEqual(env.event_type, "activity.detected")
        self.assertEqual(env.camera_id, "camera_01")
        self.assertEqual(env.track_id, "track_00042")
        self.assertNotIn("frame", env.payload["metadata"])
        self.assertNotIn("keypoints", env.payload["metadata"])
        self.assertEqual(
            env.idempotency_key,
            build_idempotency_key("evt-act-001", "activity.detected", "1.0"),
        )

        self.assertTrue(is_retryable_status_code(503))
        self.assertTrue(is_retryable_status_code(429))
        self.assertFalse(is_retryable_status_code(400))
        self.assertFalse(is_retryable_status_code(401))

        delay = compute_backoff_delay(
            attempt=2,
            initial_delay_sec=0.5,
            max_delay_sec=10.0,
            backoff_factor=2.0,
            jitter_ratio=0.1,
            jitter_fn=lambda low, high: high,
        )
        self.assertAlmostEqual(delay, 1.1)

    def test_bounded_event_store_deduplication_lifecycle_and_sqlite_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "events.sqlite"
            store = BoundedEventStore(
                EventStorageConfig(
                    max_activity_history=10,
                    max_summary_history=10,
                    max_risk_history=10,
                    sqlite_path=str(db_path),
                )
            )
            obs = ActivityObservationV1(
                event_id="dup-evt-1",
                camera_id="cam_01",
                track_id="trk_1",
                timestamp="2026-10-26T10:30:00Z",
                activity=ActivityCode.WALKING,
                observation_quality=0.90,
            )
            self.assertTrue(store.add_activity(obs))
            self.assertFalse(store.add_activity(obs))
            self.assertEqual(store.duplicates_ignored, 1)

            # Test risk alert lifecycle: TRIGGERED -> ACKNOWLEDGED -> RESOLVED
            risk = RiskEventV1(
                event_id="risk-lc-1",
                camera_id="cam_01",
                timestamp="2026-10-26T10:30:00Z",
                risk_type="UNUSUAL_ACTIVITY_CONCENTRATION",
                rule_name="unusual_running_concentration",
                severity=RiskSeverity.MEDIUM,
                explanation="Unusual running activity detected.",
            )
            self.assertTrue(store.add_risk_event(risk))
            self.assertEqual(store.count_active_risk_events("cam_01"), 1)

            ack = store.acknowledge_risk_event("risk-lc-1", actor="op_rahim", note="Investigating live feed")
            self.assertIsNotNone(ack)
            assert ack is not None
            self.assertEqual(ack.status, RiskStatus.ACKNOWLEDGED)
            self.assertEqual(ack.acknowledged_by, "op_rahim")
            self.assertEqual(store.count_active_risk_events("cam_01"), 1)

            res = store.resolve_risk_event("risk-lc-1", actor="op_rahim", note="Students rushing to bus")
            self.assertIsNotNone(res)
            assert res is not None
            self.assertEqual(res.status, RiskStatus.RESOLVED)
            self.assertEqual(res.resolution_note, "Students rushing to bus")
            self.assertEqual(store.count_active_risk_events("cam_01"), 0)

            store.close()
            self.assertTrue(db_path.exists())

    def test_outbound_delivery_retries_idempotency_and_mock_http_server(self) -> None:
        _MockIurisHttpHandler.received_requests = []
        # First 2 attempts return 503 Service Unavailable, 3rd attempt succeeds with 200 OK
        _MockIurisHttpHandler.status_sequence = [503, 503, 200]

        server = HTTPServer(("127.0.0.1", 0), _MockIurisHttpHandler)
        port = server.server_port
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                dl_path = Path(tmpdir) / "dead_letter.jsonl"
                sleep_delays: List[float] = []
                cfg = IurisConfig(
                    delivery_enabled=True,
                    base_url=f"http://127.0.0.1:{port}",
                    events_path="/v1/ingest/cctv",
                    api_token="secret_iuris_token",
                    timeout_seconds=2.0,
                    max_queue_size=10,
                    max_retry_attempts=4,
                    retry_initial_delay_sec=0.1,
                    retry_backoff_factor=2.0,
                    retry_jitter_ratio=0.0,
                    dead_letter_path=str(dl_path),
                )
                client = IurisDeliveryClient(
                    cfg, sleep_fn=sleep_delays.append, start_background_worker=False
                )

                risk_ev = RiskEventV1(
                    event_id="risk-evt-99",
                    camera_id="camera_01",
                    timestamp="2026-10-26T10:30:00Z",
                    risk_type="UNUSUAL_ACTIVITY_CONCENTRATION",
                    rule_name="unusual_running_concentration",
                    severity=RiskSeverity.MEDIUM,
                    status=RiskStatus.OPEN,
                    activity=ActivityCode.RUNNING,
                    measured_percentage=58.33,
                    threshold_percentage=40.0,
                    observed_people=12,
                    matching_people=7,
                    duration_seconds=5.0,
                    observation_quality=0.92,
                    explanation="The configured running-concentration threshold was met.",
                )

                self.assertTrue(client.enqueue(risk_ev))
                # Duplicate TRIGGERED enqueue with same event_id must be skipped
                self.assertFalse(client.enqueue(risk_ev))
                self.assertEqual(client.metrics().duplicates_skipped, 1)

                receipts = client.flush()
                self.assertEqual(len(receipts), 1)
                self.assertTrue(receipts[0].delivered)
                self.assertEqual(receipts[0].attempts, 3)
                self.assertEqual(sleep_delays, [0.1, 0.2])

                self.assertEqual(len(_MockIurisHttpHandler.received_requests), 3)
                req = _MockIurisHttpHandler.received_requests[-1]
                self.assertEqual(req["path"], "/v1/ingest/cctv")
                self.assertEqual(req["headers"].get("Authorization"), "Bearer secret_iuris_token")
                self.assertTrue(req["headers"].get("Idempotency-Key", "").startswith("iuris-risk-evt-99-"))
                self.assertEqual(req["body"]["risk_type"], "UNUSUAL_ACTIVITY_CONCENTRATION")
                self.assertEqual(req["body"]["matching_people"], 7)

                # Now transition the same alert to RESOLVED -> must be allowed through deduplication!
                resolved_ev = risk_ev.copy(deep=True)
                resolved_ev.status = RiskStatus.RESOLVED
                resolved_ev.lifecycle_state = AlertLifecycleState.RESOLVED.value
                resolved_ev.updated_at = "2026-10-26T10:30:15Z"
                self.assertTrue(client.enqueue(resolved_ev))
                res_receipts = client.flush()
                self.assertEqual(len(res_receipts), 1)
                self.assertTrue(res_receipts[0].delivered)

                m = client.metrics()
                self.assertEqual(m.events_delivered, 2)
                self.assertEqual(m.events_attempted, 4)
                self.assertEqual(m.events_retried, 2)
                self.assertEqual(m.events_failed, 0)
                self.assertGreater(m.mean_delivery_latency_ms, 0.0)
                client.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_non_retryable_401_and_timeout_or_connection_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dl_path = Path(tmpdir) / "auth_fail_dl.jsonl"
            sleep_calls: List[float] = []

            # Simulate 401 Unauthorized -> must NOT retry
            cfg_401 = IurisConfig(
                delivery_enabled=True,
                base_url="http://iuris.internal.test",
                max_retry_attempts=4,
                dead_letter_path=str(dl_path),
            )
            client_401 = IurisDeliveryClient(
                cfg_401,
                http_sender=lambda url, hdrs, body, timeout: (401, "Unauthorized token"),
                sleep_fn=sleep_calls.append,
            )
            obs = ActivityObservationV1(
                event_id="evt-401",
                camera_id="cam_01",
                track_id="trk_1",
                timestamp="2026-10-26T10:30:00Z",
                activity=ActivityCode.RUNNING,
                observation_quality=0.90,
            )
            client_401.enqueue(obs)
            rec = client_401.flush()[0]
            self.assertFalse(rec.delivered)
            self.assertEqual(rec.attempts, 1)
            self.assertEqual(sleep_calls, [])
            self.assertTrue(rec.dead_lettered)
            client_401.close()

            # Simulate TimeoutException -> retries max_retry_attempts times then dead-letters
            timeout_sleeps: List[float] = []

            def timeout_sender(url: str, hdrs: dict, body: dict, timeout: float) -> Tuple[int, str]:
                raise TimeoutError(f"Request timed out after {timeout}s")

            cfg_to = IurisConfig(
                delivery_enabled=True,
                base_url="http://iuris.internal.test",
                max_retry_attempts=2,
                retry_initial_delay_sec=0.05,
                retry_jitter_ratio=0.0,
                dead_letter_path=str(dl_path),
            )
            client_to = IurisDeliveryClient(
                cfg_to,
                http_sender=timeout_sender,
                sleep_fn=timeout_sleeps.append,
            )
            obs2 = ActivityObservationV1(
                event_id="evt-timeout",
                camera_id="cam_01",
                track_id="trk_1",
                timestamp="2026-10-26T10:30:01Z",
                activity=ActivityCode.RUNNING,
                observation_quality=0.90,
            )
            client_to.enqueue(obs2)
            rec_to = client_to.flush()[0]
            self.assertFalse(rec_to.delivered)
            self.assertEqual(rec_to.attempts, 3)
            self.assertEqual(len(timeout_sleeps), 2)
            self.assertEqual(client_to.metrics().events_failed, 1)
            client_to.close()

    def test_per_camera_fair_scheduling_prevents_flood_starvation_and_dry_run_mode(self) -> None:
        cfg = IurisConfig(
            delivery_enabled=True,
            dry_run=True,
            max_queue_size=4,
            backpressure_policy="dead_letter_and_drop_oldest",
            dead_letter_path=None,
        )
        client = IurisDeliveryClient(cfg, start_background_worker=False)

        # Camera "cam_flood" enqueues 4 low-priority activity observations (filling the queue of 4)
        for i in range(4):
            client.enqueue(
                ActivityObservationV1(
                    event_id=f"flood-{i}",
                    camera_id="cam_flood",
                    track_id=f"trk_{i}",
                    timestamp="2026-10-27T04:00:00Z",
                    activity=ActivityCode.WALKING,
                    observation_quality=0.85,
                )
            )

        # Now "cam_quiet" emits a critical risk event while queue is full ->
        # Fair eviction drops the oldest item from "cam_flood" (deepest queue), NOT "cam_quiet"!
        client.enqueue(
            RiskEventV1(
                event_id="quiet-risk-alert",
                camera_id="cam_quiet",
                timestamp="2026-10-27T04:00:01Z",
                risk_type="UNUSUAL_ACTIVITY_CONCENTRATION",
                rule_name="unusual_running_concentration",
                severity=RiskSeverity.HIGH,
                explanation="Unusual running activity detected on cam_quiet.",
            )
        )

        # Flush only 2 items: because of round-robin across cameras, cam_quiet's risk alert is delivered immediately!
        receipts = client.flush(max_items=2)
        delivered_ids = {r.event_id for r in receipts}
        self.assertIn("quiet-risk-alert", delivered_ids)
        self.assertTrue(all(r.dry_run for r in receipts))
        client.close(drain_remaining=False)

    def test_queue_saturation_and_graceful_shutdown_dead_letter_outbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dl_path = Path(tmpdir) / "saturation_dl.jsonl"
            cfg = IurisConfig(
                delivery_enabled=True,
                base_url="http://127.0.0.1:59999",
                max_queue_size=2,
                backpressure_policy="dead_letter_and_drop_oldest",
                dead_letter_path=str(dl_path),
            )
            client = IurisDeliveryClient(cfg, start_background_worker=False)

            for idx in range(3):
                summary = CameraActivitySummaryV1(
                    event_id=f"sum-{idx}",
                    camera_id="cam_01",
                    timestamp="2026-10-26T10:30:00Z",
                    window_seconds=10.0,
                    observed_people=5,
                    activity_distribution={"STANDING": 5},
                    activity_percentages={"STANDING": 100.0},
                    observation_quality=0.90,
                )
                client.enqueue(summary)

            m1 = client.metrics()
            self.assertEqual(m1.queue_depth, 2)
            self.assertEqual(m1.dead_letter_count, 1)

            client.close(drain_remaining=True)
            m2 = client.metrics()
            self.assertEqual(m2.queue_depth, 0)
            self.assertEqual(m2.dead_letter_count, 3)

            lines = dl_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
