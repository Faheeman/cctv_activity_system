"""Integration tests for the standalone FastAPI service, alert workflows, IURIS status, and crowd-running E2E."""

from pathlib import Path
import tempfile
import threading
import unittest
from fastapi.testclient import TestClient

from activity_system.api import create_api_app
from activity_system.config import (
    AppConfig,
    CameraConfig,
    ConcentrationRuleConfig,
    IurisConfig,
    PoseConfig,
    RiskEngineConfig,
    RiskRulesConfig,
)
from activity_system.main import ActivityApplication
from activity_system.schemas import ActivityCode, RiskSeverity, SourceType
from tests.fixtures.video_generator import generate_synthetic_cctv_video


class TestFastAPIStateServer(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.video_path = Path(self._tmpdir.name) / "api_cctv.avi"
        generate_synthetic_cctv_video(self.video_path, num_frames=18, width=640, height=480, fps=15.0)

        self.app_cfg = AppConfig(
            cameras=[
                CameraConfig(
                    camera_id="cam_api_01",
                    camera_name="Main Academic Building Corridor",
                    source_type=SourceType.FILE,
                    source_uri=str(self.video_path),
                    resize_width=640,
                    resize_height=480,
                )
            ],
            pose=PoseConfig(backend="synthetic"),
            risk=RiskEngineConfig(
                enabled=True,
                rules=RiskRulesConfig(
                    unusual_running_concentration=ConcentrationRuleConfig(
                        enabled=True,
                        activity=ActivityCode.WALKING,
                        percentage_threshold=50.0,
                        minimum_people=1,
                        window_seconds=10.0,
                        minimum_duration_seconds=0.0,
                        severity=RiskSeverity.LOW,
                    )
                ),
            ),
            iuris=IurisConfig(
                delivery_enabled=True,
                dry_run=True,
                deliver_activity_events=False,
                deliver_summary_events=False,
                deliver_risk_events=True,
            ),
        )
        self.pipeline_app = ActivityApplication(self.app_cfg)
        for _ in range(12):
            self.pipeline_app.step_camera("cam_api_01")

        self.fastapi_app = create_api_app(self.pipeline_app.snapshot_store)
        self.client = TestClient(self.fastapi_app)

    def tearDown(self) -> None:
        self.pipeline_app.shutdown()
        self._tmpdir.cleanup()

    def test_root_dashboard_and_frame_preview_endpoints(self) -> None:
        resp_root = self.client.get("/")
        self.assertEqual(resp_root.status_code, 200)
        self.assertIn("CCTV", resp_root.text)
        self.assertIn("/stream.mjpg", resp_root.text)

        resp_fav = self.client.get("/favicon.ico")
        self.assertEqual(resp_fav.status_code, 204)

        resp_jpg = self.client.get("/frame.jpg")
        self.assertEqual(resp_jpg.status_code, 200)
        self.assertEqual(resp_jpg.headers["content-type"], "image/jpeg")

    def test_mjpeg_live_stream_endpoint(self) -> None:
        resp_mjpg = self.client.get("/stream.mjpg", params={"camera_id": "cam_api_01", "max_chunks": 2})
        self.assertEqual(resp_mjpg.status_code, 200)
        self.assertIn("multipart/x-mixed-replace; boundary=frame", resp_mjpg.headers["content-type"])
        self.assertIn(b"--frame\r\nContent-Type: image/jpeg", resp_mjpg.content)

        resp_404 = self.client.get("/stream.mjpg", params={"camera_id": "unknown_cam", "max_chunks": 1})
        self.assertEqual(resp_404.status_code, 404)

    def test_health_and_ready_endpoints(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["healthy"])
        self.assertEqual(data["active_cameras"], 1)
        self.assertGreaterEqual(data["active_tracks"], 1)

        resp_ready = self.client.get("/ready")
        self.assertEqual(resp_ready.status_code, 200)
        ready_data = resp_ready.json()
        self.assertEqual(ready_data["status"], "ready")
        self.assertTrue(ready_data["ready"])
        self.assertTrue(ready_data["model_ready"])
        self.assertEqual(ready_data["cameras_configured"], 1)

    def test_v1_endpoints_alert_workflow_and_iuris_status(self) -> None:
        # 1. GET /v1/cameras
        r_cam = self.client.get("/v1/cameras", params={"limit": 10, "offset": 0})
        self.assertEqual(r_cam.status_code, 200)
        cam_json = r_cam.json()
        self.assertEqual(cam_json["schema_version"], "1.0")
        self.assertEqual(cam_json["pagination"]["total"], 1)
        self.assertEqual(cam_json["cameras"][0]["camera_id"], "cam_api_01")
        self.assertEqual(cam_json["cameras"][0]["camera_name"], "Main Academic Building Corridor")
        self.assertEqual(cam_json["cameras"][0]["frame_width"], 640)
        self.assertEqual(cam_json["cameras"][0]["frame_height"], 480)

        # 2. GET /v1/tracks
        r_trk = self.client.get("/v1/tracks", params={"camera_id": "cam_api_01"})
        self.assertEqual(r_trk.status_code, 200)
        trk_json = r_trk.json()
        self.assertEqual(trk_json["state_source"], "live_in_memory")
        self.assertGreaterEqual(trk_json["pagination"]["count"], 1)

        # 3. GET /v1/activities (live & history)
        r_act_live = self.client.get("/v1/activities", params={"source": "live"})
        self.assertEqual(r_act_live.status_code, 200)
        act_live_json = r_act_live.json()
        self.assertEqual(act_live_json["state_source"], "live_in_memory")
        self.assertGreaterEqual(act_live_json["pagination"]["count"], 1)

        r_act_hist = self.client.get("/v1/activities", params={"source": "history", "limit": 5})
        self.assertEqual(r_act_hist.status_code, 200)
        act_hist_json = r_act_hist.json()
        self.assertEqual(act_hist_json["state_source"], "event_history")
        self.assertEqual(len(act_hist_json["activities"]), 5)

        # 4. GET /v1/camera-summaries
        r_sum = self.client.get("/v1/camera-summaries")
        self.assertEqual(r_sum.status_code, 200)
        sum_json = r_sum.json()
        self.assertEqual(sum_json["pagination"]["count"], 1)

        # 5. GET /v1/risk-events & /v1/risk-events/active
        r_risk = self.client.get("/v1/risk-events", params={"camera_id": "cam_api_01"})
        self.assertEqual(r_risk.status_code, 200)
        risk_json = r_risk.json()
        self.assertGreaterEqual(risk_json["pagination"]["total"], 1)
        alert = risk_json["risk_events"][0]
        event_id = alert["event_id"]
        self.assertEqual(alert["camera_name"], "Main Academic Building Corridor")
        self.assertEqual(alert["status"], "OPEN")

        r_active = self.client.get("/v1/risk-events/active")
        self.assertEqual(r_active.status_code, 200)
        self.assertGreaterEqual(r_active.json()["pagination"]["total"], 1)

        # 6. Acknowledge -> Resolve workflow
        r_ack = self.client.post(
            f"/v1/risk-events/{event_id}/acknowledge",
            json={"actor": "officer_karim", "note": "Reviewing camera stream"},
        )
        self.assertEqual(r_ack.status_code, 200)
        ack_data = r_ack.json()
        self.assertEqual(ack_data["status"], "ACKNOWLEDGED")
        self.assertEqual(ack_data["acknowledged_by"], "officer_karim")

        r_res = self.client.post(
            f"/v1/risk-events/{event_id}/resolve",
            json={"actor": "officer_karim", "note": "Verified normal class dismissal"},
        )
        self.assertEqual(r_res.status_code, 200)
        res_data = r_res.json()
        self.assertEqual(res_data["status"], "RESOLVED")
        self.assertEqual(res_data["resolution_note"], "Verified normal class dismissal")
        self.assertGreaterEqual(len(res_data["status_history"]), 3)

        # Active alerts should now be 0
        r_active_after = self.client.get("/v1/risk-events/active")
        self.assertEqual(r_active_after.json()["pagination"]["total"], 0)

        # 7. GET /v1/iuris/status
        r_iuris = self.client.get("/v1/iuris/status")
        self.assertEqual(r_iuris.status_code, 200)
        iuris_json = r_iuris.json()
        self.assertEqual(iuris_json["status"], "dry_run")
        self.assertGreaterEqual(iuris_json["iuris_delivery"]["events_delivered"], 3)

    def test_cameras_endpoint_and_filtering(self) -> None:
        resp = self.client.get("/cameras")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["cameras"][0]["camera_id"], "cam_api_01")
        self.assertEqual(data["cameras"][0]["frames_emitted"], 12)

        resp_404 = self.client.get("/cameras", params={"camera_id": "nonexistent_cam"})
        self.assertEqual(resp_404.status_code, 404)

    def test_tracks_endpoint_and_validation(self) -> None:
        resp = self.client.get("/tracks")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["count"], 1)
        trk = data["tracks"][0]
        self.assertEqual(trk["camera_id"], "cam_api_01")
        self.assertIn("cam_api_01_s1_track_", trk["track_id"])
        self.assertEqual(trk["lifecycle"], "confirmed")

        self.assertEqual(self.client.get("/tracks", params={"camera_id": "bad_cam"}).status_code, 404)

    def test_activities_endpoint_and_privacy_contract(self) -> None:
        resp = self.client.get("/activities")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["count"], 1)
        act = data["activities"][0]
        self.assertEqual(act["camera_id"], "cam_api_01")
        self.assertIsNone(act["confidence"])
        self.assertIn(act["activity"], ["standing", "walking", "running", "sitting", "lying", "falling", "unknown"])
        self.assertNotIn("frame", act)
        self.assertNotIn("face", act)

        resp_422 = self.client.get("/activities", params={"activity": "flying"})
        self.assertEqual(resp_422.status_code, 422)

    def test_metrics_and_prometheus_endpoints_and_concurrent_reads(self) -> None:
        errors = []

        def reader_worker() -> None:
            try:
                for _ in range(15):
                    r1 = self.client.get("/metrics")
                    r2 = self.client.get("/metrics/prometheus")
                    if r1.status_code != 200 or r2.status_code != 200:
                        errors.append((r1.status_code, r2.status_code))
            except Exception as exc:
                errors.append(str(exc))

        t = threading.Thread(target=reader_worker)
        t.start()
        for _ in range(5):
            self.pipeline_app.step_camera("cam_api_01")
        t.join()

        self.assertEqual(errors, [])
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        m = resp.json()["metrics"]
        self.assertEqual(m["frames_processed"], 17)
        self.assertGreater(m["throughput_fps"], 0.0)
        self.assertIn("active_risk_events", m)
        self.assertIn("duplicate_alerts_suppressed", m)

        prom_resp = self.client.get("/metrics/prometheus")
        self.assertEqual(prom_resp.status_code, 200)
        self.assertIn("cctv_frames_processed_total 17", prom_resp.text)
        self.assertIn("cctv_risk_events_active_count", prom_resp.text)
        self.assertIn("cctv_iuris_deliveries_succeeded_total", prom_resp.text)


if __name__ == "__main__":
    unittest.main()
