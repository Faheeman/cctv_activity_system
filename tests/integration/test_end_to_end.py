"""End-to-end integration and reliability tests for ActivityApplication (including crowd-running IURIS E2E)."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from activity_system.config import AppConfig, CameraConfig, PoseConfig
from activity_system.main import ActivityApplication
from activity_system.schemas import SourceType, StreamState
from scripts.demo_crowd_running_alert import run_crowd_running_demo
from tests.fixtures.video_generator import generate_synthetic_cctv_video
from tests.unit.test_video_source import FakeCapture


class TestEndToEndApplication(unittest.TestCase):
    def test_end_to_end_video_processing_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_video = tmp / "input_cctv.avi"
            output_video = tmp / "annotated_out.avi"
            output_jsonl = tmp / "events.jsonl"

            generate_synthetic_cctv_video(input_video, num_frames=25, width=640, height=480, fps=15.0)

            app_cfg = AppConfig(
                cameras=[
                    CameraConfig(
                        camera_id="cam_e2e_01",
                        source_type=SourceType.FILE,
                        source_uri=str(input_video),
                        resize_width=640,
                        resize_height=480,
                    )
                ],
                pose=PoseConfig(backend="synthetic"),
            )
            app = ActivityApplication(app_cfg)
            final_snap = app.run(
                max_frames_total=25,
                output_video_path=output_video,
                events_jsonl_path=output_jsonl,
            )

            self.assertTrue(output_video.exists())
            self.assertGreater(output_video.stat().st_size, 1024)
            self.assertTrue(output_jsonl.exists())

            lines = output_jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 20)
            first_ev = json.loads(lines[0])
            self.assertEqual(first_ev["camera_id"], "cam_e2e_01")
            self.assertIn("cam_e2e_01_s1_track_", first_ev["track_id"])
            self.assertIsNone(first_ev["confidence"])
            self.assertIn(first_ev["activity"], ["standing", "walking", "running", "unknown"])

            self.assertEqual(final_snap.metrics.frames_processed, 25)
            self.assertGreater(final_snap.metrics.throughput_fps, 0.0)
            self.assertGreater(final_snap.metrics.memory_rss_mb, 0.0)

    def test_multi_camera_fault_isolation_when_one_camera_fails(self) -> None:
        good_frames = [np.full((240, 320, 3), 180, dtype=np.uint8) for _ in range(10)]
        app_cfg = AppConfig(
            cameras=[
                CameraConfig(
                    camera_id="cam_good",
                    source_type=SourceType.FILE,
                    source_uri="good.avi",
                    resize_width=320,
                    resize_height=240,
                ),
                CameraConfig(
                    camera_id="cam_bad_rtsp",
                    source_type=SourceType.RTSP,
                    source_uri="rtsp://10.0.0.250:554/offline",
                    reconnect_initial_delay_sec=0.0,
                    max_reconnect_attempts=2,
                    resize_width=320,
                    resize_height=240,
                ),
            ]
        )
        app = ActivityApplication(
            app_cfg,
            capture_factory_by_camera={
                "cam_good": lambda: FakeCapture(good_frames, opened=True),
                "cam_bad_rtsp": lambda: FakeCapture([], opened=False),
            },
        )
        snap = app.run()
        cam_states = {c.camera_id: c for c in snap.cameras}
        self.assertEqual(cam_states["cam_good"].frames_emitted, 10)
        self.assertEqual(cam_states["cam_good"].state, StreamState.EOF)
        self.assertEqual(cam_states["cam_bad_rtsp"].state, StreamState.FAILED)
        self.assertEqual(snap.metrics.frames_processed, 10)

    def test_end_to_end_12_person_crowd_running_alert_and_iuris_delivery(self) -> None:
        report = run_crowd_running_demo()
        alert = report["triggered_alert"]
        self.assertEqual(alert["camera_id"], "cam_iu_main_gate")
        self.assertEqual(alert["camera_name"], "Islamic University Main Gate")
        self.assertEqual(alert["observed_people"], 12)
        self.assertEqual(alert["matching_people"], 7)
        self.assertAlmostEqual(alert["measured_percentage"], 58.33, places=2)
        self.assertGreaterEqual(alert["duration_seconds"], 5.0)
        self.assertGreater(report["duplicate_alerts_suppressed"], 5)
        self.assertEqual(
            report["iuris_lifecycle_states_received"],
            ["TRIGGERED", "ACKNOWLEDGED", "RESOLVED"],
        )
        self.assertEqual(report["final_alert_status"], "RESOLVED")


if __name__ == "__main__":
    unittest.main()
