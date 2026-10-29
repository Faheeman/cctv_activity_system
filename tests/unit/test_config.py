"""Unit tests for configuration validation, schema constraints, env overrides, and credential redaction."""

from pathlib import Path
import unittest
from pydantic import ValidationError

from activity_system.config import AppConfig, CameraConfig, ConfigLoader, TrackerConfig
from activity_system.logging_config import CredentialRedactingFormatter, configure_logging, redact_uri
from activity_system.schemas import (
    ActivityCode,
    ActivityEvent,
    ActivityLabel,
    ActivityObservationV1,
    CameraActivitySummaryV1,
    ObservationQuality,
    PoseDetection,
    RiskEventV1,
    RiskSeverity,
    SourceType,
)


class TestConfigAndLogging(unittest.TestCase):
    def setUp(self) -> None:
        self.default_config_path = (
            Path(__file__).resolve().parents[2] / "config" / "default.yaml"
        )
        self.risk_rules_path = (
            Path(__file__).resolve().parents[2] / "config" / "risk_rules.yaml"
        )

    def test_load_default_yaml_config(self) -> None:
        cfg = ConfigLoader.load(self.default_config_path, apply_env=False)
        self.assertIsInstance(cfg, AppConfig)
        self.assertEqual(cfg.system.app_name, "cctv-activity-recognition")
        self.assertEqual(len(cfg.cameras), 1)
        self.assertEqual(cfg.cameras[0].camera_id, "cam_corridor_01")
        self.assertEqual(cfg.activity.camera_viewpoint, "cctv")

    def test_env_overrides_and_risk_rules_yaml_loading(self) -> None:
        env = {
            "AI_BACKEND_HOST": "0.0.0.0",
            "AI_BACKEND_PORT": "8081",
            "AI_DEVICE": "auto",
            "AI_LOG_LEVEL": "DEBUG",
            "IURIS_DELIVERY_ENABLED": "true",
            "IURIS_BASE_URL": "https://iuris.internal.example/api",
            "IURIS_API_TOKEN": "sec_token_123",
            "EVENT_QUEUE_MAX_SIZE": "500",
            "EVENT_RETRY_MAX_ATTEMPTS": "6",
            "ACTIVITY_WINDOW_SECONDS": "12",
            "ACTIVITY_MIN_CONFIDENCE": "0.65",
            "RISK_RULES_PATH": str(self.risk_rules_path),
        }
        cfg = ConfigLoader.load(self.default_config_path, apply_env=True, environ=env)
        self.assertEqual(cfg.system.api_host, "0.0.0.0")
        self.assertEqual(cfg.system.api_port, 8081)
        self.assertEqual(cfg.pose.device, "auto")
        self.assertEqual(cfg.system.log_level, "DEBUG")
        self.assertTrue(cfg.iuris.delivery_enabled)
        self.assertEqual(cfg.iuris.base_url, "https://iuris.internal.example/api")
        self.assertEqual(cfg.iuris.api_token, "sec_token_123")
        self.assertEqual(cfg.iuris.max_queue_size, 500)
        self.assertEqual(cfg.iuris.max_retry_attempts, 6)
        self.assertAlmostEqual(cfg.summary.window_seconds, 12.0)
        self.assertAlmostEqual(cfg.pose.min_person_confidence, 0.65)
        self.assertEqual(cfg.risk.rules.unusual_running_concentration.activity, ActivityCode.RUNNING)
        self.assertAlmostEqual(
            cfg.risk.rules.unusual_running_concentration.percentage_threshold, 40.0
        )

    def test_v1_event_schemas_serialization_and_validation(self) -> None:
        ev = ActivityEvent(
            camera_id="camera_01",
            track_id="track_00042",
            timestamp="2026-10-26T10:30:00Z",
            activity=ActivityLabel.RUNNING,
            confidence=0.91,
            evidence_score=0.91,
            observation_quality=ObservationQuality.GOOD,
            observation_quality_score=0.88,
            duration_seconds=3.2,
            bbox=[120, 80, 250, 420],
            pose_available=True,
            model_version="yolov8n-pose-1.0",
        )
        v1_obs: ActivityObservationV1 = ev.to_v1()
        self.assertEqual(v1_obs.schema_version, "1.0")
        self.assertEqual(v1_obs.event_type, "activity.detected")
        self.assertEqual(v1_obs.activity, ActivityCode.RUNNING)
        self.assertAlmostEqual(v1_obs.observation_quality, 0.88)
        self.assertAlmostEqual(v1_obs.duration_seconds, 3.2)

        summary = CameraActivitySummaryV1(
            camera_id="camera_01",
            timestamp="2026-10-26T10:30:00Z",
            window_seconds=10,
            observed_people=20,
            activity_distribution={
                "STANDING": 8,
                "WALKING": 5,
                "RUNNING": 7,
                "SITTING": 0,
                "LYING": 0,
                "FALLING": 0,
                "UNKNOWN": 0,
            },
            activity_percentages={
                "STANDING": 40.0,
                "WALKING": 25.0,
                "RUNNING": 35.0,
            },
            observation_quality=0.92,
        )
        self.assertEqual(summary.event_type, "camera.activity_summary")
        self.assertEqual(summary.observed_people, 20)

        risk_ev = RiskEventV1(
            camera_id="camera_01",
            timestamp="2026-10-26T10:30:00Z",
            risk_type="UNUSUAL_ACTIVITY_CONCENTRATION",
            rule_name="unusual_running_concentration",
            severity=RiskSeverity.MEDIUM,
            activity="RUNNING",
            measured_percentage=70.0,
            threshold_percentage=70.0,
            observed_people=20,
            duration_seconds=5.0,
            observation_quality=0.92,
            explanation="The configured running-concentration threshold was met.",
        )
        self.assertEqual(risk_ev.event_type, "risk.detected")
        self.assertEqual(risk_ev.activity, ActivityCode.RUNNING)

    def test_duplicate_camera_ids_rejected(self) -> None:
        raw = {
            "cameras": [
                {"camera_id": "cam_01", "source_type": "webcam", "source_uri": 0},
                {"camera_id": "cam_01", "source_type": "file", "source_uri": "video.mp4"},
            ]
        }
        with self.assertRaises(ValidationError):
            ConfigLoader.from_dict(raw)

    def test_invalid_rtsp_uri_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CameraConfig(
                camera_id="cam_rtsp",
                source_type=SourceType.RTSP,
                source_uri="http://192.168.1.10:554/stream1",
            )

    def test_invalid_webcam_index_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CameraConfig(
                camera_id="cam_usb",
                source_type=SourceType.WEBCAM,
                source_uri=-1,
            )

    def test_invalid_tracker_thresholds_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TrackerConfig(high_thresh=0.40, low_thresh=0.50)

    def test_invalid_bbox_rejected_in_pose_detection(self) -> None:
        with self.assertRaises(ValidationError):
            PoseDetection(
                bbox=(200, 100, 150, 300),
                confidence=0.9,
                keypoints=[],
                observation_quality="good",
            )

    def test_rtsp_credential_redaction(self) -> None:
        raw_url = "rtsp://admin:SuperSecret123!@10.20.30.40:554/Streaming/Channels/101"
        redacted = redact_uri(raw_url)
        self.assertEqual(redacted, "rtsp://***:***@10.20.30.40:554/Streaming/Channels/101")
        self.assertNotIn("admin", redacted)
        self.assertNotIn("SuperSecret123!", redacted)

    def test_structured_logger_redacts_credentials(self) -> None:
        logger = configure_logging("INFO", logger_name="test_redaction_logger")
        formatter = CredentialRedactingFormatter()
        record = logger.makeRecord(
            name="test_redaction_logger",
            level=20,
            fn="video_source.py",
            lno=42,
            msg="Failed connecting to rtsp://operator:P@ssw0rd@192.168.1.50:554/live",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        self.assertIn("rtsp://***:***@192.168.1.50:554/live", formatted)
        self.assertNotIn("P@ssw0rd", formatted)


if __name__ == "__main__":
    unittest.main()
