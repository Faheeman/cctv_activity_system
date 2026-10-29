"""Unit tests for Renderer skeleton, bounding box, and telemetry overlays."""

import unittest
import numpy as np

from activity_system.config import PoseConfig, RendererConfig
from activity_system.pose_estimator import PoseEstimator, RawPoseCandidate, SyntheticPoseBackend, build_human_coco_keypoints
from activity_system.renderer import Renderer
from activity_system.schemas import ActivityEvent, ActivityLabel, FramePacket, ObservationQuality


class TestRendererUnit(unittest.TestCase):
    def test_renderer_draws_annotations_without_mutating_input_when_copy_true(self) -> None:
        frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        kpts = build_human_coco_keypoints(320.0, 240.0, 180.0, confidence=0.95)
        cand = RawPoseCandidate(bbox=(270.0, 140.0, 370.0, 340.0), score=0.92, keypoints=kpts)
        estimator = PoseEstimator(
            PoseConfig(),
            backend=SyntheticPoseBackend(custom_generator=lambda _f, _c: [cand]),
        )
        pkt = FramePacket("cam_01", 1, 1, frame, monotonic_ts=1.0)
        detections = estimator.estimate(pkt)
        detections[0].track_id = "cam_01_s1_track_1"

        event = ActivityEvent(
            camera_id="cam_01",
            track_id="cam_01_s1_track_1",
            timestamp="2026-10-22T20:45:00Z",
            activity=ActivityLabel.WALKING,
            confidence=None,
            evidence_score=0.85,
            observation_quality=ObservationQuality.GOOD,
            bbox=list(detections[0].bbox),
            pose_available=True,
        )

        renderer = Renderer(RendererConfig())
        annotated = renderer.render(
            frame,
            detections=detections,
            events_by_track={"cam_01_s1_track_1": event},
            fps=24.5,
            camera_id="cam_01",
            stream_state="connected",
            copy=True,
        )

        self.assertTrue(np.all(frame == 30))
        self.assertFalse(np.all(annotated == 30))
        self.assertEqual(annotated.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
