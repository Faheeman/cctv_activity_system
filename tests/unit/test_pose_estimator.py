"""Unit tests for PoseEstimator validation, keypoint filtering, and ONNX tensor decoding."""

import unittest
import numpy as np

from activity_system.config import PoseConfig
from activity_system.pose_estimator import (
    PoseEstimator,
    RawPoseCandidate,
    SyntheticPoseBackend,
    build_human_coco_keypoints,
    decode_yolo_pose_output,
)
from activity_system.schemas import FramePacket, ObservationQuality


class TestPoseEstimatorUnit(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PoseConfig(
            backend="synthetic",
            min_person_confidence=0.50,
            min_keypoint_confidence=0.35,
            imgsz=640,
        )

    def test_empty_black_frame_returns_no_detections(self) -> None:
        estimator = PoseEstimator(self.config)
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pkt = FramePacket("cam_01", 1, 1, empty_frame, monotonic_ts=1.0)
        detections = estimator.estimate(pkt)
        self.assertEqual(detections, [])

    def test_low_confidence_person_filtered_out(self) -> None:
        kpts = build_human_coco_keypoints(320.0, 240.0, 180.0, confidence=0.90)
        low_cand = RawPoseCandidate(bbox=(260.0, 150.0, 380.0, 330.0), score=0.35, keypoints=kpts)
        high_cand = RawPoseCandidate(bbox=(100.0, 120.0, 200.0, 320.0), score=0.88, keypoints=kpts)

        backend = SyntheticPoseBackend(custom_generator=lambda _f, _c: [low_cand, high_cand])
        estimator = PoseEstimator(self.config, backend=backend)

        frame = np.full((480, 640, 3), 60, dtype=np.uint8)
        pkt = FramePacket("cam_01", 1, 1, frame, monotonic_ts=1.0)
        detections = estimator.estimate(pkt)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.88, places=4)

    def test_bbox_clamping_and_keypoint_visibility_filtering(self) -> None:
        kpts = build_human_coco_keypoints(320.0, 240.0, 180.0, confidence=0.90, occlude_lower_body=True)
        kpts[0, 0] = -25.0
        kpts[0, 1] = -10.0

        cand = RawPoseCandidate(bbox=(-30.0, -20.0, 700.0, 520.0), score=0.95, keypoints=kpts)
        backend = SyntheticPoseBackend(custom_generator=lambda _f, _c: [cand])
        estimator = PoseEstimator(self.config, backend=backend)

        frame = np.full((480, 640, 3), 60, dtype=np.uint8)
        pkt = FramePacket("cam_01", 1, 1, frame, monotonic_ts=1.0)
        detections = estimator.estimate(pkt)
        self.assertEqual(len(detections), 1)
        det = detections[0]

        self.assertEqual(det.bbox, (0, 0, 640, 480))
        self.assertFalse(det.keypoints[0].visible)
        self.assertFalse(det.keypoints[15].visible)
        self.assertFalse(det.keypoints[16].visible)
        self.assertTrue(det.keypoints[5].visible)
        self.assertEqual(det.observation_quality, ObservationQuality.PARTIAL)

    def test_decode_yolo_pose_output_tensor(self) -> None:
        tensor = np.zeros((1, 56, 3), dtype=np.float32)
        tensor[0, 0, 0] = 320.0
        tensor[0, 1, 0] = 240.0
        tensor[0, 2, 0] = 80.0
        tensor[0, 3, 0] = 200.0
        tensor[0, 4, 0] = 0.91
        kpts = build_human_coco_keypoints(320.0, 240.0, 180.0, confidence=0.85)
        tensor[0, 5:56, 0] = kpts.reshape(-1)

        tensor[0, :, 1] = tensor[0, :, 0]
        tensor[0, 4, 1] = 0.62

        candidates = decode_yolo_pose_output(
            tensor,
            orig_width=640,
            orig_height=480,
            imgsz=640,
            conf_thresh=0.50,
            iou_thresh=0.45,
        )
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].score, 0.91, places=4)
        self.assertEqual(candidates[0].keypoints.shape, (17, 3))


if __name__ == "__main__":
    unittest.main()
