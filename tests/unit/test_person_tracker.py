"""Unit tests for PersonTracker (ByteTrack) and TrackStateManager."""

import unittest
import numpy as np

from activity_system.config import TrackerConfig
from activity_system.person_tracker import PersonTracker
from activity_system.pose_estimator import build_human_coco_keypoints
from activity_system.schemas import (
    COCO_KEYPOINT_NAMES,
    FramePacket,
    Keypoint,
    ObservationQuality,
    PoseDetection,
    TrackLifecycle,
)


def _make_detection(
    cx: float,
    cy: float,
    body_h: float = 160.0,
    conf: float = 0.90,
) -> PoseDetection:
    kpts_arr = build_human_coco_keypoints(cx, cy, body_h, confidence=conf)
    kpts = [
        Keypoint(
            index=i,
            name=COCO_KEYPOINT_NAMES[i],
            x=float(kpts_arr[i, 0]),
            y=float(kpts_arr[i, 1]),
            confidence=float(kpts_arr[i, 2]),
            visible=True,
        )
        for i in range(17)
    ]
    w = body_h * 0.45
    bbox = (
        int(round(cx - 0.5 * w)),
        int(round(cy - 0.5 * body_h)),
        int(round(cx + 0.5 * w)),
        int(round(cy + 0.5 * body_h)),
    )
    return PoseDetection(
        bbox=bbox,
        confidence=conf,
        keypoints=kpts,
        observation_quality=ObservationQuality.GOOD,
    )


class TestPersonTrackerUnit(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.config = TrackerConfig(
            high_thresh=0.60,
            low_thresh=0.25,
            match_iou_thresh=0.30,
            second_match_iou_thresh=0.20,
            min_hits_to_confirm=2,
            track_expiration_seconds=1.5,
            max_window_duration_seconds=1.0,
            max_window_observations=10,
        )

    def test_persistent_multi_person_track_ids_and_confirmation(self) -> None:
        tracker = PersonTracker(self.config)
        # Frame 1: two people detected
        pkt1 = FramePacket("cam_01", 1, 1, self.dummy_frame, monotonic_ts=1.0)
        dets1, tracks1 = tracker.update(pkt1, [_make_detection(150, 240), _make_detection(450, 240)])
        self.assertEqual(len(tracks1), 2)
        self.assertEqual(tracks1[0].track_id, "cam_01_s1_track_1")
        self.assertEqual(tracks1[1].track_id, "cam_01_s1_track_2")
        self.assertEqual(tracks1[0].lifecycle, TrackLifecycle.TENTATIVE)

        # Frame 2: both move slightly -> tracks match and transition to CONFIRMED
        pkt2 = FramePacket("cam_01", 1, 2, self.dummy_frame, monotonic_ts=1.1)
        dets2, tracks2 = tracker.update(pkt2, [_make_detection(156, 240), _make_detection(444, 240)])
        self.assertEqual(dets2[0].track_id, "cam_01_s1_track_1")
        self.assertEqual(dets2[1].track_id, "cam_01_s1_track_2")
        self.assertEqual(tracks2[0].lifecycle, TrackLifecycle.CONFIRMED)
        self.assertEqual(tracks2[1].lifecycle, TrackLifecycle.CONFIRMED)

    def test_bytetrack_low_confidence_second_stage_and_short_occlusion(self) -> None:
        tracker = PersonTracker(self.config)
        # Confirm track 1 over 2 frames
        tracker.update(
            FramePacket("cam_01", 1, 1, self.dummy_frame, monotonic_ts=1.0),
            [_make_detection(200, 240, conf=0.92)],
        )
        tracker.update(
            FramePacket("cam_01", 1, 2, self.dummy_frame, monotonic_ts=1.1),
            [_make_detection(208, 240, conf=0.90)],
        )

        # Frame 3: partial occlusion causes low detection confidence (0.35 < high_thresh 0.60)
        dets3, tracks3 = tracker.update(
            FramePacket("cam_01", 1, 3, self.dummy_frame, monotonic_ts=1.2),
            [_make_detection(216, 240, conf=0.35)],
        )
        self.assertEqual(len(dets3), 1)
        self.assertEqual(dets3[0].track_id, "cam_01_s1_track_1")
        self.assertEqual(tracks3[0].lifecycle, TrackLifecycle.CONFIRMED)

        # Frame 4: complete 1-frame detection dropout -> track becomes OCCLUDED but not expired
        dets4, tracks4 = tracker.update(
            FramePacket("cam_01", 1, 4, self.dummy_frame, monotonic_ts=1.3),
            [],
        )
        self.assertEqual(len(dets4), 0)
        self.assertEqual(len(tracks4), 1)
        self.assertEqual(tracks4[0].lifecycle, TrackLifecycle.OCCLUDED)

        # Frame 5: person reappears near predicted position -> same track_id recovered
        dets5, tracks5 = tracker.update(
            FramePacket("cam_01", 1, 5, self.dummy_frame, monotonic_ts=1.4),
            [_make_detection(230, 240, conf=0.88)],
        )
        self.assertEqual(dets5[0].track_id, "cam_01_s1_track_1")
        self.assertEqual(tracks5[0].lifecycle, TrackLifecycle.CONFIRMED)

    def test_track_expiration_and_bounded_window_pruning(self) -> None:
        tracker = PersonTracker(self.config)
        # Feed 8 observations spaced by 0.2s (total span 1.4s > max_window_duration_seconds 1.0s)
        for idx in range(8):
            ts = 1.0 + idx * 0.2
            _, tracks = tracker.update(
                FramePacket("cam_01", 1, idx + 1, self.dummy_frame, monotonic_ts=ts),
                [_make_detection(200 + idx * 4, 240, conf=0.90)],
            )
        # Window duration is capped at 1.0s -> observations older than (2.4 - 1.0 = 1.4s) pruned
        self.assertLessEqual(len(tracks[0].observations), 6)

        # Now advance timestamp by 2.0s (> track_expiration_seconds 1.5s) with no detections
        _, tracks_after = tracker.update(
            FramePacket("cam_01", 1, 9, self.dummy_frame, monotonic_ts=4.5),
            [],
        )
        self.assertEqual(len(tracks_after), 0)

    def test_per_camera_isolation_and_session_reset_on_reconnect(self) -> None:
        tracker = PersonTracker(self.config)
        tracker.update(
            FramePacket("cam_A", 1, 1, self.dummy_frame, monotonic_ts=1.0),
            [_make_detection(200, 240)],
        )
        tracker.update(
            FramePacket("cam_B", 1, 1, self.dummy_frame, monotonic_ts=1.0),
            [_make_detection(200, 240)],
        )

        # Simulate RTSP reconnection on cam_A (session_id increments to 2)
        dets_a, _ = tracker.update(
            FramePacket("cam_A", 2, 2, self.dummy_frame, monotonic_ts=2.0),
            [_make_detection(200, 240)],
        )
        self.assertEqual(dets_a[0].track_id, "cam_A_s2_track_1")

        # cam_B remains on session 1 undisturbed
        dets_b, tracks_b = tracker.update(
            FramePacket("cam_B", 1, 2, self.dummy_frame, monotonic_ts=1.1),
            [_make_detection(204, 240)],
        )
        self.assertEqual(dets_b[0].track_id, "cam_B_s1_track_1")
        self.assertEqual(tracks_b[0].hits, 2)


if __name__ == "__main__":
    unittest.main()
