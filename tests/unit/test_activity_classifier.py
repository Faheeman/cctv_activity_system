"""Unit tests for ActivityClassifier (including SITTING, LYING, FALLING) and CameraSummaryAggregator."""

import unittest
import numpy as np

from activity_system.activity_classifier import ActivityClassifier
from activity_system.camera_summary import CameraSummaryAggregator
from activity_system.config import ActivityConfig, SummaryConfig, TrackerConfig
from activity_system.person_tracker import PersonTracker
from activity_system.pose_estimator import build_human_coco_keypoints
from activity_system.schemas import (
    ActivityCode,
    ActivityEvent,
    ActivityLabel,
    COCO_KEYPOINT_NAMES,
    FramePacket,
    Keypoint,
    ObservationQuality,
    PoseDetection,
)
from tests.fixtures.activity_sequences import _build_pose_detection, build_evaluation_dataset


def _make_custom_posture_detection(
    cx: float,
    cy: float,
    body_h: float,
    posture: str,
    bbox_w: float,
    bbox_h: float,
) -> PoseDetection:
    kpts_arr = build_human_coco_keypoints(
        center_x=cx,
        center_y=cy,
        body_height=body_h,
        posture=posture,
        confidence=0.92,
    )
    kpts = [
        Keypoint(
            index=i,
            name=COCO_KEYPOINT_NAMES[i],
            x=float(kpts_arr[i, 0]),
            y=float(kpts_arr[i, 1]),
            confidence=float(kpts_arr[i, 2]),
            visible=bool(kpts_arr[i, 2] >= 0.35),
        )
        for i in range(17)
    ]
    bbox = (
        max(0, int(round(cx - 0.5 * bbox_w))),
        max(0, int(round(cy - 0.5 * bbox_h))),
        min(640, int(round(cx + 0.5 * bbox_w))),
        min(480, int(round(cy + 0.5 * bbox_h))),
    )
    return PoseDetection(
        bbox=bbox,
        confidence=0.92,
        keypoints=kpts,
        observation_quality=ObservationQuality.GOOD,
    )


class TestActivityClassifierUnit(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker_cfg = TrackerConfig(min_hits_to_confirm=2)
        self.activity_cfg = ActivityConfig(
            min_observations=6,
            min_window_duration_seconds=0.50,
            hysteresis_count=3,
        )

    def test_insufficient_observations_returns_unknown(self) -> None:
        tracker = PersonTracker(self.tracker_cfg)
        classifier = ActivityClassifier(self.activity_cfg)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Feed only 4 frames (< min_observations=6)
        for i in range(4):
            pkt = FramePacket("cam_01", 1, i + 1, dummy_img, monotonic_ts=1.0 + i * 0.1)
            det = _build_pose_detection(200.0 + i * 8.0, 240.0, 150.0, i * 0.5, 0.12, 0.05)
            _, tracks = tracker.update(pkt, [det])
            event = classifier.classify_track(tracks[0])
            self.assertEqual(event.activity, ActivityLabel.UNKNOWN)
            self.assertIsNone(event.confidence)

    def test_perspective_scale_invariance_between_near_and_far_walking(self) -> None:
        dataset = {seq.sequence_id: seq for seq in build_evaluation_dataset()}
        classifier = ActivityClassifier(self.activity_cfg)

        for seq_id in ("seq_walking_near_01", "seq_walking_far_01"):
            tracker = PersonTracker(self.tracker_cfg)
            seq = dataset[seq_id]
            last_event = None
            for pkt, dets in seq.frames:
                _, tracks = tracker.update(pkt, dets)
                last_event = classifier.classify_track(tracks[0])
            self.assertIsNotNone(last_event)
            assert last_event is not None
            self.assertEqual(last_event.activity, ActivityLabel.WALKING)

    def test_hysteresis_prevents_single_frame_label_flicker(self) -> None:
        dataset = {seq.sequence_id: seq for seq in build_evaluation_dataset()}
        walk_seq = dataset["seq_walking_near_01"]
        tracker = PersonTracker(self.tracker_cfg)
        classifier = ActivityClassifier(self.activity_cfg)

        # Warm up to stable WALKING state (first 15 frames)
        for pkt, dets in walk_seq.frames[:15]:
            _, tracks = tracker.update(pkt, dets)
            ev = classifier.classify_track(tracks[0])
        self.assertEqual(ev.activity, ActivityLabel.WALKING)

        # Inject 1 frame with occluded lower-body keypoints (raw rule outputs UNKNOWN, but hysteresis_count=3 holds WALKING)
        pkt_occ, _ = walk_seq.frames[15]
        occ_det = _build_pose_detection(
            cx=80.0 + 15 * 10.0,
            cy=260.0,
            body_h=160.0,
            phase_rad=15 * 0.55,
            stride_amp_bh=0.11,
            knee_lift_bh=0.045,
            occlude_lower_body=True,
        )
        _, tracks = tracker.update(pkt_occ, [occ_det])
        ev_after_1_glitch = classifier.classify_track(tracks[0])
        self.assertEqual(ev_after_1_glitch.activity, ActivityLabel.WALKING)

    def test_sitting_lying_and_falling_distinction_and_single_frame_guard(self) -> None:
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        classifier = ActivityClassifier(self.activity_cfg)

        # 1. Single horizontal frame must NEVER be classified as FALLING
        tracker_single = PersonTracker(self.tracker_cfg)
        pkt1 = FramePacket("cam_01", 1, 1, dummy_img, monotonic_ts=1.0)
        det_horiz = _make_custom_posture_detection(300.0, 320.0, 150.0, "lying", bbox_w=150.0, bbox_h=55.0)
        _, trks = tracker_single.update(pkt1, [det_horiz])
        ev_single = classifier.classify_track(trks[0])
        self.assertEqual(ev_single.activity, ActivityLabel.UNKNOWN)
        self.assertNotEqual(ev_single.activity, ActivityLabel.FALLING)

        # 2. Sustained horizontal person over 12 frames (~0.8s) -> LYING (distinct from FALLING)
        tracker_lying = PersonTracker(self.tracker_cfg)
        last_lying_ev = None
        for i in range(12):
            pkt = FramePacket("cam_01", 1, i + 1, dummy_img, monotonic_ts=10.0 + i * 0.08)
            det = _make_custom_posture_detection(300.0, 340.0, 150.0, "lying", bbox_w=150.0, bbox_h=55.0)
            _, trks = tracker_lying.update(pkt, [det])
            last_lying_ev = classifier.classify_track(trks[0])
        assert last_lying_ev is not None
        self.assertEqual(last_lying_ev.activity, ActivityLabel.LYING)
        self.assertEqual(last_lying_ev.to_v1().activity, ActivityCode.LYING)

        # 3. Sustained seated person over 12 frames (~0.8s) -> SITTING
        tracker_sit = PersonTracker(self.tracker_cfg)
        last_sit_ev = None
        for i in range(12):
            pkt = FramePacket("cam_01", 1, i + 1, dummy_img, monotonic_ts=20.0 + i * 0.08)
            det = _make_custom_posture_detection(300.0, 280.0, 140.0, "sitting", bbox_w=85.0, bbox_h=110.0)
            _, trks = tracker_sit.update(pkt, [det])
            last_sit_ev = classifier.classify_track(trks[0])
        assert last_sit_ev is not None
        self.assertEqual(last_sit_ev.activity, ActivityLabel.SITTING)
        self.assertEqual(last_sit_ev.to_v1().activity, ActivityCode.SITTING)

        # 4. Multi-frame rapid vertical collapse from upright to horizontal (continuous overlapping boxes) -> FALLING
        tracker_fall = PersonTracker(self.tracker_cfg)
        last_fall_ev = None
        for i in range(10):
            ts = 30.0 + i * 0.07
            pkt = FramePacket("cam_01", 1, i + 1, dummy_img, monotonic_ts=ts)
            frac = min(1.0, max(0.0, (i - 1) / 6.0))
            cy = 220.0 + frac * 115.0
            bw = 65.0 + frac * 85.0
            bh = 155.0 - frac * 95.0
            posture = "upright" if frac < 0.35 else ("sitting" if frac < 0.70 else "lying")
            det = _make_custom_posture_detection(300.0, cy, 140.0, posture, bbox_w=bw, bbox_h=bh)
            _, trks = tracker_fall.update(pkt, [det])
            self.assertEqual(len(trks), 1)
            last_fall_ev = classifier.classify_track(trks[0])
        assert last_fall_ev is not None
        self.assertEqual(last_fall_ev.activity, ActivityLabel.FALLING)
        self.assertEqual(last_fall_ev.to_v1().activity, ActivityCode.FALLING)

    def test_camera_summary_aggregator_deduplication_and_percentages(self) -> None:
        agg = CameraSummaryAggregator(
            SummaryConfig(window_seconds=10.0, exclude_unknown_from_percentages=True)
        )
        # Create 20 unique tracks (8 STANDING, 5 WALKING, 7 RUNNING) observed across multiple frames in a 10s window
        events_frame_1 = []
        for idx in range(20):
            if idx < 8:
                act = ActivityLabel.STANDING
            elif idx < 13:
                act = ActivityLabel.WALKING
            else:
                act = ActivityLabel.RUNNING
            events_frame_1.append(
                ActivityEvent(
                    camera_id="camera_01",
                    track_id=f"track_{idx:05d}",
                    timestamp="2026-10-26T10:29:55Z",
                    activity=act,
                    observation_quality=ObservationQuality.GOOD,
                    observation_quality_score=0.92,
                    bbox=[10, 10, 50, 100],
                    pose_available=True,
                )
            )

        # Feed the exact same 20 tracks across 3 different frames in the 10s window to verify no double-counting
        agg.update("camera_01", monotonic_ts=100.0, utc_iso="2026-10-26T10:29:52Z", events=events_frame_1)
        agg.update("camera_01", monotonic_ts=104.0, utc_iso="2026-10-26T10:29:56Z", events=events_frame_1)
        summary = agg.update(
            "camera_01", monotonic_ts=108.0, utc_iso="2026-10-26T10:30:00Z", events=events_frame_1
        )

        self.assertEqual(summary.observed_people, 20)
        self.assertEqual(summary.sample_count, 60)
        self.assertEqual(summary.activity_distribution["STANDING"], 8)
        self.assertEqual(summary.activity_distribution["WALKING"], 5)
        self.assertEqual(summary.activity_distribution["RUNNING"], 7)
        self.assertAlmostEqual(summary.activity_percentages["STANDING"], 40.0)
        self.assertAlmostEqual(summary.activity_percentages["WALKING"], 25.0)
        self.assertAlmostEqual(summary.activity_percentages["RUNNING"], 35.0)
        self.assertAlmostEqual(summary.observation_quality, 0.92)


class TestSupervisedAndKalmanModules(unittest.TestCase):
    def test_supervised_logistic_classifier_predictions(self) -> None:
        from activity_system.supervised_classifier import SupervisedActivityClassifier

        clf = SupervisedActivityClassifier()
        lbl_stand, prob_stand, _ = clf.predict_proba(
            {
                "speed_bh_per_sec": 0.05,
                "ankle_std_bh": 0.005,
                "knee_std_bh": 0.005,
                "stride_amp_bh": 0.01,
                "cadence_hz": 0.0,
                "lower_body_visible_ratio": 1.0,
            }
        )
        self.assertEqual(lbl_stand, ActivityLabel.STANDING)
        self.assertGreater(prob_stand, 0.70)

        lbl_run, prob_run, _ = clf.predict_proba(
            {
                "speed_bh_per_sec": 2.40,
                "ankle_std_bh": 0.08,
                "knee_std_bh": 0.06,
                "stride_amp_bh": 0.22,
                "cadence_hz": 2.1,
                "lower_body_visible_ratio": 1.0,
            }
        )
        self.assertEqual(lbl_run, ActivityLabel.RUNNING)
        self.assertGreater(prob_run, 0.80)

    def test_prometheus_metrics_formatter(self) -> None:
        from activity_system.telemetry import ApplicationSnapshot, SystemMetricsSnapshot, format_prometheus_metrics

        snap = ApplicationSnapshot(
            app_name="test-app",
            healthy=True,
            timestamp_utc="2026-10-22T21:20:00Z",
            metrics=SystemMetricsSnapshot(
                timestamp_utc="2026-10-22T21:20:00Z",
                uptime_seconds=12.5,
                frames_received=100,
                frames_processed=100,
                throughput_fps=30.0,
            ),
        )
        prom_text = format_prometheus_metrics(snap)
        self.assertIn("cctv_frames_processed_total 100", prom_text)
        self.assertIn("cctv_throughput_fps 30.0", prom_text)


if __name__ == "__main__":
    unittest.main()
