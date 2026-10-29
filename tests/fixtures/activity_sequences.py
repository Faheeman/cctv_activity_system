"""Labeled synthetic pose and tracking sequences for activity recognition evaluation."""

from typing import List, Tuple
from activity_system.pose_estimator import build_human_coco_keypoints
from activity_system.schemas import (
    ActivityLabel,
    COCO_KEYPOINT_NAMES,
    FramePacket,
    Keypoint,
    ObservationQuality,
    PoseDetection,
)
import numpy as np


class LabeledSequence:
    """Represents a single-session video sequence with ground-truth activity label."""

    def __init__(
        self,
        sequence_id: str,
        expected_activity: ActivityLabel,
        frames: List[Tuple[FramePacket, List[PoseDetection]]],
        description: str,
    ) -> None:
        self.sequence_id = sequence_id
        self.expected_activity = expected_activity
        self.frames = frames
        self.description = description


def _build_pose_detection(
    cx: float,
    cy: float,
    body_h: float,
    phase_rad: float,
    stride_amp_bh: float,
    knee_lift_bh: float,
    occlude_lower_body: bool = False,
) -> PoseDetection:
    kpts_arr = build_human_coco_keypoints(
        center_x=cx,
        center_y=cy,
        body_height=body_h,
        phase_rad=phase_rad,
        stride_amp_bh=stride_amp_bh,
        knee_lift_bh=knee_lift_bh,
        confidence=0.92,
        occlude_lower_body=occlude_lower_body,
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
    half_w = body_h * 0.24
    half_h = body_h * 0.52
    bbox = (
        max(0, int(round(cx - half_w))),
        max(0, int(round(cy - half_h))),
        min(640, int(round(cx + half_w))),
        min(480, int(round(cy + half_h))),
    )
    quality = ObservationQuality.POOR if occlude_lower_body else ObservationQuality.GOOD
    return PoseDetection(
        bbox=bbox,
        confidence=0.92,
        keypoints=kpts,
        observation_quality=quality,
    )


def build_evaluation_dataset() -> List[LabeledSequence]:
    """Generate 6 distinct session-split sequences covering standing, walking (near/far), running, and occlusion."""
    dummy_img = np.full((480, 640, 3), 40, dtype=np.uint8)
    fps = 15.0
    dt = 1.0 / fps
    num_frames = 35  # ~2.33 seconds per sequence

    sequences: List[LabeledSequence] = []

    # 1. Standing still (near camera, body_h=160px)
    standing_frames = []
    for i in range(num_frames):
        ts = 10.0 + i * dt
        pkt = FramePacket("cam_eval", 1, i + 1, dummy_img, monotonic_ts=ts, utc_iso=f"2026-10-22T20:00:{i:02d}Z")
        det = _build_pose_detection(
            cx=300.0 + np.sin(i * 0.2) * 1.2,
            cy=250.0,
            body_h=160.0,
            phase_rad=0.1,
            stride_amp_bh=0.01,
            knee_lift_bh=0.005,
        )
        standing_frames.append((pkt, [det]))
    sequences.append(
        LabeledSequence("seq_standing_01", ActivityLabel.STANDING, standing_frames, "Standing person with minor sway")
    )

    # 2. Walking near camera (body_h=160px, speed ~0.95 BH/s = 152 px/s = 10.1 px/frame)
    walk_near_frames = []
    for i in range(num_frames):
        ts = 20.0 + i * dt
        pkt = FramePacket("cam_eval", 2, i + 1, dummy_img, monotonic_ts=ts, utc_iso=f"2026-10-22T20:01:{i:02d}Z")
        det = _build_pose_detection(
            cx=80.0 + i * 10.0,
            cy=260.0,
            body_h=160.0,
            phase_rad=i * 0.55,
            stride_amp_bh=0.11,
            knee_lift_bh=0.045,
        )
        walk_near_frames.append((pkt, [det]))
    sequences.append(
        LabeledSequence("seq_walking_near_01", ActivityLabel.WALKING, walk_near_frames, "Walking subject near camera")
    )

    # 3. Walking far from camera (body_h=80px, speed ~0.95 BH/s = 76 px/s = 5.05 px/frame)
    walk_far_frames = []
    for i in range(num_frames):
        ts = 30.0 + i * dt
        pkt = FramePacket("cam_eval", 3, i + 1, dummy_img, monotonic_ts=ts, utc_iso=f"2026-10-22T20:02:{i:02d}Z")
        det = _build_pose_detection(
            cx=120.0 + i * 5.05,
            cy=160.0,
            body_h=80.0,
            phase_rad=i * 0.55,
            stride_amp_bh=0.11,
            knee_lift_bh=0.045,
        )
        walk_far_frames.append((pkt, [det]))
    sequences.append(
        LabeledSequence("seq_walking_far_01", ActivityLabel.WALKING, walk_far_frames, "Walking subject far from camera")
    )

    # 4. Running across corridor (body_h=120px, speed ~2.4 BH/s = 288 px/s = 19.2 px/frame -> use 26 frames)
    run_frames = []
    for i in range(26):
        ts = 40.0 + i * dt
        pkt = FramePacket("cam_eval", 4, i + 1, dummy_img, monotonic_ts=ts, utc_iso=f"2026-10-22T20:03:{i:02d}Z")
        det = _build_pose_detection(
            cx=60.0 + i * 19.2,
            cy=240.0,
            body_h=120.0,
            phase_rad=i * 0.95,
            stride_amp_bh=0.22,
            knee_lift_bh=0.11,
        )
        run_frames.append((pkt, [det]))
    sequences.append(
        LabeledSequence("seq_running_01", ActivityLabel.RUNNING, run_frames, "Running subject across corridor")
    )

    # 5. Running diagonally in CCTV viewpoint (foreshortened vertical motion)
    run_diag_frames = []
    for i in range(26):
        ts = 50.0 + i * dt
        pkt = FramePacket("cam_eval", 5, i + 1, dummy_img, monotonic_ts=ts, utc_iso=f"2026-10-22T20:04:{i:02d}Z")
        det = _build_pose_detection(
            cx=100.0 + i * 12.5,
            cy=140.0 + i * 9.5,
            body_h=115.0,
            phase_rad=i * 0.95,
            stride_amp_bh=0.21,
            knee_lift_bh=0.10,
        )
        run_diag_frames.append((pkt, [det]))
    sequences.append(
        LabeledSequence("seq_running_diag_01", ActivityLabel.RUNNING, run_diag_frames, "Diagonal CCTV running subject")
    )

    # 6. Lower-body occluded sequence (legs behind obstacle -> must return UNKNOWN)
    occluded_frames = []
    for i in range(num_frames):
        ts = 60.0 + i * dt
        pkt = FramePacket("cam_eval", 6, i + 1, dummy_img, monotonic_ts=ts, utc_iso=f"2026-10-22T20:05:{i:02d}Z")
        det = _build_pose_detection(
            cx=100.0 + i * 9.0,
            cy=250.0,
            body_h=150.0,
            phase_rad=i * 0.5,
            stride_amp_bh=0.10,
            knee_lift_bh=0.04,
            occlude_lower_body=True,
        )
        occluded_frames.append((pkt, [det]))
    sequences.append(
        LabeledSequence("seq_occluded_01", ActivityLabel.UNKNOWN, occluded_frames, "Lower-body occluded subject")
    )

    return sequences
