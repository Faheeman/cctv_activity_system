"""Independent video frame renderer for bounding boxes, COCO skeletons, track IDs, and activity labels."""

from typing import Dict, Optional, Sequence, Tuple
import cv2
import numpy as np

from activity_system.config import RendererConfig
from activity_system.pose_estimator import COCO_SKELETON_EDGES
from activity_system.schemas import ActivityEvent, ActivityLabel, PoseDetection


_ACTIVITY_COLORS: Dict[ActivityLabel, Tuple[int, int, int]] = {
    ActivityLabel.STANDING: (220, 180, 40),
    ActivityLabel.WALKING: (60, 200, 60),
    ActivityLabel.RUNNING: (40, 80, 240),
    ActivityLabel.SITTING: (200, 140, 60),
    ActivityLabel.LYING: (40, 165, 255),
    ActivityLabel.FALLING: (30, 30, 255),
    ActivityLabel.UNKNOWN: (160, 160, 160),
}


class Renderer:
    """Draws pose detections, track identities, activity classifications, and stream telemetry."""

    def __init__(self, config: RendererConfig) -> None:
        self.config = config

    def render(
        self,
        frame: np.ndarray,
        detections: Sequence[PoseDetection] = (),
        events_by_track: Optional[Dict[str, ActivityEvent]] = None,
        fps: Optional[float] = None,
        camera_id: Optional[str] = None,
        stream_state: Optional[str] = None,
        copy: bool = True,
    ) -> np.ndarray:
        """Render annotations onto the video frame without coupling to classifier internals."""
        canvas = frame.copy() if copy else frame
        events_map = events_by_track or {}

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            event = events_map.get(det.track_id) if det.track_id else None
            activity = event.activity if event is not None else ActivityLabel.UNKNOWN
            box_color = _ACTIVITY_COLORS.get(activity, (160, 160, 160))

            if self.config.draw_bbox:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

            if self.config.draw_skeleton and det.keypoints:
                self._draw_skeleton(canvas, det)

            label_parts = []
            if self.config.draw_track_id and det.track_id:
                short_id = det.track_id.split("_")[-1]
                label_parts.append(f"ID:{short_id}")
            if self.config.draw_activity:
                if event is not None:
                    label_parts.append(
                        f"{event.activity.value.upper()} ({event.observation_quality.value}:{event.evidence_score:.2f})"
                    )
                else:
                    label_parts.append(f"CONF:{det.confidence:.2f}")

            if label_parts:
                banner_text = " | ".join(label_parts)
                text_y = max(18, y1 - 6)
                cv2.putText(
                    canvas,
                    banner_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )

        if self.config.draw_telemetry and (camera_id or fps is not None or stream_state):
            telemetry_items = []
            if camera_id:
                telemetry_items.append(f"CAM:{camera_id}")
            if stream_state:
                telemetry_items.append(f"STATE:{stream_state}")
            if fps is not None:
                telemetry_items.append(f"FPS:{fps:.1f}")
            telemetry_items.append(f"TRACKS:{len(detections)}")
            header = "  ".join(telemetry_items)
            cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 520), 26), (20, 20, 20), -1)
            cv2.putText(
                canvas,
                header,
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        return canvas

    @staticmethod
    def _draw_skeleton(canvas: np.ndarray, detection: PoseDetection) -> None:
        kpts = detection.keypoints
        if len(kpts) < 17:
            return
        for idx_a, idx_b in COCO_SKELETON_EDGES:
            kp_a = kpts[idx_a]
            kp_b = kpts[idx_b]
            if kp_a.visible and kp_b.visible:
                pt_a = (int(round(kp_a.x)), int(round(kp_a.y)))
                pt_b = (int(round(kp_b.x)), int(round(kp_b.y)))
                cv2.line(canvas, pt_a, pt_b, (255, 220, 80), 2, cv2.LINE_AA)

        for kp in kpts:
            if kp.visible:
                pt = (int(round(kp.x)), int(round(kp.y)))
                cv2.circle(canvas, pt, 3, (0, 255, 255), -1, cv2.LINE_AA)
