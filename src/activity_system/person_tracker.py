"""Per-camera ByteTrack multi-person tracker using KalmanBoxState and SciPy Hungarian matching."""

from typing import Dict, List, Optional, Sequence, Tuple
from scipy.optimize import linear_sum_assignment

from activity_system.config import TrackerConfig
from activity_system.kalman_filter import KalmanBoxState, compute_iou_matrix
from activity_system.schemas import FramePacket, PoseDetection
from activity_system.track_state import TrackRecord, TrackStateManager


class _CameraByteTracker:
    """Single-camera ByteTrack state machine."""

    def __init__(self, camera_id: str, session_id: int, config: TrackerConfig) -> None:
        self.camera_id = camera_id
        self.session_id = session_id
        self.config = config
        self._next_numeric_id: int = 1
        self._kalman_by_track_id: Dict[str, KalmanBoxState] = {}

    def step(
        self,
        packet: FramePacket,
        detections: Sequence[PoseDetection],
        state_manager: TrackStateManager,
    ) -> Tuple[List[PoseDetection], List[TrackRecord]]:
        now_ts = packet.monotonic_ts
        utc_iso = packet.utc_iso

        active_tracks = state_manager.get_camera_tracks(self.camera_id, include_tentative=True)

        predicted_boxes: List[Tuple[int, int, int, int]] = []
        for trk in active_tracks:
            kf = self._kalman_by_track_id.get(trk.track_id)
            if kf is not None:
                pred_box = kf.predict(now_ts)
                predicted_boxes.append(pred_box)
            else:
                predicted_boxes.append(trk.latest_bbox)

        high_indices = [
            i for i, det in enumerate(detections) if det.confidence >= self.config.high_thresh
        ]
        low_indices = [
            i
            for i, det in enumerate(detections)
            if self.config.low_thresh <= det.confidence < self.config.high_thresh
        ]

        high_boxes = [detections[i].bbox for i in high_indices]

        matched_t_s1, unmatched_t_s1, unmatched_d_high = self._associate(
            predicted_boxes,
            high_boxes,
            iou_thresh=self.config.match_iou_thresh,
        )

        annotated_detections: List[PoseDetection] = []

        for t_idx, d_rel_idx in matched_t_s1:
            trk = active_tracks[t_idx]
            det_idx = high_indices[d_rel_idx]
            det = detections[det_idx]
            kf = self._kalman_by_track_id[trk.track_id]
            kf.update(det.bbox)
            trk.mark_matched(now_ts, utc_iso, det)
            det.track_id = trk.track_id
            annotated_detections.append(det)

        rem_tracks = [active_tracks[idx] for idx in unmatched_t_s1]
        rem_pred_boxes = [predicted_boxes[idx] for idx in unmatched_t_s1]
        low_boxes = [detections[i].bbox for i in low_indices]

        matched_t_s2, unmatched_t_s2, _ = self._associate(
            rem_pred_boxes,
            low_boxes,
            iou_thresh=self.config.second_match_iou_thresh,
        )

        for t_rel_idx, d_rel_idx in matched_t_s2:
            trk = rem_tracks[t_rel_idx]
            det_idx = low_indices[d_rel_idx]
            det = detections[det_idx]
            kf = self._kalman_by_track_id[trk.track_id]
            kf.update(det.bbox)
            trk.mark_matched(now_ts, utc_iso, det)
            det.track_id = trk.track_id
            annotated_detections.append(det)

        for t_rel_idx in unmatched_t_s2:
            trk = rem_tracks[t_rel_idx]
            pred_box = rem_pred_boxes[t_rel_idx]
            trk.mark_missed(now_ts, predicted_bbox=pred_box)

        for d_rel_idx in unmatched_d_high:
            det_idx = high_indices[d_rel_idx]
            det = detections[det_idx]
            num_id = self._next_numeric_id
            self._next_numeric_id += 1
            new_record = TrackRecord(
                camera_id=self.camera_id,
                session_id=self.session_id,
                numeric_id=num_id,
                monotonic_ts=now_ts,
                utc_iso=utc_iso,
                detection=det,
                config=self.config,
            )
            self._kalman_by_track_id[new_record.track_id] = KalmanBoxState(det.bbox, now_ts)
            state_manager.register_track(new_record)
            det.track_id = new_record.track_id
            annotated_detections.append(det)

        expired_ids = state_manager.remove_expired(self.camera_id)
        for exp_id in expired_ids:
            self._kalman_by_track_id.pop(exp_id, None)

        current_tracks = state_manager.get_camera_tracks(self.camera_id, include_tentative=True)
        return annotated_detections, current_tracks

    @staticmethod
    def _associate(
        track_boxes: Sequence[Tuple[int, int, int, int]],
        det_boxes: Sequence[Tuple[int, int, int, int]],
        iou_thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(track_boxes) == 0:
            return [], [], list(range(len(det_boxes)))
        if len(det_boxes) == 0:
            return [], list(range(len(track_boxes))), []

        iou_mat = compute_iou_matrix(track_boxes, det_boxes)
        cost_mat = 1.0 - iou_mat
        row_ind, col_ind = linear_sum_assignment(cost_mat)

        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(track_boxes)))
        unmatched_dets = set(range(len(det_boxes)))

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= iou_thresh:
                matches.append((int(r), int(c)))
                unmatched_tracks.discard(int(r))
                unmatched_dets.discard(int(c))

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


class PersonTracker:
    """Multi-camera ByteTrack coordinator maintaining independent tracker instances per camera."""

    def __init__(
        self,
        config: TrackerConfig,
        state_manager: Optional[TrackStateManager] = None,
    ) -> None:
        self.config = config
        self.state_manager = state_manager or TrackStateManager(config)
        self._trackers_by_camera: Dict[str, _CameraByteTracker] = {}

    def update(
        self,
        packet: FramePacket,
        detections: Sequence[PoseDetection],
    ) -> Tuple[List[PoseDetection], List[TrackRecord]]:
        """Update per-camera tracker state and return detections annotated with persistent track_id."""
        cam_id = packet.camera_id
        sess_id = packet.session_id

        was_reset = self.state_manager.ensure_camera_session(cam_id, sess_id)
        if was_reset or cam_id not in self._trackers_by_camera:
            self._trackers_by_camera[cam_id] = _CameraByteTracker(cam_id, sess_id, self.config)

        tracker = self._trackers_by_camera[cam_id]
        return tracker.step(packet, detections, self.state_manager)

    def reset_camera(self, camera_id: str, new_session_id: int = 1) -> None:
        """Explicitly reset tracker state for a single camera."""
        self.state_manager.reset_camera(camera_id, new_session_id=new_session_id)
        self._trackers_by_camera[camera_id] = _CameraByteTracker(
            camera_id, new_session_id, self.config
        )
