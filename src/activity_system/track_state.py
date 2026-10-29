"""Bounded per-camera track state management and lifecycle tracking."""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
from pydantic import BaseModel

from activity_system.config import TrackerConfig
from activity_system.schemas import (
    ActivityLabel,
    Keypoint,
    ObservationQuality,
    PoseDetection,
    TrackLifecycle,
)


class TrackObservation:
    """Single timestamped pose and bounding-box observation for a tracked person."""

    __slots__ = (
        "monotonic_ts",
        "utc_iso",
        "bbox",
        "confidence",
        "keypoints",
        "observation_quality",
    )

    def __init__(
        self,
        monotonic_ts: float,
        utc_iso: str,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        keypoints: List[Keypoint],
        observation_quality: ObservationQuality,
    ) -> None:
        self.monotonic_ts = float(monotonic_ts)
        self.utc_iso = utc_iso
        self.bbox = bbox
        self.confidence = float(confidence)
        self.keypoints = keypoints
        self.observation_quality = observation_quality


class TrackPublicSnapshot(BaseModel):
    """Serializable read-only view of an active track for API and telemetry."""

    camera_id: str
    track_id: str
    session_id: int
    lifecycle: TrackLifecycle
    last_seen_utc_iso: str
    observations_in_window: int
    bbox: List[int]
    current_activity: ActivityLabel
    activity_code: str = "UNKNOWN"
    activity_duration_seconds: float = 0.0
    evidence_score: float
    observation_quality: ObservationQuality
    last_reliable_classification_ts: Optional[str] = None


class TrackRecord:
    """Mutable per-track state with bounded temporal history buffer."""

    def __init__(
        self,
        camera_id: str,
        session_id: int,
        numeric_id: int,
        monotonic_ts: float,
        utc_iso: str,
        detection: PoseDetection,
        config: TrackerConfig,
    ) -> None:
        self.camera_id = camera_id
        self.session_id = session_id
        self.numeric_id = numeric_id
        self.track_id = f"{camera_id}_s{session_id}_track_{numeric_id}"
        self._config = config

        self.hits: int = 1
        self.age_frames: int = 1
        self.missed_frames: int = 0
        self.created_monotonic_ts: float = float(monotonic_ts)
        self.last_seen_monotonic_ts: float = float(monotonic_ts)
        self.last_seen_utc_iso: str = utc_iso
        self.lifecycle: TrackLifecycle = (
            TrackLifecycle.CONFIRMED
            if config.min_hits_to_confirm <= 1
            else TrackLifecycle.TENTATIVE
        )

        self.observations: Deque[TrackObservation] = deque(maxlen=config.max_window_observations)
        self.latest_bbox: Tuple[int, int, int, int] = detection.bbox
        self.latest_quality: ObservationQuality = detection.observation_quality
        self.recent_features: Dict[str, float] = {}

        self.current_activity: ActivityLabel = ActivityLabel.UNKNOWN
        self.activity_started_monotonic_ts: float = float(monotonic_ts)
        self.candidate_activity: ActivityLabel = ActivityLabel.UNKNOWN
        self.candidate_streak: int = 0
        self.evidence_score: float = 0.0
        self.last_reliable_classification_ts: Optional[str] = None
        self.transition_count: int = 0

        self.add_observation(monotonic_ts, utc_iso, detection)

    @property
    def current_activity_duration_seconds(self) -> float:
        return max(0.0, float(self.last_seen_monotonic_ts - self.activity_started_monotonic_ts))

    def add_observation(
        self,
        monotonic_ts: float,
        utc_iso: str,
        detection: PoseDetection,
    ) -> None:
        obs = TrackObservation(
            monotonic_ts=monotonic_ts,
            utc_iso=utc_iso,
            bbox=detection.bbox,
            confidence=detection.confidence,
            keypoints=detection.keypoints,
            observation_quality=detection.observation_quality,
        )
        self.observations.append(obs)
        self.last_seen_monotonic_ts = float(monotonic_ts)
        self.last_seen_utc_iso = utc_iso
        self.latest_bbox = detection.bbox
        self.latest_quality = detection.observation_quality
        self._prune_old_observations(monotonic_ts)

    def mark_matched(
        self,
        monotonic_ts: float,
        utc_iso: str,
        detection: PoseDetection,
    ) -> None:
        self.hits += 1
        self.age_frames += 1
        self.missed_frames = 0
        if self.hits >= self._config.min_hits_to_confirm:
            self.lifecycle = TrackLifecycle.CONFIRMED
        self.add_observation(monotonic_ts, utc_iso, detection)

    def mark_missed(self, monotonic_ts: float, predicted_bbox: Optional[Tuple[int, int, int, int]] = None) -> None:
        self.age_frames += 1
        self.missed_frames += 1
        if predicted_bbox is not None:
            self.latest_bbox = predicted_bbox
        elapsed_unseen = float(monotonic_ts) - self.last_seen_monotonic_ts
        if elapsed_unseen > self._config.track_expiration_seconds:
            self.lifecycle = TrackLifecycle.EXPIRED
        elif self.lifecycle == TrackLifecycle.CONFIRMED:
            self.lifecycle = TrackLifecycle.OCCLUDED
        elif self.lifecycle == TrackLifecycle.TENTATIVE and self.missed_frames >= 2:
            self.lifecycle = TrackLifecycle.EXPIRED
        self._prune_old_observations(monotonic_ts)

    def _prune_old_observations(self, current_monotonic_ts: float) -> None:
        cutoff = float(current_monotonic_ts) - self._config.max_window_duration_seconds
        while self.observations and self.observations[0].monotonic_ts < cutoff:
            self.observations.popleft()

    def to_snapshot(self) -> TrackPublicSnapshot:
        return TrackPublicSnapshot(
            camera_id=self.camera_id,
            track_id=self.track_id,
            session_id=self.session_id,
            lifecycle=self.lifecycle,
            last_seen_utc_iso=self.last_seen_utc_iso,
            observations_in_window=len(self.observations),
            bbox=list(self.latest_bbox),
            current_activity=self.current_activity,
            activity_code=self.current_activity.upper_code.value,
            activity_duration_seconds=round(self.current_activity_duration_seconds, 3),
            evidence_score=round(float(self.evidence_score), 4),
            observation_quality=self.latest_quality,
            last_reliable_classification_ts=self.last_reliable_classification_ts,
        )


class TrackStateManager:
    """Maintains isolated per-camera track registries and handles stream session resets."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._tracks_by_camera: Dict[str, Dict[str, TrackRecord]] = {}
        self._session_by_camera: Dict[str, int] = {}

    def ensure_camera_session(self, camera_id: str, session_id: int) -> bool:
        """Ensure camera state matches session_id; returns True if session was reset."""
        prev_session = self._session_by_camera.get(camera_id)
        if prev_session is None:
            self._session_by_camera[camera_id] = session_id
            self._tracks_by_camera[camera_id] = {}
            return False
        if prev_session != session_id:
            self._session_by_camera[camera_id] = session_id
            self._tracks_by_camera[camera_id] = {}
            return True
        return False

    def reset_camera(self, camera_id: str, new_session_id: Optional[int] = None) -> None:
        """Clear all track histories for a specific camera without affecting other cameras."""
        self._tracks_by_camera[camera_id] = {}
        if new_session_id is not None:
            self._session_by_camera[camera_id] = new_session_id

    def register_track(self, record: TrackRecord) -> None:
        cam_map = self._tracks_by_camera.setdefault(record.camera_id, {})
        cam_map[record.track_id] = record

    def remove_expired(self, camera_id: str) -> List[str]:
        cam_map = self._tracks_by_camera.get(camera_id, {})
        expired_ids = [
            tid for tid, rec in cam_map.items() if rec.lifecycle == TrackLifecycle.EXPIRED
        ]
        for tid in expired_ids:
            del cam_map[tid]
        return expired_ids

    def get_camera_tracks(
        self,
        camera_id: str,
        include_tentative: bool = True,
    ) -> List[TrackRecord]:
        cam_map = self._tracks_by_camera.get(camera_id, {})
        tracks = [
            rec
            for rec in cam_map.values()
            if rec.lifecycle != TrackLifecycle.EXPIRED
            and (include_tentative or rec.lifecycle != TrackLifecycle.TENTATIVE)
        ]
        return sorted(tracks, key=lambda r: r.numeric_id)

    def get_all_active_snapshots(self, include_tentative: bool = False) -> List[TrackPublicSnapshot]:
        snapshots: List[TrackPublicSnapshot] = []
        for cam_id in sorted(self._tracks_by_camera.keys()):
            for rec in self.get_camera_tracks(cam_id, include_tentative=include_tentative):
                snapshots.append(rec.to_snapshot())
        return snapshots
