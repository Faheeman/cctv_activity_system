"""Clear, decoupled stage interfaces (Protocols) for the CCTV Activity Intelligence Pipeline."""

from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from activity_system.schemas import (
    ActivityEvent,
    ActivityObservationV1,
    CameraActivitySummaryV1,
    CameraStatusSnapshot,
    FramePacket,
    PoseDetection,
    RiskEventV1,
)


@runtime_checkable
class InputStreamAdapter(Protocol):
    """Pluggable input-adapter interface for Webcams, RTSP streams, Video files, and future sources."""

    @property
    def camera_id(self) -> str:
        ...

    def connect(self) -> bool:
        ...

    def read_next(self) -> Optional[FramePacket]:
        ...

    def status(self) -> CameraStatusSnapshot:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class PersonDetector(Protocol):
    """Interface for person detection stage."""

    def detect_persons(self, packet: FramePacket) -> List[PoseDetection]:
        ...


@runtime_checkable
class HumanPoseEstimator(Protocol):
    """Interface for human pose estimation stage."""

    def estimate(self, packet: FramePacket) -> List[PoseDetection]:
        ...


@runtime_checkable
class MultiObjectTracker(Protocol):
    """Interface for camera-scoped multi-object tracking stage."""

    def update(
        self,
        packet: FramePacket,
        detections: Sequence[PoseDetection],
    ) -> Tuple[List[PoseDetection], List[Any]]:
        ...

    def reset_camera(self, camera_id: str, new_session_id: int = 1) -> None:
        ...


@runtime_checkable
class ActivityFeatureExtractor(Protocol):
    """Interface for extracting scale-normalized temporal pose and trajectory features."""

    def extract_features(self, observations: Sequence[Any]) -> Dict[str, float]:
        ...


@runtime_checkable
class TemporalActivityClassifier(Protocol):
    """Interface for multi-frame temporal activity recognition."""

    def classify_track(self, track: Any) -> ActivityEvent:
        ...


@runtime_checkable
class CameraSummaryEvaluatorProtocol(Protocol):
    """Interface for generating deduplicated camera-level activity distributions and percentages."""

    def update(
        self,
        camera_id: str,
        monotonic_ts: float,
        utc_iso: str,
        events: Sequence[ActivityEvent],
    ) -> CameraActivitySummaryV1:
        ...


@runtime_checkable
class RiskRuleProtocol(Protocol):
    """Interface for pluggable, explainable risk assessment rules."""

    @property
    def rule_name(self) -> str:
        ...

    def evaluate(
        self,
        camera_id: str,
        monotonic_ts: float,
        utc_iso: str,
        activities: Sequence[ActivityObservationV1],
        summary: Optional[CameraActivitySummaryV1],
        model_version: str,
    ) -> List[RiskEventV1]:
        ...


@runtime_checkable
class RiskEngineProtocol(Protocol):
    """Interface for the independent risk assessment engine."""

    def evaluate(
        self,
        camera_id: str,
        monotonic_ts: float,
        utc_iso: str,
        activities: Sequence[ActivityObservationV1],
        summary: Optional[CameraActivitySummaryV1],
        model_version: str = "heuristic-kinematic-v1.0",
    ) -> List[RiskEventV1]:
        ...
