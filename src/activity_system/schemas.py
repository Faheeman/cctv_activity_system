"""Typed data schemas and versioned contracts for the CCTV activity intelligence and risk alerting pipeline."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import numpy as np
from pydantic import BaseModel, Field, root_validator, validator


COCO_KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def utc_now_iso() -> str:
    """Return current UTC timestamp in strict ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_utc_iso(ts: Union[str, datetime, None]) -> str:
    """Normalize a timestamp string or datetime to timezone-aware UTC ISO 8601 with trailing 'Z'."""
    if ts is None:
        return utc_now_iso()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    text = str(ts).strip()
    if not text:
        return utc_now_iso()
    if text.endswith("Z"):
        return text
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text + "Z"


class SourceType(str, Enum):
    RTSP = "rtsp"
    FILE = "file"
    WEBCAM = "webcam"


class StreamState(str, Enum):
    IDLE = "idle"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    EOF = "eof"
    STOPPED = "stopped"
    FAILED = "failed"


class ActivityLabel(str, Enum):
    STANDING = "standing"
    WALKING = "walking"
    RUNNING = "running"
    SITTING = "sitting"
    LYING = "lying"
    FALLING = "falling"
    UNKNOWN = "unknown"

    @classmethod
    def from_any(cls, value: Union["ActivityLabel", "ActivityCode", str]) -> "ActivityLabel":
        if isinstance(value, cls):
            return value
        cleaned = str(value).strip().lower()
        return cls(cleaned)

    @property
    def upper_code(self) -> "ActivityCode":
        return ActivityCode(self.value.upper())


class ActivityCode(str, Enum):
    """Canonical uppercase activity labels for v1 REST API and IURIS event contracts."""

    STANDING = "STANDING"
    WALKING = "WALKING"
    RUNNING = "RUNNING"
    SITTING = "SITTING"
    LYING = "LYING"
    FALLING = "FALLING"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_any(cls, value: Union["ActivityCode", ActivityLabel, str]) -> "ActivityCode":
        if isinstance(value, cls):
            return value
        if isinstance(value, ActivityLabel):
            return cls(value.value.upper())
        cleaned = str(value).strip().upper()
        return cls(cleaned)


class ObservationQuality(str, Enum):
    GOOD = "good"
    PARTIAL = "partial"
    POOR = "poor"

    @property
    def numeric_score(self) -> float:
        if self == ObservationQuality.GOOD:
            return 0.92
        if self == ObservationQuality.PARTIAL:
            return 0.65
        return 0.30


class TrackLifecycle(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    OCCLUDED = "occluded"
    EXPIRED = "expired"


class RiskSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskStatus(str, Enum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    EXPIRED = "EXPIRED"


class AlertLifecycleState(str, Enum):
    TRIGGERED = "TRIGGERED"
    UPDATED = "UPDATED"
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class RiskStatusHistoryEntry(BaseModel):
    """Auditable status transition record for a risk alert."""

    timestamp: str
    from_status: Optional[RiskStatus] = None
    to_status: RiskStatus
    lifecycle_state: str
    actor: str = "system"
    note: Optional[str] = None


class CameraStatusSnapshot(BaseModel):
    camera_id: str
    camera_name: Optional[str] = None
    source_type: SourceType
    redacted_uri: str
    state: StreamState
    processing_state: str = "idle"
    session_id: int = 1
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    frames_received: int = 0
    frames_emitted: int = 0
    frames_dropped: int = 0
    frames_skipped_sampling: int = 0
    stale_frames_dropped: int = 0
    is_stale: bool = False
    reconnect_attempts: int = 0
    total_reconnections: int = 0
    last_frame_monotonic_ts: Optional[float] = None
    last_frame_utc_iso: Optional[str] = None
    last_error: Optional[str] = None


class FramePacket:
    """In-memory video frame container with capture metadata."""

    __slots__ = (
        "camera_id",
        "session_id",
        "frame_index",
        "frame",
        "monotonic_ts",
        "utc_iso",
        "width",
        "height",
    )

    def __init__(
        self,
        camera_id: str,
        session_id: int,
        frame_index: int,
        frame: np.ndarray,
        monotonic_ts: float,
        utc_iso: Optional[str] = None,
    ) -> None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("FramePacket requires a non-empty numpy ndarray frame.")
        self.camera_id = camera_id
        self.session_id = session_id
        self.frame_index = frame_index
        self.frame = frame
        self.monotonic_ts = float(monotonic_ts)
        self.utc_iso = normalize_utc_iso(utc_iso) if utc_iso else utc_now_iso()
        self.height = int(frame.shape[0])
        self.width = int(frame.shape[1])


class Keypoint(BaseModel):
    index: int = Field(..., ge=0, le=16)
    name: str
    x: float
    y: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    visible: bool


class PoseDetection(BaseModel):
    bbox: Tuple[int, int, int, int]
    confidence: float = Field(..., ge=0.0, le=1.0)
    keypoints: List[Keypoint]
    observation_quality: ObservationQuality
    track_id: Optional[str] = None

    @validator("bbox")
    def validate_bbox_order(cls, v: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box coordinates: {v}; expected x2 > x1 and y2 > y1.")
        return v


class ActivityObservationV1(BaseModel):
    """Versioned v1.0 activity observation event schema for REST v1 and IURIS consumers."""

    schema_version: str = Field(default="1.0", description="Event contract version")
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(default="activity.detected")
    camera_id: str
    track_id: str
    timestamp: str
    activity: ActivityCode
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    observation_quality: float = Field(..., ge=0.0, le=1.0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    bbox: Optional[List[int]] = None
    pose_available: bool = True
    model_version: str = "heuristic-kinematic-v1.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("activity", pre=True)
    def validate_activity_code(cls, v: Any) -> ActivityCode:
        return ActivityCode.from_any(v)

    @validator("timestamp", pre=True)
    def validate_timestamp(cls, v: Any) -> str:
        return normalize_utc_iso(v)

    @validator("bbox")
    def validate_bbox_list(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        if v is not None:
            if len(v) != 4:
                raise ValueError("bbox must contain exactly 4 integers [x1, y1, x2, y2].")
            if v[2] <= v[0] or v[3] <= v[1]:
                raise ValueError(f"Invalid bbox coordinates {v}; expected x2 > x1 and y2 > y1.")
        return v


class ActivityEvent(BaseModel):
    """Internal & legacy activity event model with automatic conversion to ActivityObservationV1."""

    schema_version: str = "1.0.0"
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "activity.detected"
    camera_id: str
    track_id: str
    timestamp: str
    activity: ActivityLabel
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    observation_quality: ObservationQuality
    observation_quality_score: float = Field(default=0.88, ge=0.0, le=1.0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    bbox: List[int]
    pose_available: bool
    model_version: str = "heuristic-kinematic-v1.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("activity", pre=True)
    def normalize_activity_label(cls, v: Any) -> ActivityLabel:
        return ActivityLabel.from_any(v)

    def to_v1(self) -> ActivityObservationV1:
        """Convert internal ActivityEvent into canonical v1.0 ActivityObservationV1."""
        meta = dict(self.metadata)
        meta.setdefault("evidence_score", self.evidence_score)
        meta.setdefault("quality_grade", self.observation_quality.value)
        return ActivityObservationV1(
            schema_version="1.0",
            event_id=self.event_id,
            event_type="activity.detected",
            camera_id=self.camera_id,
            track_id=self.track_id,
            timestamp=normalize_utc_iso(self.timestamp),
            activity=self.activity.upper_code,
            confidence=self.confidence,
            observation_quality=round(float(self.observation_quality_score), 4),
            duration_seconds=round(float(self.duration_seconds), 3),
            bbox=list(self.bbox) if self.bbox else None,
            pose_available=self.pose_available,
            model_version=self.model_version,
            metadata=meta,
        )


class CameraActivitySummaryV1(BaseModel):
    """Versioned v1.0 camera-level activity distribution and percentage summary over a sliding window."""

    schema_version: str = "1.0"
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "camera.activity_summary"
    camera_id: str
    timestamp: str
    window_seconds: float = Field(..., gt=0.0)
    observed_people: int = Field(..., ge=0)
    activity_distribution: Dict[str, int]
    activity_percentages: Dict[str, float]
    observation_quality: float = Field(..., ge=0.0, le=1.0)
    sample_count: int = Field(default=0, ge=0)
    eligible_people: int = Field(default=0, ge=0)
    unknown_count: int = Field(default=0, ge=0)
    unknown_treatment: str = Field(
        default="exclude_from_percentage_denominator",
        description="How UNKNOWN tracks are treated when calculating activity_percentages.",
    )
    aggregation_method: str = Field(
        default="unique_track_latest_committed_activity",
        description="Deduplication method preventing per-frame double counting of tracked individuals.",
    )

    @validator("timestamp", pre=True)
    def validate_timestamp(cls, v: Any) -> str:
        return normalize_utc_iso(v)


_DEFAULT_RISK_TITLES: Dict[str, str] = {
    "UNUSUAL_ACTIVITY_CONCENTRATION": "Unusual Crowd Activity Concentration Detected",
    "POSSIBLE_FALL_DETECTED": "Possible Person Fall Detected",
    "PROLONGED_IMMOBILITY": "Prolonged Person Immobility Detected",
    "ZONE_RESTRICTED_ACTIVITY": "Monitored Zone Activity Alert",
    "PERSISTENT_UNUSUAL_ACTIVITY": "Persistent Unusual Activity Detected",
}


class RiskEventV1(BaseModel):
    """Versioned v1.0 risk assessment alert event with correlation key and full lifecycle metadata."""

    schema_version: str = "1.0"
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "risk.detected"
    correlation_key: str = ""
    camera_id: str
    camera_name: Optional[str] = None
    timestamp: str
    created_at: str = ""
    updated_at: str = ""
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_note: Optional[str] = None
    risk_type: str
    rule_name: str = ""
    title: str = ""
    severity: RiskSeverity
    status: RiskStatus = RiskStatus.OPEN
    lifecycle_state: str = AlertLifecycleState.TRIGGERED.value
    occurrence_count: int = Field(default=1, ge=1)
    activity: Optional[ActivityCode] = None
    measured_percentage: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    threshold_percentage: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    observed_people: int = Field(default=0, ge=0)
    matching_people: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    window_seconds: float = Field(default=10.0, gt=0.0)
    observation_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    track_id: Optional[str] = None
    relevant_track_ids: List[str] = Field(default_factory=list)
    zone_id: Optional[str] = None
    related_camera_url: Optional[str] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)
    explanation: str
    model_version: str = "heuristic-kinematic-v1.0"
    status_history: List[RiskStatusHistoryEntry] = Field(default_factory=list)

    @validator("activity", pre=True)
    def validate_optional_activity(cls, v: Any) -> Optional[ActivityCode]:
        if v is None or v == "":
            return None
        return ActivityCode.from_any(v)

    @validator("timestamp", pre=True)
    def validate_timestamp(cls, v: Any) -> str:
        return normalize_utc_iso(v)

    @root_validator(pre=False)
    def populate_lifecycle_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        ts = values.get("timestamp") or utc_now_iso()
        if not values.get("created_at"):
            values["created_at"] = ts
        if not values.get("updated_at"):
            values["updated_at"] = ts

        cam_id = values.get("camera_id", "unknown")
        rule = values.get("rule_name") or values.get("risk_type", "rule")
        act = values.get("activity")
        act_str = act.value if hasattr(act, "value") else (str(act) if act else "ALL")
        trk = values.get("track_id")
        zone = values.get("zone_id")

        if not values.get("correlation_key"):
            parts = [str(cam_id), str(rule), act_str]
            if zone:
                parts.append(str(zone))
            if trk:
                parts.append(str(trk))
            values["correlation_key"] = ":".join(parts)

        if not values.get("title"):
            rtype = str(values.get("risk_type", ""))
            if rtype == "UNUSUAL_ACTIVITY_CONCENTRATION" and act_str != "ALL":
                values["title"] = f"Unusual {act_str.title()} Concentration Detected"
            else:
                values["title"] = _DEFAULT_RISK_TITLES.get(
                    rtype, f"Operational Risk Alert ({rtype})"
                )

        if not values.get("related_camera_url"):
            values["related_camera_url"] = f"/stream.mjpg?camera_id={cam_id}"

        return values
