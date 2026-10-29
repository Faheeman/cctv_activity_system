"""Thread-safe state snapshots, JPEG frame buffer, and Prometheus metrics exposition."""

from typing import Callable, Dict, List, Optional
import resource
import threading
from pydantic import BaseModel, Field

from activity_system.event_store import BoundedEventStore
from activity_system.iuris.schemas import IurisDeliveryMetrics
from activity_system.schemas import (
    ActivityEvent,
    ActivityObservationV1,
    CameraActivitySummaryV1,
    CameraStatusSnapshot,
    RiskEventV1,
)
from activity_system.track_state import TrackPublicSnapshot


class SystemMetricsSnapshot(BaseModel):
    """Operational telemetry and resource metrics snapshot."""

    timestamp_utc: str
    uptime_seconds: float = 0.0
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    frames_skipped_sampling: int = 0
    mean_inference_ms: float = 0.0
    mean_classification_ms: float = 0.0
    mean_end_to_end_ms: float = 0.0
    throughput_fps: float = 0.0
    active_cameras: int = 0
    connected_cameras: int = 0
    disconnected_cameras: int = 0
    active_tracks: int = 0
    stream_reconnect_count: int = 0
    classification_transitions: int = 0
    low_quality_observations: int = 0
    risk_events_generated: int = 0
    active_risk_events: int = 0
    duplicate_alerts_suppressed: int = 0
    event_queue_depth: int = 0
    iuris_deliveries_attempted: int = 0
    events_delivered: int = 0
    event_delivery_failures: int = 0
    iuris_retries_total: int = 0
    mean_delivery_latency_ms: float = 0.0
    model_ready: bool = True
    pose_backend: str = "synthetic"
    cpu_utilization_pct: float = 0.0
    gpu_utilization_pct: Optional[float] = None
    memory_rss_mb: float = 0.0


class ApplicationSnapshot(BaseModel):
    """Immutable copy-on-write state snapshot served to API readers without blocking inference."""

    schema_version: str = "1.0.0"
    app_name: str
    healthy: bool = True
    ready: bool = True
    model_version: str = "heuristic-kinematic-v1.0"
    timestamp_utc: str
    cameras: List[CameraStatusSnapshot] = Field(default_factory=list)
    tracks: List[TrackPublicSnapshot] = Field(default_factory=list)
    activities: List[ActivityEvent] = Field(default_factory=list)
    activities_v1: List[ActivityObservationV1] = Field(default_factory=list)
    camera_summaries: List[CameraActivitySummaryV1] = Field(default_factory=list)
    latest_risk_events: List[RiskEventV1] = Field(default_factory=list)
    iuris_delivery: IurisDeliveryMetrics = Field(default_factory=IurisDeliveryMetrics)
    metrics: SystemMetricsSnapshot


class StateSnapshotStore:
    """Thread-safe atomic reference holder for non-blocking API state, bounded event history, and JPEG previews."""

    def __init__(
        self,
        initial: ApplicationSnapshot,
        event_store: Optional[BoundedEventStore] = None,
        on_risk_status_change: Optional[Callable[[RiskEventV1], None]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._snapshot = initial
        self._latest_jpeg_by_camera: Dict[str, bytes] = {}
        self.event_store = event_store or BoundedEventStore()
        self.on_risk_status_change = on_risk_status_change

    def publish(
        self,
        snapshot: ApplicationSnapshot,
        camera_id: Optional[str] = None,
        jpeg_bytes: Optional[bytes] = None,
    ) -> None:
        with self._lock:
            self._snapshot = snapshot
            if camera_id is not None and jpeg_bytes is not None:
                self._latest_jpeg_by_camera[camera_id] = jpeg_bytes

    def get(self) -> ApplicationSnapshot:
        with self._lock:
            return self._snapshot

    def get_latest_jpeg(self, camera_id: Optional[str] = None) -> Optional[bytes]:
        with self._lock:
            if camera_id is not None:
                return self._latest_jpeg_by_camera.get(camera_id)
            if self._latest_jpeg_by_camera:
                first_key = next(iter(self._latest_jpeg_by_camera))
                return self._latest_jpeg_by_camera[first_key]
            return None


def get_process_rss_mb() -> float:
    """Return current process resident memory in MiB on Linux."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as fh:
            parts = fh.read().split()
            if len(parts) >= 2:
                resident_pages = int(parts[1])
                return (resident_pages * 4096.0) / (1024.0 * 1024.0)
    except Exception:
        pass
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage_kb) / 1024.0


def format_prometheus_metrics(snapshot: ApplicationSnapshot) -> str:
    """Render SystemMetricsSnapshot in standard Prometheus text exposition format with bounded-cardinality labels."""
    m = snapshot.metrics
    d = snapshot.iuris_delivery
    lines = [
        "# HELP cctv_uptime_seconds Application uptime in seconds",
        "# TYPE cctv_uptime_seconds gauge",
        f"cctv_uptime_seconds {m.uptime_seconds}",
        "# HELP cctv_frames_received_total Total video frames read from all camera sources",
        "# TYPE cctv_frames_received_total counter",
        f"cctv_frames_received_total {m.frames_received}",
        "# HELP cctv_frames_processed_total Total frames processed through pose and tracking pipeline",
        "# TYPE cctv_frames_processed_total counter",
        f"cctv_frames_processed_total {m.frames_processed}",
        "# HELP cctv_frames_dropped_total Total stale frames dropped by bounded queues",
        "# TYPE cctv_frames_dropped_total counter",
        f"cctv_frames_dropped_total {m.frames_dropped}",
        "# HELP cctv_mean_inference_ms Mean pose estimation inference latency in milliseconds",
        "# TYPE cctv_mean_inference_ms gauge",
        f"cctv_mean_inference_ms {m.mean_inference_ms}",
        "# HELP cctv_mean_classification_ms Mean temporal activity classification latency in milliseconds",
        "# TYPE cctv_mean_classification_ms gauge",
        f"cctv_mean_classification_ms {m.mean_classification_ms}",
        "# HELP cctv_mean_end_to_end_ms Mean end-to-end frame processing latency in milliseconds",
        "# TYPE cctv_mean_end_to_end_ms gauge",
        f"cctv_mean_end_to_end_ms {m.mean_end_to_end_ms}",
        "# HELP cctv_throughput_fps Current pipeline processing throughput in frames per second",
        "# TYPE cctv_throughput_fps gauge",
        f"cctv_throughput_fps {m.throughput_fps}",
        "# HELP cctv_active_cameras_count Number of currently connected cameras",
        "# TYPE cctv_active_cameras_count gauge",
        f"cctv_active_cameras_count {m.active_cameras}",
        "# HELP cctv_disconnected_cameras_count Number of disconnected or failed cameras",
        "# TYPE cctv_disconnected_cameras_count gauge",
        f"cctv_disconnected_cameras_count {m.disconnected_cameras}",
        "# HELP cctv_active_tracks_count Number of confirmed active person tracks across cameras",
        "# TYPE cctv_active_tracks_count gauge",
        f"cctv_active_tracks_count {m.active_tracks}",
        "# HELP cctv_risk_events_generated_total Total risk events generated by the risk engine",
        "# TYPE cctv_risk_events_generated_total counter",
        f"cctv_risk_events_generated_total {m.risk_events_generated}",
        "# HELP cctv_risk_events_active_count Number of currently active (OPEN or ACKNOWLEDGED) risk events",
        "# TYPE cctv_risk_events_active_count gauge",
        f"cctv_risk_events_active_count {m.active_risk_events}",
        "# HELP cctv_duplicate_alerts_suppressed_total Total duplicate risk alerts suppressed during cooldown",
        "# TYPE cctv_duplicate_alerts_suppressed_total counter",
        f"cctv_duplicate_alerts_suppressed_total {m.duplicate_alerts_suppressed}",
        "# HELP cctv_event_queue_depth Current outbound IURIS event delivery queue depth",
        "# TYPE cctv_event_queue_depth gauge",
        f"cctv_event_queue_depth {m.event_queue_depth}",
        "# HELP cctv_iuris_deliveries_attempted_total Total outbound HTTP delivery attempts to IURIS",
        "# TYPE cctv_iuris_deliveries_attempted_total counter",
        f"cctv_iuris_deliveries_attempted_total {d.events_attempted}",
        "# HELP cctv_iuris_deliveries_succeeded_total Total outbound events successfully delivered to IURIS",
        "# TYPE cctv_iuris_deliveries_succeeded_total counter",
        f"cctv_iuris_deliveries_succeeded_total {d.events_delivered}",
        "# HELP cctv_event_delivery_failures_total Total outbound IURIS event delivery failures",
        "# TYPE cctv_event_delivery_failures_total counter",
        f"cctv_event_delivery_failures_total {m.event_delivery_failures}",
        "# HELP cctv_iuris_retries_total Total retry attempts made by the IURIS delivery client",
        "# TYPE cctv_iuris_retries_total counter",
        f"cctv_iuris_retries_total {d.events_retried}",
        "# HELP cctv_iuris_delivery_latency_ms Mean IURIS HTTP delivery latency in milliseconds",
        "# TYPE cctv_iuris_delivery_latency_ms gauge",
        f"cctv_iuris_delivery_latency_ms {d.mean_delivery_latency_ms}",
        "# HELP cctv_model_ready Whether the pose estimation and activity models are ready (1=ready, 0=not ready)",
        "# TYPE cctv_model_ready gauge",
        f"cctv_model_ready {1 if m.model_ready else 0}",
        "# HELP cctv_memory_rss_mb Resident set size memory in MiB",
        "# TYPE cctv_memory_rss_mb gauge",
        f"cctv_memory_rss_mb {m.memory_rss_mb}",
    ]
    return "\n".join(lines) + "\n"
