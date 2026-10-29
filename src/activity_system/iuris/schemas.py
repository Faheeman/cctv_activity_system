"""Versioned schemas for IURIS outbound event delivery and telemetry."""

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


class IurisOutboundEnvelope(BaseModel):
    """Canonical versioned event envelope transmitted to IURIS or written to the durable outbox."""

    schema_version: str = Field(default="1.0", description="Contract schema version")
    event_id: str = Field(..., description="Unique UUIDv4 identifier for idempotency")
    dedup_key: str = Field(default="", description="Internal deduplication key supporting lifecycle state updates")
    event_type: Literal["activity.detected", "camera.activity_summary", "risk.detected"]
    camera_id: str = Field(..., description="Configured camera identifier")
    track_id: Optional[str] = Field(
        default=None, description="Temporary camera-scoped tracking ID (null for camera summaries)"
    )
    timestamp: str = Field(..., description="UTC ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SSZ)")
    idempotency_key: str = Field(..., description="Deterministic idempotency token for HTTP delivery")
    payload: Dict[str, Any] = Field(
        ..., description="Validated v1.0 event payload (ActivityObservationV1, CameraActivitySummaryV1, or RiskEventV1)"
    )


class IurisDeliveryReceipt(BaseModel):
    """Outcome of a single outbound event delivery operation."""

    event_id: str
    event_type: str
    idempotency_key: str
    delivered: bool
    attempts: int
    status_code: Optional[int] = None
    latency_ms: float = 0.0
    dead_lettered: bool = False
    dry_run: bool = False
    error: Optional[str] = None


class IurisDeliveryMetrics(BaseModel):
    """Operational metrics for the optional IURIS outbound delivery subsystem."""

    delivery_enabled: bool = False
    dry_run: bool = False
    healthy: bool = True
    destination_url_redacted: str = ""
    queue_depth: int = 0
    max_queue_size: int = 1000
    events_enqueued: int = 0
    events_attempted: int = 0
    events_delivered: int = 0
    events_failed: int = 0
    events_retried: int = 0
    duplicates_skipped: int = 0
    dead_letter_count: int = 0
    mean_delivery_latency_ms: float = 0.0
    last_delivery_latency_ms: float = 0.0
    last_delivery_utc_iso: Optional[str] = None
    last_error: Optional[str] = None
