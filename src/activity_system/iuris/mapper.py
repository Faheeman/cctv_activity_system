"""Stateless mapper converting internal activity, summary, and risk models into IURIS-compatible envelopes."""

from typing import Any, Dict, Union

from activity_system.iuris.retry import build_idempotency_key
from activity_system.iuris.schemas import IurisOutboundEnvelope
from activity_system.schemas import (
    ActivityEvent,
    ActivityObservationV1,
    AlertLifecycleState,
    CameraActivitySummaryV1,
    RiskEventV1,
)


_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {"frame", "raw_frame", "image", "face_embedding", "biometric_id", "password", "keypoints"}
)


class IurisEventMapper:
    """Maps internal backend events into validated, privacy-safe IURIS v1.0 outbound envelopes."""

    @staticmethod
    def _sanitize_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for k, v in payload.items():
            if k in _FORBIDDEN_PAYLOAD_KEYS:
                continue
            if isinstance(v, dict):
                cleaned[k] = IurisEventMapper._sanitize_dict(v)
            else:
                cleaned[k] = v
        return cleaned

    @classmethod
    def map_activity(
        cls, event: Union[ActivityEvent, ActivityObservationV1]
    ) -> IurisOutboundEnvelope:
        obs: ActivityObservationV1 = event.to_v1() if isinstance(event, ActivityEvent) else event
        payload = cls._sanitize_dict(obs.dict())
        idem_key = build_idempotency_key(obs.event_id, obs.event_type, obs.schema_version)
        return IurisOutboundEnvelope(
            schema_version=obs.schema_version,
            event_id=obs.event_id,
            dedup_key=obs.event_id,
            event_type="activity.detected",
            camera_id=obs.camera_id,
            track_id=obs.track_id,
            timestamp=obs.timestamp,
            idempotency_key=idem_key,
            payload=payload,
        )

    @classmethod
    def map_camera_summary(cls, summary: CameraActivitySummaryV1) -> IurisOutboundEnvelope:
        payload = cls._sanitize_dict(summary.dict())
        idem_key = build_idempotency_key(summary.event_id, summary.event_type, summary.schema_version)
        return IurisOutboundEnvelope(
            schema_version=summary.schema_version,
            event_id=summary.event_id,
            dedup_key=summary.event_id,
            event_type="camera.activity_summary",
            camera_id=summary.camera_id,
            track_id=None,
            timestamp=summary.timestamp,
            idempotency_key=idem_key,
            payload=payload,
        )

    @classmethod
    def map_risk_event(cls, risk: RiskEventV1) -> IurisOutboundEnvelope:
        payload = cls._sanitize_dict(risk.dict())
        if payload.get("activity") is not None and hasattr(payload["activity"], "value"):
            payload["activity"] = payload["activity"].value
        if hasattr(payload.get("severity"), "value"):
            payload["severity"] = payload["severity"].value
        if hasattr(payload.get("status"), "value"):
            payload["status"] = payload["status"].value

        # For TRIGGERED events, deduplicate strictly by event_id; for subsequent lifecycle transitions
        # (UPDATED, ACKNOWLEDGED, RESOLVED), include lifecycle_state and occurrence_count so status transitions deliver.
        if risk.lifecycle_state == AlertLifecycleState.TRIGGERED.value and risk.occurrence_count == 1:
            dedup_token = risk.event_id
        else:
            dedup_token = f"{risk.event_id}:{risk.lifecycle_state}:{risk.occurrence_count}:{risk.updated_at}"

        idem_key = build_idempotency_key(dedup_token, risk.event_type, risk.schema_version)
        return IurisOutboundEnvelope(
            schema_version=risk.schema_version,
            event_id=risk.event_id,
            dedup_key=dedup_token,
            event_type="risk.detected",
            camera_id=risk.camera_id,
            track_id=risk.track_id,
            timestamp=risk.timestamp,
            idempotency_key=idem_key,
            payload=payload,
        )
