"""Synthetic end-to-end demo for crowd-running alerts and IURIS lifecycle delivery.

No camera, external network connection, or real person data is required.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from activity_system.config import (
    ConcentrationRuleConfig,
    IurisConfig,
    RiskEngineConfig,
    RiskRulesConfig,
)
from activity_system.iuris.client import IurisDeliveryClient
from activity_system.risk_engine import RiskAssessmentEngine
from activity_system.schemas import (
    ActivityCode,
    ActivityObservationV1,
    AlertLifecycleState,
    RiskEventV1,
    RiskSeverity,
    RiskStatus,
    RiskStatusHistoryEntry,
)


CAMERA_ID = "cam_iu_main_gate"
CAMERA_NAME = "Islamic University Main Gate"
MODEL_VERSION = "heuristic-kinematic-v1.0"


def _utc_at(base: datetime, seconds: float) -> str:
    return (base + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _observations(timestamp: str, running_count: int) -> List[ActivityObservationV1]:
    """Build 12 synthetic, quality-qualified track observations."""
    observations = []
    for index in range(12):
        activity = ActivityCode.RUNNING if index < running_count else ActivityCode.WALKING
        observations.append(
            ActivityObservationV1(
                camera_id=CAMERA_ID,
                track_id=f"demo-track-{index + 1:02d}",
                timestamp=timestamp,
                activity=activity,
                confidence=0.95,
                observation_quality=0.95,
                duration_seconds=0.5,
                pose_available=True,
                model_version=MODEL_VERSION,
                metadata={"source": "synthetic_demo"},
            )
        )
    return observations


def _deliver(client: IurisDeliveryClient, event: RiskEventV1) -> None:
    if not client.enqueue(event):
        raise RuntimeError(
            f"IURIS demo could not enqueue lifecycle state {event.lifecycle_state!r}"
        )
    receipts = client.flush()
    if not receipts or not all(receipt.delivered for receipt in receipts):
        raise RuntimeError(
            f"IURIS dry-run delivery failed for lifecycle state {event.lifecycle_state!r}"
        )


def run_crowd_running_demo() -> Dict[str, Any]:
    """Run the crowd-risk rule and exercise triggered, acknowledged, resolved delivery."""
    rule_config = ConcentrationRuleConfig(
        enabled=True,
        activity=ActivityCode.RUNNING,
        percentage_threshold=40.0,
        minimum_people=5,
        window_seconds=10.0,
        minimum_duration_seconds=5.0,
        min_observation_quality=0.45,
        clear_grace_seconds=1.5,
        emit_resolved_events=True,
        severity=RiskSeverity.MEDIUM,
        cooldown_seconds=10.0,
    )
    engine = RiskAssessmentEngine(
        config=RiskEngineConfig(
            enabled=True,
            rules=RiskRulesConfig(
                unusual_running_concentration=rule_config
            ),
        )
    )
    client = IurisDeliveryClient(
        IurisConfig(
            delivery_enabled=True,
            dry_run=True,
            deliver_activity_events=False,
            deliver_summary_events=False,
            deliver_risk_events=True,
        )
    )

    base = datetime.now(timezone.utc)
    triggered: RiskEventV1 | None = None
    resolved: RiskEventV1 | None = None

    def evaluate(offset: float, running_count: int) -> List[RiskEventV1]:
        timestamp = _utc_at(base, offset)
        return engine.evaluate(
            camera_id=CAMERA_ID,
            monotonic_ts=offset,
            utc_iso=timestamp,
            activities=_observations(timestamp, running_count),
            summary=None,
            model_version=MODEL_VERSION,
            session_id="synthetic-crowd-demo",
        )

    try:
        # Six seconds of sustained 7/12 running activity should trigger at >= 5 s.
        for offset in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            for event in evaluate(offset, running_count=7):
                if event.lifecycle_state == AlertLifecycleState.TRIGGERED.value:
                    triggered = event
                    break
            if triggered is not None:
                break

        if triggered is None:
            raise RuntimeError(
                "The risk engine did not trigger for 7 running people out of 12."
            )

        # Enrich with configured camera metadata before outbound delivery.
        triggered.camera_name = CAMERA_NAME
        _deliver(client, triggered)

        # Persist the alert in a real event store so acknowledgement has an audit trail.
        from activity_system.event_store import BoundedEventStore

        store = BoundedEventStore()
        store.add_risk_event(triggered)
        acknowledged = store.acknowledge_risk_event(
            triggered.event_id,
            actor="demo-operator",
            note="Acknowledged by synthetic demo operator.",
            timestamp_utc=_utc_at(base, 5.1),
        )
        if acknowledged is None:
            raise RuntimeError("Could not acknowledge the triggered demo alert.")
        acknowledged.camera_name = CAMERA_NAME
        _deliver(client, acknowledged)

        # Keep the running ratio qualified for several frames to exercise deduplication.
        for offset in (5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75):
            evaluate(offset, running_count=7)

        # Clear the rule after its grace period; the engine should emit RESOLVED.
        for event in evaluate(8.5, running_count=0):
            if event.lifecycle_state == AlertLifecycleState.RESOLVED.value:
                resolved = event
                break

        if resolved is None:
            raise RuntimeError(
                "The risk engine did not emit RESOLVED after the crowd condition cleared."
            )

        resolved.camera_name = CAMERA_NAME
        _deliver(client, resolved)

        records = client.dry_run_records
        lifecycle_states = [
            record.payload.get("lifecycle_state")
            for record in records
            if record.event_type == "risk.detected"
        ]
        expected = [
            AlertLifecycleState.TRIGGERED.value,
            AlertLifecycleState.ACKNOWLEDGED.value,
            AlertLifecycleState.RESOLVED.value,
        ]
        if lifecycle_states != expected:
            raise RuntimeError(
                f"Unexpected IURIS lifecycle sequence: {lifecycle_states!r}"
            )

        return {
            "triggered_alert": {
                "camera_id": triggered.camera_id,
                "camera_name": triggered.camera_name,
                "observed_people": triggered.observed_people,
                "matching_people": triggered.matching_people,
                "measured_percentage": triggered.measured_percentage,
                "duration_seconds": triggered.duration_seconds,
            },
            "duplicate_alerts_suppressed": engine.duplicates_suppressed,
            "iuris_lifecycle_states_received": lifecycle_states,
            "final_alert_status": resolved.status.value
            if hasattr(resolved.status, "value")
            else str(resolved.status),
        }
    finally:
        client.close(drain_remaining=True)


if __name__ == "__main__":
    import json

    print(json.dumps(run_crowd_running_demo(), indent=2))
