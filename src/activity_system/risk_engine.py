"""Independent, configurable, explainable Risk Assessment Engine with alert lifecycle and hysteresis."""

from typing import Dict, List, Optional, Sequence, Tuple

from activity_system.config import (
    ConcentrationRuleConfig,
    FallRuleConfig,
    ImmobilityRuleConfig,
    PersistentActivityPatternConfig,
    RiskEngineConfig,
    ZoneDefinitionConfig,
)
from activity_system.interfaces import RiskRuleProtocol
from activity_system.schemas import (
    ActivityCode,
    ActivityObservationV1,
    AlertLifecycleState,
    CameraActivitySummaryV1,
    RiskEventV1,
    RiskStatus,
    RiskStatusHistoryEntry,
    normalize_utc_iso,
)


def point_in_polygon(x: float, y: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon check for 2D image coordinates."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    px1, py1 = polygon[0]
    for i in range(1, n + 1):
        px2, py2 = polygon[i % n]
        if y > min(py1, py2):
            if y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xinters = (y - py1) * (px2 - px1) / float(py2 - py1) + px1
                    else:
                        xinters = px1
                    if px1 == px2 or x <= xinters:
                        inside = not inside
        px1, py1 = px2, py2
    return inside


class _CameraConcentrationState:
    """Tracks ongoing crowd-activity concentration state and active correlated alert per camera."""

    __slots__ = (
        "first_qualified_ts",
        "last_qualified_ts",
        "last_emitted_ts",
        "active_alert",
    )

    def __init__(
        self,
        first_qualified_ts: float,
        last_qualified_ts: float,
    ) -> None:
        self.first_qualified_ts = float(first_qualified_ts)
        self.last_qualified_ts = float(last_qualified_ts)
        self.last_emitted_ts: Optional[float] = None
        self.active_alert: Optional[RiskEventV1] = None


class UnusualActivityConcentrationRule:
    """Evaluates crowd-running / activity concentration with quality filtering, hysteresis grace, and lifecycle."""

    def __init__(self, config: ConcentrationRuleConfig) -> None:
        self.config = config
        self._state_by_camera: Dict[str, _CameraConcentrationState] = {}
        self.duplicates_suppressed: int = 0

    @property
    def rule_name(self) -> str:
        return "unusual_running_concentration"

    def reset_camera(self, camera_id: str) -> None:
        self._state_by_camera.pop(camera_id, None)

    def _compute_crowd_metrics(
        self,
        cfg: ConcentrationRuleConfig,
        activities: Sequence[ActivityObservationV1],
        summary: Optional[CameraActivitySummaryV1],
    ) -> Tuple[bool, int, int, float, float, List[str], Optional[float], int]:
        """Compute (has_frame_evidence, observed_people, matching_people, measured_pct, quality, track_ids, mean_conf, sample_count)."""
        target_act = cfg.activity

        if activities:
            valid_obs: List[ActivityObservationV1] = []
            for obs in activities:
                if obs.observation_quality < cfg.min_observation_quality:
                    continue
                if (
                    cfg.min_confidence is not None
                    and obs.confidence is not None
                    and obs.confidence < cfg.min_confidence
                ):
                    continue
                valid_obs.append(obs)

            if not valid_obs:
                return True, 0, 0, 0.0, 0.0, [], None, len(activities)

            matching = [
                o
                for o in valid_obs
                if o.activity == target_act and o.activity != ActivityCode.UNKNOWN
            ]
            observed_people = len(valid_obs)
            matching_count = len(matching)
            measured_pct = round((float(matching_count) * 100.0) / float(observed_people), 2)
            mean_q = round(
                sum(float(o.observation_quality) for o in valid_obs) / float(observed_people), 4
            )
            confs = [float(o.confidence) for o in matching if o.confidence is not None]
            mean_conf = round(sum(confs) / len(confs), 4) if confs else None
            track_ids = [o.track_id for o in matching[:25]]
            sample_cnt = summary.sample_count if summary is not None else observed_people
            return (
                True,
                observed_people,
                matching_count,
                measured_pct,
                mean_q,
                track_ids,
                mean_conf,
                sample_cnt,
            )

        if summary is not None:
            if summary.observation_quality < cfg.min_observation_quality:
                return True, 0, 0, 0.0, float(summary.observation_quality), [], None, summary.sample_count

            target_key = target_act.value
            if target_act == ActivityCode.UNKNOWN:
                return True, summary.observed_people, 0, 0.0, float(summary.observation_quality), [], None, summary.sample_count

            matching_count = int(summary.activity_distribution.get(target_key, 0))
            observed_people = int(summary.observed_people)
            measured_pct = float(summary.activity_percentages.get(target_key, 0.0))
            return (
                True,
                observed_people,
                matching_count,
                measured_pct,
                float(summary.observation_quality),
                [],
                None,
                int(summary.sample_count),
            )

        return False, 0, 0, 0.0, 0.0, [], None, 0

    def evaluate(
        self,
        camera_id: str,
        monotonic_ts: float,
        utc_iso: str,
        activities: Sequence[ActivityObservationV1],
        summary: Optional[CameraActivitySummaryV1],
        model_version: str,
        camera_name: Optional[str] = None,
    ) -> List[RiskEventV1]:
        cfg = self.config.for_camera(camera_id)
        if not cfg.enabled:
            return []

        (
            has_evidence,
            observed_people,
            matching_count,
            measured_pct,
            quality_score,
            matching_track_ids,
            mean_conf,
            sample_count,
        ) = self._compute_crowd_metrics(cfg, activities, summary)

        state = self._state_by_camera.get(camera_id)
        now_mono = float(monotonic_ts)
        now_iso = normalize_utc_iso(utc_iso)

        # Case 1: Missing frame / temporary drop without evidence
        if not has_evidence:
            if state is not None and (now_mono - state.last_qualified_ts) > cfg.clear_grace_seconds:
                self._state_by_camera.pop(camera_id, None)
            return []

        condition_met = (
            observed_people >= cfg.minimum_people
            and matching_count >= cfg.minimum_people
            and measured_pct >= cfg.percentage_threshold
            and quality_score >= cfg.min_observation_quality
        )

        if not condition_met:
            if state is None:
                return []
            # Hold state within clear_grace_seconds to tolerate 1-2 frame jitter/occlusion
            if (now_mono - state.last_qualified_ts) <= cfg.clear_grace_seconds:
                return []
            # Condition genuinely cleared after grace period
            self._state_by_camera.pop(camera_id, None)
            if state.active_alert is not None and cfg.emit_resolved_events:
                resolved = state.active_alert.copy(deep=True)
                prev_status = resolved.status
                resolved.status = RiskStatus.RESOLVED
                resolved.lifecycle_state = AlertLifecycleState.RESOLVED.value
                resolved.timestamp = now_iso
                resolved.updated_at = now_iso
                resolved.resolved_at = now_iso
                resolved.resolved_by = "system"
                resolved.resolution_note = (
                    f"{cfg.activity.value.title()} concentration dropped below threshold ({measured_pct:.1f}% < {cfg.percentage_threshold:.1f}%)."
                )
                resolved.status_history.append(
                    RiskStatusHistoryEntry(
                        timestamp=now_iso,
                        from_status=prev_status,
                        to_status=RiskStatus.RESOLVED,
                        lifecycle_state=AlertLifecycleState.RESOLVED.value,
                        actor="system",
                        note=resolved.resolution_note,
                    )
                )
                return [resolved]
            return []

        # Condition is met on this frame!
        if state is None:
            state = _CameraConcentrationState(
                first_qualified_ts=now_mono,
                last_qualified_ts=now_mono,
            )
            self._state_by_camera[camera_id] = state
        else:
            state.last_qualified_ts = now_mono

        sustained_sec = max(0.0, state.last_qualified_ts - state.first_qualified_ts)
        if sustained_sec + 1e-6 < cfg.minimum_duration_seconds:
            return []

        # Condition has been sustained for >= minimum_duration_seconds
        if state.active_alert is not None and state.last_emitted_ts is not None:
            if (now_mono - state.last_emitted_ts) < cfg.cooldown_seconds:
                self.duplicates_suppressed += 1
                return []

            # Cooldown elapsed while condition remains continuously active -> update existing correlated alert
            state.last_emitted_ts = now_mono
            updated = state.active_alert.copy(deep=True)
            updated.timestamp = now_iso
            updated.updated_at = now_iso
            updated.lifecycle_state = AlertLifecycleState.UPDATED.value
            updated.occurrence_count += 1
            updated.measured_percentage = round(measured_pct, 2)
            updated.observed_people = observed_people
            updated.matching_people = matching_count
            updated.duration_seconds = round(sustained_sec, 3)
            updated.observation_quality = round(quality_score, 4)
            updated.relevant_track_ids = matching_track_ids
            updated.evidence.update(
                {
                    "matching_people": matching_count,
                    "observed_people": observed_people,
                    "measured_percentage": round(measured_pct, 2),
                    "sustained_duration_seconds": round(sustained_sec, 3),
                    "occurrence_count": updated.occurrence_count,
                }
            )
            state.active_alert = updated
            return [updated]

        # Newly triggered alert!
        state.last_emitted_ts = now_mono
        target_act_str = cfg.activity.value
        win_sec = float(summary.window_seconds) if summary is not None else float(cfg.window_seconds)
        corr_key = f"{camera_id}:{self.rule_name}:{target_act_str}"

        alert = RiskEventV1(
            correlation_key=corr_key,
            camera_id=camera_id,
            camera_name=camera_name,
            timestamp=now_iso,
            created_at=now_iso,
            updated_at=now_iso,
            risk_type="UNUSUAL_ACTIVITY_CONCENTRATION",
            rule_name=self.rule_name,
            title=f"Unusual {target_act_str.title()} Concentration Detected",
            severity=cfg.severity,
            status=RiskStatus.OPEN,
            lifecycle_state=AlertLifecycleState.TRIGGERED.value,
            occurrence_count=1,
            activity=cfg.activity,
            measured_percentage=round(measured_pct, 2),
            threshold_percentage=round(float(cfg.percentage_threshold), 2),
            observed_people=observed_people,
            matching_people=matching_count,
            duration_seconds=round(sustained_sec, 3),
            window_seconds=round(win_sec, 3),
            observation_quality=round(quality_score, 4),
            confidence=mean_conf,
            relevant_track_ids=matching_track_ids,
            evidence={
                "rule": self.rule_name,
                "target_activity": target_act_str,
                "matching_people": matching_count,
                "observed_people": observed_people,
                "eligible_people": summary.eligible_people if summary is not None else observed_people,
                "measured_percentage": round(measured_pct, 2),
                "threshold_percentage": round(float(cfg.percentage_threshold), 2),
                "minimum_people_required": cfg.minimum_people,
                "minimum_duration_seconds": cfg.minimum_duration_seconds,
                "sustained_duration_seconds": round(sustained_sec, 3),
                "window_seconds": win_sec,
                "sample_count": sample_count,
            },
            explanation="The configured running-concentration threshold was met.",
            model_version=model_version,
            status_history=[
                RiskStatusHistoryEntry(
                    timestamp=now_iso,
                    from_status=None,
                    to_status=RiskStatus.OPEN,
                    lifecycle_state=AlertLifecycleState.TRIGGERED.value,
                    actor="system",
                    note=f"Triggered: {matching_count}/{observed_people} people ({measured_pct:.1f}%) {target_act_str.lower()} for {sustained_sec:.1f}s.",
                )
            ],
        )
        state.active_alert = alert
        return [alert]


class PossibleFallRule:
    """Evaluates temporal fall observations for individual tracked persons."""

    def __init__(self, config: FallRuleConfig) -> None:
        self.config = config
        self._fall_start_by_track: Dict[str, float] = {}
        self._last_alert_by_track: Dict[str, float] = {}
        self.duplicates_suppressed: int = 0

    @property
    def rule_name(self) -> str:
        return "possible_fall"

    def evaluate(
        self,
        camera_id: str,
        monotonic_ts: float,
        utc_iso: str,
        activities: Sequence[ActivityObservationV1],
        summary: Optional[CameraActivitySummaryV1],
        model_version: str,
    ) -> List[RiskEventV1]:
        if not self.config.enabled:
            return []

        events: List[RiskEventV1] = []
        now_iso = normalize_utc_iso(utc_iso)

        for obs in activities:
            tid = obs.track_id
            if (
                obs.activity == ActivityCode.FALLING
                and obs.observation_quality >= self.config.min_observation_quality
            ):
                start_ts = self._fall_start_by_track.setdefault(
                    tid, float(monotonic_ts) - float(obs.duration_seconds)
                )
                sustained = max(float(obs.duration_seconds), float(monotonic_ts) - start_ts)
                if sustained + 1e-6 >= self.config.minimum_duration_seconds:
                    last_alert = self._last_alert_by_track.get(tid)
                    if (
                        last_alert is not None
                        and (float(monotonic_ts) - last_alert) < self.config.cooldown_seconds
                    ):
                        self.duplicates_suppressed += 1
                        continue
                    self._last_alert_by_track[tid] = float(monotonic_ts)
                    events.append(
                        RiskEventV1(
                            correlation_key=f"{camera_id}:{self.rule_name}:{tid}",
                            camera_id=camera_id,
                            timestamp=now_iso,
                            risk_type="POSSIBLE_FALL_DETECTED",
                            rule_name=self.rule_name,
                            title="Possible Person Fall Detected",
                            severity=self.config.severity,
                            status=RiskStatus.OPEN,
                            activity=ActivityCode.FALLING,
                            observed_people=summary.observed_people if summary else len(activities),
                            matching_people=1,
                            duration_seconds=round(sustained, 3),
                            window_seconds=summary.window_seconds if summary else 1.5,
                            observation_quality=round(float(obs.observation_quality), 4),
                            confidence=obs.confidence,
                            track_id=tid,
                            relevant_track_ids=[tid],
                            evidence={
                                "rule": self.rule_name,
                                "track_id": tid,
                                "bbox": obs.bbox,
                                "sustained_duration_seconds": round(sustained, 3),
                                "minimum_duration_seconds": self.config.minimum_duration_seconds,
                                "min_observation_quality": self.config.min_observation_quality,
                                "metadata": obs.metadata,
                            },
                            explanation=(
                                "Temporal pose trajectory indicates a rapid vertical posture collapse consistent with a possible fall."
                            ),
                            model_version=model_version,
                            status_history=[
                                RiskStatusHistoryEntry(
                                    timestamp=now_iso,
                                    from_status=None,
                                    to_status=RiskStatus.OPEN,
                                    lifecycle_state=AlertLifecycleState.TRIGGERED.value,
                                    actor="system",
                                )
                            ],
                        )
                    )
            else:
                self._fall_start_by_track.pop(tid, None)

        return events


class ProlongedImmobilityRule:
    """Evaluates whether a tracked person remains stationary in a lying posture beyond a threshold duration."""

    def __init__(self, config: ImmobilityRuleConfig) -> None:
        self.config = config
        self._lying_start_by_track: Dict[str, float] = {}
        self._last_alert_by_track: Dict[str, float] = {}
        self.duplicates_suppressed: int = 0

    @property
    def rule_name(self) -> str:
        return "prolonged_immobility"

    def evaluate(
        self,
        camera_id: str,
        monotonic_ts: float,
        utc_iso: str,
        activities: Sequence[ActivityObservationV1],
        summary: Optional[CameraActivitySummaryV1],
        model_version: str,
    ) -> List[RiskEventV1]:
        if not self.config.enabled:
            return []

        events: List[RiskEventV1] = []
        now_iso = normalize_utc_iso(utc_iso)

        for obs in activities:
            tid = obs.track_id
            if (
                obs.activity == self.config.activity
                and obs.observation_quality >= self.config.min_observation_quality
            ):
                start_ts = self._lying_start_by_track.setdefault(
                    tid, float(monotonic_ts) - float(obs.duration_seconds)
                )
                sustained = max(float(obs.duration_seconds), float(monotonic_ts) - start_ts)
                if sustained + 1e-6 >= self.config.immobility_duration_seconds:
                    last_alert = self._last_alert_by_track.get(tid)
                    if (
                        last_alert is not None
                        and (float(monotonic_ts) - last_alert) < self.config.cooldown_seconds
                    ):
                        self.duplicates_suppressed += 1
                        continue
                    self._last_alert_by_track[tid] = float(monotonic_ts)
                    events.append(
                        RiskEventV1(
                            correlation_key=f"{camera_id}:{self.rule_name}:{tid}",
                            camera_id=camera_id,
                            timestamp=now_iso,
                            risk_type="PROLONGED_IMMOBILITY",
                            rule_name=self.rule_name,
                            title="Prolonged Person Immobility Detected",
                            severity=self.config.severity,
                            status=RiskStatus.OPEN,
                            activity=self.config.activity,
                            observed_people=summary.observed_people if summary else len(activities),
                            matching_people=1,
                            duration_seconds=round(sustained, 3),
                            window_seconds=summary.window_seconds if summary else self.config.immobility_duration_seconds,
                            observation_quality=round(float(obs.observation_quality), 4),
                            confidence=obs.confidence,
                            track_id=tid,
                            relevant_track_ids=[tid],
                            evidence={
                                "rule": self.rule_name,
                                "track_id": tid,
                                "bbox": obs.bbox,
                                "immobility_duration_seconds": round(sustained, 3),
                                "threshold_duration_seconds": self.config.immobility_duration_seconds,
                            },
                            explanation=(
                                "A tracked person has remained in a horizontal posture continuously beyond the configured immobility duration threshold."
                            ),
                            model_version=model_version,
                            status_history=[
                                RiskStatusHistoryEntry(
                                    timestamp=now_iso,
                                    from_status=None,
                                    to_status=RiskStatus.OPEN,
                                    lifecycle_state=AlertLifecycleState.TRIGGERED.value,
                                    actor="system",
                                )
                            ],
                        )
                    )
            else:
                self._lying_start_by_track.pop(tid, None)

        return events


class PersistentActivityPatternRule:
    """Evaluates configurable persistent unusual activity patterns per camera."""

    def __init__(self, patterns: Sequence[PersistentActivityPatternConfig]) -> None:
        self.patterns = list(patterns)
        self._start_by_key: Dict[Tuple[str, str], float] = {}
        self._last_alert_by_key: Dict[Tuple[str, str], float] = {}
        self.duplicates_suppressed: int = 0

    @property
    def rule_name(self) -> str:
        return "persistent_activity_pattern"

    def evaluate(
        self,
        camera_id: str,
        monotonic_ts: float,
        utc_iso: str,
        activities: Sequence[ActivityObservationV1],
        summary: Optional[CameraActivitySummaryV1],
        model_version: str,
    ) -> List[RiskEventV1]:
        events: List[RiskEventV1] = []
        now_iso = normalize_utc_iso(utc_iso)

        for pat in self.patterns:
            if not pat.enabled:
                continue
            if pat.camera_id is not None and pat.camera_id != camera_id:
                continue
            key = (camera_id, pat.pattern_id)
            matching = [
                o
                for o in activities
                if o.activity == pat.activity
                and o.observation_quality >= pat.min_observation_quality
            ]
            if len(matching) < pat.minimum_people:
                self._start_by_key.pop(key, None)
                continue

            start_ts = self._start_by_key.setdefault(key, float(monotonic_ts))
            sustained = max(0.0, float(monotonic_ts) - start_ts)
            if sustained + 1e-6 < pat.minimum_duration_seconds:
                continue

            last_alert = self._last_alert_by_key.get(key)
            if last_alert is not None and (float(monotonic_ts) - last_alert) < pat.cooldown_seconds:
                self.duplicates_suppressed += 1
                continue

            self._last_alert_by_key[key] = float(monotonic_ts)
            mean_q = sum(float(o.observation_quality) for o in matching) / len(matching)
            events.append(
                RiskEventV1(
                    correlation_key=f"{camera_id}:{pat.pattern_id}:{pat.activity.value}",
                    camera_id=camera_id,
                    timestamp=now_iso,
                    risk_type="PERSISTENT_UNUSUAL_ACTIVITY",
                    rule_name=f"{self.rule_name}:{pat.pattern_id}",
                    title=f"Persistent {pat.activity.value.title()} Pattern ({pat.pattern_id})",
                    severity=pat.severity,
                    status=RiskStatus.OPEN,
                    activity=pat.activity,
                    observed_people=len(activities),
                    matching_people=len(matching),
                    duration_seconds=round(sustained, 3),
                    observation_quality=round(mean_q, 4),
                    relevant_track_ids=[o.track_id for o in matching[:25]],
                    evidence={
                        "pattern_id": pat.pattern_id,
                        "activity": pat.activity.value,
                        "matching_people": len(matching),
                        "minimum_people": pat.minimum_people,
                        "sustained_duration_seconds": round(sustained, 3),
                        "minimum_duration_seconds": pat.minimum_duration_seconds,
                    },
                    explanation=(
                        f"Configured activity pattern '{pat.pattern_id}' ({pat.activity.value}) persisted for {sustained:.1f}s."
                    ),
                    model_version=model_version,
                )
            )
        return events


class ZoneActivityRule:
    """Evaluates configurable polygon zone activity rules when zone geometry is configured."""

    def __init__(self, zones: Sequence[ZoneDefinitionConfig]) -> None:
        self.zones = list(zones)
        self._entry_start_by_zone_track: Dict[Tuple[str, str], float] = {}
        self._last_alert_by_zone_track: Dict[Tuple[str, str], float] = {}
        self.duplicates_suppressed: int = 0

    @property
    def rule_name(self) -> str:
        return "zone_activity"

    def evaluate(
        self,
        camera_id: str,
        monotonic_ts: float,
        utc_iso: str,
        activities: Sequence[ActivityObservationV1],
        summary: Optional[CameraActivitySummaryV1],
        model_version: str,
    ) -> List[RiskEventV1]:
        events: List[RiskEventV1] = []
        camera_zones = [
            z for z in self.zones if z.enabled and z.camera_id == camera_id and len(z.polygon) >= 3
        ]
        if not camera_zones:
            return []

        now_iso = normalize_utc_iso(utc_iso)

        for zone in camera_zones:
            allowed_triggers = set(zone.trigger_activities)
            for obs in activities:
                key = (zone.zone_id, obs.track_id)
                if (
                    not obs.bbox
                    or len(obs.bbox) != 4
                    or obs.activity not in allowed_triggers
                    or obs.observation_quality < zone.min_observation_quality
                ):
                    self._entry_start_by_zone_track.pop(key, None)
                    continue

                x1, y1, x2, y2 = obs.bbox
                foot_x = 0.5 * float(x1 + x2)
                foot_y = float(y2)
                center_y = 0.5 * float(y1 + y2)

                in_zone = point_in_polygon(foot_x, foot_y, zone.polygon) or point_in_polygon(
                    foot_x, center_y, zone.polygon
                )
                if not in_zone:
                    self._entry_start_by_zone_track.pop(key, None)
                    continue

                start_ts = self._entry_start_by_zone_track.setdefault(key, float(monotonic_ts))
                sustained = max(0.0, float(monotonic_ts) - start_ts)
                if sustained + 1e-6 < zone.minimum_duration_seconds:
                    continue

                last_alert = self._last_alert_by_zone_track.get(key)
                if last_alert is not None and (float(monotonic_ts) - last_alert) < zone.cooldown_seconds:
                    self.duplicates_suppressed += 1
                    continue

                self._last_alert_by_zone_track[key] = float(monotonic_ts)
                events.append(
                    RiskEventV1(
                        correlation_key=f"{camera_id}:{zone.zone_id}:{obs.track_id}",
                        camera_id=camera_id,
                        timestamp=now_iso,
                        risk_type="ZONE_RESTRICTED_ACTIVITY",
                        rule_name=f"{self.rule_name}:{zone.zone_id}",
                        title=f"Restricted Zone Activity: {zone.name}",
                        severity=zone.severity,
                        status=RiskStatus.OPEN,
                        activity=obs.activity,
                        observed_people=summary.observed_people if summary else len(activities),
                        matching_people=1,
                        duration_seconds=round(sustained, 3),
                        window_seconds=summary.window_seconds if summary else zone.minimum_duration_seconds,
                        observation_quality=round(float(obs.observation_quality), 4),
                        confidence=obs.confidence,
                        track_id=obs.track_id,
                        relevant_track_ids=[obs.track_id],
                        zone_id=zone.zone_id,
                        evidence={
                            "rule": self.rule_name,
                            "zone_id": zone.zone_id,
                            "zone_name": zone.name,
                            "track_id": obs.track_id,
                            "activity": obs.activity.value,
                            "bbox": obs.bbox,
                            "foot_point": [foot_x, foot_y],
                            "sustained_duration_seconds": round(sustained, 3),
                            "minimum_duration_seconds": zone.minimum_duration_seconds,
                        },
                        explanation=(
                            f"Configured activity '{obs.activity.value}' was observed inside zone '{zone.name}' ({zone.zone_id}) for at least {zone.minimum_duration_seconds:.1f}s."
                        ),
                        model_version=model_version,
                    )
                )
        return events


class RiskAssessmentEngine:
    """Independent risk engine evaluating built-in and custom rules with session awareness and deduplication."""

    def __init__(
        self,
        config: Optional[RiskEngineConfig] = None,
        custom_rules: Optional[Sequence[RiskRuleProtocol]] = None,
    ) -> None:
        self.config = config or RiskEngineConfig()
        self._concentration_rule = UnusualActivityConcentrationRule(
            self.config.rules.unusual_running_concentration
        )
        self._fall_rule = PossibleFallRule(self.config.rules.possible_fall)
        self._immobility_rule = ProlongedImmobilityRule(self.config.rules.prolonged_immobility)
        self._persistent_rule = PersistentActivityPatternRule(
            self.config.rules.persistent_patterns
        )
        self._zone_rule = ZoneActivityRule(self.config.rules.zones)
        self._rules: List[RiskRuleProtocol] = [
            self._concentration_rule,
            self._fall_rule,
            self._immobility_rule,
            self._persistent_rule,
            self._zone_rule,
        ]
        if custom_rules:
            self._rules.extend(custom_rules)
        self._total_risk_events_generated: int = 0
        self._session_by_camera: Dict[str, int] = {}

    @property
    def total_risk_events_generated(self) -> int:
        return self._total_risk_events_generated

    @property
    def duplicates_suppressed(self) -> int:
        total = 0
        for r in self._rules:
            total += int(getattr(r, "duplicates_suppressed", 0))
        return total

    @property
    def rules(self) -> List[RiskRuleProtocol]:
        return list(self._rules)

    def register_rule(self, rule: RiskRuleProtocol) -> None:
        """Register an additional custom RiskRuleProtocol implementation at runtime."""
        self._rules.append(rule)

    def notify_camera_session(self, camera_id: str, session_id: int) -> None:
        """Reset transient condition timers if a camera reconnects on a new session_id."""
        prev = self._session_by_camera.get(camera_id)
        self._session_by_camera[camera_id] = session_id
        if prev is not None and prev != session_id:
            self._concentration_rule.reset_camera(camera_id)

    def evaluate(
        self,
        camera_id: str,
        monotonic_ts: float,
        utc_iso: str,
        activities: Sequence[ActivityObservationV1],
        summary: Optional[CameraActivitySummaryV1],
        model_version: str = "heuristic-kinematic-v1.0",
        session_id: Optional[int] = None,
    ) -> List[RiskEventV1]:
        if not self.config.enabled:
            return []

        if session_id is not None:
            self.notify_camera_session(camera_id, session_id)

        emitted: List[RiskEventV1] = []
        for rule in self._rules:
            rule_events = rule.evaluate(
                camera_id=camera_id,
                monotonic_ts=monotonic_ts,
                utc_iso=utc_iso,
                activities=activities,
                summary=summary,
                model_version=model_version,
            )
            if rule_events:
                for ev in rule_events:
                    if ev.lifecycle_state == AlertLifecycleState.TRIGGERED.value:
                        self._total_risk_events_generated += 1
                emitted.extend(rule_events)

        return emitted
