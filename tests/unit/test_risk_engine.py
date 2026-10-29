"""Comprehensive unit tests for RiskAssessmentEngine covering all 12+ required scenarios."""

from typing import List, Optional, Sequence
import unittest

from activity_system.config import (
    ConcentrationCameraOverrideConfig,
    ConcentrationRuleConfig,
    FallRuleConfig,
    ImmobilityRuleConfig,
    PersistentActivityPatternConfig,
    RiskEngineConfig,
    RiskRulesConfig,
    ZoneDefinitionConfig,
)
from activity_system.risk_engine import RiskAssessmentEngine
from activity_system.schemas import (
    ActivityCode,
    ActivityObservationV1,
    AlertLifecycleState,
    CameraActivitySummaryV1,
    RiskEventV1,
    RiskSeverity,
    RiskStatus,
)


def _make_crowd_observations(
    camera_id: str,
    total_people: int,
    running_people: int,
    other_activity: ActivityCode = ActivityCode.WALKING,
    quality: float = 0.90,
    timestamp: str = "2026-10-27T04:00:00Z",
) -> List[ActivityObservationV1]:
    obs: List[ActivityObservationV1] = []
    for i in range(total_people):
        act = ActivityCode.RUNNING if i < running_people else other_activity
        obs.append(
            ActivityObservationV1(
                camera_id=camera_id,
                track_id=f"{camera_id}_s1_track_{i+1}",
                timestamp=timestamp,
                activity=act,
                observation_quality=quality,
                duration_seconds=3.0,
                bbox=[20 + i * 15, 80, 60 + i * 15, 220],
                pose_available=True,
            )
        )
    return obs


def _make_summary(
    camera_id: str = "camera_01",
    observed_people: int = 12,
    running_pct: float = 58.33,
    running_count: int = 7,
) -> CameraActivitySummaryV1:
    walking_count = max(0, observed_people - running_count)
    walking_pct = round(100.0 - running_pct, 2)
    return CameraActivitySummaryV1(
        camera_id=camera_id,
        timestamp="2026-10-27T04:00:00Z",
        window_seconds=10.0,
        observed_people=observed_people,
        activity_distribution={
            "STANDING": 0,
            "WALKING": walking_count,
            "RUNNING": running_count,
            "SITTING": 0,
            "LYING": 0,
            "FALLING": 0,
            "UNKNOWN": 0,
        },
        activity_percentages={
            "STANDING": 0.0,
            "WALKING": walking_pct,
            "RUNNING": running_pct,
            "SITTING": 0.0,
            "LYING": 0.0,
            "FALLING": 0.0,
            "UNKNOWN": 0.0,
        },
        observation_quality=0.92,
        sample_count=120,
        eligible_people=observed_people,
    )


class CustomTestRule:
    """Example custom rule verifying RiskRuleProtocol extensibility."""

    @property
    def rule_name(self) -> str:
        return "custom_crowd_size_info"

    def evaluate(
        self,
        camera_id: str,
        monotonic_ts: float,
        utc_iso: str,
        activities: Sequence[ActivityObservationV1],
        summary: Optional[CameraActivitySummaryV1],
        model_version: str,
    ) -> List[RiskEventV1]:
        if summary and summary.observed_people >= 50:
            return [
                RiskEventV1(
                    camera_id=camera_id,
                    timestamp=utc_iso,
                    risk_type="HIGH_OCCUPANCY_INFO",
                    rule_name=self.rule_name,
                    severity=RiskSeverity.INFO,
                    observed_people=summary.observed_people,
                    explanation="Custom occupancy threshold reached.",
                    model_version=model_version,
                )
            ]
        return []


class TestRiskEngineUnit(unittest.TestCase):
    def setUp(self) -> None:
        self.default_rule_cfg = ConcentrationRuleConfig(
            enabled=True,
            activity=ActivityCode.RUNNING,
            minimum_people=5,
            percentage_threshold=40.0,
            minimum_duration_seconds=5.0,
            min_observation_quality=0.50,
            clear_grace_seconds=1.5,
            emit_resolved_events=True,
            severity=RiskSeverity.MEDIUM,
            cooldown_seconds=10.0,
        )
        self.engine = RiskAssessmentEngine(
            RiskEngineConfig(
                enabled=True,
                rules=RiskRulesConfig(unusual_running_concentration=self.default_rule_cfg),
            )
        )

    def test_1_normal_walking_does_not_trigger_crowd_running_alert(self) -> None:
        # 12 people walking for 10 seconds -> 0 alerts
        for step in range(12):
            obs = _make_crowd_observations("cam_01", total_people=12, running_people=0)
            res = self.engine.evaluate("cam_01", 100.0 + step * 1.0, "2026-10-27T04:00:00Z", obs, None)
            self.assertEqual(res, [])

    def test_2_single_running_person_does_not_trigger_five_person_rule(self) -> None:
        # 1 person running (100% proportion, but count 1 < 5) for 10 seconds -> 0 alerts
        for step in range(12):
            obs = _make_crowd_observations("cam_01", total_people=1, running_people=1)
            res = self.engine.evaluate("cam_01", 100.0 + step * 1.0, "2026-10-27T04:00:00Z", obs, None)
            self.assertEqual(res, [])

    def test_3_minimum_running_count_is_enforced(self) -> None:
        # 8 people observed, 4 running (50% >= 40% threshold, but running_people=4 < minimum_people=5) -> 0 alerts
        for step in range(10):
            obs = _make_crowd_observations("cam_01", total_people=8, running_people=4)
            res = self.engine.evaluate("cam_01", 100.0 + step * 1.0, "2026-10-27T04:00:00Z", obs, None)
            self.assertEqual(res, [])

    def test_4_minimum_proportion_is_enforced(self) -> None:
        # 20 people observed, 5 running (running_people=5 >= 5, but 25.0% < 40.0% threshold) -> 0 alerts
        for step in range(10):
            obs = _make_crowd_observations("cam_01", total_people=20, running_people=5)
            res = self.engine.evaluate("cam_01", 100.0 + step * 1.0, "2026-10-27T04:00:00Z", obs, None)
            self.assertEqual(res, [])

    def test_5_duration_threshold_and_12_Metadata_and_severity(self) -> None:
        # 12 people observed, 7 running (58.33% >= 40.0%, 7 >= 5)
        obs = _make_crowd_observations("cam_01", total_people=12, running_people=7)
        # t=100.0 (0s), t=102.0 (2s), t=104.5 (4.5s < 5.0s) -> no alert yet
        self.assertEqual(self.engine.evaluate("cam_01", 100.0, "2026-10-27T04:00:00Z", obs, None), [])
        self.assertEqual(self.engine.evaluate("cam_01", 102.0, "2026-10-27T04:00:02Z", obs, None), [])
        self.assertEqual(self.engine.evaluate("cam_01", 104.5, "2026-10-27T04:00:04Z", obs, None), [])

        # At t=105.2 (5.2s >= 5.0s) -> alert triggers!
        alerts = self.engine.evaluate("cam_01", 105.2, "2026-10-27T04:00:05Z", obs, None)
        self.assertEqual(len(alerts), 1)
        ev = alerts[0]
        self.assertEqual(ev.risk_type, "UNUSUAL_ACTIVITY_CONCENTRATION")
        self.assertEqual(ev.rule_name, "unusual_running_concentration")
        self.assertEqual(ev.severity, RiskSeverity.MEDIUM)
        self.assertEqual(ev.status, RiskStatus.OPEN)
        self.assertEqual(ev.lifecycle_state, AlertLifecycleState.TRIGGERED.value)
        self.assertEqual(ev.observed_people, 12)
        self.assertEqual(ev.matching_people, 7)
        self.assertAlmostEqual(ev.measured_percentage or 0.0, 58.33, places=2)
        self.assertAlmostEqual(ev.threshold_percentage or 0.0, 40.0, places=2)
        self.assertAlmostEqual(ev.duration_seconds, 5.2, places=2)
        self.assertEqual(ev.correlation_key, "cam_01:unusual_running_concentration:RUNNING")
        self.assertEqual(ev.title, "Unusual Running Concentration Detected")
        self.assertEqual(len(ev.relevant_track_ids), 7)
        self.assertEqual(len(ev.status_history), 1)

    def test_6_short_lived_spike_does_not_trigger_alert(self) -> None:
        # 2.0-second spike of 7/12 running, followed by normal walking
        obs_run = _make_crowd_observations("cam_01", total_people=12, running_people=7)
        obs_walk = _make_crowd_observations("cam_01", total_people=12, running_people=0)
        self.assertEqual(self.engine.evaluate("cam_01", 100.0, "2026-10-27T04:00:00Z", obs_run, None), [])
        self.assertEqual(self.engine.evaluate("cam_01", 102.0, "2026-10-27T04:00:02Z", obs_run, None), [])
        # Reverts to walking at t=103.0..108.0 -> never triggers
        for dt in (103.0, 104.0, 105.5, 107.0):
            self.assertEqual(self.engine.evaluate("cam_01", dt, "2026-10-27T04:00:05Z", obs_walk, None), [])

    def test_7_poor_quality_and_unknown_observations_do_not_falsely_satisfy_rule(self) -> None:
        # 12 people classified as RUNNING but with poor observation_quality=0.25 (< min_observation_quality=0.50)
        obs_poor = _make_crowd_observations(
            "cam_01", total_people=12, running_people=12, quality=0.25
        )
        for step in range(8):
            self.assertEqual(
                self.engine.evaluate("cam_01", 100.0 + step * 1.0, "2026-10-27T04:00:00Z", obs_poor, None),
                [],
            )

        # 12 people with UNKNOWN activity and high quality -> UNKNOWN is never treated as RUNNING
        obs_unk = _make_crowd_observations(
            "cam_01",
            total_people=12,
            running_people=0,
            other_activity=ActivityCode.UNKNOWN,
            quality=0.92,
        )
        for step in range(8):
            self.assertEqual(
                self.engine.evaluate("cam_01", 200.0 + step * 1.0, "2026-10-27T04:01:00Z", obs_unk, None),
                [],
            )

    def test_8_continuing_condition_suppresses_duplicates_and_9_clears_then_retriggers(self) -> None:
        obs_run = _make_crowd_observations("cam_01", total_people=12, running_people=7)
        obs_walk = _make_crowd_observations("cam_01", total_people=12, running_people=0)

        # Start at t=100.0, trigger at t=105.0
        self.engine.evaluate("cam_01", 100.0, "2026-10-27T04:00:00Z", obs_run, None)
        first_alerts = self.engine.evaluate("cam_01", 105.0, "2026-10-27T04:00:05Z", obs_run, None)
        self.assertEqual(len(first_alerts), 1)
        first_id = first_alerts[0].event_id

        # Continuing frames within cooldown (t=106, 107, 108) -> suppressed!
        for t in (106.0, 107.0, 108.0):
            self.assertEqual(self.engine.evaluate("cam_01", t, "2026-10-27T04:00:08Z", obs_run, None), [])
        self.assertEqual(self.engine.duplicates_suppressed, 3)

        # Continuing past cooldown (t=116.0 >= 105.0 + 10.0s cooldown) -> emits UPDATED on the SAME event_id!
        upd_alerts = self.engine.evaluate("cam_01", 116.0, "2026-10-27T04:00:16Z", obs_run, None)
        self.assertEqual(len(upd_alerts), 1)
        self.assertEqual(upd_alerts[0].event_id, first_id)
        self.assertEqual(upd_alerts[0].lifecycle_state, AlertLifecycleState.UPDATED.value)
        self.assertEqual(upd_alerts[0].occurrence_count, 2)

        # Now condition clears: walking at t=117.0 (within 1.5s grace -> held), then t=119.0 (> 1.5s grace -> RESOLVED)
        self.assertEqual(self.engine.evaluate("cam_01", 117.0, "2026-10-27T04:00:17Z", obs_walk, None), [])
        res_alerts = self.engine.evaluate("cam_01", 119.0, "2026-10-27T04:00:19Z", obs_walk, None)
        self.assertEqual(len(res_alerts), 1)
        self.assertEqual(res_alerts[0].event_id, first_id)
        self.assertEqual(res_alerts[0].status, RiskStatus.RESOLVED)
        self.assertEqual(res_alerts[0].lifecycle_state, AlertLifecycleState.RESOLVED.value)

        # Later recurrence at t=130.0..135.5 -> generates a brand-new alert with a new event_id!
        self.engine.evaluate("cam_01", 130.0, "2026-10-27T04:00:30Z", obs_run, None)
        second_alerts = self.engine.evaluate("cam_01", 135.5, "2026-10-27T04:00:35Z", obs_run, None)
        self.assertEqual(len(second_alerts), 1)
        self.assertNotEqual(second_alerts[0].event_id, first_id)
        self.assertEqual(second_alerts[0].lifecycle_state, AlertLifecycleState.TRIGGERED.value)

    def test_10_independent_cameras_and_camera_specific_overrides(self) -> None:
        cfg = ConcentrationRuleConfig(
            enabled=True,
            activity=ActivityCode.RUNNING,
            minimum_people=5,
            percentage_threshold=40.0,
            minimum_duration_seconds=5.0,
            severity=RiskSeverity.MEDIUM,
            camera_overrides={
                "cam_strict": ConcentrationCameraOverrideConfig(
                    minimum_people=8,
                    percentage_threshold=60.0,
                    severity=RiskSeverity.HIGH,
                )
            },
        )
        engine = RiskAssessmentEngine(
            RiskEngineConfig(enabled=True, rules=RiskRulesConfig(unusual_running_concentration=cfg))
        )

        # Feed 6/10 running (60%) to both cam_default and cam_strict
        obs_default = _make_crowd_observations("cam_default", total_people=10, running_people=6)
        obs_strict = _make_crowd_observations("cam_strict", total_people=10, running_people=6)

        engine.evaluate("cam_default", 10.0, "2026-10-27T04:00:00Z", obs_default, None)
        engine.evaluate("cam_strict", 10.0, "2026-10-27T04:00:00Z", obs_strict, None)

        # At t=15.5s, cam_default triggers (6 >= 5), while cam_strict does NOT trigger (6 < 8 override)
        res_def = engine.evaluate("cam_default", 15.5, "2026-10-27T04:00:05Z", obs_default, None)
        res_str = engine.evaluate("cam_strict", 15.5, "2026-10-27T04:00:05Z", obs_strict, None)
        self.assertEqual(len(res_def), 1)
        self.assertEqual(res_def[0].severity, RiskSeverity.MEDIUM)
        self.assertEqual(res_str, [])

    def test_11_missing_frames_and_camera_reconnections_do_not_create_false_evidence(self) -> None:
        obs_run = _make_crowd_observations("cam_01", total_people=12, running_people=7)

        # 1 frame of running at t=100.0 followed by missing frames (empty obs, summary=None) at t=101..106
        self.engine.evaluate("cam_01", 100.0, "2026-10-27T04:00:00Z", obs_run, None, session_id=1)
        for t in (101.0, 102.0, 105.5, 106.0):
            res = self.engine.evaluate("cam_01", t, "2026-10-27T04:00:06Z", [], None, session_id=1)
            self.assertEqual(res, [])

        # Now test camera reconnection (session_id increments from 1 to 2 after 3s of running):
        # Pre-reconnect 3s duration must not carry over across a stream session reset
        self.engine.evaluate("cam_01", 200.0, "2026-10-27T04:01:00Z", obs_run, None, session_id=1)
        self.engine.evaluate("cam_01", 203.0, "2026-10-27T04:01:03Z", obs_run, None, session_id=1)
        # Reconnect on session_id=2 at t=205.5 -> timer resets on session 2, so 0s elapsed on session 2 (< 5.0s)
        res_reconnect = self.engine.evaluate(
            "cam_01", 205.5, "2026-10-27T04:01:05Z", obs_run, None, session_id=2
        )
        self.assertEqual(res_reconnect, [])

    def test_possible_fall_prolonged_immobility_zone_and_custom_rules(self) -> None:
        cfg = RiskEngineConfig(
            enabled=True,
            rules=RiskRulesConfig(
                possible_fall=FallRuleConfig(
                    enabled=True,
                    minimum_duration_seconds=0.4,
                    min_observation_quality=0.6,
                    severity=RiskSeverity.HIGH,
                ),
                prolonged_immobility=ImmobilityRuleConfig(
                    enabled=True,
                    activity=ActivityCode.LYING,
                    immobility_duration_seconds=10.0,
                    min_observation_quality=0.6,
                    severity=RiskSeverity.CRITICAL,
                ),
                persistent_patterns=[
                    PersistentActivityPatternConfig(
                        pattern_id="corridor_sprint",
                        camera_id="camera_01",
                        enabled=True,
                        activity=ActivityCode.RUNNING,
                        minimum_people=1,
                        minimum_duration_seconds=0.0,
                        severity=RiskSeverity.MEDIUM,
                    )
                ],
                zones=[
                    ZoneDefinitionConfig(
                        zone_id="zone_01",
                        camera_id="camera_01",
                        name="Loading Bay",
                        enabled=True,
                        polygon=[(100.0, 100.0), (400.0, 100.0), (400.0, 400.0), (100.0, 400.0)],
                        trigger_activities=[ActivityCode.RUNNING],
                        minimum_duration_seconds=0.0,
                        severity=RiskSeverity.LOW,
                    )
                ],
            ),
        )
        engine = RiskAssessmentEngine(cfg)
        engine.register_rule(CustomTestRule())

        obs_list = [
            ActivityObservationV1(
                camera_id="camera_01",
                track_id="track_fall_1",
                timestamp="2026-10-26T10:30:00Z",
                activity=ActivityCode.FALLING,
                observation_quality=0.88,
                duration_seconds=0.6,
                bbox=[150, 200, 260, 360],
                pose_available=True,
            ),
            ActivityObservationV1(
                camera_id="camera_01",
                track_id="track_lying_2",
                timestamp="2026-10-26T10:30:00Z",
                activity=ActivityCode.LYING,
                observation_quality=0.90,
                duration_seconds=12.5,
                bbox=[20, 20, 90, 60],
                pose_available=True,
            ),
            ActivityObservationV1(
                camera_id="camera_01",
                track_id="track_zone_3",
                timestamp="2026-10-26T10:30:00Z",
                activity=ActivityCode.RUNNING,
                observation_quality=0.91,
                duration_seconds=2.0,
                bbox=[200, 180, 250, 320],
                pose_available=True,
            ),
        ]
        summary = _make_summary(observed_people=55, running_pct=20.0, running_count=11)
        risk_events = engine.evaluate(
            "camera_01", 200.0, "2026-10-26T10:30:00Z", obs_list, summary
        )
        risk_types = {e.risk_type for e in risk_events}
        self.assertIn("POSSIBLE_FALL_DETECTED", risk_types)
        self.assertIn("PROLONGED_IMMOBILITY", risk_types)
        self.assertIn("PERSISTENT_UNUSUAL_ACTIVITY", risk_types)
        self.assertIn("ZONE_RESTRICTED_ACTIVITY", risk_types)
        self.assertIn("HIGH_OCCUPANCY_INFO", risk_types)


if __name__ == "__main__":
    unittest.main()
