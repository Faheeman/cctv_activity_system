"""Scale-normalized temporal feature extraction and multi-frame activity classification with hysteresis."""

import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from activity_system.config import ActivityConfig
from activity_system.schemas import (
    ActivityEvent,
    ActivityLabel,
    ObservationQuality,
    TrackLifecycle,
)
from activity_system.supervised_classifier import SupervisedActivityClassifier
from activity_system.track_state import TrackObservation, TrackRecord


class ActivityClassifier:
    """Extracts body-scale-normalized kinematics, posture, and gait features to classify human activity."""

    def __init__(self, config: ActivityConfig) -> None:
        self.config = config
        self._supervised: Optional[SupervisedActivityClassifier] = (
            SupervisedActivityClassifier()
            if getattr(config, "classifier_type", "heuristic") == "supervised"
            else None
        )
        self._total_classifications = 0
        self._total_classification_sec = 0.0

    @property
    def mean_classification_ms(self) -> float:
        if self._total_classifications == 0:
            return 0.0
        return (self._total_classification_sec / self._total_classifications) * 1000.0

    def classify_track(self, track: TrackRecord) -> ActivityEvent:
        """Compute temporal features for a TrackRecord, update hysteresis state, and emit ActivityEvent."""
        t0 = time.perf_counter()
        obs_list = list(track.observations)
        features = self.extract_features(obs_list)
        track.recent_features = features

        raw_label, evidence_score, overall_quality = self._evaluate_rules(track, obs_list, features)
        committed_label = self._apply_hysteresis(track, raw_label, evidence_score)

        model_confidence: Optional[float] = None
        if self._supervised is not None and committed_label != ActivityLabel.UNKNOWN:
            _, sup_prob, _ = self._supervised.predict_proba(features)
            model_confidence = sup_prob

        pose_available = bool(
            obs_list and any(kp.visible for kp in obs_list[-1].keypoints)
        )
        lower_vis = features.get("lower_body_visible_ratio", 0.0)
        quality_score = round(float(np.clip(0.30 + 0.65 * lower_vis, 0.20, 0.98)), 4)
        duration_sec = round(track.current_activity_duration_seconds, 3)

        self._total_classifications += 1
        self._total_classification_sec += time.perf_counter() - t0

        return ActivityEvent(
            camera_id=track.camera_id,
            track_id=track.track_id,
            timestamp=track.last_seen_utc_iso,
            activity=committed_label,
            confidence=model_confidence,
            evidence_score=round(float(evidence_score), 4),
            observation_quality=overall_quality,
            observation_quality_score=quality_score,
            duration_seconds=duration_sec,
            bbox=list(track.latest_bbox),
            pose_available=pose_available,
            model_version=getattr(self.config, "model_version", "heuristic-kinematic-v1.0"),
            metadata={
                "activity_code": committed_label.upper_code.value,
                "speed_bh_per_sec": round(features.get("speed_bh_per_sec", 0.0), 3),
                "ankle_std_bh": round(features.get("ankle_std_bh", 0.0), 4),
                "stride_amp_bh": round(features.get("stride_amp_bh", 0.0), 4),
                "cadence_hz": round(features.get("cadence_hz", 0.0), 2),
                "aspect_ratio_median": round(features.get("aspect_ratio_median", 0.0), 3),
                "downward_velocity_bh_per_sec": round(
                    features.get("downward_velocity_bh_per_sec", 0.0), 3
                ),
                "height_drop_ratio": round(features.get("height_drop_ratio", 0.0), 3),
                "window_duration_sec": round(features.get("window_duration_sec", 0.0), 3),
                "observations": len(obs_list),
            },
        )

    def extract_features(self, observations: Sequence[TrackObservation]) -> Dict[str, float]:
        """Compute scale-normalized kinematic, posture, and lower-body gait features over real elapsed time."""
        n_obs = len(observations)
        empty_feats = {
            "n_observations": float(n_obs),
            "window_duration_sec": 0.0,
            "body_height_px": 0.0,
            "speed_bh_per_sec": 0.0,
            "ankle_std_bh": 0.0,
            "knee_std_bh": 0.0,
            "stride_amp_bh": 0.0,
            "cadence_hz": 0.0,
            "lower_body_visible_ratio": 0.0,
            "aspect_ratio_median": 0.0,
            "aspect_ratio_start": 0.0,
            "aspect_ratio_latest": 0.0,
            "hip_knee_vertical_ratio": 0.40,
            "torso_horizontal_ratio": 0.0,
            "downward_velocity_bh_per_sec": 0.0,
            "height_drop_ratio": 0.0,
        }
        if n_obs < 2:
            return empty_feats

        timestamps = np.array([o.monotonic_ts for o in observations], dtype=np.float64)
        duration = float(timestamps[-1] - timestamps[0])
        if duration <= 1e-4:
            return empty_feats

        scale_samples: List[float] = []
        centers_x: List[float] = []
        centers_y: List[float] = []
        upper_y_samples: List[float] = []
        bbox_heights: List[float] = []
        aspect_ratios: List[float] = []
        hip_knee_ratios: List[float] = []
        torso_horiz_ratios: List[float] = []
        ankle_rel_y: List[float] = []
        knee_rel_y: List[float] = []
        ankle_sep_x: List[float] = []
        lower_visible_count = 0

        for obs in observations:
            x1, y1, x2, y2 = obs.bbox
            bbox_w = max(6.0, float(x2 - x1))
            bbox_h = max(12.0, float(y2 - y1))
            bbox_max_dim = max(bbox_w, bbox_h)
            bbox_heights.append(bbox_h)
            aspect_ratios.append(bbox_w / bbox_h)
            kpts = obs.keypoints

            if len(kpts) >= 17 and kpts[5].visible and kpts[6].visible and kpts[11].visible and kpts[12].visible:
                sh_x = 0.5 * (kpts[5].x + kpts[6].x)
                sh_y = 0.5 * (kpts[5].y + kpts[6].y)
                hp_x = 0.5 * (kpts[11].x + kpts[12].x)
                hp_y = 0.5 * (kpts[11].y + kpts[12].y)
                dx_torso = abs(hp_x - sh_x)
                dy_torso = abs(hp_y - sh_y)
                torso_len = float(np.hypot(dx_torso, dy_torso))
                torso_horiz_ratios.append(dx_torso / max(1.0, dy_torso))
                if torso_len >= 8.0:
                    scale_samples.append(torso_len * 3.12)
                else:
                    scale_samples.append(bbox_max_dim)
                centers_x.append(hp_x)
                centers_y.append(hp_y)
                upper_y_samples.append(sh_y)
            else:
                scale_samples.append(bbox_max_dim)
                centers_x.append(0.5 * (x1 + x2))
                centers_y.append(0.5 * (y1 + y2))
                upper_y_samples.append(float(y1) + 0.25 * bbox_h)

            if len(kpts) >= 17:
                ankles_vis = [kpts[15].visible, kpts[16].visible]
                knees_vis = [kpts[13].visible, kpts[14].visible]
                if any(ankles_vis) and any(knees_vis):
                    lower_visible_count += 1
                if kpts[15].visible and kpts[16].visible:
                    ankle_sep_x.append(float(kpts[15].x - kpts[16].x))
                    ankle_rel_y.append(float(kpts[15].y - kpts[16].y))
                if kpts[13].visible and kpts[14].visible:
                    knee_rel_y.append(float(kpts[13].y - kpts[14].y))
                    if kpts[11].visible and kpts[12].visible:
                        hp_y = 0.5 * (kpts[11].y + kpts[12].y)
                        kn_y = 0.5 * (kpts[13].y + kpts[14].y)
                        hip_knee_ratios.append(abs(kn_y - hp_y) / max(20.0, bbox_max_dim))

        body_height = float(np.median(scale_samples)) if scale_samples else 100.0
        body_height = max(20.0, body_height)

        cx_arr = np.asarray(centers_x, dtype=np.float64)
        cy_arr = np.asarray(centers_y, dtype=np.float64)
        if len(cx_arr) >= 3:
            kernel = np.array([0.25, 0.50, 0.25], dtype=np.float64)
            cx_smooth = np.convolve(cx_arr, kernel, mode="valid")
            cy_smooth = np.convolve(cy_arr, kernel, mode="valid")
            ts_smooth = timestamps[1:-1]
        else:
            cx_smooth = cx_arr
            cy_smooth = cy_arr
            ts_smooth = timestamps

        foreshortening = (
            self.config.cctv_foreshortening_factor
            if self.config.camera_viewpoint in ("cctv", "overhead")
            else 1.0
        )

        dx_net = float(cx_smooth[-1] - cx_smooth[0])
        dy_net = float(cy_smooth[-1] - cy_smooth[0]) * foreshortening
        eff_duration = (
            max(1e-3, float(ts_smooth[-1] - ts_smooth[0])) if len(ts_smooth) >= 2 else duration
        )
        net_disp_bh = float(np.hypot(dx_net, dy_net)) / body_height
        speed_bh_per_sec = net_disp_bh / eff_duration

        ankle_std_bh = (
            float(np.std(ankle_rel_y)) / body_height if len(ankle_rel_y) >= 4 else 0.0
        )
        knee_std_bh = (
            float(np.std(knee_rel_y)) / body_height if len(knee_rel_y) >= 4 else 0.0
        )

        stride_amp_bh = 0.0
        cadence_hz = 0.0
        if len(ankle_sep_x) >= 4:
            sep_arr = np.asarray(ankle_sep_x, dtype=np.float64) / body_height
            centered = sep_arr - np.mean(sep_arr)
            stride_amp_bh = float(
                0.5 * (np.percentile(centered, 90) - np.percentile(centered, 10))
            )
            zero_crossings = int(np.sum(np.diff(np.signbit(centered))))
            cadence_hz = float(zero_crossings) / (2.0 * duration)

        lower_body_visible_ratio = float(lower_visible_count) / float(n_obs)

        k_edge = min(3, n_obs)
        aspect_ratio_median = float(np.median(aspect_ratios))
        aspect_ratio_start = float(np.mean(aspect_ratios[:k_edge]))
        aspect_ratio_latest = float(np.mean(aspect_ratios[-k_edge:]))

        h_start = float(np.mean(bbox_heights[:k_edge]))
        h_end = float(np.mean(bbox_heights[-k_edge:]))
        height_drop_ratio = (h_start - h_end) / max(12.0, h_start)

        upper_y_start = float(np.mean(upper_y_samples[:k_edge]))
        upper_y_end = float(np.mean(upper_y_samples[-k_edge:]))
        downward_velocity_bh_per_sec = (upper_y_end - upper_y_start) / (body_height * duration)

        hip_knee_vertical_ratio = (
            float(np.median(hip_knee_ratios)) if hip_knee_ratios else 0.40
        )
        torso_horizontal_ratio = (
            float(np.median(torso_horiz_ratios)) if torso_horiz_ratios else 0.0
        )

        return {
            "n_observations": float(n_obs),
            "window_duration_sec": duration,
            "body_height_px": body_height,
            "speed_bh_per_sec": speed_bh_per_sec,
            "ankle_std_bh": ankle_std_bh,
            "knee_std_bh": knee_std_bh,
            "stride_amp_bh": stride_amp_bh,
            "cadence_hz": cadence_hz,
            "lower_body_visible_ratio": lower_body_visible_ratio,
            "aspect_ratio_median": aspect_ratio_median,
            "aspect_ratio_start": aspect_ratio_start,
            "aspect_ratio_latest": aspect_ratio_latest,
            "hip_knee_vertical_ratio": hip_knee_vertical_ratio,
            "torso_horizontal_ratio": torso_horizontal_ratio,
            "downward_velocity_bh_per_sec": downward_velocity_bh_per_sec,
            "height_drop_ratio": height_drop_ratio,
        }

    def _evaluate_rules(
        self,
        track: TrackRecord,
        observations: Sequence[TrackObservation],
        features: Dict[str, float],
    ) -> Tuple[ActivityLabel, float, ObservationQuality]:
        n_obs = len(observations)
        duration = features["window_duration_sec"]
        lower_vis = features["lower_body_visible_ratio"]
        speed = features["speed_bh_per_sec"]
        ankle_std = features["ankle_std_bh"]
        stride_amp = features["stride_amp_bh"]
        cadence = features["cadence_hz"]
        ar_med = features["aspect_ratio_median"]
        ar_start = features["aspect_ratio_start"]
        ar_latest = features["aspect_ratio_latest"]
        hip_knee_ratio = features["hip_knee_vertical_ratio"]
        torso_horiz = features["torso_horizontal_ratio"]
        down_vel = features["downward_velocity_bh_per_sec"]
        h_drop = features["height_drop_ratio"]

        if lower_vis >= 0.75:
            quality = ObservationQuality.GOOD
        elif lower_vis >= 0.40:
            quality = ObservationQuality.PARTIAL
        else:
            quality = ObservationQuality.POOR

        # Guard 1: Track must be confirmed and have minimum observations & window duration
        if (
            track.lifecycle == TrackLifecycle.TENTATIVE
            or n_obs < self.config.min_observations
            or duration < self.config.min_window_duration_seconds
        ):
            warm_score = min(
                0.35, float(n_obs) / float(max(1, self.config.min_observations)) * 0.35
            )
            return ActivityLabel.UNKNOWN, warm_score, quality

        # Guard 2: Insufficient lower-body visibility -> return UNKNOWN rather than guessing
        if lower_vis < 0.40:
            return ActivityLabel.UNKNOWN, 0.25, ObservationQuality.POOR

        vis_weight = 0.7 + 0.3 * min(1.0, lower_vis)

        # Rule 0a: FALLING (Requires multi-frame transition from upright to collapsed/horizontal with rapid downward velocity)
        if (
            ar_start < 0.95
            and (ar_latest >= 0.95 or h_drop >= self.config.fall_min_height_drop_ratio)
            and down_vel >= self.config.fall_min_downward_velocity_bh_per_sec
            and duration <= self.config.fall_max_transition_seconds
        ):
            fall_factor = min(
                1.0,
                0.70
                + 0.15 * (down_vel / max(0.1, self.config.fall_min_downward_velocity_bh_per_sec))
                + 0.15 * max(0.0, h_drop),
            )
            score = float(np.clip(fall_factor * vis_weight, 0.60, 0.98))
            return ActivityLabel.FALLING, score, quality

        # Rule 0b: LYING (Sustained horizontal posture without active rapid downward collapse)
        is_horizontal = (
            ar_med >= self.config.lying_min_aspect_ratio or torso_horiz >= 1.35
        )
        if (
            is_horizontal
            and duration >= self.config.lying_min_duration_seconds
            and speed <= 0.45
            and abs(down_vel) < self.config.fall_min_downward_velocity_bh_per_sec
        ):
            score = float(np.clip(0.85 * vis_weight, 0.55, 0.97))
            return ActivityLabel.LYING, score, quality

        # Rule 0c: SITTING (Stationary compact posture with bent knees near hip level and upright torso)
        is_seated_geometry = (
            0.0 < hip_knee_ratio <= self.config.sitting_max_hip_knee_ratio
            and self.config.sitting_min_aspect_ratio <= ar_med < self.config.lying_min_aspect_ratio
            and torso_horiz < 0.85
        )
        if (
            duration >= self.config.standing_min_duration_seconds
            and speed <= self.config.standing_max_speed_bh_per_sec
            and ankle_std <= self.config.standing_max_ankle_std_bh
            and is_seated_geometry
        ):
            score = float(np.clip(0.84 * vis_weight, 0.55, 0.96))
            return ActivityLabel.SITTING, score, quality

        # Rule 1: Standing still
        if (
            duration >= self.config.standing_min_duration_seconds
            and speed <= self.config.standing_max_speed_bh_per_sec
            and ankle_std <= self.config.standing_max_ankle_std_bh
            and stride_amp <= 0.06
        ):
            stillness_margin = 1.0 - (
                speed / max(1e-3, self.config.standing_max_speed_bh_per_sec)
            )
            score = float(np.clip((0.70 + 0.30 * stillness_margin) * vis_weight, 0.50, 0.98))
            return ActivityLabel.STANDING, score, quality

        # Rule 2: Running (high normalized speed + significant lower-limb oscillation/stride)
        has_running_gait = (
            ankle_std >= self.config.running_min_ankle_std_bh or stride_amp >= 0.14
        )
        if speed >= self.config.running_min_speed_bh_per_sec and has_running_gait:
            speed_factor = min(
                1.0, (speed - self.config.running_min_speed_bh_per_sec) / 1.5 + 0.75
            )
            score = float(np.clip(speed_factor * vis_weight, 0.55, 0.98))
            return ActivityLabel.RUNNING, score, quality

        # Rule 3: Walking (moderate sustained speed + compatible gait oscillation/stride)
        has_walking_gait = (
            ankle_std >= self.config.walking_min_ankle_std_bh or stride_amp >= 0.045
        )
        if (
            self.config.walking_min_speed_bh_per_sec
            <= speed
            < self.config.running_min_speed_bh_per_sec
            and has_walking_gait
        ):
            mid = 0.5 * (
                self.config.walking_min_speed_bh_per_sec
                + self.config.running_min_speed_bh_per_sec
            )
            closeness = 1.0 - min(0.4, abs(speed - mid) / max(0.5, mid))
            score = float(np.clip((0.65 + 0.30 * closeness) * vis_weight, 0.55, 0.96))
            return ActivityLabel.WALKING, score, quality

        if (
            speed >= self.config.running_min_speed_bh_per_sec
            and has_walking_gait
            and cadence >= 1.2
        ):
            return ActivityLabel.RUNNING, float(np.clip(0.68 * vis_weight, 0.50, 0.90)), quality

        return ActivityLabel.UNKNOWN, 0.30, quality

    def _apply_hysteresis(
        self,
        track: TrackRecord,
        raw_label: ActivityLabel,
        evidence_score: float,
    ) -> ActivityLabel:
        track.evidence_score = evidence_score

        if raw_label == track.current_activity:
            track.candidate_activity = raw_label
            track.candidate_streak = 0
            if raw_label != ActivityLabel.UNKNOWN:
                track.last_reliable_classification_ts = track.last_seen_utc_iso
            return track.current_activity

        if raw_label == track.candidate_activity:
            track.candidate_streak += 1
        else:
            track.candidate_activity = raw_label
            track.candidate_streak = 1

        required_streak = (
            min(2, self.config.hysteresis_count)
            if (
                (
                    track.current_activity == ActivityLabel.UNKNOWN
                    and raw_label != ActivityLabel.UNKNOWN
                )
                or raw_label == ActivityLabel.FALLING
            )
            else self.config.hysteresis_count
        )

        if track.candidate_streak >= required_streak:
            prev_label = track.current_activity
            track.current_activity = raw_label
            track.activity_started_monotonic_ts = track.last_seen_monotonic_ts
            track.candidate_streak = 0
            if raw_label != ActivityLabel.UNKNOWN:
                track.last_reliable_classification_ts = track.last_seen_utc_iso
            if prev_label != ActivityLabel.UNKNOWN and raw_label != prev_label:
                track.transition_count += 1

        return track.current_activity
