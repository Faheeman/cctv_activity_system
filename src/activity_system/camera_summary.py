"""Camera-level activity distribution and percentage aggregator over configurable sliding time windows."""

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from activity_system.config import SummaryConfig
from activity_system.schemas import (
    ActivityCode,
    ActivityEvent,
    CameraActivitySummaryV1,
    normalize_utc_iso,
)


ALL_ACTIVITY_CODES: Tuple[str, ...] = (
    ActivityCode.STANDING.value,
    ActivityCode.WALKING.value,
    ActivityCode.RUNNING.value,
    ActivityCode.SITTING.value,
    ActivityCode.LYING.value,
    ActivityCode.FALLING.value,
    ActivityCode.UNKNOWN.value,
)


class _WindowObservationSample:
    __slots__ = ("monotonic_ts", "track_id", "activity_code", "quality_score")

    def __init__(
        self,
        monotonic_ts: float,
        track_id: str,
        activity_code: str,
        quality_score: float,
    ) -> None:
        self.monotonic_ts = float(monotonic_ts)
        self.track_id = track_id
        self.activity_code = activity_code
        self.quality_score = float(quality_score)


class CameraSummaryAggregator:
    """Computes deduplicated camera-level activity distributions and percentages over a sliding window.

    Deduplication & Percentage Calculation Semantics:
    - Each observation emitted in `[current_monotonic_ts - window_seconds, current_monotonic_ts]` is buffered.
    - To prevent double-counting the same tracked person across multiple frames in the window, each unique
      `track_id` observed within the window is counted at most ONCE (`observed_people = len(unique_tracks)`),
      using that track's most recent committed activity label in the window.
    - `sample_count` records the total number of individual frame-track observations processed in the window.
    - When `exclude_unknown_from_percentages` is True, `UNKNOWN` tracks are excluded from the percentage
      denominator (`eligible_people = observed_people - unknown_count`). When False, percentages are
      computed over `observed_people`.
    """

    def __init__(self, config: Optional[SummaryConfig] = None) -> None:
        self.config = config or SummaryConfig()
        self._samples_by_camera: Dict[str, Deque[_WindowObservationSample]] = {}
        self._latest_summary_by_camera: Dict[str, CameraActivitySummaryV1] = {}

    def update(
        self,
        camera_id: str,
        monotonic_ts: float,
        utc_iso: str,
        events: Sequence[ActivityEvent],
        window_seconds: Optional[float] = None,
    ) -> CameraActivitySummaryV1:
        """Ingest current-frame track events for camera_id, prune expired window samples, and emit summary."""
        win_sec = float(window_seconds if window_seconds is not None else self.config.window_seconds)
        buf = self._samples_by_camera.setdefault(camera_id, deque())

        for ev in events:
            code = ev.activity.upper_code.value
            q_score = float(getattr(ev, "observation_quality_score", ev.observation_quality.numeric_score))
            buf.append(
                _WindowObservationSample(
                    monotonic_ts=monotonic_ts,
                    track_id=ev.track_id,
                    activity_code=code,
                    quality_score=q_score,
                )
            )

        cutoff = float(monotonic_ts) - win_sec
        while buf and buf[0].monotonic_ts < cutoff:
            buf.popleft()

        # Deduplicate by track_id: keep latest sample per unique track_id in the window
        latest_by_track: Dict[str, _WindowObservationSample] = {}
        for sample in buf:
            latest_by_track[sample.track_id] = sample

        distribution: Dict[str, int] = {code: 0 for code in ALL_ACTIVITY_CODES}
        qualities: List[float] = []
        for sample in latest_by_track.values():
            distribution[sample.activity_code] = distribution.get(sample.activity_code, 0) + 1
            qualities.append(sample.quality_score)

        observed_people = len(latest_by_track)
        unknown_count = distribution.get(ActivityCode.UNKNOWN.value, 0)

        if self.config.exclude_unknown_from_percentages:
            eligible_people = max(0, observed_people - unknown_count)
            unknown_treatment = "exclude_from_percentage_denominator"
            denom = eligible_people
        else:
            eligible_people = observed_people
            unknown_treatment = "include_in_percentage_denominator"
            denom = observed_people

        percentages: Dict[str, float] = {}
        for code in ALL_ACTIVITY_CODES:
            if self.config.exclude_unknown_from_percentages and code == ActivityCode.UNKNOWN.value:
                percentages[code] = 0.0
                continue
            count = distribution.get(code, 0)
            pct = round((float(count) * 100.0) / float(denom), 2) if denom > 0 else 0.0
            percentages[code] = pct

        mean_quality = round(sum(qualities) / len(qualities), 4) if qualities else 1.0

        summary = CameraActivitySummaryV1(
            camera_id=camera_id,
            timestamp=normalize_utc_iso(utc_iso),
            window_seconds=win_sec,
            observed_people=observed_people,
            activity_distribution=distribution,
            activity_percentages=percentages,
            observation_quality=mean_quality,
            sample_count=len(buf),
            eligible_people=eligible_people,
            unknown_count=unknown_count,
            unknown_treatment=unknown_treatment,
            aggregation_method="unique_track_latest_committed_activity",
        )
        self._latest_summary_by_camera[camera_id] = summary
        return summary

    def get_latest(self, camera_id: Optional[str] = None) -> List[CameraActivitySummaryV1]:
        if camera_id is not None:
            s = self._latest_summary_by_camera.get(camera_id)
            return [s] if s is not None else []
        return [
            self._latest_summary_by_camera[cid]
            for cid in sorted(self._latest_summary_by_camera.keys())
        ]

    def reset_camera(self, camera_id: str) -> None:
        self._samples_by_camera.pop(camera_id, None)
        self._latest_summary_by_camera.pop(camera_id, None)
