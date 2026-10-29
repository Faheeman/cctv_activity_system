"""Sequence-split evaluation integration tests for ActivityClassifier."""

import unittest
from activity_system.activity_classifier import ActivityClassifier
from activity_system.config import ActivityConfig, TrackerConfig
from activity_system.person_tracker import PersonTracker
from activity_system.schemas import ActivityLabel
from tests.fixtures.activity_sequences import build_evaluation_dataset


class TestActivityEvaluationIntegration(unittest.TestCase):
    def test_sequence_level_evaluation_metrics(self) -> None:
        sequences = build_evaluation_dataset()
        classifier = ActivityClassifier(ActivityConfig())
        tracker_cfg = TrackerConfig(min_hits_to_confirm=2)

        correct_post_warmup = 0
        total_post_warmup = 0
        max_false_transitions = 0
        max_time_to_stable_sec = 0.0

        for seq in sequences:
            tracker = PersonTracker(tracker_cfg)
            start_ts = seq.frames[0][0].monotonic_ts
            first_stable_ts = None

            for idx, (pkt, dets) in enumerate(seq.frames):
                _, tracks = tracker.update(pkt, dets)
                self.assertEqual(len(tracks), 1)
                ev = classifier.classify_track(tracks[0])

                if ev.activity == seq.expected_activity and first_stable_ts is None and idx >= 7:
                    first_stable_ts = pkt.monotonic_ts - start_ts

                # Evaluate steady-state frames after 0.65s warm-up window (frame index >= 10)
                if idx >= 10:
                    total_post_warmup += 1
                    if ev.activity == seq.expected_activity:
                        correct_post_warmup += 1

            self.assertEqual(
                ev.activity,
                seq.expected_activity,
                f"Sequence {seq.sequence_id} expected {seq.expected_activity}, got {ev.activity}",
            )
            max_false_transitions = max(max_false_transitions, tracks[0].transition_count)
            if seq.expected_activity != ActivityLabel.UNKNOWN:
                self.assertIsNotNone(first_stable_ts)
                assert first_stable_ts is not None
                max_time_to_stable_sec = max(max_time_to_stable_sec, first_stable_ts)

        accuracy = correct_post_warmup / float(total_post_warmup)
        self.assertGreaterEqual(accuracy, 0.95)
        self.assertEqual(max_false_transitions, 0)
        self.assertLessEqual(max_time_to_stable_sec, 1.0)


if __name__ == "__main__":
    unittest.main()
