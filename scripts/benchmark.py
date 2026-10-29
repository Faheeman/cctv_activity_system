"""Reproducible sequence-level activity evaluation and hardware performance benchmark suite."""

import json
from pathlib import Path
import sys
import tempfile
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from activity_system.activity_classifier import ActivityClassifier
from activity_system.config import (
    ActivityConfig,
    AppConfig,
    CameraConfig,
    PoseConfig,
    TrackerConfig,
)
from activity_system.main import ActivityApplication
from activity_system.person_tracker import PersonTracker
from activity_system.schemas import ActivityLabel, SourceType
from tests.fixtures.activity_sequences import build_evaluation_dataset
from tests.fixtures.video_generator import generate_synthetic_cctv_video


def run_activity_evaluation() -> Dict:
    """Evaluate ActivityClassifier across session-split sequences without frame leakage."""
    sequences = build_evaluation_dataset()
    classifier = ActivityClassifier(ActivityConfig())
    tracker_cfg = TrackerConfig(min_hits_to_confirm=2)

    labels = [
        ActivityLabel.STANDING.value,
        ActivityLabel.WALKING.value,
        ActivityLabel.RUNNING.value,
        ActivityLabel.UNKNOWN.value,
    ]
    confusion: Dict[str, Dict[str, int]] = {
        true_lbl: {pred_lbl: 0 for pred_lbl in labels} for true_lbl in labels
    }

    total_frames_all = 0
    unknown_frames_all = 0
    clear_post_warmup_frames = 0
    clear_post_warmup_unknown = 0
    total_false_transitions = 0
    total_track_minutes = 0.0
    stabilization_times_sec: List[float] = []
    per_sequence_summary: List[Dict] = []

    for seq in sequences:
        tracker = PersonTracker(tracker_cfg)
        start_ts = seq.frames[0][0].monotonic_ts
        end_ts = seq.frames[-1][0].monotonic_ts
        seq_duration_min = max(1e-6, (end_ts - start_ts) / 60.0)
        total_track_minutes += seq_duration_min

        first_stable_sec = None
        seq_correct_post_warmup = 0
        seq_total_post_warmup = 0

        for idx, (pkt, dets) in enumerate(seq.frames):
            _, tracks = tracker.update(pkt, dets)
            ev = classifier.classify_track(tracks[0])
            total_frames_all += 1
            if ev.activity == ActivityLabel.UNKNOWN:
                unknown_frames_all += 1

            if ev.activity == seq.expected_activity and first_stable_sec is None and idx >= 6:
                first_stable_sec = pkt.monotonic_ts - start_ts

            # Evaluate steady-state window after initial 9 frames (0.60s warm-up)
            if idx >= 9:
                seq_total_post_warmup += 1
                true_key = seq.expected_activity.value
                pred_key = ev.activity.value
                confusion[true_key][pred_key] += 1
                if ev.activity == seq.expected_activity:
                    seq_correct_post_warmup += 1
                if seq.expected_activity != ActivityLabel.UNKNOWN:
                    clear_post_warmup_frames += 1
                    if ev.activity == ActivityLabel.UNKNOWN:
                        clear_post_warmup_unknown += 1

        transitions = tracks[0].transition_count
        total_false_transitions += transitions
        if seq.expected_activity != ActivityLabel.UNKNOWN and first_stable_sec is not None:
            stabilization_times_sec.append(round(first_stable_sec, 3))

        per_sequence_summary.append(
            {
                "sequence_id": seq.sequence_id,
                "expected": seq.expected_activity.value,
                "final_predicted": ev.activity.value,
                "post_warmup_accuracy": round(seq_correct_post_warmup / max(1, seq_total_post_warmup), 4),
                "false_transitions": transitions,
                "time_to_stable_sec": round(first_stable_sec, 3) if first_stable_sec is not None else None,
            }
        )

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    f1_scores: List[float] = []
    for lbl in labels:
        tp = confusion[lbl][lbl]
        fp = sum(confusion[other][lbl] for other in labels if other != lbl)
        fn = sum(confusion[lbl][other] for other in labels if other != lbl)
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_metrics[lbl] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
        f1_scores.append(f1)

    return {
        "per_class": per_class_metrics,
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "confusion_matrix": confusion,
        "false_transitions_per_track_minute": round(total_false_transitions / total_track_minutes, 4),
        "mean_time_to_stable_sec": round(
            sum(stabilization_times_sec) / max(1, len(stabilization_times_sec)), 3
        ),
        "max_time_to_stable_sec": max(stabilization_times_sec) if stabilization_times_sec else 0.0,
        "overall_unknown_rate_pct": round(100.0 * unknown_frames_all / max(1, total_frames_all), 2),
        "post_warmup_clear_unknown_rate_pct": round(
            100.0 * clear_post_warmup_unknown / max(1, clear_post_warmup_frames), 2
        ),
        "sequences": per_sequence_summary,
    }


def run_hardware_benchmarks() -> List[Dict]:
    """Benchmark throughput, inference latency, end-to-end latency, CPU%, and RSS memory across configurations."""
    results: List[Dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = Path(tmpdir) / "bench_cctv.avi"
        generate_synthetic_cctv_video(vid_path, num_frames=60, width=640, height=480, fps=15.0)

        configs = [
            ("1cam_640x480_sample1", 1, 640, 480, 1),
            ("1cam_640x480_sample2", 1, 640, 480, 2),
            ("1cam_480x360_sample1", 1, 480, 360, 1),
            ("2cam_640x480_sample1", 2, 640, 480, 1),
        ]

        for name, num_cams, w, h, sample_n in configs:
            cams = [
                CameraConfig(
                    camera_id=f"bench_cam_{i+1:02d}",
                    source_type=SourceType.FILE,
                    source_uri=str(vid_path),
                    sample_every_n_frames=sample_n,
                    resize_width=w,
                    resize_height=h,
                )
                for i in range(num_cams)
            ]
            app_cfg = AppConfig(
                cameras=cams,
                pose=PoseConfig(backend="synthetic", imgsz=w),
            )
            app = ActivityApplication(app_cfg)
            snap = app.run(max_frames_total=120)
            m = snap.metrics
            results.append(
                {
                    "profile": name,
                    "cameras": num_cams,
                    "resolution": f"{w}x{h}",
                    "sample_every_n_frames": sample_n,
                    "frames_received": m.frames_received,
                    "frames_processed": m.frames_processed,
                    "frames_dropped": m.frames_dropped,
                    "mean_inference_ms": m.mean_inference_ms,
                    "mean_end_to_end_ms": m.mean_end_to_end_ms,
                    "throughput_fps": m.throughput_fps,
                    "cpu_utilization_pct": m.cpu_utilization_pct,
                    "memory_rss_mb": m.memory_rss_mb,
                }
            )
    return results


def main() -> int:
    report = {
        "activity_evaluation": run_activity_evaluation(),
        "hardware_benchmarks": run_hardware_benchmarks(),
    }
    out_path = PROJECT_ROOT / "outputs" / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
