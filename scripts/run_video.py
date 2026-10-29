"""CLI runner to process a video stream or file end-to-end and write annotated output."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from activity_system.config import ConfigLoader
from activity_system.main import ActivityApplication
from tests.fixtures.video_generator import generate_synthetic_cctv_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CCTV Pose & Activity Recognition Pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "annotated_cctv.avi",
        help="Path to save annotated output video",
    )
    parser.add_argument(
        "--output-events",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "events.jsonl",
        help="Path to save structured ActivityEvent JSONL log",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=45,
        help="Maximum frames to process",
    )
    args = parser.parse_args()

    cfg = ConfigLoader.load(args.config)
    for cam in cfg.cameras:
        if cam.source_type.value == "file":
            uri_path = Path(str(cam.source_uri))
            if not uri_path.is_absolute():
                uri_path = PROJECT_ROOT / uri_path
            if not uri_path.exists():
                generate_synthetic_cctv_video(uri_path, num_frames=args.max_frames)
            cam.source_uri = str(uri_path)

    app = ActivityApplication(cfg)
    final_snapshot = app.run(
        max_frames_total=args.max_frames,
        output_video_path=args.output_video,
        events_jsonl_path=args.output_events,
    )
    print(json.dumps(final_snapshot.metrics.dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
