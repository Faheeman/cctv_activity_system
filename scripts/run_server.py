"""Production service entrypoint running the video processing pipeline and FastAPI server concurrently."""

import argparse
import os
from pathlib import Path
import signal
import sys
import threading
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from activity_system.api import create_api_app
from activity_system.config import CameraConfig, ConfigLoader
from activity_system.main import ActivityApplication
from activity_system.schemas import SourceType
from tests.fixtures.video_generator import generate_synthetic_cctv_video


def _apply_live_camera_overrides(
    cfg,
    source_type_arg: str | None,
    source_uri_arg: str | None,
    camera_id_arg: str | None,
    pose_backend_arg: str | None,
) -> None:
    """Apply CLI or environment variable overrides for instant live camera streaming."""
    env_type = source_type_arg or os.environ.get("CAMERA_SOURCE_TYPE")
    env_uri = source_uri_arg or os.environ.get("CAMERA_SOURCE_URI")
    env_cam_id = camera_id_arg or os.environ.get("CAMERA_ID")
    env_backend = pose_backend_arg or os.environ.get("POSE_BACKEND")

    if env_backend:
        cfg.pose.backend = env_backend.strip().lower()

    if env_type or env_uri:
        base_cam = cfg.cameras[0]
        resolved_type_str = (env_type or base_cam.source_type.value).strip().lower()
        resolved_uri = env_uri if env_uri is not None else base_cam.source_uri
        if not env_type and isinstance(resolved_uri, str):
            u = resolved_uri.strip()
            if u.isdigit() or u.startswith("/dev/video"):
                resolved_type_str = "webcam"
            elif u.startswith(("rtsp://", "rtsps://", "http://", "https://")):
                resolved_type_str = "rtsp"
            else:
                resolved_type_str = "file"

        resolved_type = SourceType(resolved_type_str)
        override_cam = CameraConfig(
            camera_id=env_cam_id or base_cam.camera_id,
            source_type=resolved_type,
            source_uri=resolved_uri,
            enabled=True,
            loop_file=(resolved_type == SourceType.FILE),
            sample_every_n_frames=base_cam.sample_every_n_frames,
            target_fps=base_cam.target_fps,
            max_frame_queue_size=base_cam.max_frame_queue_size,
            resize_width=base_cam.resize_width,
            resize_height=base_cam.resize_height,
            connection_timeout_ms=base_cam.connection_timeout_ms,
            read_timeout_ms=base_cam.read_timeout_ms,
            reconnect_initial_delay_sec=base_cam.reconnect_initial_delay_sec,
            reconnect_max_delay_sec=base_cam.reconnect_max_delay_sec,
            reconnect_backoff_factor=base_cam.reconnect_backoff_factor,
            max_reconnect_attempts=50 if resolved_type != SourceType.FILE else base_cam.max_reconnect_attempts,
        )
        cfg.cameras = [override_cam]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CCTV Activity Recognition Service + FastAPI Server")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        choices=["webcam", "rtsp", "file"],
        default=None,
        help="Override camera source_type (webcam, rtsp, file) or set CAMERA_SOURCE_TYPE env var",
    )
    parser.add_argument(
        "--source-uri",
        type=str,
        default=None,
        help="Override camera source_uri (e.g. '0', '/dev/video0', 'rtsp://user:pass@ip:554/stream') or set CAMERA_SOURCE_URI env var",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=None,
        help="Override camera_id when using --source-uri or CAMERA_ID env var",
    )
    parser.add_argument(
        "--pose-backend",
        type=str,
        choices=["synthetic", "onnx", "ultralytics"],
        default=None,
        help="Override pose backend (synthetic, onnx, ultralytics) or set POSE_BACKEND env var",
    )
    parser.add_argument("--host", type=str, default=None, help="Override API bind host")
    parser.add_argument("--port", type=int, default=None, help="Override API bind port")
    args = parser.parse_args()

    cfg = ConfigLoader.load(args.config)
    _apply_live_camera_overrides(
        cfg,
        source_type_arg=args.source_type,
        source_uri_arg=args.source_uri,
        camera_id_arg=args.camera_id,
        pose_backend_arg=args.pose_backend,
    )

    all_file_sources = True
    for cam in cfg.cameras:
        if cam.source_type == SourceType.FILE:
            cam.loop_file = True
            uri_path = Path(str(cam.source_uri))
            if not uri_path.is_absolute():
                uri_path = PROJECT_ROOT / uri_path
            if not uri_path.exists():
                generate_synthetic_cctv_video(uri_path, num_frames=120)
            cam.source_uri = str(uri_path)
        else:
            all_file_sources = False

    stop_event = threading.Event()
    pipeline_app = ActivityApplication(cfg, stop_event=stop_event)
    fastapi_app = create_api_app(pipeline_app.snapshot_store)

    worker_thread = threading.Thread(
        target=pipeline_app.run,
        kwargs={
            "output_video_path": PROJECT_ROOT / "outputs" / "annotated_live.avi",
            "pace_fps": 15.0 if all_file_sources else None,
        },
        daemon=True,
        name="pipeline-worker",
    )
    worker_thread.start()

    def _handle_sig(_signum, _frame) -> None:
        stop_event.set()
        pipeline_app.shutdown()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    host = args.host or cfg.system.api_host
    port = args.port or cfg.system.api_port
    uvicorn.run(fastapi_app, host=host, port=port, log_level=cfg.system.log_level.lower())
    stop_event.set()
    worker_thread.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
