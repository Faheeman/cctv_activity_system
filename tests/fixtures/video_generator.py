"""Deterministic synthetic CCTV video generator for offline unit and integration tests."""

from pathlib import Path
from typing import Union
import cv2
import numpy as np


def generate_synthetic_cctv_video(
    output_path: Union[str, Path],
    num_frames: int = 30,
    width: int = 640,
    height: int = 480,
    fps: float = 15.0,
) -> Path:
    """Create a deterministic MJPG .avi video simulating a CCTV corridor scene."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open cv2.VideoWriter for {path}")

    try:
        for idx in range(num_frames):
            frame = np.full((height, width, 3), 36, dtype=np.uint8)
            # Draw static corridor floor/wall reference lines
            cv2.line(frame, (0, int(height * 0.75)), (width, int(height * 0.75)), (80, 80, 80), 2)
            # Draw moving synthetic person silhouette
            cx = int(80 + (idx * 12) % (width - 160))
            cy = int(height * 0.55)
            cv2.rectangle(frame, (cx - 25, cy - 70), (cx + 25, cy + 70), (200, 180, 140), -1)
            cv2.circle(frame, (cx, cy - 85), 14, (220, 200, 180), -1)
            cv2.putText(
                frame,
                f"CCTV_FRAME_{idx:03d}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                1,
            )
            writer.write(frame)
    finally:
        writer.release()

    return path
