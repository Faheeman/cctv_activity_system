"""8-state constant-velocity Kalman filter and vectorized IoU computation for multi-person tracking."""

from typing import Sequence, Tuple
import numpy as np


def compute_iou_matrix(
    boxes_a: Sequence[Tuple[int, int, int, int]],
    boxes_b: Sequence[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Compute pairwise IoU matrix of shape (len(boxes_a), len(boxes_b))."""
    n_a = len(boxes_a)
    n_b = len(boxes_b)
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=np.float32)

    arr_a = np.asarray(boxes_a, dtype=np.float32)
    arr_b = np.asarray(boxes_b, dtype=np.float32)

    x1 = np.maximum(arr_a[:, None, 0], arr_b[None, :, 0])
    y1 = np.maximum(arr_a[:, None, 1], arr_b[None, :, 1])
    x2 = np.minimum(arr_a[:, None, 2], arr_b[None, :, 2])
    y2 = np.minimum(arr_a[:, None, 3], arr_b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(1.0, (arr_a[:, 2] - arr_a[:, 0]) * (arr_a[:, 3] - arr_a[:, 1]))
    area_b = np.maximum(1.0, (arr_b[:, 2] - arr_b[:, 0]) * (arr_b[:, 3] - arr_b[:, 1]))

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return np.clip(inter_area / np.maximum(1e-6, union_area), 0.0, 1.0)


class KalmanBoxState:
    """Constant-velocity Kalman filter for 2D bounding boxes in (cx, cy, w, h, vx, vy, vw, vh) space."""

    def __init__(self, bbox: Tuple[int, int, int, int], timestamp: float) -> None:
        x1, y1, x2, y2 = bbox
        w = max(4.0, float(x2 - x1))
        h = max(8.0, float(y2 - y1))
        cx = float(x1) + 0.5 * w
        cy = float(y1) + 0.5 * h

        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 50.0, 50.0]).astype(np.float64)
        self.R = np.diag([4.0, 4.0, 6.0, 6.0]).astype(np.float64)
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0
        self.last_ts = float(timestamp)

    def predict(self, timestamp: float) -> Tuple[int, int, int, int]:
        dt = max(1e-3, min(1.0, float(timestamp) - self.last_ts))
        self.last_ts = float(timestamp)

        F = np.eye(8, dtype=np.float64)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt

        q_pos = 2.0 * dt
        q_vel = 8.0 * dt
        Q = np.diag([q_pos, q_pos, q_pos, q_pos, q_vel, q_vel, q_vel * 0.5, q_vel * 0.5])

        self.x = F @ self.x
        self.x[2] = max(6.0, self.x[2])
        self.x[3] = max(12.0, self.x[3])
        self.P = F @ self.P @ F.T + Q
        return self.current_bbox()

    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        w = max(4.0, float(x2 - x1))
        h = max(8.0, float(y2 - y1))
        z = np.array([float(x1) + 0.5 * w, float(y1) + 0.5 * h, w, h], dtype=np.float64)

        y_res = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y_res)
        self.P = (np.eye(8, dtype=np.float64) - K @ self.H) @ self.P
        return self.current_bbox()

    def current_bbox(self) -> Tuple[int, int, int, int]:
        cx, cy, w, h = self.x[:4]
        w = max(6.0, w)
        h = max(12.0, h)
        x1 = int(round(cx - 0.5 * w))
        y1 = int(round(cy - 0.5 * h))
        x2 = int(round(cx + 0.5 * w))
        y2 = int(round(cy + 0.5 * h))
        return (x1, y1, max(x1 + 6, x2), max(y1 + 12, y2))
