"""Human pose estimation and person detection module with COCO 17-keypoint validation and pluggable backends."""

import logging
from pathlib import Path
import time
from typing import Callable, List, Optional, Protocol, Sequence, Tuple
import cv2
import numpy as np

from activity_system.config import PoseConfig
from activity_system.schemas import (
    COCO_KEYPOINT_NAMES,
    FramePacket,
    Keypoint,
    ObservationQuality,
    PoseDetection,
)


COCO_SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


class RawPoseCandidate:
    """Unvalidated output from a pose backend in original image coordinates."""

    __slots__ = ("bbox", "score", "keypoints")

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        score: float,
        keypoints: np.ndarray,
    ) -> None:
        self.bbox = bbox
        self.score = float(score)
        self.keypoints = np.asarray(keypoints, dtype=np.float32)


class PoseBackend(Protocol):
    """Protocol for pluggable single-pass multi-person pose inference engines."""

    def infer(self, frame: np.ndarray) -> List[RawPoseCandidate]:
        ...


def build_human_coco_keypoints(
    center_x: float,
    center_y: float,
    body_height: float,
    phase_rad: float = 0.0,
    stride_amp_bh: float = 0.0,
    knee_lift_bh: float = 0.0,
    confidence: float = 0.92,
    occlude_lower_body: bool = False,
    posture: str = "upright",
) -> np.ndarray:
    """Synthesize anatomically proportioned 17 COCO keypoints for upright, sitting, or lying postures."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    half_h = body_height * 0.5
    shoulder_w = body_height * 0.22
    hip_w = body_height * 0.16

    if posture == "lying":
        # Horizontal body posture along X-axis with width = body_height and vertical thickness ~0.28 * body_height
        head_x = center_x - half_h * 0.80
        shoulder_x = center_x - half_h * 0.48
        elbow_x = center_x - half_h * 0.15
        wrist_x = center_x + half_h * 0.05
        hip_x = center_x + half_h * 0.05
        knee_x = center_x + half_h * 0.45
        ankle_x = center_x + half_h * 0.85
        coords = [
            (head_x, center_y - 6.0, confidence),
            (head_x + 4.0, center_y - 10.0, confidence),
            (head_x + 4.0, center_y - 2.0, confidence),
            (head_x + 8.0, center_y - 12.0, confidence * 0.9),
            (head_x + 8.0, center_y, confidence * 0.9),
            (shoulder_x, center_y - 14.0, confidence),
            (shoulder_x, center_y + 14.0, confidence),
            (elbow_x, center_y - 18.0, confidence),
            (elbow_x, center_y + 18.0, confidence),
            (wrist_x, center_y - 16.0, confidence),
            (wrist_x, center_y + 16.0, confidence),
            (hip_x, center_y - 12.0, confidence),
            (hip_x, center_y + 12.0, confidence),
            (knee_x, center_y - 12.0, 0.10 if occlude_lower_body else confidence),
            (knee_x, center_y + 12.0, 0.10 if occlude_lower_body else confidence),
            (ankle_x, center_y - 10.0, 0.08 if occlude_lower_body else confidence),
            (ankle_x, center_y + 10.0, 0.08 if occlude_lower_body else confidence),
        ]
        for idx, (x, y, c) in enumerate(coords):
            kpts[idx, 0] = x
            kpts[idx, 1] = y
            kpts[idx, 2] = c
        return kpts

    if posture == "sitting":
        # Seated posture: upright upper torso, knees bent horizontally forward near hip level
        nose_y = center_y - half_h * 0.65
        shoulder_y = center_y - half_h * 0.35
        elbow_y = center_y - half_h * 0.05
        wrist_y = center_y + half_h * 0.08
        hip_y = center_y + half_h * 0.15
        knee_y = center_y + half_h * 0.22  # knees nearly level with hips
        ankle_y = center_y + half_h * 0.62
        coords = [
            (center_x, nose_y, confidence),
            (center_x - shoulder_w * 0.2, nose_y - 3.0, confidence),
            (center_x + shoulder_w * 0.2, nose_y - 3.0, confidence),
            (center_x - shoulder_w * 0.35, nose_y, confidence * 0.9),
            (center_x + shoulder_w * 0.35, nose_y, confidence * 0.9),
            (center_x - shoulder_w * 0.5, shoulder_y, confidence),
            (center_x + shoulder_w * 0.5, shoulder_y, confidence),
            (center_x - shoulder_w * 0.6, elbow_y, confidence),
            (center_x + shoulder_w * 0.6, elbow_y, confidence),
            (center_x - shoulder_w * 0.4, wrist_y, confidence),
            (center_x + shoulder_w * 0.4, wrist_y, confidence),
            (center_x - hip_w * 0.5, hip_y, confidence),
            (center_x + hip_w * 0.5, hip_y, confidence),
            (center_x - hip_w * 0.8 - body_height * 0.14, knee_y, 0.10 if occlude_lower_body else confidence),
            (center_x + hip_w * 0.8 + body_height * 0.14, knee_y, 0.10 if occlude_lower_body else confidence),
            (center_x - hip_w * 0.7 - body_height * 0.12, ankle_y, 0.08 if occlude_lower_body else confidence),
            (center_x + hip_w * 0.7 + body_height * 0.12, ankle_y, 0.08 if occlude_lower_body else confidence),
        ]
        for idx, (x, y, c) in enumerate(coords):
            kpts[idx, 0] = x
            kpts[idx, 1] = y
            kpts[idx, 2] = c
        return kpts

    nose_y = center_y - half_h * 0.85
    shoulder_y = center_y - half_h * 0.52
    elbow_y = center_y - half_h * 0.15
    wrist_y = center_y + half_h * 0.12
    hip_y = center_y + half_h * 0.05
    base_knee_y = center_y + half_h * 0.48
    base_ankle_y = center_y + half_h * 0.90

    left_swing = float(np.sin(phase_rad))
    right_swing = float(np.sin(phase_rad + np.pi))

    left_ankle_x = center_x - hip_w * 0.5 + left_swing * stride_amp_bh * body_height
    right_ankle_x = center_x + hip_w * 0.5 + right_swing * stride_amp_bh * body_height

    left_ankle_y = base_ankle_y - max(0.0, left_swing) * knee_lift_bh * body_height
    right_ankle_y = base_ankle_y - max(0.0, right_swing) * knee_lift_bh * body_height

    left_knee_x = 0.5 * (center_x - hip_w * 0.5 + left_ankle_x)
    right_knee_x = 0.5 * (center_x + hip_w * 0.5 + right_ankle_x)
    left_knee_y = base_knee_y - max(0.0, left_swing) * (knee_lift_bh * 0.75) * body_height
    right_knee_y = base_knee_y - max(0.0, right_swing) * (knee_lift_bh * 0.75) * body_height

    coords = [
        (center_x, nose_y, confidence),
        (center_x - shoulder_w * 0.2, nose_y - 4.0, confidence),
        (center_x + shoulder_w * 0.2, nose_y - 4.0, confidence),
        (center_x - shoulder_w * 0.35, nose_y, confidence * 0.9),
        (center_x + shoulder_w * 0.35, nose_y, confidence * 0.9),
        (center_x - shoulder_w * 0.5, shoulder_y, confidence),
        (center_x + shoulder_w * 0.5, shoulder_y, confidence),
        (center_x - shoulder_w * 0.65, elbow_y, confidence),
        (center_x + shoulder_w * 0.65, elbow_y, confidence),
        (center_x - shoulder_w * 0.6 - left_swing * 8.0, wrist_y, confidence),
        (center_x + shoulder_w * 0.6 - right_swing * 8.0, wrist_y, confidence),
        (center_x - hip_w * 0.5, hip_y, confidence),
        (center_x + hip_w * 0.5, hip_y, confidence),
        (left_knee_x, left_knee_y, 0.10 if occlude_lower_body else confidence),
        (right_knee_x, right_knee_y, 0.10 if occlude_lower_body else confidence),
        (left_ankle_x, left_ankle_y, 0.08 if occlude_lower_body else confidence),
        (right_ankle_x, right_ankle_y, 0.08 if occlude_lower_body else confidence),
    ]
    for idx, (x, y, c) in enumerate(coords):
        kpts[idx, 0] = x
        kpts[idx, 1] = y
        kpts[idx, 2] = c
    return kpts


class SyntheticPoseBackend:
    """Deterministic pose backend for offline verification, evaluation sequences, and fallback live camera detection."""

    def __init__(
        self,
        custom_generator: Optional[Callable[[np.ndarray, int], List[RawPoseCandidate]]] = None,
    ) -> None:
        self._custom_generator = custom_generator
        self._call_count = 0
        self._hog: Optional[cv2.HOGDescriptor] = None

    def _get_hog(self) -> cv2.HOGDescriptor:
        if self._hog is None:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self._hog = hog
        return self._hog

    def infer(self, frame: np.ndarray) -> List[RawPoseCandidate]:
        self._call_count += 1
        if frame is None or frame.size == 0:
            return []
        if self._custom_generator is not None:
            return self._custom_generator(frame, self._call_count)

        if int(np.max(frame)) <= 15:
            return []

        mask = cv2.inRange(frame, (110, 130, 110), (245, 225, 215))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[RawPoseCandidate] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw >= 16 and bh >= 45:
                cx = x + bw * 0.5
                cy = y + bh * 0.5
                body_h = float(bh * 1.15)
                phase = self._call_count * 0.45
                kpts = build_human_coco_keypoints(
                    center_x=cx,
                    center_y=cy,
                    body_height=body_h,
                    phase_rad=phase,
                    stride_amp_bh=0.14,
                    knee_lift_bh=0.05,
                    confidence=0.92,
                )
                bbox = (
                    float(cx - body_h * 0.25),
                    float(cy - body_h * 0.55),
                    float(cx + body_h * 0.25),
                    float(cy + body_h * 0.55),
                )
                candidates.append(RawPoseCandidate(bbox=bbox, score=0.93, keypoints=kpts))

        if candidates:
            return candidates

        # Fallback for real camera frames when running in synthetic mode: fast downscaled HOG pedestrian detector
        orig_h, orig_w = frame.shape[:2]
        if orig_w >= 160 and orig_h >= 120 and float(np.std(frame)) > 8.0:
            scale_w = 320
            scale_h = max(120, int(round(orig_h * (320.0 / float(orig_w)))))
            small = cv2.resize(frame, (scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
            hog = self._get_hog()
            rects, weights = hog.detectMultiScale(
                small, winStride=(8, 8), padding=(8, 8), scale=1.08
            )
            sx = float(orig_w) / float(scale_w)
            sy = float(orig_h) / float(scale_h)
            for (rx, ry, rw, rh), weight in zip(rects, weights):
                conf = float(np.clip(0.55 + 0.35 * float(weight), 0.55, 0.95))
                cx = (rx + rw * 0.5) * sx
                cy = (ry + rh * 0.5) * sy
                body_h = float(rh * sy * 0.90)
                kpts = build_human_coco_keypoints(
                    center_x=cx,
                    center_y=cy,
                    body_height=body_h,
                    phase_rad=self._call_count * 0.35,
                    stride_amp_bh=0.08,
                    knee_lift_bh=0.03,
                    confidence=conf,
                )
                bbox = (
                    float(rx * sx),
                    float(ry * sy),
                    float((rx + rw) * sx),
                    float((ry + rh) * sy),
                )
                candidates.append(RawPoseCandidate(bbox=bbox, score=conf, keypoints=kpts))

        return candidates


def decode_yolo_pose_output(
    output_tensor: np.ndarray,
    orig_width: int,
    orig_height: int,
    imgsz: int,
    conf_thresh: float,
    iou_thresh: float,
    input_width: Optional[int] = None,
    input_height: Optional[int] = None,
) -> List[RawPoseCandidate]:
    """Decode standard YOLOv8/YOLO11-pose output tensor of shape (1, 56, N) into RawPoseCandidates."""
    preds = np.squeeze(output_tensor)
    if preds.ndim != 2:
        return []
    if preds.shape[0] == 56 and preds.shape[1] != 56:
        preds = preds.T
    if preds.shape[1] < 56:
        return []

    scores = preds[:, 4]
    keep_mask = scores >= conf_thresh
    preds = preds[keep_mask]
    scores = scores[keep_mask]
    if len(preds) == 0:
        return []

    input_width = input_width or imgsz
    input_height = input_height or imgsz
    scale_x = float(orig_width) / float(input_width)
    scale_y = float(orig_height) / float(input_height)

    cx = preds[:, 0] * scale_x
    cy = preds[:, 1] * scale_y
    bw = preds[:, 2] * scale_x
    bh = preds[:, 3] * scale_y

    x1 = cx - bw * 0.5
    y1 = cy - bh * 0.5
    x2 = cx + bw * 0.5
    y2 = cy + bh * 0.5

    boxes_xywh = [[float(x1[i]), float(y1[i]), float(bw[i]), float(bh[i])] for i in range(len(preds))]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), float(conf_thresh), float(iou_thresh))
    if len(indices) == 0:
        return []

    flat_indices = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indices]
    candidates: List[RawPoseCandidate] = []
    for idx in flat_indices:
        kpt_raw = preds[idx, 5:56].reshape(17, 3).copy()
        kpt_raw[:, 0] *= scale_x
        kpt_raw[:, 1] *= scale_y
        kpt_raw[:, 2] = np.clip(kpt_raw[:, 2], 0.0, 1.0)
        candidates.append(
            RawPoseCandidate(
                bbox=(float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])),
                score=float(np.clip(scores[idx], 0.0, 1.0)),
                keypoints=kpt_raw,
            )
        )
    return candidates


class OnnxPoseBackend:
    """ONNX Runtime backend for exported YOLOv8-pose / YOLO11-pose models."""

    def __init__(self, config: PoseConfig) -> None:
        import onnxruntime as ort

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX pose model not found at {model_path}. See models/README.md for export instructions."
            )
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = config.cpu_threads
        available = ort.get_available_providers()
        dev = config.device.lower()
        if dev == "auto":
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        elif dev.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif dev == "tensorrt" and "TensorrtExecutionProvider" in available:
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        shape = input_meta.shape
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"Expected ONNX input shape [1, 3, H, W], got {shape}")
        self._input_height = int(shape[2]) if isinstance(shape[2], int) else config.imgsz
        self._input_width = int(shape[3]) if isinstance(shape[3], int) else config.imgsz
        self._config = config

    def infer(self, frame: np.ndarray) -> List[RawPoseCandidate]:
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(
            frame,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        outputs = self._session.run(None, {self._input_name: blob})
        return decode_yolo_pose_output(
            outputs[0],
            orig_width=orig_w,
            orig_height=orig_h,
            imgsz=self._config.imgsz,
            input_width=self._input_width,
            input_height=self._input_height,
            conf_thresh=self._config.min_person_confidence,
            iou_thresh=self._config.iou_threshold,
        )


class UltralyticsPoseBackend:
    """Ultralytics YOLO pose backend for hosts with the `ultralytics` package installed."""

    def __init__(self, config: PoseConfig) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics package is not installed. Install optional dependency group `pip install .[yolo]` "
                "or use `backend: onnx` / `backend: synthetic`."
            ) from exc
        self._model = YOLO(config.model_path)
        self._config = config

    def infer(self, frame: np.ndarray) -> List[RawPoseCandidate]:
        dev = None if self._config.device.lower() == "auto" else self._config.device
        results = self._model.predict(
            source=frame,
            imgsz=self._config.imgsz,
            conf=self._config.min_person_confidence,
            iou=self._config.iou_threshold,
            device=dev,
            verbose=False,
        )
        if not results:
            return []
        res = results[0]
        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kpts_xy = res.keypoints.xy.cpu().numpy()
        kpts_conf = (
            res.keypoints.conf.cpu().numpy()
            if res.keypoints.conf is not None
            else np.ones((len(xyxy), 17), dtype=np.float32)
        )
        candidates: List[RawPoseCandidate] = []
        for i in range(len(xyxy)):
            kpt_arr = np.zeros((17, 3), dtype=np.float32)
            kpt_arr[:, 0:2] = kpts_xy[i]
            kpt_arr[:, 2] = np.clip(kpts_conf[i], 0.0, 1.0)
            candidates.append(
                RawPoseCandidate(
                    bbox=(float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3])),
                    score=float(np.clip(confs[i], 0.0, 1.0)),
                    keypoints=kpt_arr,
                )
            )
        return candidates


class PoseEstimator:
    """Validates person detections and COCO 17 keypoints from a configured PoseBackend."""

    def __init__(
        self,
        config: PoseConfig,
        backend: Optional[PoseBackend] = None,
    ) -> None:
        self.config = config
        self._is_ready = True
        self._backend_name = config.backend
        if backend is not None:
            self._backend = backend
        else:
            self._backend, self._is_ready, self._backend_name = self._create_backend(config)
        self._total_inferences = 0
        self._total_inference_sec = 0.0

    @staticmethod
    def _create_backend(config: PoseConfig) -> Tuple[PoseBackend, bool, str]:
        logger = logging.getLogger("activity_system")
        if config.backend == "synthetic":
            return SyntheticPoseBackend(), True, "synthetic"
        if config.backend == "onnx":
            try:
                return OnnxPoseBackend(config), True, "onnx"
            except FileNotFoundError as exc:
                if getattr(config, "allow_synthetic_fallback", True):
                    logger.warning(
                        f"{exc} Falling back to SyntheticPoseBackend (HOG + keypoint estimator) until model_path is provided."
                    )
                    return SyntheticPoseBackend(), True, "synthetic-fallback"
                raise
        if config.backend == "ultralytics":
            return UltralyticsPoseBackend(config), True, "ultralytics"
        raise ValueError(f"Unsupported pose backend: {config.backend}")

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self._backend is not None

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def mean_inference_ms(self) -> float:
        if self._total_inferences == 0:
            return 0.0
        return (self._total_inference_sec / self._total_inferences) * 1000.0

    def detect_persons(self, packet: FramePacket) -> List[PoseDetection]:
        """PersonDetector stage interface delegating to validated single-pass estimation."""
        return self.estimate(packet)

    def estimate(self, packet: FramePacket) -> List[PoseDetection]:
        """Run single-pass pose estimation on a FramePacket and return validated PoseDetections."""
        if packet.frame is None or packet.frame.size == 0:
            return []

        t0 = time.perf_counter()
        raw_candidates = self._backend.infer(packet.frame)
        elapsed = time.perf_counter() - t0
        self._total_inferences += 1
        self._total_inference_sec += elapsed

        return self._validate_candidates(
            raw_candidates,
            frame_width=packet.width,
            frame_height=packet.height,
        )

    def _validate_candidates(
        self,
        candidates: Sequence[RawPoseCandidate],
        frame_width: int,
        frame_height: int,
    ) -> List[PoseDetection]:
        validated: List[PoseDetection] = []
        min_person_conf = self.config.min_person_confidence
        min_kpt_conf = self.config.min_keypoint_confidence

        for cand in candidates:
            if cand.score < min_person_conf:
                continue

            x1 = int(np.clip(np.floor(cand.bbox[0]), 0, frame_width - 1))
            y1 = int(np.clip(np.floor(cand.bbox[1]), 0, frame_height - 1))
            x2 = int(np.clip(np.ceil(cand.bbox[2]), 0, frame_width))
            y2 = int(np.clip(np.ceil(cand.bbox[3]), 0, frame_height))

            if (x2 - x1) < 6 or (y2 - y1) < 12:
                continue

            kpt_arr = cand.keypoints
            if kpt_arr is None or kpt_arr.shape != (17, 3):
                keypoints = [
                    Keypoint(
                        index=idx,
                        name=COCO_KEYPOINT_NAMES[idx],
                        x=0.0,
                        y=0.0,
                        confidence=0.0,
                        visible=False,
                    )
                    for idx in range(17)
                ]
                quality = ObservationQuality.POOR
            else:
                keypoints = []
                for idx in range(17):
                    kx = float(kpt_arr[idx, 0])
                    ky = float(kpt_arr[idx, 1])
                    kconf = float(np.clip(kpt_arr[idx, 2], 0.0, 1.0))
                    in_bounds = (0.0 <= kx <= float(frame_width)) and (0.0 <= ky <= float(frame_height))
                    is_visible = bool(in_bounds and kconf >= min_kpt_conf and not np.isnan(kx) and not np.isnan(ky))
                    keypoints.append(
                        Keypoint(
                            index=idx,
                            name=COCO_KEYPOINT_NAMES[idx],
                            x=float(np.clip(kx, 0.0, float(frame_width))) if not np.isnan(kx) else 0.0,
                            y=float(np.clip(ky, 0.0, float(frame_height))) if not np.isnan(ky) else 0.0,
                            confidence=kconf if not np.isnan(kconf) else 0.0,
                            visible=is_visible,
                        )
                    )
                quality = self._assess_observation_quality(keypoints)

            validated.append(
                PoseDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(np.clip(cand.score, 0.0, 1.0)),
                    keypoints=keypoints,
                    observation_quality=quality,
                )
            )

        return validated

    @staticmethod
    def _assess_observation_quality(keypoints: Sequence[Keypoint]) -> ObservationQuality:
        visible_indices = {kp.index for kp in keypoints if kp.visible}
        torso_visible = len(visible_indices.intersection({5, 6, 11, 12}))
        lower_visible = len(visible_indices.intersection({13, 14, 15, 16}))
        total_visible = len(visible_indices)

        if total_visible >= 11 and torso_visible >= 3 and lower_visible >= 2:
            return ObservationQuality.GOOD
        if total_visible >= 6 and (torso_visible >= 2 or lower_visible >= 2):
            return ObservationQuality.PARTIAL
        return ObservationQuality.POOR
