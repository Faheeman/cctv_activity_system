"""Video ingestion abstraction with pluggable adapters (Webcam, RTSP, VideoFile) and StreamManager."""

from collections import deque
from datetime import datetime, timezone
import logging
import os
import random
import threading
import time
from typing import Callable, Deque, Dict, List, Optional, Protocol, Tuple
import cv2
import numpy as np

from activity_system.config import CameraConfig
from activity_system.logging_config import get_camera_logger, redact_uri
from activity_system.schemas import CameraStatusSnapshot, FramePacket, SourceType, StreamState


class CaptureBackend(Protocol):
    """Protocol matching cv2.VideoCapture for dependency injection in tests."""

    def isOpened(self) -> bool:
        ...

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ...

    def release(self) -> None:
        ...

    def set(self, prop_id: int, value: float) -> bool:
        ...


def default_opencv_capture_factory(config: CameraConfig) -> CaptureBackend:
    """Create an OpenCV VideoCapture instance with bounded buffer, low-latency V4L2/FFmpeg, and timeout properties."""
    if config.source_type == SourceType.WEBCAM:
        dev_idx = int(config.source_uri)
        if hasattr(cv2, "CAP_V4L2"):
            cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(dev_idx)
        else:
            cap = cv2.VideoCapture(dev_idx)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, float(config.max_frame_queue_size))
        if config.resize_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(config.resize_width))
        if config.resize_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(config.resize_height))
        if config.target_fps:
            cap.set(cv2.CAP_PROP_FPS, float(config.target_fps))
        return cap

    uri_str = str(config.source_uri)
    if config.source_type == SourceType.RTSP:
        if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            )
        params = [
            cv2.CAP_PROP_BUFFERSIZE,
            int(config.max_frame_queue_size),
        ]
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(config.connection_timeout_ms)])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(config.read_timeout_ms)])
        try:
            cap = cv2.VideoCapture(uri_str, cv2.CAP_FFMPEG, params)
        except TypeError:
            cap = cv2.VideoCapture(uri_str, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, float(config.max_frame_queue_size))
        return cap

    # Prerecorded video file
    cap = cv2.VideoCapture(uri_str, cv2.CAP_FFMPEG)
    return cap


class VideoSource:
    """Manages single-camera video capture, frame sampling, bounded buffering, stale-frame handling, and reconnection."""

    def __init__(
        self,
        config: CameraConfig,
        capture_factory: Optional[Callable[[CameraConfig], CaptureBackend]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
        jitter_fn: Optional[Callable[[float, float], float]] = None,
        stop_event: Optional[threading.Event] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self._capture_factory = capture_factory or default_opencv_capture_factory
        self._sleep_fn = sleep_fn
        self._clock_fn = clock_fn
        self._jitter_fn = jitter_fn or random.uniform
        self._stop_event = stop_event or threading.Event()
        self._logger = get_camera_logger(config.camera_id, logger)

        self._lock = threading.Lock()
        self._capture: Optional[CaptureBackend] = None
        self._frame_buffer: Deque[FramePacket] = deque(maxlen=config.max_frame_queue_size)

        self._state: StreamState = StreamState.IDLE
        self._session_id: int = 1
        self._has_connected_once: bool = False
        self._frame_width: Optional[int] = config.resize_width
        self._frame_height: Optional[int] = config.resize_height
        self._frames_received: int = 0
        self._frames_emitted: int = 0
        self._frames_dropped: int = 0
        self._frames_skipped_sampling: int = 0
        self._stale_frames_dropped: int = 0
        self._reconnect_attempts: int = 0
        self._total_reconnections: int = 0
        self._last_frame_monotonic_ts: Optional[float] = None
        self._last_wall_monotonic_ts: Optional[float] = None
        self._last_frame_utc_iso: Optional[str] = None
        self._last_error: Optional[str] = None

    @property
    def camera_id(self) -> str:
        return self.config.camera_id

    @property
    def redacted_uri(self) -> str:
        return redact_uri(self.config.source_uri)

    def connect(self) -> bool:
        """Open the underlying video capture device or stream."""
        if self._stop_event.is_set():
            with self._lock:
                self._state = StreamState.STOPPED
            return False

        self._release_capture()
        try:
            cap = self._capture_factory(self.config)
            if cap is not None and cap.isOpened():
                with self._lock:
                    self._capture = cap
                    if self._has_connected_once and self.config.source_type in (
                        SourceType.RTSP,
                        SourceType.WEBCAM,
                    ):
                        self._session_id += 1
                        self._total_reconnections += 1
                    self._has_connected_once = True
                    self._state = StreamState.CONNECTED
                    self._reconnect_attempts = 0
                    self._last_error = None
                self._logger.info(
                    f"Connected video source {self.redacted_uri} (session={self._session_id})"
                )
                return True
            else:
                if cap is not None:
                    cap.release()
                err_msg = f"Failed to open video source {self.redacted_uri}"
                with self._lock:
                    self._state = StreamState.DISCONNECTED
                    self._last_error = err_msg
                self._logger.warning(err_msg)
                return False
        except Exception as exc:
            err_msg = f"Exception opening {self.redacted_uri}: {redact_uri(str(exc))}"
            with self._lock:
                self._state = StreamState.DISCONNECTED
                self._last_error = err_msg
            self._logger.error(err_msg)
            return False

    def _compute_backoff_delay(self, attempt: int) -> float:
        base_delay = min(
            self.config.reconnect_initial_delay_sec
            * (self.config.reconnect_backoff_factor ** (attempt - 1)),
            self.config.reconnect_max_delay_sec,
        )
        jitter_ratio = getattr(self.config, "reconnect_jitter_ratio", 0.0)
        if jitter_ratio > 0.0 and base_delay > 0.0:
            jitter_span = base_delay * jitter_ratio
            delay = base_delay + self._jitter_fn(-jitter_span, jitter_span)
            return float(max(0.0, min(self.config.reconnect_max_delay_sec, delay)))
        return float(base_delay)

    def _attempt_reconnect(self) -> bool:
        """Attempt reconnection using exponential backoff with optional jitter up to max_reconnect_attempts."""
        if self.config.source_type == SourceType.FILE:
            with self._lock:
                self._state = StreamState.EOF
            return False

        max_attempts = self.config.max_reconnect_attempts
        while not self._stop_event.is_set():
            with self._lock:
                if self._reconnect_attempts >= max_attempts:
                    self._state = StreamState.FAILED
                    self._last_error = (
                        f"Exceeded max_reconnect_attempts ({max_attempts}) for {self.redacted_uri}"
                    )
                    self._logger.error(self._last_error)
                    return False
                self._reconnect_attempts += 1
                attempt = self._reconnect_attempts
                self._state = StreamState.RECONNECTING

            delay = self._compute_backoff_delay(attempt)
            self._logger.warning(
                f"Reconnecting to {self.redacted_uri} (attempt {attempt}/{max_attempts}) after {delay:.2f}s"
            )
            if delay > 0:
                self._sleep_fn(delay)
            if self._stop_event.is_set():
                with self._lock:
                    self._state = StreamState.STOPPED
                return False
            if self.connect():
                return True
        with self._lock:
            self._state = StreamState.STOPPED
        return False

    def ingest_step(self) -> bool:
        """Read from capture until one sampled frame is pushed to the bounded buffer or stream ends."""
        if self._stop_event.is_set():
            with self._lock:
                self._state = StreamState.STOPPED
            return False

        if self._capture is None or not self._capture.isOpened():
            if not self._has_connected_once:
                if not self.connect() and not self._attempt_reconnect():
                    return False
            else:
                if not self._attempt_reconnect():
                    return False

        sample_interval = self.config.sample_every_n_frames
        while not self._stop_event.is_set():
            assert self._capture is not None
            try:
                ok, raw_frame = self._capture.read()
            except Exception as exc:
                ok, raw_frame = False, None
                with self._lock:
                    self._last_error = f"Capture read error: {redact_uri(str(exc))}"
                self._logger.error(self._last_error)

            if not ok or raw_frame is None or raw_frame.size == 0:
                if self.config.source_type == SourceType.FILE:
                    if getattr(self.config, "loop_file", False):
                        self._release_capture()
                        if self.connect():
                            continue
                    with self._lock:
                        self._state = StreamState.EOF
                    self._release_capture()
                    return False
                # Live stream frame failure -> trigger reconnect
                with self._lock:
                    self._state = StreamState.DISCONNECTED
                    if self._last_error is None:
                        self._last_error = f"Frame read failed on {self.redacted_uri}"
                self._release_capture()
                if not self._attempt_reconnect():
                    return False
                continue

            wall_mono = self._clock_fn()
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            with self._lock:
                self._frames_received += 1
                current_idx = self._frames_received
                if self.config.source_type == SourceType.FILE:
                    now_mono = float(current_idx - 1) / float(self.config.target_fps)
                else:
                    now_mono = wall_mono
                self._last_frame_monotonic_ts = now_mono
                self._last_wall_monotonic_ts = wall_mono
                self._last_frame_utc_iso = now_utc

            # Apply frame sampling (emit frame 1, 1 + N, 1 + 2N, ...)
            if (current_idx - 1) % sample_interval != 0:
                with self._lock:
                    self._frames_skipped_sampling += 1
                continue

            processed_frame = self._preprocess_frame(raw_frame)
            packet = FramePacket(
                camera_id=self.config.camera_id,
                session_id=self._session_id,
                frame_index=current_idx,
                frame=processed_frame,
                monotonic_ts=now_mono,
                utc_iso=now_utc,
            )

            with self._lock:
                self._frame_width = packet.width
                self._frame_height = packet.height
                if len(self._frame_buffer) == self._frame_buffer.maxlen:
                    self._frames_dropped += 1
                self._frame_buffer.append(packet)
            return True

        with self._lock:
            self._state = StreamState.STOPPED
        return False

    def _drop_stale_buffered_frames_locked(self, now_mono: float) -> None:
        """Evict buffered frames older than stale_frame_timeout_sec on live streams."""
        if self.config.source_type == SourceType.FILE:
            return
        stale_timeout = getattr(self.config, "stale_frame_timeout_sec", 5.0)
        while self._frame_buffer:
            oldest = self._frame_buffer[0]
            if (now_mono - oldest.monotonic_ts) > stale_timeout:
                self._frame_buffer.popleft()
                self._frames_dropped += 1
                self._stale_frames_dropped += 1
            else:
                break

    def read_next(self) -> Optional[FramePacket]:
        """Return the newest available non-stale sampled FramePacket, ingesting if buffer is empty."""
        if self._stop_event.is_set():
            with self._lock:
                self._state = StreamState.STOPPED
            return None

        now_mono = self._clock_fn()
        with self._lock:
            self._drop_stale_buffered_frames_locked(now_mono)
            if self._frame_buffer:
                packet = self._frame_buffer.pop()
                dropped_now = len(self._frame_buffer)
                if dropped_now > 0:
                    self._frames_dropped += dropped_now
                    self._frame_buffer.clear()
                self._frames_emitted += 1
                return packet

        if not self.ingest_step():
            return None

        now_mono = self._clock_fn()
        with self._lock:
            self._drop_stale_buffered_frames_locked(now_mono)
            if not self._frame_buffer:
                return None
            packet = self._frame_buffer.pop()
            self._frames_emitted += 1
            return packet

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to configured dimensions if needed."""
        target_w = self.config.resize_width
        target_h = self.config.resize_height
        if target_w and target_h:
            h, w = frame.shape[:2]
            if w != target_w or h != target_h:
                return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return frame

    def status(self) -> CameraStatusSnapshot:
        """Return a thread-safe snapshot of stream telemetry including dimensions and processing state."""
        now_mono = self._clock_fn()
        stale_timeout = getattr(self.config, "stale_frame_timeout_sec", 5.0)
        with self._lock:
            is_stale = False
            if (
                self.config.source_type in (SourceType.RTSP, SourceType.WEBCAM)
                and self._state == StreamState.CONNECTED
                and self._last_wall_monotonic_ts is not None
                and (now_mono - self._last_wall_monotonic_ts) > stale_timeout
            ):
                is_stale = True

            if self._state == StreamState.CONNECTED:
                proc_state = "stale" if is_stale else "active"
            else:
                proc_state = self._state.value

            return CameraStatusSnapshot(
                camera_id=self.config.camera_id,
                source_type=self.config.source_type,
                redacted_uri=self.redacted_uri,
                state=self._state,
                processing_state=proc_state,
                session_id=self._session_id,
                frame_width=self._frame_width,
                frame_height=self._frame_height,
                frames_received=self._frames_received,
                frames_emitted=self._frames_emitted,
                frames_dropped=self._frames_dropped,
                frames_skipped_sampling=self._frames_skipped_sampling,
                stale_frames_dropped=self._stale_frames_dropped,
                is_stale=is_stale,
                reconnect_attempts=self._reconnect_attempts,
                total_reconnections=self._total_reconnections,
                last_frame_monotonic_ts=self._last_frame_monotonic_ts,
                last_frame_utc_iso=self._last_frame_utc_iso,
                last_error=self._last_error,
            )

    def _release_capture(self) -> None:
        with self._lock:
            cap = self._capture
            self._capture = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def close(self) -> None:
        """Signal shutdown and release OpenCV capture resources."""
        self._stop_event.set()
        self._release_capture()
        with self._lock:
            self._frame_buffer.clear()
            if self._state not in (StreamState.EOF, StreamState.FAILED):
                self._state = StreamState.STOPPED

    def __enter__(self) -> "VideoSource":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class WebcamStreamAdapter(VideoSource):
    """Pluggable input adapter specialized for USB and integrated webcams."""

    def __init__(self, config: CameraConfig, **kwargs) -> None:
        if config.source_type != SourceType.WEBCAM:
            raise ValueError(f"WebcamStreamAdapter requires source_type=webcam, got {config.source_type}")
        super().__init__(config, **kwargs)


class RtspStreamAdapter(VideoSource):
    """Pluggable input adapter specialized for RTSP/RTSPS IP cameras with credential redaction."""

    def __init__(self, config: CameraConfig, **kwargs) -> None:
        if config.source_type != SourceType.RTSP:
            raise ValueError(f"RtspStreamAdapter requires source_type=rtsp, got {config.source_type}")
        super().__init__(config, **kwargs)


class VideoFileStreamAdapter(VideoSource):
    """Pluggable input adapter specialized for local video file replay and testing."""

    def __init__(self, config: CameraConfig, **kwargs) -> None:
        if config.source_type != SourceType.FILE:
            raise ValueError(f"VideoFileStreamAdapter requires source_type=file, got {config.source_type}")
        super().__init__(config, **kwargs)


def create_stream_adapter(config: CameraConfig, **kwargs) -> VideoSource:
    """Factory creating the appropriate InputStreamAdapter implementation for a CameraConfig."""
    if config.source_type == SourceType.WEBCAM:
        return WebcamStreamAdapter(config, **kwargs)
    if config.source_type == SourceType.RTSP:
        return RtspStreamAdapter(config, **kwargs)
    if config.source_type == SourceType.FILE:
        return VideoFileStreamAdapter(config, **kwargs)
    return VideoSource(config, **kwargs)


class StreamManager:
    """Manages multiple pluggable camera stream adapters with per-camera error isolation."""

    def __init__(
        self,
        adapters: Optional[Dict[str, VideoSource]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._adapters: Dict[str, VideoSource] = dict(adapters or {})
        self._logger = logger or logging.getLogger("activity_system")

    @property
    def sources(self) -> Dict[str, VideoSource]:
        return self._adapters

    def register(self, adapter: VideoSource) -> None:
        self._adapters[adapter.camera_id] = adapter

    def get(self, camera_id: str) -> Optional[VideoSource]:
        return self._adapters.get(camera_id)

    def read_next_safe(self, camera_id: str) -> Optional[FramePacket]:
        """Read next frame from camera_id while isolating unexpected adapter exceptions."""
        adapter = self._adapters.get(camera_id)
        if adapter is None:
            raise KeyError(f"Unknown camera_id: {camera_id}")
        try:
            return adapter.read_next()
        except Exception as exc:
            self._logger.error(
                f"Isolated stream error on camera {camera_id}: {redact_uri(str(exc))}",
                extra={"camera_id": camera_id},
            )
            return None

    def statuses(self) -> List[CameraStatusSnapshot]:
        return [adapter.status() for adapter in self._adapters.values()]

    def close_all(self) -> None:
        for adapter in self._adapters.values():
            try:
                adapter.close()
            except Exception:
                pass
