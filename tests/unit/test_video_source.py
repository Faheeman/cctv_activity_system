"""Unit tests for VideoSource, pluggable stream adapters, jitter, stale frames, and StreamManager."""

import threading
from typing import List, Optional, Tuple
import unittest
import numpy as np

from activity_system.config import CameraConfig
from activity_system.schemas import SourceType, StreamState
from activity_system.video_source import (
    CaptureBackend,
    RtspStreamAdapter,
    StreamManager,
    VideoFileStreamAdapter,
    VideoSource,
    WebcamStreamAdapter,
    create_stream_adapter,
)


class FakeCapture(CaptureBackend):
    """Deterministic fake capture backend simulating file, webcam, and RTSP behaviors."""

    def __init__(
        self,
        frames: List[Optional[np.ndarray]],
        opened: bool = True,
    ) -> None:
        self._frames = list(frames)
        self._opened = opened
        self.read_idx = 0
        self.released = False
        self.props = {}

    def isOpened(self) -> bool:
        return self._opened and not self.released

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.isOpened():
            return False, None
        if self.read_idx >= len(self._frames):
            return False, None
        frame = self._frames[self.read_idx]
        self.read_idx += 1
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self.released = True
        self._opened = False

    def set(self, prop_id: int, value: float) -> bool:
        self.props[prop_id] = value
        return True


def _make_frame(width: int = 320, height: int = 240, val: int = 100) -> np.ndarray:
    return np.full((height, width, 3), val, dtype=np.uint8)


class TestVideoSourceUnit(unittest.TestCase):
    def test_file_source_reads_and_reaches_eof(self) -> None:
        frames = [_make_frame(320, 240, i) for i in range(5)]
        fake_cap = FakeCapture(frames)
        cfg = CameraConfig(
            camera_id="cam_file",
            source_type=SourceType.FILE,
            source_uri="dummy.avi",
            resize_width=160,
            resize_height=120,
        )
        src = VideoSource(cfg, capture_factory=lambda _: fake_cap)
        emitted = []
        while True:
            pkt = src.read_next()
            if pkt is None:
                break
            emitted.append(pkt)

        self.assertEqual(len(emitted), 5)
        self.assertEqual(emitted[0].width, 160)
        self.assertEqual(emitted[0].height, 120)
        status = src.status()
        self.assertEqual(status.state, StreamState.EOF)
        self.assertEqual(status.frame_width, 160)
        self.assertEqual(status.frame_height, 120)
        self.assertEqual(status.frames_received, 5)
        self.assertEqual(status.frames_emitted, 5)
        self.assertTrue(fake_cap.released)

    def test_pluggable_adapter_factory_and_stream_manager_isolation(self) -> None:
        file_cfg = CameraConfig(camera_id="cam_f", source_type=SourceType.FILE, source_uri="a.avi")
        web_cfg = CameraConfig(camera_id="cam_w", source_type=SourceType.WEBCAM, source_uri=0)
        rtsp_cfg = CameraConfig(
            camera_id="cam_r", source_type=SourceType.RTSP, source_uri="rtsp://10.0.0.1:554/live"
        )
        self.assertIsInstance(
            create_stream_adapter(file_cfg, capture_factory=lambda _: FakeCapture([_make_frame()])),
            VideoFileStreamAdapter,
        )
        self.assertIsInstance(
            create_stream_adapter(web_cfg, capture_factory=lambda _: FakeCapture([_make_frame()])),
            WebcamStreamAdapter,
        )
        self.assertIsInstance(
            create_stream_adapter(rtsp_cfg, capture_factory=lambda _: FakeCapture([_make_frame()])),
            RtspStreamAdapter,
        )

        # Verify StreamManager isolates camera errors
        good_adapter = create_stream_adapter(
            file_cfg, capture_factory=lambda _: FakeCapture([_make_frame(160, 120, 42)])
        )
        mgr = StreamManager({"cam_f": good_adapter})
        pkt = mgr.read_next_safe("cam_f")
        self.assertIsNotNone(pkt)
        mgr.close_all()

    def test_frame_sampling_interval(self) -> None:
        frames = [_make_frame(160, 120, i) for i in range(10)]
        fake_cap = FakeCapture(frames)
        cfg = CameraConfig(
            camera_id="cam_sampled",
            source_type=SourceType.FILE,
            source_uri="sampled.avi",
            sample_every_n_frames=3,
            resize_width=160,
            resize_height=120,
        )
        src = VideoSource(cfg, capture_factory=lambda _: fake_cap)
        indices = []
        while True:
            pkt = src.read_next()
            if pkt is None:
                break
            indices.append(pkt.frame_index)

        # Expect frames 1, 4, 7, 10 (1-indexed)
        self.assertEqual(indices, [1, 4, 7, 10])
        status = src.status()
        self.assertEqual(status.frames_received, 10)
        self.assertEqual(status.frames_emitted, 4)
        self.assertEqual(status.frames_skipped_sampling, 6)

    def test_bounded_buffer_drops_stale_frames(self) -> None:
        frames = [_make_frame(160, 120, i + 1) for i in range(4)]
        fake_cap = FakeCapture(frames)
        cfg = CameraConfig(
            camera_id="cam_webcam",
            source_type=SourceType.WEBCAM,
            source_uri=0,
            max_frame_queue_size=1,
            resize_width=160,
            resize_height=120,
        )
        src = VideoSource(cfg, capture_factory=lambda _: fake_cap)
        # Ingest 3 frames before reading any
        self.assertTrue(src.ingest_step())
        self.assertTrue(src.ingest_step())
        self.assertTrue(src.ingest_step())

        pkt = src.read_next()
        self.assertIsNotNone(pkt)
        assert pkt is not None
        # Newest frame (frame 3) should be returned; frames 1 and 2 were dropped
        self.assertEqual(pkt.frame_index, 3)
        status = src.status()
        self.assertEqual(status.frames_dropped, 2)
        src.close()

    def test_stale_frame_eviction_and_status_reporting(self) -> None:
        clock = [100.0]
        frames = [_make_frame(160, 120, 10), _make_frame(160, 120, 20)]
        fake_cap = FakeCapture(frames)
        cfg = CameraConfig(
            camera_id="cam_stale_test",
            source_type=SourceType.WEBCAM,
            source_uri=0,
            max_frame_queue_size=2,
            stale_frame_timeout_sec=2.0,
        )
        src = VideoSource(cfg, capture_factory=lambda _: fake_cap, clock_fn=lambda: clock[0])
        self.assertTrue(src.ingest_step())  # ingested at t=100.0
        clock[0] = 105.0  # 5s later (> stale_frame_timeout_sec=2.0)
        st = src.status()
        self.assertTrue(st.is_stale)
        self.assertEqual(st.processing_state, "stale")
        # Reading now evicts the stale buffered frame from t=100.0 and ingests fresh frame 2 at t=105.0
        pkt = src.read_next()
        self.assertIsNotNone(pkt)
        assert pkt is not None
        self.assertEqual(pkt.frame_index, 2)
        self.assertEqual(src.status().stale_frames_dropped, 1)
        src.close()

    def test_rtsp_disconnection_and_exponential_backoff_recovery(self) -> None:
        # Session 1 yields 1 frame then fails; reconnect attempt 1 fails to open; reconnect attempt 2 succeeds
        cap1 = FakeCapture([_make_frame(160, 120, 10), None], opened=True)
        cap_fail = FakeCapture([], opened=False)
        cap2 = FakeCapture([_make_frame(160, 120, 20)], opened=True)
        captures = [cap1, cap_fail, cap2]
        sleep_calls: List[float] = []

        def factory(_: CameraConfig) -> CaptureBackend:
            return captures.pop(0)

        cfg = CameraConfig(
            camera_id="cam_rtsp_01",
            source_type=SourceType.RTSP,
            source_uri="rtsp://user:secret@10.0.0.5:554/stream",
            reconnect_initial_delay_sec=0.5,
            reconnect_backoff_factor=2.0,
            reconnect_max_delay_sec=4.0,
            max_reconnect_attempts=3,
            resize_width=160,
            resize_height=120,
        )
        src = VideoSource(cfg, capture_factory=factory, sleep_fn=sleep_calls.append)

        pkt1 = src.read_next()
        self.assertIsNotNone(pkt1)
        assert pkt1 is not None
        self.assertEqual(pkt1.session_id, 1)

        # Next read encounters None on cap1 -> triggers reconnect 1 (fails) -> reconnect 2 (succeeds on cap2)
        pkt2 = src.read_next()
        self.assertIsNotNone(pkt2)
        assert pkt2 is not None
        self.assertEqual(pkt2.session_id, 2)
        self.assertEqual(sleep_calls, [0.5, 1.0])

        status = src.status()
        self.assertEqual(status.total_reconnections, 1)
        self.assertEqual(status.redacted_uri, "rtsp://***:***@10.0.0.5:554/stream")
        src.close()

    def test_rtsp_backoff_with_jitter(self) -> None:
        sleep_calls: List[float] = []
        cfg = CameraConfig(
            camera_id="cam_rtsp_jitter",
            source_type=SourceType.RTSP,
            source_uri="rtsp://10.0.0.99:554/jitter",
            reconnect_initial_delay_sec=1.0,
            reconnect_backoff_factor=2.0,
            reconnect_max_delay_sec=10.0,
            reconnect_jitter_ratio=0.20,
            max_reconnect_attempts=2,
        )
        src = VideoSource(
            cfg,
            capture_factory=lambda _: FakeCapture([], opened=False),
            sleep_fn=sleep_calls.append,
            jitter_fn=lambda low, high: high,  # deterministic positive jitter
        )
        src.read_next()
        self.assertEqual(len(sleep_calls), 2)
        self.assertAlmostEqual(sleep_calls[0], 1.2)
        self.assertAlmostEqual(sleep_calls[1], 2.4)

    def test_rtsp_exceeds_max_reconnect_attempts(self) -> None:
        sleep_calls: List[float] = []
        cfg = CameraConfig(
            camera_id="cam_rtsp_dead",
            source_type=SourceType.RTSP,
            source_uri="rtsp://10.0.0.99:554/dead",
            reconnect_initial_delay_sec=0.25,
            reconnect_backoff_factor=2.0,
            reconnect_max_delay_sec=1.0,
            max_reconnect_attempts=3,
        )
        src = VideoSource(
            cfg,
            capture_factory=lambda _: FakeCapture([], opened=False),
            sleep_fn=sleep_calls.append,
        )
        pkt = src.read_next()
        self.assertIsNone(pkt)
        self.assertEqual(sleep_calls, [0.25, 0.5, 1.0])
        self.assertEqual(src.status().state, StreamState.FAILED)

    def test_graceful_shutdown_stops_ingestion(self) -> None:
        stop_event = threading.Event()
        fake_cap = FakeCapture([_make_frame(160, 120, 1) for _ in range(10)])
        cfg = CameraConfig(
            camera_id="cam_shutdown",
            source_type=SourceType.WEBCAM,
            source_uri=0,
        )
        src = VideoSource(cfg, capture_factory=lambda _: fake_cap, stop_event=stop_event)
        pkt1 = src.read_next()
        self.assertIsNotNone(pkt1)
        src.close()
        self.assertTrue(stop_event.is_set())
        self.assertTrue(fake_cap.released)
        self.assertIsNone(src.read_next())
        self.assertEqual(src.status().state, StreamState.STOPPED)


if __name__ == "__main__":
    unittest.main()
