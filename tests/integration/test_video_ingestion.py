"""Integration tests for Phase 1 video ingestion using real OpenCV video files."""

from pathlib import Path
import tempfile
import unittest

from activity_system.config import CameraConfig, ConfigLoader
from activity_system.schemas import SourceType, StreamState
from activity_system.video_source import VideoSource
from tests.fixtures.video_generator import generate_synthetic_cctv_video


class TestVideoIngestionIntegration(unittest.TestCase):
    def test_end_to_end_opencv_video_file_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "sample_cctv.avi"
            generate_synthetic_cctv_video(video_path, num_frames=24, width=640, height=480, fps=15.0)
            self.assertTrue(video_path.exists())

            cfg = CameraConfig(
                camera_id="cam_corridor_01",
                source_type=SourceType.FILE,
                source_uri=str(video_path),
                sample_every_n_frames=2,
                resize_width=320,
                resize_height=240,
            )

            packets = []
            with VideoSource(cfg) as source:
                while True:
                    pkt = source.read_next()
                    if pkt is None:
                        break
                    packets.append(pkt)
                final_status = source.status()

            self.assertEqual(len(packets), 12)
            self.assertEqual(packets[0].width, 320)
            self.assertEqual(packets[0].height, 240)
            self.assertEqual(packets[0].frame_index, 1)
            self.assertEqual(packets[-1].frame_index, 23)
            self.assertEqual(final_status.state, StreamState.EOF)
            self.assertEqual(final_status.frames_received, 24)
            self.assertEqual(final_status.frames_emitted, 12)
            self.assertEqual(final_status.frames_skipped_sampling, 12)

    def test_default_yaml_fixture_video_ingestion(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "config" / "default.yaml"
        app_cfg = ConfigLoader.load(config_path)
        cam_cfg = app_cfg.cameras[0]

        fixture_video_path = project_root / str(cam_cfg.source_uri)
        generate_synthetic_cctv_video(fixture_video_path, num_frames=15, width=640, height=480, fps=15.0)
        cam_cfg.source_uri = str(fixture_video_path)

        source = VideoSource(cam_cfg)
        count = 0
        while True:
            pkt = source.read_next()
            if pkt is None:
                break
            count += 1
        source.close()

        self.assertEqual(count, 15)
        self.assertEqual(source.status().state, StreamState.EOF)


if __name__ == "__main__":
    unittest.main()
