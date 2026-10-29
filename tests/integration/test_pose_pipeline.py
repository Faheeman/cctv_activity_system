"""Integration test for VideoSource + PoseEstimator + Renderer on prerecorded CCTV video."""

from pathlib import Path
import tempfile
import unittest

from activity_system.config import CameraConfig, PoseConfig, RendererConfig
from activity_system.pose_estimator import PoseEstimator
from activity_system.renderer import Renderer
from activity_system.schemas import ObservationQuality, SourceType
from activity_system.video_source import VideoSource
from tests.fixtures.video_generator import generate_synthetic_cctv_video


class TestPosePipelineIntegration(unittest.TestCase):
    def test_video_to_pose_and_skeleton_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "pose_test.avi"
            generate_synthetic_cctv_video(video_path, num_frames=12, width=640, height=480, fps=15.0)

            cam_cfg = CameraConfig(
                camera_id="cam_pose_01",
                source_type=SourceType.FILE,
                source_uri=str(video_path),
                resize_width=640,
                resize_height=480,
            )
            estimator = PoseEstimator(PoseConfig(backend="synthetic"))
            renderer = Renderer(RendererConfig())

            total_frames = 0
            frames_with_detections = 0
            with VideoSource(cam_cfg) as source:
                while True:
                    pkt = source.read_next()
                    if pkt is None:
                        break
                    total_frames += 1
                    dets = estimator.estimate(pkt)
                    if dets:
                        frames_with_detections += 1
                        self.assertEqual(len(dets[0].keypoints), 17)
                        self.assertEqual(dets[0].observation_quality, ObservationQuality.GOOD)
                    rendered = renderer.render(pkt.frame, detections=dets, camera_id="cam_pose_01", fps=15.0)
                    self.assertEqual(rendered.shape, (480, 640, 3))

            self.assertEqual(total_frames, 12)
            self.assertGreaterEqual(frames_with_detections, 10)
            self.assertGreater(estimator.mean_inference_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
