"""End-to-end application coordinator with thread-safe state snapshots, summaries, risk engine, and IURIS delivery."""

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from activity_system.activity_classifier import ActivityClassifier
from activity_system.camera_summary import CameraSummaryAggregator
from activity_system.config import AppConfig
from activity_system.event_store import BoundedEventStore
from activity_system.iuris.client import IurisDeliveryClient
from activity_system.logging_config import configure_logging
from activity_system.person_tracker import PersonTracker
from activity_system.pose_estimator import PoseBackend, PoseEstimator
from activity_system.renderer import Renderer
from activity_system.risk_engine import RiskAssessmentEngine
from activity_system.schemas import (
    ActivityEvent,
    ActivityObservationV1,
    CameraActivitySummaryV1,
    FramePacket,
    ObservationQuality,
    RiskEventV1,
    StreamState,
)
from activity_system.telemetry import (
    ApplicationSnapshot,
    StateSnapshotStore,
    SystemMetricsSnapshot,
    get_process_rss_mb,
)
from activity_system.track_state import TrackStateManager
from activity_system.video_source import (
    CaptureBackend,
    StreamManager,
    VideoSource,
    create_stream_adapter,
)


class ActivityApplication:
    """Coordinates multi-camera ingestion, pose inference, tracking, classification, summaries, risk engine, and IURIS delivery."""

    def __init__(
        self,
        config: AppConfig,
        pose_backend: Optional[PoseBackend] = None,
        capture_factory_by_camera: Optional[Dict[str, Callable[[], CaptureBackend]]] = None,
        iuris_client: Optional[IurisDeliveryClient] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.config = config
        self.logger = configure_logging(config.system.log_level)
        self.stop_event = stop_event or threading.Event()

        self.state_manager = TrackStateManager(config.tracker)
        self.pose_estimator = PoseEstimator(config.pose, backend=pose_backend)
        self.person_tracker = PersonTracker(config.tracker, state_manager=self.state_manager)
        self.activity_classifier = ActivityClassifier(config.activity)
        self.summary_aggregator = CameraSummaryAggregator(config.summary)
        self.risk_engine = RiskAssessmentEngine(config.risk)
        self.event_store = BoundedEventStore(config.storage)
        self.iuris_client = iuris_client or IurisDeliveryClient(
            config.iuris,
            start_background_worker=config.iuris.delivery_enabled and not config.iuris.dry_run,
            logger=self.logger,
        )
        self.renderer = Renderer(config.renderer)

        self.sources: Dict[str, VideoSource] = {}
        self._camera_names: Dict[str, Optional[str]] = {}
        custom_factories = capture_factory_by_camera or {}
        for cam_cfg in config.cameras:
            if not cam_cfg.enabled:
                continue
            self._camera_names[cam_cfg.camera_id] = cam_cfg.camera_name
            factory = None
            if cam_cfg.camera_id in custom_factories:
                user_fn = custom_factories[cam_cfg.camera_id]
                factory = lambda _c, fn=user_fn: fn()
            self.sources[cam_cfg.camera_id] = create_stream_adapter(
                cam_cfg,
                capture_factory=factory,
                stop_event=self.stop_event,
                logger=self.logger,
            )
        self.stream_manager = StreamManager(self.sources, logger=self.logger)

        self._start_wall_ts = time.perf_counter()
        self._start_cpu_ts = time.process_time()
        self._frames_processed = 0
        self._total_e2e_sec = 0.0
        self._low_quality_count = 0
        self._latest_events_by_camera: Dict[str, Dict[str, ActivityEvent]] = {
            cid: {} for cid in self.sources
        }
        self._latest_risk_events: List[RiskEventV1] = []

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        initial_metrics = SystemMetricsSnapshot(
            timestamp_utc=now_iso,
            active_cameras=len(self.sources),
            model_ready=self.pose_estimator.is_ready,
            pose_backend=self.pose_estimator.backend_name,
            memory_rss_mb=round(get_process_rss_mb(), 2),
        )
        self.snapshot_store = StateSnapshotStore(
            ApplicationSnapshot(
                app_name=config.system.app_name,
                healthy=True,
                ready=self.pose_estimator.is_ready,
                model_version=config.activity.model_version,
                timestamp_utc=now_iso,
                cameras=[src.status() for src in self.sources.values()],
                tracks=[],
                activities=[],
                activities_v1=[],
                camera_summaries=[],
                latest_risk_events=[],
                iuris_delivery=self.iuris_client.metrics(),
                metrics=initial_metrics,
            ),
            event_store=self.event_store,
            on_risk_status_change=self._handle_operator_risk_status_change,
        )

    def _handle_operator_risk_status_change(self, updated_event: RiskEventV1) -> None:
        """Enqueue operator acknowledgment or resolution updates to IURIS when delivery is enabled."""
        if self.config.iuris.delivery_enabled:
            self.iuris_client.enqueue(updated_event)
            if self.config.iuris.dry_run:
                self.iuris_client.flush()
        self._publish_snapshot()

    def step_camera(
        self, camera_id: str
    ) -> Optional[Tuple[FramePacket, np.ndarray, List[ActivityEvent]]]:
        """Process the next available sampled frame for a single camera through the full pipeline."""
        if self.stop_event.is_set():
            return None
        source = self.sources.get(camera_id)
        if source is None:
            raise KeyError(f"Unknown camera_id: {camera_id}")

        t0 = time.perf_counter()
        packet = self.stream_manager.read_next_safe(camera_id)
        if packet is None:
            self._publish_snapshot()
            return None

        detections = self.pose_estimator.estimate(packet)
        for det in detections:
            if det.observation_quality == ObservationQuality.POOR:
                self._low_quality_count += 1

        tracked_dets, active_tracks = self.person_tracker.update(packet, detections)

        events: List[ActivityEvent] = []
        v1_observations: List[ActivityObservationV1] = []
        events_by_track: Dict[str, ActivityEvent] = {}
        for trk in active_tracks:
            ev = self.activity_classifier.classify_track(trk)
            events.append(ev)
            events_by_track[trk.track_id] = ev
            v1_obs = ev.to_v1()
            v1_observations.append(v1_obs)
            self.event_store.add_activity(v1_obs)
            if self.config.iuris.delivery_enabled:
                self.iuris_client.enqueue(v1_obs)

        self._latest_events_by_camera[camera_id] = events_by_track

        # Stage: Camera-Level Activity Summary
        cam_summary: CameraActivitySummaryV1 = self.summary_aggregator.update(
            camera_id=camera_id,
            monotonic_ts=packet.monotonic_ts,
            utc_iso=packet.utc_iso,
            events=events,
        )
        self.event_store.add_summary(cam_summary)
        if self.config.iuris.delivery_enabled:
            self.iuris_client.enqueue(cam_summary)

        # Stage: Independent Risk Assessment Engine
        risk_events = self.risk_engine.evaluate(
            camera_id=camera_id,
            monotonic_ts=packet.monotonic_ts,
            utc_iso=packet.utc_iso,
            activities=v1_observations,
            summary=cam_summary,
            model_version=self.config.activity.model_version,
            session_id=packet.session_id,
        )
        cam_name = self._camera_names.get(camera_id)
        for rev in risk_events:
            if cam_name and not rev.camera_name:
                rev.camera_name = cam_name
            self.event_store.add_risk_event(rev)
            if self.config.iuris.delivery_enabled:
                self.iuris_client.enqueue(rev)
        if risk_events:
            self._latest_risk_events = (risk_events + self._latest_risk_events)[:50]

        if self.config.iuris.delivery_enabled and self.config.iuris.dry_run:
            self.iuris_client.flush()

        e2e_sec = time.perf_counter() - t0
        self._frames_processed += 1
        self._total_e2e_sec += e2e_sec
        instant_fps = 1.0 / max(1e-6, e2e_sec)

        cam_status = source.status()
        annotated_frame = self.renderer.render(
            packet.frame,
            detections=tracked_dets,
            events_by_track=events_by_track,
            fps=instant_fps,
            camera_id=camera_id,
            stream_state=cam_status.state.value,
            copy=True,
        )

        ok_jpg, jpg_buf = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        jpeg_bytes = jpg_buf.tobytes() if ok_jpg else None

        self._publish_snapshot(camera_id=camera_id, jpeg_bytes=jpeg_bytes)
        return packet, annotated_frame, events

    def run(
        self,
        max_frames_total: Optional[int] = None,
        output_video_path: Optional[Path] = None,
        events_jsonl_path: Optional[Path] = None,
        pace_fps: Optional[float] = None,
    ) -> ApplicationSnapshot:
        """Run round-robin processing across all enabled cameras until completion or max_frames_total."""
        writer: Optional[cv2.VideoWriter] = None
        jsonl_fh = None

        try:
            if events_jsonl_path is not None:
                events_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                jsonl_fh = events_jsonl_path.open("w", encoding="utf-8")

            camera_ids = list(self.sources.keys())
            if not camera_ids:
                return self.snapshot_store.get()

            while not self.stop_event.is_set():
                loop_t0 = time.perf_counter()
                any_frame_read = False
                for cam_id in camera_ids:
                    if self.stop_event.is_set():
                        break
                    src_state = self.sources[cam_id].status().state
                    if src_state in (StreamState.EOF, StreamState.FAILED, StreamState.STOPPED):
                        continue

                    step_res = self.step_camera(cam_id)
                    if step_res is not None:
                        any_frame_read = True
                        packet, annotated, events = step_res

                        if output_video_path is not None:
                            if writer is None:
                                output_video_path.parent.mkdir(parents=True, exist_ok=True)
                                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                                writer = cv2.VideoWriter(
                                    str(output_video_path),
                                    fourcc,
                                    15.0,
                                    (packet.width, packet.height),
                                )
                            writer.write(annotated)

                        if jsonl_fh is not None:
                            for ev in events:
                                jsonl_fh.write(json.dumps(ev.dict()) + "\n")

                        if max_frames_total is not None and self._frames_processed >= max_frames_total:
                            return self.snapshot_store.get()

                all_terminal = all(
                    src.status().state in (StreamState.EOF, StreamState.FAILED, StreamState.STOPPED)
                    for src in self.sources.values()
                )
                if all_terminal or not any_frame_read:
                    break

                if pace_fps and pace_fps > 0:
                    target_dt = 1.0 / float(pace_fps)
                    elapsed = time.perf_counter() - loop_t0
                    if elapsed < target_dt:
                        time.sleep(target_dt - elapsed)

            return self.snapshot_store.get()
        finally:
            if writer is not None:
                writer.release()
            if jsonl_fh is not None:
                jsonl_fh.close()
            self.shutdown()

    def _publish_snapshot(
        self,
        camera_id: Optional[str] = None,
        jpeg_bytes: Optional[bytes] = None,
    ) -> None:
        now_perf = time.perf_counter()
        now_cpu = time.process_time()
        uptime = max(1e-6, now_perf - self._start_wall_ts)
        cpu_elapsed = max(0.0, now_cpu - self._start_cpu_ts)
        cpu_pct = min(100.0 * 2.0, (cpu_elapsed / uptime) * 100.0)

        cam_snapshots = []
        for cid, src in self.sources.items():
            cs = src.status()
            cs.camera_name = self._camera_names.get(cid)
            cam_snapshots.append(cs)

        total_received = sum(c.frames_received for c in cam_snapshots)
        total_dropped = sum(c.frames_dropped for c in cam_snapshots)
        total_skipped = sum(c.frames_skipped_sampling for c in cam_snapshots)
        total_reconnects = sum(c.total_reconnections for c in cam_snapshots)

        track_snapshots = self.state_manager.get_all_active_snapshots(include_tentative=False)

        total_transitions = 0
        for cam_id in self.sources:
            for trk in self.state_manager.get_camera_tracks(cam_id, include_tentative=True):
                total_transitions += trk.transition_count

        active_track_ids = {t.track_id for t in track_snapshots}
        current_activities: List[ActivityEvent] = []
        current_activities_v1: List[ActivityObservationV1] = []
        for cam_id in sorted(self._latest_events_by_camera.keys()):
            for tid, ev in self._latest_events_by_camera[cam_id].items():
                if tid in active_track_ids:
                    current_activities.append(ev)
                    current_activities_v1.append(ev.to_v1())

        mean_e2e_ms = (
            (self._total_e2e_sec / self._frames_processed) * 1000.0
            if self._frames_processed > 0
            else 0.0
        )
        throughput_fps = (
            float(self._frames_processed) / uptime if self._frames_processed > 0 else 0.0
        )

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        any_failed = any(c.state == StreamState.FAILED for c in cam_snapshots)
        all_failed = bool(cam_snapshots) and all(c.state == StreamState.FAILED for c in cam_snapshots)
        connected_cams = sum(1 for c in cam_snapshots if c.state == StreamState.CONNECTED)
        disconnected_cams = sum(
            1
            for c in cam_snapshots
            if c.state in (StreamState.DISCONNECTED, StreamState.RECONNECTING, StreamState.FAILED)
        )

        iuris_metrics = self.iuris_client.metrics()
        is_ready = bool(self.pose_estimator.is_ready and not all_failed)
        active_risks = self.event_store.count_active_risk_events()

        metrics = SystemMetricsSnapshot(
            timestamp_utc=now_iso,
            uptime_seconds=round(uptime, 3),
            frames_received=total_received,
            frames_processed=self._frames_processed,
            frames_dropped=total_dropped,
            frames_skipped_sampling=total_skipped,
            mean_inference_ms=round(self.pose_estimator.mean_inference_ms, 3),
            mean_classification_ms=round(self.activity_classifier.mean_classification_ms, 3),
            mean_end_to_end_ms=round(mean_e2e_ms, 3),
            throughput_fps=round(throughput_fps, 2),
            active_cameras=connected_cams,
            connected_cameras=connected_cams,
            disconnected_cameras=disconnected_cams,
            active_tracks=len(track_snapshots),
            stream_reconnect_count=total_reconnects,
            classification_transitions=total_transitions,
            low_quality_observations=self._low_quality_count,
            risk_events_generated=self.risk_engine.total_risk_events_generated,
            active_risk_events=active_risks,
            duplicate_alerts_suppressed=self.risk_engine.duplicates_suppressed,
            event_queue_depth=iuris_metrics.queue_depth,
            iuris_deliveries_attempted=iuris_metrics.events_attempted,
            events_delivered=iuris_metrics.events_delivered,
            event_delivery_failures=iuris_metrics.events_failed,
            iuris_retries_total=iuris_metrics.events_retried,
            mean_delivery_latency_ms=iuris_metrics.mean_delivery_latency_ms,
            model_ready=self.pose_estimator.is_ready,
            pose_backend=self.pose_estimator.backend_name,
            cpu_utilization_pct=round(cpu_pct, 2),
            gpu_utilization_pct=None,
            memory_rss_mb=round(get_process_rss_mb(), 2),
        )

        snapshot = ApplicationSnapshot(
            app_name=self.config.system.app_name,
            healthy=not all_failed and not any_failed,
            ready=is_ready,
            model_version=self.config.activity.model_version,
            timestamp_utc=now_iso,
            cameras=cam_snapshots,
            tracks=track_snapshots,
            activities=current_activities,
            activities_v1=current_activities_v1,
            camera_summaries=self.summary_aggregator.get_latest(),
            latest_risk_events=list(self._latest_risk_events),
            iuris_delivery=iuris_metrics,
            metrics=metrics,
        )
        self.snapshot_store.publish(snapshot, camera_id=camera_id, jpeg_bytes=jpeg_bytes)

    def shutdown(self) -> None:
        """Signal shutdown, close stream adapters, flush/close IURIS client, and close event store."""
        self.stop_event.set()
        self.stream_manager.close_all()
        self.iuris_client.close(drain_remaining=True)
        self.event_store.close()
        self._publish_snapshot()
