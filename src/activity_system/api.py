"""Standalone FastAPI service exposing versioned /v1 REST endpoints, risk alert workflows, IURIS delivery health, and metrics."""

import time
from typing import Iterator, List, Literal, Optional
from fastapi import Body, FastAPI, HTTPException, Path as FastAPIPath, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from activity_system.iuris.schemas import IurisDeliveryMetrics
from activity_system.schemas import (
    ActivityCode,
    ActivityEvent,
    ActivityLabel,
    ActivityObservationV1,
    CameraActivitySummaryV1,
    CameraStatusSnapshot,
    RiskEventV1,
    RiskSeverity,
    RiskStatus,
    StreamState,
    TrackLifecycle,
)
from activity_system.telemetry import (
    ApplicationSnapshot,
    StateSnapshotStore,
    SystemMetricsSnapshot,
    format_prometheus_metrics,
)
from activity_system.track_state import TrackPublicSnapshot


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CCTV Pose, Activity &amp; IURIS Risk Operations Center</title>
  <style>
    :root { --bg: #0f172a; --panel: #1e293b; --accent: #38bdf8; --ok: #22c55e; --warn: #f59e0b; --danger: #ef4444; --text: #f8fafc; --muted: #94a3b8; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); }
    header { padding: 16px 24px; background: #020617; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }
    header h1 { margin: 0; font-size: 1.2rem; color: var(--accent); }
    .links a { color: var(--muted); margin-left: 14px; text-decoration: none; font-size: 0.9rem; }
    .links a:hover { color: var(--accent); }
    main { max-width: 1280px; margin: 20px auto; padding: 0 20px; display: grid; grid-template-columns: 1.15fr 1fr; gap: 20px; }
    .card { background: var(--panel); border-radius: 8px; padding: 16px; border: 1px solid #334155; }
    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .card h2 { margin: 0; font-size: 1rem; color: var(--accent); }
    select#cam-select { background: #0f172a; color: var(--text); border: 1px solid #475569; border-radius: 4px; padding: 4px 8px; font-size: 0.85rem; }
    .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }
    .kpi { background: #0f172a; padding: 10px; border-radius: 6px; text-align: center; }
    .kpi .val { font-size: 1.3rem; font-weight: 700; color: var(--ok); }
    .kpi .lbl { font-size: 0.75rem; color: var(--muted); }
    img#live-feed { width: 100%; border-radius: 6px; background: #000; border: 1px solid #475569; min-height: 320px; object-fit: contain; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #334155; }
    th { color: var(--muted); }
  </style>
</head>
<body>
  <header>
    <h1>Real-Time CCTV Activity &amp; IURIS Risk Intelligence</h1>
    <div class="links">
      <a href="/stream.mjpg" target="_blank">/stream.mjpg</a>
      <a href="/docs" target="_blank">OpenAPI (/docs)</a>
      <a href="/health" target="_blank">/health</a>
      <a href="/ready" target="_blank">/ready</a>
      <a href="/v1/cameras" target="_blank">/v1/cameras</a>
      <a href="/v1/tracks" target="_blank">/v1/tracks</a>
      <a href="/v1/activities" target="_blank">/v1/activities</a>
      <a href="/v1/camera-summaries" target="_blank">/v1/camera-summaries</a>
      <a href="/v1/risk-events" target="_blank">/v1/risk-events</a>
      <a href="/v1/risk-events/active" target="_blank">/v1/risk-events/active</a>
      <a href="/v1/iuris/status" target="_blank">/v1/iuris/status</a>
      <a href="/metrics" target="_blank">/metrics</a>
    </div>
  </header>
  <main>
    <div class="card">
      <div class="card-header">
        <h2>Live Annotated Camera Stream (MJPEG + 17-Keypoint Pose + ByteTrack)</h2>
        <select id="cam-select" title="Select Camera Stream"><option value="">Default Camera</option></select>
      </div>
      <img id="live-feed" src="/stream.mjpg" alt="Annotated Live Camera Stream">
    </div>
    <div>
      <div class="kpis">
        <div class="kpi"><div class="val" id="kpi-fps">--</div><div class="lbl">Throughput FPS</div></div>
        <div class="kpi"><div class="val" id="kpi-tracks">--</div><div class="lbl">Active Tracks</div></div>
        <div class="kpi"><div class="val" id="kpi-frames">--</div><div class="lbl">Frames Processed</div></div>
        <div class="kpi"><div class="val" id="kpi-mem">--</div><div class="lbl">Memory (MiB)</div></div>
      </div>
      <div class="card" style="margin-bottom: 16px;">
        <h2>Live Classified Activities</h2>
        <table>
          <thead><tr><th>Camera</th><th>Track ID</th><th>Activity</th><th>Quality</th><th>Evidence</th></tr></thead>
          <tbody id="activities-body"><tr><td colspan="5">Loading...</td></tr></tbody>
        </table>
      </div>
      <div class="card">
        <h2>Camera Stream Telemetry</h2>
        <table>
          <thead><tr><th>Camera ID</th><th>Type</th><th>State</th><th>Emitted</th><th>Dropped</th></tr></thead>
          <tbody id="cameras-body"><tr><td colspan="5">Loading...</td></tr></tbody>
        </table>
      </div>
    </div>
  </main>
  <script>
    const camSelect = document.getElementById('cam-select');
    const liveFeed = document.getElementById('live-feed');
    let knownCamIds = [];

    camSelect.addEventListener('change', () => {
      const cid = camSelect.value;
      liveFeed.src = cid ? `/stream.mjpg?camera_id=${encodeURIComponent(cid)}` : '/stream.mjpg';
    });

    async function refreshState() {
      try {
        const [mRes, aRes, cRes] = await Promise.all([
          fetch('/metrics'), fetch('/activities'), fetch('/cameras')
        ]);
        const mData = await mRes.json();
        const aData = await aRes.json();
        const cData = await cRes.json();
        const m = mData.metrics;
        document.getElementById('kpi-fps').textContent = m.throughput_fps.toFixed(1);
        document.getElementById('kpi-tracks').textContent = m.active_tracks;
        document.getElementById('kpi-frames').textContent = m.frames_processed;
        document.getElementById('kpi-mem').textContent = m.memory_rss_mb.toFixed(0);

        const cIds = cData.cameras.map(c => c.camera_id);
        if (JSON.stringify(cIds) !== JSON.stringify(knownCamIds)) {
          knownCamIds = cIds;
          const currentVal = camSelect.value;
          camSelect.innerHTML = cIds.map(id => `<option value="${id}">${id}</option>`).join('');
          if (currentVal && cIds.includes(currentVal)) camSelect.value = currentVal;
        }

        const aRows = aData.activities.map(a =>
          `<tr><td>${a.camera_id}</td><td>${a.track_id}</td><td><strong>${a.activity.toUpperCase()}</strong></td><td>${a.observation_quality}</td><td>${a.evidence_score.toFixed(2)}</td></tr>`
        ).join('');
        document.getElementById('activities-body').innerHTML = aRows || '<tr><td colspan="5">No active tracks in current frame</td></tr>';

        const cRows = cData.cameras.map(c =>
          `<tr><td>${c.camera_id}</td><td>${c.source_type}</td><td>${c.state}</td><td>${c.frames_emitted}</td><td>${c.frames_dropped}</td></tr>`
        ).join('');
        document.getElementById('cameras-body').innerHTML = cRows;
      } catch (e) {}
    }
    setInterval(refreshState, 500);
    refreshState();
  </script>
</body>
</html>
"""


class HealthResponse(BaseModel):
    schema_version: str = "1.0.0"
    app_name: str
    status: str = Field(..., description="Process liveness status ('ok' or 'degraded')")
    healthy: bool
    timestamp_utc: str
    uptime_seconds: float
    active_cameras: int
    active_tracks: int


class ReadyResponse(BaseModel):
    schema_version: str = "1.0"
    app_name: str
    status: str = Field(..., description="Readiness status ('ready' or 'not_ready')")
    ready: bool
    model_ready: bool
    pose_backend: str
    model_version: str
    cameras_configured: int
    active_cameras: int
    timestamp_utc: str


class PaginationMeta(BaseModel):
    total: int
    count: int
    limit: int
    offset: int


class CamerasResponse(BaseModel):
    schema_version: str = "1.0.0"
    timestamp_utc: str
    count: int
    cameras: List[CameraStatusSnapshot]


class CamerasV1Response(BaseModel):
    schema_version: str = "1.0"
    timestamp_utc: str
    pagination: PaginationMeta
    cameras: List[CameraStatusSnapshot]


class TracksResponse(BaseModel):
    schema_version: str = "1.0.0"
    timestamp_utc: str
    count: int
    tracks: List[TrackPublicSnapshot]


class TracksV1Response(BaseModel):
    schema_version: str = "1.0"
    timestamp_utc: str
    state_source: Literal["live_in_memory"] = "live_in_memory"
    pagination: PaginationMeta
    tracks: List[TrackPublicSnapshot]


class ActivitiesResponse(BaseModel):
    schema_version: str = "1.0.0"
    timestamp_utc: str
    count: int
    activities: List[ActivityEvent]


class ActivitiesV1Response(BaseModel):
    schema_version: str = "1.0"
    timestamp_utc: str
    state_source: Literal["live_in_memory", "event_history"]
    pagination: PaginationMeta
    activities: List[ActivityObservationV1]


class CameraSummariesV1Response(BaseModel):
    schema_version: str = "1.0"
    timestamp_utc: str
    state_source: Literal["live_in_memory", "event_history"]
    pagination: PaginationMeta
    summaries: List[CameraActivitySummaryV1]


class RiskEventsV1Response(BaseModel):
    schema_version: str = "1.0"
    timestamp_utc: str
    state_source: Literal["event_history"] = "event_history"
    pagination: PaginationMeta
    risk_events: List[RiskEventV1]


class RiskStatusUpdateRequest(BaseModel):
    actor: str = Field(default="security_operator", min_length=1, max_length=128)
    note: Optional[str] = Field(default=None, max_length=1000)


class IurisStatusV1Response(BaseModel):
    schema_version: str = "1.0"
    timestamp_utc: str
    status: Literal["disabled", "healthy", "degraded", "dry_run"]
    iuris_delivery: IurisDeliveryMetrics


class MetricsResponse(BaseModel):
    schema_version: str = "1.0.0"
    timestamp_utc: str
    metrics: SystemMetricsSnapshot
    iuris_delivery: Optional[IurisDeliveryMetrics] = None


def create_api_app(snapshot_store: StateSnapshotStore) -> FastAPI:
    """Create the standalone FastAPI application exposing v1 and legacy endpoints."""
    api = FastAPI(
        title="Standalone Real-Time CCTV Activity & Risk Intelligence Backend",
        version="1.0.0",
        description=(
            "Production-oriented AI backend for person detection, multi-object tracking, temporal activity "
            "recognition, camera-level activity summaries, configurable risk assessment, and IURIS-compatible "
            "alert delivery. Never exposes raw unannotated frames, biometric identities, or RTSP credentials."
        ),
    )

    @api.exception_handler(RuntimeError)
    async def runtime_exception_handler(_request: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": f"State snapshot temporarily unavailable: {str(exc)}"},
        )

    def _get_snapshot() -> ApplicationSnapshot:
        snap = snapshot_store.get()
        if snap is None:
            raise RuntimeError("Application snapshot store is not initialized.")
        return snap

    def _verify_camera_exists(snap: ApplicationSnapshot, camera_id: str) -> None:
        known_ids = {c.camera_id for c in snap.cameras}
        if camera_id not in known_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{camera_id}' is not registered. Known cameras: {sorted(known_ids)}",
            )

    @api.get(
        "/",
        response_class=HTMLResponse,
        summary="Live CCTV Pose & Activity Operations Dashboard",
        tags=["Dashboard"],
    )
    def get_dashboard() -> HTMLResponse:
        return HTMLResponse(content=_DASHBOARD_HTML)

    @api.get("/favicon.ico", include_in_schema=False)
    def get_favicon() -> Response:
        return Response(status_code=204)

    @api.get(
        "/frame.jpg",
        summary="Get latest annotated CCTV preview frame (JPEG)",
        tags=["Dashboard"],
    )
    def get_latest_frame_jpeg(
        camera_id: Optional[str] = Query(default=None, description="Optional camera_id")
    ) -> Response:
        jpg = snapshot_store.get_latest_jpeg(camera_id=camera_id)
        if jpg is None:
            return Response(status_code=204)
        return Response(content=jpg, media_type="image/jpeg")

    @api.get(
        "/stream.mjpg",
        summary="Continuous multipart MJPEG live video stream of annotated frames",
        tags=["Dashboard"],
    )
    def get_mjpeg_stream(
        camera_id: Optional[str] = Query(default=None, description="Optional camera_id"),
        max_chunks: Optional[int] = Query(
            default=None, ge=1, le=10000, description="Optional limit on MJPEG frames (useful for tests)"
        ),
    ) -> StreamingResponse:
        snap = _get_snapshot()
        if camera_id is not None:
            _verify_camera_exists(snap, camera_id)

        def _frame_generator() -> Iterator[bytes]:
            emitted = 0
            while True:
                jpg = snapshot_store.get_latest_jpeg(camera_id=camera_id)
                if jpg is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpg)).encode("ascii") + b"\r\n\r\n" + jpg + b"\r\n"
                    )
                    emitted += 1
                    if max_chunks is not None and emitted >= max_chunks:
                        break
                elif max_chunks is not None:
                    break
                time.sleep(0.065)

        return StreamingResponse(
            _frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @api.get(
        "/health",
        response_model=HealthResponse,
        summary="Process liveness check",
        tags=["Observability"],
    )
    def get_health() -> HealthResponse:
        snap = _get_snapshot()
        return HealthResponse(
            schema_version=snap.schema_version,
            app_name=snap.app_name,
            status="ok" if snap.healthy else "degraded",
            healthy=snap.healthy,
            timestamp_utc=snap.timestamp_utc,
            uptime_seconds=snap.metrics.uptime_seconds,
            active_cameras=snap.metrics.active_cameras,
            active_tracks=snap.metrics.active_tracks,
        )

    @api.get(
        "/ready",
        response_model=ReadyResponse,
        summary="Readiness of required models and services",
        tags=["Observability"],
    )
    def get_ready(response: Response) -> ReadyResponse:
        snap = _get_snapshot()
        is_ready = bool(snap.ready and snap.metrics.model_ready)
        if not is_ready:
            response.status_code = 503
        return ReadyResponse(
            schema_version="1.0",
            app_name=snap.app_name,
            status="ready" if is_ready else "not_ready",
            ready=is_ready,
            model_ready=snap.metrics.model_ready,
            pose_backend=snap.metrics.pose_backend,
            model_version=snap.model_version,
            cameras_configured=len(snap.cameras),
            active_cameras=snap.metrics.active_cameras,
            timestamp_utc=snap.timestamp_utc,
        )

    # -------------------------------------------------------------------------
    # Versioned /v1 REST API Endpoints
    # -------------------------------------------------------------------------

    @api.get(
        "/v1/cameras",
        response_model=CamerasV1Response,
        summary="Camera list, connection status, frame timestamps, dimensions, and processing state",
        tags=["v1 API"],
    )
    def get_v1_cameras(
        camera_id: Optional[str] = Query(default=None, description="Filter by camera_id"),
        state: Optional[StreamState] = Query(default=None, description="Filter by connection state"),
        limit: int = Query(default=50, ge=1, le=500, description="Pagination limit"),
        offset: int = Query(default=0, ge=0, description="Pagination offset"),
    ) -> CamerasV1Response:
        snap = _get_snapshot()
        cameras = snap.cameras
        if camera_id is not None:
            _verify_camera_exists(snap, camera_id)
            cameras = [c for c in cameras if c.camera_id == camera_id]
        if state is not None:
            cameras = [c for c in cameras if c.state == state]
        total = len(cameras)
        paged = cameras[offset : offset + limit]
        return CamerasV1Response(
            schema_version="1.0",
            timestamp_utc=snap.timestamp_utc,
            pagination=PaginationMeta(total=total, count=len(paged), limit=limit, offset=offset),
            cameras=paged,
        )

    @api.get(
        "/v1/tracks",
        response_model=TracksV1Response,
        summary="Currently active tracked people across cameras",
        tags=["v1 API"],
    )
    def get_v1_tracks(
        camera_id: Optional[str] = Query(default=None, description="Filter by camera_id"),
        lifecycle: Optional[TrackLifecycle] = Query(default=None, description="Filter by track lifecycle"),
        limit: int = Query(default=50, ge=1, le=500, description="Pagination limit"),
        offset: int = Query(default=0, ge=0, description="Pagination offset"),
    ) -> TracksV1Response:
        snap = _get_snapshot()
        if camera_id is not None:
            _verify_camera_exists(snap, camera_id)
        tracks = snap.tracks
        if camera_id is not None:
            tracks = [t for t in tracks if t.camera_id == camera_id]
        if lifecycle is not None:
            tracks = [t for t in tracks if t.lifecycle == lifecycle]
        total = len(tracks)
        paged = tracks[offset : offset + limit]
        return TracksV1Response(
            schema_version="1.0",
            timestamp_utc=snap.timestamp_utc,
            state_source="live_in_memory",
            pagination=PaginationMeta(total=total, count=len(paged), limit=limit, offset=offset),
            tracks=paged,
        )

    @api.get(
        "/v1/activities",
        response_model=ActivitiesV1Response,
        summary="Latest activity observations (live in-memory state or bounded event history)",
        tags=["v1 API"],
    )
    def get_v1_activities(
        camera_id: Optional[str] = Query(default=None, description="Filter by camera_id"),
        activity: Optional[ActivityCode] = Query(
            default=None, description="Filter by canonical activity label (STANDING, WALKING, RUNNING, SITTING, LYING, FALLING, UNKNOWN)"
        ),
        track_id: Optional[str] = Query(default=None, description="Filter by track_id"),
        since: Optional[str] = Query(default=None, description="Filter events with UTC ISO timestamp >= since"),
        source: Literal["live", "history"] = Query(
            default="live",
            description="Select 'live' for current active track observations or 'history' for bounded historical event log",
        ),
        limit: int = Query(default=50, ge=1, le=500, description="Pagination limit"),
        offset: int = Query(default=0, ge=0, description="Pagination offset"),
    ) -> ActivitiesV1Response:
        snap = _get_snapshot()
        if camera_id is not None:
            _verify_camera_exists(snap, camera_id)

        if source == "history":
            paged, total = snapshot_store.event_store.query_activities(
                camera_id=camera_id,
                activity=activity,
                track_id=track_id,
                since=since,
                limit=limit,
                offset=offset,
            )
            return ActivitiesV1Response(
                schema_version="1.0",
                timestamp_utc=snap.timestamp_utc,
                state_source="event_history",
                pagination=PaginationMeta(total=total, count=len(paged), limit=limit, offset=offset),
                activities=paged,
            )

        items = snap.activities_v1
        if camera_id is not None:
            items = [a for a in items if a.camera_id == camera_id]
        if activity is not None:
            items = [a for a in items if a.activity == activity]
        if track_id is not None:
            items = [a for a in items if a.track_id == track_id]
        if since is not None:
            items = [a for a in items if a.timestamp >= since]
        total = len(items)
        paged = items[offset : offset + limit]
        return ActivitiesV1Response(
            schema_version="1.0",
            timestamp_utc=snap.timestamp_utc,
            state_source="live_in_memory",
            pagination=PaginationMeta(total=total, count=len(paged), limit=limit, offset=offset),
            activities=paged,
        )

    @api.get(
        "/v1/camera-summaries",
        response_model=CameraSummariesV1Response,
        summary="Camera-level activity distributions and percentages over configurable windows",
        tags=["v1 API"],
    )
    def get_v1_camera_summaries(
        camera_id: Optional[str] = Query(default=None, description="Filter by camera_id"),
        since: Optional[str] = Query(default=None, description="Filter summaries with UTC ISO timestamp >= since"),
        source: Literal["live", "history"] = Query(
            default="live",
            description="Select 'live' for latest per-camera summary or 'history' for bounded window summary history",
        ),
        limit: int = Query(default=50, ge=1, le=500, description="Pagination limit"),
        offset: int = Query(default=0, ge=0, description="Pagination offset"),
    ) -> CameraSummariesV1Response:
        snap = _get_snapshot()
        if camera_id is not None:
            _verify_camera_exists(snap, camera_id)

        if source == "history":
            paged, total = snapshot_store.event_store.query_summaries(
                camera_id=camera_id,
                since=since,
                limit=limit,
                offset=offset,
            )
            return CameraSummariesV1Response(
                schema_version="1.0",
                timestamp_utc=snap.timestamp_utc,
                state_source="event_history",
                pagination=PaginationMeta(total=total, count=len(paged), limit=limit, offset=offset),
                summaries=paged,
            )

        items = snap.camera_summaries
        if camera_id is not None:
            items = [s for s in items if s.camera_id == camera_id]
        if since is not None:
            items = [s for s in items if s.timestamp >= since]
        total = len(items)
        paged = items[offset : offset + limit]
        return CameraSummariesV1Response(
            schema_version="1.0",
            timestamp_utc=snap.timestamp_utc,
            state_source="live_in_memory",
            pagination=PaginationMeta(total=total, count=len(paged), limit=limit, offset=offset),
            summaries=paged,
        )

    @api.get(
        "/v1/risk-events",
        response_model=RiskEventsV1Response,
        summary="Risk-event history generated by the independent RiskAssessmentEngine",
        tags=["v1 API"],
    )
    def get_v1_risk_events(
        camera_id: Optional[str] = Query(default=None, description="Filter by camera_id"),
        severity: Optional[RiskSeverity] = Query(
            default=None, description="Filter by operational risk severity (INFO, LOW, MEDIUM, HIGH, CRITICAL)"
        ),
        risk_type: Optional[str] = Query(default=None, description="Filter by risk_type"),
        rule_name: Optional[str] = Query(default=None, description="Filter by rule_name"),
        status: Optional[RiskStatus] = Query(default=None, description="Filter by risk event status"),
        active_only: bool = Query(default=False, description="Only return OPEN or ACKNOWLEDGED alerts"),
        since: Optional[str] = Query(default=None, description="Filter risk events with UTC ISO timestamp >= since"),
        until: Optional[str] = Query(default=None, description="Filter risk events with UTC ISO timestamp <= until"),
        limit: int = Query(default=50, ge=1, le=500, description="Pagination limit"),
        offset: int = Query(default=0, ge=0, description="Pagination offset"),
    ) -> RiskEventsV1Response:
        snap = _get_snapshot()
        if camera_id is not None:
            _verify_camera_exists(snap, camera_id)

        paged, total = snapshot_store.event_store.query_risk_events(
            camera_id=camera_id,
            severity=severity,
            risk_type=risk_type,
            rule_name=rule_name,
            status=status,
            active_only=active_only,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )
        return RiskEventsV1Response(
            schema_version="1.0",
            timestamp_utc=snap.timestamp_utc,
            state_source="event_history",
            pagination=PaginationMeta(total=total, count=len(paged), limit=limit, offset=offset),
            risk_events=paged,
        )

    @api.get(
        "/v1/risk-events/active",
        response_model=RiskEventsV1Response,
        summary="Currently active (OPEN or ACKNOWLEDGED) risk alerts",
        tags=["v1 API"],
    )
    def get_v1_active_risk_events(
        camera_id: Optional[str] = Query(default=None, description="Filter by camera_id"),
        severity: Optional[RiskSeverity] = Query(default=None, description="Filter by severity"),
        limit: int = Query(default=50, ge=1, le=500, description="Pagination limit"),
        offset: int = Query(default=0, ge=0, description="Pagination offset"),
    ) -> RiskEventsV1Response:
        snap = _get_snapshot()
        if camera_id is not None:
            _verify_camera_exists(snap, camera_id)

        paged, total = snapshot_store.event_store.query_risk_events(
            camera_id=camera_id,
            severity=severity,
            active_only=True,
            limit=limit,
            offset=offset,
        )
        return RiskEventsV1Response(
            schema_version="1.0",
            timestamp_utc=snap.timestamp_utc,
            state_source="event_history",
            pagination=PaginationMeta(total=total, count=len(paged), limit=limit, offset=offset),
            risk_events=paged,
        )

    @api.get(
        "/v1/risk-events/{event_id}",
        response_model=RiskEventV1,
        summary="Retrieve a specific risk alert by event_id including audit status history",
        tags=["v1 API"],
    )
    def get_v1_risk_event_by_id(
        event_id: str = FastAPIPath(..., description="Unique risk event_id")
    ) -> RiskEventV1:
        ev = snapshot_store.event_store.get_risk_event(event_id)
        if ev is None:
            raise HTTPException(status_code=404, detail=f"Risk event '{event_id}' not found.")
        return ev

    @api.post(
        "/v1/risk-events/{event_id}/acknowledge",
        response_model=RiskEventV1,
        summary="Acknowledge an active risk alert and record audit metadata",
        tags=["v1 API"],
    )
    def acknowledge_v1_risk_event(
        event_id: str = FastAPIPath(..., description="Unique risk event_id"),
        body: RiskStatusUpdateRequest = Body(default_factory=RiskStatusUpdateRequest),
    ) -> RiskEventV1:
        updated = snapshot_store.event_store.acknowledge_risk_event(
            event_id=event_id,
            actor=body.actor,
            note=body.note,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Risk event '{event_id}' not found.")
        if snapshot_store.on_risk_status_change is not None:
            snapshot_store.on_risk_status_change(updated)
        return updated

    @api.post(
        "/v1/risk-events/{event_id}/resolve",
        response_model=RiskEventV1,
        summary="Resolve a risk alert and record resolution note and audit metadata",
        tags=["v1 API"],
    )
    def resolve_v1_risk_event(
        event_id: str = FastAPIPath(..., description="Unique risk event_id"),
        body: RiskStatusUpdateRequest = Body(default_factory=RiskStatusUpdateRequest),
    ) -> RiskEventV1:
        updated = snapshot_store.event_store.resolve_risk_event(
            event_id=event_id,
            actor=body.actor,
            note=body.note,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Risk event '{event_id}' not found.")
        if snapshot_store.on_risk_status_change is not None:
            snapshot_store.on_risk_status_change(updated)
        return updated

    @api.get(
        "/v1/iuris/status",
        response_model=IurisStatusV1Response,
        summary="Check IURIS outbound delivery health, queue depth, latency, and error metrics",
        tags=["v1 API"],
    )
    def get_v1_iuris_status() -> IurisStatusV1Response:
        snap = _get_snapshot()
        d = snap.iuris_delivery
        if not d.delivery_enabled:
            st: Literal["disabled", "healthy", "degraded", "dry_run"] = "disabled"
        elif d.dry_run:
            st = "dry_run"
        elif d.healthy:
            st = "healthy"
        else:
            st = "degraded"
        return IurisStatusV1Response(
            schema_version="1.0",
            timestamp_utc=snap.timestamp_utc,
            status=st,
            iuris_delivery=d,
        )

    # -------------------------------------------------------------------------
    # Legacy Compatible Endpoints (Preserved for Existing Consumers & Tests)
    # -------------------------------------------------------------------------

    @api.get(
        "/cameras",
        response_model=CamerasResponse,
        summary="List configured camera stream statuses with redacted URIs",
        tags=["State"],
    )
    def get_cameras(
        camera_id: Optional[str] = Query(default=None, description="Optional camera_id filter")
    ) -> CamerasResponse:
        snap = _get_snapshot()
        cameras = snap.cameras
        if camera_id is not None:
            _verify_camera_exists(snap, camera_id)
            cameras = [c for c in cameras if c.camera_id == camera_id]
        return CamerasResponse(
            schema_version=snap.schema_version,
            timestamp_utc=snap.timestamp_utc,
            count=len(cameras),
            cameras=cameras,
        )

    @api.get(
        "/tracks",
        response_model=TracksResponse,
        summary="List active multi-person tracks across cameras",
        tags=["State"],
    )
    def get_tracks(
        camera_id: Optional[str] = Query(default=None, description="Filter by camera_id"),
        lifecycle: Optional[TrackLifecycle] = Query(
            default=None, description="Filter by track lifecycle state"
        ),
    ) -> TracksResponse:
        snap = _get_snapshot()
        if camera_id is not None:
            _verify_camera_exists(snap, camera_id)
        tracks = snap.tracks
        if camera_id is not None:
            tracks = [t for t in tracks if t.camera_id == camera_id]
        if lifecycle is not None:
            tracks = [t for t in tracks if t.lifecycle == lifecycle]
        return TracksResponse(
            schema_version=snap.schema_version,
            timestamp_utc=snap.timestamp_utc,
            count=len(tracks),
            tracks=tracks,
        )

    @api.get(
        "/activities",
        response_model=ActivitiesResponse,
        summary="List latest classified activity events for active tracks",
        tags=["Activity Recognition"],
    )
    def get_activities(
        camera_id: Optional[str] = Query(default=None, description="Filter by camera_id"),
        activity: Optional[ActivityLabel] = Query(
            default=None, description="Filter by activity label"
        ),
    ) -> ActivitiesResponse:
        snap = _get_snapshot()
        if camera_id is not None:
            _verify_camera_exists(snap, camera_id)
        activities = snap.activities
        if camera_id is not None:
            activities = [a for a in activities if a.camera_id == camera_id]
        if activity is not None:
            activities = [a for a in activities if a.activity == activity]
        return ActivitiesResponse(
            schema_version=snap.schema_version,
            timestamp_utc=snap.timestamp_utc,
            count=len(activities),
            activities=activities,
        )

    @api.get(
        "/metrics",
        response_model=MetricsResponse,
        summary="Get pipeline throughput, latency, resource usage, IURIS delivery, and reliability counters",
        tags=["Observability"],
    )
    def get_metrics() -> MetricsResponse:
        snap = _get_snapshot()
        return MetricsResponse(
            schema_version=snap.schema_version,
            timestamp_utc=snap.timestamp_utc,
            metrics=snap.metrics,
            iuris_delivery=snap.iuris_delivery,
        )

    @api.get(
        "/metrics/prometheus",
        response_class=PlainTextResponse,
        summary="Export operational telemetry in Prometheus exposition format",
        tags=["Observability"],
    )
    def get_prometheus_metrics() -> PlainTextResponse:
        snap = _get_snapshot()
        return PlainTextResponse(
            content=format_prometheus_metrics(snap), media_type="text/plain; version=0.0.4"
        )

    return api
