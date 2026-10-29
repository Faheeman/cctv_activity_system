"""Authenticated, idempotent, per-camera fair HTTP delivery client for IURIS integration."""

from collections import deque
import json
import logging
from pathlib import Path
import threading
import time
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple, Union
import httpx

from activity_system.config import IurisConfig
from activity_system.iuris.mapper import IurisEventMapper
from activity_system.iuris.retry import compute_backoff_delay, is_retryable_status_code
from activity_system.iuris.schemas import (
    IurisDeliveryMetrics,
    IurisDeliveryReceipt,
    IurisOutboundEnvelope,
)
from activity_system.logging_config import redact_uri
from activity_system.schemas import (
    ActivityEvent,
    ActivityObservationV1,
    CameraActivitySummaryV1,
    RiskEventV1,
    utc_now_iso,
)


HttpSenderFn = Callable[[str, Dict[str, str], Dict[str, object], float], Tuple[int, str]]


def default_httpx_sender(
    url: str,
    headers: Dict[str, str],
    json_payload: Dict[str, object],
    timeout_sec: float,
) -> Tuple[int, str]:
    """Send an HTTP POST request via httpx with explicit timeout."""
    with httpx.Client(timeout=timeout_sec) as client:
        resp = client.post(url, headers=headers, json=json_payload)
        return int(resp.status_code), resp.text


class IurisDeliveryClient:
    """Optional outbound HTTP delivery client with fair per-camera scheduling, retries, idempotency, and outbox.

    Architectural guarantees:
    - Never blocks or crashes the computer-vision pipeline if IURIS is unreachable.
    - Uses per-camera sub-queues with round-robin draining and priority placement for `risk.detected`
      events so one camera's event flood cannot starve alerts from other cameras.
    - Deduplicates outbound events by `dedup_key` and attaches `Idempotency-Key` headers.
    - Supports `dry_run` mode for safe local testing without contacting external servers.
    - Writes overflowed or permanently failed envelopes to a JSONL dead-letter outbox.
    """

    def __init__(
        self,
        config: IurisConfig,
        http_sender: Optional[HttpSenderFn] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        jitter_fn: Optional[Callable[[float, float], float]] = None,
        start_background_worker: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self._http_sender = http_sender or default_httpx_sender
        self._sleep_fn = sleep_fn
        self._jitter_fn = jitter_fn
        self._logger = logger or logging.getLogger("activity_system")

        self._lock = threading.Lock()
        self._queues_by_camera: Dict[str, Deque[IurisOutboundEnvelope]] = {}
        self._camera_rr_order: List[str] = []
        self._rr_cursor: int = 0
        self._seen_ids: Deque[str] = deque(maxlen=max(1000, config.max_queue_size * 4))
        self._seen_id_set: Set[str] = set()
        self._dry_run_delivered: Deque[IurisOutboundEnvelope] = deque(maxlen=200)

        self._events_enqueued: int = 0
        self._events_attempted: int = 0
        self._events_delivered: int = 0
        self._events_failed: int = 0
        self._events_retried: int = 0
        self._duplicates_skipped: int = 0
        self._dead_letter_count: int = 0
        self._total_delivery_latency_ms: float = 0.0
        self._last_delivery_latency_ms: float = 0.0
        self._last_delivery_utc_iso: Optional[str] = None
        self._last_error: Optional[str] = None

        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        if start_background_worker and self.config.delivery_enabled:
            self.start_worker()

    @property
    def destination_url(self) -> str:
        base = self.config.base_url.strip().rstrip("/")
        if not base:
            return ""
        path = self.config.events_path.strip()
        if not path.startswith("/"):
            path = "/" + path
        return f"{base}{path}"

    @property
    def dry_run_records(self) -> List[IurisOutboundEnvelope]:
        with self._lock:
            return list(self._dry_run_delivered)

    def _total_queue_depth_locked(self) -> int:
        return sum(len(q) for q in self._queues_by_camera.values())

    def _evict_oldest_from_deepest_camera_locked(self) -> Optional[IurisOutboundEnvelope]:
        """Evict the oldest item from the camera with the largest queue depth to preserve fairness."""
        if not self._queues_by_camera:
            return None
        deepest_cam = max(
            self._queues_by_camera.keys(),
            key=lambda cid: len(self._queues_by_camera[cid]),
        )
        q = self._queues_by_camera[deepest_cam]
        if not q:
            return None
        return q.popleft()

    def _pop_next_fair_locked(self) -> Optional[IurisOutboundEnvelope]:
        """Pop next envelope in round-robin order across cameras."""
        active_cams = [cid for cid in self._camera_rr_order if self._queues_by_camera.get(cid)]
        if not active_cams:
            return None
        idx = self._rr_cursor % len(active_cams)
        cam_id = active_cams[idx]
        self._rr_cursor = (idx + 1) % len(active_cams)
        return self._queues_by_camera[cam_id].popleft()

    def _build_headers(self, envelope: IurisOutboundEnvelope) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Idempotency-Key": envelope.idempotency_key,
            "X-Event-Id": envelope.event_id,
            "X-Event-Type": envelope.event_type,
            "X-Schema-Version": envelope.schema_version,
        }
        token = self.config.api_token.strip()
        if token:
            if self.config.auth_scheme == "bearer":
                headers[self.config.auth_header_name] = f"Bearer {token}"
            else:
                header_name = (
                    self.config.auth_header_name
                    if self.config.auth_header_name != "Authorization"
                    else "X-API-Key"
                )
                headers[header_name] = token
        return headers

    def _write_dead_letter(self, envelope: IurisOutboundEnvelope, reason: str) -> None:
        with self._lock:
            self._dead_letter_count += 1
        if not self.config.dead_letter_path:
            return
        try:
            dl_path = Path(self.config.dead_letter_path)
            dl_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "dead_letter_timestamp": utc_now_iso(),
                "reason": redact_uri(reason),
                "envelope": envelope.dict(),
            }
            with dl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            self._logger.error(f"Failed writing to dead-letter outbox: {redact_uri(str(exc))}")

    def enqueue(
        self,
        item: Union[
            IurisOutboundEnvelope,
            ActivityEvent,
            ActivityObservationV1,
            CameraActivitySummaryV1,
            RiskEventV1,
        ],
    ) -> bool:
        """Convert and enqueue an event for outbound delivery with per-camera fair queueing."""
        if not self.config.delivery_enabled:
            return False

        if isinstance(item, IurisOutboundEnvelope):
            envelope = item
        elif isinstance(item, (ActivityEvent, ActivityObservationV1)):
            if not self.config.deliver_activity_events:
                return False
            envelope = IurisEventMapper.map_activity(item)
        elif isinstance(item, CameraActivitySummaryV1):
            if not self.config.deliver_summary_events:
                return False
            envelope = IurisEventMapper.map_camera_summary(item)
        elif isinstance(item, RiskEventV1):
            if not self.config.deliver_risk_events:
                return False
            envelope = IurisEventMapper.map_risk_event(item)
        else:
            raise TypeError(f"Unsupported outbound event type: {type(item)}")

        dedup_key = envelope.dedup_key or envelope.event_id
        overflow_to_dead_letter: Optional[Tuple[IurisOutboundEnvelope, str]] = None

        with self._lock:
            if dedup_key in self._seen_id_set:
                self._duplicates_skipped += 1
                return False

            if len(self._seen_ids) == self._seen_ids.maxlen:
                evicted_id = self._seen_ids.popleft()
                self._seen_id_set.discard(evicted_id)
            self._seen_ids.append(dedup_key)
            self._seen_id_set.add(dedup_key)

            cam_id = envelope.camera_id or "default"
            if cam_id not in self._queues_by_camera:
                self._queues_by_camera[cam_id] = deque()
                self._camera_rr_order.append(cam_id)
            cam_q = self._queues_by_camera[cam_id]

            if self._total_queue_depth_locked() >= self.config.max_queue_size:
                policy = self.config.backpressure_policy
                if policy == "drop_newest":
                    overflow_to_dead_letter = (
                        envelope,
                        f"Queue full ({self.config.max_queue_size}); policy=drop_newest",
                    )
                else:
                    oldest = self._evict_oldest_from_deepest_camera_locked()
                    if oldest is not None and policy == "dead_letter_and_drop_oldest":
                        overflow_to_dead_letter = (
                            oldest,
                            f"Queue full ({self.config.max_queue_size}); policy=dead_letter_and_drop_oldest",
                        )
                    if envelope.event_type == "risk.detected":
                        cam_q.appendleft(envelope)
                    else:
                        cam_q.append(envelope)
                    self._events_enqueued += 1
            else:
                if envelope.event_type == "risk.detected":
                    cam_q.appendleft(envelope)
                else:
                    cam_q.append(envelope)
                self._events_enqueued += 1

        if overflow_to_dead_letter is not None:
            env_dl, reason = overflow_to_dead_letter
            self._write_dead_letter(env_dl, reason)
            if self.config.backpressure_policy == "drop_newest":
                return False

        self._wake_event.set()
        return True

    def deliver_envelope(self, envelope: IurisOutboundEnvelope) -> IurisDeliveryReceipt:
        """Deliver a single envelope with exponential backoff retries, idempotency headers, or safe dry-run."""
        if self.config.dry_run:
            now_iso = utc_now_iso()
            with self._lock:
                self._events_attempted += 1
                self._events_delivered += 1
                self._last_delivery_utc_iso = now_iso
                self._last_delivery_latency_ms = 0.05
                self._total_delivery_latency_ms += 0.05
                self._last_error = None
                self._dry_run_delivered.append(envelope)
            return IurisDeliveryReceipt(
                event_id=envelope.event_id,
                event_type=envelope.event_type,
                idempotency_key=envelope.idempotency_key,
                delivered=True,
                attempts=1,
                status_code=200,
                latency_ms=0.05,
                dry_run=True,
            )

        dest_url = self.destination_url
        if not dest_url:
            err = "IURIS_BASE_URL is not configured while delivery is enabled."
            with self._lock:
                self._events_failed += 1
                self._last_error = err
            self._write_dead_letter(envelope, err)
            return IurisDeliveryReceipt(
                event_id=envelope.event_id,
                event_type=envelope.event_type,
                idempotency_key=envelope.idempotency_key,
                delivered=False,
                attempts=0,
                dead_lettered=True,
                error=err,
            )

        headers = self._build_headers(envelope)
        max_attempts = max(1, 1 + int(self.config.max_retry_attempts))
        last_status: Optional[int] = None
        last_err: Optional[str] = None
        t0_overall = time.perf_counter()

        for attempt in range(1, max_attempts + 1):
            if self._stop_event.is_set() and attempt > 1:
                break

            with self._lock:
                self._events_attempted += 1

            t_req = time.perf_counter()
            try:
                status_code, body_text = self._http_sender(
                    dest_url,
                    headers,
                    envelope.payload,
                    float(self.config.timeout_seconds),
                )
                req_ms = (time.perf_counter() - t_req) * 1000.0
                last_status = status_code
                if 200 <= status_code < 300 or status_code == 409:
                    now_iso = utc_now_iso()
                    with self._lock:
                        self._events_delivered += 1
                        self._last_delivery_utc_iso = now_iso
                        self._last_delivery_latency_ms = round(req_ms, 3)
                        self._total_delivery_latency_ms += req_ms
                        self._last_error = None
                    return IurisDeliveryReceipt(
                        event_id=envelope.event_id,
                        event_type=envelope.event_type,
                        idempotency_key=envelope.idempotency_key,
                        delivered=True,
                        attempts=attempt,
                        status_code=status_code,
                        latency_ms=round(req_ms, 3),
                    )

                last_err = f"HTTP {status_code}: {redact_uri(body_text[:200])}"
                if not is_retryable_status_code(status_code):
                    break
            except Exception as exc:
                last_err = f"Transport error: {redact_uri(str(exc))}"

            if attempt < max_attempts and not self._stop_event.is_set():
                with self._lock:
                    self._events_retried += 1
                delay = compute_backoff_delay(
                    attempt=attempt,
                    initial_delay_sec=self.config.retry_initial_delay_sec,
                    max_delay_sec=self.config.retry_max_delay_sec,
                    backoff_factor=self.config.retry_backoff_factor,
                    jitter_ratio=self.config.retry_jitter_ratio,
                    jitter_fn=self._jitter_fn,
                )
                if delay > 0:
                    self._sleep_fn(delay)

        elapsed_ms = round((time.perf_counter() - t0_overall) * 1000.0, 3)
        final_err = last_err or "Delivery failed after maximum retry attempts."
        with self._lock:
            self._events_failed += 1
            self._last_error = final_err
        self._logger.error(
            f"IURIS event delivery failed for {envelope.event_id} ({envelope.event_type}): {final_err}"
        )
        self._write_dead_letter(envelope, final_err)
        return IurisDeliveryReceipt(
            event_id=envelope.event_id,
            event_type=envelope.event_type,
            idempotency_key=envelope.idempotency_key,
            delivered=False,
            attempts=attempt,
            status_code=last_status,
            latency_ms=elapsed_ms,
            dead_lettered=True,
            error=final_err,
        )

    def flush(self, max_items: Optional[int] = None) -> List[IurisDeliveryReceipt]:
        """Synchronously drain queued events in round-robin camera order and return delivery receipts."""
        receipts: List[IurisDeliveryReceipt] = []
        processed = 0
        while True:
            if max_items is not None and processed >= max_items:
                break
            with self._lock:
                envelope = self._pop_next_fair_locked()
            if envelope is None:
                break
            receipts.append(self.deliver_envelope(envelope))
            processed += 1
        return receipts

    def start_worker(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="iuris-delivery-worker"
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=0.25)
            self._wake_event.clear()
            self.flush(max_items=25)

    def metrics(self) -> IurisDeliveryMetrics:
        with self._lock:
            q_depth = self._total_queue_depth_locked()
            mean_lat = (
                round(self._total_delivery_latency_ms / float(self._events_delivered), 3)
                if self._events_delivered > 0
                else 0.0
            )
            healthy = (
                not self.config.delivery_enabled
                or self.config.dry_run
                or (bool(self.destination_url) and self._last_error is None)
            )
            return IurisDeliveryMetrics(
                delivery_enabled=self.config.delivery_enabled,
                dry_run=self.config.dry_run,
                healthy=healthy,
                destination_url_redacted=redact_uri(self.destination_url),
                queue_depth=q_depth,
                max_queue_size=self.config.max_queue_size,
                events_enqueued=self._events_enqueued,
                events_attempted=self._events_attempted,
                events_delivered=self._events_delivered,
                events_failed=self._events_failed,
                events_retried=self._events_retried,
                duplicates_skipped=self._duplicates_skipped,
                dead_letter_count=self._dead_letter_count,
                mean_delivery_latency_ms=mean_lat,
                last_delivery_latency_ms=self._last_delivery_latency_ms,
                last_delivery_utc_iso=self._last_delivery_utc_iso,
                last_error=self._last_error,
            )

    def close(self, drain_remaining: bool = True) -> None:
        """Gracefully stop background delivery worker and optionally drain remaining queue to dead-letter outbox."""
        self._stop_event.set()
        self._wake_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        if drain_remaining:
            while True:
                with self._lock:
                    env = self._pop_next_fair_locked()
                if env is None:
                    break
                self._write_dead_letter(env, "Service shutdown with undelivered event in queue")
