"""Bounded, deduplicated event history store with lifecycle state updates and optional SQLite persistence."""

from collections import deque
import json
from pathlib import Path
import sqlite3
import threading
from typing import Deque, Dict, List, Optional, Set, Tuple

from activity_system.config import EventStorageConfig
from activity_system.schemas import (
    ActivityCode,
    ActivityObservationV1,
    AlertLifecycleState,
    CameraActivitySummaryV1,
    RiskEventV1,
    RiskSeverity,
    RiskStatus,
    RiskStatusHistoryEntry,
    utc_now_iso,
)


class BoundedEventStore:
    """Thread-safe bounded ring-buffer store (with optional SQLite backing) for historical and active events.

    Supports in-place lifecycle transitions (`TRIGGERED` -> `UPDATED` -> `ACKNOWLEDGED` -> `RESOLVED`)
    keyed by `event_id` and `correlation_key` so continuing conditions update existing alerts rather
    than creating unbounded duplicate records.
    """

    def __init__(self, config: Optional[EventStorageConfig] = None) -> None:
        self.config = config or EventStorageConfig()
        self._lock = threading.Lock()

        self._activities: Deque[ActivityObservationV1] = deque(
            maxlen=self.config.max_activity_history
        )
        self._summaries: Deque[CameraActivitySummaryV1] = deque(
            maxlen=self.config.max_summary_history
        )
        self._risks: Deque[RiskEventV1] = deque(maxlen=self.config.max_risk_history)
        self._risk_by_id: Dict[str, RiskEventV1] = {}

        self._seen_event_ids: Deque[str] = deque(
            maxlen=(
                self.config.max_activity_history
                + self.config.max_summary_history
                + self.config.max_risk_history
            )
        )
        self._seen_event_id_set: Set[str] = set()
        self._duplicates_ignored: int = 0

        self._sqlite_conn: Optional[sqlite3.Connection] = None
        if self.config.sqlite_path:
            db_path = Path(self.config.sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._sqlite_conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        assert self._sqlite_conn is not None
        with self._sqlite_conn:
            self._sqlite_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def _record_event_id_locked(self, event_id: str) -> bool:
        """Return True if event_id is new; False if duplicate."""
        if event_id in self._seen_event_id_set:
            self._duplicates_ignored += 1
            return False
        if len(self._seen_event_ids) == self._seen_event_ids.maxlen:
            evicted = self._seen_event_ids.popleft()
            self._seen_event_id_set.discard(evicted)
        self._seen_event_ids.append(event_id)
        self._seen_event_id_set.add(event_id)
        return True

    def _persist_sqlite_locked(
        self,
        event_id: str,
        event_type: str,
        camera_id: str,
        timestamp: str,
        payload_dict: dict,
        replace: bool = False,
    ) -> None:
        if self._sqlite_conn is None:
            return
        try:
            stmt = (
                "INSERT OR REPLACE INTO events (event_id, event_type, camera_id, timestamp, payload_json) VALUES (?, ?, ?, ?, ?)"
                if replace
                else "INSERT OR IGNORE INTO events (event_id, event_type, camera_id, timestamp, payload_json) VALUES (?, ?, ?, ?, ?)"
            )
            with self._sqlite_conn:
                self._sqlite_conn.execute(
                    stmt,
                    (event_id, event_type, camera_id, timestamp, json.dumps(payload_dict)),
                )
        except Exception:
            pass

    @property
    def duplicates_ignored(self) -> int:
        with self._lock:
            return self._duplicates_ignored

    def add_activity(self, event: ActivityObservationV1) -> bool:
        with self._lock:
            if not self._record_event_id_locked(event.event_id):
                return False
            self._activities.append(event)
            self._persist_sqlite_locked(
                event.event_id, event.event_type, event.camera_id, event.timestamp, event.dict()
            )
            return True

    def add_summary(self, summary: CameraActivitySummaryV1) -> bool:
        with self._lock:
            if not self._record_event_id_locked(summary.event_id):
                return False
            self._summaries.append(summary)
            self._persist_sqlite_locked(
                summary.event_id,
                summary.event_type,
                summary.camera_id,
                summary.timestamp,
                summary.dict(),
            )
            return True

    def add_risk_event(self, risk_event: RiskEventV1) -> bool:
        """Add a new risk event or update an existing risk event when lifecycle_state is UPDATED/RESOLVED."""
        with self._lock:
            existing = self._risk_by_id.get(risk_event.event_id)
            if existing is not None:
                if risk_event.lifecycle_state in (
                    AlertLifecycleState.UPDATED.value,
                    AlertLifecycleState.ACKNOWLEDGED.value,
                    AlertLifecycleState.RESOLVED.value,
                ):
                    for idx, item in enumerate(self._risks):
                        if item.event_id == risk_event.event_id:
                            self._risks[idx] = risk_event
                            break
                    self._risk_by_id[risk_event.event_id] = risk_event
                    self._persist_sqlite_locked(
                        risk_event.event_id,
                        risk_event.event_type,
                        risk_event.camera_id,
                        risk_event.timestamp,
                        risk_event.dict(),
                        replace=True,
                    )
                    return True
                self._duplicates_ignored += 1
                return False

            if not self._record_event_id_locked(risk_event.event_id):
                return False

            if len(self._risks) == self._risks.maxlen:
                oldest = self._risks.popleft()
                self._risk_by_id.pop(oldest.event_id, None)
            self._risks.append(risk_event)
            self._risk_by_id[risk_event.event_id] = risk_event
            self._persist_sqlite_locked(
                risk_event.event_id,
                risk_event.event_type,
                risk_event.camera_id,
                risk_event.timestamp,
                risk_event.dict(),
                replace=True,
            )
            return True

    def get_risk_event(self, event_id: str) -> Optional[RiskEventV1]:
        with self._lock:
            ev = self._risk_by_id.get(event_id)
            return ev.copy(deep=True) if ev is not None else None

    def acknowledge_risk_event(
        self,
        event_id: str,
        actor: str = "operator",
        note: Optional[str] = None,
        timestamp_utc: Optional[str] = None,
    ) -> Optional[RiskEventV1]:
        """Transition an OPEN risk alert to ACKNOWLEDGED with an audit trail entry."""
        now_iso = timestamp_utc or utc_now_iso()
        with self._lock:
            ev = self._risk_by_id.get(event_id)
            if ev is None:
                return None
            prev_status = ev.status
            ev.status = RiskStatus.ACKNOWLEDGED
            ev.lifecycle_state = AlertLifecycleState.ACKNOWLEDGED.value
            ev.acknowledged_at = now_iso
            ev.acknowledged_by = actor
            ev.updated_at = now_iso
            ev.status_history.append(
                RiskStatusHistoryEntry(
                    timestamp=now_iso,
                    from_status=prev_status,
                    to_status=RiskStatus.ACKNOWLEDGED,
                    lifecycle_state=AlertLifecycleState.ACKNOWLEDGED.value,
                    actor=actor,
                    note=note,
                )
            )
            self._persist_sqlite_locked(
                ev.event_id, ev.event_type, ev.camera_id, ev.timestamp, ev.dict(), replace=True
            )
            return ev.copy(deep=True)

    def resolve_risk_event(
        self,
        event_id: str,
        actor: str = "operator",
        note: Optional[str] = None,
        timestamp_utc: Optional[str] = None,
    ) -> Optional[RiskEventV1]:
        """Transition an OPEN or ACKNOWLEDGED risk alert to RESOLVED with an audit trail entry."""
        now_iso = timestamp_utc or utc_now_iso()
        with self._lock:
            ev = self._risk_by_id.get(event_id)
            if ev is None:
                return None
            prev_status = ev.status
            ev.status = RiskStatus.RESOLVED
            ev.lifecycle_state = AlertLifecycleState.RESOLVED.value
            ev.resolved_at = now_iso
            ev.resolved_by = actor
            ev.updated_at = now_iso
            if note is not None:
                ev.resolution_note = note
            ev.status_history.append(
                RiskStatusHistoryEntry(
                    timestamp=now_iso,
                    from_status=prev_status,
                    to_status=RiskStatus.RESOLVED,
                    lifecycle_state=AlertLifecycleState.RESOLVED.value,
                    actor=actor,
                    note=note,
                )
            )
            self._persist_sqlite_locked(
                ev.event_id, ev.event_type, ev.camera_id, ev.timestamp, ev.dict(), replace=True
            )
            return ev.copy(deep=True)

    def count_active_risk_events(self, camera_id: Optional[str] = None) -> int:
        with self._lock:
            return sum(
                1
                for x in self._risks
                if x.status in (RiskStatus.OPEN, RiskStatus.ACKNOWLEDGED)
                and (camera_id is None or x.camera_id == camera_id)
            )

    def query_activities(
        self,
        camera_id: Optional[str] = None,
        activity: Optional[ActivityCode] = None,
        track_id: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[ActivityObservationV1], int]:
        with self._lock:
            items = list(reversed(self._activities))
        if camera_id is not None:
            items = [x for x in items if x.camera_id == camera_id]
        if activity is not None:
            items = [x for x in items if x.activity == activity]
        if track_id is not None:
            items = [x for x in items if x.track_id == track_id]
        if since is not None:
            items = [x for x in items if x.timestamp >= since]
        total = len(items)
        return items[offset : offset + limit], total

    def query_summaries(
        self,
        camera_id: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[CameraActivitySummaryV1], int]:
        with self._lock:
            items = list(reversed(self._summaries))
        if camera_id is not None:
            items = [x for x in items if x.camera_id == camera_id]
        if since is not None:
            items = [x for x in items if x.timestamp >= since]
        total = len(items)
        return items[offset : offset + limit], total

    def query_risk_events(
        self,
        camera_id: Optional[str] = None,
        severity: Optional[RiskSeverity] = None,
        risk_type: Optional[str] = None,
        rule_name: Optional[str] = None,
        status: Optional[RiskStatus] = None,
        active_only: bool = False,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[RiskEventV1], int]:
        with self._lock:
            items = [x.copy(deep=True) for x in reversed(self._risks)]
        if camera_id is not None:
            items = [x for x in items if x.camera_id == camera_id]
        if severity is not None:
            items = [x for x in items if x.severity == severity]
        if risk_type is not None:
            items = [x for x in items if x.risk_type.upper() == risk_type.strip().upper()]
        if rule_name is not None:
            items = [x for x in items if x.rule_name.lower() == rule_name.strip().lower()]
        if status is not None:
            items = [x for x in items if x.status == status]
        if active_only:
            items = [x for x in items if x.status in (RiskStatus.OPEN, RiskStatus.ACKNOWLEDGED)]
        if since is not None:
            items = [x for x in items if x.timestamp >= since]
        if until is not None:
            items = [x for x in items if x.timestamp <= until]
        total = len(items)
        return items[offset : offset + limit], total

    def close(self) -> None:
        with self._lock:
            if self._sqlite_conn is not None:
                try:
                    self._sqlite_conn.close()
                except Exception:
                    pass
                self._sqlite_conn = None
