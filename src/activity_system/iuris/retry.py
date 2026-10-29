"""Retry, exponential backoff with jitter, and idempotency utilities for IURIS delivery."""

import hashlib
import random
from typing import Callable, Optional


RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def is_retryable_status_code(status_code: int) -> bool:
    """Return True if the HTTP status code warrants a retry with backoff."""
    return int(status_code) in RETRYABLE_HTTP_STATUS_CODES


def build_idempotency_key(event_id: str, event_type: str, schema_version: str = "1.0") -> str:
    """Derive a deterministic idempotency key from event_id, event_type, and schema_version."""
    cleaned_id = event_id.strip()
    if not cleaned_id:
        raise ValueError("event_id must not be empty when generating an idempotency key.")
    digest = hashlib.sha256(f"{schema_version}:{event_type}:{cleaned_id}".encode("utf-8")).hexdigest()[:24]
    return f"iuris-{cleaned_id}-{digest}"


def compute_backoff_delay(
    attempt: int,
    initial_delay_sec: float = 0.25,
    max_delay_sec: float = 15.0,
    backoff_factor: float = 2.0,
    jitter_ratio: float = 0.20,
    jitter_fn: Optional[Callable[[float, float], float]] = None,
) -> float:
    """Calculate exponential backoff delay in seconds with symmetric bounded jitter."""
    if attempt <= 0:
        return 0.0
    base = min(float(initial_delay_sec) * (float(backoff_factor) ** (attempt - 1)), float(max_delay_sec))
    if jitter_ratio > 0.0 and base > 0.0:
        rng = jitter_fn or random.uniform
        span = base * float(jitter_ratio)
        delay = base + rng(-span, span)
        return float(max(0.0, min(float(max_delay_sec), delay)))
    return float(base)
