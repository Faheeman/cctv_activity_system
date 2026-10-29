"""Structured logging configuration with RTSP credential redaction."""

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Dict, Optional, Union


_RTSP_CREDENTIAL_PATTERN = re.compile(r"(rtsps?://)([^:/@\s]+):([^/\s]+)@", re.IGNORECASE)


def redact_uri(uri: Union[str, int, None]) -> str:
    """Redact username and password from RTSP/HTTP URIs before logging or API exposure."""
    if uri is None:
        return ""
    text = str(uri)
    return _RTSP_CREDENTIAL_PATTERN.sub(r"\1***:***@", text)


class CredentialRedactingFormatter(logging.Formatter):
    """JSON structured log formatter that strips embedded stream credentials."""

    def format(self, record: logging.LogRecord) -> str:
        message = redact_uri(record.getMessage())
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }
        camera_id = getattr(record, "camera_id", None)
        if camera_id:
            payload["camera_id"] = camera_id
        event_type = getattr(record, "event_type", None)
        if event_type:
            payload["event_type"] = event_type
        if record.exc_info:
            payload["exception"] = redact_uri(self.formatException(record.exc_info))
        return json.dumps(payload, sort_keys=False)


def configure_logging(level: str = "INFO", logger_name: str = "activity_system") -> logging.Logger:
    """Configure and return the root application logger with structured JSON output."""
    logger = logging.getLogger(logger_name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(CredentialRedactingFormatter())
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setFormatter(CredentialRedactingFormatter())
    logger.propagate = False
    return logger


def get_camera_logger(camera_id: str, base_logger: Optional[logging.Logger] = None) -> logging.LoggerAdapter:
    """Return a LoggerAdapter bound to a specific camera_id."""
    logger = base_logger or logging.getLogger("activity_system")
    return logging.LoggerAdapter(logger, {"camera_id": camera_id})
