"""Configuration loading, environment variable overrides, and validation using YAML and Pydantic."""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union
import yaml
from pydantic import BaseModel, Field, validator

from activity_system.schemas import ActivityCode, RiskSeverity, SourceType


class SystemConfig(BaseModel):
    app_name: str = "cctv-activity-recognition"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    enable_api: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)


class CameraConfig(BaseModel):
    camera_id: str
    camera_name: Optional[str] = None
    source_type: SourceType
    source_uri: Union[str, int]
    enabled: bool = True
    loop_file: bool = False
    sample_every_n_frames: int = Field(default=1, ge=1, le=120)
    target_fps: float = Field(default=15.0, gt=0.0, le=120.0)
    max_frame_queue_size: int = Field(default=1, ge=1, le=32)
    resize_width: Optional[int] = Field(default=640, ge=64, le=3840)
    resize_height: Optional[int] = Field(default=480, ge=64, le=2160)
    connection_timeout_ms: int = Field(default=5000, ge=100, le=60000)
    read_timeout_ms: int = Field(default=5000, ge=100, le=60000)
    reconnect_initial_delay_sec: float = Field(default=0.5, ge=0.0, le=60.0)
    reconnect_max_delay_sec: float = Field(default=8.0, ge=0.1, le=300.0)
    reconnect_backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    reconnect_jitter_ratio: float = Field(default=0.0, ge=0.0, le=0.5)
    max_reconnect_attempts: int = Field(default=5, ge=0, le=1000)
    stale_frame_timeout_sec: float = Field(default=5.0, gt=0.1, le=300.0)

    @validator("camera_id")
    def validate_camera_id(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("camera_id must not be empty.")
        return cleaned

    @validator("source_uri")
    def validate_source_uri(cls, v: Union[str, int], values: dict) -> Union[str, int]:
        source_type = values.get("source_type")
        if source_type == SourceType.WEBCAM:
            if isinstance(v, str) and v.strip().startswith("/dev/video"):
                suffix = v.strip().replace("/dev/video", "")
                if suffix.isdigit():
                    return int(suffix)
            try:
                idx = int(v)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "Webcam source_uri must be a non-negative integer index or /dev/videoN."
                ) from exc
            if idx < 0:
                raise ValueError("Webcam source_uri index must be >= 0.")
            return idx
        if isinstance(v, str) and not v.strip():
            raise ValueError("source_uri must not be an empty string.")
        if source_type == SourceType.RTSP and isinstance(v, str):
            if not (v.startswith("rtsp://") or v.startswith("rtsps://")):
                raise ValueError("RTSP source_uri must start with rtsp:// or rtsps://.")
        return v


class PoseConfig(BaseModel):
    backend: Literal["ultralytics", "onnx", "synthetic"] = "synthetic"
    model_path: str = "models/yolov8n-pose.onnx"
    model_version: str = "yolov8n-pose-1.0"
    device: str = "cpu"
    imgsz: int = Field(default=640, ge=128, le=1920)
    min_person_confidence: float = Field(default=0.50, gt=0.0, le=1.0)
    min_keypoint_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, gt=0.0, le=1.0)
    cpu_threads: int = Field(default=2, ge=1, le=64)
    allow_synthetic_fallback: bool = True

    @validator("device")
    def validate_device(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("device must not be empty.")
        if cleaned not in ("auto", "cpu", "cuda", "mps", "tensorrt") and not cleaned.startswith("cuda:"):
            raise ValueError(
                f"Unsupported device '{v}'. Expected one of 'auto', 'cpu', 'cuda', 'cuda:N', 'mps', 'tensorrt'."
            )
        return cleaned


class TrackerConfig(BaseModel):
    high_thresh: float = Field(default=0.60, gt=0.0, le=1.0)
    low_thresh: float = Field(default=0.25, gt=0.0, le=1.0)
    match_iou_thresh: float = Field(default=0.30, gt=0.0, le=1.0)
    second_match_iou_thresh: float = Field(default=0.20, gt=0.0, le=1.0)
    min_hits_to_confirm: int = Field(default=2, ge=1, le=30)
    track_expiration_seconds: float = Field(default=2.0, gt=0.0, le=60.0)
    max_window_duration_seconds: float = Field(default=1.5, gt=0.1, le=30.0)
    max_window_observations: int = Field(default=60, ge=5, le=600)

    @validator("low_thresh")
    def validate_low_less_than_high(cls, v: float, values: dict) -> float:
        high = values.get("high_thresh")
        if high is not None and v >= high:
            raise ValueError(f"low_thresh ({v}) must be strictly less than high_thresh ({high}).")
        return v


class ActivityConfig(BaseModel):
    classifier_type: Literal["heuristic", "supervised"] = "heuristic"
    model_version: str = "heuristic-kinematic-v1.0"
    min_observations: int = Field(default=6, ge=2, le=100)
    min_window_duration_seconds: float = Field(default=0.50, gt=0.0, le=10.0)
    standing_min_duration_seconds: float = Field(default=0.60, gt=0.0, le=30.0)
    standing_max_speed_bh_per_sec: float = Field(default=0.35, gt=0.0, le=5.0)
    walking_min_speed_bh_per_sec: float = Field(default=0.35, gt=0.0, le=5.0)
    running_min_speed_bh_per_sec: float = Field(default=1.80, gt=0.1, le=15.0)
    standing_max_ankle_std_bh: float = Field(default=0.035, ge=0.0, le=1.0)
    walking_min_ankle_std_bh: float = Field(default=0.025, ge=0.0, le=1.0)
    running_min_ankle_std_bh: float = Field(default=0.055, ge=0.0, le=1.0)
    sitting_max_hip_knee_ratio: float = Field(default=0.22, gt=0.02, le=0.50)
    sitting_min_aspect_ratio: float = Field(default=0.52, gt=0.20, le=1.50)
    lying_min_aspect_ratio: float = Field(default=1.15, gt=0.70, le=5.0)
    lying_min_duration_seconds: float = Field(default=0.60, gt=0.1, le=30.0)
    fall_min_downward_velocity_bh_per_sec: float = Field(default=0.80, gt=0.1, le=10.0)
    fall_min_height_drop_ratio: float = Field(default=0.25, gt=0.05, le=0.90)
    fall_max_transition_seconds: float = Field(default=1.50, gt=0.2, le=5.0)
    hysteresis_count: int = Field(default=3, ge=1, le=30)
    camera_viewpoint: Literal["cctv", "eye_level", "overhead"] = "cctv"
    cctv_foreshortening_factor: float = Field(default=1.25, ge=1.0, le=3.0)

    @validator("running_min_speed_bh_per_sec")
    def validate_speed_hierarchy(cls, v: float, values: dict) -> float:
        walk_min = values.get("walking_min_speed_bh_per_sec")
        if walk_min is not None and v <= walk_min:
            raise ValueError("running_min_speed_bh_per_sec must exceed walking_min_speed_bh_per_sec.")
        return v


class SummaryConfig(BaseModel):
    window_seconds: float = Field(default=10.0, gt=0.5, le=3600.0)
    exclude_unknown_from_percentages: bool = False
    max_history_per_camera: int = Field(default=200, ge=10, le=10000)


class ConcentrationCameraOverrideConfig(BaseModel):
    """Per-camera threshold overrides for the crowd-running / activity concentration rule."""

    enabled: Optional[bool] = None
    activity: Optional[ActivityCode] = None
    percentage_threshold: Optional[float] = Field(default=None, gt=0.0, le=100.0)
    minimum_people: Optional[int] = Field(default=None, ge=1, le=1000)
    window_seconds: Optional[float] = Field(default=None, gt=0.5, le=3600.0)
    minimum_duration_seconds: Optional[float] = Field(default=None, ge=0.0, le=600.0)
    min_observation_quality: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    clear_grace_seconds: Optional[float] = Field(default=None, ge=0.0, le=120.0)
    severity: Optional[RiskSeverity] = None
    cooldown_seconds: Optional[float] = Field(default=None, ge=0.0, le=3600.0)


class ConcentrationRuleConfig(BaseModel):
    enabled: bool = True
    activity: ActivityCode = ActivityCode.RUNNING
    percentage_threshold: float = Field(default=40.0, gt=0.0, le=100.0)
    minimum_people: int = Field(default=5, ge=1, le=1000)
    window_seconds: float = Field(default=10.0, gt=0.5, le=3600.0)
    minimum_duration_seconds: float = Field(default=5.0, ge=0.0, le=600.0)
    min_observation_quality: float = Field(default=0.45, ge=0.0, le=1.0)
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    clear_grace_seconds: float = Field(default=1.5, ge=0.0, le=120.0)
    emit_resolved_events: bool = True
    severity: RiskSeverity = RiskSeverity.MEDIUM
    cooldown_seconds: float = Field(default=10.0, ge=0.0, le=3600.0)
    camera_overrides: Dict[str, ConcentrationCameraOverrideConfig] = Field(default_factory=dict)

    def for_camera(self, camera_id: str) -> "ConcentrationRuleConfig":
        """Return effective rule configuration with camera-specific overrides applied."""
        override = self.camera_overrides.get(camera_id)
        if override is None:
            return self
        merged = self.dict(exclude={"camera_overrides"})
        for k, v in override.dict(exclude_unset=True, exclude_none=True).items():
            merged[k] = v
        merged["camera_overrides"] = {}
        return ConcentrationRuleConfig(**merged)


class FallRuleConfig(BaseModel):
    enabled: bool = True
    minimum_duration_seconds: float = Field(default=0.40, ge=0.0, le=30.0)
    min_observation_quality: float = Field(default=0.50, ge=0.0, le=1.0)
    clear_grace_seconds: float = Field(default=1.5, ge=0.0, le=60.0)
    severity: RiskSeverity = RiskSeverity.HIGH
    cooldown_seconds: float = Field(default=10.0, ge=0.0, le=3600.0)


class ImmobilityRuleConfig(BaseModel):
    enabled: bool = True
    activity: ActivityCode = ActivityCode.LYING
    immobility_duration_seconds: float = Field(default=10.0, gt=0.5, le=3600.0)
    min_observation_quality: float = Field(default=0.50, ge=0.0, le=1.0)
    clear_grace_seconds: float = Field(default=2.0, ge=0.0, le=60.0)
    severity: RiskSeverity = RiskSeverity.HIGH
    cooldown_seconds: float = Field(default=15.0, ge=0.0, le=3600.0)


class PersistentActivityPatternConfig(BaseModel):
    """Configurable high-risk or persistent activity pattern rule in a monitored camera view."""

    pattern_id: str
    camera_id: Optional[str] = None
    enabled: bool = True
    activity: ActivityCode = ActivityCode.RUNNING
    minimum_people: int = Field(default=1, ge=1, le=500)
    minimum_duration_seconds: float = Field(default=8.0, ge=0.0, le=3600.0)
    min_observation_quality: float = Field(default=0.50, ge=0.0, le=1.0)
    severity: RiskSeverity = RiskSeverity.MEDIUM
    cooldown_seconds: float = Field(default=15.0, ge=0.0, le=3600.0)


class ZoneDefinitionConfig(BaseModel):
    zone_id: str
    camera_id: str
    name: str = "Restricted Zone"
    enabled: bool = True
    polygon: List[Tuple[float, float]] = Field(
        ..., description="List of (x, y) vertices in image pixel coordinates (>= 3 points)."
    )
    trigger_activities: List[ActivityCode] = Field(
        default_factory=lambda: [
            ActivityCode.STANDING,
            ActivityCode.WALKING,
            ActivityCode.RUNNING,
            ActivityCode.LYING,
            ActivityCode.FALLING,
        ]
    )
    minimum_duration_seconds: float = Field(default=1.0, ge=0.0, le=600.0)
    min_observation_quality: float = Field(default=0.45, ge=0.0, le=1.0)
    severity: RiskSeverity = RiskSeverity.MEDIUM
    cooldown_seconds: float = Field(default=10.0, ge=0.0, le=3600.0)

    @validator("polygon")
    def validate_polygon_vertices(cls, v: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(v) < 3:
            raise ValueError("Zone polygon requires at least 3 vertices.")
        return v


class RiskRulesConfig(BaseModel):
    unusual_running_concentration: ConcentrationRuleConfig = Field(
        default_factory=ConcentrationRuleConfig
    )
    possible_fall: FallRuleConfig = Field(default_factory=FallRuleConfig)
    prolonged_immobility: ImmobilityRuleConfig = Field(default_factory=ImmobilityRuleConfig)
    persistent_patterns: List[PersistentActivityPatternConfig] = Field(default_factory=list)
    zones: List[ZoneDefinitionConfig] = Field(default_factory=list)


class RiskEngineConfig(BaseModel):
    enabled: bool = True
    rules_path: Optional[str] = None
    rules: RiskRulesConfig = Field(default_factory=RiskRulesConfig)


class IurisConfig(BaseModel):
    delivery_enabled: bool = False
    dry_run: bool = False
    base_url: str = ""
    events_path: str = "/api/v1/cctv/events"
    api_token: str = ""
    auth_scheme: Literal["bearer", "api_key_header"] = "bearer"
    auth_header_name: str = "Authorization"
    timeout_seconds: float = Field(default=5.0, gt=0.1, le=60.0)
    max_queue_size: int = Field(default=1000, ge=1, le=100000)
    max_retry_attempts: int = Field(default=8, ge=0, le=50)
    retry_initial_delay_sec: float = Field(default=0.25, ge=0.0, le=60.0)
    retry_max_delay_sec: float = Field(default=15.0, ge=0.05, le=300.0)
    retry_backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    retry_jitter_ratio: float = Field(default=0.20, ge=0.0, le=0.5)
    backpressure_policy: Literal["dead_letter_and_drop_oldest", "drop_oldest", "drop_newest"] = (
        "dead_letter_and_drop_oldest"
    )
    dead_letter_path: Optional[str] = "outputs/iuris_dead_letter.jsonl"
    deliver_activity_events: bool = True
    deliver_summary_events: bool = True
    deliver_risk_events: bool = True


class EventStorageConfig(BaseModel):
    max_activity_history: int = Field(default=1000, ge=10, le=50000)
    max_summary_history: int = Field(default=500, ge=10, le=10000)
    max_risk_history: int = Field(default=500, ge=10, le=10000)
    sqlite_path: Optional[str] = None


class RendererConfig(BaseModel):
    draw_bbox: bool = True
    draw_skeleton: bool = True
    draw_track_id: bool = True
    draw_activity: bool = True
    draw_telemetry: bool = True


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    cameras: List[CameraConfig]
    pose: PoseConfig = Field(default_factory=PoseConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    risk: RiskEngineConfig = Field(default_factory=RiskEngineConfig)
    iuris: IurisConfig = Field(default_factory=IurisConfig)
    storage: EventStorageConfig = Field(default_factory=EventStorageConfig)
    renderer: RendererConfig = Field(default_factory=RendererConfig)

    @validator("cameras")
    def validate_unique_cameras(cls, v: List[CameraConfig]) -> List[CameraConfig]:
        if not v:
            raise ValueError("At least one camera must be configured.")
        ids = [cam.camera_id for cam in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate camera_id values found in configuration.")
        return v


class ConfigLoader:
    """Loads and validates YAML runtime configuration and applies environment variable overrides."""

    @staticmethod
    def load(
        config_path: Union[str, Path],
        apply_env: bool = True,
        environ: Optional[Dict[str, str]] = None,
    ) -> AppConfig:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw_data = yaml.safe_load(fh)
        if not isinstance(raw_data, dict):
            raise ValueError(f"Configuration file {path} must contain a top-level YAML mapping.")
        cfg = AppConfig(**raw_data)
        base_dir = path.parent.parent if path.parent.name == "config" else path.parent
        ConfigLoader._load_external_risk_rules(cfg, base_dir=base_dir, environ=environ)
        if apply_env:
            ConfigLoader.apply_env_overrides(cfg, environ=environ, base_dir=base_dir)
        return cfg

    @staticmethod
    def from_dict(data: dict, apply_env: bool = False, environ: Optional[Dict[str, str]] = None) -> AppConfig:
        if not isinstance(data, dict):
            raise ValueError("Configuration input must be a dictionary.")
        cfg = AppConfig(**data)
        if apply_env:
            ConfigLoader.apply_env_overrides(cfg, environ=environ)
        return cfg

    @staticmethod
    def _load_external_risk_rules(
        cfg: AppConfig,
        base_dir: Optional[Path] = None,
        environ: Optional[Dict[str, str]] = None,
    ) -> None:
        env = environ if environ is not None else os.environ
        rules_path_str = env.get("RISK_RULES_PATH") or cfg.risk.rules_path
        if not rules_path_str:
            return
        rules_path = Path(rules_path_str)
        if not rules_path.is_absolute() and base_dir is not None:
            candidate = base_dir / rules_path
            if candidate.exists():
                rules_path = candidate
        if not rules_path.exists():
            raise FileNotFoundError(f"Risk rules configuration file not found: {rules_path}")
        with rules_path.open("r", encoding="utf-8") as fh:
            raw_rules = yaml.safe_load(fh)
        if not isinstance(raw_rules, dict):
            raise ValueError(f"Risk rules file {rules_path} must contain a YAML mapping.")
        rules_payload = raw_rules.get("rules", raw_rules)
        cfg.risk.rules = RiskRulesConfig(**rules_payload)

    @staticmethod
    def apply_env_overrides(
        cfg: AppConfig,
        environ: Optional[Dict[str, str]] = None,
        base_dir: Optional[Path] = None,
    ) -> AppConfig:
        """Apply documented AI_*, IURIS_*, EVENT_*, ACTIVITY_*, and RISK_* environment variables."""
        env = environ if environ is not None else os.environ

        host = env.get("AI_BACKEND_HOST") or env.get("API_HOST")
        if host:
            cfg.system.api_host = host.strip()

        port = env.get("AI_BACKEND_PORT") or env.get("API_PORT")
        if port:
            cfg.system.api_port = int(port)

        log_level = env.get("AI_LOG_LEVEL") or env.get("LOG_LEVEL")
        if log_level:
            upper_lvl = log_level.strip().upper()
            if upper_lvl in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
                cfg.system.log_level = upper_lvl  # type: ignore[assignment]

        ai_device = env.get("AI_DEVICE")
        if ai_device:
            cfg.pose.device = PoseConfig.validate_device(ai_device)

        ai_model_path = env.get("AI_MODEL_PATH")
        if ai_model_path:
            cfg.pose.model_path = ai_model_path.strip()

        min_conf = env.get("ACTIVITY_MIN_CONFIDENCE")
        if min_conf:
            val = float(min_conf)
            if not (0.0 < val <= 1.0):
                raise ValueError("ACTIVITY_MIN_CONFIDENCE must be in (0.0, 1.0].")
            cfg.pose.min_person_confidence = val

        win_sec = env.get("ACTIVITY_WINDOW_SECONDS")
        if win_sec:
            val = float(win_sec)
            if val <= 0.0:
                raise ValueError("ACTIVITY_WINDOW_SECONDS must be > 0.")
            cfg.summary.window_seconds = val

        iuris_enabled = env.get("IURIS_DELIVERY_ENABLED")
        if iuris_enabled is not None and iuris_enabled.strip() != "":
            cfg.iuris.delivery_enabled = iuris_enabled.strip().lower() in ("1", "true", "yes", "on")

        iuris_dry_run = env.get("IURIS_DRY_RUN")
        if iuris_dry_run is not None and iuris_dry_run.strip() != "":
            cfg.iuris.dry_run = iuris_dry_run.strip().lower() in ("1", "true", "yes", "on")

        iuris_url = env.get("IURIS_BASE_URL")
        if iuris_url is not None and iuris_url.strip() != "":
            cfg.iuris.base_url = iuris_url.strip()

        iuris_path = env.get("IURIS_EVENTS_PATH")
        if iuris_path is not None and iuris_path.strip() != "":
            cfg.iuris.events_path = iuris_path.strip()

        iuris_token = env.get("IURIS_API_TOKEN")
        if iuris_token is not None and iuris_token.strip() != "":
            cfg.iuris.api_token = iuris_token.strip()

        q_max = env.get("EVENT_QUEUE_MAX_SIZE")
        if q_max:
            cfg.iuris.max_queue_size = int(q_max)

        retry_max = env.get("EVENT_RETRY_MAX_ATTEMPTS")
        if retry_max:
            cfg.iuris.max_retry_attempts = int(retry_max)

        if env.get("RISK_RULES_PATH"):
            ConfigLoader._load_external_risk_rules(cfg, base_dir=base_dir, environ=env)

        cfg.system = SystemConfig(**cfg.system.dict())
        cfg.pose = PoseConfig(**cfg.pose.dict())
        cfg.summary = SummaryConfig(**cfg.summary.dict())
        cfg.iuris = IurisConfig(**cfg.iuris.dict())
        return cfg
