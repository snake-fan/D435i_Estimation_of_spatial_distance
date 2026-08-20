"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import math
from pathlib import Path
from typing import Any, TypeVar


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


@dataclass(frozen=True)
class CameraConfig:
    width: int = 848
    height: int = 480
    fps: int = 30
    emitter_enabled: bool = False
    warmup_frames: int = 30
    serial_number: str | None = None
    frame_timeout_ms: int = 5000


@dataclass(frozen=True)
class QRConfig:
    expected_ids: list[str] = field(default_factory=lambda: ["QR_A", "QR_B"])
    polygon_shrink_ratio: float = 0.75
    min_edge_pixels: float = 35.0
    corner_refinement: bool = True


@dataclass(frozen=True)
class PlaneConfig:
    ransac_iterations: int = 200
    distance_threshold: float = 0.004
    min_points: int = 80
    min_inlier_ratio: float = 0.65
    max_rms: float = 0.003
    sample_stride: int = 2
    random_seed: int | None = 0
    degenerate_epsilon: float = 1.0e-12


@dataclass(frozen=True)
class MeasurementConfig:
    min_depth: float = 0.25
    max_depth: float = 2.0
    min_valid_depth_ratio: float = 0.35
    min_spatial_quadrants: int = 3
    min_points_per_quadrant: int = 5
    warning_tilt_deg: float = 45.0
    max_tilt_deg: float = 60.0
    ray_parallel_epsilon: float = 1.0e-10


@dataclass(frozen=True)
class TemporalConfig:
    window_size: int = 20
    min_valid_frames: int = 10


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool = True
    depth_min_m: float = 0.25
    depth_max_m: float = 2.0


@dataclass(frozen=True)
class RecordingConfig:
    csv_enabled: bool = False
    flush_every_row: bool = True


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    qr: QRConfig = field(default_factory=QRConfig)
    plane: PlaneConfig = field(default_factory=PlaneConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _construct(cls: type[T], values: Any, section: str) -> T:
    if not isinstance(values, dict):
        raise ConfigError(f"Configuration section '{section}' must be a mapping")

    known = {item.name: item for item in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ConfigError(
            f"Unknown option(s) in '{section}': {', '.join(unknown)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        field_info = known[name]
        nested_type = field_info.type
        if isinstance(nested_type, type) and is_dataclass(nested_type):
            kwargs[name] = _construct(nested_type, value, f"{section}.{name}")
        else:
            kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"Invalid configuration in '{section}': {exc}") from exc


def validate_config(config: AppConfig) -> None:
    camera = config.camera
    _require_int("camera.width", camera.width)
    _require_int("camera.height", camera.height)
    _require_int("camera.fps", camera.fps)
    _require_bool("camera.emitter_enabled", camera.emitter_enabled)
    _require_int("camera.warmup_frames", camera.warmup_frames)
    _require_int("camera.frame_timeout_ms", camera.frame_timeout_ms)
    if camera.serial_number is not None and (
        not isinstance(camera.serial_number, str) or not camera.serial_number.strip()
    ):
        raise ConfigError("camera.serial_number must be null or a non-empty string")
    if camera.width <= 0 or camera.height <= 0 or camera.fps <= 0:
        raise ConfigError("camera width, height and fps must be positive")
    if camera.warmup_frames < 0 or camera.frame_timeout_ms <= 0:
        raise ConfigError("camera warmup_frames must be >= 0 and frame_timeout_ms > 0")

    qr = config.qr
    if not isinstance(qr.expected_ids, list) or not all(
        isinstance(value, str) for value in qr.expected_ids
    ):
        raise ConfigError("qr.expected_ids must be a list of strings")
    _require_number("qr.polygon_shrink_ratio", qr.polygon_shrink_ratio)
    _require_number("qr.min_edge_pixels", qr.min_edge_pixels)
    _require_bool("qr.corner_refinement", qr.corner_refinement)
    if len(qr.expected_ids) != 2 or len(set(qr.expected_ids)) != 2:
        raise ConfigError("qr.expected_ids must contain exactly two distinct payloads")
    if any(not str(qr_id).strip() for qr_id in qr.expected_ids):
        raise ConfigError("qr.expected_ids may not contain empty payloads")
    if not 0.0 < qr.polygon_shrink_ratio <= 1.0:
        raise ConfigError("qr.polygon_shrink_ratio must be in (0, 1]")
    if qr.min_edge_pixels <= 0:
        raise ConfigError("qr.min_edge_pixels must be positive")

    plane = config.plane
    _require_int("plane.ransac_iterations", plane.ransac_iterations)
    _require_number("plane.distance_threshold", plane.distance_threshold)
    _require_int("plane.min_points", plane.min_points)
    _require_number("plane.min_inlier_ratio", plane.min_inlier_ratio)
    _require_number("plane.max_rms", plane.max_rms)
    _require_int("plane.sample_stride", plane.sample_stride)
    _require_number("plane.degenerate_epsilon", plane.degenerate_epsilon)
    if plane.random_seed is not None:
        _require_int("plane.random_seed", plane.random_seed)
        if plane.random_seed < 0:
            raise ConfigError("plane.random_seed must be null or non-negative")
    if plane.ransac_iterations <= 0 or plane.min_points < 3:
        raise ConfigError("plane iterations must be > 0 and min_points must be >= 3")
    if plane.distance_threshold <= 0 or plane.max_rms <= 0:
        raise ConfigError("plane distance_threshold and max_rms must be positive")
    if plane.max_rms >= plane.distance_threshold:
        raise ConfigError(
            "plane.max_rms must be smaller than plane.distance_threshold "
            "because RMS is computed on RANSAC inliers"
        )
    if not 0.0 <= plane.min_inlier_ratio <= 1.0:
        raise ConfigError("plane.min_inlier_ratio must be in [0, 1]")
    if plane.sample_stride <= 0 or plane.degenerate_epsilon <= 0:
        raise ConfigError("plane sample_stride and degenerate_epsilon must be positive")

    measurement = config.measurement
    _require_number("measurement.min_depth", measurement.min_depth)
    _require_number("measurement.max_depth", measurement.max_depth)
    _require_number(
        "measurement.min_valid_depth_ratio", measurement.min_valid_depth_ratio
    )
    _require_int(
        "measurement.min_spatial_quadrants", measurement.min_spatial_quadrants
    )
    _require_int(
        "measurement.min_points_per_quadrant", measurement.min_points_per_quadrant
    )
    _require_number("measurement.warning_tilt_deg", measurement.warning_tilt_deg)
    _require_number("measurement.max_tilt_deg", measurement.max_tilt_deg)
    _require_number("measurement.ray_parallel_epsilon", measurement.ray_parallel_epsilon)
    if not 0.0 < measurement.min_depth < measurement.max_depth:
        raise ConfigError("measurement depth range must satisfy 0 < min_depth < max_depth")
    if not 0.0 < measurement.min_valid_depth_ratio <= 1.0:
        raise ConfigError("measurement.min_valid_depth_ratio must be in (0, 1]")
    if not 1 <= measurement.min_spatial_quadrants <= 4:
        raise ConfigError("measurement.min_spatial_quadrants must be in [1, 4]")
    if measurement.min_points_per_quadrant <= 0:
        raise ConfigError("measurement.min_points_per_quadrant must be positive")
    if not 0.0 <= measurement.warning_tilt_deg <= measurement.max_tilt_deg < 90.0:
        raise ConfigError(
            "measurement tilt limits must satisfy 0 <= warning <= max < 90 degrees"
        )
    if measurement.ray_parallel_epsilon <= 0:
        raise ConfigError("measurement.ray_parallel_epsilon must be positive")

    temporal = config.temporal
    _require_int("temporal.window_size", temporal.window_size)
    _require_int("temporal.min_valid_frames", temporal.min_valid_frames)
    if temporal.window_size <= 0:
        raise ConfigError("temporal.window_size must be positive")
    if not 1 <= temporal.min_valid_frames <= temporal.window_size:
        raise ConfigError(
            "temporal.min_valid_frames must be between 1 and window_size"
        )

    visualization = config.visualization
    _require_bool("visualization.enabled", visualization.enabled)
    _require_number("visualization.depth_min_m", visualization.depth_min_m)
    _require_number("visualization.depth_max_m", visualization.depth_max_m)
    if not 0.0 <= visualization.depth_min_m < visualization.depth_max_m:
        raise ConfigError("visualization depth range is invalid")

    _require_bool("recording.csv_enabled", config.recording.csv_enabled)
    _require_bool("recording.flush_every_row", config.recording.flush_every_row)


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")


def _require_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")


def _require_number(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ConfigError(f"{name} must be finite")


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load an ``AppConfig`` from YAML and reject misspelled options."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ConfigError(
            "PyYAML is required to read config.yaml; install project dependencies first"
        ) from exc

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not read {config_path}: {exc}") from exc

    if raw is None:
        raw = {}
    elif not isinstance(raw, dict):
        raise ConfigError("The root configuration value must be a mapping")

    section_types: dict[str, type[Any]] = {
        "camera": CameraConfig,
        "qr": QRConfig,
        "plane": PlaneConfig,
        "measurement": MeasurementConfig,
        "temporal": TemporalConfig,
        "visualization": VisualizationConfig,
        "recording": RecordingConfig,
    }
    unknown_sections = sorted(set(raw) - set(section_types))
    if unknown_sections:
        raise ConfigError(f"Unknown configuration section(s): {', '.join(unknown_sections)}")

    kwargs = {
        name: _construct(section_types[name], section_values, name)
        for name, section_values in raw.items()
    }
    config = AppConfig(**kwargs)
    validate_config(config)
    return config
