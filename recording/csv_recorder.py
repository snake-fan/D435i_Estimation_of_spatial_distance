"""CSV recorder for accepted and rejected measurement frames."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TextIO

import numpy as np


CSV_FIELDS = [
    "wall_time_utc",
    "source_timestamp_ms",
    "frame_number",
    "qr_a_id",
    "qr_a_status",
    "qr_a_reject_reason",
    "qr_a_x_m",
    "qr_a_y_m",
    "qr_a_z_m",
    "qr_b_id",
    "qr_b_status",
    "qr_b_reject_reason",
    "qr_b_x_m",
    "qr_b_y_m",
    "qr_b_z_m",
    "distance_mm",
    "qr_a_plane_rms_mm",
    "qr_b_plane_rms_mm",
    "qr_a_inlier_ratio",
    "qr_b_inlier_ratio",
    "qr_a_tilt_deg",
    "qr_b_tilt_deg",
    "qr_a_edge_pixels",
    "qr_b_edge_pixels",
    "qr_a_valid_depth_points",
    "qr_b_valid_depth_points",
    "qr_a_valid_depth_ratio",
    "qr_b_valid_depth_ratio",
    "qr_a_spatial_quadrants",
    "qr_b_spatial_quadrants",
    "temporal_sample_count",
    "temporal_ready",
    "temporal_median_mm",
    "temporal_mean_mm",
    "temporal_std_mm",
    "temporal_mad_mm",
    "method_a_distance_mm",
    "method_b_distance_mm",
    "method_c_distance_mm",
    "status",
    "reject_reason",
]


def _number(value: Any, scale: float = 1.0) -> float | str:
    if value is None:
        return ""
    try:
        result = float(value) * scale
    except (TypeError, ValueError):
        return ""
    return result if math.isfinite(result) else ""


def _coordinates(result: Any) -> tuple[float | str, float | str, float | str]:
    point = getattr(result, "point_xyz", None)
    if point is None:
        return "", "", ""
    array = np.asarray(point, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        return "", "", ""
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def build_csv_row(
    *,
    source_timestamp_ms: float,
    frame_number: int,
    qr_results: Mapping[str, Any],
    expected_ids: list[str] | tuple[str, str],
    distance_result: Any = None,
    temporal: Any = None,
    status: str,
    reject_reason: str | None = None,
    method_a_distance_m: float | None = None,
    method_b_distance_m: float | None = None,
    method_c_distance_m: float | None = None,
) -> dict[str, Any]:
    if len(expected_ids) != 2:
        raise ValueError("Exactly two expected QR ids are required")
    qr_a_id, qr_b_id = expected_ids
    qr_a = qr_results.get(qr_a_id)
    qr_b = qr_results.get(qr_b_id)
    a_xyz = _coordinates(qr_a)
    b_xyz = _coordinates(qr_b)
    plane_a = getattr(qr_a, "plane", None)
    plane_b = getattr(qr_b, "plane", None)
    return {
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        "source_timestamp_ms": _number(source_timestamp_ms),
        "frame_number": int(frame_number),
        "qr_a_id": qr_a_id,
        "qr_a_status": str(getattr(qr_a, "status", "INVALID")),
        "qr_a_reject_reason": getattr(qr_a, "reject_reason", None) or "",
        "qr_a_x_m": a_xyz[0],
        "qr_a_y_m": a_xyz[1],
        "qr_a_z_m": a_xyz[2],
        "qr_b_id": qr_b_id,
        "qr_b_status": str(getattr(qr_b, "status", "INVALID")),
        "qr_b_reject_reason": getattr(qr_b, "reject_reason", None) or "",
        "qr_b_x_m": b_xyz[0],
        "qr_b_y_m": b_xyz[1],
        "qr_b_z_m": b_xyz[2],
        "distance_mm": _number(
            getattr(distance_result, "distance_m", None), 1000.0
        ),
        "qr_a_plane_rms_mm": _number(getattr(plane_a, "rms", None), 1000.0),
        "qr_b_plane_rms_mm": _number(getattr(plane_b, "rms", None), 1000.0),
        "qr_a_inlier_ratio": _number(getattr(plane_a, "inlier_ratio", None)),
        "qr_b_inlier_ratio": _number(getattr(plane_b, "inlier_ratio", None)),
        "qr_a_tilt_deg": _number(getattr(qr_a, "tilt_deg", None)),
        "qr_b_tilt_deg": _number(getattr(qr_b, "tilt_deg", None)),
        "qr_a_edge_pixels": _number(
            getattr(qr_a, "qr_min_edge_pixels", None)
        ),
        "qr_b_edge_pixels": _number(
            getattr(qr_b, "qr_min_edge_pixels", None)
        ),
        "qr_a_valid_depth_points": int(
            getattr(qr_a, "valid_depth_points", 0) or 0
        ),
        "qr_b_valid_depth_points": int(
            getattr(qr_b, "valid_depth_points", 0) or 0
        ),
        "qr_a_valid_depth_ratio": _number(
            getattr(qr_a, "valid_depth_ratio", None)
        ),
        "qr_b_valid_depth_ratio": _number(
            getattr(qr_b, "valid_depth_ratio", None)
        ),
        "qr_a_spatial_quadrants": int(
            getattr(qr_a, "spatial_quadrants", 0) or 0
        ),
        "qr_b_spatial_quadrants": int(
            getattr(qr_b, "spatial_quadrants", 0) or 0
        ),
        "temporal_sample_count": int(getattr(temporal, "count", 0) or 0),
        "temporal_ready": bool(getattr(temporal, "ready", False)),
        "temporal_median_mm": _number(
            getattr(temporal, "median_m", None), 1000.0
        ),
        "temporal_mean_mm": _number(
            getattr(temporal, "mean_m", None), 1000.0
        ),
        "temporal_std_mm": _number(getattr(temporal, "std_m", None), 1000.0),
        "temporal_mad_mm": _number(getattr(temporal, "mad_m", None), 1000.0),
        "method_a_distance_mm": _number(method_a_distance_m, 1000.0),
        "method_b_distance_mm": _number(method_b_distance_m, 1000.0),
        "method_c_distance_mm": _number(method_c_distance_m, 1000.0),
        "status": str(status),
        "reject_reason": reject_reason or "",
    }


class CSVRecorder:
    def __init__(self, path: str | Path, flush_every_row: bool = True):
        self.path = Path(path)
        self.flush_every_row = flush_every_row
        self._stream: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self.enabled = True

    def open(self) -> None:
        if self._stream is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._stream.flush()

    def write(self, row: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        if self._writer is None:
            self.open()
        assert self._writer is not None
        self._writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        if self.flush_every_row and self._stream is not None:
            self._stream.flush()

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = None
        self._writer = None

    def __enter__(self) -> "CSVRecorder":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
