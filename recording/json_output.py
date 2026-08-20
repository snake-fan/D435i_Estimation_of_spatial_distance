"""Stable JSON serialization for per-frame measurement results."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, TextIO

import numpy as np


def _number(value: Any, scale: float = 1.0) -> float | None:
    if value is None:
        return None
    try:
        result = float(value) * scale
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _point(value: Any) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        return None
    return [float(item) for item in array]


def _qr_payload(qr_id: str, result: Any) -> dict[str, Any]:
    plane = getattr(result, "plane", None)
    return {
        "id": qr_id,
        "valid": bool(getattr(result, "valid", False)),
        "status": str(getattr(result, "status", "INVALID")),
        "point_m": _point(getattr(result, "point_xyz", None)),
        "plane_rms_mm": _number(getattr(plane, "rms", None), 1000.0),
        "inlier_ratio": _number(getattr(plane, "inlier_ratio", None)),
        "tilt_deg": _number(getattr(result, "tilt_deg", None)),
        "min_edge_pixels": _number(
            getattr(result, "qr_min_edge_pixels", None)
        ),
        "valid_depth_points": int(
            getattr(result, "valid_depth_points", 0) or 0
        ),
        "valid_depth_ratio": _number(
            getattr(result, "valid_depth_ratio", None)
        ),
        "spatial_quadrants": int(
            getattr(result, "spatial_quadrants", 0) or 0
        ),
        "reject_reason": getattr(result, "reject_reason", None),
    }


def build_json_result(
    *,
    source_timestamp_ms: float,
    frame_number: int,
    qr_results: Mapping[str, Any],
    expected_ids: list[str] | tuple[str, str],
    distance_result: Any = None,
    temporal: Any = None,
    status: str,
    reject_reason: str | None = None,
) -> dict[str, Any]:
    """Build the documented JSON object without emitting NaN/Infinity."""

    if len(expected_ids) != 2:
        raise ValueError("Exactly two expected QR ids are required")
    qr_a_id, qr_b_id = expected_ids
    distance_m = getattr(distance_result, "distance_m", None)
    temporal_ready = bool(getattr(temporal, "ready", False)) if temporal else False
    return {
        "source_timestamp_ms": _number(source_timestamp_ms),
        "frame_number": int(frame_number),
        "qr_a": _qr_payload(qr_a_id, qr_results.get(qr_a_id)),
        "qr_b": _qr_payload(qr_b_id, qr_results.get(qr_b_id)),
        "distance_mm": _number(distance_m, 1000.0),
        "temporal": {
            "sample_count": int(getattr(temporal, "count", 0) or 0),
            "ready": temporal_ready,
            "median_mm": _number(getattr(temporal, "median_m", None), 1000.0),
            "mean_mm": _number(getattr(temporal, "mean_m", None), 1000.0),
            "std_mm": _number(getattr(temporal, "std_m", None), 1000.0),
            "mad_mm": _number(getattr(temporal, "mad_m", None), 1000.0),
        },
        "status": str(status),
        "reject_reason": reject_reason,
    }


class JsonLinesWriter:
    """Write one compact JSON object per frame to a file or stdout."""

    def __init__(self, destination: str | Path, flush_every_row: bool = True):
        self.destination = str(destination)
        self.flush_every_row = flush_every_row
        self._stream: TextIO | None = None
        self._owns_stream = False

    def open(self) -> None:
        if self._stream is not None:
            return
        if self.destination == "-":
            self._stream = sys.stdout
        else:
            path = Path(self.destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("x", encoding="utf-8")
            self._owns_stream = True

    def write(self, payload: Mapping[str, Any]) -> None:
        if self._stream is None:
            self.open()
        assert self._stream is not None
        self._stream.write(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        )
        if self.flush_every_row:
            self._stream.flush()

    def close(self) -> None:
        if self._stream is not None and self._owns_stream:
            self._stream.close()
        self._stream = None
        self._owns_stream = False

    def __enter__(self) -> "JsonLinesWriter":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
