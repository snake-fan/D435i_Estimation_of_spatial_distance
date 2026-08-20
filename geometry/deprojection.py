"""Depth ROI deprojection with an optional offline backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

Deprojector = Callable[[Any, list[float], float], Any]


class DeprojectionBackendError(RuntimeError):
    """Raised when valid depth samples exist but none can be deprojected."""


def _resolve_deprojector(deprojector: Deprojector | None) -> Deprojector:
    if deprojector is not None:
        return deprojector
    try:
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - environment specific
        raise DeprojectionBackendError(
            "pyrealsense2 is required for hardware deprojection; provide a "
            "deprojector callback for offline use"
        ) from exc
    return rs.rs2_deproject_pixel_to_point


def deproject_roi(
    depth_image: np.ndarray,
    mask: np.ndarray,
    intrinsics: Any,
    depth_scale: float,
    stride: int,
    min_depth: float,
    max_depth: float,
    *,
    deprojector: Deprojector | None = None,
    return_pixels: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert valid, sampled depth pixels under ``mask`` into ``N x 3`` metres.

    The callback follows ``rs2_deproject_pixel_to_point(intrinsics, [u, v], z)``.
    This injection point makes the function testable without librealsense.
    With ``return_pixels=True``, also return the matching ``N x 2`` ``[u, v]``
    coordinates so callers can verify spatial support of fitted inliers.
    """

    depth = np.asarray(depth_image)
    roi = np.asarray(mask)
    if depth.ndim != 2 or roi.shape != depth.shape:
        raise ValueError("depth_image and mask must be equally shaped 2-D arrays")
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be positive and finite")
    if isinstance(stride, bool) or int(stride) != stride or stride <= 0:
        raise ValueError("stride must be a positive integer")
    stride = int(stride)
    if (
        not np.isfinite(min_depth)
        or not np.isfinite(max_depth)
        or min_depth < 0.0
        or min_depth >= max_depth
    ):
        raise ValueError("depth range must satisfy 0 <= min_depth < max_depth")

    sampled_mask = np.zeros(depth.shape, dtype=bool)
    sampled_mask[::stride, ::stride] = np.asarray(roi[::stride, ::stride], dtype=bool)
    def empty_result() -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        empty_points = np.empty((0, 3), dtype=np.float64)
        if return_pixels:
            return empty_points, np.empty((0, 2), dtype=np.float64)
        return empty_points

    rows, columns = np.nonzero(sampled_mask)
    if rows.size == 0:
        return empty_result()

    raw_depths = np.asarray(depth[rows, columns], dtype=np.float64)
    depths_m = raw_depths * float(depth_scale)
    valid = (
        (raw_depths != 0.0)
        & np.isfinite(raw_depths)
        & np.isfinite(depths_m)
        & (depths_m >= float(min_depth))
        & (depths_m <= float(max_depth))
    )
    rows = rows[valid]
    columns = columns[valid]
    depths_m = depths_m[valid]
    if rows.size == 0:
        return empty_result()

    backend = _resolve_deprojector(deprojector)
    points: list[np.ndarray] = []
    pixels: list[tuple[float, float]] = []
    for row, column, depth_m in zip(rows, columns, depths_m, strict=True):
        try:
            point = np.asarray(
                backend(
                    intrinsics,
                    [float(column), float(row)],
                    float(depth_m),
                ),
                dtype=np.float64,
            )
        except (RuntimeError, TypeError, ValueError):
            continue
        if point.shape == (3,) and np.all(np.isfinite(point)):
            points.append(point)
            pixels.append((float(column), float(row)))

    if not points:
        raise DeprojectionBackendError(
            "deprojection backend produced no valid 3-D point for "
            f"{len(rows)} valid depth sample(s)"
        )
    point_array = np.vstack(points).astype(np.float64, copy=False)
    if return_pixels:
        return point_array, np.asarray(pixels, dtype=np.float64)
    return point_array


def deproject_pixel_to_ray(
    intrinsics: Any,
    pixel_uv: np.ndarray,
    *,
    deprojector: Deprojector | None = None,
) -> np.ndarray | None:
    """Create a camera ray by deprojecting ``pixel_uv`` at one metre depth."""

    pixel = np.asarray(pixel_uv, dtype=np.float64)
    if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
        return None
    backend = _resolve_deprojector(deprojector)
    try:
        ray = np.asarray(
            backend(intrinsics, [float(pixel[0]), float(pixel[1])], 1.0),
            dtype=np.float64,
        )
    except (RuntimeError, TypeError, ValueError):
        return None
    if ray.shape != (3,) or not np.all(np.isfinite(ray)) or np.linalg.norm(ray) == 0.0:
        return None
    return ray


def deproject_pixel(
    intrinsics: Any,
    pixel_uv: np.ndarray,
    depth_m: float,
    *,
    deprojector: Deprojector | None = None,
) -> np.ndarray | None:
    """Deproject one pixel at a metric depth through the SDK backend."""

    pixel = np.asarray(pixel_uv, dtype=np.float64)
    if (
        pixel.shape != (2,)
        or not np.all(np.isfinite(pixel))
        or not np.isfinite(depth_m)
        or depth_m <= 0.0
    ):
        return None
    backend = _resolve_deprojector(deprojector)
    try:
        point = np.asarray(
            backend(
                intrinsics,
                [float(pixel[0]), float(pixel[1])],
                float(depth_m),
            ),
            dtype=np.float64,
        )
    except (RuntimeError, TypeError, ValueError):
        return None
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    return point


__all__ = [
    "DeprojectionBackendError",
    "Deprojector",
    "deproject_pixel",
    "deproject_pixel_to_ray",
    "deproject_roi",
]
