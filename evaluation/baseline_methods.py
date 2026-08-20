"""Research-only depth baselines; the production path always uses plane fitting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from geometry.deprojection import deproject_pixel


Deprojector = Callable[[Any, list[float], float], list[float] | tuple[float, ...] | np.ndarray]


def _deproject(
    intrinsics: Any,
    center_uv: np.ndarray,
    depth_m: float,
    deprojector: Deprojector | None,
) -> np.ndarray | None:
    return deproject_pixel(
        intrinsics,
        center_uv,
        depth_m,
        deprojector=deprojector,
    )


def center_single_pixel_point(
    depth_image: np.ndarray,
    center_uv: np.ndarray,
    intrinsics: Any,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    deprojector: Deprojector | None = None,
) -> np.ndarray | None:
    """Method A: use the nearest depth pixel at the detected QR center."""

    image = np.asarray(depth_image)
    center = np.asarray(center_uv, dtype=np.float64)
    if image.ndim != 2 or center.shape != (2,) or not np.isfinite(center).all():
        return None
    u, v = (int(round(center[0])), int(round(center[1])))
    if u < 0 or v < 0 or v >= image.shape[0] or u >= image.shape[1]:
        return None
    depth_m = float(image[v, u]) * float(depth_scale)
    if not np.isfinite(depth_m) or not min_depth <= depth_m <= max_depth:
        return None
    return _deproject(intrinsics, center, depth_m, deprojector)


def roi_median_depth_point(
    depth_image: np.ndarray,
    roi_mask: np.ndarray,
    center_uv: np.ndarray,
    intrinsics: Any,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    deprojector: Deprojector | None = None,
) -> np.ndarray | None:
    """Method B: project the center using the median valid depth in the QR ROI."""

    image = np.asarray(depth_image)
    mask = np.asarray(roi_mask, dtype=bool)
    center = np.asarray(center_uv, dtype=np.float64)
    if (
        image.ndim != 2
        or mask.shape != image.shape
        or center.shape != (2,)
        or not np.isfinite(center).all()
    ):
        return None
    depths = image[mask].astype(np.float64) * float(depth_scale)
    valid = np.isfinite(depths) & (depths >= min_depth) & (depths <= max_depth)
    if not valid.any():
        return None
    median_depth = float(np.median(depths[valid]))
    return _deproject(intrinsics, center, median_depth, deprojector)
