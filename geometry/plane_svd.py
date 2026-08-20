"""Least-squares plane fitting using singular-value decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneModel:
    """Normalized plane ``normal @ point + d = 0`` and fit diagnostics."""

    normal: np.ndarray
    d: float
    centroid: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    inlier_ratio: float
    rms: float


def _canonicalize_normal(
    normal: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Choose a deterministic sign, with the required ``z >= 0`` convention."""

    canonical = np.asarray(normal, dtype=np.float64)
    if canonical[2] < 0.0:
        return -canonical
    if abs(canonical[2]) <= epsilon:
        for component in canonical:
            if abs(component) > epsilon:
                if component < 0.0:
                    return -canonical
                break
    return canonical


def fit_plane_svd(
    points: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    *,
    degenerate_epsilon: float = 1.0e-12,
) -> PlaneModel | None:
    """Fit a plane to all selected finite points, returning ``None`` if degenerate."""

    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1:] != (3,):
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(degenerate_epsilon) or degenerate_epsilon <= 0.0:
        raise ValueError("degenerate_epsilon must be positive and finite")

    point_count = cloud.shape[0]
    if inlier_mask is None:
        selected = np.all(np.isfinite(cloud), axis=1)
    else:
        selected = np.asarray(inlier_mask, dtype=bool)
        if selected.shape != (point_count,):
            raise ValueError("inlier_mask must have shape (N,)")
        selected = selected.copy() & np.all(np.isfinite(cloud), axis=1)

    selected_points = cloud[selected]
    if selected_points.shape[0] < 3:
        return None

    centroid = np.mean(selected_points, axis=0)
    centered = selected_points - centroid
    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular_values.size < 2 or singular_values[1] <= degenerate_epsilon:
        return None

    normal = np.asarray(vt[-1], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm <= degenerate_epsilon:
        return None
    normal = _canonicalize_normal(normal / normal_norm, epsilon=degenerate_epsilon)
    d = -float(normal @ centroid)
    residuals = np.abs(selected_points @ normal + d)
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    inlier_count = int(np.count_nonzero(selected))

    return PlaneModel(
        normal=normal,
        d=d,
        centroid=centroid,
        inlier_mask=selected,
        inlier_count=inlier_count,
        inlier_ratio=(float(inlier_count / point_count) if point_count else 0.0),
        rms=rms,
    )


__all__ = ["PlaneModel", "fit_plane_svd"]
