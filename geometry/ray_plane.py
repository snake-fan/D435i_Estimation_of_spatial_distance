"""Camera-ray and plane intersection."""

from __future__ import annotations

import numpy as np

from .plane_svd import PlaneModel


def intersect_ray_plane(
    ray: np.ndarray,
    plane: PlaneModel,
    *,
    epsilon: float = 1.0e-10,
) -> np.ndarray | None:
    """Intersect the origin ray ``P(t)=t*ray`` with ``plane`` for ``t > 0``."""

    direction = np.asarray(ray, dtype=np.float64)
    normal = np.asarray(plane.normal, dtype=np.float64)
    if (
        direction.shape != (3,)
        or normal.shape != (3,)
        or not np.all(np.isfinite(direction))
        or not np.all(np.isfinite(normal))
        or not np.isfinite(plane.d)
        or not np.isfinite(epsilon)
        or epsilon <= 0.0
    ):
        return None

    ray_norm = float(np.linalg.norm(direction))
    normal_norm = float(np.linalg.norm(normal))
    if ray_norm == 0.0 or normal_norm == 0.0:
        return None
    denominator = float(normal @ direction)
    if abs(denominator) <= float(epsilon) * normal_norm * ray_norm:
        return None

    t = -float(plane.d) / denominator
    if not np.isfinite(t) or t <= 0.0:
        return None
    intersection = t * direction
    if not np.all(np.isfinite(intersection)):
        return None
    return intersection


__all__ = ["intersect_ray_plane"]
