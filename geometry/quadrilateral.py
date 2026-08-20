"""Image-space quadrilateral geometry used by QR localization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _corners_array(corners: np.ndarray) -> np.ndarray | None:
    array = np.asarray(corners, dtype=np.float64)
    if array.shape != (4, 2) or not np.all(np.isfinite(array)):
        return None
    return array


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def polygon_area(corners: np.ndarray) -> float:
    """Return the unsigned shoelace area, or zero for malformed input."""

    points = _corners_array(corners)
    if points is None:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def edge_lengths(corners: np.ndarray) -> np.ndarray | None:
    """Return four perimeter edge lengths for ordered corners."""

    points = _corners_array(corners)
    if points is None:
        return None
    return np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)


def is_valid_convex_quadrilateral(
    corners: np.ndarray,
    *,
    epsilon: float = 1.0e-9,
) -> bool:
    """Return whether four perimeter-ordered points form a convex polygon."""

    points = _corners_array(corners)
    if points is None or epsilon <= 0.0:
        return False

    edges = np.roll(points, -1, axis=0) - points
    crosses = np.array(
        [_cross_2d(edges[index], edges[(index + 1) % 4]) for index in range(4)]
    )
    scale = max(float(np.max(np.linalg.norm(edges, axis=1))) ** 2, 1.0)
    tolerance = epsilon * scale
    if np.any(np.abs(crosses) <= tolerance):
        return False
    if not (np.all(crosses > 0.0) or np.all(crosses < 0.0)):
        return False
    return polygon_area(points) > tolerance


def quadrilateral_center(
    corners: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> np.ndarray | None:
    """Return the intersection of diagonals ``q0-q2`` and ``q1-q3``.

    The operation uses the strict projective definition of a QR center rather
    than averaging its projected corners.  Parallel or otherwise degenerate
    diagonals return ``None``.
    """

    points = _corners_array(corners)
    if points is None or epsilon <= 0.0:
        return None

    first_origin = points[0]
    first_direction = points[2] - points[0]
    second_origin = points[1]
    second_direction = points[3] - points[1]
    denominator = _cross_2d(first_direction, second_direction)
    direction_scale = float(
        np.linalg.norm(first_direction) * np.linalg.norm(second_direction)
    )
    if direction_scale == 0.0 or abs(denominator) <= epsilon * direction_scale:
        return None

    offset = second_origin - first_origin
    first_t = _cross_2d(offset, second_direction) / denominator
    second_t = _cross_2d(offset, first_direction) / denominator
    tolerance = max(epsilon * 10.0, np.finfo(np.float64).eps * 10.0)
    if (
        first_t < -tolerance
        or first_t > 1.0 + tolerance
        or second_t < -tolerance
        or second_t > 1.0 + tolerance
    ):
        return None

    center = first_origin + first_t * first_direction
    if not np.all(np.isfinite(center)):
        return None
    return center.astype(np.float64, copy=False)


def shrink_quadrilateral(
    corners: np.ndarray,
    ratio: float,
    *,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Shrink ordered QR corners toward their diagonal-intersection center."""

    points = _corners_array(corners)
    if points is None:
        raise ValueError("corners must be a finite array with shape (4, 2)")
    if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")

    if center is None:
        resolved_center = quadrilateral_center(points)
        if resolved_center is None:
            raise ValueError("quadrilateral diagonals do not have a valid intersection")
    else:
        resolved_center = np.asarray(center, dtype=np.float64)
        if resolved_center.shape != (2,) or not np.all(np.isfinite(resolved_center)):
            raise ValueError("center must be a finite array with shape (2,)")

    return resolved_center + float(ratio) * (points - resolved_center)


def _numpy_convex_fill(height: int, width: int, corners: np.ndarray) -> np.ndarray:
    """Small NumPy fallback for environments with unavailable OpenCV wheels."""

    min_x = max(0, int(np.floor(np.min(corners[:, 0]))))
    max_x = min(width - 1, int(np.ceil(np.max(corners[:, 0]))))
    min_y = max(0, int(np.floor(np.min(corners[:, 1]))))
    max_y = min(height - 1, int(np.ceil(np.max(corners[:, 1]))))
    mask = np.zeros((height, width), dtype=bool)
    if min_x > max_x or min_y > max_y:
        return mask

    xs, ys = np.meshgrid(
        np.arange(min_x, max_x + 1, dtype=np.float64),
        np.arange(min_y, max_y + 1, dtype=np.float64),
    )
    sample_points = np.stack((xs, ys), axis=-1)
    signs: list[np.ndarray] = []
    for start, end in zip(corners, np.roll(corners, -1, axis=0), strict=True):
        edge = end - start
        relative = sample_points - start
        signs.append(edge[0] * relative[..., 1] - edge[1] * relative[..., 0])
    signed = np.stack(signs, axis=0)
    inside = np.all(signed >= -1.0e-9, axis=0) | np.all(signed <= 1.0e-9, axis=0)
    mask[min_y : max_y + 1, min_x : max_x + 1] = inside
    return mask


def polygon_mask(
    image_shape: Sequence[int],
    corners: np.ndarray,
) -> np.ndarray:
    """Rasterize a convex polygon as a boolean ROI mask.

    ``cv2.fillConvexPoly`` is used in normal installations.  A deterministic
    NumPy fallback keeps offline geometry tests usable if a local OpenCV binary
    cannot be loaded.
    """

    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    points = _corners_array(corners)
    if points is None or not is_valid_convex_quadrilateral(points):
        raise ValueError("corners must form a finite convex quadrilateral")

    integer_points = np.rint(points).astype(np.int32)
    try:
        import cv2

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, integer_points, 1)
        return mask.astype(bool)
    except Exception:  # pragma: no cover - exercised only by broken/missing wheels
        return _numpy_convex_fill(height, width, integer_points.astype(np.float64))


__all__ = [
    "edge_lengths",
    "is_valid_convex_quadrilateral",
    "polygon_area",
    "polygon_mask",
    "quadrilateral_center",
    "shrink_quadrilateral",
]
