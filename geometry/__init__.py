"""Camera-independent geometry primitives for QR distance measurement."""

from .deprojection import deproject_pixel, deproject_pixel_to_ray, deproject_roi
from .plane_ransac import fit_plane_ransac
from .plane_svd import PlaneModel, fit_plane_svd
from .quadrilateral import (
    edge_lengths,
    is_valid_convex_quadrilateral,
    polygon_area,
    polygon_mask,
    quadrilateral_center,
    shrink_quadrilateral,
)
from .ray_plane import intersect_ray_plane

__all__ = [
    "PlaneModel",
    "deproject_pixel",
    "deproject_pixel_to_ray",
    "deproject_roi",
    "edge_lengths",
    "fit_plane_ransac",
    "fit_plane_svd",
    "intersect_ray_plane",
    "is_valid_convex_quadrilateral",
    "polygon_area",
    "polygon_mask",
    "quadrilateral_center",
    "shrink_quadrilateral",
]
