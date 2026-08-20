"""Unit tests for ray-plane intersection."""

from __future__ import annotations

import unittest

import numpy as np

from geometry.plane_svd import PlaneModel
from geometry.ray_plane import intersect_ray_plane


def make_plane(normal: list[float], d: float) -> PlaneModel:
    return PlaneModel(
        normal=np.asarray(normal, dtype=np.float64),
        d=float(d),
        centroid=np.zeros(3, dtype=np.float64),
        inlier_mask=np.ones(3, dtype=bool),
        inlier_count=3,
        inlier_ratio=1.0,
        rms=0.0,
    )


class RayPlaneTests(unittest.TestCase):
    def test_intersects_z_equals_one(self) -> None:
        plane = make_plane([0.0, 0.0, 1.0], -1.0)
        ray = np.array([0.2, 0.1, 1.0])

        point = intersect_ray_plane(ray, plane)

        self.assertIsNotNone(point)
        np.testing.assert_allclose(point, [0.2, 0.1, 1.0], atol=1.0e-12)

    def test_parallel_ray_returns_none(self) -> None:
        plane = make_plane([0.0, 0.0, 1.0], -1.0)
        self.assertIsNone(intersect_ray_plane([1.0, 0.0, 0.0], plane))

    def test_intersection_behind_camera_returns_none(self) -> None:
        plane = make_plane([0.0, 0.0, 1.0], 1.0)
        self.assertIsNone(intersect_ray_plane([0.0, 0.0, 1.0], plane))

    def test_ray_scale_does_not_change_point(self) -> None:
        plane = make_plane([0.0, 0.0, 1.0], -2.0)
        first = intersect_ray_plane([0.1, -0.2, 1.0], plane)
        second = intersect_ray_plane([0.2, -0.4, 2.0], plane)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        np.testing.assert_allclose(first, second, atol=1.0e-12)

    def test_invalid_ray_returns_none(self) -> None:
        plane = make_plane([0.0, 0.0, 1.0], -1.0)
        self.assertIsNone(intersect_ray_plane([0.0, 0.0, 0.0], plane))
        self.assertIsNone(intersect_ray_plane([0.0, np.nan, 1.0], plane))


if __name__ == "__main__":
    unittest.main()
