"""Unit tests for deterministic SVD plane fitting."""

from __future__ import annotations

import unittest

import numpy as np

from geometry.plane_svd import fit_plane_svd


class PlaneSvdTests(unittest.TestCase):
    def test_horizontal_plane(self) -> None:
        x, y = np.meshgrid(np.linspace(-0.5, 0.5, 11), np.linspace(-0.3, 0.3, 9))
        points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))

        plane = fit_plane_svd(points)

        self.assertIsNotNone(plane)
        assert plane is not None
        np.testing.assert_allclose(plane.normal, [0.0, 0.0, 1.0], atol=1.0e-12)
        self.assertAlmostEqual(plane.d, -1.0, places=12)
        np.testing.assert_allclose(plane.centroid, [0.0, 0.0, 1.0], atol=1.0e-12)
        self.assertEqual(plane.inlier_count, points.shape[0])
        self.assertAlmostEqual(plane.inlier_ratio, 1.0)
        self.assertLess(plane.rms, 1.0e-12)

    def test_tilted_plane_and_positive_z_convention(self) -> None:
        normal = np.array([0.2, 0.3, 0.932], dtype=np.float64)
        d = -1.0
        xs, ys = np.meshgrid(np.linspace(-0.3, 0.3, 12), np.linspace(-0.2, 0.2, 10))
        zs = -(normal[0] * xs + normal[1] * ys + d) / normal[2]
        points = np.column_stack((xs.ravel(), ys.ravel(), zs.ravel()))

        plane = fit_plane_svd(points)

        self.assertIsNotNone(plane)
        assert plane is not None
        expected_normal = normal / np.linalg.norm(normal)
        np.testing.assert_allclose(plane.normal, expected_normal, atol=1.0e-10)
        self.assertGreaterEqual(plane.normal[2], 0.0)
        self.assertAlmostEqual(plane.d, d / np.linalg.norm(normal), places=10)
        self.assertLess(plane.rms, 1.0e-12)

    def test_refit_uses_only_selected_inliers(self) -> None:
        plane_points = np.array(
            [
                [-1.0, -1.0, 2.0],
                [1.0, -1.0, 2.0],
                [1.0, 1.0, 2.0],
                [-1.0, 1.0, 2.0],
            ]
        )
        points = np.vstack((plane_points, [[0.0, 0.0, 10.0]]))
        mask = np.array([True, True, True, True, False])

        plane = fit_plane_svd(points, mask)

        self.assertIsNotNone(plane)
        assert plane is not None
        np.testing.assert_allclose(plane.normal, [0.0, 0.0, 1.0], atol=1.0e-12)
        self.assertAlmostEqual(plane.d, -2.0, places=12)
        self.assertEqual(plane.inlier_count, 4)
        self.assertAlmostEqual(plane.inlier_ratio, 0.8)
        np.testing.assert_array_equal(plane.inlier_mask, mask)

    def test_collinear_points_are_degenerate(self) -> None:
        values = np.linspace(0.0, 1.0, 10)
        points = np.column_stack((values, 2.0 * values, 3.0 * values))
        self.assertIsNone(fit_plane_svd(points))


if __name__ == "__main__":
    unittest.main()
