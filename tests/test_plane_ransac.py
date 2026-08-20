"""Unit tests for robust RANSAC plane recovery."""

from __future__ import annotations

import unittest

import numpy as np

from geometry.plane_ransac import fit_plane_ransac


class PlaneRansacTests(unittest.TestCase):
    def test_horizontal_plane_with_noise_and_outliers(self) -> None:
        generator = np.random.default_rng(1234)
        xy = generator.uniform(-0.5, 0.5, size=(1000, 2))
        z = 1.0 + generator.normal(0.0, 0.001, size=1000)
        inliers = np.column_stack((xy, z))
        outliers = generator.uniform([-0.5, -0.5, 0.3], [0.5, 0.5, 1.7], size=(100, 3))
        points = np.vstack((inliers, outliers))

        plane = fit_plane_ransac(
            points,
            iterations=200,
            distance_threshold=0.004,
            min_inlier_ratio=0.8,
            random_seed=11,
        )

        self.assertIsNotNone(plane)
        assert plane is not None
        self.assertGreater(plane.inlier_ratio, 0.8)
        self.assertLess(plane.rms, 0.002)
        self.assertGreaterEqual(plane.normal[2], 0.0)
        self.assertGreater(float(plane.normal @ np.array([0.0, 0.0, 1.0])), 0.999)
        plane_z = -plane.d / plane.normal[2]
        self.assertAlmostEqual(float(plane_z), 1.0, delta=0.001)

    def test_tilted_plane(self) -> None:
        generator = np.random.default_rng(9)
        normal = np.array([0.2, 0.3, 0.932], dtype=np.float64)
        xy = generator.uniform(-0.4, 0.4, size=(500, 2))
        z = (1.0 - normal[0] * xy[:, 0] - normal[1] * xy[:, 1]) / normal[2]
        z += generator.normal(0.0, 0.0005, size=z.size)
        inliers = np.column_stack((xy, z))
        outliers = generator.uniform(-1.0, 1.0, size=(80, 3))
        points = np.vstack((inliers, outliers))

        plane = fit_plane_ransac(
            points,
            iterations=200,
            distance_threshold=0.003,
            random_seed=5,
        )

        self.assertIsNotNone(plane)
        assert plane is not None
        expected = normal / np.linalg.norm(normal)
        self.assertGreater(float(plane.normal @ expected), 0.9999)
        self.assertLess(plane.rms, 0.001)

    def test_low_ratio_model_is_returned_for_quality_gate(self) -> None:
        generator = np.random.default_rng(321)
        xy = generator.uniform(-0.2, 0.2, size=(40, 2))
        plane_points = np.column_stack((xy, np.ones(40)))
        outliers = generator.uniform([-1.0, -1.0, 0.0], [1.0, 1.0, 2.0], size=(160, 3))
        points = np.vstack((plane_points, outliers))

        plane = fit_plane_ransac(
            points,
            iterations=300,
            distance_threshold=0.001,
            min_inlier_ratio=0.9,
            random_seed=44,
        )

        self.assertIsNotNone(plane)
        assert plane is not None
        self.assertLess(plane.inlier_ratio, 0.9)
        self.assertGreaterEqual(plane.inlier_count, 40)

    def test_seed_makes_result_reproducible(self) -> None:
        generator = np.random.default_rng(77)
        points = generator.normal(size=(100, 3))
        first = fit_plane_ransac(points, 40, 0.05, random_seed=8)
        second = fit_plane_ransac(points, 40, 0.05, random_seed=8)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        np.testing.assert_array_equal(first.inlier_mask, second.inlier_mask)
        np.testing.assert_allclose(first.normal, second.normal, atol=0.0, rtol=0.0)

    def test_svd_refit_reclassifies_threshold_boundary_points(self) -> None:
        class FixedSampleRng:
            def choice(
                self,
                _population: np.ndarray,
                *,
                size: int,
                replace: bool,
            ) -> np.ndarray:
                if size != 3 or replace:
                    raise AssertionError("RANSAC must request three unique points")
                return np.array([0, 1, 2])

        xy = np.array(
            [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]
        )
        base = np.column_stack((xy, np.zeros(4)))
        upper = np.tile(np.column_stack((xy, np.full(4, 0.9))), (2, 1))
        points = np.vstack((base, upper, [0.0, 0.0, -0.5], [0.0, 0.0, 1.1]))
        threshold = 1.0

        plane = fit_plane_ransac(
            points,
            iterations=1,
            distance_threshold=threshold,
            rng=FixedSampleRng(),
        )

        self.assertIsNotNone(plane)
        assert plane is not None
        residuals = np.abs(points @ plane.normal + plane.d)
        expected_mask = residuals < threshold
        np.testing.assert_array_equal(plane.inlier_mask, expected_mask)
        self.assertFalse(plane.inlier_mask[-2])
        self.assertTrue(plane.inlier_mask[-1])
        self.assertEqual(plane.inlier_count, int(np.count_nonzero(expected_mask)))
        self.assertAlmostEqual(
            plane.inlier_ratio,
            float(np.count_nonzero(expected_mask) / len(points)),
        )
        self.assertAlmostEqual(
            plane.rms,
            float(np.sqrt(np.mean(np.square(residuals[expected_mask])))),
        )

    def test_all_collinear_points_fail(self) -> None:
        values = np.linspace(0.0, 1.0, 30)
        points = np.column_stack((values, 2.0 * values, -values))
        self.assertIsNone(
            fit_plane_ransac(points, 50, 0.004, random_seed=1)
        )


if __name__ == "__main__":
    unittest.main()
