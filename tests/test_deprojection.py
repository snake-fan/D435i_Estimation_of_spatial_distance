from __future__ import annotations

import unittest

import numpy as np

from geometry.deprojection import (
    DeprojectionBackendError,
    deproject_pixel_to_ray,
    deproject_roi,
)


def _pinhole(_intrinsics: object, pixel: list[float], depth_m: float) -> list[float]:
    return [pixel[0] * depth_m, pixel[1] * depth_m, depth_m]


class DeprojectionTests(unittest.TestCase):
    def test_roi_filters_zero_and_out_of_range_depth(self) -> None:
        depth = np.array([[0, 1000], [200, 2500]], dtype=np.uint16)
        points = deproject_roi(
            depth,
            np.ones_like(depth, dtype=bool),
            object(),
            0.001,
            1,
            0.25,
            2.0,
            deprojector=_pinhole,
        )
        self.assertEqual(points.shape, (1, 3))
        np.testing.assert_allclose(points[0], [1.0, 0.0, 1.0])

    def test_center_ray_uses_unit_z_deprojection(self) -> None:
        ray = deproject_pixel_to_ray(
            object(), np.array([0.2, 0.1]), deprojector=_pinhole
        )
        np.testing.assert_allclose(ray, [0.2, 0.1, 1.0])

    def test_roi_can_return_matching_pixel_coordinates(self) -> None:
        depth = np.array([[1000, 0], [0, 1200]], dtype=np.uint16)
        points, pixels = deproject_roi(
            depth,
            np.ones_like(depth, dtype=bool),
            object(),
            0.001,
            1,
            0.25,
            2.0,
            deprojector=_pinhole,
            return_pixels=True,
        )
        self.assertEqual(points.shape, (2, 3))
        np.testing.assert_allclose(pixels, [[0.0, 0.0], [1.0, 1.0]])

    def test_all_backend_calls_failing_is_distinct_from_no_valid_depth(self) -> None:
        calls = 0

        def failing_backend(
            _intrinsics: object, _pixel: list[float], _depth_m: float
        ) -> list[float]:
            nonlocal calls
            calls += 1
            raise RuntimeError("synthetic backend failure")

        with self.assertRaises(DeprojectionBackendError):
            deproject_roi(
                np.full((2, 2), 1000, dtype=np.uint16),
                np.ones((2, 2), dtype=bool),
                object(),
                0.001,
                1,
                0.25,
                2.0,
                deprojector=failing_backend,
            )
        self.assertEqual(calls, 4)

        calls = 0
        points = deproject_roi(
            np.zeros((2, 2), dtype=np.uint16),
            np.ones((2, 2), dtype=bool),
            object(),
            0.001,
            1,
            0.25,
            2.0,
            deprojector=failing_backend,
        )
        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
