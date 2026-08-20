"""Unit tests for projective QR quadrilateral helpers."""

from __future__ import annotations

import unittest

import numpy as np

from geometry.quadrilateral import (
    edge_lengths,
    is_valid_convex_quadrilateral,
    polygon_mask,
    quadrilateral_center,
    shrink_quadrilateral,
)


class QuadrilateralCenterTests(unittest.TestCase):
    def test_standard_rectangle(self) -> None:
        corners = np.array(
            [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]]
        )
        center = quadrilateral_center(corners)
        self.assertIsNotNone(center)
        np.testing.assert_allclose(center, [30.0, 30.0], atol=1.0e-12)

    def test_perspective_quadrilateral_uses_diagonal_intersection(self) -> None:
        # Both diagonals were constructed to cross at exactly (2, 1.5), while
        # the arithmetic mean of the four corners is a different point.
        corners = np.array(
            [[0.0, 0.0], [5.0, 0.0], [3.0, 2.25], [-0.25, 2.625]]
        )
        center = quadrilateral_center(corners)
        self.assertIsNotNone(center)
        np.testing.assert_allclose(center, [2.0, 1.5], atol=1.0e-12)
        self.assertFalse(np.allclose(center, np.mean(corners, axis=0)))

    def test_extreme_coordinate_scale(self) -> None:
        offset = np.array([1.0e6, -2.0e6])
        corners = offset + np.array(
            [[-1000.0, -0.5], [1500.0, -0.4], [700.0, 0.8], [-1300.0, 0.6]]
        )
        center = quadrilateral_center(corners)
        self.assertIsNotNone(center)
        self.assertTrue(np.all(np.isfinite(center)))
        # The returned point must lie on both diagonals.
        first_direction = corners[2] - corners[0]
        first_offset = center - corners[0]
        first_cross = (
            first_direction[0] * first_offset[1]
            - first_direction[1] * first_offset[0]
        )
        second_direction = corners[3] - corners[1]
        second_offset = center - corners[1]
        second_cross = (
            second_direction[0] * second_offset[1]
            - second_direction[1] * second_offset[0]
        )
        self.assertAlmostEqual(float(first_cross), 0.0, places=6)
        self.assertAlmostEqual(float(second_cross), 0.0, places=6)

    def test_degenerate_quadrilateral_returns_none(self) -> None:
        corners = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        )
        self.assertIsNone(quadrilateral_center(corners))

    def test_nonfinite_or_wrong_shape_returns_none(self) -> None:
        self.assertIsNone(quadrilateral_center(np.zeros((3, 2))))
        malformed = np.zeros((4, 2))
        malformed[2, 1] = np.nan
        self.assertIsNone(quadrilateral_center(malformed))


class QuadrilateralRoiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corners = np.array(
            [[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]]
        )

    def test_edge_lengths_and_convexity(self) -> None:
        self.assertTrue(is_valid_convex_quadrilateral(self.corners))
        lengths = edge_lengths(self.corners)
        self.assertIsNotNone(lengths)
        np.testing.assert_allclose(lengths, [6.0, 6.0, 6.0, 6.0])

    def test_shrink_keeps_center_and_scales_corners(self) -> None:
        shrunk = shrink_quadrilateral(self.corners, 0.5)
        expected = np.array(
            [[3.5, 3.5], [6.5, 3.5], [6.5, 6.5], [3.5, 6.5]]
        )
        np.testing.assert_allclose(shrunk, expected)
        np.testing.assert_allclose(quadrilateral_center(shrunk), [5.0, 5.0])

    def test_mask_is_boolean_and_fills_polygon(self) -> None:
        mask = polygon_mask((12, 12), self.corners)
        self.assertEqual(mask.shape, (12, 12))
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertTrue(mask[5, 5])
        self.assertFalse(mask[0, 0])

    def test_invalid_shrink_ratio_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            shrink_quadrilateral(self.corners, 0.0)


if __name__ == "__main__":
    unittest.main()
