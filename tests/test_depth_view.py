from __future__ import annotations

import unittest

import numpy as np

try:
    from visualization.depth_view import colorize_depth
except ImportError:  # pragma: no cover - depends on local OpenCV installation
    colorize_depth = None  # type: ignore[assignment]


class DepthViewTests(unittest.TestCase):
    @unittest.skipIf(colorize_depth is None, "OpenCV is not importable")
    def test_colorize_depth_marks_invalid_pixels(self) -> None:
        assert colorize_depth is not None
        depth = np.array([[0, 1000], [500, 2500]], dtype=np.uint16)
        result = colorize_depth(depth, 0.001, 0.25, 2.0)
        self.assertEqual(result.shape, (2, 2, 3))
        self.assertTrue(np.array_equal(result[0, 0], np.array([40, 40, 120])))
        self.assertTrue(np.array_equal(result[1, 1], np.array([40, 40, 120])))


if __name__ == "__main__":
    unittest.main()
