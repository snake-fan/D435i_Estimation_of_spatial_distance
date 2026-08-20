from __future__ import annotations

import unittest

import numpy as np

from detection import QRDetector


class _Backend:
    def detectAndDecodeMulti(self, _image: np.ndarray):
        points = np.array(
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[20, 20], [60, 20], [60, 60], [20, 60]],
                [[70, 20], [100, 20], [100, 50], [70, 50]],
            ],
            dtype=np.float32,
        )
        return True, ("QR_A", "QR_A", "UNEXPECTED"), points, ()


class QRDetectorTests(unittest.TestCase):
    def test_filters_payloads_and_keeps_largest_duplicate(self) -> None:
        detector = QRDetector(
            ["QR_A", "QR_B"], refine_corners=False, detector=_Backend()
        )
        detections = detector.detect(np.zeros((120, 120), dtype=np.uint8))
        self.assertEqual(set(detections), {"QR_A"})
        self.assertAlmostEqual(detections["QR_A"].min_edge_length, 40.0)
        np.testing.assert_allclose(detections["QR_A"].center, [40.0, 40.0])

    def test_non_grayscale_input_is_rejected(self) -> None:
        detector = QRDetector(detector=_Backend())
        with self.assertRaises(ValueError):
            detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
