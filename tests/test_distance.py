from __future__ import annotations

import unittest

import numpy as np

from measurement.distance_measure import DistanceMeasure, euclidean_distance
from measurement.qr_3d_locator import QR3DResult
from measurement.quality_gate import MeasurementStatus


class EuclideanDistanceTests(unittest.TestCase):
    def test_three_four_five_triangle(self) -> None:
        distance = euclidean_distance(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.3, 0.4, 0.0]),
        )
        self.assertAlmostEqual(distance, 0.5)

    def test_rejects_nonfinite_or_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            euclidean_distance([0.0, 0.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            euclidean_distance([0.0, 0.0, np.nan], [1.0, 2.0, 3.0])


class DistanceMeasureTests(unittest.TestCase):
    @staticmethod
    def result(
        qr_id: str,
        point: list[float],
        status: MeasurementStatus = MeasurementStatus.GOOD,
    ) -> QR3DResult:
        return QR3DResult(
            qr_id=qr_id,
            point_xyz=np.asarray(point, dtype=np.float64),
            status=status,
        )

    def test_computes_only_from_two_valid_qrs(self) -> None:
        result = DistanceMeasure().compute(
            self.result("QR_A", [0.0, 0.0, 1.0]),
            self.result("QR_B", [0.0, 0.5, 1.0]),
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.status, MeasurementStatus.GOOD)
        self.assertAlmostEqual(result.distance_m or -1.0, 0.5)
        self.assertIsNone(result.reject_reason)

    def test_warning_propagates_to_pair(self) -> None:
        result = DistanceMeasure().compute(
            self.result("QR_A", [0.0, 0.0, 1.0], MeasurementStatus.WARNING),
            self.result("QR_B", [0.0, 0.5, 1.0]),
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.status, MeasurementStatus.WARNING)

    def test_invalid_reasons_are_a_then_b_and_distance_is_none(self) -> None:
        qr_a = QR3DResult.missing("QR_A")
        qr_b = QR3DResult(
            qr_id="QR_B",
            status=MeasurementStatus.INVALID,
            reject_reasons=("plane_rms_too_large", "low_inlier_ratio"),
        )
        result = DistanceMeasure().compute(qr_a, qr_b)
        self.assertFalse(result.valid)
        self.assertIsNone(result.distance_m)
        self.assertEqual(
            result.reject_reasons,
            (
                "QR_A:qr_not_found",
                "QR_B:low_inlier_ratio",
                "QR_B:plane_rms_too_large",
            ),
        )

    def test_none_result_is_explicitly_rejected(self) -> None:
        result = DistanceMeasure().compute(
            None,
            self.result("QR_B", [0.0, 0.0, 1.0]),
        )
        self.assertEqual(result.status, MeasurementStatus.INVALID)
        self.assertEqual(result.reject_reasons, ("QR_A:qr_not_found",))


if __name__ == "__main__":
    unittest.main()
