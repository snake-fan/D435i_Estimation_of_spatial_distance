from __future__ import annotations

import unittest

import numpy as np

from geometry.plane_svd import PlaneModel
from measurement.quality_gate import (
    MeasurementStatus,
    QualityGate,
    order_reject_reasons,
)


def plane(*, ratio: float = 0.9, rms: float = 0.001) -> PlaneModel:
    points = 10
    mask = np.ones(points, dtype=bool)
    return PlaneModel(
        normal=np.array([0.0, 0.0, 1.0]),
        d=-1.0,
        centroid=np.array([0.0, 0.0, 1.0]),
        inlier_mask=mask,
        inlier_count=int(round(points * ratio)),
        inlier_ratio=ratio,
        rms=rms,
    )


class RejectReasonOrderingTests(unittest.TestCase):
    def test_reasons_are_deduplicated_in_stage_order(self) -> None:
        self.assertEqual(
            order_reject_reasons(
                [
                    "invalid_intersection",
                    "low_inlier_ratio",
                    "qr_too_small",
                    "low_inlier_ratio",
                ]
            ),
            ("qr_too_small", "low_inlier_ratio", "invalid_intersection"),
        )


class QualityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = QualityGate(
            min_edge_pixels=35.0,
            min_points=20,
            min_inlier_ratio=0.65,
            max_rms=0.003,
            min_depth=0.25,
            max_depth=2.0,
            warning_tilt_deg=45.0,
            max_tilt_deg=60.0,
        )

    def assess(self, **overrides: object):
        arguments: dict[str, object] = {
            "qr_min_edge_pixels": 50.0,
            "valid_depth_points": 100,
            "plane": plane(),
            "tilt_deg": 10.0,
            "point_xyz": np.array([0.0, 0.0, 1.0]),
        }
        arguments.update(overrides)
        return self.gate.assess(**arguments)

    def test_good_and_warning_statuses_are_valid(self) -> None:
        good = self.assess()
        warning = self.assess(tilt_deg=45.0)
        self.assertEqual(good.status, MeasurementStatus.GOOD)
        self.assertTrue(good.valid)
        self.assertEqual(warning.status, MeasurementStatus.WARNING)
        self.assertTrue(warning.valid)

    def test_missing_qr_has_only_missing_reason(self) -> None:
        assessment = self.gate.assess(qr_found=False)
        self.assertEqual(assessment.status, MeasurementStatus.INVALID)
        self.assertEqual(assessment.reject_reasons, ("qr_not_found",))

    def test_low_ratio_is_not_misreported_as_ransac_failure(self) -> None:
        assessment = self.assess(plane=plane(ratio=0.5))
        self.assertEqual(assessment.reject_reasons, ("low_inlier_ratio",))
        self.assertNotIn("ransac_failed", assessment.reject_reasons)

    def test_depth_density_and_spatial_support_are_hard_gates(self) -> None:
        assessment = self.assess(
            valid_depth_ratio=0.1,
            spatial_quadrants=1,
        )
        self.assertEqual(
            assessment.reject_reasons,
            ("low_valid_depth_ratio", "poor_spatial_support"),
        )

    def test_absent_plane_is_ransac_failure(self) -> None:
        assessment = self.assess(plane=None)
        self.assertEqual(assessment.reject_reasons, ("ransac_failed",))

    def test_multiple_failures_follow_pipeline_order(self) -> None:
        assessment = self.assess(
            qr_min_edge_pixels=10.0,
            valid_depth_points=5,
            plane=plane(ratio=0.5, rms=0.01),
            tilt_deg=70.0,
            point_xyz=None,
        )
        self.assertEqual(
            assessment.reject_reasons,
            (
                "qr_too_small",
                "not_enough_depth_points",
                "low_inlier_ratio",
                "plane_rms_too_large",
                "qr_tilt_too_large",
                "invalid_intersection",
            ),
        )

    def test_out_of_range_depth_is_distinct_from_bad_intersection(self) -> None:
        assessment = self.assess(point_xyz=np.array([0.0, 0.0, 3.0]))
        self.assertEqual(assessment.reject_reasons, ("out_of_depth_range",))


if __name__ == "__main__":
    unittest.main()
