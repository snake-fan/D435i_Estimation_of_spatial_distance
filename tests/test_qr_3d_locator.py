from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from detection.qr_detector import QRDetection
from measurement.qr_3d_locator import QR3DLocator, QR3DResult
from measurement.quality_gate import MeasurementStatus


@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float = 100.0
    fy: float = 100.0
    ppx: float = 50.0
    ppy: float = 50.0


def pinhole_deproject(
    intrinsics: PinholeIntrinsics,
    pixel: list[float],
    depth_m: float,
) -> np.ndarray:
    u, v = pixel
    return np.array(
        [
            (u - intrinsics.ppx) * depth_m / intrinsics.fx,
            (v - intrinsics.ppy) * depth_m / intrinsics.fy,
            depth_m,
        ],
        dtype=np.float64,
    )


class SyntheticQR3DLocatorTests(unittest.TestCase):
    def setUp(self) -> None:
        corners = np.array(
            [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]],
            dtype=np.float64,
        )
        self.detection = QRDetection(
            payload="QR_A",
            corners=corners,
            center=np.array([50.0, 50.0]),
            min_edge_length=60.0,
        )
        self.depth = np.full((100, 100), 1_000, dtype=np.uint16)
        self.locator = QR3DLocator(
            polygon_shrink_ratio=0.75,
            min_edge_pixels=20.0,
            ransac_iterations=50,
            distance_threshold=0.001,
            min_points=20,
            min_inlier_ratio=0.9,
            max_rms=0.0005,
            sample_stride=2,
            min_depth=0.25,
            max_depth=2.0,
            random_seed=7,
            deprojector=pinhole_deproject,
        )

    def test_full_pipeline_recovers_center_on_one_meter_plane_without_sdk(self) -> None:
        result = self.locator.locate(
            self.detection,
            self.depth,
            PinholeIntrinsics(),
            0.001,
            qr_id="QR_A",
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.status, MeasurementStatus.GOOD)
        np.testing.assert_allclose(result.point_xyz, [0.0, 0.0, 1.0], atol=1.0e-10)
        self.assertIsNotNone(result.plane)
        self.assertGreater(result.valid_depth_points, 20)
        self.assertEqual(result.roi_points.shape, (result.valid_depth_points, 3))
        self.assertGreater(result.roi_sample_points, 0)
        self.assertAlmostEqual(result.valid_depth_ratio, 1.0)
        self.assertAlmostEqual(result.tilt_deg or 0.0, 0.0)

    def test_missing_qr_is_first_class_result(self) -> None:
        result = self.locator.locate(
            None,
            self.depth,
            PinholeIntrinsics(),
            0.001,
            qr_id="QR_B",
        )
        self.assertIsInstance(result, QR3DResult)
        self.assertFalse(result.valid)
        self.assertEqual(result.qr_id, "QR_B")
        self.assertEqual(result.reject_reasons, ("qr_not_found",))
        self.assertEqual(result.roi_points.shape, (0, 3))

    def test_zero_depth_retains_empty_cloud_and_specific_reason(self) -> None:
        result = self.locator.locate(
            self.detection,
            np.zeros_like(self.depth),
            PinholeIntrinsics(),
            0.001,
            qr_id="QR_A",
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reject_reasons, ("not_enough_depth_points",))
        self.assertEqual(result.valid_depth_points, 0)
        self.assertGreater(result.roi_sample_points, 0)

    def test_sparse_one_corner_depth_is_not_extrapolated_as_good(self) -> None:
        depth = np.zeros_like(self.depth)
        depth[28:50, 28:50] = 1_000
        result = self.locator.locate(
            self.detection,
            depth,
            PinholeIntrinsics(),
            0.001,
            qr_id="QR_A",
        )
        self.assertFalse(result.valid)
        self.assertIn("low_valid_depth_ratio", result.reject_reasons)
        self.assertIn("poor_spatial_support", result.reject_reasons)
        self.assertEqual(result.spatial_quadrants, 1)

    def test_ransac_inliers_must_cover_center_not_only_raw_depth(self) -> None:
        depth = np.zeros_like(self.depth)
        depth[28:50, 28:50] = 1_000
        depth[28:34, 52:58] = 1_500
        depth[52:58, 28:34] = 1_500
        depth[52:58, 52:58] = 1_500
        locator = QR3DLocator(
            polygon_shrink_ratio=0.75,
            min_edge_pixels=20.0,
            ransac_iterations=80,
            distance_threshold=0.001,
            min_points=20,
            min_inlier_ratio=0.6,
            max_rms=0.0005,
            sample_stride=2,
            min_depth=0.25,
            max_depth=2.0,
            min_valid_depth_ratio=0.2,
            min_spatial_quadrants=3,
            min_points_per_quadrant=5,
            random_seed=4,
            deprojector=pinhole_deproject,
        )
        result = locator.locate(
            self.detection,
            depth,
            PinholeIntrinsics(),
            0.001,
            qr_id="QR_A",
        )
        self.assertIsNotNone(result.plane)
        self.assertFalse(result.valid)
        self.assertEqual(result.reject_reasons, ("poor_spatial_support",))
        self.assertEqual(result.spatial_quadrants, 1)

    def test_invalid_depth_scale_is_deprojection_failure(self) -> None:
        result = self.locator.locate(
            self.detection,
            self.depth,
            PinholeIntrinsics(),
            0.0,
            qr_id="QR_A",
        )
        self.assertEqual(result.reject_reasons, ("deprojection_failed",))

    def test_systematic_backend_failure_is_deprojection_failure(self) -> None:
        def failing_backend(
            _intrinsics: object, _pixel: list[float], _depth_m: float
        ) -> np.ndarray:
            raise RuntimeError("synthetic backend failure")

        locator = QR3DLocator(
            min_edge_pixels=20.0,
            min_points=20,
            sample_stride=2,
            ransac_iterations=30,
            random_seed=2,
            deprojector=failing_backend,
        )
        result = locator.locate(
            self.detection,
            self.depth,
            PinholeIntrinsics(),
            0.001,
            qr_id="QR_A",
        )
        self.assertEqual(result.reject_reasons, ("deprojection_failed",))
        self.assertEqual(result.valid_depth_points, 0)

    def test_direct_constructor_rejects_inoperative_rms_gate(self) -> None:
        for max_rms in (0.004, 0.006):
            with self.subTest(max_rms=max_rms):
                with self.assertRaisesRegex(
                    ValueError, "max_rms must be smaller than distance_threshold"
                ):
                    QR3DLocator(distance_threshold=0.004, max_rms=max_rms)

    def test_frame_source_style_two_argument_deprojector_is_supported(self) -> None:
        intrinsics = PinholeIntrinsics()

        def source_deproject(pixel: list[float], depth_m: float) -> np.ndarray:
            return pinhole_deproject(intrinsics, pixel, depth_m)

        locator = QR3DLocator(
            min_edge_pixels=20.0,
            min_points=20,
            sample_stride=2,
            ransac_iterations=30,
            random_seed=2,
            deprojector=source_deproject,
        )
        result = locator.locate(
            self.detection,
            self.depth,
            intrinsics,
            0.001,
        )
        self.assertTrue(result.valid)


if __name__ == "__main__":
    unittest.main()
