"""Locate a QR code's physical center by intersecting its ray with a plane."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

import numpy as np

from detection.qr_detector import QRDetection
from geometry.deprojection import deproject_pixel_to_ray, deproject_roi
from geometry.plane_ransac import fit_plane_ransac
from geometry.plane_svd import PlaneModel
from geometry.quadrilateral import (
    edge_lengths,
    is_valid_convex_quadrilateral,
    polygon_mask,
    quadrilateral_center,
    shrink_quadrilateral,
)
from geometry.ray_plane import intersect_ray_plane

from .quality_gate import (
    MeasurementStatus,
    QualityAssessment,
    QualityGate,
    order_reject_reasons,
)


Deprojector = Callable[..., Any]


def _empty_points() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


@dataclass(slots=True)
class QR3DResult:
    """Full per-QR result, including a first-class missing/invalid state."""

    qr_id: str
    point_xyz: np.ndarray | None = None
    plane: PlaneModel | None = None
    center_uv: np.ndarray | None = None
    corners: np.ndarray | None = None
    roi_corners: np.ndarray | None = None
    roi_points: np.ndarray = field(default_factory=_empty_points)
    valid_depth_points: int = 0
    roi_sample_points: int = 0
    valid_depth_ratio: float = 0.0
    spatial_quadrants: int = 0
    qr_min_edge_pixels: float | None = None
    tilt_deg: float | None = None
    status: MeasurementStatus = MeasurementStatus.INVALID
    reject_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.qr_id = str(self.qr_id)
        self.reject_reasons = order_reject_reasons(self.reject_reasons)
        if not isinstance(self.status, MeasurementStatus):
            self.status = MeasurementStatus(str(self.status))
        points = np.asarray(self.roi_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("roi_points must have shape (N, 3)")
        self.roi_points = points
        self.valid_depth_points = int(self.valid_depth_points)
        self.roi_sample_points = int(self.roi_sample_points)

    @property
    def valid(self) -> bool:
        if self.status is MeasurementStatus.INVALID or self.point_xyz is None:
            return False
        point = np.asarray(self.point_xyz, dtype=np.float64)
        return point.shape == (3,) and bool(np.all(np.isfinite(point)))

    @property
    def reject_reason(self) -> str | None:
        return ";".join(self.reject_reasons) if self.reject_reasons else None

    @property
    def quality_status(self) -> str:
        return self.status.value

    @classmethod
    def missing(cls, qr_id: str) -> "QR3DResult":
        return cls(
            qr_id=qr_id,
            status=MeasurementStatus.INVALID,
            reject_reasons=("qr_not_found",),
        )


class QR3DLocator:
    """QR polygon -> depth cloud -> RANSAC plane -> center intersection."""

    def __init__(
        self,
        *,
        polygon_shrink_ratio: float = 0.75,
        min_edge_pixels: float = 35.0,
        ransac_iterations: int = 200,
        distance_threshold: float = 0.004,
        min_points: int = 80,
        min_inlier_ratio: float = 0.65,
        max_rms: float = 0.003,
        sample_stride: int = 2,
        min_depth: float = 0.25,
        max_depth: float = 2.0,
        min_valid_depth_ratio: float = 0.35,
        min_spatial_quadrants: int = 3,
        min_points_per_quadrant: int = 5,
        warning_tilt_deg: float = 45.0,
        max_tilt_deg: float = 60.0,
        ray_parallel_epsilon: float = 1.0e-10,
        random_seed: int | None = None,
        degenerate_epsilon: float = 1.0e-12,
        deprojector: Deprojector | None = None,
    ) -> None:
        if not math.isfinite(polygon_shrink_ratio) or not 0.0 < polygon_shrink_ratio <= 1.0:
            raise ValueError("polygon_shrink_ratio must be in (0, 1]")
        if isinstance(ransac_iterations, bool) or int(ransac_iterations) != ransac_iterations:
            raise ValueError("ransac_iterations must be a positive integer")
        if int(ransac_iterations) <= 0:
            raise ValueError("ransac_iterations must be a positive integer")
        if not math.isfinite(distance_threshold) or distance_threshold <= 0.0:
            raise ValueError("distance_threshold must be positive and finite")
        if not math.isfinite(max_rms) or max_rms <= 0.0:
            raise ValueError("max_rms must be positive and finite")
        if max_rms >= distance_threshold:
            raise ValueError("max_rms must be smaller than distance_threshold")
        if isinstance(sample_stride, bool) or int(sample_stride) != sample_stride:
            raise ValueError("sample_stride must be a positive integer")
        if int(sample_stride) <= 0:
            raise ValueError("sample_stride must be a positive integer")
        if not math.isfinite(ray_parallel_epsilon) or ray_parallel_epsilon <= 0.0:
            raise ValueError("ray_parallel_epsilon must be positive and finite")
        if not math.isfinite(degenerate_epsilon) or degenerate_epsilon <= 0.0:
            raise ValueError("degenerate_epsilon must be positive and finite")
        if (
            isinstance(min_points_per_quadrant, bool)
            or int(min_points_per_quadrant) != min_points_per_quadrant
            or min_points_per_quadrant <= 0
        ):
            raise ValueError("min_points_per_quadrant must be a positive integer")

        self.polygon_shrink_ratio = float(polygon_shrink_ratio)
        self.ransac_iterations = int(ransac_iterations)
        self.distance_threshold = float(distance_threshold)
        self.sample_stride = int(sample_stride)
        self.ray_parallel_epsilon = float(ray_parallel_epsilon)
        self.random_seed = random_seed
        self.degenerate_epsilon = float(degenerate_epsilon)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.min_points_per_quadrant = int(min_points_per_quadrant)
        self._deprojector = self._adapt_deprojector(deprojector)
        self.quality_gate = QualityGate(
            min_edge_pixels=min_edge_pixels,
            min_points=min_points,
            min_inlier_ratio=min_inlier_ratio,
            max_rms=max_rms,
            min_depth=min_depth,
            max_depth=max_depth,
            min_valid_depth_ratio=min_valid_depth_ratio,
            min_spatial_quadrants=min_spatial_quadrants,
            warning_tilt_deg=warning_tilt_deg,
            max_tilt_deg=max_tilt_deg,
        )

    @staticmethod
    def _adapt_deprojector(deprojector: Deprojector | None) -> Deprojector | None:
        """Accept both SDK-style 3-arg and FrameSource-style 2-arg callbacks."""

        if deprojector is None:
            return None

        def adapted(intrinsics: Any, pixel: list[float], depth_m: float) -> Any:
            try:
                return deprojector(intrinsics, pixel, depth_m)
            except TypeError as three_argument_error:
                try:
                    return deprojector(pixel, depth_m)
                except TypeError:
                    raise three_argument_error

        return adapted

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        deprojector: Deprojector | None = None,
    ) -> "QR3DLocator":
        """Construct from ``AppConfig`` without coupling this module to YAML."""

        qr = config.qr
        plane = config.plane
        measurement = config.measurement
        return cls(
            polygon_shrink_ratio=qr.polygon_shrink_ratio,
            min_edge_pixels=qr.min_edge_pixels,
            ransac_iterations=plane.ransac_iterations,
            distance_threshold=plane.distance_threshold,
            min_points=plane.min_points,
            min_inlier_ratio=plane.min_inlier_ratio,
            max_rms=plane.max_rms,
            sample_stride=plane.sample_stride,
            min_depth=measurement.min_depth,
            max_depth=measurement.max_depth,
            min_valid_depth_ratio=measurement.min_valid_depth_ratio,
            min_spatial_quadrants=measurement.min_spatial_quadrants,
            min_points_per_quadrant=measurement.min_points_per_quadrant,
            warning_tilt_deg=measurement.warning_tilt_deg,
            max_tilt_deg=measurement.max_tilt_deg,
            ray_parallel_epsilon=measurement.ray_parallel_epsilon,
            random_seed=plane.random_seed,
            degenerate_epsilon=plane.degenerate_epsilon,
            deprojector=deprojector,
        )

    @staticmethod
    def _tilt_deg(plane: PlaneModel) -> float | None:
        normal = np.asarray(plane.normal, dtype=np.float64)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            return None
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            return None
        cosine = float(np.clip(abs(normal[2]) / norm, 0.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _assessment_result(
        *,
        qr_id: str,
        assessment: QualityAssessment,
        point_xyz: np.ndarray | None = None,
        plane: PlaneModel | None = None,
        center_uv: np.ndarray | None = None,
        corners: np.ndarray | None = None,
        roi_corners: np.ndarray | None = None,
        roi_points: np.ndarray | None = None,
        roi_sample_points: int = 0,
        spatial_quadrants: int = 0,
        qr_min_edge_pixels: float | None = None,
        tilt_deg: float | None = None,
    ) -> QR3DResult:
        points = _empty_points() if roi_points is None else roi_points
        valid_count = int(len(points))
        valid_ratio = (
            float(valid_count / roi_sample_points) if roi_sample_points > 0 else 0.0
        )
        return QR3DResult(
            qr_id=qr_id,
            point_xyz=point_xyz,
            plane=plane,
            center_uv=center_uv,
            corners=corners,
            roi_corners=roi_corners,
            roi_points=points,
            valid_depth_points=valid_count,
            roi_sample_points=roi_sample_points,
            valid_depth_ratio=valid_ratio,
            spatial_quadrants=spatial_quadrants,
            qr_min_edge_pixels=qr_min_edge_pixels,
            tilt_deg=tilt_deg,
            status=assessment.status,
            reject_reasons=assessment.reject_reasons,
        )

    def _count_spatial_quadrants(
        self,
        pixel_uv: np.ndarray,
        center_uv: np.ndarray,
    ) -> int:
        """Count center quadrants with enough supplied sampled pixels."""

        pixels = np.asarray(pixel_uv, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1:] != (2,) or len(pixels) == 0:
            return 0
        pixels = pixels[np.all(np.isfinite(pixels), axis=1)]
        if len(pixels) == 0:
            return 0
        columns = pixels[:, 0]
        rows = pixels[:, 1]
        left = columns < float(center_uv[0])
        top = rows < float(center_uv[1])
        quadrant_counts = (
            np.count_nonzero(left & top),
            np.count_nonzero(~left & top),
            np.count_nonzero(left & ~top),
            np.count_nonzero(~left & ~top),
        )
        return sum(
            count >= self.min_points_per_quadrant for count in quadrant_counts
        )

    def locate(
        self,
        detection: QRDetection | None,
        depth_image: np.ndarray,
        intrinsics: Any,
        depth_scale: float,
        *,
        qr_id: str | None = None,
    ) -> QR3DResult:
        """Locate one expected QR, returning INVALID instead of raising per-frame.

        ``qr_id`` is required only when ``detection`` is ``None`` and lets the
        caller produce an explicit missing result for every configured marker.
        """

        resolved_id = str(qr_id or (detection.payload if detection is not None else "UNKNOWN"))
        if detection is None:
            return QR3DResult.missing(resolved_id)

        payload_matches = qr_id is None or detection.payload == qr_id
        corners = np.asarray(detection.corners, dtype=np.float64)
        center = quadrilateral_center(corners)
        lengths = edge_lengths(corners)
        geometry_valid = (
            is_valid_convex_quadrilateral(corners)
            and center is not None
            and lengths is not None
        )
        min_edge = float(np.min(lengths)) if lengths is not None else None
        if not payload_matches or not geometry_valid:
            assessment = self.quality_gate.assess(
                payload_matches=payload_matches,
                geometry_valid=geometry_valid,
                qr_min_edge_pixels=min_edge,
                depth_evaluated=False,
                plane_fit_attempted=False,
                intersection_attempted=False,
            )
            return self._assessment_result(
                qr_id=resolved_id,
                assessment=assessment,
                center_uv=center,
                corners=corners,
                qr_min_edge_pixels=min_edge,
            )
        assert center is not None and min_edge is not None

        if min_edge < self.quality_gate.min_edge_pixels:
            assessment = self.quality_gate.assess(
                qr_min_edge_pixels=min_edge,
                depth_evaluated=False,
                plane_fit_attempted=False,
                intersection_attempted=False,
            )
            return self._assessment_result(
                qr_id=resolved_id,
                assessment=assessment,
                center_uv=center,
                corners=corners,
                qr_min_edge_pixels=min_edge,
            )

        if not math.isfinite(depth_scale) or depth_scale <= 0.0:
            assessment = self.quality_gate.from_reasons(("deprojection_failed",))
            return self._assessment_result(
                qr_id=resolved_id,
                assessment=assessment,
                center_uv=center,
                corners=corners,
                qr_min_edge_pixels=min_edge,
            )

        try:
            roi_corners = shrink_quadrilateral(
                corners,
                self.polygon_shrink_ratio,
                center=center,
            )
            mask = polygon_mask(np.asarray(depth_image).shape, roi_corners)
            roi_sample_points = int(
                np.count_nonzero(mask[:: self.sample_stride, :: self.sample_stride])
            )
            points, point_pixels = deproject_roi(
                depth_image,
                mask,
                intrinsics,
                depth_scale,
                self.sample_stride,
                self.min_depth,
                self.max_depth,
                deprojector=self._deprojector,
                return_pixels=True,
            )
            spatial_quadrants = self._count_spatial_quadrants(
                point_pixels,
                center,
            )
        except (RuntimeError, TypeError, ValueError):
            assessment = self.quality_gate.from_reasons(("deprojection_failed",))
            return self._assessment_result(
                qr_id=resolved_id,
                assessment=assessment,
                center_uv=center,
                corners=corners,
                qr_min_edge_pixels=min_edge,
            )

        valid_depth_ratio = (
            float(len(points) / roi_sample_points) if roi_sample_points else 0.0
        )
        support_assessment = self.quality_gate.assess(
            qr_min_edge_pixels=min_edge,
            valid_depth_points=len(points),
            valid_depth_ratio=valid_depth_ratio,
            spatial_quadrants=spatial_quadrants,
            plane_fit_attempted=False,
            intersection_attempted=False,
        )
        if not support_assessment.valid:
            return self._assessment_result(
                qr_id=resolved_id,
                assessment=support_assessment,
                center_uv=center,
                corners=corners,
                roi_corners=roi_corners,
                roi_points=points,
                roi_sample_points=roi_sample_points,
                spatial_quadrants=spatial_quadrants,
                qr_min_edge_pixels=min_edge,
            )

        plane = fit_plane_ransac(
            points,
            self.ransac_iterations,
            self.distance_threshold,
            0.0,
            random_seed=self.random_seed,
            degenerate_epsilon=self.degenerate_epsilon,
        )
        if plane is None:
            assessment = self.quality_gate.assess(
                qr_min_edge_pixels=min_edge,
                valid_depth_points=len(points),
                valid_depth_ratio=valid_depth_ratio,
                spatial_quadrants=spatial_quadrants,
                plane_fit_attempted=True,
                plane=None,
                intersection_attempted=False,
            )
            return self._assessment_result(
                qr_id=resolved_id,
                assessment=assessment,
                center_uv=center,
                corners=corners,
                roi_corners=roi_corners,
                roi_points=points,
                roi_sample_points=roi_sample_points,
                spatial_quadrants=spatial_quadrants,
                qr_min_edge_pixels=min_edge,
            )

        inlier_pixels = point_pixels[np.asarray(plane.inlier_mask, dtype=bool)]
        inlier_spatial_quadrants = self._count_spatial_quadrants(
            inlier_pixels,
            center,
        )

        tilt = self._tilt_deg(plane)
        ray = deproject_pixel_to_ray(
            intrinsics,
            center,
            deprojector=self._deprojector,
        )
        point = (
            intersect_ray_plane(ray, plane, epsilon=self.ray_parallel_epsilon)
            if ray is not None
            else None
        )
        assessment = self.quality_gate.assess(
            qr_min_edge_pixels=min_edge,
            valid_depth_points=len(points),
            valid_depth_ratio=valid_depth_ratio,
            spatial_quadrants=inlier_spatial_quadrants,
            plane_fit_attempted=True,
            plane=plane,
            tilt_deg=tilt,
            intersection_attempted=True,
            point_xyz=point,
        )
        return self._assessment_result(
            qr_id=resolved_id,
            assessment=assessment,
            point_xyz=point,
            plane=plane,
            center_uv=center,
            corners=corners,
            roi_corners=roi_corners,
            roi_points=points,
            roi_sample_points=roi_sample_points,
            spatial_quadrants=inlier_spatial_quadrants,
            qr_min_edge_pixels=min_edge,
            tilt_deg=tilt,
        )


__all__ = ["QR3DLocator", "QR3DResult"]
