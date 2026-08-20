"""Deterministic per-QR quality classification and reject reasons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

import numpy as np

from geometry.plane_svd import PlaneModel


class MeasurementStatus(str, Enum):
    """A QR or pair measurement's user-visible quality state."""

    GOOD = "GOOD"
    WARNING = "WARNING"
    INVALID = "INVALID"

    def __str__(self) -> str:
        return self.value


# This tuple is the single source of truth for reject-reason ordering.  Stage
# order is more useful to operators than incidental discovery order.
REJECT_REASON_ORDER: tuple[str, ...] = (
    "qr_not_found",
    "payload_mismatch",
    "invalid_qr_geometry",
    "qr_too_small",
    "not_enough_depth_points",
    "low_valid_depth_ratio",
    "poor_spatial_support",
    "deprojection_failed",
    "ransac_failed",
    "low_inlier_ratio",
    "plane_rms_too_large",
    "qr_tilt_too_large",
    "invalid_intersection",
    "out_of_depth_range",
)
_REJECT_REASON_RANK = {
    reason: index for index, reason in enumerate(REJECT_REASON_ORDER)
}


def order_reject_reasons(reasons: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate reasons and return them in stable pipeline-stage order."""

    unique: list[str] = []
    seen: set[str] = set()
    for raw_reason in reasons:
        reason = str(raw_reason).strip()
        if reason and reason not in seen:
            seen.add(reason)
            unique.append(reason)
    insertion_order = {reason: index for index, reason in enumerate(unique)}
    return tuple(
        sorted(
            unique,
            key=lambda reason: (
                _REJECT_REASON_RANK.get(reason, len(REJECT_REASON_ORDER)),
                insertion_order[reason],
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    status: MeasurementStatus
    reject_reasons: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status is not MeasurementStatus.INVALID

    @property
    def reject_reason(self) -> str | None:
        return ";".join(self.reject_reasons) if self.reject_reasons else None


class QualityGate:
    """Evaluate the quality metrics produced by the locator.

    Plane existence and inlier ratio are deliberately separate checks.  A
    fitted low-inlier plane therefore reports ``low_inlier_ratio`` rather than
    the less diagnostic ``ransac_failed``.
    """

    def __init__(
        self,
        *,
        min_edge_pixels: float = 35.0,
        min_points: int = 80,
        min_inlier_ratio: float = 0.65,
        max_rms: float = 0.003,
        min_depth: float = 0.25,
        max_depth: float = 2.0,
        min_valid_depth_ratio: float = 0.35,
        min_spatial_quadrants: int = 3,
        warning_tilt_deg: float = 45.0,
        max_tilt_deg: float = 60.0,
    ) -> None:
        numeric = (
            min_edge_pixels,
            min_inlier_ratio,
            max_rms,
            min_depth,
            max_depth,
            min_valid_depth_ratio,
            warning_tilt_deg,
            max_tilt_deg,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("quality thresholds must be finite")
        if min_edge_pixels <= 0.0:
            raise ValueError("min_edge_pixels must be positive")
        if isinstance(min_points, bool) or int(min_points) != min_points or min_points < 3:
            raise ValueError("min_points must be an integer >= 3")
        if not 0.0 <= min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be in [0, 1]")
        if max_rms <= 0.0:
            raise ValueError("max_rms must be positive")
        if not 0.0 < min_depth < max_depth:
            raise ValueError("depth range must satisfy 0 < min_depth < max_depth")
        if not 0.0 < min_valid_depth_ratio <= 1.0:
            raise ValueError("min_valid_depth_ratio must be in (0, 1]")
        if (
            isinstance(min_spatial_quadrants, bool)
            or int(min_spatial_quadrants) != min_spatial_quadrants
            or not 1 <= int(min_spatial_quadrants) <= 4
        ):
            raise ValueError("min_spatial_quadrants must be an integer in [1, 4]")
        if not 0.0 <= warning_tilt_deg <= max_tilt_deg < 90.0:
            raise ValueError("tilt limits must satisfy 0 <= warning <= max < 90")

        self.min_edge_pixels = float(min_edge_pixels)
        self.min_points = int(min_points)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_rms = float(max_rms)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.min_valid_depth_ratio = float(min_valid_depth_ratio)
        self.min_spatial_quadrants = int(min_spatial_quadrants)
        self.warning_tilt_deg = float(warning_tilt_deg)
        self.max_tilt_deg = float(max_tilt_deg)

    def from_reasons(
        self,
        reasons: Iterable[str],
        *,
        tilt_deg: float | None = None,
    ) -> QualityAssessment:
        """Build a status from already discovered stage failures."""

        ordered = order_reject_reasons(reasons)
        if ordered:
            return QualityAssessment(MeasurementStatus.INVALID, ordered)
        if (
            tilt_deg is not None
            and math.isfinite(float(tilt_deg))
            and float(tilt_deg) >= self.warning_tilt_deg
        ):
            return QualityAssessment(MeasurementStatus.WARNING)
        return QualityAssessment(MeasurementStatus.GOOD)

    def assess(
        self,
        *,
        qr_found: bool = True,
        payload_matches: bool = True,
        geometry_valid: bool = True,
        qr_min_edge_pixels: float | None = None,
        depth_evaluated: bool = True,
        valid_depth_points: int = 0,
        valid_depth_ratio: float | None = None,
        spatial_quadrants: int | None = None,
        plane_fit_attempted: bool = True,
        plane: PlaneModel | None = None,
        tilt_deg: float | None = None,
        intersection_attempted: bool = True,
        point_xyz: np.ndarray | None = None,
    ) -> QualityAssessment:
        """Assess all reached stages and preserve canonical reason ordering."""

        if not qr_found:
            return self.from_reasons(("qr_not_found",))

        reasons: list[str] = []
        if not payload_matches:
            reasons.append("payload_mismatch")
        if not geometry_valid:
            reasons.append("invalid_qr_geometry")

        if qr_min_edge_pixels is not None:
            edge = float(qr_min_edge_pixels)
            if not math.isfinite(edge) or edge < self.min_edge_pixels:
                reasons.append("qr_too_small")

        if depth_evaluated and valid_depth_points < self.min_points:
            reasons.append("not_enough_depth_points")
        enough_depth_points = valid_depth_points >= self.min_points
        if depth_evaluated and enough_depth_points and valid_depth_ratio is not None:
            ratio = float(valid_depth_ratio)
            if not math.isfinite(ratio) or ratio < self.min_valid_depth_ratio:
                reasons.append("low_valid_depth_ratio")
        if depth_evaluated and enough_depth_points and spatial_quadrants is not None:
            if spatial_quadrants < self.min_spatial_quadrants:
                reasons.append("poor_spatial_support")

        if plane_fit_attempted:
            if plane is None:
                reasons.append("ransac_failed")
            else:
                if (
                    not math.isfinite(float(plane.inlier_ratio))
                    or float(plane.inlier_ratio) < self.min_inlier_ratio
                ):
                    reasons.append("low_inlier_ratio")
                if (
                    not math.isfinite(float(plane.rms))
                    or float(plane.rms) > self.max_rms
                ):
                    reasons.append("plane_rms_too_large")

        if tilt_deg is not None:
            tilt = float(tilt_deg)
            if not math.isfinite(tilt) or tilt > self.max_tilt_deg:
                reasons.append("qr_tilt_too_large")

        if intersection_attempted:
            point = None if point_xyz is None else np.asarray(point_xyz, dtype=np.float64)
            if point is None or point.shape != (3,) or not np.all(np.isfinite(point)):
                reasons.append("invalid_intersection")
            elif not self.min_depth <= float(point[2]) <= self.max_depth:
                reasons.append("out_of_depth_range")

        return self.from_reasons(reasons, tilt_deg=tilt_deg)

    # A common verb used by callers and tests.
    evaluate = assess


__all__ = [
    "MeasurementStatus",
    "QualityAssessment",
    "QualityGate",
    "REJECT_REASON_ORDER",
    "order_reject_reasons",
]
