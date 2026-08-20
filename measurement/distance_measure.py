"""Three-dimensional Euclidean distance and pair-level status handling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .quality_gate import MeasurementStatus


def _finite_point(value: Any, *, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite three-vector")
    return point


def euclidean_distance(point_a: Any, point_b: Any) -> float:
    """Return ``||point_b - point_a||`` in the points' existing unit."""

    first = _finite_point(point_a, name="point_a")
    second = _finite_point(point_b, name="point_b")
    distance = float(np.linalg.norm(second - first))
    if not math.isfinite(distance):  # defensive; finite vectors should suffice
        raise ValueError("computed distance is not finite")
    return distance


@dataclass(frozen=True, slots=True)
class DistanceResult:
    """Pair result; invalid QR inputs are represented without a fake zero."""

    distance_m: float | None
    status: MeasurementStatus
    qr_a_id: str
    qr_b_id: str
    reject_reasons: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status is not MeasurementStatus.INVALID and self.distance_m is not None

    @property
    def reject_reason(self) -> str | None:
        return ";".join(self.reject_reasons) if self.reject_reasons else None


def _result_id(result: Any, fallback: str) -> str:
    value = getattr(result, "qr_id", None) if result is not None else None
    return str(value) if value else fallback


def _is_valid(result: Any) -> bool:
    return result is not None and bool(getattr(result, "valid", False))


def _status(result: Any) -> MeasurementStatus:
    raw = getattr(result, "status", None)
    if isinstance(raw, MeasurementStatus):
        return raw
    try:
        return MeasurementStatus(str(raw))
    except ValueError:
        return MeasurementStatus.GOOD if _is_valid(result) else MeasurementStatus.INVALID


def _prefixed_reasons(result: Any, label: str) -> list[str]:
    if result is None:
        return [f"{label}:qr_not_found"]
    raw_reasons = getattr(result, "reject_reasons", ()) or ()
    if isinstance(raw_reasons, str):
        raw_reasons = (raw_reasons,)
    reasons = [str(reason) for reason in raw_reasons if str(reason)]
    if not reasons:
        singular = getattr(result, "reject_reason", None)
        if singular:
            reasons = [part for part in str(singular).split(";") if part]
    if not reasons and not _is_valid(result):
        reasons = ["invalid_qr_result"]
    return [f"{label}:{reason}" for reason in reasons]


class DistanceMeasure:
    """Compute a distance only when both QR results are valid."""

    def compute(self, qr_a: Any, qr_b: Any) -> DistanceResult:
        qr_a_id = _result_id(qr_a, "QR_A")
        qr_b_id = _result_id(qr_b, "QR_B")
        reasons: list[str] = []
        if not _is_valid(qr_a):
            reasons.extend(_prefixed_reasons(qr_a, qr_a_id))
        if not _is_valid(qr_b):
            reasons.extend(_prefixed_reasons(qr_b, qr_b_id))
        if reasons:
            return DistanceResult(
                distance_m=None,
                status=MeasurementStatus.INVALID,
                qr_a_id=qr_a_id,
                qr_b_id=qr_b_id,
                reject_reasons=tuple(reasons),
            )

        try:
            distance = euclidean_distance(qr_a.point_xyz, qr_b.point_xyz)
        except (AttributeError, TypeError, ValueError):
            return DistanceResult(
                distance_m=None,
                status=MeasurementStatus.INVALID,
                qr_a_id=qr_a_id,
                qr_b_id=qr_b_id,
                reject_reasons=("invalid_point_xyz",),
            )

        status = (
            MeasurementStatus.WARNING
            if MeasurementStatus.WARNING in (_status(qr_a), _status(qr_b))
            else MeasurementStatus.GOOD
        )
        return DistanceResult(
            distance_m=distance,
            status=status,
            qr_a_id=qr_a_id,
            qr_b_id=qr_b_id,
        )


def compute_distance(qr_a: Any, qr_b: Any) -> DistanceResult:
    """Functional wrapper around :class:`DistanceMeasure`."""

    return DistanceMeasure().compute(qr_a, qr_b)


__all__ = [
    "DistanceMeasure",
    "DistanceResult",
    "compute_distance",
    "euclidean_distance",
]
