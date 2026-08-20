"""Multi-QR detection using OpenCV's :class:`QRCodeDetector`."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable

import numpy as np

from geometry.quadrilateral import (
    edge_lengths,
    is_valid_convex_quadrilateral,
    polygon_area,
    quadrilateral_center,
)

from .corner_refinement import refine_qr_corners


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QRDetection:
    """A decoded QR payload and its image-space geometry."""

    payload: str
    corners: np.ndarray
    center: np.ndarray
    min_edge_length: float


class QRDetector:
    """Detect expected QR payloads in a grayscale image.

    Unexpected payloads and undecoded candidates are ignored.  If OpenCV
    reports the same payload more than once, the largest valid convex
    quadrilateral is retained.
    """

    def __init__(
        self,
        expected_ids: Iterable[str] | None = None,
        *,
        refine_corners: bool | None = None,
        corner_refinement: bool | None = None,
        detector: Any | None = None,
    ) -> None:
        if expected_ids is None:
            self._expected_ids: frozenset[str] | None = None
        else:
            normalized = [str(value) for value in expected_ids]
            if any(not value for value in normalized):
                raise ValueError("expected QR payloads may not be empty")
            self._expected_ids = frozenset(normalized)

        if refine_corners is not None and corner_refinement is not None:
            raise ValueError(
                "provide either refine_corners or corner_refinement, not both"
            )
        refinement_option = (
            corner_refinement
            if corner_refinement is not None
            else refine_corners
        )
        self._refine_corners = (
            True if refinement_option is None else bool(refinement_option)
        )
        self._detector = detector
        self._consecutive_backend_errors = 0

    def _get_detector(self) -> Any:
        if self._detector is None:
            try:
                import cv2
            except Exception as exc:  # pragma: no cover - environment specific
                raise RuntimeError(
                    "OpenCV is required for QR detection; install opencv-python"
                ) from exc
            self._detector = cv2.QRCodeDetector()
        return self._detector

    @staticmethod
    def _parse_multi_result(result: Any) -> tuple[bool, list[str], Any]:
        """Normalize OpenCV version differences in detectAndDecodeMulti."""

        if not isinstance(result, tuple):
            return False, [], None

        if len(result) == 4:
            ok, decoded_info, points, _ = result
        elif len(result) == 3 and isinstance(result[0], (bool, np.bool_)):
            ok, decoded_info, points = result
        elif len(result) == 3:
            decoded_info, points, _ = result
            ok = points is not None
        elif len(result) == 2:
            decoded_info, points = result
            ok = points is not None
        else:
            return False, [], None

        if decoded_info is None:
            return bool(ok), [], points
        if isinstance(decoded_info, str):
            payloads = [decoded_info]
        else:
            payloads = [str(value) for value in decoded_info]
        return bool(ok), payloads, points

    def detect(self, gray_image: np.ndarray) -> dict[str, QRDetection]:
        """Detect and decode multiple QR codes from a 2-D grayscale image."""

        image = np.asarray(gray_image)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("gray_image must be a non-empty 2-D array")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)

        # Dependency/setup failures are process-level diagnostics and should be
        # surfaced.  Only an actual per-frame OpenCV detection failure is
        # converted to an empty detection set.
        detector = self._get_detector()
        try:
            raw_result = detector.detectAndDecodeMulti(image)
        except Exception as exc:
            # Detection failures are frame-local in the real-time pipeline.
            self._consecutive_backend_errors += 1
            if (
                self._consecutive_backend_errors == 1
                or self._consecutive_backend_errors % 100 == 0
            ):
                LOGGER.warning(
                    "OpenCV QR backend failed for %d consecutive frame(s): %s",
                    self._consecutive_backend_errors,
                    exc,
                )
            return {}
        self._consecutive_backend_errors = 0

        ok, payloads, raw_points = self._parse_multi_result(raw_result)
        if not ok or raw_points is None or not payloads:
            return {}

        points = np.asarray(raw_points, dtype=np.float64)
        if points.shape == (4, 2):
            points = points[np.newaxis, ...]
        if points.ndim != 3 or points.shape[1:] != (4, 2):
            return {}

        detections: dict[str, QRDetection] = {}
        areas: dict[str, float] = {}

        for payload, original_corners in zip(payloads, points, strict=False):
            if not payload:
                continue
            if self._expected_ids is not None and payload not in self._expected_ids:
                continue
            if not is_valid_convex_quadrilateral(original_corners):
                continue

            corners = np.asarray(original_corners, dtype=np.float64)
            if self._refine_corners:
                refined = refine_qr_corners(image, corners)
                if is_valid_convex_quadrilateral(refined):
                    corners = refined

            center = quadrilateral_center(corners)
            if center is None:
                continue
            lengths = edge_lengths(corners)
            area = polygon_area(corners)
            if lengths is None or not np.isfinite(area) or area <= 0.0:
                continue

            if payload in detections and area <= areas[payload]:
                continue

            detections[payload] = QRDetection(
                payload=payload,
                corners=corners.copy(),
                center=center,
                min_edge_length=float(np.min(lengths)),
            )
            areas[payload] = float(area)

        return detections


__all__ = ["QRDetection", "QRDetector"]
