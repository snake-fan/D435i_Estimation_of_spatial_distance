"""Interactive OpenCV view for measurements and diagnostics."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .depth_view import colorize_depth


class ViewAction(str, Enum):
    NONE = "none"
    QUIT = "quit"
    TOGGLE_RECORDING = "toggle_recording"
    TOGGLE_DEPTH = "toggle_depth"


_GOOD = (80, 220, 80)
_WARNING = (0, 190, 255)
_INVALID = (80, 80, 255)
_TEXT = (225, 225, 225)


def _finite_point(value: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        return None
    return array


class DebugView:
    def __init__(
        self,
        *,
        expected_ids: Sequence[str],
        depth_scale: float,
        depth_min_m: float,
        depth_max_m: float,
        enabled: bool = True,
        detailed: bool = False,
        window_name: str = "D435i QR Distance Measurement",
    ):
        if len(expected_ids) != 2:
            raise ValueError("DebugView requires exactly two QR identifiers")
        self.expected_ids = tuple(expected_ids)
        self.depth_scale = float(depth_scale)
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.enabled = bool(enabled)
        self.detailed = bool(detailed)
        self.window_name = window_name
        self.show_depth = False

    @staticmethod
    def _roi_corners(result: Any) -> np.ndarray | None:
        explicit = _finite_point(getattr(result, "roi_corners", None), (4, 2))
        if explicit is not None:
            return explicit
        corners = _finite_point(getattr(result, "corners", None), (4, 2))
        center = _finite_point(getattr(result, "center_uv", None), (2,))
        shrink_ratio = getattr(result, "polygon_shrink_ratio", None)
        if corners is None or center is None or shrink_ratio is None:
            return None
        return center + float(shrink_ratio) * (corners - center)

    def _draw_geometry(self, image: np.ndarray, qr_results: Mapping[str, Any]) -> None:
        centers: list[np.ndarray] = []
        for qr_id in self.expected_ids:
            result = qr_results.get(qr_id)
            if result is None:
                continue
            valid = bool(getattr(result, "valid", False))
            raw_status = str(getattr(result, "status", "INVALID"))
            color = _WARNING if valid and raw_status == "WARNING" else _GOOD if valid else _INVALID
            corners = _finite_point(getattr(result, "corners", None), (4, 2))
            center = _finite_point(getattr(result, "center_uv", None), (2,))
            roi = self._roi_corners(result)
            if corners is not None:
                cv2.polylines(
                    image,
                    [np.rint(corners).astype(np.int32)],
                    True,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            if roi is not None:
                cv2.polylines(
                    image,
                    [np.rint(roi).astype(np.int32)],
                    True,
                    (255, 180, 40),
                    1,
                    cv2.LINE_AA,
                )
            if center is not None:
                center_i = tuple(np.rint(center).astype(int))
                cv2.circle(image, center_i, 4, color, -1, cv2.LINE_AA)
                centers.append(center)
                reason = getattr(result, "reject_reason", None)
                label = qr_id if valid else f"{qr_id}: {reason or 'INVALID'}"
                cv2.putText(
                    image,
                    label,
                    (center_i[0] + 8, center_i[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        if len(centers) == 2 and all(
            bool(getattr(qr_results.get(qr_id), "valid", False))
            for qr_id in self.expected_ids
        ):
            cv2.line(
                image,
                tuple(np.rint(centers[0]).astype(int)),
                tuple(np.rint(centers[1]).astype(int)),
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _put_lines(panel: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
        step = max(14, min(18, (panel.shape[0] - 20) // max(len(lines), 1)))
        font_scale = 0.40 if step < 17 else 0.42
        y = 18
        for text, color in lines:
            cv2.putText(
                panel,
                text,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
            y += step

    def _result_lines(self, qr_id: str, result: Any) -> list[tuple[str, tuple[int, int, int]]]:
        valid = result is not None and bool(getattr(result, "valid", False))
        raw_status = str(getattr(result, "status", "INVALID")) if result else "INVALID"
        header_color = (
            _WARNING if valid and raw_status == "WARNING" else _GOOD if valid else _INVALID
        )
        lines: list[tuple[str, tuple[int, int, int]]] = [(f"{qr_id}: {raw_status}", header_color)]
        if not valid:
            reason = getattr(result, "reject_reason", None) if result else None
            lines.append((f"  {reason or 'qr_not_found'}", _INVALID))
        if result is None:
            return lines

        point = _finite_point(getattr(result, "point_xyz", None), (3,))
        if point is not None:
            xyz_mm = point * 1000.0
            lines.append((f"  XYZ [{xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f}] mm", _TEXT))
        plane = getattr(result, "plane", None)
        if plane is not None:
            lines.extend(
                [
                    (f"  RMS {float(plane.rms) * 1000.0:.2f} mm", _TEXT),
                    (f"  Inliers {float(plane.inlier_ratio) * 100.0:.1f}%", _TEXT),
                ]
            )
        tilt = getattr(result, "tilt_deg", None)
        if tilt is not None:
            lines.append((f"  Tilt {float(tilt):.1f} deg", _TEXT))
        edge = getattr(result, "qr_min_edge_pixels", None)
        if edge is not None:
            lines.append((f"  Pixel size {float(edge):.1f}", _TEXT))
        if self.detailed:
            lines.append((f"  Depth points {int(getattr(result, 'valid_depth_points', 0))}", _TEXT))
            lines.append(
                (
                    f"  Depth valid {float(getattr(result, 'valid_depth_ratio', 0.0)) * 100.0:.1f}% "
                    f"Q={int(getattr(result, 'spatial_quadrants', 0))}/4",
                    _TEXT,
                )
            )
        return lines

    def render(
        self,
        *,
        frame: Any,
        qr_results: Mapping[str, Any],
        distance_result: Any,
        temporal: Any,
        status: str,
        reject_reason: str | None,
        fps: float | None = None,
        recording: bool = False,
    ) -> ViewAction:
        if not self.enabled:
            return ViewAction.NONE

        ir = np.asarray(frame.ir_image)
        if ir.ndim == 2:
            base = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        elif ir.ndim == 3 and ir.shape[2] == 3:
            base = ir.copy()
        else:
            raise ValueError("IR view must be grayscale or BGR")

        if self.show_depth:
            rois = [
                roi
                for result in qr_results.values()
                if (roi := self._roi_corners(result)) is not None
            ]
            base = colorize_depth(
                frame.depth_image,
                self.depth_scale,
                self.depth_min_m,
                self.depth_max_m,
                rois,
            )
        self._draw_geometry(base, qr_results)

        panel = np.full((base.shape[0], 430, 3), 25, dtype=np.uint8)
        status_color = _GOOD if status == "GOOD" else _WARNING if status == "WARNING" else _INVALID
        lines: list[tuple[str, tuple[int, int, int]]] = [
            ("D435i QR Distance Measurement", _TEXT),
            (f"Status: {status}", status_color),
        ]
        if reject_reason:
            lines.append((f"Reason: {reject_reason}", _INVALID))
        for qr_id in self.expected_ids:
            lines.append(("", _TEXT))
            lines.extend(self._result_lines(qr_id, qr_results.get(qr_id)))
        distance_m = getattr(distance_result, "distance_m", None)
        temporal_ready = bool(getattr(temporal, "ready", False))
        temporal_stale = bool(getattr(temporal, "stale", False))
        if temporal_stale:
            temporal_line = "Temporal: STALE"
        elif not temporal_ready:
            temporal_line = f"Temporal: warming ({int(getattr(temporal, 'count', 0))})"
        else:
            temporal_line = f"Median ({int(getattr(temporal, 'count', 0))}): {float(temporal.median_m) * 1000.0:.2f} mm"
        lines.extend(
            [
                ("", _TEXT),
                (
                    "Instant: --" if distance_m is None else f"Instant: {float(distance_m) * 1000.0:.2f} mm",
                    _TEXT,
                ),
                (
                    temporal_line,
                    _TEXT,
                ),
                (
                    "STD: --" if not temporal_ready or getattr(temporal, "std_m", None) is None else f"STD: {float(temporal.std_m) * 1000.0:.2f} mm",
                    _TEXT,
                ),
                (
                    "MAD: --" if not temporal_ready or getattr(temporal, "mad_m", None) is None else f"MAD: {float(temporal.mad_m) * 1000.0:.2f} mm",
                    _TEXT,
                ),
            ]
        )
        if fps is not None:
            lines.append((f"Processing FPS: {fps:.1f}", _TEXT))
        lines.extend(
            [
                (f"CSV recording: {'ON' if recording else 'OFF'}", _WARNING if recording else _TEXT),
                ("Q Quit | R Record | D Depth", _TEXT),
            ]
        )
        self._put_lines(panel, lines)
        canvas = np.concatenate([base, panel], axis=1)
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return ViewAction.QUIT
        if key == ord("r"):
            return ViewAction.TOGGLE_RECORDING
        if key == ord("d"):
            self.show_depth = not self.show_depth
            return ViewAction.TOGGLE_DEPTH
        return ViewAction.NONE

    def close(self) -> None:
        if self.enabled:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
