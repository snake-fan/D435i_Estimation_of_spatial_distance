from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

import main
from camera import CameraFrame, FrameAcquisitionError
from detection import QRDetection
from geometry.quadrilateral import quadrilateral_center
from utils.config import AppConfig, PlaneConfig, QRConfig, VisualizationConfig


class _Diagnostics:
    def format_text(self) -> str:
        return "synthetic depth-viewport source"


class _SyntheticSource:
    def __init__(self) -> None:
        self.stopped = False
        depth = np.full((80, 120), 1000, dtype=np.uint16)
        self.frame = CameraFrame(
            ir_image=np.zeros_like(depth, dtype=np.uint8),
            depth_image=depth,
            timestamp_ms=100.0,
            frame_number=1,
        )

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    def get_diagnostics(self) -> _Diagnostics:
        return _Diagnostics()

    def get_depth_intrinsics(self) -> object:
        return object()

    def get_depth_scale(self) -> float:
        return 0.001

    def get_frames(self) -> CameraFrame:
        return self.frame

    def deproject_pixel(self, pixel: list[float], depth_m: float) -> np.ndarray:
        u, v = pixel
        return np.array(
            [(float(u) - 60.0) * depth_m / 100.0, (float(v) - 40.0) * depth_m / 100.0, depth_m]
        )


class _FlakySyntheticSource(_SyntheticSource):
    def __init__(self) -> None:
        super().__init__()
        second = CameraFrame(
            ir_image=self.frame.ir_image.copy(),
            depth_image=self.frame.depth_image.copy(),
            timestamp_ms=3_100.0,
            frame_number=2,
        )
        self._events: list[CameraFrame | BaseException] = [
            self.frame,
            FrameAcquisitionError("synthetic timeout"),
            second,
        ]

    def get_frames(self) -> CameraFrame:
        event = self._events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


def _detection(payload: str, corners: list[list[float]]) -> QRDetection:
    array = np.asarray(corners, dtype=np.float64)
    center = quadrilateral_center(array)
    assert center is not None
    return QRDetection(payload, array, center, 40.0)


class _SyntheticDetector:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def detect(self, _image: np.ndarray) -> dict[str, QRDetection]:
        return {
            "QR_A": _detection("QR_A", [[10, 20], [50, 20], [50, 60], [10, 60]]),
            "QR_B": _detection("QR_B", [[70, 20], [110, 20], [110, 60], [70, 60]]),
        }


class MainIntegrationTests(unittest.TestCase):
    def test_one_synthetic_frame_reaches_json_and_comparison_csv(self) -> None:
        config = replace(
            AppConfig(),
            qr=replace(QRConfig(), min_edge_pixels=10.0, corner_refinement=False),
            plane=replace(
                PlaneConfig(),
                ransac_iterations=40,
                min_points=20,
                max_rms=0.003,
            ),
            visualization=VisualizationConfig(enabled=False),
        )
        source = _SyntheticSource()
        dummy_cv2 = types.ModuleType("cv2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "result.jsonl"
            csv_path = root / "result.csv"
            args = main.build_parser().parse_args(
                [
                    "--no-display",
                    "--max-frames",
                    "1",
                    "--json-output",
                    str(json_path),
                    "--record",
                    str(csv_path),
                ]
            )
            with (
                patch.object(main, "_create_source", return_value=source),
                patch.object(main, "QRDetector", _SyntheticDetector),
                patch.dict(sys.modules, {"cv2": dummy_cv2}),
            ):
                self.assertEqual(main.run(args, config), 0)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "GOOD")
            self.assertAlmostEqual(payload["distance_mm"], 600.0, places=5)
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("method_a_distance_mm", csv_text)
            self.assertIn("method_c_distance_mm", csv_text)
            self.assertTrue(source.stopped)

    def test_acquisition_failure_starts_a_fresh_temporal_epoch(self) -> None:
        config = replace(
            AppConfig(),
            qr=replace(QRConfig(), min_edge_pixels=10.0, corner_refinement=False),
            plane=replace(
                PlaneConfig(),
                ransac_iterations=40,
                min_points=20,
                max_rms=0.003,
            ),
            visualization=VisualizationConfig(enabled=False),
        )
        source = _FlakySyntheticSource()
        dummy_cv2 = types.ModuleType("cv2")
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "result.jsonl"
            args = main.build_parser().parse_args(
                [
                    "--no-display",
                    "--max-frames",
                    "2",
                    "--json-output",
                    str(json_path),
                ]
            )
            with (
                patch.object(main, "_create_source", return_value=source),
                patch.object(main, "QRDetector", _SyntheticDetector),
                patch.dict(sys.modules, {"cv2": dummy_cv2}),
            ):
                self.assertEqual(main.run(args, config), 0)

            payloads = [
                json.loads(line)
                for line in json_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(payloads), 2)
            self.assertEqual(payloads[0]["temporal"]["sample_count"], 1)
            self.assertEqual(payloads[1]["temporal"]["sample_count"], 1)
            self.assertTrue(source.stopped)

    def test_output_initialization_rolls_back_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "result.csv"
            json_path = root / "result.jsonl"
            args = main.build_parser().parse_args(
                [
                    "--record",
                    str(csv_path),
                    "--json-output",
                    str(json_path),
                ]
            )
            with patch.object(
                main.JsonLinesWriter,
                "open",
                side_effect=OSError("synthetic JSON open failure"),
            ):
                with self.assertRaises(OSError):
                    main._open_outputs(args, AppConfig())

            self.assertFalse(csv_path.exists())
            self.assertFalse(json_path.exists())


if __name__ == "__main__":
    unittest.main()
