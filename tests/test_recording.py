from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from recording.csv_recorder import CSVRecorder, build_csv_row
from recording.json_output import JsonLinesWriter, build_json_result
from recording.point_cloud_dump import PointCloudDumper


def _qr(qr_id: str, point: list[float]) -> SimpleNamespace:
    plane = SimpleNamespace(
        rms=0.0015,
        inlier_ratio=0.9,
        inlier_mask=np.array([True, False]),
    )
    return SimpleNamespace(
        qr_id=qr_id,
        valid=True,
        status="GOOD",
        point_xyz=np.asarray(point),
        plane=plane,
        tilt_deg=12.0,
        qr_min_edge_pixels=55.0,
        valid_depth_points=100,
        valid_depth_ratio=0.9,
        spatial_quadrants=4,
        reject_reason=None,
        roi_points=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
    )


class RecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = _qr("QR_A", [0.0, 0.0, 1.0])
        self.b = _qr("QR_B", [0.3, 0.4, 1.0])
        self.results = {"QR_A": self.a, "QR_B": self.b}
        self.distance = SimpleNamespace(distance_m=0.5)
        self.temporal = SimpleNamespace(
            count=10,
            ready=True,
            median_m=0.5,
            mean_m=0.5,
            std_m=0.001,
            mad_m=0.0007,
        )

    def test_json_contains_documented_units_and_no_nan(self) -> None:
        payload = build_json_result(
            source_timestamp_ms=123.0,
            frame_number=9,
            qr_results=self.results,
            expected_ids=["QR_A", "QR_B"],
            distance_result=self.distance,
            temporal=self.temporal,
            status="GOOD",
        )
        self.assertEqual(payload["distance_mm"], 500.0)
        self.assertEqual(payload["qr_a"]["point_m"], [0.0, 0.0, 1.0])
        self.assertEqual(payload["qr_a"]["status"], "GOOD")
        json.dumps(payload, allow_nan=False)

    def test_csv_and_jsonl_writers_create_parseable_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = build_csv_row(
                source_timestamp_ms=123.0,
                frame_number=9,
                qr_results=self.results,
                expected_ids=["QR_A", "QR_B"],
                distance_result=self.distance,
                temporal=self.temporal,
                status="GOOD",
            )
            csv_path = root / "measurements.csv"
            with CSVRecorder(csv_path) as recorder:
                recorder.write(row)
            self.assertIn("distance_mm", csv_path.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                CSVRecorder(csv_path).open()

            json_path = root / "measurements.jsonl"
            payload = build_json_result(
                source_timestamp_ms=123.0,
                frame_number=9,
                qr_results=self.results,
                expected_ids=["QR_A", "QR_B"],
                distance_result=self.distance,
                temporal=self.temporal,
                status="GOOD",
            )
            with JsonLinesWriter(json_path) as writer:
                writer.write(payload)
            self.assertEqual(json.loads(json_path.read_text())["status"], "GOOD")
            with self.assertRaises(FileExistsError):
                JsonLinesWriter(json_path).open()

    def test_point_cloud_dump_has_inlier_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = PointCloudDumper(directory).dump("QR_A", self.a)
            self.assertIsNotNone(path)
            assert path is not None
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "x,y,z,inlier")
            self.assertTrue(lines[1].endswith(",1"))


if __name__ == "__main__":
    unittest.main()
