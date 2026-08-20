from __future__ import annotations

import unittest
import csv
import tempfile
from pathlib import Path

from evaluation.compare_methods import compare_csv
from evaluation.evaluate_ground_truth import accuracy_metrics


class EvaluationTests(unittest.TestCase):
    def test_accuracy_metrics(self) -> None:
        metrics = accuracy_metrics([99.0, 100.0, 101.0], 100.0)
        self.assertEqual(metrics["sample_count"], 3)
        self.assertAlmostEqual(float(metrics["bias_mm"]), 0.0)
        self.assertAlmostEqual(float(metrics["mae_mm"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(metrics["rmse_mm"]), (2.0 / 3.0) ** 0.5)

    def test_paired_method_comparison_uses_common_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "methods.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "method_a_distance_mm",
                        "method_b_distance_mm",
                        "method_c_distance_mm",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "method_a_distance_mm": 99,
                        "method_b_distance_mm": 100,
                        "method_c_distance_mm": 101,
                    }
                )
                writer.writerow(
                    {
                        "method_a_distance_mm": 98,
                        "method_b_distance_mm": "",
                        "method_c_distance_mm": 102,
                    }
                )
            independent = compare_csv(path, 100.0)
            paired = compare_csv(path, 100.0, paired_only=True)
            self.assertEqual(independent["method_a_single_pixel"]["sample_count"], 2)
            self.assertEqual(paired["method_a_single_pixel"]["sample_count"], 1)
            self.assertTrue(paired["method_c_ransac_plane"]["paired_only"])


if __name__ == "__main__":
    unittest.main()
