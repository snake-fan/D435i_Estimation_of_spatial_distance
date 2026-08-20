"""Compare pre-recorded baseline A/B/C distance columns against ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

if __package__:
    from .evaluate_ground_truth import accuracy_metrics
else:  # Supports ``python evaluation/compare_methods.py`` from the repo root.
    from evaluate_ground_truth import accuracy_metrics


DEFAULT_COLUMNS = {
    "method_a_single_pixel": "method_a_distance_mm",
    "method_b_roi_median": "method_b_distance_mm",
    "method_c_ransac_plane": "method_c_distance_mm",
}


def compare_csv(
    path: str | Path,
    ground_truth_mm: float,
    columns: dict[str, str] | None = None,
    *,
    paired_only: bool = False,
) -> dict[str, dict[str, float | int | bool]]:
    selected = columns or DEFAULT_COLUMNS
    values: dict[str, list[float]] = {name: [] for name in selected}
    total_rows = 0
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in selected.values() if column not in fieldnames]
        if missing:
            raise ValueError(f"CSV is missing method column(s): {', '.join(missing)}")
        for row in reader:
            total_rows += 1
            row_values: dict[str, float] = {}
            for name, column in selected.items():
                try:
                    value = float(row[column])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    row_values[name] = value
            if paired_only:
                if len(row_values) == len(selected):
                    for name, value in row_values.items():
                        values[name].append(value)
            else:
                for name, value in row_values.items():
                    values[name].append(value)
    empty = [name for name, method_values in values.items() if not method_values]
    if empty:
        raise ValueError(
            "No finite measurements for method(s): " + ", ".join(empty)
        )
    result: dict[str, dict[str, float | int | bool]] = {}
    for name, method_values in values.items():
        metrics = accuracy_metrics(method_values, ground_truth_mm)
        metrics["input_row_count"] = total_rows
        metrics["valid_rate"] = (
            float(len(method_values) / total_rows) if total_rows else 0.0
        )
        metrics["paired_only"] = paired_only
        result[name] = metrics
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--ground-truth-mm", type=float, required=True)
    parser.add_argument(
        "--method",
        action="append",
        metavar="NAME=COLUMN",
        help="Override method mapping; repeat for multiple methods",
    )
    parser.add_argument(
        "--paired-only",
        action="store_true",
        help="Compare only rows where every selected method is finite",
    )
    return parser


def _parse_methods(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --method '{value}', expected NAME=COLUMN")
        name, column = (part.strip() for part in value.split("=", 1))
        if not name or not column:
            raise ValueError(f"Invalid --method '{value}', expected NAME=COLUMN")
        result[name] = column
    return result


def main() -> int:
    args = build_parser().parse_args()
    methods = _parse_methods(args.method)
    comparison = compare_csv(
        args.csv_file,
        args.ground_truth_mm,
        methods,
        paired_only=args.paired_only,
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
