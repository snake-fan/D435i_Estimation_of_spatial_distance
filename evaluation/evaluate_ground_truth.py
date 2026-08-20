"""Compute absolute accuracy metrics from a measurement CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def accuracy_metrics(measurements_mm: Iterable[float], ground_truth_mm: float) -> dict[str, float | int]:
    values = np.asarray(list(measurements_mm), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite measurement values were provided")
    if not math.isfinite(ground_truth_mm):
        raise ValueError("Ground truth must be finite")
    errors = values - float(ground_truth_mm)
    absolute = np.abs(errors)
    return {
        "sample_count": int(values.size),
        "ground_truth_mm": float(ground_truth_mm),
        "mean_mm": float(np.mean(values)),
        "median_mm": float(np.median(values)),
        "repeatability_std_mm": float(np.std(values, ddof=0)),
        "bias_mm": float(np.mean(errors)),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))),
        "p95_absolute_error_mm": float(np.percentile(absolute, 95)),
    }


def read_measurements(
    path: str | Path,
    value_column: str = "distance_mm",
    accepted_statuses: set[str] | None = None,
) -> list[float]:
    accepted = {"GOOD", "WARNING"} if accepted_statuses is None else accepted_statuses
    values: list[float] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or value_column not in reader.fieldnames:
            raise ValueError(f"CSV does not contain column '{value_column}'")
        for row in reader:
            raw_status = row.get("status")
            if raw_status is None or not raw_status.strip():
                continue
            status = raw_status.upper()
            if accepted and status not in accepted:
                continue
            raw = row.get(value_column, "")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--ground-truth-mm", type=float, required=True)
    parser.add_argument("--column", default="distance_mm")
    parser.add_argument(
        "--include-status",
        action="append",
        dest="statuses",
        help="Accepted status (repeatable; defaults to GOOD and WARNING)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    statuses = {value.upper() for value in args.statuses} if args.statuses else None
    values = read_measurements(args.csv_file, args.column, statuses)
    metrics = accuracy_metrics(values, args.ground_truth_mm)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
