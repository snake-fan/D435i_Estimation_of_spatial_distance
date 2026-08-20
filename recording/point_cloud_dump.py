"""Debug export of the latest QR ROI point cloud and RANSAC labels."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


class PointCloudDumper:
    """Overwrite one file per QR with the latest available valid ROI cloud."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def dump(self, qr_id: str, result: Any) -> Path | None:
        points_value = getattr(result, "roi_points", None)
        if points_value is None:
            return None
        points = np.asarray(points_value, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
            return None
        if not np.isfinite(points).all():
            raise ValueError(f"{qr_id} point cloud contains non-finite coordinates")

        plane = getattr(result, "plane", None)
        mask_value = getattr(plane, "inlier_mask", None)
        if mask_value is None:
            inliers = np.zeros(len(points), dtype=bool)
        else:
            inliers = np.asarray(mask_value, dtype=bool)
            if inliers.shape != (len(points),):
                raise ValueError(
                    f"{qr_id} inlier mask shape {inliers.shape} does not match "
                    f"point count {len(points)}"
                )

        safe_name = "".join(
            character.lower() if character.isalnum() else "_" for character in qr_id
        ).strip("_") or "qr"
        destination = self.directory / f"{safe_name}_points.csv"
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["x", "y", "z", "inlier"])
            for point, is_inlier in zip(points, inliers, strict=True):
                writer.writerow(
                    [float(point[0]), float(point[1]), float(point[2]), int(is_inlier)]
                )
        temporary.replace(destination)
        return destination
