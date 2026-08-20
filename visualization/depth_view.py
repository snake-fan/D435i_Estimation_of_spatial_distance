"""Depth heatmap construction with invalid-pixel and ROI overlays."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


def colorize_depth(
    depth_image: np.ndarray,
    depth_scale: float,
    min_depth_m: float,
    max_depth_m: float,
    roi_polygons: Iterable[np.ndarray] = (),
) -> np.ndarray:
    image = np.asarray(depth_image)
    if image.ndim != 2:
        raise ValueError("depth_image must be a two-dimensional array")
    if (
        not np.isfinite(depth_scale)
        or not np.isfinite(min_depth_m)
        or not np.isfinite(max_depth_m)
        or depth_scale <= 0
        or not 0 <= min_depth_m < max_depth_m
    ):
        raise ValueError("Invalid depth visualization range")

    depth_m = image.astype(np.float32) * float(depth_scale)
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
        & (image != 0)
    )
    normalized = np.zeros(image.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (depth_m[valid] - min_depth_m)
        * (255.0 / (max_depth_m - min_depth_m)),
        0,
        255,
    ).astype(np.uint8)
    # Reverse so near objects are warm and far objects are cool.
    heatmap = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    heatmap[~valid] = (40, 40, 120)

    for polygon in roi_polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if points.shape == (4, 2) and np.isfinite(points).all():
            cv2.polylines(
                heatmap,
                [np.rint(points).astype(np.int32)],
                True,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return heatmap
