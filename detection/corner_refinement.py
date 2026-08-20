"""Best-effort QR corner refinement.

OpenCV's QR detector already returns usable corners.  Refinement is therefore
deliberately non-fatal: malformed input, unavailable OpenCV bindings, or a
``cornerSubPix`` failure all return the original corners unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def refine_qr_corners(
    gray_image: np.ndarray,
    corners: np.ndarray,
    *,
    window_size: int = 5,
    max_iterations: int = 30,
    epsilon: float = 0.01,
    cv2_module: Any | None = None,
) -> np.ndarray:
    """Return sub-pixel corners, falling back to ``corners`` on any failure.

    Parameters are intentionally small and conservative because QR vertices
    can be close to an image boundary.  A returned array always has dtype
    ``float64`` and shape ``(4, 2)`` when the supplied corners have that shape.
    """

    original = np.asarray(corners, dtype=np.float64).copy()
    image = np.asarray(gray_image)

    if original.shape != (4, 2) or not np.all(np.isfinite(original)):
        return original
    if image.ndim != 2 or image.size == 0:
        return original
    if window_size <= 0 or max_iterations <= 0 or epsilon <= 0:
        return original

    height, width = image.shape
    # cornerSubPix requires every initial point to lie inside the image.  It
    # can still refine points close to an edge, so only reject actual OOB data.
    if (
        np.any(original[:, 0] < 0.0)
        or np.any(original[:, 0] >= width)
        or np.any(original[:, 1] < 0.0)
        or np.any(original[:, 1] >= height)
    ):
        return original

    try:
        if cv2_module is None:
            import cv2 as cv2_module  # type: ignore[no-redef]

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        working = original.astype(np.float32).reshape(-1, 1, 2)
        criteria = (
            cv2_module.TERM_CRITERIA_EPS | cv2_module.TERM_CRITERIA_MAX_ITER,
            int(max_iterations),
            float(epsilon),
        )
        returned = cv2_module.cornerSubPix(
            np.ascontiguousarray(image),
            working,
            (int(window_size), int(window_size)),
            (-1, -1),
            criteria,
        )
        candidate = working if returned is None else np.asarray(returned)
        candidate = np.asarray(candidate, dtype=np.float64).reshape(4, 2)
    except Exception:
        return original

    if not np.all(np.isfinite(candidate)):
        return original
    return candidate


# A concise alias is convenient for callers and preserves the module name from
# the project design without duplicating implementation.
refine_corners = refine_qr_corners


__all__ = ["refine_corners", "refine_qr_corners"]
