"""Robust plane recovery followed by SVD refinement."""

from __future__ import annotations

from typing import Any

import numpy as np

from .plane_svd import PlaneModel, fit_plane_svd


def _classify_plane(
    cloud: np.ndarray,
    finite: np.ndarray,
    plane: PlaneModel,
    distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify every finite point against ``plane`` using RANSAC semantics."""

    distances = np.full(cloud.shape[0], np.inf, dtype=np.float64)
    distances[finite] = np.abs(cloud[finite] @ plane.normal + plane.d)
    return distances < distance_threshold, distances


def fit_plane_ransac(
    points: np.ndarray,
    iterations: int,
    distance_threshold: float,
    min_inlier_ratio: float = 0.0,
    *,
    random_seed: int | None = None,
    rng: Any | None = None,
    degenerate_epsilon: float = 1.0e-12,
) -> PlaneModel | None:
    """Find a plane's inliers with RANSAC and refit them with SVD.

    ``min_inlier_ratio`` is validated but deliberately not used to suppress a
    valid best model.  The quality gate needs that low-ratio model in order to
    distinguish ``low_inlier_ratio`` from a true ``ransac_failed`` condition.
    Candidate models are ordered by inlier count; RMS breaks only exact count
    ties, as required by the project design.
    """

    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1:] != (3,):
        raise ValueError("points must have shape (N, 3)")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    iterations = int(iterations)
    if not np.isfinite(distance_threshold) or distance_threshold <= 0.0:
        raise ValueError("distance_threshold must be positive and finite")
    if not np.isfinite(min_inlier_ratio) or not 0.0 <= min_inlier_ratio <= 1.0:
        raise ValueError("min_inlier_ratio must be in [0, 1]")
    if not np.isfinite(degenerate_epsilon) or degenerate_epsilon <= 0.0:
        raise ValueError("degenerate_epsilon must be positive and finite")
    if rng is not None and random_seed is not None:
        raise ValueError("provide either rng or random_seed, not both")

    finite = np.all(np.isfinite(cloud), axis=1)
    finite_indices = np.flatnonzero(finite)
    if finite_indices.size < 3:
        return None
    generator = np.random.default_rng(random_seed) if rng is None else rng

    best_mask: np.ndarray | None = None
    best_count = -1
    best_rms = float("inf")

    for _ in range(iterations):
        try:
            sample_indices = generator.choice(finite_indices, size=3, replace=False)
        except (TypeError, ValueError):
            return None
        first, second, third = cloud[sample_indices]
        normal = np.cross(second - first, third - first)
        normal_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal_norm) or normal_norm <= degenerate_epsilon:
            continue
        normal /= normal_norm
        d = -float(normal @ first)

        distances = np.full(cloud.shape[0], np.inf, dtype=np.float64)
        distances[finite] = np.abs(cloud[finite] @ normal + d)
        candidate_mask = distances < float(distance_threshold)
        candidate_count = int(np.count_nonzero(candidate_mask))
        if candidate_count < 3:
            continue
        candidate_rms = float(
            np.sqrt(np.mean(np.square(distances[candidate_mask])))
        )

        if candidate_count > best_count or (
            candidate_count == best_count and candidate_rms < best_rms
        ):
            best_mask = candidate_mask
            best_count = candidate_count
            best_rms = candidate_rms

    if best_mask is None:
        return None

    # A least-squares refit can move the plane enough for threshold-boundary
    # points to enter or leave the consensus set.  Reclassify against that
    # refitted plane, then refit once more to the updated consensus.
    first_refit = fit_plane_svd(
        cloud,
        best_mask,
        degenerate_epsilon=degenerate_epsilon,
    )
    if first_refit is None:
        return None
    refined_mask, _ = _classify_plane(
        cloud,
        finite,
        first_refit,
        float(distance_threshold),
    )
    if np.count_nonzero(refined_mask) < 3:
        return None

    second_refit = fit_plane_svd(
        cloud,
        refined_mask,
        degenerate_epsilon=degenerate_epsilon,
    )
    if second_refit is None:
        return None

    # The second SVD can make one last, usually tiny, boundary change.  The
    # returned mask and diagnostics must describe the final plane itself, not
    # the preceding consensus set used to estimate it.
    final_mask, final_distances = _classify_plane(
        cloud,
        finite,
        second_refit,
        float(distance_threshold),
    )
    final_count = int(np.count_nonzero(final_mask))
    if final_count < 3:
        return None
    final_rms = float(
        np.sqrt(np.mean(np.square(final_distances[final_mask])))
    )
    return PlaneModel(
        normal=second_refit.normal,
        d=second_refit.d,
        centroid=second_refit.centroid,
        inlier_mask=final_mask,
        inlier_count=final_count,
        inlier_ratio=float(final_count / cloud.shape[0]),
        rms=final_rms,
    )


__all__ = ["PlaneModel", "fit_plane_ransac"]
