"""Regularized detector flat-field utilities used by GROND row-scale fits.

Detector factors are nuisance parameters, not a substitute for the dynamical
model.  The solver therefore shrinks every tile toward unity, penalizes jumps
between neighboring tiles, and refuses to let sparsely populated tiles acquire
an independent data-driven value.
"""

from __future__ import annotations

import math

import numpy as np


def infer_square_grid_size(bin_ids: np.ndarray, requested: int | None = None) -> int:
    """Return the NxN detector-grid width represented by integer bin IDs."""

    if requested is not None and int(requested) > 1:
        return int(requested)
    finite = np.asarray(bin_ids, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 1
    return max(1, int(math.ceil(math.sqrt(int(np.max(finite)) + 1))))


def solve_regularized_detector_field(
    diag: np.ndarray,
    rhs: np.ndarray,
    counts: np.ndarray,
    *,
    n_bins: int,
    ridge: float,
    smoothness: float,
    min_rows: int,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
) -> np.ndarray:
    """Solve one square detector field with dimensionless regularization.

    ``diag`` and ``rhs`` are the independent weighted normal-equation terms for
    each tile.  Ridge and Laplacian strengths are expressed relative to the
    median adequately populated tile information, which keeps the controls
    meaningful across intensity units and datasets.  Tiles below ``min_rows``
    contribute no independent data term and are determined only by the unity
    prior and their populated neighbors.
    """

    n_bins = int(n_bins)
    if n_bins <= 1:
        return np.ones(1, dtype=float)
    n_tiles = n_bins * n_bins
    diag = np.asarray(diag, dtype=float).reshape(-1)
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    counts = np.asarray(counts, dtype=int).reshape(-1)
    if not (len(diag) == len(rhs) == len(counts) == n_tiles):
        raise ValueError("detector normal-equation arrays do not match the requested square grid")
    if ridge < 0.0 or smoothness < 0.0:
        raise ValueError("detector ridge and smoothness penalties must be nonnegative")
    if int(min_rows) < 2:
        raise ValueError("detector min_rows must be at least two")

    adequate = (
        (counts >= int(min_rows))
        & np.isfinite(diag)
        & np.isfinite(rhs)
        & (diag > 0.0)
    )
    if not np.any(adequate):
        return np.ones(n_tiles, dtype=float)
    information = float(np.median(diag[adequate]))
    information = max(information, 1e-30)
    ridge_strength = float(ridge) * information
    smooth_strength = float(smoothness) * information

    matrix = np.zeros((n_tiles, n_tiles), dtype=float)
    vector = np.zeros(n_tiles, dtype=float)
    matrix[np.diag_indices(n_tiles)] = np.where(adequate, diag, 0.0) + ridge_strength
    vector[:] = np.where(adequate, rhs, 0.0) + ridge_strength

    if smooth_strength > 0.0:
        for y in range(n_bins):
            for x in range(n_bins):
                here = y * n_bins + x
                if x + 1 < n_bins:
                    other = here + 1
                    matrix[here, here] += smooth_strength
                    matrix[other, other] += smooth_strength
                    matrix[here, other] -= smooth_strength
                    matrix[other, here] -= smooth_strength
                if y + 1 < n_bins:
                    other = here + n_bins
                    matrix[here, here] += smooth_strength
                    matrix[other, other] += smooth_strength
                    matrix[here, other] -= smooth_strength
                    matrix[other, here] -= smooth_strength

    try:
        scales = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        scales = np.linalg.lstsq(matrix, vector, rcond=None)[0]
    scales = np.asarray(scales, dtype=float)
    scales[~np.isfinite(scales)] = 1.0
    scales = np.clip(scales, float(minimum_scale), float(maximum_scale))
    return scales


def detector_field_diagnostics(
    scales: np.ndarray,
    counts: np.ndarray,
    *,
    n_bins: int,
    min_rows: int,
) -> dict[str, object]:
    """Return auditable detector-field support and range diagnostics."""

    scales = np.asarray(scales, dtype=float).reshape(-1)
    counts = np.asarray(counts, dtype=int).reshape(-1)
    n_tiles = int(n_bins) * int(n_bins) if int(n_bins) > 1 else 0
    adequate = counts >= int(min_rows)
    finite = scales[np.isfinite(scales) & (scales > 0.0)]
    return {
        "grid_bins_per_axis": int(n_bins) if int(n_bins) > 1 else 0,
        "nominal_parameter_count": int(n_tiles),
        "adequately_populated_tiles": int(np.count_nonzero(adequate)),
        "underpopulated_tiles": int(np.count_nonzero(~adequate)) if n_tiles else 0,
        "minimum_rows_per_tile": int(np.min(counts)) if len(counts) else 0,
        "median_rows_per_tile": float(np.median(counts)) if len(counts) else 0.0,
        "required_rows_per_tile": int(min_rows),
        "all_tiles_adequate": bool(n_tiles == 0 or np.all(adequate)),
        "scale_min": float(np.min(finite)) if len(finite) else 1.0,
        "scale_max": float(np.max(finite)) if len(finite) else 1.0,
    }
