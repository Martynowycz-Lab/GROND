"""P-spline frame scaling with automatic smoothing selection and positive scale guarantees.

Implements cubic B-spline bases with second-difference penalties (Eilers & Marx, 1996),
analytical Effective Degrees of Freedom (EDF), Generalized Cross-Validation (GCV) /
orbit-based cross-validation for lambda selection, stage-slip jump handling, and
strictly positive scale bounds.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import block_diag
from scipy.optimize import nnls


def calculate_weights(
    y: np.ndarray,
    sigma: np.ndarray,
    objective: Literal["sigma", "row_mr"] = "row_mr",
) -> np.ndarray:
    """Return weights for the intensity-residual scale fit.

    ``sigma`` uses inverse intensity variances. ``row_mr`` derives empirical weights
    from amplitude uncertainties but still applies them to intensity residuals.
    """
    y = np.asarray(y, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-30)

    if objective == "sigma":
        return 1.0 / (sigma**2)

    # row_mr: robust amplitude-domain / counting weighting
    weak = y < 0.01 * sigma
    f = np.sqrt(np.maximum(np.where(weak, 0.0, y), 0.0))
    sf = sigma / np.maximum(2.0 * f, 1e-30)
    sf[weak] = 5.0 * np.sqrt(sigma[weak])
    return 1.0 / (sf**2 + (0.01 * f) ** 2)


def detect_stage_slip_jumps(
    angle: np.ndarray,
    group: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    raw: np.ndarray,
    fit_mask: np.ndarray,
    objective: Literal["sigma", "row_mr"] = "row_mr",
    max_jumps: int = 2,
    coarse_bin_size: int = 16,
) -> list[float]:
    """Detect significant discrete scale steps (e.g. stage slips) using training data only."""
    if max_jumps <= 0:
        return []

    coarse = group // coarse_bin_size
    w = calculate_weights(y, sigma, objective)
    values = []

    unique_coarse = np.unique(coarse[fit_mask])
    for g in unique_coarse:
        m = fit_mask & (coarse == g)
        if m.sum() < min(10, coarse_bin_size):
            continue
        sum_w_raw2 = float(np.sum(w[m] * raw[m] ** 2))
        sum_w_raw_y = float(np.sum(w[m] * raw[m] * y[m]))
        if sum_w_raw2 <= 1e-30:
            continue
        scale = sum_w_raw_y / sum_w_raw2
        if scale > 0:
            med_angle = float(np.median(angle[m]))
            var_log_scale = 1.0 / max(sum_w_raw2 * scale**2, 1e-30)
            values.append((g, med_angle, np.log(scale), var_log_scale))

    if len(values) < 2:
        return []

    candidates = []
    for a, b in zip(values[:-1], values[1:], strict=False):
        if b[0] - a[0] > 2:
            continue
        z_contrast = abs(b[2] - a[2]) / math.sqrt(a[3] + b[3])
        location = 0.5 * (a[1] + b[1])
        # Require substantial statistical contrast and keep clear of edges
        lo_bound = float(np.min(angle[fit_mask])) + 8.0
        hi_bound = float(np.max(angle[fit_mask])) - 8.0
        if z_contrast >= 3.5 and lo_bound < location < hi_bound:
            candidates.append((z_contrast, location))

    chosen = []
    for _, loc in sorted(candidates, reverse=True):
        if all(abs(loc - prev) > 8.0 for prev in chosen):
            chosen.append(loc)
        if len(chosen) >= max_jumps:
            break

    return sorted(chosen)


def construct_spline_basis(
    angle: np.ndarray,
    fit_mask: np.ndarray,
    n_basis_target: int = 32,
    breakpoints: list[float] | None = None,
    min_points_per_segment: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Construct the B-spline basis and second-difference penalty on the fitted angle range.

    This basis does not extrapolate. Callers that want constant endpoint scales must
    clip prediction angles to the training range before calling; otherwise out-of-range
    rows are rejected rather than receiving an all-zero basis and scale.
    """
    angle = np.asarray(angle, dtype=float)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if angle.ndim != 1 or fit_mask.shape != angle.shape or not fit_mask.any():
        raise ValueError("spline angles need a matching nonempty training mask")
    if not np.all(np.isfinite(angle)):
        raise ValueError("spline angles must be finite")
    lo = float(np.min(angle[fit_mask]))
    hi = float(np.max(angle[fit_mask]))
    if np.any((angle < lo) | (angle > hi)):
        raise ValueError(
            "spline prediction angles lie outside the training range; "
            "clip them to the training endpoints for constant extrapolation"
        )
    if hi <= lo:
        hi = lo + 1.0

    # Physical gaps (> 3.0 deg without reflections)
    unique_angles = np.unique(angle[fit_mask])
    dif = np.diff(unique_angles)
    gap_locs = [float(0.5 * (unique_angles[i] + unique_angles[i + 1])) for i in np.where(dif > 3.0)[0]]

    all_cuts = sorted(set(gap_locs + (breakpoints or [])))
    valid_cuts = []
    for cut in all_cuts:
        left = lo if not valid_cuts else valid_cuts[-1]
        n_left = np.count_nonzero(fit_mask & (angle >= left) & (angle < cut))
        n_right = np.count_nonzero(fit_mask & (angle >= cut))
        if n_left >= min_points_per_segment and n_right >= min_points_per_segment:
            valid_cuts.append(cut)

    boundaries = [lo] + valid_cuts + [hi]
    blocks = []
    penalties = []

    for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=False)):
        m = (angle >= a) & (angle <= b if i == len(boundaries) - 2 else angle < b)
        seg_frac = (b - a) / (hi - lo)
        k_seg = max(4, int(round(n_basis_target * seg_frac)))
        pts_in_seg = int(np.count_nonzero(fit_mask & m))
        k_seg = min(k_seg, max(4, pts_in_seg // 3))

        knots = np.r_[np.repeat(a, 4), np.linspace(a, b, k_seg - 2)[1:-1], np.repeat(b, 4)]
        basis_sub = BSpline.design_matrix(angle[m], knots, 3, extrapolate=False).toarray()

        block = np.zeros((len(angle), basis_sub.shape[1]), dtype=float)
        block[m] = basis_sub
        blocks.append(block)

        # 2nd difference penalty matrix D (size (k_seg-2) x k_seg)
        D_seg = np.diff(np.eye(basis_sub.shape[1]), n=2, axis=0)
        penalties.append(D_seg)

    B = np.hstack(blocks)
    D = block_diag(*penalties)
    return B, D, valid_cuts


def pseudo_inverse(matrix: np.ndarray, rcond: float) -> np.ndarray:
    """numpy.linalg.pinv(matrix, rcond); the same pseudo-inverse from LAPACK gesvd when gesdd does not converge.

    Apple Accelerate's gesdd (numpy's SVD on macOS) can report "SVD did not converge" for a well-conditioned
    matrix: a 32 x 32 P-spline system of condition number 27 from an AUTO nested nuisance selection did. gesvd
    computes the same decomposition, and the cutoff is numpy's (singular values <= rcond * max are dropped), so a
    result is identical to rounding; matrices numpy handles are unchanged.
    """
    try:
        return np.linalg.pinv(matrix, rcond=rcond)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd

        u, s, vt = svd(np.asarray(matrix, dtype=float), full_matrices=False, lapack_driver="gesvd")
        large = s > float(rcond) * np.max(s, initial=0.0)
        inverse = np.zeros_like(s)
        inverse[large] = 1.0 / s[large]
        return (vt.T * inverse) @ u.T


class PSplineScaleFitter:
    """Production P-spline model for continuous illumination and frame scaling."""

    def __init__(
        self,
        n_basis: int = 32,
        objective: Literal["sigma", "row_mr"] = "row_mr",
        max_jumps: int = 0,
        floor_factor: float = 1e-12,
    ) -> None:
        """Set the target basis size, weight objective, stage-slip jumps to detect (0 = none) and coefficient floor."""
        self.n_basis = n_basis
        self.objective = objective
        self.max_jumps = max_jumps
        self.floor_factor = floor_factor

    def _prepare_system(
        self,
        y: np.ndarray,
        sigma: np.ndarray,
        raw: np.ndarray,
        B: np.ndarray,
        D: np.ndarray,
        fit_mask: np.ndarray,
    ) -> dict[str, Any]:
        """Build the lambda-independent normal equations of a P-spline scale fit on the fit_mask rows.

        Returns the weights, raw_scale (normalizes raw to the target), A = raw x B, G = A'WA, rhs = A'Wy, P = D'D
        and penalty_scale (median diag G / median diag P), which makes lambda dimensionless.
        """
        w = calculate_weights(y, sigma, self.objective)
        fit_y = y[fit_mask]
        fit_raw = raw[fit_mask]
        fit_w = w[fit_mask]

        denom = max(np.sum(fit_w * fit_raw**2), 1e-300)
        raw_scale = max(float(np.sqrt(np.sum(fit_w * fit_y**2) / denom)), np.finfo(float).tiny)

        A = (raw * raw_scale)[:, None] * B
        A_fit = A[fit_mask]
        G = A_fit.T @ (fit_w[:, None] * A_fit)
        rhs = A_fit.T @ (fit_w * fit_y)
        P = D.T @ D

        diag_G = np.diag(G)
        diag_P = np.diag(P)
        pos_G = diag_G[diag_G > 0]
        pos_P = diag_P[diag_P > 0]
        med_G = float(np.median(pos_G)) if len(pos_G) else 1.0
        med_P = float(np.median(pos_P)) if len(pos_P) else 1.0
        penalty_scale = med_G / max(med_P, 1e-30)

        return {
            "w": w,
            "raw_scale": raw_scale,
            "A": A,
            "G": G,
            "rhs": rhs,
            "P": P,
            "penalty_scale": penalty_scale,
        }

    def _solve_for_lambda(
        self,
        system: dict[str, Any],
        B: np.ndarray,
        lam: float,
        fit_mask: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """Solve one penalized fit (G + lambda penalty_scale P) beta = rhs with every coefficient at least floor_factor.

        Returns the prediction and scale for every row, EDF (trace of the hat matrix on the coefficients off the
        bound), GCV on the fit rows and the number of coefficients held at the bound.
        EDF is conditional on the weights, lambda and active bounds; it does not count
        selection of smoothing or jump locations.
        """
        G = system["G"]
        rhs = system["rhs"]
        P = system["P"]
        penalty_scale = system["penalty_scale"]
        raw_scale = system["raw_scale"]
        A = system["A"]

        K = G + lam * penalty_scale * P
        eig, V = np.linalg.eigh(K)
        tol = max(float(eig.max()) * 1e-12, 1e-30)
        keep = eig > tol

        beta = V[:, keep] @ ((V[:, keep].T @ rhs) / eig[keep])
        floor = self.floor_factor

        # Active-set bound enforcement if any coefficient violates floor
        if np.min(beta) < floor:
            M = np.sqrt(eig[keep])[:, None] * V[:, keep].T
            b = (V[:, keep].T @ (rhs - K @ np.full(len(beta), floor))) / np.sqrt(eig[keep])
            beta = nnls(M, b, maxiter=max(1000, 10 * len(beta)))[0] + floor

        free = beta > floor + 1e-9 * max(float(np.max(beta)), np.finfo(float).tiny)
        KF = K[np.ix_(free, free)]
        GF = G[np.ix_(free, free)]
        F = pseudo_inverse(KF, rcond=1e-12) @ GF

        edf = float(np.trace(F))
        trace_hth = float(np.trace(F @ F))

        prediction = A @ beta
        scale = (B @ beta) * raw_scale

        # GCV calculation on training reflections
        fit_res = y[fit_mask] - prediction[fit_mask]
        fit_w = system["w"][fit_mask]
        n_fit = len(fit_res)
        denom_gcv = max(1.0 - edf / max(n_fit, 1), 1e-6)
        rss = float(np.sum(fit_w * fit_res**2))
        gcv = (rss / max(n_fit, 1)) / (denom_gcv**2)

        return {
            "lambda": lam,
            "beta": beta * raw_scale,
            "prediction": prediction,
            "scale": scale,
            "edf": edf,
            "trace_hth": trace_hth,
            "gcv": gcv,
            "active_bounds": int((~free).sum()),
            "raw_count": len(beta),
        }

    def fit(
        self,
        y: np.ndarray,
        sigma: np.ndarray,
        raw: np.ndarray,
        angle: np.ndarray,
        group: np.ndarray | None = None,
        fit_mask: np.ndarray | None = None,
        lambda_smooth: float | None = None,
        orbit_folds: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Fit the P-spline model, automatically selecting lambda if not specified."""
        y = np.asarray(y, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        raw = np.asarray(raw, dtype=float)
        angle = np.asarray(angle, dtype=float)
        group = np.arange(len(y), dtype=int) if group is None else np.asarray(group, dtype=int)

        fit_mask = np.ones(len(y), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)

        # Detect any jumps
        breakpoints = []
        if self.max_jumps > 0:
            breakpoints = detect_stage_slip_jumps(
                angle, group, y, sigma, raw, fit_mask, self.objective, self.max_jumps
            )

        B, D, cuts = construct_spline_basis(
            angle, fit_mask, n_basis_target=self.n_basis, breakpoints=breakpoints
        )
        system = self._prepare_system(y, sigma, raw, B, D, fit_mask)

        # Lambda selection
        if lambda_smooth is not None:
            best_sol = self._solve_for_lambda(system, B, lambda_smooth, fit_mask, y)
        elif orbit_folds is not None:
            # Whole-orbit cross validation within training data
            candidate_lambdas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
            best_lam = 1.0
            best_cv_score = float("inf")

            for lam in candidate_lambdas:
                oof_preds = np.zeros(len(y), dtype=float)
                for f in range(3):
                    inner_train = fit_mask & (orbit_folds != f)
                    inner_val = fit_mask & (orbit_folds == f)
                    if not inner_val.any():
                        continue
                    sys_f = self._prepare_system(y, sigma, raw, B, D, inner_train)
                    sol_f = self._solve_for_lambda(sys_f, B, lam, inner_train, y)
                    oof_preds[inner_val] = sol_f["prediction"][inner_val]

                val_mask = fit_mask
                r_all = float(np.abs(np.sqrt(np.maximum(y[val_mask], 0.0)) - np.sqrt(np.maximum(oof_preds[val_mask], 0.0))).sum() / max(np.sqrt(np.maximum(y[val_mask], 0.0)).sum(), 1e-30))
                if r_all < best_cv_score:
                    best_cv_score = r_all
                    best_lam = lam

            best_sol = self._solve_for_lambda(system, B, best_lam, fit_mask, y)
        else:
            # GCV optimization over grid
            candidate_lambdas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0]
            sols = [self._solve_for_lambda(system, B, lam, fit_mask, y) for lam in candidate_lambdas]
            best_sol = min(sols, key=lambda s: s["gcv"])

        best_sol["B"] = B
        best_sol["D"] = D
        best_sol["cuts"] = cuts
        return best_sol
