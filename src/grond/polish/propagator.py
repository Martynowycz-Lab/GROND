"""Bloch-wave row intensity and its exact adjoint with respect to the row Hamiltonian.

One row is a set of profile points p (rocking samples, mosaic-cone tilts) that share one static beam
coupling matrix and differ only in the real excitation-error diagonal:

    H_p = H_static + diag(Sg_p) + i*mu*I,     I = sum_p w_p sum_t w_t exp(-4*pi*mu*t) |[exp(2*pi*i*t*H0_p)]_10|^2

(the engine's ``propagate_direct_to_target_thickness_average`` with a uniform absorptive potential).  With
H0_p = V diag(lambda) V^H the derivative of exp(X), X = 2*pi*i*t*H0, is the Daleckii-Krein formula
dexp(X) = V (F o V^H dX V) V^H with the divided differences F_jk = (e^x_j - e^x_k) / (x_j - x_k).  For a
purely imaginary spectrum F_jk = exp(i*pi*t*(l_j + l_k)) * sinc(t*(l_j - l_k)), which is finite and smooth
at degenerate eigenvalues. The adjoint therefore has a finite limit without
differentiating individual eigenvectors, whose derivatives can be singular.

For implementation, first form R_j = sum_t w_t conj(a_t) exp(2*pi*i*lambda_j*t),
including absorption in w_t. The eigenbasis gradient is
2 conj(u_j v_k) conj[(R_j-R_k)/(lambda_j-lambda_k)], with the derivative limit
for repeated eigenvalues. This is the same divided difference, evaluated with
beam-by-thickness matrix-vector products instead of beam-pair thickness sums.

``row_intensity`` returns I, dI/dt for every thickness node and Gamma = sum_p w_p dI/dH (the torch/complex
convention: dI = Re(sum_ab conj(Gamma_ab) dH_ab)). The optional fourth result is dI/dw for the undamped
thickness weights, used when refining a distribution on fixed support nodes.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh

from grond.bloch.thickness import uniform_grid_phase_factors, uniform_grid_spec

# |pi*t*(l_j - l_k)| below which the divided difference uses its symmetric Taylor expansion.
DIVIDED_DIFFERENCE_TAYLOR_BELOW = 1e-4


def _thickness_factors(t_nodes: np.ndarray, t_weights: np.ndarray, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the thickness nodes (Å) and their weights times the uniform absorption factor exp(-4*pi*mu*t)."""

    t = np.asarray(t_nodes, dtype=float)
    return t, np.asarray(t_weights, dtype=float) * np.exp(-4.0 * math.pi * float(mu) * t)


def _phase_products(lam: np.ndarray, t: np.ndarray):
    """Forward and plain-transpose products with exp(2*pi*i*lambda_j*t_m).

    Uniform grids use the native block factorization for both products. The transpose sum is
    sum_r E[j,r]*(S.T@Q)[j,r], where Q is the node vector reshaped into zero-padded blocks. It is deliberately
    not a conjugate transpose; the intensity adjoint supplies its conjugated amplitude in Q.
    """

    grid = uniform_grid_spec(t) if len(t) >= 64 else None
    if grid is not None:
        reconstructed = grid[0] + grid[1] * np.arange(grid[2])
        # Reject near-uniform but deliberately perturbed nodes; allow only representation-level roundoff.
        if np.max(np.abs(t - reconstructed)) > 4 * np.spacing(np.max(np.abs(t))):
            grid = None
    if grid is None:
        phase = np.exp(2j * math.pi * lam[:, None] * t[None, :])
        return lambda coefficients: coefficients @ phase, lambda values: phase @ values

    within, shifts = uniform_grid_phase_factors(lam, *grid)
    within, shifts = within[0], shifts[0]

    def forward(coefficients):
        return ((shifts * coefficients[None, :]) @ within).ravel()[:len(t)]

    def transpose(values):
        blocks = np.zeros((shifts.shape[0], within.shape[1]), dtype=np.complex128)
        blocks.ravel()[:len(t)] = values
        return np.sum(within * (shifts.T @ blocks), axis=1)

    return forward, transpose


def row_intensity(
    h_static: np.ndarray,
    sg: np.ndarray,
    profile_weights: np.ndarray,
    t_nodes: np.ndarray,
    t_weights: np.ndarray,
    mu: float = 0.0,
    *,
    adjoint: bool = True,
    return_weight_gradient: bool = False,
) -> tuple:
    """Profile- and thickness-averaged intensity and its derivatives.

    Returns (I, dI/dt_nodes, dI/dH). With return_weight_gradient, appends
    dI/dw_nodes before absorption is applied. Process one profile point at a
    time so memory does not grow as profile_points * beams * thickness_nodes.
    """
    h_static = np.asarray(h_static, dtype=np.complex128)
    sg = np.atleast_2d(np.asarray(sg, dtype=float))
    weights = np.asarray(profile_weights, dtype=float)
    t, wt = _thickness_factors(t_nodes, t_weights, mu)
    n = h_static.shape[0]
    if h_static.shape != (n, n) or sg.shape[1] != n or sg.shape[0] != len(weights):
        raise ValueError("row_intensity: inconsistent Hamiltonian, excitation-error and weight shapes")
    intensity = 0.0
    d_t = np.zeros_like(t)
    d_w = np.zeros_like(t) if return_weight_gradient else None
    gamma = np.zeros((n, n), dtype=np.complex128) if adjoint else None
    damping = 4.0 * math.pi * float(mu)
    absorption = np.exp(-damping * t)
    t_max = float(np.max(np.abs(t))) if len(t) else 0.0
    diagonal = np.diag_indices(n)
    for excitation, profile_weight in zip(sg, weights, strict=True):
        if profile_weight == 0:
            continue
        matrix = h_static.copy()
        matrix[diagonal] += excitation
        lam, vecs = eigh(matrix)
        u = vecs[1, :]
        v = np.conj(vecs[0, :])
        coupling = u * v
        forward, transpose = _phase_products(lam, t)
        amplitude = forward(coupling)
        probability = np.abs(amplitude)**2
        intensity += float(profile_weight * (probability @ wt))
        amplitude_dt = forward(coupling * (2j * math.pi * lam))
        d_t += profile_weight * (2 * np.real(np.conj(amplitude) * amplitude_dt) - damping * probability) * wt
        if d_w is not None:
            d_w += profile_weight * probability * absorption
        if not adjoint:
            continue
        # R_j = sum_t w_t conj(a_t) exp(2*pi*i*lambda_j*t).
        # The eigenbasis gradient uses (R_j-R_k)/(lambda_j-lambda_k).
        # Forming R first avoids a beams-by-beams-by-thickness contraction.
        weighted_conjugate = wt * np.conj(amplitude)
        response = transpose(weighted_conjugate)
        difference = lam[:, None] - lam[None, :]
        small = math.pi * np.abs(difference) * t_max < DIVIDED_DIFFERENCE_TAYLOR_BELOW
        divided = (response[:, None] - response[None, :]) / np.where(small, 1.0, difference)
        if np.any(small):
            frequency = 2j * math.pi * t
            first = transpose(weighted_conjugate * frequency)
            divided[diagonal] = first
            off_diagonal_small = small.copy()
            off_diagonal_small[diagonal] = False
            # Symmetric expansion of the divided difference, accurate through
            # cubic order and finite at repeated eigenvalues.
            if np.any(off_diagonal_small):
                third = transpose(weighted_conjugate * frequency**3)
                i, j = np.nonzero(off_diagonal_small)
                divided[i, j] = (first[i] + first[j]) / 2 - difference[i, j]**2 * (third[i] + third[j]) / 24
        k_matrix = 2 * np.conj(u)[:, None] * np.conj(v)[None, :] * np.conj(divided)
        gamma += profile_weight * (vecs @ k_matrix @ np.conj(vecs).T)
    result = (float(intensity), d_t, gamma)
    return (*result, d_w) if return_weight_gradient else result


def born_intensity(
    h10_abs2: float,
    delta_sg: np.ndarray,
    profile_weights: np.ndarray,
    t_nodes: np.ndarray,
    t_weights: np.ndarray,
) -> float:
    """Kinematic two-beam row intensity of the engine (its matched Born control) for one row."""

    t = np.asarray(t_nodes, dtype=float)
    w = np.asarray(t_weights, dtype=float)
    delta = np.asarray(delta_sg, dtype=float)
    sinc = np.sinc(delta[:, None] * t[None, :])
    per_point = (2.0 * math.pi) ** 2 * float(h10_abs2) * ((t[None, :] ** 2) * sinc**2) @ w
    return float(np.asarray(profile_weights, dtype=float) @ per_point)
