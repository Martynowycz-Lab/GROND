"""Shared orientation grid: one Bloch eigensystem per scan orientation, shared by every row that rocks through it.

The per-row engine solves one Hamiltonian per profile point of every row (rows x points x tilts matrices). A
rotation scan visits each orientation once, and every row whose rocking window covers that orientation needs the
same eigensystem, only for a different target beam. This evaluator solves one Hamiltonian per orientation of a
uniform scan-angle grid (step 1/per_degree degrees, one grid per transverse mosaic tilt), extracts the couplings of
every target active there from one Householder back-transformation, and maps the grid onto each row's own profile
points and weights with not-a-knot cubic-spline cardinal weights (4 margin points). Profile weights, Jacobians,
mosaic tilts, windows, Born terms and the thickness evaluation are the per-row engine's.

The beams of a grid Hamiltonian are those with |Sg| <= max_support_sg at the grid orientation (the per-row rule
applied at the grid angle rather than at each row's centre), plus every active target; with
Config.basis_centring_coset the pool holds only the direct beam's centring coset, which is exact. Dose damage uses
the grid angle's frame. Grid solves run in float64 by default (GROND_SHARED_GRID_PRECISION=single for float32): one
float32 MRRR failure in 16,888 SU-102 rows was measured where float64 agreed to 1e-11 in either beam order.

Measured on SU-102 (triage_evidence/su102_shared_grid_20261013): at cutoffs >= 0.02 1/A the grid reproduces per-row
values with the same basis rule to median 0.05-0.08%, p95 0.24-0.34% at 100 per degree; a full pass at Plan D's
settings took 4.3 CPU-hours against about 760 for the per-row pass, with R equal to 0.01-0.02 pp on work and free rows.

Selection (GROND_SHARED_GRID): off (default), auto (use the grid when the configuration is supported and the cost
model says it is clearly cheaper; see choose_path), force (always when supported). GROND_SHARED_GRID_PER_DEGREE
(default 100) sets the grid density.

Supported: target-Sg profiles, mosaicity none or transverse_target, no Sg jitter, no absorption, the table thickness
evaluator (predictions) or response tables, scipy backends, XDS/DIALS rotation geometry (not PETS zone files).
Anything else falls back to the per-row engine.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal, lapack

from grond.bloch.thickness import resolved_config_evaluator, uniform_grid_phase_factors, uniform_grid_spec

ENV_MODE = "GROND_SHARED_GRID"
ENV_PER_DEGREE = "GROND_SHARED_GRID_PER_DEGREE"
ENV_PRECISION = "GROND_SHARED_GRID_PRECISION"
ENV_MIN_ROWS = "GROND_SHARED_GRID_MIN_ROWS"
ENV_TABLE_PER_DEGREE = "GROND_SHARED_GRID_TABLE_PER_DEGREE"
ENV_GPUS = "GROND_SHARED_GRID_GPUS"
ENV_GPU_PRECISION = "GROND_SHARED_GRID_GPU_PRECISION"
GPU_BUCKET = 128
MARGIN = 4
TARGET_CHUNK = 32

_G: dict = {}


# ----------------------------------------------------------------------------- settings and the path chooser


def mode() -> str:
    """off, auto or force from GROND_SHARED_GRID (1/on/true mean auto)."""
    value = str(os.environ.get(ENV_MODE, "off") or "off").strip().lower()
    if value in {"1", "on", "true", "yes", "auto"}:
        return "auto"
    if value == "force":
        return "force"
    return "off"


def per_degree() -> float:
    value = float(os.environ.get(ENV_PER_DEGREE, "100") or 100)
    if not math.isfinite(value) or value < 10.0:
        raise ValueError(f"{ENV_PER_DEGREE} must be at least 10 grid points per degree")
    return value


def table_per_degree() -> float:
    """Grid density for response tables (nested selection, thickness updates): GROND_SHARED_GRID_TABLE_PER_DEGREE."""
    value = os.environ.get(ENV_TABLE_PER_DEGREE, "")
    if not value:
        return per_degree()
    value = float(value)
    if not math.isfinite(value) or value < 10.0:
        raise ValueError(f"{ENV_TABLE_PER_DEGREE} must be at least 10 grid points per degree")
    return value


def precision() -> str:
    value = str(os.environ.get(ENV_PRECISION, "double") or "double").strip().lower()
    if value not in {"double", "single"}:
        raise ValueError(f"{ENV_PRECISION} must be double or single")
    return value


def table_precision() -> str:
    """CPU solve precision for response tables (GROND_SHARED_GRID_TABLE_PRECISION, default as predictions)."""
    value = str(os.environ.get("GROND_SHARED_GRID_TABLE_PRECISION", "") or "").strip().lower()
    if not value:
        return precision()
    if value not in {"double", "single"}:
        raise ValueError("GROND_SHARED_GRID_TABLE_PRECISION must be double or single")
    return value


def gpu_devices() -> list[str]:
    """CUDA device ids for GPU grid workers (GROND_SHARED_GRID_GPUS, e.g. "0,1,2"); predictions only."""
    value = str(os.environ.get(ENV_GPUS, "") or "").strip()
    return [v.strip() for v in value.split(",") if v.strip()]


def gpu_precision() -> str:
    """GPU eigensolver precision (default single: cuSOLVER divide and conquer, max 4.6e-4 from float64 measured)."""
    value = str(os.environ.get(ENV_GPU_PRECISION, "single") or "single").strip().lower()
    if value not in {"double", "single"}:
        raise ValueError(f"{ENV_GPU_PRECISION} must be double or single")
    return value


def unsupported_reason(port, purpose: str = "predict") -> str | None:
    """Why the shared grid cannot reproduce this port's physics (None when it can)."""
    cfg = port.cfg
    if getattr(port.data, "pets_exact_a_center", None) is not None:
        return "PETS zone orientations"
    if str(getattr(cfg, "profile_coordinate", "target_sg") or "target_sg").lower() != "target_sg":
        return "profile_coordinate is not target_sg"
    if str(getattr(cfg, "mosaicity_mode", "none") or "none").lower() not in {"none", "transverse_target"}:
        return f"mosaicity_mode {cfg.mosaicity_mode}"
    if str(getattr(cfg, "sg_jitter_mode", "none") or "none").lower() != "none":
        return "target-Sg jitter"
    if port.absorption_enabled():
        return "absorption"
    backend = str(getattr(cfg, "compute_backend", "scipy") or "scipy").lower()
    if backend not in {"scipy", "numpy_scipy_cpu"}:
        return f"compute_backend {backend}"
    if str(getattr(cfg, "matrix_convention", "j_minus_i") or "j_minus_i") != "j_minus_i":
        return "matrix_convention"
    if purpose == "predict" and resolved_config_evaluator(cfg) != "table":
        return f"thickness evaluator {resolved_config_evaluator(cfg)} (the grid evaluates the table)"
    if purpose == "predict" and port.thickness_uniform_grid() is None:
        return "non-uniform thickness nodes"
    return None


def _tilt_count(cfg) -> int:
    if str(getattr(cfg, "mosaicity_mode", "none") or "none").lower() == "transverse_target" and \
            float(getattr(cfg, "mosaicity_sigma_deg", 0.0)) > 0.0:
        return max(1, int(getattr(cfg, "mosaicity_points", 3)))
    return 1


def estimate_costs(port, indices, workers: int = 1, purpose: str = "predict") -> dict:
    """Relative cost of the per-row engine and of the grid for these rows (the path chooser's model).

    Both scale with the number of eigensolves times N^2.5 (measured: LAPACK ?sytrd + ?stemr, N 1,000-3,000); the grid
    adds the per-target thickness evaluation, linear in N: one target costs about 1/900 of an eigensolve at
    N = 2,340 (SU-102, M3 Max and EPYC alike), i.e. 126 N in these units. Grid solves in float64 count twice. N is
    sampled at a few row centres.
    """
    cfg = port.cfg
    rows = np.asarray(indices, dtype=int)
    rng = np.random.default_rng(0)
    sample = rows if len(rows) <= 8 else rng.choice(rows, 8, replace=False)
    pool = np.asarray(port.pool_hkl, dtype=int)
    cut = float(cfg.max_support_sg)
    n_row, omegas, half = [], [], []
    for r in sample:
        oc = port.omega_from_z(port.data.zcal[int(r)])
        hkl = port.data.hkl[int(r)]
        win, _m = port.row_sg_window(int(r), hkl, oc)
        d = abs(float(port.sg_derivative_per_deg(hkl, oc)))
        half.append(0.5 * float(win) / max(d, 1e-6))
        sg = port.sg_for_hkls(pool, oc)
        n_row.append(2 + int(np.count_nonzero(np.abs(sg) <= cut)))
    cap = float(getattr(cfg, "n_basis", 0) or 0) or 1e9
    n_cap = min(float(np.median(n_row)), cap)
    points = max(1, int(cfg.n_points)) * _tilt_count(cfg)
    per_row = len(rows) * points * n_cap ** 2.5
    om_all = np.asarray([port.omega_from_z(port.data.zcal[int(r)]) for r in rows])
    density_grid = per_degree() if purpose == "predict" else table_per_degree()
    span = float(np.ptp(om_all)) + 2.0 * float(np.median(half)) + 2.0 * (MARGIN + 1) / density_grid
    density = len(rows) / max(span, 1e-6)
    active = density * (2.0 * float(np.median(half)) + 2.0 * (MARGIN + 1) / density_grid)
    n_grid = n_cap + (0.2 if n_cap >= float(np.median(n_row)) else 0.6) * active
    prec = 2.0 if precision() == "double" else 1.0
    grid_points = span * density_grid * _tilt_count(cfg)
    shared = grid_points * (prec * n_grid ** 2.5 + active * 126.0 * n_grid)
    return {"rows": int(len(rows)), "per_row_cost": float(per_row), "shared_cost": float(shared),
            "ratio": float(per_row / max(shared, 1.0)), "n_row_median": float(np.median(n_row)), "n_row_capped": n_cap,
            "n_grid_estimate": n_grid, "span_deg": span, "active_targets": active, "grid_points": grid_points}


def choose_path(port, indices, workers: int = 1, purpose: str = "predict") -> tuple[str, str]:
    """('shared_grid' or 'per_row', reason). The guesser behind GROND_SHARED_GRID=auto."""
    m = mode()
    if m == "off":
        return "per_row", f"{ENV_MODE} off"
    reason = unsupported_reason(port, purpose)
    if reason is not None:
        return "per_row", f"unsupported: {reason}"
    if m == "force":
        return "shared_grid", "forced"
    n = len(indices)
    min_rows = int(os.environ.get(ENV_MIN_ROWS, "200") or 200)
    if n < min_rows:
        return "per_row", f"{n} rows < {min_rows}"
    est = estimate_costs(port, indices, workers, purpose)
    if est["ratio"] >= 3.0:
        return "shared_grid", f"estimated {est['ratio']:.0f}x cheaper"
    return "per_row", f"estimated grid/per-row cost ratio only {est['ratio']:.1f}"


# ----------------------------------------------------------------------------- per-row light plans


def _tilt_key(value: float) -> float:
    return float(np.round(float(value), 12))


def light_plan(port, row: int, step: float, support_t=None):
    """A row's plan without its matrices: public metadata, Born values, and its cardinal weights per tilt.

    Mirrors XDSBlochPort.row_prediction_plan for everything except the basis and the Hamiltonians.
    """
    hkl = np.asarray(port.data.hkl[row], dtype=int)
    omega_center = port.omega_from_z(port.data.zcal[row])
    center_sg = float(port.sg_for_hkls(np.asarray([hkl], dtype=int), omega_center)[0])
    omega_bragg, omega_bragg_root_available = port.row_diagnostic_omega_bragg(hkl, omega_center)
    sg_window_i, sg_width_meta = port.row_sg_window(row, hkl, omega_center)
    (omegas, tilts, weights, sg_resid, jitter_meta,
     mosaicity_meta) = port.row_profile_orientation_points(row, hkl, omega_center, center_sg, sg_window_i)
    omegas = np.asarray(omegas, dtype=float).reshape(-1)
    tilts = np.asarray(tilts, dtype=float).reshape(-1, 2)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if np.any(tilts[:, 0] != 0.0):
        raise ValueError("shared grid: along-scan mosaic tilts are not supported")
    pair = np.asarray([[0, 0, 0], hkl], dtype=int)
    h10 = port.hamiltonian_static_part(pair, absorb=False, row_idx=row)[1, 0]
    deltas = np.empty(len(omegas))
    for i, (omega, (along, transverse)) in enumerate(zip(omegas, tilts, strict=True)):
        sg = port.sg_for_hkls_at_orientation(pair, omega, along, transverse)
        deltas[i] = float(np.real(sg[1] - sg[0]))
    born_vals = np.zeros(len(omegas))
    if support_t is None:
        # XDSBlochPort.born_thickness_average for every point at once (same expressions).
        cf = port.thickness_cf_parameters()
        if cf is not None:
            from grond.bloch.thickness import gamma_cf_born

            born_vals = np.asarray(gamma_cf_born(abs(h10) ** 2, deltas, cf["shape"], cf["theta"]), dtype=float)
        else:
            thick, thick_w = np.asarray(port.thick), np.asarray(port.thick_w)
            sinc = np.sinc(deltas[:, None] * thick[None, :])
            born_vals = ((2.0 * math.pi) ** 2 * (abs(h10) ** 2) * (thick[None, :] ** 2) * (sinc ** 2)) @ thick_w
    blocks = {}
    for tau in np.unique(tilts[:, 1]):
        sel = tilts[:, 1] == tau
        om, w = omegas[sel], weights[sel]
        k0 = int(math.floor(om.min() / step)) - MARGIN
        k1 = int(math.ceil(om.max() / step)) + MARGIN
        grid = step * np.arange(k0, k1 + 1)
        W = w @ CubicSpline(grid, np.eye(len(grid)), bc_type="not-a-knot")(om)
        blocks[_tilt_key(tau)] = (k0, k1, W)
    meta = {
        "support_hkls": pair.tolist(),
        "center_sg_inv_A": center_sg,
        "omega_center_deg": float(omega_center),
        "omega_bragg_deg": float(omega_bragg),
        "omega_bragg_minus_center_deg": float(omega_bragg - omega_center),
        "omega_bragg_root_available": bool(omega_bragg_root_available),
        "sg_window_inv_A": float(sg_window_i),
        "dose_delta_b_A2": float(port.row_delta_b(row)),
        "thickness_model": str(getattr(port.cfg, "thickness_model", "uniform_gradient")),
        "thickness_sigma_frac": float(getattr(port.cfg, "thickness_sigma_frac", 0.0)),
        **sg_width_meta, **jitter_meta, **mosaicity_meta,
        "profile_basis_mode": "shared_grid",
        "born": float(np.sum(weights * born_vals)) if support_t is None else 0.0,
        "max_sg_solve_residual": float(np.max(np.abs(sg_resid))) if len(sg_resid) else 0.0,
    }
    out = {"row": int(row), "hkl": hkl, "meta": meta, "blocks": blocks, "n_p": int(len(omegas))}
    if support_t is not None:
        sinc = np.sinc(deltas[:, None] * support_t[None, :])
        out["born_table"] = weights @ ((2.0 * np.pi) ** 2 * float(abs(h10) ** 2) * (support_t[None, :] ** 2) * sinc ** 2)
    return out


# ----------------------------------------------------------------------------- one grid orientation


def _static_matrix(port, basis, delta_b, dtype):
    """U(h_j - h_i) / 2k with a zero diagonal, dose Delta B at the grid angle; the cube gather avoids an (N, N, 3) array."""
    h, k, l = (basis[:, i].astype(np.int64) for i in range(3))
    dh, dk, dl = h[None, :] - h[:, None], k[None, :] - k[:, None], l[None, :] - l[:, None]
    cube = getattr(port, "ug_cube", None)
    if cube is not None:
        hl, kl, ll = int(port.h_lim), int(port.k_lim), int(port.l_lim)
        ok = (np.abs(dh) <= hl) & (np.abs(dk) <= kl) & (np.abs(dl) <= ll)
        u = np.empty(dh.shape, dtype=np.complex128)
        u[ok] = cube[dh[ok] + hl, dk[ok] + kl, dl[ok] + ll]
        if not np.all(ok):
            u[~ok] = port.ug_array(np.stack([dh[~ok], dk[~ok], dl[~ok]], axis=-1))
    else:
        u = port.ug_array(np.stack([dh, dk, dl], axis=-1))
    if delta_b != 0.0:
        diffs = np.stack([dh, dk, dl], axis=-1).reshape(-1, 3)
        u = u * np.exp(-0.25 * delta_b * port.inv_d2_for_hkls(diffs).reshape(u.shape))
    np.fill_diagonal(u, 0.0)
    return (u / (2.0 * port.data.k)).astype(dtype)


def _grid_delta_b(port, omega: float) -> float:
    mode_ = str(getattr(port.cfg, "dose_b_mode", "none") or "none").lower()
    rate = float(getattr(port.cfg, "dose_b_per_frame", 0.0))
    if mode_ == "none" or rate == 0.0:
        return 0.0
    z = port.nominal_z_from_corrected_omega(float(omega))
    return rate * max(0.0, z - port.dose_ref_frame)


def grid_basis(port, omega: float, tau: float, targets, cut: float):
    """(basis, target positions, Sg of the basis) of the grid Hamiltonian at one orientation."""
    order = {(0, 0, 0): 0}
    for hkl in targets:
        order.setdefault((int(hkl[0]), int(hkl[1]), int(hkl[2])), len(order))
    pool = _G.get("pool")
    if pool is None:
        pool = np.asarray(port.pool_hkl, dtype=int)
    sg_pool = port.sg_for_hkls_at_orientation(pool, omega, 0.0, tau)
    candidates = np.flatnonzero(np.abs(sg_pool) <= cut)
    cap = int(_G.get("n_basis", getattr(port.cfg, "n_basis", 0)) or 0)
    if cap and len(candidates) > max(0, cap - 2):
        pool_ug = _G.get("pool_ug")
        if pool_ug is None:
            pool_ug = np.asarray(port.pool_ug)
        delta_b = _grid_delta_b(port, omega)
        coupling = np.abs(pool_ug[candidates]) ** 2
        if delta_b != 0.0:
            coupling = coupling * np.exp(-0.5 * delta_b * np.asarray(port.pool_inv_d2)[candidates])
        score = coupling / (np.abs(sg_pool[candidates]) + 1e-5)
        candidates = candidates[np.lexsort((candidates, -score))[: max(0, cap - 2)]]
    for i in candidates:
        order.setdefault((int(pool[i, 0]), int(pool[i, 1]), int(pool[i, 2])), len(order))
    basis = np.asarray(list(order), dtype=np.int64)
    pos = np.asarray([order[(int(h[0]), int(h[1]), int(h[2]))] for h in targets], dtype=int)
    sg = np.asarray(port.sg_for_hkls_at_orientation(basis, omega, 0.0, tau), dtype=float)
    return basis, pos, sg


def solve_grid_point(port, omega: float, tau: float, targets, cut: float, prec: str = "double"):
    """(eigenvalues, couplings[target, j] = V[t, j] conj(V[0, j]), basis size) at one grid orientation."""
    order = {(0, 0, 0): 0}
    for hkl in targets:
        order.setdefault((int(hkl[0]), int(hkl[1]), int(hkl[2])), len(order))
    pool = _G.get("pool")
    if pool is None:
        pool = np.asarray(port.pool_hkl, dtype=int)
    sg_pool = port.sg_for_hkls_at_orientation(pool, omega, 0.0, tau)
    candidates = np.flatnonzero(np.abs(sg_pool) <= cut)
    cap = int(_G.get("n_basis", getattr(port.cfg, "n_basis", 0)) or 0)
    if cap and len(candidates) > max(0, cap - 2):
        # Production's top-up rule (select_basis) at the grid angle: the cap - 2 best |U_g|^2 / (|Sg| + 1e-5), ties by
        # pool order, on top of the direct beam and the targets (each target sees the same number of top-up beams).
        pool_ug = _G.get("pool_ug")
        if pool_ug is None:
            pool_ug = np.asarray(port.pool_ug)
        delta_b = _grid_delta_b(port, omega)
        coupling = np.abs(pool_ug[candidates]) ** 2
        if delta_b != 0.0:
            coupling = coupling * np.exp(-0.5 * delta_b * np.asarray(port.pool_inv_d2)[candidates])
        score = coupling / (np.abs(sg_pool[candidates]) + 1e-5)
        candidates = candidates[np.lexsort((candidates, -score))[: max(0, cap - 2)]]
    for i in candidates:
        order.setdefault((int(pool[i, 0]), int(pool[i, 1]), int(pool[i, 2])), len(order))
    basis = np.asarray(list(order), dtype=np.int64)
    n = len(basis)
    complex_dtype = np.complex128 if prec == "double" else np.complex64
    H = _static_matrix(port, basis, _grid_delta_b(port, omega), complex_dtype)
    sg = np.asarray(port.sg_for_hkls_at_orientation(basis, omega, 0.0, tau), dtype=float)
    real_dtype = np.float64 if prec == "double" else np.float32
    if np.max(np.abs(H.imag)) <= 8.0 * np.finfo(real_dtype).eps * max(float(np.max(np.abs(H.real))), 1e-300):
        A = np.ascontiguousarray(H.real)
        A[np.diag_indices(n)] += sg.astype(real_dtype)
        prefix = "d" if prec == "double" else "s"
        reduce, apply, trans = prefix + "sytrd", prefix + "ormqr", "T"
    else:
        A = np.ascontiguousarray(H)
        A[np.diag_indices(n)] += sg.astype(real_dtype)
        prefix = "z" if prec == "double" else "c"
        reduce, apply, trans = prefix + "hetrd", prefix + "unmqr", "C"
    lwork = max(1, int(np.real(getattr(lapack, reduce + "_lwork")(n, lower=1)[0])))
    a, d, e, tau_h, info = getattr(lapack, reduce)(A, lower=1, lwork=lwork)
    if info != 0:
        raise np.linalg.LinAlgError(f"{reduce} failed (info={info})")
    lam, Z = eigh_tridiagonal(d, e, lapack_driver="stemr")
    pos = np.asarray([order[(int(h[0]), int(h[1]), int(h[2]))] for h in targets], dtype=int)
    m = len(pos)
    unit = np.zeros((n - 1, m), dtype=a.dtype, order="F")
    unit[pos - 1, np.arange(m)] = 1.0
    q, _work, info = getattr(lapack, apply)("L", trans, a[1:, : n - 1], tau_h[: n - 1], unit, lwork=max(1, n * m))
    if info != 0:
        raise np.linalg.LinAlgError(f"{apply} failed (info={info})")
    rows_q = np.zeros((m, n), dtype=a.dtype)
    rows_q[:, 1:] = np.conj(q.T)
    V = rows_q @ Z.astype(rows_q.dtype if np.iscomplexobj(rows_q) else Z.dtype)
    C = V * np.conj(Z[0, :])[None, :]
    return np.asarray(lam, dtype=float), np.asarray(C, dtype=np.complex128), n


def target_intensities(lam, C, t_spec, hands, weights=None):
    """|sum_j C[a, j] exp(2 pi i s lam_j t)|^2 on a uniform t grid, per hand s; summed with weights when given."""
    t0, dt, nt = t_spec
    within, shift = uniform_grid_phase_factors(lam[None, :], t0, dt, nt)
    within, shift = within[0], shift[0]
    out = {}
    for h in hands:
        wv, sv = (within, shift) if h > 0 else (np.conj(within), np.conj(shift))
        res = np.empty((C.shape[0],) if weights is not None else (C.shape[0], nt))
        for s in range(0, C.shape[0], TARGET_CHUNK):
            c = C[s:s + TARGET_CHUNK]
            amp = np.matmul(c[:, None, :] * sv[None, :, :], wv).reshape(len(c), -1)[:, :nt]
            inten = np.abs(amp) ** 2
            res[s:s + TARGET_CHUNK] = inten @ weights if weights is not None else inten
        out[h] = res
    return out


# ----------------------------------------------------------------------------- the pass


def _init_light(port, step, support_t):
    _G.clear()
    _G.update(port=port, step=step, support_t=support_t)


def _light_worker(rows):
    return [light_plan(_G["port"], int(r), _G["step"], _G["support_t"]) for r in rows]


def _block_worker(task):
    """Solve grid points k0..k1 of one tilt; return per-point intensities (predict) or partial row tables (table)."""
    tilt_index, k_lo, k_hi = task
    g = _G
    port, tau, step = g["port"], g["taus"][tilt_index], g["step"]
    k0s, k1s, Ws = g["k0"][tilt_index], g["k1"][tilt_index], g["W"][tilt_index]
    hkls, hands, kind = g["hkls"], g["hands"], g["kind"]
    partial = {}
    points = []
    cpu = {"solve": 0.0, "thickness": 0.0}
    for k in range(k_lo, k_hi + 1):
        act = np.flatnonzero((k0s <= k) & (k1s >= k))
        if not len(act):
            continue
        t0 = time.process_time()
        lam, C, n = solve_grid_point(port, k * step, tau, hkls[act], g["cut"], g["precision"])
        t1 = time.process_time()
        if kind == "predict":
            inten = target_intensities(lam, C, g["t_spec"], hands, weights=g["t_weights"])
            points.append((k, act, {h: inten[h] for h in hands}, n))
        else:
            inten = target_intensities(lam, C, g["t_spec"], hands)
            for i, p in enumerate(act):
                wk = float(Ws[p][k - k0s[p]])
                acc = partial.get(int(p))
                if acc is None:
                    acc = partial[int(p)] = np.zeros((len(hands), g["t_spec"][2]))
                for j, h in enumerate(hands):
                    acc[j] += wk * inten[h][i]
            points.append((k, None, None, n))
        cpu["solve"] += t1 - t0
        cpu["thickness"] += time.process_time() - t1
    return tilt_index, points, partial, cpu


class _GpuSolver:
    """One CUDA device: Hamiltonian from the U_g cube, eigh (sizes padded to GPU_BUCKET with decoupled beams),
    target couplings and the thickness-weighted intensities, all on the device."""

    def __init__(self, port, prec):
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        self.jax, self.jnp, self.prec = jax, jnp, prec
        self.rdt = jnp.float32 if prec == "single" else jnp.float64
        self.cdt = jnp.complex64 if prec == "single" else jnp.complex128
        cube = np.asarray(port.ug_cube)
        self.real = bool(np.max(np.abs(cube.imag)) <= 1e-12 * max(float(np.max(np.abs(cube.real))), 1e-300))
        scale = 1.0 / (2.0 * port.data.k)
        self.cube = jnp.asarray((cube.real if self.real else cube) * scale, dtype=self.rdt if self.real else self.cdt)
        self.lims = (int(port.h_lim), int(port.k_lim), int(port.l_lim))
        self.shape = cube.shape
        self.t_spec = port.thickness_uniform_grid()
        self.w = jnp.asarray(np.asarray(port.thick_w), dtype=jnp.float64)
        self.eigh = jax.jit(jnp.linalg.eigh)
        self.A = jnp.asarray(np.asarray(port.data.A, dtype=float), dtype=jnp.float64)
        self.port = port

    def intensities(self, basis, pos, sg, hands, delta_b=0.0):
        jnp = self.jnp
        n = len(basis)
        size = int(math.ceil(n / GPU_BUCKET) * GPU_BUCKET)
        hl, kl, ll = self.lims
        b = jnp.asarray(basis, dtype=jnp.int32)
        dh = b[None, :, 0] - b[:, None, 0]
        dk = b[None, :, 1] - b[:, None, 1]
        dl = b[None, :, 2] - b[:, None, 2]
        ok = (jnp.abs(dh) <= hl) & (jnp.abs(dk) <= kl) & (jnp.abs(dl) <= ll)
        if not bool(jnp.all(ok)):
            raise ValueError("GPU grid: a beam difference lies outside the U_g cube")
        H = self.cube[dh + hl, dk + kl, dl + ll]
        if delta_b != 0.0:
            # Dose Debye-Waller factor of each difference at the grid angle, as _static_matrix (1/d^2 = |A h|^2).
            g = jnp.stack([dh, dk, dl], axis=-1).astype(jnp.float64) @ self.A.T
            H = H * jnp.exp(-0.25 * float(delta_b) * jnp.sum(g * g, axis=-1)).astype(H.real.dtype)
        H = H.at[jnp.diag_indices(n)].set(jnp.asarray(sg, dtype=H.dtype))
        if size > n:
            pad = jnp.zeros((size, size), dtype=H.dtype)
            pad = pad.at[:n, :n].set(H)
            # Decoupled padding beams with distinct diagonals just above the physical range: an eigensolver's accuracy is
            # relative to the matrix norm, so large padding values would cost float32 its precision at the scale of
            # the couplings (measured: 1e3 pads gave 35% median errors in float32; these keep the norm unchanged).
            top = float(np.max(np.abs(sg))) + 0.01  # decoupled, so coinciding with a physical eigenvalue is harmless
            fill = top * (1.0 + np.arange(size - n) / float(size))
            pad = pad.at[jnp.arange(n, size), jnp.arange(n, size)].set(jnp.asarray(fill, dtype=pad.dtype))
            H = pad
        lam, V = self.eigh(H)
        cdt = jnp.complex128 if self.prec == "double" else jnp.complex64
        C = (V[jnp.asarray(pos)] * jnp.conj(V[0])[None, :]).astype(cdt)
        # Production's factorised phases (uniform_grid_phase_factors), built in float64 on the host from the
        # eigenvalues, then one batched product on the device: amp[a, b*B + r] = sum_j C[a, j] shift[b, j] within[j, r].
        lam_h = np.asarray(lam, dtype=np.float64)
        t0, dt, nt = self.t_spec
        within, shift = uniform_grid_phase_factors(lam_h[None, :], t0, dt, nt)
        within, shift = jnp.asarray(within[0], dtype=cdt), jnp.asarray(shift[0], dtype=cdt)
        out = {}
        for h in hands:
            wv, sv = (within, shift) if h > 0 else (jnp.conj(within), jnp.conj(shift))
            parts = []
            for s0 in range(0, C.shape[0], 64):  # bounded (64, blocks, N) intermediates per device call
                amp = ((C[s0:s0 + 64, None, :] * sv[None, :, :]) @ wv).reshape(-1, sv.shape[0] * wv.shape[1])[:, :nt]
                parts.append((jnp.abs(amp) ** 2).astype(jnp.float64) @ self.w)
            out[h] = np.asarray(jnp.concatenate(parts))
        return out, n


def _queue_worker(kind, device, task_q, result_q):
    """Pull grid blocks until the sentinel; device None = CPU (float64 path), else one GPU."""
    try:
        if device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            gpu = _GpuSolver(_G["port"], gpu_precision())
        while True:
            task = task_q.get()
            if task is None:
                break
            if device is None:
                result_q.put(("ok", _block_worker(task)))
                continue
            tilt_index, k_lo, k_hi = task
            g = _G
            port, tau, step = g["port"], g["taus"][tilt_index], g["step"]
            k0s, k1s = g["k0"][tilt_index], g["k1"][tilt_index]
            points, cpu = [], {"solve": 0.0, "thickness": 0.0}
            for k in range(k_lo, k_hi + 1):
                act = np.flatnonzero((k0s <= k) & (k1s >= k))
                if not len(act):
                    continue
                t0 = time.time()
                basis, pos, sg = grid_basis(port, k * step, tau, g["hkls"][act], g["cut"])
                inten, n = gpu.intensities(basis, pos, sg, g["hands"], _grid_delta_b(port, k * step))
                points.append((k, act, inten, n))
                cpu["solve"] += time.time() - t0
            result_q.put(("ok", (tilt_index, points, {}, cpu)))
    except BaseException as exc:  # report and stop; the parent raises
        import traceback

        result_q.put(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"))


def _run_queue(ctx, tasks, n_cpu, devices, kind, absorb):
    task_q, result_q = ctx.Queue(), ctx.Queue()
    for task in tasks:
        task_q.put(task)
    procs = [ctx.Process(target=_queue_worker, args=(kind, d, task_q, result_q)) for d in devices]
    procs += [ctx.Process(target=_queue_worker, args=(kind, None, task_q, result_q)) for _ in range(n_cpu)]
    for _ in procs:
        task_q.put(None)
    for p in procs:
        p.start()
    try:
        for _ in range(len(tasks)):
            status, payload = result_q.get()
            if status != "ok":
                raise RuntimeError(f"shared grid worker failed: {payload}")
            absorb(payload)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()


def _context():
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return None


def _run(port, rows, workers, kind, hands, support=None, log=print):
    """The shared pass over ``rows``; returns (light plans, per-row values or tables, diagnostics)."""
    t_start = time.time()
    step = 1.0 / (per_degree() if kind == "predict" else table_per_degree())
    prec = precision() if kind == "predict" else table_precision()
    support_t = None if support is None else support.grid()
    workers = max(1, int(workers))
    ctx = _context()
    rows = [int(r) for r in rows]
    _init_light(port, step, support_t)
    size = max(1, min(64, len(rows) // (4 * workers) or 1))
    chunks = [rows[i:i + size] for i in range(0, len(rows), size)]
    if ctx is not None and workers > 1:
        with ctx.Pool(min(workers, len(chunks))) as pool:
            plans = [p for part in pool.map(_light_worker, chunks) for p in part]
    else:
        plans = [p for part in map(_light_worker, chunks) for p in part]
    t_light = time.time() - t_start
    taus = sorted({tau for p in plans for tau in p["blocks"]})
    k0 = [np.asarray([p["blocks"][tau][0] if tau in p["blocks"] else 1 for p in plans], dtype=np.int64) for tau in taus]
    k1 = [np.asarray([p["blocks"][tau][1] if tau in p["blocks"] else 0 for p in plans], dtype=np.int64) for tau in taus]
    W = [[p["blocks"][tau][2] if tau in p["blocks"] else None for p in plans] for tau in taus]
    hkls = np.asarray([p["hkl"] for p in plans], dtype=np.int64)
    if kind == "predict":
        t_spec = port.thickness_uniform_grid()
        t_weights = np.asarray(port.thick_w, dtype=float)
    else:
        t_spec = uniform_grid_spec(support_t)
        t_weights = None
    total_points = sum(int(k1[i].max() - k0[i][k0[i] <= k1[i]].min() + 1) for i in range(len(taus)))
    gpu_workers = len(gpu_devices()) * max(1, int(os.environ.get("GROND_SHARED_GRID_WORKERS_PER_GPU", "4") or 4))
    n_tasks = max(16 * gpu_workers if gpu_devices() and kind == "predict" else 4 * workers, 1)
    span = max(1, total_points // n_tasks)
    tasks = []
    for i in range(len(taus)):
        valid = k0[i] <= k1[i]
        lo, hi = int(k0[i][valid].min()), int(k1[i][valid].max())
        for a in range(lo, hi + 1, span):
            tasks.append((i, a, min(hi, a + span - 1)))
    _G.update(taus=taus, k0=k0, k1=k1, W=W, hkls=hkls, hands=tuple(hands), kind=kind, t_spec=t_spec,
              t_weights=t_weights, cut=float(port.cfg.max_support_sg), precision=prec,
              pool=np.asarray(port.pool_hkl, dtype=np.int64), pool_ug=np.asarray(port.pool_ug),
              n_basis=int(getattr(port.cfg, "n_basis", 0) or 0))
    log(f"shared grid: {len(rows)} rows, {len(taus)} tilt(s), {total_points} grid points at {1.0 / step:g}/deg, "
        f"|Sg| <= {port.cfg.max_support_sg:g}, n_basis cap {port.cfg.n_basis}, {prec} precision, {len(tasks)} tasks "
        f"on {workers} workers "
        f"(light plans {t_light:.0f} s)")
    values = {h: np.zeros(len(plans)) for h in hands} if kind == "predict" else None
    tables = {h: np.zeros((len(plans), t_spec[2])) for h in hands} if kind == "table" else None
    n_at_centre = np.zeros(len(plans), dtype=int)
    centre_k = np.rint(np.asarray([p["meta"]["omega_center_deg"] for p in plans]) / step).astype(np.int64)
    sizes, cpu = [], {"solve": 0.0, "thickness": 0.0}
    t_pass = time.time()
    done = 0

    def absorb(result):
        nonlocal done
        tilt_index, points, partial, c = result
        for key in cpu:
            cpu[key] += c[key]
        for k, act, inten, n in points:
            sizes.append(n)
            if taus[tilt_index] == 0.0 or len(taus) == 1:
                n_at_centre[centre_k == k] = n
            if act is not None:
                for j, p in enumerate(act):
                    wk = float(W[tilt_index][p][k - k0[tilt_index][p]])
                    for h in hands:
                        values[h][p] += wk * float(inten[h][j])
        for p, acc in partial.items():
            for j, h in enumerate(hands):
                tables[h][p] += acc[j]
        done += 1
        if done % max(1, len(tasks) // 10) == 0:
            log(f"  shared grid: {done}/{len(tasks)} tasks, {time.time() - t_pass:.0f} s")

    devices = gpu_devices() if kind == "predict" and getattr(port, "ug_cube", None) is not None else []
    if ctx is not None and devices:
        workers_per_gpu = max(1, int(os.environ.get("GROND_SHARED_GRID_WORKERS_PER_GPU", "4") or 4))
        dev_list = [d for d in devices for _ in range(workers_per_gpu)]
        # CPU workers next to the GPUs only on request: at large N they are memory-bandwidth bound and slowed the
        # GPU workers' host work as much as they added (SU-102, |Sg| <= 0.04: 743 s mixed against 760 s GPU only).
        with_cpu = str(os.environ.get("GROND_SHARED_GRID_CPU_WITH_GPU", "0")).strip().lower() in {"1", "true", "on", "yes"}
        n_cpu = max(0, workers - len(dev_list)) if with_cpu else 0
        log(f"  shared grid: {len(dev_list)} GPU worker(s) on device(s) {','.join(devices)} ({gpu_precision()}), "
            f"{n_cpu} CPU worker(s) ({prec})")
        _run_queue(ctx, tasks, n_cpu, dev_list, kind, absorb)
    elif ctx is not None and workers > 1:
        with ctx.Pool(min(workers, len(tasks))) as pool:
            for result in pool.imap_unordered(_block_worker, tasks, chunksize=1):
                absorb(result)
    else:
        for task in tasks:
            absorb(_block_worker(task))
    diag = {"rows": len(rows), "tilts": len(taus), "grid_points": int(len(sizes)), "per_degree": 1.0 / step,
            "precision": prec, "cut": float(port.cfg.max_support_sg), "workers": workers,
            "N_median": float(np.median(sizes)) if sizes else 0.0, "N_max": int(max(sizes)) if sizes else 0,
            "cpu_solve_s": cpu["solve"], "cpu_thickness_s": cpu["thickness"], "light_plan_wall_s": t_light,
            "pass_wall_s": time.time() - t_pass, "wall_s": time.time() - t_start}
    log(f"shared grid done: {diag['grid_points']} solves, N median {diag['N_median']:.0f} max {diag['N_max']}, "
        f"CPU solve {cpu['solve']:.0f} s + thickness {cpu['thickness']:.0f} s, {diag['wall_s']:.0f} s wall")
    _G.clear()
    return plans, values, tables, n_at_centre, diag


def predict_rows(port, indices, workers=1, log=print):
    """Per-row results in the order of ``indices`` (the dicts row_prediction returns), from the shared grid."""
    hands = (1, -1) if bool(getattr(port.cfg, "paired_inverse_hand", False)) else (1,)
    plans, values, _t, n_c, diag = _run(port, indices, workers, "predict", hands, log=log)
    results = []
    for i, p in enumerate(plans):
        r = dict(p["meta"])
        r["dyn"] = float(values[1][i])
        if -1 in hands:
            r["dyn_inverse"] = float(values[-1][i])
        r["basis_size"] = int(n_c[i])
        results.append(r)
    port.shared_grid_diagnostics = diag
    return results


def response_table(port, rows, support, hands=(1,), workers=1, log=print):
    """Response-table arrays {hand: I(t), 'born', 'row_ids', 'mu', 'n_p', 'n_b', 'reused_points'} from the grid."""
    hands = tuple(dict.fromkeys(int(h) for h in hands))
    plans, _v, tables, n_c, diag = _run(port, rows, workers, "table", hands, support=support, log=log)
    out = {h: tables[h] for h in hands}
    out["born"] = np.asarray([p["born_table"] for p in plans])
    out["row_ids"] = np.asarray([p["row"] for p in plans], dtype=int)
    out["mu"] = np.zeros(len(plans))
    out["n_p"] = np.asarray([p["n_p"] for p in plans], dtype=int)
    out["n_b"] = np.asarray(n_c, dtype=int)
    out["reused_points"] = np.zeros(len(plans), dtype=int)
    port.shared_grid_diagnostics = diag
    return out, diag
