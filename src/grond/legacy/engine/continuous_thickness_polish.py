#!/usr/bin/env python3
"""Continuous crystal-thickness regularization for GROND.

This script tests the key implementation idea needed to move beyond a coarse
thickness grid: for a fixed structure, geometry, beam basis, and absorption
model, each Bloch Hamiltonian is independent of crystal thickness.  We can
therefore build and eigensolve each row/profile Hamiltonian once, then re-score
many continuous thickness and thickness-gradient values by changing only the
propagation phases.

The implementation deliberately reuses GROND's native nuisance objective and
scale fitting.  It is run after a cheap grid screen: the grid finds a defensible
bracket, and this script turns that bracket into a continuous thickness value
without repeating all of the eigensolver work for each trial point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eig, eigh
from scipy.optimize import minimize, minimize_scalar


ENGINE_ROOT = Path(__file__).resolve().parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from bloch_engine import Config, XDSBlochPort, XDSData, parse_model, sha256, split_masks  # noqa: E402
import native_row_regularize as nrf  # noqa: E402


DEFAULT_HKL = Path("INTEGRATE.HKL")
DEFAULT_RES = Path("model.res")


def thickness_grid(t0: float, grad: float, points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return GROND's uniform-gradient thickness quadrature for arbitrary t0."""

    t0 = float(t0)
    grad = float(grad)
    points = max(1, int(points))
    if points <= 1 or abs(grad) <= 0.0:
        return np.asarray([t0], dtype=float), np.asarray([1.0], dtype=float)
    x, w = leggauss(points)
    t = t0 * (1.0 + grad * x)
    ok = t > 0.0
    if not np.any(ok):
        return np.asarray([t0], dtype=float), np.asarray([1.0], dtype=float)
    t = np.asarray(t[ok], dtype=float)
    w = np.asarray(w[ok], dtype=float)
    return t, w / np.sum(w)


def build_base_config(args) -> Config:
    """Create a GROND Bloch config whose non-thickness terms define the contract."""

    return Config(
        n_basis=int(args.n_basis),
        n_points=int(args.n_points),
        sg_window=float(args.sg_window),
        sg_window_mode=str(args.sg_window_mode),
        sg_window_min=float(args.sg_window_min),
        sg_window_max=float(args.sg_window_max),
        sg_window_scale=float(args.sg_window_scale),
        sg_window_frame_floor=float(args.sg_window_frame_floor),
        profile_quadrature=str(args.profile_quadrature),
        profile_accumulation=str(args.profile_accumulation),
        topup_d_min=float(args.topup_d_min),
        max_support_sg=float(args.max_support_sg),
        thickness_A=float(args.start_thickness),
        thickness_model="uniform_gradient",
        thickness_grad_frac=float(args.start_grad),
        thickness_points=int(args.thickness_points),
        group_width_deg=float(args.group_width_deg),
        frame_offset=float(args.frame_offset),
        matrix_convention=str(args.matrix_convention),
        free_fraction=float(args.free_fraction),
        split_seed=int(args.split_seed),
    )


def selected_indices(data: XDSData, target: np.ndarray, args) -> np.ndarray:
    """Choose either all eligible rows or the strongest eligible subset."""

    idx = nrf.row_selection(
        data,
        target,
        positive_only=not bool(args.include_nonpositive),
        min_d=float(args.min_d_spacing),
        min_peak=float(args.min_peak),
        max_rows=0,
        geometry_percentile=float(args.geometry_percentile),
    )
    if int(args.strong_rows) > 0 and len(idx) > int(args.strong_rows):
        sig = np.asarray(data.sigma, dtype=float)
        score = np.asarray(target, dtype=float) / np.maximum(sig, 1e-12)
        order = np.argsort(score[idx])[::-1]
        idx = idx[order[: int(args.strong_rows)]]
        idx = np.sort(idx)
    if int(args.max_rows) > 0 and len(idx) > int(args.max_rows):
        idx = idx[: int(args.max_rows)]
    return np.asarray(idx, dtype=int)


def eig_coupling(H: np.ndarray, absorptive: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and direct-to-target coupling for one Hamiltonian."""

    if absorptive:
        vals, vecs = eig(H)
        try:
            inv_vecs = np.linalg.inv(vecs)
        except np.linalg.LinAlgError:
            inv_vecs = np.linalg.pinv(vecs)
        coupling = vecs[1, :] * inv_vecs[:, 0]
    else:
        vals, vecs = eigh(H)
        coupling = vecs[1, :] * np.conj(vecs[0, :])
    return np.asarray(vals, dtype=np.complex128), np.asarray(coupling, dtype=np.complex128)


def build_response_cache(
    data: XDSData,
    model,
    cfg: Config,
    indices: np.ndarray,
    res_hash: str,
    cache_dir: Path,
) -> list[dict]:
    """Build row/profile eigensystems that can be re-scored for any thickness."""

    port = XDSBlochPort(data, model, cfg, cache_dir=cache_dir, res_hash=res_hash)
    absorptive = port.absorption_enabled()
    records: list[dict] = []
    t0 = time.perf_counter()
    for n, row_idx in enumerate(indices, start=1):
        hkl = data.hkl[int(row_idx)]
        omega_center = port.omega_from_z(data.zcal[int(row_idx)])
        center_sg = float(port.sg_for_hkls(np.asarray([hkl], dtype=int), omega_center)[0])
        omega_bragg = port.solve_omega_for_sg(hkl, omega_center, 0.0)
        basis = port.select_basis(hkl, omega_bragg, row_idx=int(row_idx))
        sg_window_i, sg_meta = port.row_sg_window(int(row_idx), hkl, omega_bragg)
        requested, profile_weights, jitter_meta = port.row_profile_sg_points(
            int(row_idx), hkl, sg_window_i
        )

        dyn_specs = []
        born_specs = []
        sg_resid = []
        for target_sg in requested:
            omega = port.solve_omega_for_sg(hkl, omega_bragg, target_sg)
            sg_basis = port.sg_for_hkls(basis, omega)
            sg_resid.append(float(sg_basis[1] - target_sg))
            H_dyn = port.hamiltonian(basis, sg_basis, absorb=True, row_idx=int(row_idx))
            H_born = port.hamiltonian(basis, sg_basis, absorb=False, row_idx=int(row_idx))
            vals, coupling = eig_coupling(H_dyn, absorptive=absorptive)
            dyn_specs.append({"vals": vals, "coupling": coupling})
            h10 = H_born[1, 0]
            delta = float(np.real(H_born[1, 1] - H_born[0, 0]))
            born_specs.append({"h10_abs2": float(abs(h10) ** 2), "delta": delta})

        records.append(
            {
                "row_index": int(row_idx),
                "profile_weights": np.asarray(profile_weights, dtype=float),
                "dyn_specs": dyn_specs,
                "born_specs": born_specs,
                "basis_size": int(len(basis)),
                "center_sg_inv_A": center_sg,
                "omega_bragg_deg": float(omega_bragg),
                "sg_window_inv_A": float(sg_window_i),
                "max_sg_solve_residual": float(np.max(np.abs(sg_resid))) if sg_resid else 0.0,
                **sg_meta,
                **jitter_meta,
            }
        )
        if n % int(args_progress_every) == 0 or n == len(indices):
            elapsed = time.perf_counter() - t0
            rate = n / max(elapsed, 1e-12)
            eta = (len(indices) - n) / max(rate, 1e-12)
            print(
                f"response cache {n}/{len(indices)} rows | {rate:.2f} rows/s | ETA {eta/60:.1f}m",
                flush=True,
            )
    return records


def dyn_from_spec(vals: np.ndarray, coupling: np.ndarray, thick: np.ndarray, weights: np.ndarray) -> float:
    phase = np.exp((2j * np.pi) * vals[:, None] * thick[None, :])
    per_t = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2
    return float(np.sum(weights * per_t))


def born_from_spec(h10_abs2: float, delta: float, thick: np.ndarray, weights: np.ndarray) -> float:
    arg = np.pi * float(delta) * thick
    sinc = np.ones_like(arg, dtype=float)
    nz = np.abs(arg) >= 1e-12
    sinc[nz] = np.sin(arg[nz]) / arg[nz]
    per_t = (2.0 * math.pi) ** 2 * float(h10_abs2) * (thick**2) * (sinc**2)
    return float(np.sum(weights * per_t))


def predict_from_response(records: list[dict], n_rows: int, t0: float, grad: float, points: int) -> dict:
    """Return a native_row_regularize prediction dictionary for arbitrary thickness."""

    thick, thick_w = thickness_grid(float(t0), float(grad), int(points))
    indices = np.asarray([r["row_index"] for r in records], dtype=int)
    dyn = np.zeros(len(records), dtype=float)
    born = np.zeros(len(records), dtype=float)
    basis = np.zeros(len(records), dtype=int)
    for i, rec in enumerate(records):
        pw = np.asarray(rec["profile_weights"], dtype=float)
        dyn_points = [
            dyn_from_spec(s["vals"], s["coupling"], thick, thick_w)
            for s in rec["dyn_specs"]
        ]
        born_points = [
            born_from_spec(s["h10_abs2"], s["delta"], thick, thick_w)
            for s in rec["born_specs"]
        ]
        dyn[i] = float(np.sum(pw * np.asarray(dyn_points, dtype=float)))
        born[i] = float(np.sum(pw * np.asarray(born_points, dtype=float)))
        basis[i] = int(rec["basis_size"])
    return {
        "indices": indices,
        "dyn": dyn,
        "born": born,
        "basis_size": basis,
        "response_rows_total": int(n_rows),
    }


def make_nuisance(args, t0: float, grad: float) -> nrf.Nuisance:
    return nrf.Nuisance(
        thickness_A=float(t0),
        thickness_grad_frac=float(grad),
        frame_offset=float(args.frame_offset),
        group_width_deg=float(args.group_width_deg),
        detector_scale_bins=int(args.detector_scale_bins),
        sigma_a=float(args.sigma_a),
        sigma_b_corr=float(args.sigma_b_corr),
        sigma_c_peak=float(args.sigma_c_peak),
        thickness_model="uniform_gradient",
        thickness_sigma_frac=0.0,
        matrix_convention=str(args.matrix_convention),
        row_resolution_b=float(args.row_resolution_b),
        native_wght_a=float("nan"),
        native_wght_b=float("nan"),
        scale_smoothness=float(args.scale_smoothness),
    )


def metric_value(ev: dict, metric: str) -> float:
    if metric == "all_wR2":
        return float(ev["metrics"]["all"]["dyn"]["wR2_all"])
    if metric == "free_wR2":
        return float(ev["metrics"]["free"]["dyn"]["wR2_all"])
    if metric == "all_R1":
        return float(ev["metrics"]["all"]["dyn"]["R1_all"])
    if metric == "all_R1gt":
        return float(ev["metrics"]["all"]["dyn"]["R1_gt"])
    raise ValueError(f"unknown metric {metric!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--out-dir", default="outputs/continuous_thickness_polish")
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument("--sg-window-mode", default="fixed")
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument("--sg-window-frame-floor", type=float, default=0.5)
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum")
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument("--max-support-sg", type=float, default=0.005)
    ap.add_argument("--min-d-spacing", type=float, default=0.7)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--geometry-percentile", type=float, default=100.0)
    ap.add_argument("--target-mode", choices=list(nrf.TARGET_MODES), default="iobs")
    ap.add_argument("--include-nonpositive", action="store_true")
    ap.add_argument("--start-thickness", type=float, default=1200.0)
    ap.add_argument("--start-grad", type=float, default=1.0)
    ap.add_argument("--thickness-points", type=int, default=5)
    ap.add_argument("--frame-offset", type=float, default=0.2)
    ap.add_argument("--group-width-deg", type=float, default=0.168968)
    ap.add_argument("--detector-scale-bins", type=int, default=8)
    ap.add_argument("--sigma-a", type=float, default=0.0)
    ap.add_argument("--sigma-b-corr", type=float, default=0.0)
    ap.add_argument("--sigma-c-peak", type=float, default=0.15)
    ap.add_argument("--row-resolution-b", type=float, default=0.0)
    ap.add_argument("--scale-smoothness", type=float, default=0.0)
    ap.add_argument("--matrix-convention", default="j_minus_i")
    ap.add_argument("--free-fraction", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--scale-fit-split", choices=["all", "work", "free"], default="all")
    ap.add_argument("--selection-metric", choices=["all_wR2", "free_wR2", "all_R1", "all_R1gt"], default="all_wR2")
    ap.add_argument("--strong-rows", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--bracket-frac", type=float, default=0.35)
    ap.add_argument("--t-min", type=float, default=400.0)
    ap.add_argument("--t-max", type=float, default=2600.0)
    ap.add_argument("--grad-min", type=float, default=0.0)
    ap.add_argument("--grad-max", type=float, default=1.2)
    ap.add_argument("--grid-thickness-step", type=float, default=100.0)
    ap.add_argument("--grid-grad-step", type=float, default=0.2)
    ap.add_argument("--powell-maxiter", type=int, default=24)
    ap.add_argument("--min-promotion-gain", type=float, default=5e-5, help="Minimum selected-objective gain required before replacing the grid value.")
    ap.add_argument("--max-free-wr2-loss", type=float, default=0.002, help="Maximum tolerated free-row wR2 loss for a promoted continuous thickness.")
    ap.add_argument("--max-all-r1-loss", type=float, default=0.001, help="Maximum tolerated all-row R1 loss for a promoted continuous thickness.")
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    global args_progress_every
    args_progress_every = max(1, int(args.progress_every))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    target, base_sig = nrf.target_arrays(data, args.target_mode)
    indices = selected_indices(data, target, args)
    masks_all = split_masks(data.hkl, float(args.free_fraction), int(args.split_seed))
    selected = np.zeros(len(target), dtype=bool)
    selected[indices] = True
    masks = {k: v & selected for k, v in masks_all.items()}
    model = parse_model(Path(args.res))
    cfg = build_base_config(args)
    wght_a, wght_b = nrf.parse_wght(Path(args.res))
    print(
        f"selected {len(indices)} rows for continuous thickness polish | "
        f"N={args.n_basis} Q={args.n_points} start={args.start_thickness:.1f} A grad={args.start_grad:.3f}",
        flush=True,
    )

    cache_t0 = time.perf_counter()
    records = build_response_cache(
        data,
        model,
        cfg,
        indices,
        res_hash=sha256(Path(args.res)),
        cache_dir=out / "ug_cache",
    )
    cache_seconds = time.perf_counter() - cache_t0

    eval_history: list[dict] = []

    def evaluate(t0: float, grad: float, label: str = "eval") -> tuple[float, dict]:
        if not (float(args.t_min) <= t0 <= float(args.t_max)) or not (float(args.grad_min) <= grad <= float(args.grad_max)):
            return 1.0e9, {}
        pred = predict_from_response(records, len(target), float(t0), float(grad), int(args.thickness_points))
        nui = make_nuisance(args, float(t0), float(grad))
        ev = nrf.evaluate_trial(
            data,
            target,
            base_sig,
            pred,
            indices,
            masks,
            nui,
            wght_a,
            wght_b,
            scale_fit_split=str(args.scale_fit_split),
        )
        val = metric_value(ev, str(args.selection_metric))
        row = {
            "label": label,
            "thickness_A": float(t0),
            "thickness_nm": float(t0) / 10.0,
            "thickness_grad_frac": float(grad),
            "objective": float(val),
            "all_wR2": float(ev["metrics"]["all"]["dyn"]["wR2_all"]),
            "all_R1": float(ev["metrics"]["all"]["dyn"]["R1_all"]),
            "all_R1gt": float(ev["metrics"]["all"]["dyn"]["R1_gt"]),
            "free_wR2": float(ev["metrics"]["free"]["dyn"]["wR2_all"]),
            "scale_count": int(ev["scale_count"]),
            "valid_rows": int(ev["valid_rows"]),
        }
        eval_history.append(row)
        return val, ev

    # 1. Baseline at the grid-promoted point.
    baseline_val, baseline_ev = evaluate(float(args.start_thickness), float(args.start_grad), "baseline")
    print(f"baseline objective={baseline_val:.6f}", flush=True)

    # 2. Local scalar bounded regularization of thickness at fixed gradient.
    t0 = float(args.start_thickness)
    lo = max(float(args.t_min), t0 * (1.0 - float(args.bracket_frac)))
    hi = min(float(args.t_max), t0 * (1.0 + float(args.bracket_frac)))
    scalar_t0 = time.perf_counter()
    scalar_res = minimize_scalar(
        lambda x: evaluate(float(x), float(args.start_grad), "scalar_t")[0],
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 0.05, "maxiter": 32},
    )
    scalar_seconds = time.perf_counter() - scalar_t0
    scalar_val, scalar_ev = evaluate(float(scalar_res.x), float(args.start_grad), "scalar_t_final")
    print(
        f"scalar thickness optimum t={scalar_res.x:.3f} A objective={scalar_val:.6f} "
        f"in {scalar_seconds:.2f}s",
        flush=True,
    )

    # 3. Parabolic/Newton-like one-step from three local points.
    step = max(10.0, 0.05 * t0)
    xs = np.asarray([max(float(args.t_min), t0 - step), t0, min(float(args.t_max), t0 + step)], dtype=float)
    ys = np.asarray([evaluate(float(x), float(args.start_grad), "parabolic_probe")[0] for x in xs], dtype=float)
    coeff = np.polyfit(xs, ys, 2)
    if coeff[0] > 0:
        t_parab = float(np.clip(-coeff[1] / (2.0 * coeff[0]), lo, hi))
    else:
        t_parab = float(xs[int(np.argmin(ys))])
    parab_val, parab_ev = evaluate(t_parab, float(args.start_grad), "parabolic_newton")
    print(f"parabolic step t={t_parab:.3f} A objective={parab_val:.6f}", flush=True)

    # 4. Small direct grid over t and gradient to guard against local minima.
    grid_t0 = time.perf_counter()
    t_values = np.arange(
        max(float(args.t_min), t0 * (1.0 - float(args.bracket_frac))),
        min(float(args.t_max), t0 * (1.0 + float(args.bracket_frac))) + 0.5 * float(args.grid_thickness_step),
        float(args.grid_thickness_step),
    )
    g_values = np.arange(
        float(args.grad_min),
        float(args.grad_max) + 0.5 * float(args.grid_grad_step),
        float(args.grid_grad_step),
    )
    grid_best = None
    for tv in t_values:
        for gv in g_values:
            val, ev = evaluate(float(tv), float(gv), "guard_grid")
            if grid_best is None or val < grid_best[0]:
                grid_best = (val, float(tv), float(gv), ev)
    grid_seconds = time.perf_counter() - grid_t0
    print(
        f"guard grid best t={grid_best[1]:.3f} A grad={grid_best[2]:.3f} "
        f"objective={grid_best[0]:.6f} in {grid_seconds:.2f}s",
        flush=True,
    )

    # 5. Bounded Powell in thickness and gradient, seeded from the guard grid.
    powell_t0 = time.perf_counter()
    x0 = np.asarray([grid_best[1], grid_best[2]], dtype=float)
    powell_res = minimize(
        lambda x: evaluate(float(x[0]), float(x[1]), "powell")[0],
        x0,
        method="Powell",
        bounds=[(float(args.t_min), float(args.t_max)), (float(args.grad_min), float(args.grad_max))],
        options={"maxiter": int(args.powell_maxiter), "xtol": 0.05, "ftol": 1e-5},
    )
    powell_seconds = time.perf_counter() - powell_t0
    powell_val, powell_ev = evaluate(float(powell_res.x[0]), float(powell_res.x[1]), "powell_final")
    print(
        f"Powell optimum t={powell_res.x[0]:.3f} A grad={powell_res.x[1]:.4f} "
        f"objective={powell_val:.6f} in {powell_seconds:.2f}s",
        flush=True,
    )

    candidates = [
        ("baseline", baseline_val, float(args.start_thickness), float(args.start_grad), baseline_ev),
        ("scalar_t", scalar_val, float(scalar_res.x), float(args.start_grad), scalar_ev),
        ("parabolic_newton", parab_val, t_parab, float(args.start_grad), parab_ev),
        ("guard_grid", grid_best[0], grid_best[1], grid_best[2], grid_best[3]),
        ("powell", powell_val, float(powell_res.x[0]), float(powell_res.x[1]), powell_ev),
    ]
    baseline_free = float(baseline_ev["metrics"]["free"]["dyn"]["wR2_all"])
    baseline_r1 = float(baseline_ev["metrics"]["all"]["dyn"]["R1_all"])
    eligible = []
    rejected = []
    for name, val, tv, gv, ev in candidates:
        gain = float(baseline_val - val)
        free_loss = float(ev["metrics"]["free"]["dyn"]["wR2_all"] - baseline_free)
        all_r1_loss = float(ev["metrics"]["all"]["dyn"]["R1_all"] - baseline_r1)
        reasons = []
        if name != "baseline" and gain < float(args.min_promotion_gain):
            reasons.append("gain_below_minimum")
        if free_loss > float(args.max_free_wr2_loss):
            reasons.append("free_wR2_loss")
        if all_r1_loss > float(args.max_all_r1_loss):
            reasons.append("all_R1_loss")
        rec = {
            "strategy": name,
            "gain": gain,
            "free_wR2_loss": free_loss,
            "all_R1_loss": all_r1_loss,
            "reasons": reasons,
        }
        if reasons:
            rejected.append(rec)
        else:
            eligible.append((name, val, tv, gv, ev, rec))
    if not eligible:
        eligible = [("baseline", baseline_val, float(args.start_thickness), float(args.start_grad), baseline_ev, {
            "strategy": "baseline",
            "gain": 0.0,
            "free_wR2_loss": 0.0,
            "all_R1_loss": 0.0,
            "reasons": [],
        })]
    best_label, best_val, best_t, best_g, best_ev, best_decision = min(eligible, key=lambda x: x[1])
    best_nui = make_nuisance(args, best_t, best_g)
    best_native = {k: v for k, v in best_ev.items() if k != "arrays"}
    best_native["nuisance"] = asdict(best_nui)
    best_native["continuous_thickness_strategy"] = best_label

    with (out / "continuous_thickness_trials.csv").open("w", newline="") as f:
        fields = list(eval_history[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(eval_history)

    summary = {
        "inputs": {
            "hkl": str(Path(args.hkl).resolve()),
            "res": str(Path(args.res).resolve()),
            "hkl_transform": str(args.hkl_transform),
        },
        "config": vars(args),
        "contract": {
            "n_basis": int(args.n_basis),
            "n_points": int(args.n_points),
            "profile_quadrature": str(args.profile_quadrature),
            "profile_accumulation": str(args.profile_accumulation),
            "thickness_points": int(args.thickness_points),
            "topup_d_min": float(args.topup_d_min),
            "max_support_sg": float(args.max_support_sg),
            "frame_offset": float(args.frame_offset),
            "group_width_deg": float(args.group_width_deg),
            "detector_scale_bins": int(args.detector_scale_bins),
            "sigma_c_peak": float(args.sigma_c_peak),
            "selection_metric": str(args.selection_metric),
            "scale_fit_split": str(args.scale_fit_split),
            "target_mode": str(args.target_mode),
        },
        "row_selection": {
            "eligible_rows": int(len(indices)),
            "strong_rows": int(args.strong_rows),
            "max_rows": int(args.max_rows),
        },
        "timings": {
            "response_cache_seconds": float(cache_seconds),
            "scalar_seconds": float(scalar_seconds),
            "guard_grid_seconds": float(grid_seconds),
            "powell_seconds": float(powell_seconds),
            "total_seconds": float(time.perf_counter() - t_start),
            "evaluations": int(len(eval_history)),
        },
        "strategies": {
            name: {
                "objective": float(val),
                "thickness_A": float(tv),
                "thickness_nm": float(tv) / 10.0,
                "thickness_grad_frac": float(gv),
                "all_wR2": float(ev["metrics"]["all"]["dyn"]["wR2_all"]),
                "all_R1": float(ev["metrics"]["all"]["dyn"]["R1_all"]),
                "all_R1gt": float(ev["metrics"]["all"]["dyn"]["R1_gt"]),
                "free_wR2": float(ev["metrics"]["free"]["dyn"]["wR2_all"]),
                "scale_count": int(ev["scale_count"]),
                "valid_rows": int(ev["valid_rows"]),
            }
            for name, val, tv, gv, ev in candidates
        },
        "best": {
            "strategy": best_label,
            "objective": float(best_val),
            "thickness_A": float(best_t),
            "thickness_nm": float(best_t) / 10.0,
            "thickness_grad_frac": float(best_g),
            "all_wR2": float(best_ev["metrics"]["all"]["dyn"]["wR2_all"]),
            "all_R1": float(best_ev["metrics"]["all"]["dyn"]["R1_all"]),
            "all_R1gt": float(best_ev["metrics"]["all"]["dyn"]["R1_gt"]),
            "free_wR2": float(best_ev["metrics"]["free"]["dyn"]["wR2_all"]),
            "scale_count": int(best_ev["scale_count"]),
            "valid_rows": int(best_ev["valid_rows"]),
        },
        "promotion_decision": {
            "selected": best_decision,
            "rejected": rejected,
            "min_promotion_gain": float(args.min_promotion_gain),
            "max_free_wR2_loss": float(args.max_free_wr2_loss),
            "max_all_R1_loss": float(args.max_all_r1_loss),
        },
        "best_native": best_native,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    (out / "best_native_fit.json").write_text(json.dumps(best_native, indent=2, allow_nan=True) + "\n")
    report = [
        "# Continuous Thickness Regularization",
        "",
        f"Rows: {len(indices)} selected; N={args.n_basis}; Q={args.n_points}; thickness quadrature={args.thickness_points}.",
        f"Response cache build: {cache_seconds:.2f} s. Total: {summary['timings']['total_seconds']:.2f} s for {len(eval_history)} objective evaluations.",
        "",
        "| strategy | thickness A | thickness nm | grad | all wR2 | all R1 | R1gt | free wR2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, val, tv, gv, ev in candidates:
        report.append(
            f"| {name} | {tv:.3f} | {tv / 10.0:.3f} | {gv:.4f} | "
            f"{ev['metrics']['all']['dyn']['wR2_all']:.6f} | "
            f"{ev['metrics']['all']['dyn']['R1_all']:.6f} | "
            f"{ev['metrics']['all']['dyn']['R1_gt']:.6f} | "
            f"{ev['metrics']['free']['dyn']['wR2_all']:.6f} |"
        )
    report.extend(
        [
            "",
            f"Selected result: `{best_label}` at `{best_t:.3f} A` (`{best_t / 10.0:.3f} nm`) with gradient `{best_g:.4f}`.",
            f"Promotion gate: gain `{best_decision['gain']:.6f}`, free-wR2 loss `{best_decision['free_wR2_loss']:.6f}`, all-R1 loss `{best_decision['all_R1_loss']:.6f}`.",
            "",
            "The important speed result is not the subset metric itself; it is that once the response cache is built, scalar, parabolic, grid, and Powell searches reuse the same eigensystems and only change propagation phases.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"wrote {out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
