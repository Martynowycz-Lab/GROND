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

import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

if not __package__:
    # Direct legacy execution must import this checkout, not another installed GROND.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from grond.regularize.arguments_thickness import build_parser as build_thickness_parser
from grond.regularize.thickness_response import (
    PHYSICAL_GRAD_MAX as PHYSICAL_GRAD_MAX,
    PHYSICAL_GRAD_MIN as PHYSICAL_GRAD_MIN,
    ResponseRecords as ResponseRecords,
    build_base_config as build_base_config,
    dyn_from_spec as dyn_from_spec,
    estimated_packed_response_bytes as estimated_packed_response_bytes,
    inclusive_bounded_axis as inclusive_bounded_axis,
    local_powell_thickness_bounds as local_powell_thickness_bounds,
    local_thickness_search_bounds as local_thickness_search_bounds,
    metric_value as metric_value,
    pack_response_records as pack_response_records,
    predict_candidates_from_response as predict_candidates_from_response,
    predict_from_response as predict_from_response,
    predict_from_response_scalar as predict_from_response_scalar,
    propagation_spec as propagation_spec,
    require_physical_thickness_candidate as require_physical_thickness_candidate,
    thickness_candidate_domain_reasons as thickness_candidate_domain_reasons,
    thickness_grid as thickness_grid,
)

if __package__:
    from . import native_row_regularize as nrf
    from .bloch_engine import DEFAULT_MAX_SUPPORT_SG, Config, XDSBlochPort, XDSData, parse_model, sha256
else:  # pragma: no cover - direct script execution
    import native_row_regularize as nrf
    from bloch_engine import DEFAULT_MAX_SUPPORT_SG, Config, XDSBlochPort, XDSData, parse_model, sha256


DEFAULT_HKL = Path("INTEGRATE.HKL")
DEFAULT_RES = Path("model.res")


def selected_indices(data: XDSData, target: np.ndarray, args, excluded=None) -> np.ndarray:
    """Choose either all eligible rows or the strongest eligible subset.

    Rows in ``excluded`` (held-out orbit exclusion) are dropped before the
    strongest-subset ranking so they never occupy a selection slot.
    """

    idx = nrf.row_selection(
        data,
        target,
        positive_only=not bool(args.include_nonpositive),
        min_d=float(args.min_d_spacing),
        min_peak=float(args.min_peak),
        max_rows=0,
        geometry_percentile=float(args.geometry_percentile),
    )
    if excluded is not None:
        idx = np.asarray([i for i in idx if not excluded[int(i)]], dtype=int)
    if int(args.strong_rows) > 0 and len(idx) > int(args.strong_rows):
        sig = np.asarray(data.sigma, dtype=float)
        score = np.asarray(target, dtype=float) / np.maximum(sig, 1e-12)
        order = np.argsort(score[idx])[::-1]
        idx = idx[order[: int(args.strong_rows)]]
        idx = np.sort(idx)
    if int(args.max_rows) > 0 and len(idx) > int(args.max_rows):
        idx = idx[: int(args.max_rows)]
    return np.asarray(idx, dtype=int)


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
        plan = port.row_prediction_plan(int(row_idx))
        dyn_specs = [
            propagation_spec(matrix, absorptive=absorptive)
            for matrix in plan["_dyn_matrices"]
        ]
        born_specs = [
            {
                "h10_abs2": float(plan["_born_h10_abs2"]),
                "delta": float(delta),
            }
            for delta in plan["_born_deltas"]
        ]
        metadata = {key: value for key, value in plan.items() if not key.startswith("_")}
        sg_resid = np.asarray(plan["_sg_resid"], dtype=float)

        records.append(
            {
                "row_index": int(row_idx),
                "profile_weights": np.asarray(plan["_profile_weights"], dtype=float),
                "dyn_specs": dyn_specs,
                "born_specs": born_specs,
                "max_sg_solve_residual": float(np.max(np.abs(sg_resid))) if sg_resid.size else 0.0,
                **metadata,
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
    response_records = ResponseRecords(records)
    estimated_bytes = estimated_packed_response_bytes(response_records)
    max_gib = float(os.environ.get("GROND_MAX_PACKED_THICKNESS_GIB", "2.0"))
    max_bytes = max(0, int(max_gib * 1024**3))
    if estimated_bytes > max_bytes:
        response_records.packed_status = {
            "status": "skipped_memory_limit",
            "estimated_bytes": estimated_bytes,
            "maximum_bytes": max_bytes,
        }
    else:
        try:
            response_records.packed = pack_response_records(response_records)
            response_records.packed_status = {
                "status": "built",
                "estimated_bytes": estimated_bytes,
                "maximum_bytes": max_bytes,
            }
        except MemoryError:
            response_records.packed = None
            response_records.packed_status = {
                "status": "skipped_memory_error",
                "estimated_bytes": estimated_bytes,
                "maximum_bytes": max_bytes,
            }
    return response_records


def make_nuisance(args, t0: float, grad: float) -> nrf.Nuisance:
    """Return the native nuisance for thickness t0 (Å) and gradient fraction grad, other terms from args.

    Raises when (t0, grad) is outside the physical thickness domain.
    """

    require_physical_thickness_candidate(t0, grad)
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


def free_wr2_veto_on(args) -> bool:
    """Only ``--max-free-wr2-loss +inf`` switches the free-wR2 veto off (-inf still vetoes every candidate)."""

    limit = float(args.max_free_wr2_loss)
    return not (math.isinf(limit) and limit > 0.0)


def promotion_threshold_errors(args) -> list[str]:
    """Promotion thresholds that are NaN; a NaN comparison would silently switch its veto off."""

    return [
        option
        for option, value in (
            ("--min-promotion-gain", args.min_promotion_gain),
            ("--max-free-wr2-loss", args.max_free_wr2_loss),
            ("--max-all-r1-loss", args.max_all_r1_loss),
        )
        if math.isnan(float(value))
    ]


def promotion_vetoes(name: str, gain: float, ev: dict, baseline_ev: dict, args) -> dict:
    """Promotion record of one evaluated candidate against the grid baseline.

    ``--promotion-score-rows all`` (default) vetoes on the all-row R1 loss;
    ``work`` vetoes on the work-row R1 loss.  ``--max-free-wr2-loss inf`` switches the free-wR2 veto
    off, so under work the promotion reads no free row.  All three losses are recorded.
    """

    r1_rows = "work" if str(args.promotion_score_rows) == "work" else "all"
    free_veto = free_wr2_veto_on(args)
    free_loss = float(ev["metrics"]["free"]["dyn"]["wR2_all"] - baseline_ev["metrics"]["free"]["dyn"]["wR2_all"])
    all_r1_loss = float(ev["metrics"]["all"]["dyn"]["R1_all"] - baseline_ev["metrics"]["all"]["dyn"]["R1_all"])
    work_r1_loss = float(ev["metrics"]["work"]["dyn"]["R1_all"] - baseline_ev["metrics"]["work"]["dyn"]["R1_all"])
    reasons = []
    if name != "baseline" and gain < float(args.min_promotion_gain):
        reasons.append("gain_below_minimum")
    if free_veto and free_loss > float(args.max_free_wr2_loss):
        reasons.append("free_wR2_loss")
    if (work_r1_loss if r1_rows == "work" else all_r1_loss) > float(args.max_all_r1_loss):
        reasons.append(f"{r1_rows}_R1_loss")
    return {
        "strategy": name,
        "gain": gain,
        "free_wR2_loss": free_loss,
        "all_R1_loss": all_r1_loss,
        "work_R1_loss": work_r1_loss,
        "reasons": reasons,
    }


def main() -> int:
    """Search thickness and gradient continuously from cached eigensystems and return 0.

    Writes the trial CSV, summary.json, best_native_fit.json and REPORT.md to --out-dir.
    """

    ap = build_thickness_parser(
        description=__doc__,
        default_hkl=DEFAULT_HKL,
        default_res=DEFAULT_RES,
        max_support_sg=DEFAULT_MAX_SUPPORT_SG,
        native_regularizer=nrf,
        physical_grad_max=PHYSICAL_GRAD_MAX,
    )
    args = ap.parse_args()

    try:
        args.t_min, args.t_max = local_thickness_search_bounds(
            float(args.start_thickness),
            float(args.bracket_frac),
            t_min=args.t_min,
            t_max=args.t_max,
        )
    except ValueError as exc:
        ap.error(str(exc))
    search_values = {
        "t_min": float(args.t_min),
        "t_max": float(args.t_max),
        "grad_min": float(args.grad_min),
        "grad_max": float(args.grad_max),
        "grid_thickness_step": float(args.grid_thickness_step),
        "grid_grad_step": float(args.grid_grad_step),
    }
    if not all(math.isfinite(value) for value in search_values.values()):
        ap.error("continuous-thickness search bounds and steps must be finite")
    if not PHYSICAL_GRAD_MIN <= float(args.grad_min) <= float(args.grad_max) <= PHYSICAL_GRAD_MAX:
        ap.error(
            "continuous-thickness gradient bounds must satisfy "
            f"{PHYSICAL_GRAD_MIN:g} <= grad_min <= grad_max <= {PHYSICAL_GRAD_MAX:g}"
        )
    if float(args.grid_thickness_step) <= 0.0 or float(args.grid_grad_step) <= 0.0:
        ap.error("continuous-thickness grid steps must be positive")
    if int(args.powell_maxiter) <= 0:
        ap.error("continuous-thickness Powell iterations must be positive")
    if promotion_threshold_errors(args):
        ap.error(
            "continuous-thickness promotion thresholds must not be NaN (use --max-free-wr2-loss inf to switch "
            "the free-wR2 veto off): " + ", ".join(promotion_threshold_errors(args))
        )
    start_reasons = thickness_candidate_domain_reasons(
        float(args.start_thickness),
        float(args.start_grad),
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        grad_min=float(args.grad_min),
        grad_max=float(args.grad_max),
    )
    if start_reasons:
        ap.error("invalid starting continuous-thickness point: " + ", ".join(start_reasons))

    global args_progress_every
    args_progress_every = max(1, int(args.progress_every))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    target, base_sig = nrf.target_arrays(data, args.target_mode)
    model = parse_model(Path(args.res))
    from grond.bloch.splits import resolve_split_masks

    masks_all, excluded_rows, split_contract_summary, split_scheme_summary = resolve_split_masks(
        data,
        model,
        split_contract=str(getattr(args, "split_contract", "") or ""),
        free_fraction=float(args.free_fraction),
        split_seed=int(args.split_seed),
        split_scheme=str(getattr(args, "split_scheme", "legacy")),
        split_salt=str(getattr(args, "split_salt", "")),
        exclude_orbits_json=str(getattr(args, "exclude_orbits_json", "") or ""),
    )
    indices = selected_indices(data, target, args, excluded=excluded_rows if np.any(excluded_rows) else None)
    selected = np.zeros(len(target), dtype=bool)
    selected[indices] = True
    masks = {k: v & selected for k, v in masks_all.items()}
    cfg = build_base_config(args)
    wght_a, wght_b = nrf.parse_wght(Path(args.res))
    scale_model = nrf.scale_model_from_args(args)
    scale_orbit_keys = None
    if scale_model.kind == "pspline":
        scale_orbit_keys = np.full(len(target), "", dtype=object)
        scale_orbit_keys[indices] = nrf.heldout.orbit_keys(data.hkl[indices], model.get("symmetry_ops"), include_friedel=True)
    scale_state = {"model": scale_model, "pinned_from_baseline": None}
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
        """Score one (thickness Å, gradient) candidate and append it to the trial history.

        Returns (objective, native evaluation). A candidate outside the search bounds returns 1e9 and an empty
        evaluation.
        """

        if thickness_candidate_domain_reasons(
            t0,
            grad,
            t_min=float(args.t_min),
            t_max=float(args.t_max),
            grad_min=float(args.grad_min),
            grad_max=float(args.grad_max),
        ):
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
            scale_model=scale_state["model"],
            orbit_keys=scale_orbit_keys,
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
    if scale_model.kind == "pspline":
        # Hold the inner-CV choice made at the start point fixed so the thickness objective stays smooth.
        chosen = baseline_ev["scale_model"]
        scale_state["pinned_from_baseline"] = {"n_basis": int(chosen["n_basis"]), "lambda": float(chosen["lambda"])}
        scale_state["model"] = replace(
            scale_model, n_basis_grid=(int(chosen["n_basis"]),), lambda_grid=(float(chosen["lambda"]),)
        )

    # 2. Local scalar bounded regularization of thickness at fixed gradient.
    t0 = float(args.start_thickness)
    lo = float(args.t_min)
    hi = float(args.t_max)
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
    xs = np.unique(np.asarray([max(lo, t0 - step), t0, min(hi, t0 + step)], dtype=float))
    ys = np.asarray([evaluate(float(x), float(args.start_grad), "parabolic_probe")[0] for x in xs], dtype=float)
    if len(xs) == 3:
        coeff = np.polyfit(xs, ys, 2)
        if coeff[0] > 0:  # noqa: SIM108 - keep the fitted and fallback branches explicit
            t_parab = float(np.clip(-coeff[1] / (2.0 * coeff[0]), lo, hi))
        else:
            t_parab = float(xs[int(np.argmin(ys))])
    else:
        t_parab = float(xs[int(np.argmin(ys))])
    parab_val, parab_ev = evaluate(t_parab, float(args.start_grad), "parabolic_newton")
    print(f"parabolic step t={t_parab:.3f} A objective={parab_val:.6f}", flush=True)

    # 4. Small direct grid over t and gradient to guard against local minima.
    grid_t0 = time.perf_counter()
    t_values = inclusive_bounded_axis(lo, hi, float(args.grid_thickness_step))
    g_values = inclusive_bounded_axis(
        float(args.grad_min), float(args.grad_max), float(args.grid_grad_step)
    )
    grid_candidates = [
        (float(tv), float(gv)) for tv in t_values for gv in g_values
    ]
    grid_predictions = predict_candidates_from_response(
        records,
        len(target),
        np.asarray([candidate[0] for candidate in grid_candidates]),
        np.asarray([candidate[1] for candidate in grid_candidates]),
        int(args.thickness_points),
    )
    grid_best = None
    for (tv, gv), pred in zip(
        grid_candidates, grid_predictions, strict=True
    ):
        nui = make_nuisance(args, tv, gv)
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
            scale_model=scale_state["model"],
            orbit_keys=scale_orbit_keys,
        )
        val = metric_value(ev, str(args.selection_metric))
        eval_history.append(
            {
                "label": "guard_grid_batched",
                "thickness_A": tv,
                "thickness_nm": tv / 10.0,
                "thickness_grad_frac": gv,
                "objective": float(val),
                "all_wR2": float(ev["metrics"]["all"]["dyn"]["wR2_all"]),
                "all_R1": float(ev["metrics"]["all"]["dyn"]["R1_all"]),
                "all_R1gt": float(ev["metrics"]["all"]["dyn"]["R1_gt"]),
                "free_wR2": float(ev["metrics"]["free"]["dyn"]["wR2_all"]),
                "scale_count": int(ev["scale_count"]),
                "valid_rows": int(ev["valid_rows"]),
            }
        )
        if grid_best is None or val < grid_best[0]:
            grid_best = (val, tv, gv, ev)
    grid_seconds = time.perf_counter() - grid_t0
    print(
        f"guard grid best t={grid_best[1]:.3f} A grad={grid_best[2]:.3f} "
        f"objective={grid_best[0]:.6f} in {grid_seconds:.2f}s",
        flush=True,
    )

    # 5. Bounded Powell in thickness and gradient, seeded from the guard grid.
    powell_t0 = time.perf_counter()
    x0 = np.asarray([grid_best[1], grid_best[2]], dtype=float)
    powell_t_lower, powell_t_upper = local_powell_thickness_bounds(
        t_values, float(grid_best[1]), lo, hi
    )
    powell_res = minimize(
        lambda x: evaluate(float(x[0]), float(x[1]), "powell")[0],
        x0,
        method="Powell",
        bounds=[(powell_t_lower, powell_t_upper), (float(args.grad_min), float(args.grad_max))],
        options={"maxiter": int(args.powell_maxiter), "xtol": 0.05, "ftol": 1e-5},
    )
    powell_seconds = time.perf_counter() - powell_t0
    powell_t = float(powell_res.x[0])
    powell_g = float(powell_res.x[1])
    powell_domain_reasons = thickness_candidate_domain_reasons(
        powell_t,
        powell_g,
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        grad_min=float(args.grad_min),
        grad_max=float(args.grad_max),
    )
    if powell_domain_reasons:
        powell_val, powell_ev = 1.0e9, {}
        print(
            f"Powell result rejected before promotion: t={powell_t!r} A grad={powell_g!r}; "
            + ", ".join(powell_domain_reasons),
            flush=True,
        )
    else:
        powell_val, powell_ev = evaluate(powell_t, powell_g, "powell_final")
    print(
        f"Powell optimum t={powell_t:.3f} A grad={powell_g:.4f} "
        f"objective={powell_val:.6f} in {powell_seconds:.2f}s",
        flush=True,
    )

    candidates = [
        ("baseline", baseline_val, float(args.start_thickness), float(args.start_grad), baseline_ev),
        ("scalar_t", scalar_val, float(scalar_res.x), float(args.start_grad), scalar_ev),
        ("parabolic_newton", parab_val, t_parab, float(args.start_grad), parab_ev),
        ("guard_grid", grid_best[0], grid_best[1], grid_best[2], grid_best[3]),
        ("powell", powell_val, powell_t, powell_g, powell_ev),
    ]
    # Under --promotion-score-rows work the R1 veto reads work rows; with --max-free-wr2-loss inf the free
    # veto is off and the promotion reads no free row.
    r1_rows = "work" if str(args.promotion_score_rows) == "work" else "all"
    free_veto = free_wr2_veto_on(args)
    eligible = []
    rejected = []
    for name, val, tv, gv, ev in candidates:
        reasons = thickness_candidate_domain_reasons(
            tv,
            gv,
            t_min=float(args.t_min),
            t_max=float(args.t_max),
            grad_min=float(args.grad_min),
            grad_max=float(args.grad_max),
        )
        if not math.isfinite(float(val)):
            reasons.append("nonfinite_objective")
        if not ev:
            reasons.append("missing_valid_evaluation")
        if reasons:
            rejected.append(
                {
                    "strategy": name,
                    "gain": None,
                    "free_wR2_loss": None,
                    "all_R1_loss": None,
                    "work_R1_loss": None,
                    "reasons": list(dict.fromkeys(reasons)),
                }
            )
            continue
        rec = promotion_vetoes(name, float(baseline_val - val), ev, baseline_ev, args)
        reasons = rec["reasons"]
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
            "work_R1_loss": 0.0,
            "reasons": [],
        })]
    best_label, best_val, best_t, best_g, best_ev, best_decision = min(eligible, key=lambda x: x[1])
    best_nui = make_nuisance(args, best_t, best_g)
    boundary_tolerance = max(0.1, 0.01 * float(args.grid_thickness_step))
    thickness_at_search_boundary = bool(
        math.isclose(best_t, lo, rel_tol=0.0, abs_tol=boundary_tolerance)
        or math.isclose(best_t, hi, rel_tol=0.0, abs_tol=boundary_tolerance)
    )
    gradient_at_physical_boundary = bool(
        math.isclose(best_g, PHYSICAL_GRAD_MAX, rel_tol=0.0, abs_tol=1e-6)
    )
    identifiability_warnings = []
    if thickness_at_search_boundary:
        identifiability_warnings.append(
            "selected thickness is on the local search boundary; the mean is not bracketed by this run"
        )
    if gradient_at_physical_boundary:
        identifiability_warnings.append(
            "selected gradient is at the widest physical uniform-gradient boundary; interpret it as effective dephasing unless independently calibrated"
        )
    elif best_g >= 0.8:
        identifiability_warnings.append(
            "selected thickness distribution is broad enough to suppress coherent thickness oscillations; the mean may be weakly identified"
        )
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
            "scale_model": {**scale_model.to_dict(), "pinned_from_baseline": scale_state["pinned_from_baseline"]},
            "target_mode": str(args.target_mode),
        },
        "row_selection": {
            "eligible_rows": int(len(indices)),
            "strong_rows": int(args.strong_rows),
            "max_rows": int(args.max_rows),
            "split_scheme": split_scheme_summary,
            "transform_hand_split_contract": split_contract_summary,
        },
        "timings": {
            "response_cache_seconds": float(cache_seconds),
            "scalar_seconds": float(scalar_seconds),
            "guard_grid_seconds": float(grid_seconds),
            "powell_seconds": float(powell_seconds),
            "total_seconds": float(time.perf_counter() - t_start),
            "evaluations": int(len(eval_history)),
            "packed_response": dict(getattr(records, "packed_status", {})),
        },
        "strategies": {
            name: {
                "status": "ok" if ev else "rejected_invalid_domain",
                "objective": float(val) if ev else None,
                "thickness_A": float(tv),
                "thickness_nm": float(tv) / 10.0,
                "thickness_grad_frac": float(gv),
                "all_wR2": float(ev["metrics"]["all"]["dyn"]["wR2_all"]) if ev else None,
                "all_R1": float(ev["metrics"]["all"]["dyn"]["R1_all"]) if ev else None,
                "all_R1gt": float(ev["metrics"]["all"]["dyn"]["R1_gt"]) if ev else None,
                "free_wR2": float(ev["metrics"]["free"]["dyn"]["wR2_all"]) if ev else None,
                "scale_count": int(ev["scale_count"]) if ev else None,
                "valid_rows": int(ev["valid_rows"]) if ev else None,
                "domain_reasons": thickness_candidate_domain_reasons(
                    tv,
                    gv,
                    t_min=float(args.t_min),
                    t_max=float(args.t_max),
                    grad_min=float(args.grad_min),
                    grad_max=float(args.grad_max),
                ),
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
        "thickness_identifiability": {
            "status": "warning" if identifiability_warnings else "locally_bracketed",
            "mean_thickness_A": float(best_t),
            "absolute_half_width_A": float(best_t * best_g),
            "nominal_support_A": [float(best_t * (1.0 - best_g)), float(best_t * (1.0 + best_g))],
            "thickness_at_local_search_boundary": thickness_at_search_boundary,
            "gradient_at_physical_boundary": gradient_at_physical_boundary,
            "warnings": identifiability_warnings,
        },
        "promotion_decision": {
            "selected": best_decision,
            "rejected": rejected,
            "min_promotion_gain": float(args.min_promotion_gain),
            "max_free_wR2_loss": float(args.max_free_wr2_loss),
            "max_all_R1_loss": float(args.max_all_r1_loss),
            "selection_rows": r1_rows,
            "free_veto": free_veto,
            "free_rows_in_selection": bool(
                r1_rows != "work"
                or free_veto
                or str(args.selection_metric) != "work_wR2"
                or str(args.scale_fit_split) != "work"
            ),
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
    for name, _val, tv, gv, ev in candidates:
        if ev:
            report.append(
                f"| {name} | {tv:.3f} | {tv / 10.0:.3f} | {gv:.4f} | "
                f"{ev['metrics']['all']['dyn']['wR2_all']:.6f} | "
                f"{ev['metrics']['all']['dyn']['R1_all']:.6f} | "
                f"{ev['metrics']['all']['dyn']['R1_gt']:.6f} | "
                f"{ev['metrics']['free']['dyn']['wR2_all']:.6f} |"
            )
        else:
            report.append(
                f"| {name} | {tv:.3f} | {tv / 10.0:.3f} | {gv:.4f} | rejected | rejected | rejected | rejected |"
            )
    report.extend(
        [
            "",
            f"Selected result: `{best_label}` at `{best_t:.3f} A` (`{best_t / 10.0:.3f} nm`) with gradient `{best_g:.4f}`.",
            f"Promotion gate: gain `{best_decision['gain']:.6f}`, free-wR2 loss `{best_decision['free_wR2_loss']:.6f}`, all-R1 loss `{best_decision['all_R1_loss']:.6f}`."
            + (
                f" Work-row R1 loss `{best_decision['work_R1_loss']:.6f}` (work-row R1 veto"
                + ("" if free_veto else "; free-wR2 veto off")
                + ")."
                if r1_rows == "work"
                else ""
            ),
            f"Thickness identifiability: `{summary['thickness_identifiability']['status']}`; absolute half-width `{summary['thickness_identifiability']['absolute_half_width_A']:.3f} A`.",
            *(f"Warning: {warning}." for warning in summary["thickness_identifiability"]["warnings"]),
            "",
            "The important speed result is not the subset metric itself; it is that once the response cache is built, scalar, parabolic, grid, and Powell searches reuse the same eigensystems and only change propagation phases.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"wrote {out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
