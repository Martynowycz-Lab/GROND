#!/usr/bin/env python3
"""Native Bloch row structure update for XDS row data.

This is the diagnostic gradient-bearing layer above the frozen Bloch row scorer.
It updates selected fractional coordinates and ADPs against a native intensity
least-squares target with validation-gated block acceptance. By default,
nuisance scales are frozen inside each coordinate/ADP microcycle and refreshed
only between blocks, so a moving structure cannot be hidden by per-evaluation
scale refits. It is intentionally conservative and is not the default single-dataset
path when the SHELXL feedback loop is available.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

if not __package__:
    # Direct legacy execution must import this checkout, not another installed GROND.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.optimize import least_squares

from grond.refine.arguments_native import build_parser as build_native_update_parser
from grond.refine.geometry_restraints import (
    COVALENT_RADII_A as COVALENT_RADII_A,
    adp_similarity_residuals as adp_similarity_residuals,
    atom_distance_A as atom_distance_A,
    bond_restraint_residuals as bond_restraint_residuals,
    build_start_restraints as build_start_restraints,
    cart_delta as cart_delta,
    frac_delta_min_image as frac_delta_min_image,
    frac_xyz as frac_xyz,
    write_cctbx_restraint_view as write_cctbx_restraint_view,
    write_restraints_csv as write_restraints_csv,
)
from grond.refine.native_objective import (
    apply_row_resolution_to_prediction as apply_row_resolution_to_prediction,
    apply_scale_contract as apply_scale_contract,
    choose_objective_rows as choose_objective_rows,
    fit_scale_contract as fit_scale_contract,
    residual_vector as residual_vector,
    scaled_prediction as scaled_prediction,
    validation_score as validation_score,
)
from grond.refine.native_output import (
    format_native_value as format_native_value,
    update_res_coordinates as update_res_coordinates,
    write_model_delta_csv as write_model_delta_csv,
    write_objective_rows as write_objective_rows,
    write_parameter_csv as write_parameter_csv,
)
from grond.refine.parameters import (
    absolute_finite_difference_jacobian as absolute_finite_difference_jacobian,
    atom_is_refinable as atom_is_refinable,
    atom_ueq as atom_ueq,
    model_from_vector as model_from_vector,
    parameter_absolute_steps as parameter_absolute_steps,
    parameter_bounds as parameter_bounds,
    parameter_layout as parameter_layout,
    parameter_prior as parameter_prior,
    set_atom_ueq as set_atom_ueq,
    special_position_coordinate_policy as special_position_coordinate_policy,
)

if __package__:
    from .bloch_engine import (
        DEFAULT_MAX_SUPPORT_SG,
        Config,
        XDSBlochPort,
        XDSData,
        alternate_hand_model,
        parse_model,
        sha256,
        shelx_u_cart_matrix,
        split_masks,
    )
    from .native_row_regularize import (
        DEFAULT_BASELINE_HKL,
        DEFAULT_HKL,
        DEFAULT_RES,
        TARGET_MODES,
        Nuisance,
        copy_baseline,
        evaluate_trial,
        export_corrected_endpoint,
        native_metrics,
        parse_float_grid,
        parse_wght,
        row_selection,
        sigma_eff,
        target_arrays,
    )
else:  # pragma: no cover - direct script execution
    from bloch_engine import (
        DEFAULT_MAX_SUPPORT_SG,
        Config,
        XDSBlochPort,
        XDSData,
        alternate_hand_model,
        parse_model,
        sha256,
        shelx_u_cart_matrix,
        split_masks,
    )
    from native_row_regularize import (
        DEFAULT_BASELINE_HKL,
        DEFAULT_HKL,
        DEFAULT_RES,
        TARGET_MODES,
        Nuisance,
        copy_baseline,
        evaluate_trial,
        export_corrected_endpoint,
        native_metrics,
        parse_float_grid,
        parse_wght,
        row_selection,
        sigma_eff,
        target_arrays,
    )


class EarlyStopOptimization(Exception):
    """Stop a structure block after a real plateau while preserving its best vector."""

    def __init__(self, reason: str, x, metric: float, eval_no: int):
        super().__init__(reason)
        self.reason = reason
        self.x = np.asarray(x, dtype=float).copy()
        self.metric = float(metric)
        self.eval_no = int(eval_no)


NATIVE_UPDATE_TRACE_FIELDS = (
    "block",
    "mode",
    "eval",
    "rms_residual",
    "data_rms",
    "prior_rms",
    "geometry_rms",
    "adp_similarity_rms",
    "max_abs_shift",
    "scale_count",
    "objective_scale_mode",
    "objective_group_width_deg",
    "pets_objective_cache_only",
)


def write_native_update_trace(path: Path, trace: list[dict]) -> None:
    """Write the update trace without silently dropping provenance fields."""

    known = set(NATIVE_UPDATE_TRACE_FIELDS)
    unknown = sorted({key for row in trace for key in row} - known)
    if unknown:
        raise ValueError(f"native update trace schema is missing fields: {', '.join(unknown)}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NATIVE_UPDATE_TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(trace)


def early_stop_min_evals(args, params: list[dict]) -> int:
    """Return the minimum residual calls before a plateau gate is allowed to fire."""

    explicit = int(getattr(args, "early_stop_min_evals", 0))
    if explicit > 0:
        return explicit
    window = max(1, int(getattr(args, "early_stop_window", 0)))
    return max(2 * (len(params) + 1), window)


def make_early_stop_result(exc: EarlyStopOptimization, eval_count: int):
    """Build a small scipy-like result object for a block stopped by our gate."""

    return SimpleNamespace(
        x=exc.x,
        success=False,
        status=0,
        message=f"early stopped: {exc.reason}",
        cost=float("nan"),
        optimality=float("nan"),
        nfev=int(eval_count),
        early_stop=True,
        early_stop_reason=exc.reason,
        early_stop_metric=float(exc.metric),
        early_stop_eval=int(exc.eval_no),
    )


def inspect_cctbx_bridge(src: Path, out_dir: Path):
    view = out_dir / "cctbx_restraint_view.res"
    status = {"available": False, "parsed": False, "view": str(view)}
    try:
        from iotbx import shelx

        write_cctbx_restraint_view(src, view)
        parser = getattr(shelx, "parse_smtbx_" + "ref" + "inement_model")
        builder = parser(filename=str(view), strictly_shelxl=False)
        constraints = getattr(builder, "constraints", []) or []
        manager = getattr(builder, "restraints_manager", None)
        structure = getattr(builder, "structure", None)
        status.update(
            {
                "available": True,
                "parsed": True,
                "scatterers": int(structure.scatterers().size()) if structure is not None else None,
                "constraints_count": int(len(constraints)),
                "constraints_preview": [str(c).splitlines()[0] for c in constraints[:12]],
                "restraints_manager_type": type(manager).__name__ if manager is not None else None,
            }
        )
    except Exception as exc:
        status.update({"error": f"{type(exc).__name__}: {exc}"})
    return status


class CctbxRiguEvaluator:
    """Evaluate cctbx/SHELX RIGU residuals for the current native model."""

    def __init__(self, src: Path, out_dir: Path):
        self.view = out_dir / "cctbx_restraint_view.res"
        self.status = {"available": False, "parsed": False, "view": str(self.view), "rigu_count": 0}
        try:
            from cctbx import adp_restraints
            from iotbx import shelx
            from scitbx.array_family import flex

            write_cctbx_restraint_view(src, self.view)
            parser = getattr(shelx, "parse_smtbx_" + "ref" + "inement_model")
            builder = parser(filename=str(self.view), strictly_shelxl=False)
            self._adp_restraints = adp_restraints
            self._flex = flex
            self._xray_structure = builder.structure
            self._restraints_manager = builder.restraints_manager
            self._rigu_proxies = getattr(self._restraints_manager, "rigu_proxies", None)
            self._labels = list(self._xray_structure.scatterers().extract_labels())
            base_scatterers = self._xray_structure.scatterers()
            self._base_sites = [tuple(v) for v in base_scatterers.extract_sites()]
            self._base_u_cart = [tuple(v) for v in base_scatterers.extract_u_cart(self._xray_structure.unit_cell())]
            self._base_u_iso = list(base_scatterers.extract_u_iso())
            self._base_use_u_aniso = list(base_scatterers.extract_use_u_aniso())
            constraints = getattr(builder, "constraints", []) or []
            rigu_count = int(self._rigu_proxies.size()) if self._rigu_proxies is not None else 0
            self.status.update(
                {
                    "available": True,
                    "parsed": True,
                    "scatterers": int(self._xray_structure.scatterers().size()),
                    "constraints_count": int(len(constraints)),
                    "constraints_preview": [str(c).splitlines()[0] for c in constraints[:12]],
                    "restraints_manager_type": type(self._restraints_manager).__name__ if self._restraints_manager is not None else None,
                    "rigu_count": rigu_count,
                }
            )
        except Exception as exc:
            self._adp_restraints = None
            self._flex = None
            self._xray_structure = None
            self._restraints_manager = None
            self._rigu_proxies = None
            self._labels = []
            self._base_sites = []
            self._base_u_cart = []
            self._base_u_iso = []
            self._base_use_u_aniso = []
            self.status.update({"error": f"{type(exc).__name__}: {exc}"})

    @property
    def count(self) -> int:
        return int(self.status.get("rigu_count") or 0)

    def residuals(self, model):
        if not self.status.get("parsed") or self._rigu_proxies is None or self.count <= 0:
            return np.zeros(0, dtype=float)
        atom_by_name = {a["name"]: a for a in model["atoms"]}
        sites = []
        u_cart = []
        for label, base_site, base_u in zip(self._labels, self._base_sites, self._base_u_cart, strict=False):
            atom = atom_by_name.get(label)
            if atom is None:
                sites.append(base_site)
                u_cart.append(base_u)
                continue
            sites.append((float(atom["x"]) % 1.0, float(atom["y"]) % 1.0, float(atom["z"]) % 1.0))
            if atom.get("adp_type") == "anis":
                u = shelx_u_cart_matrix(atom, model["cell"])
                u_cart.append((u[0, 0], u[1, 1], u[2, 2], u[0, 1], u[0, 2], u[1, 2]))
            else:
                u = atom_ueq(atom, model["cell"])
                u_cart.append((u, u, u, 0.0, 0.0, 0.0))
        xs = self._xray_structure.deep_copy_scatterers()
        xs.set_sites_frac(self._flex.vec3_double(sites))
        xs.set_u_cart(self._flex.sym_mat3_double(u_cart))
        sc = xs.scatterers()
        params = self._adp_restraints.adp_restraint_params(
            xs.sites_cart(),
            sc.extract_u_cart(xs.unit_cell()),
            sc.extract_u_iso(),
            sc.extract_use_u_aniso(),
        )
        values = np.asarray(list(self._adp_restraints.rigu_residuals(params, self._rigu_proxies)), dtype=float)
        return np.sqrt(np.maximum(values, 0.0))


def prediction_for_model(data, model, cfg, indices, cache_root: Path, model_tag: str, workers: int, chunksize: int):
    # Do not use the persistent prediction cache for moving models.  The Ug cube
    # cache is still content-addressed by model hash inside XDSBlochPort.
    port = XDSBlochPort(data, model, cfg, cache_dir=cache_root / "ug", res_hash=model_tag)
    return port.predict_rows(indices, workers=workers, chunksize=chunksize)


def evaluate_model(data, model, cfg, target, sig, indices, masks, fit_indices, wght_a, wght_b, cache_root, tag, workers, chunksize, detector_scale_bins: int = 0, row_resolution_b: float = 0.0):
    pred = prediction_for_model(data, model, cfg, indices, cache_root, tag, workers, chunksize)
    full, dyn_scaled, scales, groups, scale_count = scaled_prediction(data, target, sig, pred, cfg, fit_indices, wght_a, wght_b, detector_scale_bins=detector_scale_bins, row_resolution_b=row_resolution_b)
    out = {"splits": {}, "scale_count": int(scale_count), "prediction_seconds": float(pred.get("seconds", 0.0))}
    valid = np.isfinite(dyn_scaled)
    for split, mask in masks.items():
        out["splits"][split] = native_metrics(target, sig, dyn_scaled, mask & valid, wght_a, wght_b, n_params=int(scale_count))
    out["arrays"] = {"dyn_raw": full, "dyn_scaled": dyn_scaled, "groups": groups, "valid": valid}
    return out


def main():
    ap = build_native_update_parser(
        description=__doc__,
        default_hkl=DEFAULT_HKL,
        default_res=DEFAULT_RES,
        default_baseline_hkl=DEFAULT_BASELINE_HKL,
        target_modes=TARGET_MODES,
        max_support_sg=DEFAULT_MAX_SUPPORT_SG,
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = out / "cache"
    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    base_model = parse_model(Path(args.res))
    model_tag = "alternate_fractional_xyz" if args.alternate_hand else "original"
    model_tag = f"{model_tag};hkl_transform={data.hkl_transform_spec}"
    if args.alternate_hand:
        base_model = alternate_hand_model(base_model)
    target, base_sig = target_arrays(data, args.target_mode)
    target_indices = row_selection(data, target, not args.include_nonpositive, args.min_d_spacing, args.min_peak, 0, args.geometry_percentile)
    masks = split_masks(
        data.hkl,
        args.free_fraction,
        args.split_seed,
        symmetry_ops=base_model.get("symmetry_ops"),
    )
    selected_mask = np.zeros(len(target), dtype=bool)
    selected_mask[target_indices] = True
    masks = {k: v & selected_mask for k, v in masks.items()}
    nuisance = Nuisance(
        args.thickness,
        args.thickness_grad_frac,
        args.frame_offset,
        args.group_width_deg,
        args.detector_scale_bins,
        args.sigma_a,
        args.sigma_b_corr,
        args.sigma_c_peak,
        row_resolution_b=args.row_resolution_b,
    )
    sig = sigma_eff(data, target, base_sig, nuisance)
    objective_indices = choose_objective_rows(data, target_indices, masks, target, sig, args.objective_rows, args.objective_bins, args.split_seed)
    wght_a, wght_b = parse_wght(Path(args.res))
    block_schedule = [x.strip() for x in re.split(r"[,\s]+", args.block_schedule) if x.strip()]
    allowed_blocks = {"xyz", "adp_ueq", "xyz_adp_ueq"}
    unknown_blocks = [b for b in block_schedule if b not in allowed_blocks]
    if unknown_blocks:
        raise ValueError(f"unknown model-update block(s): {unknown_blocks}; allowed: {sorted(allowed_blocks)}")
    cfg = Config(
        n_basis=args.n_basis,
        n_points=args.n_points,
        sg_window=args.sg_window,
        sg_window_mode=args.sg_window_mode,
        sg_window_min=args.sg_window_min,
        sg_window_max=args.sg_window_max,
        sg_window_scale=args.sg_window_scale,
        sg_window_frame_floor=args.sg_window_frame_floor,
        thickness_A=args.thickness,
        thickness_grad_frac=args.thickness_grad_frac,
        thickness_points=args.thickness_points,
        topup_d_min=min(float(args.topup_d_min), float(args.min_d_spacing)) if args.min_d_spacing > 0 else float(args.topup_d_min),
        max_support_sg=args.max_support_sg,
        group_width_deg=args.group_width_deg,
        frame_offset=args.frame_offset,
        free_fraction=args.free_fraction,
        split_seed=args.split_seed,
        variance_a=args.sigma_a,
        variance_b_corr=args.sigma_b_corr,
        variance_c_peak=args.sigma_c_peak,
        compute_backend=args.compute_backend,
        jax_batch_size=args.jax_batch_size,
        jax_matrix_bucket_size=args.jax_matrix_bucket_size,
        bethe_strong_count=args.bethe_strong_count,
        bethe_variant=args.bethe_variant,
        bethe_promote_denom=args.bethe_promote_denom,
        bethe_denom_eps=args.bethe_denom_eps,
        profile_quadrature=args.profile_quadrature,
        profile_accumulation=args.profile_accumulation,
        sg_jitter_mode=args.sg_jitter_mode,
        sg_jitter_sigma0=args.sg_jitter_sigma0,
        sg_jitter_sigma_d_inv=args.sg_jitter_sigma_d_inv,
        sg_jitter_sigma_frame=args.sg_jitter_sigma_frame,
        sg_jitter_points=args.sg_jitter_points,
        sg_jitter_student_df=args.sg_jitter_student_df,
        sg_jitter_width_factor=args.sg_jitter_width_factor,
        absorption_mode=args.absorption_mode,
        absorption_mu=args.absorption_mu,
        absorption_frac=args.absorption_frac,
        absorption_mu_g2=args.absorption_mu_g2,
        absorption_mu_theta2=args.absorption_mu_theta2,
        absorption_tds_b=args.absorption_tds_b,
        ug_d_min=args.ug_d_min,
        row_plan_cache=args.row_plan_cache,
    )
    objective_cfg = copy.deepcopy(cfg)
    objective_cfg.group_width_deg = args.objective_group_width_deg
    pets_objective_cache_only = bool(
        getattr(data, "pets_exact_a_center", None) is not None
        and str(args.row_plan_cache).lower() == "readwrite"
    )
    if pets_objective_cache_only:
        # A full N~400/Q~40 PETS cache stores O(rows*N^2) coupling indices and
        # can consume many gigabytes. The finite-difference objective is the
        # repeated hot path and uses only its stratified row subset, so cache
        # that geometry while leaving one-off full validation passes uncached.
        cfg.row_plan_cache = "off"
        objective_cfg.row_plan_cache = "readwrite"
    trace = []
    block_results = []
    restraints = []
    if args.restraints == "start_geometry":
        restraints = build_start_restraints(
            base_model,
            args.atom_selection,
            args.atom_labels,
            args.bond_detection_tolerance,
            args.bond_detection_max_distance,
        )
    write_restraints_csv(out / "restraints.csv", restraints)
    cctbx_bridge = inspect_cctbx_bridge(Path(args.res), out)
    write_objective_rows(out / "objective_rows.csv", data, objective_indices, target, sig, masks)

    if args.scale_fit_split == "all":
        validation_fit_indices = np.asarray(target_indices, dtype=int)
    else:
        validation_fit_indices = np.asarray([i for i in target_indices if masks["work"][i]], dtype=int)
    start_eval = evaluate_model(data, base_model, cfg, target, sig, target_indices, masks, validation_fit_indices, wght_a, wght_b, cache_root, "start_validation", args.workers, args.chunksize, detector_scale_bins=args.detector_scale_bins, row_resolution_b=args.row_resolution_b)
    current_model = copy.deepcopy(base_model)
    current_eval = start_eval
    accepted = False
    acceptance_reason = "no model-update blocks were accepted"
    last_candidate_eval = start_eval

    for block_no, block_mode in enumerate(block_schedule, start=1):
        block_tag = f"block_{block_no:02d}_{block_mode}"
        coordinate_policy = special_position_coordinate_policy(current_model, args.atom_selection, args.atom_labels, block_mode)
        try:
            params, x0 = parameter_layout(current_model, args.atom_selection, args.atom_labels, block_mode)
        except ValueError as exc:
            block_results.append({"block": block_no, "mode": block_mode, "skipped": True, "reason": str(exc), "coordinate_constraints": coordinate_policy})
            continue
        has_xyz_params = any(p.get("kind", "xyz") == "xyz" for p in params)
        has_adp_params = any(p.get("kind") == "adp_ueq" for p in params)
        bounds = parameter_bounds(params, x0, args.max_shift_frac, args.adp_shift, args.adp_min, args.adp_max)
        absolute_steps = parameter_absolute_steps(
            params,
            xyz_step_frac=args.xyz_diff_step_frac,
            adp_step_a2=args.adp_diff_step_a2,
            legacy_step=args.diff_step,
        )
        eval_counter = {"n": 0}
        residual_cache = {"x": None, "value": None}
        block_scale_contract = None
        if args.objective_scale_mode == "frozen_block":
            anchor_pred = prediction_for_model(data, current_model, objective_cfg, objective_indices, cache_root, f"{block_tag}_scale_anchor", args.workers, args.chunksize)
            anchor_full = np.full(len(target), np.nan, dtype=float)
            anchor_full[anchor_pred["indices"]] = anchor_pred["dyn"]
            anchor_full, _, _ = apply_row_resolution_to_prediction(data, anchor_full, args.row_resolution_b)
            block_scale_contract = fit_scale_contract(
                data,
                target,
                sig,
                anchor_full,
                objective_cfg,
                objective_indices,
                wght_a,
                wght_b,
                detector_scale_bins=args.detector_scale_bins,
            )
            block_scale_contract["mode"] = "frozen_block"
            block_scale_contract["anchor_prediction_seconds"] = float(anchor_pred.get("seconds", 0.0))
        early_stop_state = {
            "enabled": int(args.early_stop_window) > 0,
            "window": max(1, int(args.early_stop_window)),
            "min_evals": early_stop_min_evals(args, params),
            "min_improvement": max(0.0, float(args.early_stop_min_improvement)),
            "best_metric": float("inf"),
            "best_eval": 0,
            "best_x": np.asarray(x0, dtype=float).copy(),
            "initial_metric": float("nan"),
        }

        def residuals(
            x,
            *,
            block_no=block_no,
            block_mode=block_mode,
            block_tag=block_tag,
            params=params,
            x0=x0,
            current_model=current_model,
            has_xyz_params=has_xyz_params,
            has_adp_params=has_adp_params,
        ):
            eval_counter["n"] += 1
            model = model_from_vector(current_model, params, x)
            pred = prediction_for_model(data, model, objective_cfg, objective_indices, cache_root, f"{block_tag}_moving", args.workers, args.chunksize)
            _, scaled, scales, _, scale_count = scaled_prediction(
                data,
                target,
                sig,
                pred,
                objective_cfg,
                objective_indices,
                wght_a,
                wght_b,
                detector_scale_bins=args.detector_scale_bins,
                scale_contract=block_scale_contract,
                row_resolution_b=args.row_resolution_b,
            )
            data_resid = residual_vector(data, target, sig, scaled, objective_indices, wght_a, wght_b)
            prior = parameter_prior(params, x, x0, args.xyz_restraint_sigma, args.adp_restraint_sigma)
            geom_resid = bond_restraint_residuals(model, restraints, args.bond_restraint_sigma) if has_xyz_params else np.zeros(0, dtype=float)
            adp_resid = adp_similarity_residuals(model, restraints, args.adp_similarity_sigma) if has_adp_params else np.zeros(0, dtype=float)
            total = np.concatenate([data_resid, prior, geom_resid, adp_resid])
            rms = float(np.sqrt(np.mean(total * total))) if len(total) else float("nan")
            data_rms = float(np.sqrt(np.mean(data_resid * data_resid))) if len(data_resid) else float("nan")
            max_shift = float(np.max(np.abs(x - x0))) if len(x) else 0.0
            trace.append(
                {
                    "block": block_no,
                    "mode": block_mode,
                    "eval": eval_counter["n"],
                    "rms_residual": rms,
                    "data_rms": data_rms,
                    "prior_rms": float(np.sqrt(np.mean(prior * prior))) if len(prior) else 0.0,
                    "geometry_rms": float(np.sqrt(np.mean(geom_resid * geom_resid))) if len(geom_resid) else 0.0,
                    "adp_similarity_rms": float(np.sqrt(np.mean(adp_resid * adp_resid))) if len(adp_resid) else 0.0,
                    "max_abs_shift": max_shift,
                    "scale_count": int(scale_count),
                    "objective_scale_mode": args.objective_scale_mode,
                    "objective_group_width_deg": float(args.objective_group_width_deg),
                    "pets_objective_cache_only": pets_objective_cache_only,
                }
            )
            print(
                f"block {block_no:02d} {block_mode} eval {eval_counter['n']:03d}: "
                f"data_rms={trace[-1]['data_rms']:.4f} total_rms={rms:.4f} max_shift={max_shift:.5f}",
                flush=True,
            )
            if early_stop_state["enabled"]:
                stop_metric = rms if args.early_stop_metric == "total_rms" else data_rms
                if math.isfinite(stop_metric):
                    if not math.isfinite(float(early_stop_state["initial_metric"])):
                        early_stop_state["initial_metric"] = float(stop_metric)
                    if stop_metric < float(early_stop_state["best_metric"]) - float(early_stop_state["min_improvement"]):
                        early_stop_state["best_metric"] = float(stop_metric)
                        early_stop_state["best_eval"] = int(eval_counter["n"])
                        early_stop_state["best_x"] = np.asarray(x, dtype=float).copy()
                    stale = int(eval_counter["n"]) - int(early_stop_state["best_eval"])
                    if int(eval_counter["n"]) >= int(early_stop_state["min_evals"]) and stale >= int(early_stop_state["window"]):
                        reason = (
                            f"{args.early_stop_metric} plateau for {stale} residual calls "
                            f"after eval {early_stop_state['best_eval']} "
                            f"(best {float(early_stop_state['best_metric']):.6g})"
                        )
                        raise EarlyStopOptimization(
                            reason,
                            early_stop_state["best_x"],
                            float(early_stop_state["best_metric"]),
                            int(early_stop_state["best_eval"]),
                        )
            residual_cache["x"] = np.asarray(x, dtype=float).copy()
            residual_cache["value"] = np.asarray(total, dtype=float).copy()
            return total

        def jacobian(x):
            cached = residual_cache.get("value")
            cached_x = residual_cache.get("x")
            if cached_x is None or not np.array_equal(np.asarray(cached_x, dtype=float), np.asarray(x, dtype=float)):
                cached = None
            return absolute_finite_difference_jacobian(
                residuals,
                x,
                bounds,
                absolute_steps,
                f0=cached,
            )

        x_scale = []
        for p in params:
            x_scale.append(max(float(args.adp_shift if p.get("kind") == "adp_ueq" else args.max_shift_frac), 1e-5))
        try:
            result = least_squares(
                residuals,
                x0,
                bounds=bounds,
                jac=jacobian,
                loss="linear",
                max_nfev=args.max_nfev,
                verbose=1,
                x_scale=np.asarray(x_scale, dtype=float),
            )
        except EarlyStopOptimization as exc:
            result = make_early_stop_result(exc, eval_counter["n"])
            print(f"block {block_no:02d} {block_mode}: {result.message}", flush=True)
        candidate_model = model_from_vector(current_model, params, result.x)
        candidate_eval = evaluate_model(data, candidate_model, cfg, target, sig, target_indices, masks, validation_fit_indices, wght_a, wght_b, cache_root, f"{block_tag}_candidate_validation", args.workers, args.chunksize, detector_scale_bins=args.detector_scale_bins, row_resolution_b=args.row_resolution_b)
        last_candidate_eval = candidate_eval

        start_metric = validation_score(current_eval, args.acceptance_metric)
        candidate_metric = validation_score(candidate_eval, args.acceptance_metric)
        free_start = validation_score(current_eval, "free_wR2")
        free_candidate = validation_score(candidate_eval, "free_wR2")
        metric_ok = candidate_metric <= start_metric - float(args.acceptance_min_delta)
        free_ok = args.allow_free_worse or args.acceptance_metric == "free_wR2" or free_candidate <= free_start + float(args.free_wR2_tolerance)
        block_accepted = bool(metric_ok and free_ok)
        block_reason = (
            f"accepted: {args.acceptance_metric} {start_metric:.6g} -> {candidate_metric:.6g}"
            if block_accepted
            else f"rejected: {args.acceptance_metric} {start_metric:.6g} -> {candidate_metric:.6g}; free_wR2 {free_start:.6g} -> {free_candidate:.6g}; metric_ok={metric_ok}; free_ok={free_ok}"
        )
        candidate_res = out / f"{block_tag}_candidate.res"
        accepted_res_block = out / f"{block_tag}_accepted.res"
        update_res_coordinates(Path(args.res), candidate_res, candidate_model)
        update_res_coordinates(Path(args.res), accepted_res_block, candidate_model if block_accepted else current_model)
        write_parameter_csv(out / f"{block_tag}_candidate_parameters.csv", params, x0, result.x)
        write_parameter_csv(out / f"{block_tag}_accepted_parameters.csv", params, x0, result.x if block_accepted else x0)
        block_record = {
            "block": block_no,
            "mode": block_mode,
            "skipped": False,
            "parameter_count": int(len(params)),
            "coordinate_constraints": coordinate_policy,
            "xyz_parameter_count": int(sum(1 for p in params if p.get("kind", "xyz") == "xyz")),
            "adp_parameter_count": int(sum(1 for p in params if p.get("kind") == "adp_ueq")),
            "finite_difference": {
                "scheme": "bounded_two_point_absolute",
                "xyz_step_frac": float(args.diff_step if args.diff_step is not None else args.xyz_diff_step_frac),
                "adp_step_a2": float(args.diff_step if args.diff_step is not None else args.adp_diff_step_a2),
                "legacy_diff_step_shorthand_used": bool(args.diff_step is not None),
                "minimum_step": float(np.min(absolute_steps)),
                "maximum_step": float(np.max(absolute_steps)),
            },
            "accepted": block_accepted,
            "reason": block_reason,
            "objective_scale_mode": args.objective_scale_mode,
            "objective_scale_count": int(block_scale_contract["scale_count"]) if block_scale_contract is not None else None,
            "objective_scale_anchor_seconds": float(block_scale_contract.get("anchor_prediction_seconds", 0.0)) if block_scale_contract is not None else None,
            "least_squares": {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "early_stop": bool(getattr(result, "early_stop", False)),
                "early_stop_reason": str(getattr(result, "early_stop_reason", "")),
                "early_stop_eval": int(getattr(result, "early_stop_eval", 0)),
                "early_stop_metric": float(getattr(result, "early_stop_metric", float("nan"))),
            },
            "current_metric": start_metric,
            "candidate_metric": candidate_metric,
            "free_wR2_start": free_start,
            "free_wR2_candidate": free_candidate,
            "candidate_res": str(candidate_res),
            "accepted_res": str(accepted_res_block),
        }
        block_results.append(block_record)
        print(f"block {block_no:02d} {block_mode}: {block_reason}", flush=True)
        if block_accepted:
            current_model = candidate_model
            current_eval = candidate_eval
            accepted = True
            acceptance_reason = block_reason

    accepted_model = current_model
    accepted_eval = current_eval
    candidate_eval = last_candidate_eval

    accepted_res = out / "accepted_model.res"
    updated_res = out / "bloch_updated_model.res"
    update_res_coordinates(Path(args.res), accepted_res, accepted_model)
    update_res_coordinates(Path(args.res), updated_res, accepted_model)
    write_model_delta_csv(out / "model_delta.csv", base_model, accepted_model)
    # Backward-compatible name for quick inspection scripts.
    write_model_delta_csv(out / "coordinate_shifts.csv", base_model, accepted_model)
    write_native_update_trace(out / "trace.csv", trace)

    # Export a corrected-HKL endpoint only from the validation-accepted model.
    # If the coordinate step overfits, this intentionally falls back to the
    # starting model rather than exporting a worse dynamic model.
    accepted_pred = prediction_for_model(data, accepted_model, cfg, target_indices, cache_root, "accepted_endpoint", args.workers, args.chunksize)
    accepted_best = evaluate_trial(data, target, base_sig, accepted_pred, target_indices, masks, nuisance, wght_a, wght_b, scale_fit_split=args.scale_fit_split)
    endpoints = []
    for lam in parse_float_grid(args.lambda_grid):
        ep_dir = out / f"endpoint_lam_{str(lam).replace('.', 'p')}"
        endpoints.append(export_corrected_endpoint(ep_dir, data, target, accepted_best["arrays"]["sigma_eff"], accepted_best, masks, updated_res, lam, "shelx_fixed", "median-iobs", args.make_maps))
    endpoints = sorted(endpoints, key=lambda r: (r["shelxl"].get("wR2_all", 999), r["shelxl"].get("R1_all", 999)))
    baseline = copy_baseline(out / "baseline_exti", Path(args.res), Path(args.baseline_hkl)) if Path(args.baseline_hkl).exists() else {}

    summary = {
        "inputs": {
            "hkl": str(Path(args.hkl).resolve()),
            "hkl_sha256": sha256(Path(args.hkl)),
            "res": str(Path(args.res).resolve()),
            "res_sha256": sha256(Path(args.res)),
            "model_tag": model_tag,
        },
        "config": vars(args),
        "nuisance": asdict(nuisance),
        "objective_config": {
            "group_width_deg": float(objective_cfg.group_width_deg),
            "validation_group_width_deg": float(cfg.group_width_deg),
            "scale_mode": str(args.objective_scale_mode),
            "scale_mode_note": "frozen_block refreshes scales once per model-update block, then keeps them fixed for every coordinate/ADP finite-difference evaluation.",
            "scale_fit_split": str(args.scale_fit_split),
            "pets_objective_cache_only": pets_objective_cache_only,
        },
        "restraints": {
            "mode": args.restraints,
            "count": int(len(restraints)),
            "bond_restraint_sigma": float(args.bond_restraint_sigma),
            "adp_similarity_sigma": float(args.adp_similarity_sigma),
            "source": "start model inferred covalent graph",
            "path": str(out / "restraints.csv"),
            "cctbx_bridge": cctbx_bridge,
        },
        "row_counts": {
            "target_rows": int(len(target_indices)),
            "objective_rows": int(len(objective_indices)),
            "work_rows": int(np.count_nonzero(masks["work"])),
            "free_rows": int(np.count_nonzero(masks["free"])),
            "max_block_parameter_count": int(max([b.get("parameter_count", 0) for b in block_results] + [0])),
        },
        "blocks": block_results,
        "acceptance": {
            "accepted": accepted,
            "metric": args.acceptance_metric,
            "start_metric": validation_score(start_eval, args.acceptance_metric),
            "accepted_metric": validation_score(accepted_eval, args.acceptance_metric),
            "acceptance_min_delta": float(args.acceptance_min_delta),
            "free_wR2_start": validation_score(start_eval, "free_wR2"),
            "free_wR2_accepted": validation_score(accepted_eval, "free_wR2"),
            "free_wR2_tolerance": float(args.free_wR2_tolerance),
            "allow_free_worse": bool(args.allow_free_worse),
            "reason": acceptance_reason,
        },
        "start_eval": {k: v for k, v in start_eval.items() if k != "arrays"},
        "candidate_eval": {k: v for k, v in candidate_eval.items() if k != "arrays"},
        "accepted_eval": {k: v for k, v in accepted_eval.items() if k != "arrays"},
        "endpoint_results_sorted": endpoints,
        "best_endpoint": endpoints[0] if endpoints else {},
        "baseline_exti": baseline,
        "wall_seconds": time.perf_counter() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    lines = [
        "# Native Row-Parameter Regularization",
        "",
        f"Objective rows: `{len(objective_indices)}` of `{len(target_indices)}` selected rows. Blocks: `{','.join(block_schedule)}`. N/Q: `{args.n_basis}/{args.n_points}`.",
        "",
        f"This is a finite-difference coordinate regularization against `sqrt(w) * (Iobs - Idyn_scaled)`, with `{args.objective_group_width_deg:.2f}` degree objective scale groups. Objective scale mode is `{args.objective_scale_mode}`. In `frozen_block`, scales are fitted from the block-start model and held fixed during every coordinate/ADP finite-difference evaluation, then refreshed before the next block. Broad validation and corrected-HKL export use `{args.group_width_deg:.2f}` degree groups, `{args.detector_scale_bins}x{args.detector_scale_bins}` detector bins, and `{args.scale_fit_split}` rows for fitting nuisance scales.",
        f"Restraints: `{args.restraints}` with `{len(restraints)}` inferred bond restraints, bond sigma `{args.bond_restraint_sigma:.3f} A`, and bonded-Ueq sigma `{args.adp_similarity_sigma:.4f}`.",
        f"cctbx/smtbx bridge: parsed=`{cctbx_bridge.get('parsed')}`, constraints=`{cctbx_bridge.get('constraints_count')}`. The bridge uses `{cctbx_bridge.get('view')}` as a restraint-only SHELX view with sanitized SFAC rows.",
        "",
        f"Validation-gated acceptance: `{acceptance_reason}`. The corrected-HKL endpoint below was generated from the accepted model.",
        "",
        "## Regularization Blocks",
        "",
        "| block | mode | params | accepted | validation decision |",
        "|---:|---|---:|---|---|",
    ]
    for br in block_results:
        if br.get("skipped"):
            lines.append(f"| {br['block']} | {br['mode']} | 0 | skipped | {br['reason']} |")
        else:
            lines.append(f"| {br['block']} | {br['mode']} | {br['parameter_count']} | {str(br['accepted']).lower()} | {br['reason']} |")
    lines.extend([
        "",
        "## Native Validation",
        "",
        "| split | start R1all | candidate R1all | accepted R1all | start wR2all | candidate wR2all | accepted wR2all | start CC | candidate CC | accepted CC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for split in ("work", "free", "all"):
        s = start_eval["splits"][split]
        c = candidate_eval["splits"][split]
        a = accepted_eval["splits"][split]
        lines.append(
            f"| {split} | {100*s['R1_all']:.2f} | {100*c['R1_all']:.2f} | {100*a['R1_all']:.2f} | "
            f"{100*s['wR2_all']:.2f} | {100*c['wR2_all']:.2f} | {100*a['wR2_all']:.2f} | "
            f"{s['CC']:.4f} | {c['CC']:.4f} | {a['CC']:.4f} |"
        )
    lines.extend([
        "",
        "## Corrected-HKL Endpoint From Updated Model",
        "",
        "| lambda_q | R1gt % | R1all % | wR2all % | GooF | Ngt/Nall |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for ep in endpoints:
        sh = ep["shelxl"]
        lines.append(f"| {ep['lambda_q']:.2f} | {100*sh.get('R1_gt', float('nan')):.2f} | {100*sh.get('R1_all', float('nan')):.2f} | {100*sh.get('wR2_all', float('nan')):.2f} | {sh.get('GooF', float('nan')):.3f} | {sh.get('N_gt', '')}/{sh.get('N_all', '')} |")
    if baseline:
        lines.extend([
            "",
            f"Baseline EXTI rerun: `R1gt={100*baseline.get('R1_gt', float('nan')):.2f}%`, `R1all={100*baseline.get('R1_all', float('nan')):.2f}%`, `wR2all={100*baseline.get('wR2_all', float('nan')):.2f}%`, `GooF={baseline.get('GooF', float('nan')):.3f}`.",
        ])
    lines.extend([
        "",
        "## Files",
        "",
        f"Accepted RES: `{updated_res}`",
        f"Accepted model delta: `{out / 'model_delta.csv'}`",
        f"Restraints: `{out / 'restraints.csv'}`",
        f"Objective rows: `{out / 'objective_rows.csv'}`",
        f"Trace: `{out / 'trace.csv'}`",
    ])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "out_dir": str(out.resolve()),
                "report": str((out / "REPORT.md").resolve()),
                "accepted": accepted,
                "acceptance_reason": acceptance_reason,
                "best_endpoint": summary["best_endpoint"],
            },
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
