"""Enumerate the declared nuisance grid with the existing candidate policy.

Prediction/evaluation callables and result collectors are explicit inputs.
This keeps the legacy worker hooks local without runtime global rebinding.
Only an infeasible optional candidate is skipped; its exact error is retained.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict
from pathlib import Path

from grond.bloch.config import resolve_double_precision
from grond.engine.bloch_engine import Config, InfeasibleSgJitterProfileError, NonPassiveAbsorptionError

from .contracts import (
    Nuisance,
    absorption_parameter_grid,
    parse_float_grid,
    thickness_shape_grids,
    thickness_support_fields,
)

MOSAICITY_MODES = {"none", "gaussian_omega", "transverse_target"}


def mosaicity_candidate_grid(args) -> list[tuple[str, float]]:
    """Return the declared (mosaicity_mode, mosaicity_sigma_deg) candidates.

    Without --mosaicity-mode-grid/--mosaicity-sigma-deg-grid the scalar options
    are used exactly as given, so single-candidate runs keep their established
    configuration.  In a grid, mode none is one candidate with sigma 0.
    """

    mode_text = str(getattr(args, "mosaicity_mode_grid", "") or "").strip()
    sigma_text = str(getattr(args, "mosaicity_sigma_deg_grid", "") or "").strip()
    if not mode_text and not sigma_text:
        return [(str(args.mosaicity_mode), float(args.mosaicity_sigma_deg))]
    modes = [m.strip() for m in mode_text.replace(",", " ").split() if m.strip()] if mode_text else [str(args.mosaicity_mode)]
    sigmas = parse_float_grid(sigma_text) if sigma_text else [float(args.mosaicity_sigma_deg)]
    candidates: list[tuple[str, float]] = []
    for mode in modes:
        if mode not in MOSAICITY_MODES:
            raise ValueError(f"unknown mosaicity_mode {mode!r}")
        for sigma in [0.0] if mode == "none" else sigmas:
            if (mode, float(sigma)) not in candidates:
                candidates.append((mode, float(sigma)))
    return candidates


def populate_nuisance_trials(
    *,
    args,
    thickness_models,
    frame_offset_values,
    sg_jitter_modes,
    absorption_modes,
    dose_modes,
    effective_topup_d_min,
    group_width_values,
    row_plan_cache_decision,
    predict_with_cache,
    data,
    model,
    res_hash,
    indices,
    model_tag,
    sg_jitter_candidate_rejection,
    rejected_prediction_candidates,
    geom_trials,
    wght_a_values,
    wght_b_values,
    evaluate_trial,
    target,
    base_sig,
    masks,
    wght_a,
    wght_b,
    all_trials,
) -> None:
    """Append predictions, evaluated trials, and rejected-candidate records."""

    mosaicity_candidates = mosaicity_candidate_grid(args)
    support_fields = thickness_support_fields(args)
    for thickness in parse_float_grid(args.thickness_grid):
        for thickness_model in thickness_models:
            grad_values, shape_values = thickness_shape_grids(args, thickness_model)
            for grad, thickness_gamma_shape in itertools.product(grad_values, shape_values):
                for frame_offset in frame_offset_values:
                    for sg_jitter_mode in sg_jitter_modes:
                        if sg_jitter_mode not in {"none", "gaussian", "student_t"}:
                            raise ValueError(f"unknown sg_jitter_mode {sg_jitter_mode!r}")
                        jitter_sigma0_values = parse_float_grid(args.sg_jitter_sigma0_grid) if sg_jitter_mode != "none" else [0.0]
                        jitter_d_inv_values = parse_float_grid(args.sg_jitter_sigma_d_inv_grid) if sg_jitter_mode != "none" else [0.0]
                        jitter_frame_values = parse_float_grid(args.sg_jitter_sigma_frame_grid) if sg_jitter_mode != "none" else [0.0]
                        for sg_jitter_sigma0 in jitter_sigma0_values:
                            for sg_jitter_sigma_d_inv in jitter_d_inv_values:
                                for sg_jitter_sigma_frame in jitter_frame_values:
                                    for absorption_mode in absorption_modes:
                                        for abs_par in absorption_parameter_grid(args, absorption_mode):
                                            for dose_b_mode in dose_modes:
                                                if dose_b_mode not in {"none", "linear_frame"}:
                                                    raise ValueError(f"unknown dose_b_mode {dose_b_mode!r}")
                                                dose_b_values = parse_float_grid(args.dose_b_per_frame_grid) if dose_b_mode == "linear_frame" else [0.0]
                                                for dose_b_per_frame, (mosaicity_mode, mosaicity_sigma_deg) in itertools.product(
                                                    dose_b_values, mosaicity_candidates
                                                ):
                                                    cfg = Config(
                                                        n_basis=args.n_basis,
                                                        n_points=args.n_points,
                                                        sg_window=args.sg_window,
                                                        sg_window_mode=args.sg_window_mode,
                                                        sg_window_min=args.sg_window_min,
                                                        sg_window_max=args.sg_window_max,
                                                        sg_window_scale=args.sg_window_scale,
                                                        sg_window_frame_floor=args.sg_window_frame_floor,
                                                        sg_jitter_mode=sg_jitter_mode,
                                                        sg_jitter_sigma0=sg_jitter_sigma0,
                                                        sg_jitter_sigma_d_inv=sg_jitter_sigma_d_inv,
                                                        sg_jitter_sigma_frame=sg_jitter_sigma_frame,
                                                        sg_jitter_points=args.sg_jitter_points,
                                                        sg_jitter_student_df=args.sg_jitter_student_df,
                                                        sg_jitter_width_factor=args.sg_jitter_width_factor,
                                                        mosaicity_mode=mosaicity_mode,
                                                        mosaicity_sigma_deg=mosaicity_sigma_deg,
                                                        mosaicity_max_sigma_deg=args.mosaicity_max_sigma_deg,
                                                        mosaicity_points=args.mosaicity_points,
                                                        geometry_path_file=args.geometry_path,
                                                        profile_quadrature=args.profile_quadrature,
                                                        profile_accumulation=args.profile_accumulation,
                                                        topup_d_min=effective_topup_d_min,
                                                        max_support_sg=args.max_support_sg,
                                                        thickness_A=thickness,
                                                        thickness_model=thickness_model,
                                                        thickness_grad_frac=grad,
                                                        thickness_points=args.thickness_points,
                                                        thickness_gamma_shape=thickness_gamma_shape,
                                                        **support_fields,
                                                        group_width_deg=group_width_values[0],
                                                        frame_offset=frame_offset,
                                                        dose_b_mode=dose_b_mode,
                                                        dose_b_per_frame=dose_b_per_frame,
                                                        dose_b_ref_frame=args.dose_b_ref_frame,
                                                        matrix_convention=args.matrix_convention,
                                                        row_plan_cache=str(row_plan_cache_decision["resolved"]),
                                                        paired_inverse_hand=args.paired_inverse_hand,
                                                        double_precision=resolve_double_precision(
                                                            getattr(args, "double_precision", False)
                                                        ),
                                                        free_fraction=args.free_fraction,
                                                        **abs_par,
                                                    )
                                                    print(
                                                        f"predict t={thickness:g} thickness_model={thickness_model} grad={grad:g} "
                                                        + (
                                                            f"gamma_shape={thickness_gamma_shape:g} "
                                                            f"evaluator={support_fields['thickness_evaluator']} "
                                                            if thickness_model == "gamma" or support_fields["thickness_evaluator"] != "quadrature"
                                                            else ""
                                                        )
                                                        + f"frame_offset={frame_offset:g} "
                                                        f"sg_jitter={sg_jitter_mode} sigma0={sg_jitter_sigma0:g} "
                                                        f"sigma_d_inv={sg_jitter_sigma_d_inv:g} sigma_frame={sg_jitter_sigma_frame:g} "
                                                        f"abs={abs_par['absorption_mode']} mu={abs_par['absorption_mu']:g} "
                                                        f"dose_b={dose_b_mode} per_frame={dose_b_per_frame:g} "
                                                        f"mosaicity={mosaicity_mode} sigma_deg={mosaicity_sigma_deg:g}",
                                                        flush=True,
                                                    )
                                                    if mosaicity_mode == "transverse_target" and sg_jitter_mode != "none" and any(
                                                        abs(float(value)) > 0.0
                                                        for value in (sg_jitter_sigma0, sg_jitter_sigma_d_inv, sg_jitter_sigma_frame)
                                                    ):
                                                        # The re-centred cone solves each block on the untilted
                                                        # target-Sg samples; a target-Sg jitter convolution has
                                                        # no defined meaning inside it.
                                                        rejection = {
                                                            "status": "rejected",
                                                            "reason": "transverse_target_with_sg_jitter",
                                                            "message": "transverse_target mosaicity cannot be combined with target-Sg jitter",
                                                            "config": asdict(cfg),
                                                        }
                                                        rejected_prediction_candidates.append(rejection)
                                                        print(
                                                            "rejected nuisance candidate: "
                                                            f"{rejection['reason']}: {rejection['message']}",
                                                            flush=True,
                                                        )
                                                        continue
                                                    try:
                                                        pred = predict_with_cache(
                                                            data,
                                                            model,
                                                            res_hash,
                                                            Path(args.hkl),
                                                            Path(args.res),
                                                            cfg,
                                                            indices,
                                                            Path(args.prediction_cache_dir),
                                                            args.workers,
                                                            args.chunksize,
                                                            model_tag=model_tag,
                                                            cache_compression=args.cache_compression,
                                                            ug_cache_dir=Path(args.ug_cache_dir),
                                                        )
                                                    except (InfeasibleSgJitterProfileError, NonPassiveAbsorptionError) as exc:
                                                        rejection = (
                                                            sg_jitter_candidate_rejection(cfg, exc)
                                                            if isinstance(exc, InfeasibleSgJitterProfileError)
                                                            else {
                                                                "status": "rejected",
                                                                "reason": "nonpassive_absorption",
                                                                "message": str(exc),
                                                                "details": exc.details,
                                                                "config": asdict(cfg),
                                                            }
                                                        )
                                                        rejected_prediction_candidates.append(rejection)
                                                        print(
                                                            "rejected nuisance candidate: "
                                                            f"{rejection['reason']}: {rejection['message']}",
                                                            flush=True,
                                                        )
                                                        continue
                                                    geom_trials.append(
                                                        {
                                                            "cfg": cfg,
                                                            "pred": pred,
                                                            "thickness": thickness,
                                                            "thickness_model": thickness_model,
                                                            "grad": grad,
                                                            "thickness_gamma_shape": thickness_gamma_shape,
                                                            **support_fields,
                                                            "frame_offset": frame_offset,
                                                            "sg_jitter_mode": sg_jitter_mode,
                                                            "sg_jitter_sigma0": sg_jitter_sigma0,
                                                            "sg_jitter_sigma_d_inv": sg_jitter_sigma_d_inv,
                                                            "sg_jitter_sigma_frame": sg_jitter_sigma_frame,
                                                            "dose_b_mode": dose_b_mode,
                                                            "dose_b_per_frame": dose_b_per_frame,
                                                            "mosaicity_mode": mosaicity_mode,
                                                            "mosaicity_sigma_deg": mosaicity_sigma_deg,
                                                            **abs_par,
                                                        }
                                                    )
                                                    sigma_a_values = parse_float_grid(args.sigma_a_grid)
                                                    sigma_b_values = parse_float_grid(args.sigma_b_corr_grid)
                                                    sigma_c_values = parse_float_grid(args.sigma_c_peak_grid)
                                                    row_b_values = parse_float_grid(args.row_resolution_b_grid)
                                                    for grid_values in itertools.product(
                                                        group_width_values,
                                                        sigma_a_values,
                                                        sigma_b_values,
                                                        sigma_c_values,
                                                        row_b_values,
                                                        wght_a_values,
                                                        wght_b_values,
                                                    ):
                                                        group_width, sig_a, sig_b, sig_c, row_resolution_b, native_wght_a, native_wght_b = grid_values
                                                        nui = Nuisance(
                                                            thickness,
                                                            grad,
                                                            frame_offset,
                                                            group_width,
                                                            0,  # detector_scale_bins: no detector scale field
                                                            sig_a,
                                                            sig_b,
                                                            sig_c,
                                                            thickness_model=thickness_model,
                                                            sg_jitter_mode=sg_jitter_mode,
                                                            sg_jitter_sigma0=sg_jitter_sigma0,
                                                            sg_jitter_sigma_d_inv=sg_jitter_sigma_d_inv,
                                                            sg_jitter_sigma_frame=sg_jitter_sigma_frame,
                                                            sg_jitter_points=args.sg_jitter_points,
                                                            sg_jitter_student_df=args.sg_jitter_student_df,
                                                            sg_jitter_width_factor=args.sg_jitter_width_factor,
                                                            mosaicity_mode=mosaicity_mode,
                                                            mosaicity_sigma_deg=mosaicity_sigma_deg,
                                                            mosaicity_points=args.mosaicity_points,
                                                            geometry_path_file=args.geometry_path,
                                                            dose_b_mode=dose_b_mode,
                                                            dose_b_per_frame=dose_b_per_frame,
                                                            dose_b_ref_frame=args.dose_b_ref_frame,
                                                            matrix_convention=args.matrix_convention,
                                                            row_resolution_b=row_resolution_b,
                                                            native_wght_a=native_wght_a,
                                                            native_wght_b=native_wght_b,
                                                            thickness_gamma_shape=thickness_gamma_shape,
                                                            **support_fields,
                                                            **abs_par,
                                                        )
                                                        ev = evaluate_trial(data, target, base_sig, pred, indices, masks, nui, wght_a, wght_b, scale_fit_split=args.scale_fit_split)
                                                        ev["prediction_cache"] = {"status": pred.get("cache_status"), "path": pred.get("cache_path")}
                                                        all_trials.append(ev)
