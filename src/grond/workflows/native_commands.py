"""Build the native row-calculation command with the complete pinned physics contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import FULL_Q_LAMBDA
from grond.io.config import DatasetSpec
from grond.paths import engine_script
from grond.workflows.crystal_parameters import crystal_arguments
from grond.workflows.nuisance_policy import (
    effective_topup_d_min,
    grid_from_nuisance,
    prediction_cache_dir,
    resolved_post_bloch_sigma_policy,
    resolved_row_plan_cache,
    resolved_sg_window_mode,
)


def native_stage_cmd(
    python: str,
    ds: DatasetSpec,
    structure: str,
    baseline_hkl: str,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    n_basis: int,
    n_points: int,
    stage: str,
    pinned: dict | None = None,
    alternate_hand: bool = False,
    make_maps: bool = True,
    endpoint: bool = True,
    max_rows_override: int | None = None,
    workers_override: int | None = None,
    sg_window_override: float | None = None,
    sg_window_mode_override: str | None = None,
    scale_fit_split_override: str | None = None,
    selection_metric_override: str | None = None,
    strict_pinned_contract: bool = False,
    allow_pinned_jitter_fallback: bool = True,
    promotion_baseline: Mapping[str, object] | None = None,
    paired_inverse_hand: bool = False,
) -> list[str]:
    """Build one native nuisance-regularization command for the vendored engine."""

    from grond.cli_support.tools import default_shelxl_executable
    from grond.io.structure_contract import is_shelx_model

    args = crystal_arguments(args, ds.name)
    fixed_thickness = getattr(args, "fixed_thickness", {})
    pinned = {**pinned, **fixed_thickness} if pinned else {}
    n_basis = getattr(args, "fixed_n_basis", None) or n_basis
    n_points = getattr(args, "fixed_n_points", None) or n_points
    fast = bool(args.fast)
    endpoint_ok = bool(endpoint and is_shelx_model(structure) and not args.native_only)
    lambda_grid = FULL_Q_LAMBDA if endpoint_ok else ""
    endpoint_b_grid = args.endpoint_b_grid if endpoint_ok and not fast else "0.0"
    endpoint_exti_grid = args.endpoint_exti_grid if endpoint_ok and not fast else "off"
    shelxl_iterations = args.shelxl_iterations if endpoint_ok and not fast else 1
    finalish_stage = stage in {"final", "noH", "hand", "adaptive_final"} or stage.endswith("_final")
    post_bloch_sigma_policy = resolved_post_bloch_sigma_policy(
        args, final_endpoint=bool(endpoint_ok and finalish_stage and not fast)
    )
    thickness_grid = (
        grid_from_nuisance(pinned, "thickness_A", "1200")
        if pinned
        else ("1000,1200,1400" if fast else args.thickness_grid)
    )
    grad_grid = (
        grid_from_nuisance(pinned, "thickness_grad_frac", "0.4")
        if pinned
        else ("0.0,0.4" if fast else args.thickness_grad_grid)
    )
    frame_grid = (
        grid_from_nuisance(pinned, "frame_offset", "0.0") if pinned else ("0.0" if fast else args.frame_offset_grid)
    )
    group_grid = (
        grid_from_nuisance(pinned, "group_width_deg", "1.0")
        if pinned
        else ("frame,2frame" if fast else args.group_width_grid)
    )
    sigma_a = grid_from_nuisance(pinned, "sigma_a", "0.0") if pinned else ("0.0" if fast else args.sigma_a_grid)
    sigma_b = grid_from_nuisance(pinned, "sigma_b_corr", "0.0") if pinned else "0.0"
    sigma_c = (
        grid_from_nuisance(pinned, "sigma_c_peak", "0.0") if pinned else ("0.0" if fast else args.sigma_c_peak_grid)
    )
    row_resolution_b = (
        grid_from_nuisance(pinned, "row_resolution_b", "0.0")
        if pinned
        else ("0.0" if fast else args.row_resolution_b_grid)
    )
    detector_scale_bins = (
        grid_from_nuisance(pinned, "detector_scale_bins", "0")
        if pinned
        else ("0" if fast else args.detector_scale_bins)
    )
    scale_fit_split = scale_fit_split_override or ("all" if args.production_scales else "work")
    if scale_fit_split not in {"work", "all"}:
        raise ValueError(f"invalid scale-fit split override {scale_fit_split!r}")
    pinned_jitter_mode = grid_from_nuisance(pinned, "sg_jitter_mode", "none") if pinned else ""
    jitter_mode_grid = pinned_jitter_mode if pinned else ("none" if fast else args.sg_jitter_mode_grid)
    if (
        pinned
        and allow_pinned_jitter_fallback
        and str(getattr(args, "pinned_sg_jitter_fallback", "none")) == "none"
        and pinned_jitter_mode.strip().lower() != "none"
    ):
        # A previously selected convolution can become geometrically
        # infeasible after a resolution/model change. Keep the strict local
        # root bound and score the ordinary no-jitter physics as an explicit
        # fallback candidate instead of widening, truncating, or crashing.
        jitter_mode_grid = f"{pinned_jitter_mode},none"
    cmd = [
        python,
        str(engine_script("native_row_regularize.py")),
        "--hkl",
        ds.hkl,
        "--res",
        structure,
        "--baseline-hkl",
        baseline_hkl or str(out_dir / "NO_BASELINE_HKL_SUPPLIED.hkl"),
        f"--hkl-transform={ds.hkl_transform}",
        "--out-dir",
        str(out_dir),
        "--n-basis",
        str(n_basis),
        "--n-points",
        str(n_points),
        "--sg-window",
        f"{float(args.sg_window if sg_window_override is None else sg_window_override):g}",
        "--sg-window-mode",
        sg_window_mode_override or resolved_sg_window_mode(args, ds),
        "--sg-window-min",
        f"{args.sg_window_min:g}",
        "--sg-window-max",
        f"{args.sg_window_max:g}",
        "--sg-window-scale",
        f"{args.sg_window_scale:g}",
        "--sg-window-frame-floor",
        f"{args.sg_window_frame_floor:g}",
        "--min-d-spacing",
        f"{ds.min_d_spacing:g}",
        "--min-peak",
        f"{ds.min_peak:g}",
        "--geometry-percentile",
        f"{ds.geometry_percentile:g}",
        "--target-mode",
        ds.target_mode,
        "--thickness-grid",
        thickness_grid,
        "--thickness-model-grid",
        grid_from_nuisance(pinned, "thickness_model", "uniform_gradient") if pinned else args.thickness_model_grid,
        "--thickness-grad-grid",
        grad_grid,
        "--thickness-sigma-frac-grid",
        grid_from_nuisance(pinned, "thickness_sigma_frac", "0.0") if pinned else "0.0",
        f"--frame-offset-grid={frame_grid}",
        "--group-width-grid",
        group_grid,
        "--detector-scale-bins",
        detector_scale_bins,
        "--detector-scale-ridge",
        str(getattr(args, "detector_scale_ridge", 0.05)),
        "--detector-scale-smoothness",
        str(getattr(args, "detector_scale_smoothness", 0.10)),
        "--detector-scale-min-rows",
        str(getattr(args, "detector_scale_min_rows", 20)),
        "--sigma-a-grid",
        sigma_a,
        "--sigma-b-corr-grid",
        sigma_b,
        "--sigma-c-peak-grid",
        sigma_c,
        "--sg-jitter-mode-grid",
        jitter_mode_grid,
        "--sg-jitter-sigma0-grid",
        grid_from_nuisance(pinned, "sg_jitter_sigma0", "0.0")
        if pinned
        else ("0.0" if fast else args.sg_jitter_sigma0_grid),
        "--sg-jitter-sigma-d-inv-grid",
        grid_from_nuisance(pinned, "sg_jitter_sigma_d_inv", "0.0")
        if pinned
        else ("0.0" if fast else args.sg_jitter_sigma_d_inv_grid),
        "--sg-jitter-sigma-frame-grid",
        grid_from_nuisance(pinned, "sg_jitter_sigma_frame", "0.0") if pinned else "0.0",
        "--sg-jitter-points",
        str(int(float(pinned.get("sg_jitter_points", args.sg_jitter_points if not fast else 1)))),
        "--profile-quadrature",
        args.profile_quadrature,
        "--profile-accumulation",
        args.profile_accumulation,
        "--absorption-mode-grid",
        grid_from_nuisance(pinned, "absorption_mode", "none")
        if pinned
        else ("none" if fast else args.absorption_mode_grid),
        "--absorption-mu-grid",
        grid_from_nuisance(pinned, "absorption_mu", "0.0") if pinned else ("0.0" if fast else args.absorption_mu_grid),
        "--absorption-frac-grid",
        grid_from_nuisance(pinned, "absorption_frac", "0.0")
        if pinned
        else ("0.0" if fast else args.absorption_frac_grid),
        "--absorption-mu-g2-grid",
        grid_from_nuisance(pinned, "absorption_mu_g2", "0.0")
        if pinned
        else ("0.0" if fast else args.absorption_mu_g2_grid),
        "--absorption-mu-theta2-grid",
        grid_from_nuisance(pinned, "absorption_mu_theta2", "0.0") if pinned else "0.0",
        "--absorption-tds-b-grid",
        grid_from_nuisance(pinned, "absorption_tds_b", "0.0") if pinned else "0.0",
        "--dose-b-mode-grid",
        grid_from_nuisance(pinned, "dose_b_mode", "none") if pinned else ("none" if fast else args.dose_b_mode_grid),
        "--dose-b-per-frame-grid",
        grid_from_nuisance(pinned, "dose_b_per_frame", "0.0")
        if pinned
        else ("0.0" if fast else args.dose_b_per_frame_grid),
        "--row-resolution-b-grid",
        row_resolution_b,
        "--thickness-points",
        str(args.final_thickness_points if (pinned and finalish_stage) else (3 if fast else args.thickness_points)),
        "--topup-d-min",
        f"{effective_topup_d_min(args, ds):g}",
        "--max-support-sg",
        f"{args.max_support_sg:g}",
        "--free-fraction",
        f"{args.free_fraction:g}",
        "--split-seed",
        str(args.split_seed),
        "--scale-fit-split",
        scale_fit_split,
        "--selection-metric",
        selection_metric_override or args.selection_metric,
        "--lambda-grid",
        lambda_grid,
        "--scale-smoothness",
        str(args.scale_smoothness),
        "--endpoint-b-grid",
        endpoint_b_grid,
        "--endpoint-exti-grid",
        endpoint_exti_grid,
        "--shelxl",
        str(getattr(args, "shelxl", "") or default_shelxl_executable()),
        "--shelxl-iterations",
        str(shelxl_iterations),
        "--endpoint-wght-mode",
        str(getattr(args, "endpoint_wght_mode", "freeze-zero")),
        "--post-bloch-sigma-policy",
        post_bloch_sigma_policy,
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{args.post_bloch_sigma_min_scale:g}",
        "--post-bloch-sigma-max-scale",
        f"{args.post_bloch_sigma_max_scale:g}",
        "--workers",
        str(args.workers if workers_override is None else max(1, int(workers_override))),
        "--chunksize",
        str(args.chunksize),
        "--prediction-cache-dir",
        str(prediction_cache_dir(args, ds, stage)),
        "--cache-compression",
        args.prediction_cache_compression,
        "--compute-backend",
        args.compute_backend,
        "--row-plan-cache",
        resolved_row_plan_cache(args, fixed_model=True),
        "--jax-batch-size",
        str(args.jax_batch_size),
        "--jax-matrix-bucket-size",
        str(args.jax_matrix_bucket_size),
        "--bethe-strong-count",
        str(args.bethe_strong_count),
        "--bethe-variant",
        args.bethe_variant,
        "--bethe-promote-denom",
        f"{args.bethe_promote_denom:g}",
        "--bethe-denom-eps",
        f"{args.bethe_denom_eps:g}",
        "--matrix-convention",
        str(pinned.get("matrix_convention", args.matrix_convention))
        if strict_pinned_contract
        else args.matrix_convention,
        "--hkl-float-format",
        "shelx_fixed",
    ]
    # Apply these last: a thickness neighbourhood or feature scout must not
    # reopen a parameter the caller explicitly fixed.
    for field, option in (
        ("thickness_A", "--thickness-grid"),
        ("thickness_model", "--thickness-model-grid"),
        ("thickness_grad_frac", "--thickness-grad-grid"),
        ("thickness_sigma_frac", "--thickness-sigma-frac-grid"),
    ):
        if field in fixed_thickness:
            cmd[cmd.index(option) + 1] = grid_from_nuisance(fixed_thickness, field, "")
    if strict_pinned_contract:
        cmd.extend(
            [
                "--sg-jitter-student-df",
                grid_from_nuisance(pinned, "sg_jitter_student_df", "4.0"),
                "--sg-jitter-width-factor",
                grid_from_nuisance(pinned, "sg_jitter_width_factor", "3.0"),
                "--dose-b-ref-frame",
                grid_from_nuisance(pinned, "dose_b_ref_frame", "nan"),
                "--wght-a-grid",
                grid_from_nuisance(pinned, "native_wght_a", "") if "native_wght_a" in pinned else "",
                "--wght-b-grid",
                grid_from_nuisance(pinned, "native_wght_b", "") if "native_wght_b" in pinned else "",
            ]
        )
    if promotion_baseline:
        cmd.extend(
            [
                "--promotion-baseline-json",
                json.dumps(dict(promotion_baseline), sort_keys=True, allow_nan=True),
                "--promotion-min-gain",
                str(float(args.continuous_thickness_min_gain)),
                "--promotion-max-free-wr2-loss",
                str(float(args.continuous_thickness_max_free_wr2_loss)),
                "--promotion-max-all-r1-loss",
                str(float(args.continuous_thickness_max_all_r1_loss)),
            ]
        )
    if ds.include_nonpositive:
        cmd.append("--include-nonpositive")
    else:
        cmd.append("--positive-only")
    max_rows = args.max_rows if max_rows_override is None else max_rows_override
    if max_rows:
        cmd += ["--max-rows", str(max_rows)]
    row_contracts = getattr(args, "dataset_row_contracts", {})
    row_contract = row_contracts.get(ds.name, "") if isinstance(row_contracts, Mapping) else ""
    if row_contract:
        cmd += ["--row-contract", str(row_contract)]
    if ds.split_contract:
        cmd += ["--split-contract", str(ds.split_contract)]
    if alternate_hand:
        cmd.append("--alternate-hand")
    if paired_inverse_hand:
        cmd.append("--paired-inverse-hand")
    if make_maps and endpoint_ok:
        cmd.append("--make-maps")
    if not endpoint_ok:
        cmd.append("--skip-endpoint-refinement")
    return cmd
