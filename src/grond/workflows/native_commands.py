"""Build the native row-calculation command with the complete pinned physics contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import FULL_Q_LAMBDA
from grond.io.config import DatasetSpec
from grond.paths import engine_script
from grond.regularize.contracts import thickness_distribution_options
from grond.validation.heldout import DEFAULT_FREE_SALT
from grond.workflows.crystal_parameters import crystal_arguments
from grond.workflows.endpoint_commands import endpoint_handoff_args
from grond.workflows.nuisance_policy import (
    effective_topup_d_min,
    grid_from_nuisance,
    grid_has_variation,
    nuisance_grid,
    prediction_cache_dir,
    resolved_post_bloch_sigma_policy,
    resolved_row_plan_cache,
    resolved_sg_window_max,
    resolved_sg_window_mode,
)
from grond.workflows.scale_model_policy import scale_model_cli_arguments

# Discrete thickness promotion: minimum work-row score gain and the work-row R1 no-harm limit.
THICKNESS_PROMOTION_MIN_GAIN = 5e-05
THICKNESS_PROMOTION_MAX_ALL_R1_LOSS = 0.001


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
    trial_rows_npz: Path | None = None,
) -> list[str]:
    """Build one native nuisance-regularization command for the engine."""

    from grond.cli_support.tools import default_shelxl_executable
    from grond.io.structure_contract import is_shelx_model

    args = crystal_arguments(args, ds.name)
    fixed_thickness = getattr(args, "fixed_thickness", {})
    pinned = {**pinned, **fixed_thickness} if pinned else {}
    n_basis = getattr(args, "fixed_n_basis", None) or n_basis
    n_points = getattr(args, "fixed_n_points", None) or n_points
    if n_points is None or isinstance(n_points, bool) or int(n_points) < 1:
        raise ValueError(f"native stage {stage!r} for {ds.name} needs a resolved positive n_points, got {n_points!r}")
    endpoint_ok = bool(endpoint and is_shelx_model(structure) and not getattr(args, "native_only", False))
    lambda_grid = FULL_Q_LAMBDA if endpoint_ok else ""
    endpoint_b_grid = args.endpoint_b_grid if endpoint_ok else "0.0"
    endpoint_exti_grid = args.endpoint_exti_grid if endpoint_ok else "off"
    shelxl_iterations = args.shelxl_iterations if endpoint_ok else 1
    finalish_stage = stage in {"final", "noH", "hand", "adaptive_final"} or stage.endswith("_final")
    post_bloch_sigma_policy = resolved_post_bloch_sigma_policy(
        args, final_endpoint=bool(endpoint_ok and finalish_stage)
    )
    thickness_grid = (
        grid_from_nuisance(pinned, "thickness_A", "1200")
        if pinned
        else args.thickness_grid
    )
    grad_grid = (
        grid_from_nuisance(pinned, "thickness_grad_frac", "0.4")
        if pinned
        else args.thickness_grad_grid
    )
    frame_grid = (
        grid_from_nuisance(pinned, "frame_offset", "0.0") if pinned else nuisance_grid(args, "frame_offset_grid")
    )
    group_grid = (
        grid_from_nuisance(pinned, "group_width_deg", "1.0")
        if pinned
        else nuisance_grid(args, "group_width_grid")
    )
    sigma_a = grid_from_nuisance(pinned, "sigma_a", "0.0") if pinned else nuisance_grid(args, "sigma_a_grid")
    sigma_b = grid_from_nuisance(pinned, "sigma_b_corr", "0.0") if pinned else "0.0"
    sigma_c = (
        grid_from_nuisance(pinned, "sigma_c_peak", "0.0") if pinned else nuisance_grid(args, "sigma_c_peak_grid")
    )
    row_resolution_b = (
        grid_from_nuisance(pinned, "row_resolution_b", "0.0")
        if pinned
        else nuisance_grid(args, "row_resolution_b_grid")
    )
    # hand-check replays the recorded split of its source stage (production_scales); AUTO fits on work rows.
    scale_fit_split = scale_fit_split_override or ("all" if getattr(args, "production_scales", False) else "work")
    if scale_fit_split not in {"work", "all"}:
        raise ValueError(f"invalid scale-fit split override {scale_fit_split!r}")
    pinned_jitter_mode = grid_from_nuisance(pinned, "sg_jitter_mode", "none") if pinned else ""
    jitter_mode_grid = pinned_jitter_mode if pinned else nuisance_grid(args, "sg_jitter_mode_grid")
    if (
        pinned
        and allow_pinned_jitter_fallback
        and pinned_jitter_mode.strip().lower() != "none"
    ):
        # A previously selected convolution can become geometrically
        # infeasible after a resolution/model change. Keep the strict local
        # root bound and score the ordinary no-jitter physics as an explicit
        # fallback candidate instead of widening, truncating, or crashing.
        jitter_mode_grid = f"{pinned_jitter_mode},none"
    # An unpinned (non-adaptive) grid searches dose only on an explicit per-frame grid; the
    # total-B-over-the-scan grid is converted per crystal by the dose feature gate.
    unpinned_dose_grid = str(nuisance_grid(args, "dose_b_per_frame_grid") or "").strip()
    # The mosaic cone is part of the pinned physics: one candidate is forwarded as the
    # engine's scalar options (recorded verbatim in its config), a feature grid as grids.
    mosaicity_modes = grid_from_nuisance(pinned, "mosaicity_mode", "none") if pinned else "none"
    mosaicity_sigmas = grid_from_nuisance(pinned, "mosaicity_sigma_deg", "0.0") if pinned else "0.0"
    if grid_has_variation(mosaicity_modes) or grid_has_variation(mosaicity_sigmas):
        mosaicity_options = ["--mosaicity-mode-grid", mosaicity_modes, "--mosaicity-sigma-deg-grid", mosaicity_sigmas]
    else:
        mosaicity_options = ["--mosaicity-mode", mosaicity_modes, "--mosaicity-sigma-deg", mosaicity_sigmas]
    if any(mode != "none" for mode in mosaicity_modes.replace(",", " ").split()):
        # Without a cone the engine keeps its own quadrature default.
        points = int(float(pinned.get("mosaicity_points", getattr(args, "mosaicity_points", 3))))
        mosaicity_options += ["--mosaicity-points", str(points)]
    mosaicity_options += ["--mosaicity-max-sigma-deg", f"{float(getattr(args, 'mosaicity_max_sigma_deg', 0.5)):g}"]
    for field in ("mosaicity_support_sigma_deg", "mosaicity_support_points", "mosaicity_support_width_sigma"):
        value = pinned.get(field, getattr(args, field, None))
        if value is not None:
            mosaicity_options += ["--" + field.replace("_", "-"), str(value)]
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
        f"{resolved_sg_window_max(args, ds):g}",
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
        grid_from_nuisance(pinned, "thickness_model", "uniform_gradient") if pinned else nuisance_grid(args, "thickness_model_grid"),
        "--thickness-grad-grid",
        grad_grid,
        "--thickness-sigma-frac-grid",
        grid_from_nuisance(pinned, "thickness_sigma_frac", "0.0") if pinned else "0.0",
        f"--frame-offset-grid={frame_grid}",
        "--group-width-grid",
        group_grid,
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
        else nuisance_grid(args, "sg_jitter_sigma0_grid"),
        "--sg-jitter-sigma-d-inv-grid",
        grid_from_nuisance(pinned, "sg_jitter_sigma_d_inv", "0.0")
        if pinned
        else nuisance_grid(args, "sg_jitter_sigma_d_inv_grid"),
        "--sg-jitter-sigma-frame-grid",
        grid_from_nuisance(pinned, "sg_jitter_sigma_frame", "0.0") if pinned else "0.0",
        "--sg-jitter-points",
        str(int(float(pinned.get("sg_jitter_points", args.sg_jitter_points)))),
        "--profile-coordinate",
        str(getattr(args, "profile_coordinate", "target_sg") or "target_sg"),
        "--profile-normalization",
        str(getattr(args, "profile_normalization", "legacy") or "legacy"),
        "--profile-angle-max-half-width-deg",
        str(getattr(args, "profile_angle_max_half_width_deg", 2.0) or 2.0),
        "--profile-basis-mode",
        str(getattr(args, "profile_basis_mode", "center") or "center"),
        "--profile-quadrature",
        args.profile_quadrature,
        "--profile-accumulation",
        args.profile_accumulation,
        "--absorption-mode-grid",
        grid_from_nuisance(pinned, "absorption_mode", "none")
        if pinned
        else nuisance_grid(args, "absorption_mode_grid"),
        "--absorption-mu-grid",
        grid_from_nuisance(pinned, "absorption_mu", "0.0") if pinned else nuisance_grid(args, "absorption_mu_grid"),
        "--absorption-frac-grid",
        grid_from_nuisance(pinned, "absorption_frac", "0.0")
        if pinned
        else nuisance_grid(args, "absorption_frac_grid"),
        "--absorption-mu-g2-grid",
        grid_from_nuisance(pinned, "absorption_mu_g2", "0.0")
        if pinned
        else nuisance_grid(args, "absorption_mu_g2_grid"),
        "--absorption-mu-theta2-grid",
        grid_from_nuisance(pinned, "absorption_mu_theta2", "0.0") if pinned else "0.0",
        "--absorption-tds-b-grid",
        grid_from_nuisance(pinned, "absorption_tds_b", "0.0") if pinned else "0.0",
        "--dose-b-mode-grid",
        grid_from_nuisance(pinned, "dose_b_mode", "none")
        if pinned
        else ("none" if not unpinned_dose_grid else nuisance_grid(args, "dose_b_mode_grid")),
        "--dose-b-per-frame-grid",
        grid_from_nuisance(pinned, "dose_b_per_frame", "0.0")
        if pinned
        else (unpinned_dose_grid if unpinned_dose_grid else "0.0"),
        "--row-resolution-b-grid",
        row_resolution_b,
        *mosaicity_options,
        "--thickness-points",
        str(args.final_thickness_points if (pinned and finalish_stage) else args.thickness_points),
        "--topup-d-min",
        f"{effective_topup_d_min(args, ds):g}",
        "--max-support-sg",
        f"{args.max_support_sg:g}",
        "--free-fraction",
        f"{args.free_fraction:g}",
        "--split-seed",
        str(args.split_seed),
        "--split-scheme",
        str(getattr(args, "split_scheme", "legacy")),
        "--split-salt",
        str(getattr(args, "split_salt", DEFAULT_FREE_SALT)),
        "--scale-fit-split",
        scale_fit_split,
        "--selection-metric",
        selection_metric_override or args.selection_metric,
        "--lambda-grid",
        lambda_grid,
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
        *endpoint_handoff_args(args),
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
        "--row-plan-cache",
        resolved_row_plan_cache(args, fixed_model=True),
        "--matrix-convention",
        str(pinned.get("matrix_convention", args.matrix_convention))
        if strict_pinned_contract
        else args.matrix_convention,
        "--hkl-float-format",
        "shelx_fixed",
    ]
    if endpoint_ok and str(getattr(args, "endpoint_wght_mode", "freeze-zero")) == "converge":
        # converge calls are L.S. 10 (the combined endpoints' --shelxl-feedback-endpoint-ls-cycles), not the
        # engine's one-cycle default, so each call's shift/esd test and weight suggestion see a converged model.
        cmd += ["--endpoint-ls-cycles", str(int(getattr(args, "shelxl_feedback_endpoint_ls_cycles", 10) or 10))]
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
    if pinned:
        # A pinned support thickness distribution (gamma, table/cf/auto) travels with its shape, evaluator and
        # support; the legacy quadrature adds nothing, so legacy commands are unchanged.
        cmd.extend(thickness_distribution_options(pinned))
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
        # Held-out free rows: the composite score and the R1 veto read work rows only, and the free-row no-harm
        # veto is off (inf).
        cmd.extend(
            [
                "--promotion-baseline-json",
                json.dumps(dict(promotion_baseline), sort_keys=True, allow_nan=True),
                "--promotion-min-gain",
                str(THICKNESS_PROMOTION_MIN_GAIN),
                "--promotion-max-free-wr2-loss",
                "inf",
                "--promotion-max-all-r1-loss",
                str(THICKNESS_PROMOTION_MAX_ALL_R1_LOSS),
                "--promotion-score-rows",
                "work",
            ]
        )
    cmd += scale_model_cli_arguments(args)
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
    if str(getattr(args, "exclude_orbits_json", "") or "").strip():
        cmd += ["--exclude-orbits-json", str(Path(args.exclude_orbits_json).expanduser().resolve())]
    if alternate_hand:
        cmd.append("--alternate-hand")
    if paired_inverse_hand:
        cmd.append("--paired-inverse-hand")
    if trial_rows_npz is not None:
        cmd += ["--trial-rows-npz", str(trial_rows_npz)]
    if make_maps and endpoint_ok:
        cmd.append("--make-maps")
    if not endpoint_ok:
        cmd.append("--skip-endpoint-refinement")
    return cmd
