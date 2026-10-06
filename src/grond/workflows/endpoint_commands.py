"""Build coordinate-update and SHELXL-feedback commands with explicit numerical settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from grond.cli_support.values import FULL_Q_LAMBDA
from grond.io.config import DatasetSpec
from grond.paths import engine_script
from grond.workflows.nuisance_policy import (
    effective_topup_d_min,
    error_model_sigma_active,
    grid_from_nuisance,
    refuse_unrepresented_structure_update_physics,
    resolved_feedback_wght_mode,
    resolved_final_n_points,
    resolved_post_bloch_sigma_policy,
    resolved_row_plan_cache,
    resolved_sg_window_max,
    resolved_sg_window_mode,
    resolved_structure_n_points,
)
from grond.workflows.split_args import split_scheme_args


def endpoint_intensity_rescale(args: argparse.Namespace) -> str:
    """Engine ``--rescale`` value for the AUTO ``--endpoint-intensity-scale`` choice (legacy when absent)."""

    requested = str(getattr(args, "endpoint_intensity_scale", "median-iobs") or "median-iobs")
    if requested not in {"absolute", "median-iobs"}:
        raise ValueError(f"unknown --endpoint-intensity-scale {requested!r}")
    return requested


def endpoint_handoff_args(args: argparse.Namespace) -> list[str]:
    """Engine options of the dynamical SHELXL handoff shared by combine, native and feedback-loop stages.

    Missing attributes resolve to the legacy contract so that partial namespaces (continuations, tests)
    never pick up a new behaviour implicitly; the AUTO parser supplies the new defaults explicitly.
    """

    cmd = [
        "--shelxl-convergence-shift-esd",
        f"{float(getattr(args, 'shelxl_convergence_shift_esd', 0.0) or 0.0):g}",
        "--shelxl-npd-recovery",
        str(getattr(args, "shelxl_npd_recovery", "damp") or "damp"),
        "--endpoint-q-policy",
        str(getattr(args, "endpoint_q_policy", "clip") or "clip"),
        "--rescale",
        endpoint_intensity_rescale(args),
        "--endpoint-include-free-rows"
        if bool(getattr(args, "endpoint_include_free_rows", True))
        else "--no-endpoint-include-free-rows",
    ]
    if bool(getattr(args, "shelxl_carry_damp", False)):
        cmd.append("--keep-damp")
    return cmd + endpoint_native_output_args(args)


def endpoint_native_output_args(args: argparse.Namespace, *, final: bool = False) -> list[str]:
    """Engine options of the SHELXL-native outputs shared by every endpoint writer.

    The converge call cap (when an endpoint or macro-cycle rule is converge), the error-model refit count
    (under the error model), the resolved cell e.s.d.s (ZERR), the Olex2/SHELXLE input cleanup (no stale
    Q-peaks, LIST 6) and, for ``final`` endpoints only, ACTA.  An option is emitted only when it acts, so
    missing attributes and the legacy flags add no option to the engine commands.
    """

    cmd: list[str] = []
    if "converge" in {str(getattr(args, "endpoint_wght_mode", "freeze-zero")), resolved_feedback_wght_mode(args)}:
        cmd += ["--wght-converge-max-calls", str(int(getattr(args, "wght_converge_max_calls", 20) or 20))]
    if error_model_sigma_active(args) or str(getattr(args, "post_bloch_sigma_policy", "auto")) == "error_model":
        cmd += ["--error-model-refits", str(int(getattr(args, "error_model_refits", 1)))]
    if error_model_sigma_active(args) and str(getattr(args, "error_model_dataset_weight", "apply")) == "off":
        cmd += ["--error-model-dataset-weight", "off"]
    zerr = str(getattr(args, "resolved_zerr_esds", "") or "").strip()
    if zerr:
        cmd += ["--zerr-esds", zerr]
    if str(getattr(args, "shelx_ins_cleanup", "off") or "off") == "on":
        cmd += ["--strip-q-peaks", "--endpoint-list-code", "6"]
    if final and bool(getattr(args, "shelx_acta", False)):
        cmd.append("--acta")
    return cmd


def legacy_handoff_conflicts(args: argparse.Namespace) -> list[str]:
    """Requested handoff settings that a legacy-contract corrected-HKL writer cannot honour.

    The endpoint-calibration replay (scale_contract_replay.py) and the observation-polish endpoint write
    their own corrected HKL with the legacy contract: every row including split=free, the median-IOBS
    scale, no correction-ratio exclusion, the DAMP ladder and a fixed SHELXL call count.  Both can become
    the publication endpoint, so under the default handoff they would silently republish free rows and an
    unconverged, per-cycle-scaled refinement.  Missing attributes resolve to the legacy contract.
    """

    conflicts = []
    if not bool(getattr(args, "endpoint_include_free_rows", True)):
        conflicts.append("--no-endpoint-include-free-rows (the legacy HKL contains every row, free rows included)")
    if endpoint_intensity_rescale(args) != "median-iobs":
        conflicts.append("--endpoint-intensity-scale absolute (the legacy writer only has the median-IOBS scale)")
    if str(getattr(args, "endpoint_q_policy", "clip") or "clip") != "clip":
        conflicts.append("--endpoint-q-policy exclude (the legacy writer keeps every correction ratio)")
    if str(getattr(args, "shelxl_npd_recovery", "damp") or "damp") != "damp":
        conflicts.append("--shelxl-npd-recovery isor (the legacy writer uses the DAMP ladder)")
    if float(getattr(args, "shelxl_convergence_shift_esd", 0.0) or 0.0) > 0.0:
        conflicts.append("--shelxl-convergence-shift-esd > 0 (the legacy writer runs a fixed SHELXL call count)")
    if str(getattr(args, "dataset_sigma_policy", "auto") or "auto") == "error_model":
        conflicts.append("--dataset-sigma-policy error_model (the legacy writer only has uniform_z_rms sigmas)")
    if str(getattr(args, "endpoint_wght_mode", "freeze-zero") or "freeze-zero") == "converge":
        conflicts.append("--endpoint-wght-mode converge (the legacy writer has no converged-weight loop)")
    return conflicts


def structure_cmd(
    python: str, ds: DatasetSpec, structure: str, baseline_hkl: str, out_dir: Path, args: argparse.Namespace, nui: dict
) -> list[str]:
    """Build the optional restrained structure-update command."""

    refuse_unrepresented_structure_update_physics(ds.name, nui, dose_supported=False)
    return [
        python,
        str(engine_script("native_structure_update.py")),
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
        str(args.structure_n_basis),
        "--n-points",
        str(resolved_structure_n_points(args, ds)),
        "--topup-d-min",
        f"{effective_topup_d_min(args, ds):g}",
        "--max-support-sg",
        f"{args.max_support_sg:g}",
        "--sg-window",
        f"{args.sg_window:g}",
        "--sg-window-mode",
        resolved_sg_window_mode(args, ds),
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
        "--objective-rows",
        str(args.structure_objective_rows),
        "--objective-bins",
        "10",
        "--objective-group-width-deg",
        grid_from_nuisance(nui, "group_width_deg", "1.0"),
        *split_scheme_args(args, split_contract=ds.split_contract),
        "--block-schedule",
        args.structure_blocks,
        "--atom-selection",
        args.structure_atom_selection,
        "--max-nfev",
        str(args.structure_max_nfev),
        "--compute-backend",
        args.compute_backend,
        "--row-plan-cache",
        resolved_row_plan_cache(args, fixed_model=False),
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
        "--profile-quadrature",
        args.profile_quadrature,
        "--profile-accumulation",
        args.profile_accumulation,
        "--early-stop-window",
        str(args.structure_early_stop_window),
        "--early-stop-min-evals",
        str(args.structure_early_stop_min_evals),
        "--early-stop-min-improvement",
        f"{args.structure_early_stop_min_improvement:g}",
        "--early-stop-metric",
        args.structure_early_stop_metric,
        "--thickness",
        grid_from_nuisance(nui, "thickness_A", "1200"),
        "--thickness-grad-frac",
        grid_from_nuisance(nui, "thickness_grad_frac", "0.4"),
        "--frame-offset",
        grid_from_nuisance(nui, "frame_offset", "0.0"),
        "--row-resolution-b",
        grid_from_nuisance(nui, "row_resolution_b", "0.0"),
        "--sg-jitter-mode",
        str(nui.get("sg_jitter_mode", "none")),
        "--sg-jitter-sigma0",
        grid_from_nuisance(nui, "sg_jitter_sigma0", "0.0"),
        "--sg-jitter-sigma-d-inv",
        grid_from_nuisance(nui, "sg_jitter_sigma_d_inv", "0.0"),
        "--sg-jitter-sigma-frame",
        grid_from_nuisance(nui, "sg_jitter_sigma_frame", "0.0"),
        "--sg-jitter-points",
        str(int(float(nui.get("sg_jitter_points", 1)))),
        "--sg-jitter-student-df",
        grid_from_nuisance(nui, "sg_jitter_student_df", "4.0"),
        "--sg-jitter-width-factor",
        grid_from_nuisance(nui, "sg_jitter_width_factor", "3.0"),
        "--absorption-mode",
        str(nui.get("absorption_mode", "none")),
        "--absorption-mu",
        grid_from_nuisance(nui, "absorption_mu", "0.0"),
        "--absorption-frac",
        grid_from_nuisance(nui, "absorption_frac", "0.0"),
        "--absorption-mu-g2",
        grid_from_nuisance(nui, "absorption_mu_g2", "0.0"),
        "--absorption-mu-theta2",
        grid_from_nuisance(nui, "absorption_mu_theta2", "0.0"),
        "--absorption-tds-b",
        grid_from_nuisance(nui, "absorption_tds_b", "0.0"),
        "--group-width-deg",
        grid_from_nuisance(nui, "group_width_deg", "1.0"),
        "--detector-scale-bins",
        str(int(float(nui.get("detector_scale_bins", 0)))),
        "--sigma-a",
        grid_from_nuisance(nui, "sigma_a", "0.0"),
        "--sigma-b-corr",
        grid_from_nuisance(nui, "sigma_b_corr", "0.0"),
        "--sigma-c-peak",
        grid_from_nuisance(nui, "sigma_c_peak", "0.0"),
        "--lambda-grid",
        FULL_Q_LAMBDA,
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--acceptance-metric",
        "all_wR2",
        "--free-wR2-tolerance",
        "0.02",
    ]


def shelxl_feedback_cmd(
    python: str,
    ds: DatasetSpec,
    start_model: str,
    source_summary: Path,
    baseline_hkl: str,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    n_basis: int,
) -> list[str]:
    """Build the high-N/Q SHELXL feedback-loop command."""

    cmd = [
        python,
        str(engine_script("shelxl_feedback_loop.py")),
        "--source-summary",
        str(source_summary),
        "--start-res",
        start_model,
        "--out-dir",
        str(out_dir),
        "--python",
        python,
        "--hkl",
        ds.hkl,
        "--baseline-hkl",
        baseline_hkl or "",
        f"--hkl-transform={ds.hkl_transform}",
        "--cycles",
        str(args.shelxl_feedback_cycles),
        "--n-basis",
        str(n_basis),
        "--n-points",
        str(resolved_final_n_points(args, ds)),
        "--final-n-basis",
        "0",
        "--final-n-points",
        str(resolved_final_n_points(args, ds)),
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--shelxl",
        args.shelxl,
        "--shelxl-iterations",
        str(args.shelxl_feedback_shelxl_iterations),
        "--endpoint-ls-cycles",
        str(args.shelxl_feedback_endpoint_ls_cycles),
        "--endpoint-wght-mode",
        # The macro-cycle rule; a single-crystal loop's cycle endpoints are also its publication candidates.
        resolved_feedback_wght_mode(args),
        "--post-bloch-sigma-policy",
        resolved_post_bloch_sigma_policy(args, feedback_loop=True),
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{args.post_bloch_sigma_min_scale:g}",
        "--post-bloch-sigma-max-scale",
        f"{args.post_bloch_sigma_max_scale:g}",
        *endpoint_handoff_args(args),
        "--shelxl-feedback-xyz-tol",
        f"{float(getattr(args, 'shelxl_feedback_xyz_tol', 0.0) or 0.0):g}",
        "--shelxl-feedback-keep",
        str(getattr(args, "shelxl_feedback_keep", "last") or "last"),
        "--lambda-grid",
        FULL_Q_LAMBDA,
        "--cache-compression",
        args.prediction_cache_compression,
        "--compute-backend",
        args.compute_backend,
        "--row-plan-cache",
        resolved_row_plan_cache(args, fixed_model=False),
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
        "--min-d-spacing",
        f"{ds.min_d_spacing:g}",
        "--topup-d-min",
        f"{effective_topup_d_min(args, ds):g}",
        "--min-peak",
        f"{ds.min_peak:g}",
        "--geometry-percentile",
        f"{ds.geometry_percentile:g}",
    ]
    if args.max_rows:
        cmd += ["--max-rows", str(args.max_rows)]
    if args.make_maps:
        cmd.append("--make-maps-on-final")
    return cmd
