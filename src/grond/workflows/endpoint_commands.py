"""Build SHELXL-feedback commands with explicit numerical settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from grond.cli_support.values import FULL_Q_LAMBDA
from grond.io.config import DatasetSpec
from grond.paths import engine_script
from grond.workflows.nuisance_policy import (
    effective_topup_d_min,
    error_model_sigma_active,
    resolved_feedback_wght_mode,
    resolved_final_n_points,
    resolved_post_bloch_sigma_policy,
    resolved_row_plan_cache,
)

# Most SHELXL calls of the WGHT convergence loop of a converge endpoint.
WGHT_CONVERGE_MAX_CALLS = 20
# Error-model refit passes of the per-crystal sigma model.
ERROR_MODEL_REFITS = 1


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


def endpoint_native_output_args(args: argparse.Namespace) -> list[str]:
    """Engine options of the SHELXL-native outputs shared by every endpoint writer.

    The converge call cap (when an endpoint or macro-cycle rule is converge), the error-model refit count
    (under the error model), the resolved cell e.s.d.s (ZERR) and the Olex2/SHELXLE input cleanup (no stale
    Q-peaks, LIST 6).  An option is emitted only when it acts, so missing attributes and the legacy flags add
    no option to the engine commands.
    """

    cmd: list[str] = []
    if "converge" in {str(getattr(args, "endpoint_wght_mode", "freeze-zero")), resolved_feedback_wght_mode(args)}:
        cmd += ["--wght-converge-max-calls", str(WGHT_CONVERGE_MAX_CALLS)]
    if error_model_sigma_active(args) or str(getattr(args, "post_bloch_sigma_policy", "auto")) == "error_model":
        cmd += ["--error-model-refits", str(ERROR_MODEL_REFITS)]
    if error_model_sigma_active(args):
        # The error model is fitted per crystal; its dataset weight is not applied (AUTO).
        cmd += ["--error-model-dataset-weight", "off"]
    zerr = str(getattr(args, "resolved_zerr_esds", "") or "").strip()
    if zerr:
        cmd += ["--zerr-esds", zerr]
    cmd += ["--strip-q-peaks", "--endpoint-list-code", "6"]
    return cmd


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
        "--row-plan-cache",
        resolved_row_plan_cache(args, fixed_model=False),
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
