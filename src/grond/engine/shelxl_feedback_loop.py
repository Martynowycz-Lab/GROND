#!/usr/bin/env python3
"""Iterate fixed-nuisance Bloch corrections with SHELXL structure update.

This diagnostic asks whether expensive native finite-difference dynamical
structure gradients can be replaced by a cheaper loop:

1. calculate fixed-structure Bloch row predictions,
2. export a merged corrected HKLF4 endpoint,
3. run no-EXTI SHELXL model update with the model's existing restraints/ADPs,
4. feed the updated ``.res`` into the next Bloch calculation.

The nuisance parameter values are read from an existing GROND
``native_row_regularize.py`` summary and pinned during the loop.  This makes
the benchmark a test of the structure-update hook, not another grid search.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

if not __package__:
    # Direct legacy execution must import this checkout, not another installed GROND.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from grond.refine.arguments_feedback import build_parser as build_feedback_parser
from grond.refine.feedback_contract import (
    endpoint_res as endpoint_res,
    feedback_prediction_cache_dir as feedback_prediction_cache_dir,
    grid_value as grid_value,
    nuisance as nuisance,
    read_json as read_json,
    selected_metrics as selected_metrics,
    summary_value as summary_value,
)
from grond.refine.feedback_trajectory import (
    COMPLETED_LOOP_STATUSES,
    KEEP_POLICIES,
    completed_loop_status,
    row_r_by_split,
    select_kept_cycle,
    trajectory_entry,
)
from grond.regularize.contracts import thickness_distribution_options

try:
    from .native_row_regularize import DEFAULT_MAX_SUPPORT_SG, shelx_res_health
except ImportError:  # Direct script execution from the legacy engine directory.
    from native_row_regularize import DEFAULT_MAX_SUPPORT_SG, shelx_res_health


def native_output_args(args: argparse.Namespace) -> list[str]:
    """SHELXL weight/sigma and SHELXL-native output options forwarded to every cycle's native endpoint.

    Each option is forwarded only when it acts (the converge call cap under converge, the refit count under
    the error model), so a legacy loop's per-cycle commands carry none of them.
    """

    out: list[str] = []
    if str(getattr(args, "endpoint_wght_mode", "freeze-zero")) == "converge":
        out += ["--wght-converge-max-calls", str(int(getattr(args, "wght_converge_max_calls", 20) or 20))]
    if str(getattr(args, "post_bloch_sigma_policy", "none")) == "error_model":
        out += ["--error-model-refits", str(int(getattr(args, "error_model_refits", 1)))]
    if str(getattr(args, "zerr_esds", "") or "").strip():
        out += ["--zerr-esds", str(args.zerr_esds)]
    if bool(getattr(args, "strip_q_peaks", False)):
        out.append("--strip-q-peaks")
    if int(getattr(args, "endpoint_list_code", 0) or 0) > 0:
        out += ["--endpoint-list-code", str(int(args.endpoint_list_code))]
    return out


def native_command(
    args: argparse.Namespace,
    source_summary: dict,
    model: Path,
    run_dir: Path,
    *,
    n_basis: int,
    n_points: int,
    make_maps: bool = False,
    skip_endpoint_refinement: bool = False,
) -> list[str]:
    """Build one fixed-nuisance native Bloch/SHELXL endpoint command."""

    nui = nuisance(source_summary)
    inputs = source_summary.get("inputs") or {}
    cfg = source_summary.get("config") or {}
    hkl = args.hkl or inputs.get("hkl") or cfg.get("hkl")
    baseline_hkl = (
        ""
        if bool(getattr(args, "ignore_source_baseline_hkl", False))
        else args.baseline_hkl or inputs.get("baseline_hkl") or cfg.get("baseline_hkl") or ""
    )
    hkl_transform = args.hkl_transform or inputs.get("hkl_transform") or cfg.get("hkl_transform") or "identity"
    if not hkl:
        raise SystemExit("No HKL was supplied and source summary does not record inputs.hkl/config.hkl")

    script = Path(__file__).with_name("native_row_regularize.py")
    cmd = [
        args.python,
        str(script),
        "--hkl",
        str(hkl),
        "--res",
        str(model),
        f"--hkl-transform={hkl_transform}",
        "--out-dir",
        str(run_dir),
        "--n-basis",
        str(int(n_basis)),
        "--n-points",
        str(int(n_points)),
        "--sg-window",
        grid_value(summary_value(source_summary, "sg_window", 0.004), "0.004"),
        "--sg-window-mode",
        str(summary_value(source_summary, "sg_window_mode", "fixed")),
        "--min-d-spacing",
        grid_value(args.min_d_spacing if args.min_d_spacing > 0 else summary_value(source_summary, "min_d_spacing", 0.7), "0.7"),
        "--min-peak",
        grid_value(args.min_peak if args.min_peak >= 0 else summary_value(source_summary, "min_peak", 0.0), "0.0"),
        "--geometry-percentile",
        grid_value(args.geometry_percentile if args.geometry_percentile > 0 else summary_value(source_summary, "geometry_percentile", 100.0), "100.0"),
        "--target-mode",
        str(summary_value(source_summary, "target_mode", "iobs")),
        "--thickness-grid",
        grid_value(nui.get("thickness_A")),
        "--thickness-model-grid",
        str(nui.get("thickness_model", "single")),
        "--thickness-grad-grid",
        grid_value(nui.get("thickness_grad_frac")),
        "--thickness-sigma-frac-grid",
        grid_value(nui.get("thickness_sigma_frac")),
        "--frame-offset-grid",
        grid_value(nui.get("frame_offset")),
        "--group-width-grid",
        grid_value(nui.get("group_width_deg")),
        "--sigma-a-grid",
        grid_value(nui.get("sigma_a")),
        "--sigma-b-corr-grid",
        grid_value(nui.get("sigma_b_corr")),
        "--sigma-c-peak-grid",
        grid_value(nui.get("sigma_c_peak")),
        "--sg-jitter-mode-grid",
        str(nui.get("sg_jitter_mode", "none")),
        "--sg-jitter-sigma0-grid",
        grid_value(nui.get("sg_jitter_sigma0")),
        "--sg-jitter-sigma-d-inv-grid",
        grid_value(nui.get("sg_jitter_sigma_d_inv")),
        "--sg-jitter-sigma-frame-grid",
        grid_value(nui.get("sg_jitter_sigma_frame")),
        "--sg-jitter-points",
        str(int(float(nui.get("sg_jitter_points", 5)))),
        "--sg-jitter-student-df",
        grid_value(nui.get("sg_jitter_student_df", 4.0), "4.0"),
        "--sg-jitter-width-factor",
        grid_value(nui.get("sg_jitter_width_factor", 3.0), "3.0"),
        "--profile-quadrature",
        str(getattr(args, "profile_quadrature", None) or summary_value(source_summary, "profile_quadrature", "uniform")),
        "--profile-accumulation",
        str(getattr(args, "profile_accumulation", None) or summary_value(source_summary, "profile_accumulation", "sum")),
        "--absorption-mode-grid",
        str(nui.get("absorption_mode", "none")),
        "--absorption-mu-grid",
        grid_value(nui.get("absorption_mu")),
        "--dose-b-mode-grid",
        str(nui.get("dose_b_mode", "none")),
        "--dose-b-per-frame-grid",
        grid_value(nui.get("dose_b_per_frame")),
        "--row-resolution-b-grid",
        grid_value(nui.get("row_resolution_b")),
        "--wght-a-grid",
        grid_value(nui.get("native_wght_a"), ""),
        "--wght-b-grid",
        grid_value(nui.get("native_wght_b"), ""),
        "--thickness-points",
        str(int(float(summary_value(source_summary, "thickness_points", 5)))),
        "--topup-d-min",
        grid_value(args.topup_d_min if args.topup_d_min > 0 else summary_value(source_summary, "topup_d_min", 0.8), "0.8"),
        "--max-support-sg",
        grid_value(
            summary_value(source_summary, "max_support_sg", DEFAULT_MAX_SUPPORT_SG),
            f"{DEFAULT_MAX_SUPPORT_SG:g}",
        ),
        "--free-fraction",
        grid_value(summary_value(source_summary, "free_fraction", 0.10), "0.10"),
        "--split-seed",
        str(int(float(summary_value(source_summary, "split_seed", 42)))),
        "--scale-fit-split",
        str(summary_value(source_summary, "scale_fit_split", "all")),
        "--selection-metric",
        str(summary_value(source_summary, "selection_metric", "all_wR2")),
        "--lambda-grid",
        str(args.lambda_grid),
        "--endpoint-b-grid",
        "0.0",
        "--endpoint-exti-grid",
        "off",
        "--shelxl",
        str(args.shelxl),
        "--shelxl-iterations",
        str(int(args.shelxl_iterations)),
        "--endpoint-ls-cycles",
        str(int(args.endpoint_ls_cycles)),
        "--endpoint-wght-mode",
        str(getattr(args, "endpoint_wght_mode", "freeze-zero")),
        *native_output_args(args),
        "--shelxl-convergence-shift-esd",
        grid_value(getattr(args, "shelxl_convergence_shift_esd", 0.0), "0"),
        "--shelxl-npd-recovery",
        str(getattr(args, "shelxl_npd_recovery", "damp")),
        "--endpoint-q-policy",
        str(getattr(args, "endpoint_q_policy", "clip")),
        "--endpoint-include-free-rows" if bool(getattr(args, "endpoint_include_free_rows", True)) else "--no-endpoint-include-free-rows",
        "--post-bloch-sigma-policy",
        str(args.post_bloch_sigma_policy),
        "--post-bloch-sigma-holdout-fraction",
        grid_value(args.post_bloch_sigma_holdout_fraction, "0.20"),
        "--post-bloch-sigma-min-scale",
        grid_value(args.post_bloch_sigma_min_scale, "0.25"),
        "--post-bloch-sigma-max-scale",
        grid_value(args.post_bloch_sigma_max_scale, "4.0"),
        "--rescale",
        str(args.rescale),
        "--hkl-float-format",
        "shelx_fixed",
        "--workers",
        str(int(args.workers)),
        "--chunksize",
        str(int(args.chunksize)),
        "--prediction-cache-dir",
        # The prediction filename and embedded metadata already hash the model,
        # row contract, and full effective Bloch configuration.  Sharing this
        # directory lets a rejected/pathological output model reuse the exact
        # preceding input-model prediction during the final row recompute.
        str(feedback_prediction_cache_dir(run_dir)),
        "--cache-compression",
        str(args.cache_compression),
        "--row-plan-cache",
        str(getattr(args, "row_plan_cache", "off")),
        "--matrix-convention",
        str(nui.get("matrix_convention", summary_value(source_summary, "matrix_convention", "j_minus_i"))),
        # Support thickness distributions (gamma, table/cf/auto); nothing is added for the legacy quadrature.
        *thickness_distribution_options(nui),
    ]
    cmd.append(
        "--positive-only"
        if bool(summary_value(source_summary, "positive_only", True))
        else "--include-nonpositive"
    )
    # These values are part of the selected physical/observation contract, not
    # parameters to reset when the atomic model is updated. Keep optional
    # fields absent only for historical summaries that never recorded them.
    pinned_fields = (
        "sg_window_min", "sg_window_max", "sg_window_scale", "sg_window_frame_floor",
        "mosaicity_mode", "mosaicity_sigma_deg", "mosaicity_max_sigma_deg", "mosaicity_points",
        "dose_b_ref_frame",
        "row_contract", "split_contract",
    )
    for key in pinned_fields:
        value = nui.get(key, cfg.get(key))
        if value is not None and str(value) != "":
            cmd.extend(["--" + key.replace("_", "-"), str(value)])
    # The free-split scheme travels with the split contract: an orbit-hash
    # contract refuses the engine's legacy default.  Historical summaries that
    # never recorded a scheme keep the engine's legacy split.
    for key in ("split_scheme", "split_salt", "exclude_orbits_json"):
        value = cfg.get(key)
        if value is not None and str(value) != "":
            cmd.extend(["--" + key.replace("_", "-"), str(value)])
    # The P-spline scale grids are part of the source stage contract; a summary that records none keeps the
    # engine's default grids.
    scale_model = (source_summary.get("best_native") or {}).get("scale_model") or {}
    n_basis_grid = cfg.get("scale_spline_n_basis_grid") or ",".join(str(v) for v in scale_model.get("n_basis_grid") or [])
    lambda_grid = cfg.get("scale_spline_lambda_grid") or ",".join(f"{float(v):g}" for v in scale_model.get("lambda_grid") or [])
    if n_basis_grid:
        cmd.extend(["--scale-spline-n-basis-grid", str(n_basis_grid)])
    if lambda_grid:
        cmd.extend(["--scale-spline-lambda-grid", str(lambda_grid)])
    geometry_path = nui.get("geometry_path_file") or cfg.get("geometry_path") or cfg.get("geometry_path_file")
    if geometry_path:
        cmd.extend(["--geometry-path", str(geometry_path)])
    if baseline_hkl:
        cmd.extend(["--baseline-hkl", str(baseline_hkl)])
    max_rows = int(args.max_rows) if args.max_rows > 0 else int(cfg.get("max_rows") or 0)
    if max_rows > 0:
        cmd.extend(["--max-rows", str(max_rows)])
    if bool(getattr(args, "keep_damp", False)):
        cmd.append("--keep-damp")
    if make_maps:
        cmd.append("--make-maps")
    if skip_endpoint_refinement:
        cmd.append("--skip-endpoint-refinement")
    return cmd


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> dict:
    """Write child output as it arrives and keep partial logs on failure."""

    t0 = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = {**env, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, text=True, stdout=log, stderr=subprocess.STDOUT, env=child_env)
    elapsed = time.perf_counter() - t0
    return {"returncode": proc.returncode, "elapsed_seconds": elapsed, "log": str(log_path)}


def run_cycle(
    args: argparse.Namespace,
    source_summary: dict,
    model: Path,
    run_dir: Path,
    *,
    n_basis: int,
    n_points: int,
    label: str,
    env: dict[str, str],
    make_maps: bool = False,
    skip_endpoint_refinement: bool = False,
) -> dict:
    """Run one Bloch/SHELXL feedback cycle."""

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = native_command(
        args,
        source_summary,
        model,
        run_dir,
        n_basis=n_basis,
        n_points=n_points,
        make_maps=make_maps,
        skip_endpoint_refinement=skip_endpoint_refinement,
    )
    (run_dir / "COMMAND.sh").write_text(" ".join(shlex.quote(str(x)) for x in cmd) + "\n")
    status = run_command(cmd, run_dir / "run.log", env)
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    updated_res = Path("")
    endpoint_health: dict = {"ok": False, "reason": "not_run"}
    selected_endpoint = summary.get("best_endpoint") if isinstance(summary, dict) else {}
    selected_endpoint = selected_endpoint if isinstance(selected_endpoint, dict) else {}
    selected_shelxl = selected_endpoint.get("shelxl")
    selected_shelxl = selected_shelxl if isinstance(selected_shelxl, dict) else {}
    endpoint_shelxl_status = str(selected_shelxl.get("status") or "")
    if status["returncode"] == 0 and skip_endpoint_refinement:
        updated_res = model
        endpoint_health = shelx_res_health(model, require_lst=False)
        endpoint_shelxl_status = "reused_validated_input_model"
        if not endpoint_health.get("ok") or endpoint_health.get("safe_for_feedback") is False:
            status["returncode"] = 3
            status["endpoint_error"] = f"input model is not safe for final row recompute: {endpoint_health}"
    elif status["returncode"] == 0:
        try:
            updated_res = endpoint_res(run_dir, summary=summary)
            endpoint_health = shelx_res_health(updated_res, updated_res.with_suffix(".lst"))
        except FileNotFoundError as exc:
            endpoint_health = {"ok": False, "reason": "missing_or_invalid_endpoint_res", "error": str(exc)}
            status["returncode"] = 3
            status["endpoint_error"] = str(exc)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "input_model": str(model),
        "output_model": str(updated_res) if updated_res else "",
        "endpoint_shelxl_status": endpoint_shelxl_status,
        "endpoint_res_health": endpoint_health,
        "n_basis": int(n_basis),
        "n_points": int(n_points),
        "command": cmd,
        **status,
        "metrics": selected_metrics(summary) if summary else {},
        "row_stats_model": str(model),
        "skip_endpoint_refinement": bool(skip_endpoint_refinement),
    }


def _endpoint_shelxl(run_dir: Path) -> tuple[dict, dict]:
    """The selected endpoint record and its SHELXL block from one cycle's native summary."""

    summary = read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
    endpoint = summary.get("best_endpoint") if isinstance(summary.get("best_endpoint"), dict) else {}
    shelxl = endpoint.get("shelxl") if isinstance(endpoint.get("shelxl"), dict) else {}
    return endpoint, shelxl


def _summary_with_nuisance(source_summary: dict, nui: dict) -> dict:
    """A copy of the source summary whose selected nuisance is ``nui`` (every other contract field unchanged)."""

    out = dict(source_summary)
    out["best_native"] = {**(source_summary.get("best_native") or {}), "nuisance": dict(nui)}
    return out


def _thickness_tracker(args: argparse.Namespace, out: Path, env: dict[str, str]):
    """The per-structure thickness of --feedback-thickness-update on (None when off)."""

    if str(getattr(args, "feedback_thickness_update", "off")) != "on":
        return None
    if not str(getattr(args, "nested_nuisance_contract", "") or "").strip():
        raise SystemExit("--feedback-thickness-update on needs --nested-nuisance-contract")
    from grond.engine.nested_nuisance_stage import FeedbackThickness

    def run_stage(request_path, result_path, label):
        """Run nested_nuisance_stage.py on one re-selection request; raise if it fails."""

        cmd = [args.python, str(Path(__file__).with_name("nested_nuisance_stage.py")), "--request", str(request_path),
               "--out", str(result_path), "--workers", str(int(args.workers))]
        status = run_command(cmd, Path(result_path).with_suffix(".log"), env)
        if status["returncode"] != 0:
            raise RuntimeError(f"thickness re-selection {label} returned {status['returncode']}; see {status['log']}")

    return FeedbackThickness(Path(args.nested_nuisance_contract).expanduser(), out, run_stage)


def main() -> None:
    """Run the single-dataset macro-cycles as a monitored fixed-point iteration.

    Each cycle is the lagged map p_{k+1} = T(p_k) (see grond.refine.feedback_trajectory): its stationary
    point is not the dynamical least-squares solution, so the loop records a ``trajectory`` (row R of the
    entering model on work and free rows, converged SHELXL statistics, coordinate shift, NPD/ISOR/DAMP),
    stops on ``--shelxl-feedback-xyz-tol`` (stop_reason ``converged``) or the cycle cap (``cap``), still
    stops on an unsafe or NPD model, and with ``--shelxl-feedback-keep best-free`` keeps the output model
    with the lowest free-row R_obs (``kept_cycle``).  Legacy settings (tolerance 0, keep last) reproduce
    the fixed-count loop and record status ``ok``; otherwise the status is the stop reason.

    With ``--feedback-thickness-update on`` every structure the loop corrects with (each entering model and the
    final model) has its own thickness distribution and lambda, re-selected on its spectra with the nested
    physics variant fixed (grond.engine.nested_nuisance_stage); each trajectory entry records it.
    """

    ap = build_feedback_parser(
        description=__doc__,
    )
    args = ap.parse_args()
    if args.shelxl_feedback_keep not in KEEP_POLICIES:
        ap.error(f"--shelxl-feedback-keep must be one of {', '.join(KEEP_POLICIES)}")

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    source_summary_path = Path(args.source_summary).expanduser()
    source_summary = read_json(source_summary_path)
    current_model = Path(args.start_res).expanduser().resolve()
    env = os.environ.copy()
    env["SHELXL"] = str(args.shelxl)
    xyz_tol = float(args.shelxl_feedback_xyz_tol or 0.0)
    keep_policy = str(args.shelxl_feedback_keep)
    monitored = xyz_tol > 0.0 or keep_policy != "last"

    result = {
        "source_summary": str(source_summary_path.resolve()),
        "start_res": str(current_model),
        "current_model": str(current_model),
        "shelxl": str(args.shelxl),
        "endpoint_wght_mode": str(args.endpoint_wght_mode),
        "wght_converge_max_calls": int(getattr(args, "wght_converge_max_calls", 20) or 20),
        "native_output_args": native_output_args(args),
        "post_bloch_sigma_policy": str(args.post_bloch_sigma_policy),
        "post_bloch_sigma_holdout_fraction": float(args.post_bloch_sigma_holdout_fraction),
        "post_bloch_sigma_min_scale": float(args.post_bloch_sigma_min_scale),
        "post_bloch_sigma_max_scale": float(args.post_bloch_sigma_max_scale),
        "endpoint_ls_cycles": int(args.endpoint_ls_cycles),
        "cycles_requested": int(args.cycles),
        "fast_n_basis": int(args.n_basis),
        "fast_n_points": int(args.n_points),
        "final_n_basis": int(args.final_n_basis),
        "final_n_points": int(args.final_n_points),
        "handoff": {
            "shelxl_iterations": int(args.shelxl_iterations),
            "shelxl_convergence_shift_esd": float(args.shelxl_convergence_shift_esd),
            "shelxl_npd_recovery": str(args.shelxl_npd_recovery),
            "keep_damp": bool(args.keep_damp),
            "endpoint_include_free_rows": bool(args.endpoint_include_free_rows),
            "free_rows_in_hkl": bool(args.endpoint_include_free_rows),
            "endpoint_q_policy": str(args.endpoint_q_policy),
            "rescale": str(args.rescale),
        },
        "xyz_tol_A": xyz_tol,
        "keep_policy": keep_policy,
        "status": "running",
        "kept_cycle": 0,
        "trajectory": [],
        "runs": [],
    }
    t0 = time.perf_counter()
    thickness = _thickness_tracker(args, out, env)
    last_run_dir: Path | None = None

    def summary_for(model: Path, label: str) -> tuple[dict, dict | None]:
        """The source contract with the thickness selected for ``model`` (re-selected when new)."""

        if thickness is None:
            return source_summary, None
        from grond.engine.nested_nuisance_stage import apply_thickness

        if not thickness.known(model):
            if last_run_dir is None:
                raise RuntimeError(f"the loop starts from a model the nested nuisance stage did not select on ({model})")
            thickness.update(model, last_run_dir, label)
        fields = thickness.fields(model)
        return _summary_with_nuisance(source_summary, apply_thickness(nuisance(source_summary), fields)), thickness.record(model)

    def record_thickness_update() -> None:
        """Store the thickness-update contract, updates and the current model's thickness in the result."""

        if thickness is not None:
            result["thickness_update"] = {
                "enabled": True,
                "contract": str(thickness.contract_path),
                "updates": thickness.updates,
                "final": ({"record": thickness.record(current_model), "fields": thickness.fields(current_model)}
                          if thickness.known(current_model) else None),
            }

    def fail_thickness(stage: str, exc: Exception) -> None:
        """Mark the loop as failed at a thickness re-selection and write the summary JSON."""

        result["status"] = "failed_thickness_update"
        result["stopped_reason"] = f"thickness re-selection {stage}: {type(exc).__name__}: {exc}"
        record_thickness_update()
        (out / "shelxl_feedback_loop_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")

    adopted: list[dict] = []
    stopped = False
    stop_status = "cap"
    for i in range(1, int(args.cycles) + 1):
        # A monitored loop may stop at any cycle, so every cycle's endpoint carries its maps.
        maps_this_cycle = bool(
            args.make_maps_on_final and int(args.final_n_basis) <= 0 and (monitored or i == int(args.cycles))
        )
        try:
            cycle_summary, thickness_record = summary_for(current_model, f"cycle_{i:02d}")
        except Exception as exc:
            fail_thickness(f"before cycle {i}", exc)
            raise
        rec = run_cycle(
            args,
            cycle_summary,
            current_model,
            out / f"cycle_{i:02d}_N{int(args.n_basis)}Q{int(args.n_points)}",
            n_basis=args.n_basis,
            n_points=args.n_points,
            label=f"cycle_{i:02d}",
            env=env,
            make_maps=maps_this_cycle,
        )
        if thickness_record is not None:
            rec["thickness"] = thickness_record
            if rec["returncode"] == 0:
                from grond.engine.nested_nuisance_stage import thickness_drift

                run_summary = Path(rec["run_dir"]) / "summary.json"
                recorded = ((read_json(run_summary) if run_summary.exists() else {}).get("best_native") or {}).get("nuisance") or {}
                drifted = thickness_drift(thickness.fields(current_model), recorded)
                if drifted:
                    result["runs"].append(rec)
                    exc = RuntimeError(f"cycle {i} did not reproduce the selected thickness: " + "; ".join(drifted))
                    fail_thickness(f"in cycle {i}", exc)
                    raise exc
        last_run_dir = Path(rec["run_dir"])
        result["runs"].append(rec)
        # Rows of the model entering this cycle score the previous cycle's output model, also when this
        # cycle's own endpoint failed after its Bloch rows were written.
        entering_rows = row_r_by_split([rec["run_dir"]])
        if adopted:
            adopted[-1]["rows"] = {**entering_rows, "source": f"cycle_{i:02d} entering-model rows"}
        if rec["returncode"] != 0:
            result["stopped_reason"] = rec.get("endpoint_error") or f"cycle returned {rec['returncode']}"
            result["status"] = "failed_cycle"
            stopped = True
            break
        endpoint, endpoint_shelxl = _endpoint_shelxl(Path(rec["run_dir"]))
        trajectory = trajectory_entry(
            i,
            entering_rows,
            endpoint_shelxl,
            input_model=str(current_model),
            output_model=str(rec.get("output_model") or ""),
            n_excluded_q_rows=endpoint.get("n_excluded_q_rows"),
        )
        if thickness_record is not None:
            trajectory["thickness"] = thickness_record
        result["trajectory"].append(trajectory)
        health = rec.get("endpoint_res_health") or {}
        if not health.get("ok") or health.get("safe_for_feedback") is False:
            result["stopped_reason"] = (
                "SHELXL feedback cycle did not produce a structurally valid, pathology-free model; "
                "keeping the previous clean model. "
                f"reason={health.get('reason', 'unknown')}, NPD atoms={health.get('npd_count', 0)}, "
                f"split warnings={health.get('split_count', 0)}."
            )
            result["rejected_model"] = rec.get("output_model", "")
            result["status"] = (
                "stopped_adp_pathology"
                if int(health.get("npd_count", 0) or 0) > 0 or health.get("has_npd") is True
                else "stopped_unsafe_endpoint"
            )
            trajectory["adopted_for_feedback"] = False
            stopped = True
            break
        trajectory["adopted_for_feedback"] = True
        current_model = Path(rec["output_model"]).resolve()
        result["current_model"] = str(current_model)
        adopted.append({"cycle": i, "model": str(current_model), "rows": None})
        shift = trajectory["max_xyz_shift_A"]
        if xyz_tol > 0.0 and shift is not None and shift < xyz_tol:
            stop_status = "converged"
            result["convergence_reason"] = f"max non-H coordinate shift {shift:.5f} A < {xyz_tol:g} A"
            break
    if not stopped:
        result["stop_reason"] = stop_status
        result["status"] = completed_loop_status(stop_status, xyz_tol=xyz_tol, keep_policy=keep_policy)

    final_summary = source_summary
    if thickness is not None and (not result["runs"] or result["runs"][-1].get("returncode") == 0):
        try:
            final_summary, _final_record = summary_for(current_model, "final")
        except Exception as exc:
            fail_thickness("for the final model", exc)
            raise
    recompute_n_basis = int(args.final_n_basis) if int(args.final_n_basis) > 0 else int(args.n_basis)
    recompute_n_points = int(args.final_n_points) if int(args.final_n_basis) > 0 else int(args.n_points)
    last_scored_recompute = None
    scored_dir: Path | None = None
    loop_ok = not result["runs"] or result["runs"][-1].get("returncode") == 0
    if loop_ok and keep_policy == "best-free" and len(adopted) > 1 and adopted[-1]["rows"] is None:
        # The last output is scored with the same N/Q as the cycles that scored the others.
        last_cycle = int(adopted[-1]["cycle"])
        score_dir = (
            out / f"final_row_recompute_N{recompute_n_basis}Q{recompute_n_points}"
            if int(args.final_n_basis) <= 0
            else out / f"score_cycle_{last_cycle:02d}_N{int(args.n_basis)}Q{int(args.n_points)}"
        )
        rec = run_cycle(
            args,
            final_summary,
            current_model,
            score_dir,
            n_basis=args.n_basis,
            n_points=args.n_points,
            label="final_row_recompute" if int(args.final_n_basis) <= 0 else f"score_cycle_{last_cycle:02d}",
            env=env,
            make_maps=False,
            skip_endpoint_refinement=True,
        )
        result["runs"].append(rec)
        scored_dir = score_dir
        if rec["returncode"] == 0:
            adopted[-1]["rows"] = {**row_r_by_split([rec["run_dir"]]), "source": rec["label"]}
            if int(args.final_n_basis) <= 0:
                last_scored_recompute = rec
        else:
            result["stopped_reason"] = rec.get("endpoint_error") or f"{rec['label']} returned {rec['returncode']}"
    result["adopted_cycle_scores"] = [
        {"cycle": item["cycle"], "free_r_obs": (item.get("rows") or {}).get("free_r_obs"),
         "work_r_obs": (item.get("rows") or {}).get("work_r_obs"), "source": (item.get("rows") or {}).get("source")}
        for item in adopted
    ]
    try:
        kept_cycle, kept_reason = select_kept_cycle(
            [(item["cycle"], (item.get("rows") or {}).get("free_r_obs")) for item in adopted], keep_policy
        )
    except RuntimeError as exc:
        # Refuse to guess, but leave the trajectory and the reason on disk first.
        result["status"] = "failed_keep_best" if result.get("status") in COMPLETED_LOOP_STATUSES else result["status"]
        result["kept_reason"] = str(exc)
        result["wall_seconds"] = time.perf_counter() - t0
        (out / "shelxl_feedback_loop_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
        detail = f" ({result['stopped_reason']})" if result.get("stopped_reason") else ""
        raise RuntimeError(f"{exc}{detail}") from exc
    result["kept_cycle"] = kept_cycle
    result["kept_reason"] = kept_reason
    if kept_cycle > 0 and keep_policy != "last":
        current_model = Path(next(item["model"] for item in adopted if item["cycle"] == kept_cycle)).resolve()
        result["current_model"] = str(current_model)
        if kept_cycle != int(adopted[-1]["cycle"]):
            last_scored_recompute = None
            if thickness is not None:
                # An earlier kept output entered the next cycle, where its thickness was selected.
                try:
                    final_summary, _final_record = summary_for(current_model, "final")
                except Exception as exc:
                    fail_thickness("for the kept model", exc)
                    raise

    if (not result["runs"] or result["runs"][-1].get("returncode") == 0) and int(args.final_n_basis) > 0:
        rec = run_cycle(
            args,
            final_summary,
            current_model,
            out / f"final_N{int(args.final_n_basis)}Q{int(args.final_n_points)}",
            n_basis=args.final_n_basis,
            n_points=args.final_n_points,
            label="final_polish",
            env=env,
            make_maps=bool(args.make_maps_on_final),
        )
        result["runs"].append(rec)
        final_health = rec.get("endpoint_res_health") or {}
        if rec["returncode"] == 0 and final_health.get("ok") and final_health.get("safe_for_feedback") is not False:
            result["current_model"] = rec["output_model"]
            current_model = Path(rec["output_model"]).resolve()
            if thickness is not None:
                last_run_dir = Path(rec["run_dir"])
                try:
                    final_summary, _final_record = summary_for(current_model, "final_polish")
                except Exception as exc:
                    fail_thickness("for the final-polish model", exc)
                    raise
        elif rec["returncode"] == 0:
            result["stopped_reason"] = (
                "final polish produced an unsafe endpoint; keeping the previous clean model. "
                f"health={final_health}"
            )
            result["rejected_model"] = rec.get("output_model", "")
        else:
            result["stopped_reason"] = rec.get("endpoint_error") or f"final polish returned {rec['returncode']}"

    if last_scored_recompute is not None:
        result["current_model"] = str(current_model)
        result["final_row_stats_run"] = last_scored_recompute["run_dir"]
        result["final_row_stats_model"] = str(current_model)
    elif not result["runs"] or result["runs"][-1].get("returncode") == 0:
        recompute_dir = out / f"final_row_recompute_N{recompute_n_basis}Q{recompute_n_points}"
        if scored_dir is not None and recompute_dir == scored_dir:
            # The standard directory already holds the rows that scored the last output; never overwrite it.
            recompute_dir = out / f"final_row_recompute_kept_cycle_{kept_cycle:02d}_N{recompute_n_basis}Q{recompute_n_points}"
        rec = run_cycle(
            args,
            final_summary,
            current_model,
            recompute_dir,
            n_basis=recompute_n_basis,
            n_points=recompute_n_points,
            label="final_row_recompute",
            env=env,
            make_maps=False,
            skip_endpoint_refinement=True,
        )
        result["runs"].append(rec)
        if rec["returncode"] == 0:
            result["current_model"] = str(current_model)
            result["final_row_stats_run"] = rec["run_dir"]
            result["final_row_stats_model"] = str(current_model)
        else:
            result["stopped_reason"] = rec.get("endpoint_error") or f"final row recompute returned {rec['returncode']}"

    record_thickness_update()
    result["wall_seconds"] = time.perf_counter() - t0
    result["summary_path"] = str((out / "shelxl_feedback_loop_summary.json").resolve())
    (out / "shelxl_feedback_loop_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
