"""Build and evaluate discrete or continuous thickness candidates under one row contract."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.io.config import DatasetSpec, PrepConfig
from grond.paths import engine_script
from grond.workflows.decision import safe_float, score_native_summary
from grond.workflows.model_selection import best_nuisance, dataset_baseline_hkl
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nuisance_policy import effective_topup_d_min, first, grid_from_nuisance, resolved_sg_window_mode


def continuous_thickness_cmd(
    python: str,
    ds: DatasetSpec,
    structure: str,
    out_dir: Path,
    args: argparse.Namespace,
    pinned: dict,
    *,
    n_basis: int,
    n_points: int,
    stage: str,
) -> list[str]:
    """Build the phase-only continuous thickness-polish command."""

    thickness = grid_from_nuisance(pinned, "thickness_A", "1200")
    grad = grid_from_nuisance(pinned, "thickness_grad_frac", "0.4")
    frame_offset = grid_from_nuisance(pinned, "frame_offset", "0.0")
    group_width = grid_from_nuisance(pinned, "group_width_deg", "1.0")
    detector_bins = grid_from_nuisance(pinned, "detector_scale_bins", "0")
    sigma_a = grid_from_nuisance(pinned, "sigma_a", "0.0")
    sigma_b = grid_from_nuisance(pinned, "sigma_b_corr", "0.0")
    sigma_c = grid_from_nuisance(pinned, "sigma_c_peak", "0.0")
    row_resolution_b = grid_from_nuisance(pinned, "row_resolution_b", "0.0")
    cmd = [
        python,
        str(engine_script("continuous_thickness_polish.py")),
        "--hkl",
        ds.hkl,
        "--res",
        structure,
        f"--hkl-transform={ds.hkl_transform}",
        "--out-dir",
        str(out_dir),
        "--n-basis",
        str(n_basis),
        "--n-points",
        str(n_points),
        "--sg-window",
        f"{args.sg_window:g}",
        "--sg-window-mode",
        resolved_sg_window_mode(args, ds),
        "--sg-window-min",
        f"{args.sg_window_min:g}",
        "--sg-window-max",
        f"{args.sg_window_max:g}",
        "--sg-window-scale",
        f"{args.sg_window_scale:g}",
        "--sg-window-frame-floor",
        f"{args.sg_window_frame_floor:g}",
        "--profile-quadrature",
        args.profile_quadrature,
        "--profile-accumulation",
        args.profile_accumulation,
        "--topup-d-min",
        f"{effective_topup_d_min(args, ds):g}",
        "--max-support-sg",
        f"{args.max_support_sg:g}",
        "--min-d-spacing",
        f"{ds.min_d_spacing:g}",
        "--min-peak",
        f"{ds.min_peak:g}",
        "--geometry-percentile",
        f"{ds.geometry_percentile:g}",
        "--target-mode",
        ds.target_mode,
        "--start-thickness",
        thickness,
        "--start-grad",
        grad,
        "--thickness-points",
        str(args.thickness_points),
        "--frame-offset",
        frame_offset,
        "--group-width-deg",
        group_width,
        "--detector-scale-bins",
        detector_bins,
        "--sigma-a",
        sigma_a,
        "--sigma-b-corr",
        sigma_b,
        "--sigma-c-peak",
        sigma_c,
        "--row-resolution-b",
        row_resolution_b,
        "--scale-smoothness",
        str(args.scale_smoothness),
        "--matrix-convention",
        args.matrix_convention,
        "--free-fraction",
        f"{args.free_fraction:g}",
        "--split-seed",
        str(args.split_seed),
        "--scale-fit-split",
        "all" if args.production_scales else "work",
        "--selection-metric",
        args.continuous_thickness_metric,
        "--strong-rows",
        str(args.continuous_thickness_strong_rows),
        "--bracket-frac",
        str(args.continuous_thickness_bracket_frac),
        "--grid-thickness-step",
        str(args.continuous_thickness_grid_step),
        "--grid-grad-step",
        str(args.continuous_thickness_grad_step),
        "--powell-maxiter",
        str(args.continuous_thickness_powell_maxiter),
        "--min-promotion-gain",
        str(args.continuous_thickness_min_gain),
        "--max-free-wr2-loss",
        str(args.continuous_thickness_max_free_wr2_loss),
        "--max-all-r1-loss",
        str(args.continuous_thickness_max_all_r1_loss),
    ]
    if ds.include_nonpositive:
        cmd.append("--include-nonpositive")
    max_rows = int(getattr(args, "continuous_thickness_max_rows", 0) or 0)
    if max_rows > 0:
        cmd += ["--max-rows", str(max_rows)]
    return cmd


def continuous_thickness_enabled(args: argparse.Namespace) -> bool:
    """Return true when automatic continuous thickness polish should run."""

    mode = str(getattr(args, "continuous_thickness_polish", "auto") or "auto").lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    return not bool(getattr(args, "fast", False))


def continuous_thickness_seed_nuisance(args: argparse.Namespace, *candidates: Mapping[str, object] | None) -> dict:
    """Return the pinned nuisance state for continuous thickness polish.

    Real runs use the best JSON summary from the preceding screen/regularize stage.
    Dry-runs do not have those summaries yet, so this supplies a conservative
    single-point seed only to make the generated execution plan complete.
    """

    for candidate in candidates:
        if candidate:
            return dict(candidate)
    if not bool(getattr(args, "dry_run", False)):
        return {}
    return {
        "thickness_A": first(getattr(args, "thickness_grid", "1200"), "1200"),
        "thickness_grad_frac": first(getattr(args, "thickness_grad_grid", "0.4"), "0.4"),
        "frame_offset": first(getattr(args, "frame_offset_grid", "0.0"), "0.0"),
        "group_width_deg": "1.0",
        "detector_scale_bins": "0",
        "sigma_a": "0.0",
        "sigma_b_corr": "0.0",
        "sigma_c_peak": "0.0",
        "row_resolution_b": "0.0",
    }


def numeric_nuisance_contract_matches(expected_value: object, actual_value: object) -> bool:
    """Compare one numeric nuisance value without mishandling N/A NaNs.

    Controlled engine summaries use NaN for numeric fields that do not apply,
    such as ``dose_b_ref_frame`` when dose correction is disabled.  Matching
    NaN sentinels describe the same contract; one-sided NaNs, infinities, and
    non-numeric values do not.  Finite values retain the existing relative
    replay tolerance.
    """

    try:
        expected = float(expected_value)
        actual = float(actual_value)
    except (TypeError, ValueError):
        return False
    if math.isnan(expected) and math.isnan(actual):
        return True
    if not math.isfinite(expected) or not math.isfinite(actual):
        return False
    tolerance = 1e-8 * max(1.0, abs(expected))
    return abs(actual - expected) <= tolerance


def discrete_thickness_promotion_record(run: Path, *, dry_run: bool) -> dict:
    """Return the engine's auditable post-ranking thickness decision."""

    run = Path(run)
    artifact = run / "discrete_thickness_promotion.json"
    if dry_run:
        return {
            "family": "discrete_thickness_polish",
            "status": "planned",
            "run": str(run),
            "artifact": str(artifact),
        }
    summary_path = run / "summary.json"
    summary = try_read_json(summary_path)
    decision = summary.get("promotion_decision")
    if not isinstance(decision, Mapping) or str(decision.get("status", "")) == "not_requested":
        raise RuntimeError(
            f"discrete thickness stage did not record the required held-out promotion decision: {summary_path}"
        )
    return {
        **dict(decision),
        "run": str(run),
        "artifact": str(artifact),
    }


def discrete_thickness_baseline_nuisance(args: argparse.Namespace, nuisance: Mapping[str, object] | None) -> dict:
    """Require a real starting nuisance, with a planning-only dry-run seed."""

    if nuisance:
        return dict(nuisance)
    if bool(getattr(args, "dry_run", False)):
        return {
            "thickness_A": 1200.0,
            "thickness_grad_frac": 0.0,
            "frame_offset": 0.0,
        }
    raise RuntimeError(
        "discrete thickness polishing has no starting nuisance from the preceding stage; "
        "refusing to run an unguarded work-only search"
    )


def materialize_continuous_nuisance_run(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    *,
    active_model: str,
    baseline_run: Path,
    continuous_run: Path,
    selected_nuisance: Mapping[str, object],
    n_basis: int,
    n_points: int,
    stage: str,
) -> tuple[dict, Path, dict]:
    """Materialize a continuous-thickness choice on the full row contract.

    ``continuous_thickness_polish.py`` intentionally optimizes a bounded row
    subset and writes a summary rather than ``native_dyn_calc.csv``.  Feature
    gates and multi-dataset policy decisions require a complete, comparable
    row table.  Reusing the optimizer directory as if it were a native run
    silently disables those gates, so the selected nuisance is replayed once
    through the ordinary native stage before it can flow downstream.
    """

    from grond.cli_support.execution import run_logged

    nuisance = dict(selected_nuisance or {})
    if not nuisance:
        raise RuntimeError(
            f"continuous thickness stage for {ds.name} did not record a selected nuisance: "
            f"{continuous_run / 'summary.json'}"
        )
    source_row_table = continuous_run / "native_dyn_calc.csv"
    materialized = continuous_run if source_row_table.exists() else continuous_run / "materialized_full_rows"
    if not source_row_table.exists():
        run_logged(
            native_stage_cmd(
                python,
                ds,
                active_model,
                dataset_baseline_hkl(prep, ds),
                materialized,
                args,
                n_basis=int(n_basis),
                n_points=int(n_points),
                stage=stage,
                pinned=nuisance,
                make_maps=False,
                endpoint=False,
            ),
            logs,
            f"{ds.name}_{stage}",
            dry_run=args.dry_run,
        )
    row_table = materialized / "native_dyn_calc.csv"
    if not args.dry_run and not row_table.exists():
        raise RuntimeError(
            f"continuous-thickness materialization did not produce the required full row table: {row_table}"
        )
    materialized_nuisance = best_nuisance(materialized) if not args.dry_run else nuisance
    if not args.dry_run and not materialized_nuisance:
        raise RuntimeError(
            "continuous-thickness materialization did not record a reusable nuisance block: "
            f"{materialized / 'summary.json'}"
        )
    numeric_contract_keys = (
        "thickness_A",
        "thickness_grad_frac",
        "thickness_sigma_frac",
        "frame_offset",
        "group_width_deg",
        "detector_scale_bins",
        "sigma_a",
        "sigma_b_corr",
        "sigma_c_peak",
        "row_resolution_b",
        "scale_smoothness",
        "sg_jitter_sigma0",
        "sg_jitter_sigma_d_inv",
        "sg_jitter_sigma_frame",
        "sg_jitter_points",
        "sg_jitter_student_df",
        "sg_jitter_width_factor",
        "absorption_mu",
        "absorption_frac",
        "absorption_mu_g2",
        "absorption_mu_theta2",
        "absorption_tds_b",
        "dose_b_per_frame",
        "dose_b_ref_frame",
    )
    categorical_contract_keys = (
        "thickness_model",
        "sg_jitter_mode",
        "absorption_mode",
        "dose_b_mode",
        "matrix_convention",
    )
    mismatches: list[str] = []
    if not args.dry_run:
        for key in numeric_contract_keys:
            if key not in nuisance:
                continue
            if key not in materialized_nuisance or not numeric_nuisance_contract_matches(
                nuisance.get(key), materialized_nuisance.get(key)
            ):
                mismatches.append(
                    f"{key}: selected={nuisance.get(key)!r}, materialized={materialized_nuisance.get(key)!r}"
                )
        for key in categorical_contract_keys:
            if key in nuisance and str(materialized_nuisance.get(key)) != str(nuisance.get(key)):
                mismatches.append(
                    f"{key}: selected={nuisance.get(key)!r}, materialized={materialized_nuisance.get(key)!r}"
                )
        if mismatches:
            raise RuntimeError(
                "continuous-thickness full-row materialization changed the selected nuisance contract: "
                + "; ".join(mismatches)
            )
    baseline_summary_path = Path(baseline_run) / "summary.json"
    materialized_summary_path = materialized / "summary.json"
    if not args.dry_run and not baseline_summary_path.exists():
        raise RuntimeError(
            f"continuous-thickness full-row promotion has no comparable baseline summary: {baseline_summary_path}"
        )
    baseline_summary = try_read_json(baseline_summary_path) if not args.dry_run else {}
    candidate_summary = try_read_json(materialized_summary_path) if not args.dry_run else {}
    baseline_score = score_native_summary(baseline_summary) if not args.dry_run else {}
    candidate_score = score_native_summary(candidate_summary) if not args.dry_run else {}
    promotion_reasons: list[str] = []
    score_gain = float("nan")
    free_wR2_loss = float("nan")
    all_R1_loss = float("nan")
    baseline_free_rows = 0
    candidate_free_rows = 0
    if not args.dry_run:

        def positive_free_rows(summary: Mapping[str, object]) -> int:
            best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
            metrics = best.get("metrics") if isinstance(best.get("metrics"), Mapping) else {}
            free = metrics.get("free") if isinstance(metrics.get("free"), Mapping) else {}
            free_dyn = free.get("dyn") if isinstance(free.get("dyn"), Mapping) else {}
            count = safe_float(free_dyn.get("n"), float("nan"))
            if not math.isfinite(count):
                selection = summary.get("row_selection") if isinstance(summary.get("row_selection"), Mapping) else {}
                count = safe_float(selection.get("free_rows"), 0.0)
            return max(0, int(count)) if math.isfinite(count) else 0

        baseline_free_rows = positive_free_rows(baseline_summary)
        candidate_free_rows = positive_free_rows(candidate_summary)
        if baseline_free_rows <= 0:
            promotion_reasons.append("baseline_missing_positive_free_rows")
        if candidate_free_rows <= 0:
            promotion_reasons.append("candidate_missing_positive_free_rows")
        baseline_total = safe_float(baseline_score.get("score"), float("nan"))
        candidate_total = safe_float(candidate_score.get("score"), float("nan"))
        baseline_free = safe_float(baseline_score.get("free_dyn_wR2_all"), float("nan"))
        candidate_free = safe_float(candidate_score.get("free_dyn_wR2_all"), float("nan"))
        baseline_r1 = safe_float(baseline_score.get("dyn_R1_all"), float("nan"))
        candidate_r1 = safe_float(candidate_score.get("dyn_R1_all"), float("nan"))
        if not all(
            math.isfinite(value)
            for value in (baseline_total, candidate_total, baseline_free, candidate_free, baseline_r1, candidate_r1)
        ):
            promotion_reasons.append("missing_finite_full_row_metrics")
        else:
            score_gain = baseline_total - candidate_total
            free_wR2_loss = candidate_free - baseline_free
            all_R1_loss = candidate_r1 - baseline_r1
            if score_gain < float(args.continuous_thickness_min_gain):
                promotion_reasons.append("full_row_gain_below_minimum")
            if free_wR2_loss > float(args.continuous_thickness_max_free_wr2_loss):
                promotion_reasons.append("full_row_free_wR2_loss")
            if all_R1_loss > float(args.continuous_thickness_max_all_r1_loss):
                promotion_reasons.append("full_row_all_R1_loss")
    accepted = bool(args.dry_run or not promotion_reasons)
    selected_run = materialized if accepted else Path(baseline_run)
    selected_nuisance = dict(materialized_nuisance or nuisance) if accepted else best_nuisance(Path(baseline_run))
    if not args.dry_run and not selected_nuisance:
        raise RuntimeError(
            "continuous-thickness full-row rejection could not restore the preceding nuisance contract: "
            f"{baseline_summary_path}"
        )
    return (
        selected_nuisance,
        selected_run,
        {
            "family": "continuous_thickness_materialization",
            "status": "dry_run" if args.dry_run else ("accepted_full_row" if accepted else "rejected_full_row"),
            "source": str(continuous_run),
            "run": str(materialized),
            "row_table": str(row_table),
            "baseline_run": str(baseline_run),
            "selected_run": str(selected_run),
            "full_row_contract": True,
            "nuisance_contract_verified": not mismatches,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "full_row_score_gain": score_gain,
            "full_row_free_wR2_loss": free_wR2_loss,
            "full_row_all_R1_loss": all_R1_loss,
            "baseline_free_rows": baseline_free_rows,
            "candidate_free_rows": candidate_free_rows,
            "min_promotion_gain": float(args.continuous_thickness_min_gain),
            "max_free_wR2_loss": float(args.continuous_thickness_max_free_wr2_loss),
            "max_all_R1_loss": float(args.continuous_thickness_max_all_r1_loss),
            "promotion_reasons": promotion_reasons,
        },
    )
