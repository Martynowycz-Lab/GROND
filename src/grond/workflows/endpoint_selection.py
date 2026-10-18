"""Run endpoint calibration and choose a usable, provenance-bound final endpoint."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support import execution
from grond.cli_support.execution import run_logged
from grond.cli_support.values import try_read_json
from grond.io.config import PrepConfig
from grond.paths import engine_script
from grond.workflows.conventional_baseline import (
    _endpoint_baseline_no_harm_gate,
    _validated_fresh_conventional_baseline,
)
from grond.workflows.decision import safe_float, safe_label, write_json
from grond.workflows.endpoint_candidates import (
    _endpoint_calibration_chooser_candidate,
    _multi_feedback_endpoint_candidates,
    _publication_model_endpoint_candidate,
)
from grond.workflows.endpoint_chooser import write_publication_endpoint_choice
from grond.workflows.endpoint_commands import legacy_handoff_conflicts
from grond.workflows.nuisance_policy import resolved_post_bloch_sigma_policy
from grond.workflows.report_commands import publication_report_cmd
from grond.workflows.scale_model_policy import resolved_endpoint_calibration_mode, resolved_scale_polish_mode


def run_publication_report(python: str, prep: PrepConfig, result: dict, args: argparse.Namespace, logs: Path) -> dict:
    """Write top-level MD/TeX/PDF publication tables for a completed run."""

    if args.dry_run:
        return {"status": "dry_run"}
    if not (result.get("primary_run") or result.get("runs")):
        return {"status": "no_final_row_tables"}
    out_dir = Path(args.out_dir) / "tables"
    cmd = publication_report_cmd(
        python,
        prep,
        result,
        out_dir,
        title=f"GROND regularization summary: {', '.join(ds.name for ds in prep.datasets)}",
        instability_u=float(getattr(args, "instability_u", 0.01)),
    )
    run_logged(cmd, logs, "publication_row_mr_tables", dry_run=False)
    artifacts_path = out_dir / "publication_artifacts.json"
    artifacts = try_read_json(artifacts_path)
    if artifacts:
        if result.get("resolution_limits"):
            artifacts["resolution_limits"] = result.get("resolution_limits")
        if result.get("merged_resolution_limit"):
            artifacts["merged_resolution_limit"] = result.get("merged_resolution_limit")
        if result.get("dataset_policy"):
            artifacts["dataset_policy"] = result.get("dataset_policy")
        if result.get("publication_runs"):
            artifacts["publication_runs"] = result.get("publication_runs")
        if result.get("publication_full_runs"):
            artifacts["publication_full_runs"] = result.get("publication_full_runs")
        if result.get("scale_contract_selection"):
            artifacts["scale_contract_selection"] = result.get("scale_contract_selection")
        with execution._RUN_LOG_LOCK:
            write_json(artifacts_path, artifacts)
            if execution._PROGRESS is not None and not execution._PROGRESS.refresh_output_fingerprints(
                "publication_row_mr_tables"
            ):
                raise RuntimeError(
                    "publication report was written, but its resumable output fingerprint could not be refreshed"
                )
        return {"status": "ok", **artifacts}
    return {
        "status": "missing_artifact_summary",
        "markdown": str((out_dir / "publication_table.md").resolve()),
        "latex": str((out_dir / "publication_table.tex").resolve()),
        "summary": str((out_dir / "row_mr_metrics.json").resolve()),
    }


def _multi_hand_publication_blockers(result: Mapping[str, object]) -> list[str]:
    """Return fail-closed publication blockers from the final shared-hand audit."""

    if not result.get("combined"):
        return []
    raw = result.get("post_feedback_multi_hand_diagnostic")
    diagnostic = raw if isinstance(raw, Mapping) else {}
    if not diagnostic:
        path_text = str(result.get("post_feedback_multi_hand_diagnostic_path") or "").strip()
        diagnostic = try_read_json(Path(path_text)) if path_text else {}
    status = str(diagnostic.get("status") or "missing")
    method = str(diagnostic.get("method") or "")
    final_datasets = diagnostic.get("datasets") if isinstance(diagnostic.get("datasets"), Mapping) else {}
    consistency = diagnostic.get("consistency_gate") if isinstance(diagnostic.get("consistency_gate"), Mapping) else {}
    symmetry = diagnostic.get("symmetry_policy") if isinstance(diagnostic.get("symmetry_policy"), Mapping) else {}
    explicitly_not_applicable = status in {"not_applicable", "skipped_not_applicable"} and (
        str(consistency.get("status") or "") == "not_applicable"
        or (method == "nakane_negative_thickness_elastic_pair_v1" and symmetry.get("run_hand_control") is False)
    )
    completed_final_elastic_selection = bool(
        method == "nakane_negative_thickness_elastic_pair_v1"
        and status in {"kept", "switched"}
        and isinstance(diagnostic.get("aggregate"), Mapping)
        and bool(final_datasets)
        and all(
            isinstance(record, Mapping) and record.get("status") == "complete" for record in final_datasets.values()
        )
    )
    completed_conditional_diagnostic = bool(final_datasets) and all(
        isinstance(record, Mapping)
        and record.get("status") in {"consistent_no_material_opposite_support", "warning_material_opposite_support"}
        for record in final_datasets.values()
    )
    if (
        status == "consistent"
        or explicitly_not_applicable
        or completed_final_elastic_selection
        or completed_conditional_diagnostic
    ):
        return []
    warnings = [str(value) for value in (consistency.get("warnings") or diagnostic.get("warnings") or []) if str(value)]
    detail = "; ".join(warnings[:4]) or str(
        diagnostic.get("reason") or "the final matched hand comparison is missing or incomplete"
    )
    return [
        f"shared post-feedback hand diagnostic status {status!r} is not publication-consistent: {detail}; "
        "GROND cannot verify the final conditional hand representation"
    ]


def run_publication_endpoint_chooser(prep: PrepConfig, result: dict, args: argparse.Namespace) -> dict:
    """Write the publication-vs-diagnostic endpoint chooser report."""

    if not getattr(args, "publication_endpoint_chooser", True):
        return {"status": "disabled"}
    out_dir = Path(getattr(args, "out_dir", ".")) / "endpoint_chooser"
    if getattr(args, "dry_run", False):
        return {"status": "dry_run", "out_dir": str(out_dir)}
    if result.get("combined"):
        runs = {name: Path(run) for name, run in (result.get("publication_runs") or result.get("runs") or {}).items()}
        combined_run = Path(result["combined"])
    elif result.get("primary_run"):
        label = prep.datasets[0].name if prep.datasets else "dataset"
        runs = {label: Path(result["primary_run"])}
        combined_run = None
    else:
        return {"status": "no_endpoint_runs"}
    if not runs and not combined_run:
        return {"status": "no_endpoint_runs"}
    extra_candidates = []
    calibration_candidate = _endpoint_calibration_chooser_candidate(result)
    if calibration_candidate:
        extra_candidates.append(calibration_candidate)
    model_candidate = _publication_model_endpoint_candidate(result)
    if model_candidate:
        extra_candidates.append(model_candidate)
    extra_candidates.extend(
        _multi_feedback_endpoint_candidates(
            result, Path(getattr(args, "out_dir", ".")), str(getattr(args, "publication_feedback_cycles", "all") or "all")
        )
    )
    combined_baseline = _validated_fresh_conventional_baseline(result) if result.get("combined") else {}
    global_publication_blockers = _multi_hand_publication_blockers(result)
    return write_publication_endpoint_choice(
        runs,
        out_dir,
        combined_run=combined_run,
        extra_candidates=extra_candidates,
        min_hkl_safe_factor=getattr(args, "publication_endpoint_min_hkl_safe_factor", 0.25),
        max_modulation_p95=getattr(args, "publication_endpoint_max_modulation_p95", 20.0),
        publication_goof_min=getattr(args, "endpoint_calibration_goof_min", 0.85),
        publication_goof_max=getattr(args, "endpoint_calibration_goof_max", 1.15),
        allow_npd=bool(getattr(args, "publication_allow_npd", False)),
        combined_baseline_r1_all=(
            safe_float(combined_baseline.get("R1_all"), float("nan")) if combined_baseline else None
        ),
        combined_baseline_source=str(combined_baseline.get("baseline_source") or ""),
        require_combined_baseline=bool(result.get("combined")),
        global_publication_blockers=global_publication_blockers,
    )


def _endpoint_calibration_lambda(summary: Mapping[str, object], args: argparse.Namespace) -> float:
    """Return the diagnostic q exponent to replay for a frozen row table."""

    del summary
    value = safe_float(getattr(args, "endpoint_calibration_lambda_q", 1.0), 1.0)
    return value if value > 0.0 else 1.0


def _recorded_native_row_tables(
    result: Mapping[str, object],
    *,
    include_endpoint_only: bool = False,
) -> dict[str, str]:
    """Return the native row tables that define the raw-row result.

    ``publication_runs`` excludes completeness-only crystals from raw-row
    metrics while retaining primary and downweighted datasets.  The broader
    ``combined_input_runs`` contract is useful only when endpoint-completeness
    provenance is explicitly requested.
    """

    if result.get("combined"):
        runs = (
            (result.get("combined_input_runs") if include_endpoint_only else result.get("publication_runs"))
            or result.get("combined_input_runs")
            or result.get("runs")
            or {}
        )
        if isinstance(runs, Mapping):
            return {
                str(name): str(Path(str(run_dir)).expanduser() / "native_dyn_calc.csv")
                for name, run_dir in sorted(runs.items(), key=lambda item: str(item[0]))
                if str(run_dir)
            }
        return {}
    primary = str(result.get("primary_run") or "").strip()
    return {"dataset": str(Path(primary).expanduser() / "native_dyn_calc.csv")} if primary else {}


def _multi_endpoint_calibration_not_applicable(result: Mapping[str, object]) -> dict[str, object]:
    """Describe why single-table scale replay is unsafe for a multi run."""

    combined = Path(str(result.get("combined") or "")).expanduser()
    row_tables = _recorded_native_row_tables(result, include_endpoint_only=True)
    return {
        "status": "not_applicable_multi_dataset",
        "reason": (
            "Multi-dataset scale contracts remain independent by crystal. The single-table endpoint-calibration "
            "replay would incorrectly share frame groups and detector scales, and would bypass the validated "
            "per-dataset sigma, policy-weight, overlap-scale, symmetry-merge, and overdispersion sequence. "
            "The combined corrected-HKL endpoint and publication chooser remain authoritative."
        ),
        "combined_summary": str(combined / "combined_summary.json") if str(combined) not in {"", "."} else "",
        "row_tables": row_tables,
        "row_tables_exist": {name: Path(path).exists() for name, path in row_tables.items()},
        "future_contract": "replay each scale candidate independently per dataset, then evaluate through the normal combined endpoint path",
    }


def run_endpoint_calibration(prep: PrepConfig, result: dict, args: argparse.Namespace, python: str, logs: Path) -> dict:
    """Replay the final frozen row table through the endpoint-calibration chooser."""

    mode, off_reason = resolved_endpoint_calibration_mode(args)
    if mode == "off":
        return {"status": "disabled", **({"reason": off_reason} if off_reason else {})}
    if mode == "auto" and getattr(args, "fast", False):
        return {"status": "disabled_fast_mode"}
    if args.dry_run:
        return {"status": "dry_run", "out_dir": str(Path(args.out_dir) / "endpoint_calibration")}
    if result.get("combined"):
        if mode == "on":
            raise RuntimeError(
                "--endpoint-calibration on is not supported for multi-dataset runs: scale contracts must be "
                "replayed independently per dataset and then passed through the normal combined endpoint. "
                "Use --endpoint-calibration auto (safe explicit skip) or calibrate each dataset separately."
            )
        return _multi_endpoint_calibration_not_applicable(result)
    conflicts = legacy_handoff_conflicts(args)
    if conflicts:
        reason = (
            "The endpoint-calibration replay writes its corrected HKL and runs SHELXL with the legacy handoff "
            "contract and cannot honour: " + "; ".join(conflicts) + ". The feedback-loop model endpoint and the "
            "publication chooser remain authoritative."
        )
        if mode == "on":
            raise RuntimeError(
                f"--endpoint-calibration on: {reason} To calibrate, select the single-step handoff with "
                "--shelxl-feedback-shelxl-iterations 1 --shelxl-feedback-endpoint-ls-cycles 1 "
                "--shelxl-convergence-shift-esd 0 --shelxl-feedback-xyz-tol 0 --shelxl-npd-recovery damp "
                "--shelxl-carry-damp --endpoint-include-free-rows --endpoint-q-policy clip "
                "--endpoint-intensity-scale median-iobs --endpoint-wght-mode freeze-zero --dataset-sigma-policy auto, "
                "or use --endpoint-calibration auto (explicit skip)."
            )
        return {"status": "not_applicable_handoff_contract", "reason": reason, "conflicts": conflicts}
    label = "combined"
    run_dir: Path | None = None
    if result.get("combined"):
        run_dir = Path(str(result["combined"]))
    elif result.get("primary_run"):
        label = prep.datasets[0].name if prep.datasets else "dataset"
        run_dir = Path(str(result["primary_run"]))
    if run_dir is None:
        return {"status": "no_frozen_row_table"}
    row_table = run_dir / "native_dyn_calc.csv"
    summary_path = run_dir / "summary.json"
    if not row_table.exists() or not summary_path.exists():
        return {
            "status": "missing_frozen_row_table",
            "row_table": str(row_table),
            "summary": str(summary_path),
        }
    summary = try_read_json(summary_path)
    lambda_q = _endpoint_calibration_lambda(summary, args)
    out_dir = Path(args.out_dir) / "endpoint_calibration" / safe_label(label)
    cmd = [
        python,
        str(engine_script("scale_contract_replay.py")),
        "--row-table",
        str(row_table),
        "--summary",
        str(summary_path),
        "--out-dir",
        str(out_dir),
        "--specs",
        str(getattr(args, "endpoint_calibration_specs", "")),
        "--lambda-q",
        f"{lambda_q:g}",
        "--post-bloch-sigma-policy",
        resolved_post_bloch_sigma_policy(args, final_endpoint=True),
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{getattr(args, 'endpoint_calibration_sigma_min_scale', 0.05):g}",
        "--post-bloch-sigma-max-scale",
        f"{getattr(args, 'endpoint_calibration_sigma_max_scale', 4.0):g}",
        "--shelxl",
        args.shelxl,
        "--shelxl-iterations",
        str(getattr(args, "endpoint_calibration_shelxl_iterations", 1)),
        "--wght-modes",
        str(getattr(args, "endpoint_calibration_wght_modes", "freeze-zero,promote")),
        "--publication-goof-min",
        f"{getattr(args, 'endpoint_calibration_goof_min', 0.85):g}",
        "--publication-goof-max",
        f"{getattr(args, 'endpoint_calibration_goof_max', 1.15):g}",
        "--publication-max-r1-cost",
        f"{getattr(args, 'endpoint_calibration_max_r1_cost', 0.010):g}",
        "--publication-free-wr2-tolerance",
        f"{getattr(args, 'endpoint_calibration_free_wr2_tolerance', 0.040):g}",
        "--publication-free-rall-tolerance",
        f"{getattr(args, 'endpoint_calibration_free_rall_tolerance', 0.010):g}",
        "--publication-work-rall-tolerance",
        f"{getattr(args, 'endpoint_calibration_work_rall_tolerance', 0.010):g}",
        "--publication-free-wrall-tolerance",
        f"{getattr(args, 'endpoint_calibration_free_wrall_tolerance', 0.050):g}",
        "--publication-max-modulation-p95",
        f"{getattr(args, 'publication_endpoint_max_modulation_p95', 20.0):g}",
        "--publication-prefer-wght-mode",
        str(getattr(args, "endpoint_calibration_prefer_wght_mode", "freeze-zero")),
        "--publication-selection-policy",
        str(getattr(args, "endpoint_calibration_selection_policy", "gated-smallest")),
        "--publication-information-criterion",
        str(getattr(args, "endpoint_calibration_information_criterion", "bic_fitted_variance")),
        "--instability-u",
        f"{getattr(args, 'instability_u', 0.01):g}",
        "--detector-ridge",
        f"{getattr(args, 'detector_scale_ridge', 0.05):g}",
        "--detector-smoothness",
        f"{getattr(args, 'detector_scale_smoothness', 0.10):g}",
        "--detector-min-rows",
        str(getattr(args, "detector_scale_min_rows", 20)),
        "--detector-min-holdout-improvement",
        f"{getattr(args, 'detector_scale_min_holdout_improvement', 0.002):g}",
        "--detector-required-improved-splits",
        str(getattr(args, "detector_scale_required_improved_splits", 2)),
        "--detector-max-bloch-gain-loss",
        f"{getattr(args, 'detector_scale_max_bloch_gain_loss', 0.002):g}",
        "--detector-min-field-correlation",
        f"{getattr(args, 'detector_scale_min_field_correlation', 0.50):g}",
        "--detector-max-field-log-rms",
        f"{getattr(args, 'detector_scale_max_field_log_rms', 0.10):g}",
    ]
    if bool(getattr(args, "shelxl_carry_damp", False)):
        cmd.append("--keep-damp")
    if bool(getattr(args, "publication_allow_npd", False)):
        cmd.append("--publication-allow-npd")
    if bool(getattr(args, "detector_scale_independent_calibration", False)):
        cmd.append("--detector-independent-calibration")
    try:
        run_logged(cmd, logs, f"{safe_label(label)}_endpoint_calibration", dry_run=False)
    except subprocess.CalledProcessError as exc:
        if mode == "on":
            raise
        return {"status": "failed", "returncode": exc.returncode, "command": exc.cmd, "out_dir": str(out_dir)}
    replay_summary = try_read_json(out_dir / "summary.json")
    selection = replay_summary.get("publication_selection", {})
    selection = dict(selection) if isinstance(selection, Mapping) else {}
    endpoint = selection.get("publication_endpoint")
    endpoint = endpoint if isinstance(endpoint, Mapping) else {}
    if endpoint:
        no_harm_gate = _endpoint_baseline_no_harm_gate(summary, endpoint)
        selection["baseline_no_harm_gate"] = no_harm_gate
        if bool(no_harm_gate.get("passed", False)):
            checked_endpoint = dict(endpoint)
            checked_endpoint["baseline_no_harm_gate"] = no_harm_gate
            selection["publication_endpoint"] = checked_endpoint
        else:
            selection["status"] = "failed_baseline_no_harm_gate"
            selection["baseline_rejected_endpoint"] = dict(endpoint)
            selection.pop("publication_endpoint", None)
    return {
        "status": "ok",
        "label": label,
        "run_dir": str(run_dir),
        "lambda_q": lambda_q,
        "out_dir": str(out_dir),
        "report": str((out_dir / "REPORT.md").resolve()),
        "endpoint_csv": str((out_dir / "scale_contract_endpoint_replay.csv").resolve()),
        "publication_selection": selection,
    }


def run_scale_window_polish(
    python: str, run_dir: Path, out_dir: Path, logs: Path, label: str, args: argparse.Namespace
) -> dict:
    """Run the post-hoc scale-window compression gate for a finished row table."""

    mode, off_reason = resolved_scale_polish_mode(args)
    if mode == "off":
        return {"status": "disabled", **({"reason": off_reason} if off_reason else {})}
    row_table = run_dir / "native_dyn_calc.csv"
    summary = run_dir / "summary.json"
    if args.dry_run:
        return {"status": "dry_run", "row_table": str(row_table), "out_dir": str(out_dir)}
    if not row_table.exists():
        return {"status": "missing_row_table", "row_table": str(row_table)}
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(engine_script("scale_compression_polish.py")),
        "--row-table",
        str(row_table),
        "--summary",
        str(summary),
        "--out-dir",
        str(out_dir),
        "--title",
        f"GROND scale-window polish: {label}",
        "--group-strides",
        "1",
        "--window-scan-deg",
        str(getattr(args, "scale_polish_window_deg", 5.0)),
        "--detector-bins",
        getattr(args, "scale_polish_detector_bins", "0,2,4,8"),
        "--detector-ridge",
        str(getattr(args, "detector_scale_ridge", 0.05)),
        "--detector-smoothness",
        str(getattr(args, "detector_scale_smoothness", 0.10)),
        "--detector-min-rows",
        str(getattr(args, "detector_scale_min_rows", 20)),
        "--detector-min-holdout-improvement",
        str(getattr(args, "detector_scale_min_holdout_improvement", 0.002)),
        "--detector-required-improved-splits",
        str(getattr(args, "detector_scale_required_improved_splits", 2)),
        "--detector-max-bloch-gain-loss",
        str(getattr(args, "detector_scale_max_bloch_gain_loss", 0.002)),
        "--detector-min-field-correlation",
        str(getattr(args, "detector_scale_min_field_correlation", 0.50)),
        "--detector-max-field-log-rms",
        str(getattr(args, "detector_scale_max_field_log_rms", 0.10)),
        "--scale-objective",
        getattr(args, "scale_polish_objective", "row_mr"),
        "--tolerance",
        str(getattr(args, "scale_polish_tolerance", 0.01)),
        "--selection-metrics",
        getattr(
            args,
            "scale_polish_selection_metrics",
            "free_Rall,work_Rall",
        ),
        "--frame-models",
        getattr(args, "scale_polish_frame_models", "adaptive,fixed"),
        "--adaptive-frame-counts",
        getattr(args, "scale_polish_adaptive_counts", "auto"),
    ]
    if getattr(args, "scale_polish_scan_offsets", False):
        cmd.append("--scan-offsets")
    if getattr(args, "detector_scale_independent_calibration", False):
        cmd.append("--detector-independent-calibration")
    try:
        run_logged(cmd, logs, f"{safe_label(label)}_scale_window_polish", dry_run=False)
    except subprocess.CalledProcessError as exc:
        if mode == "on":
            raise
        return {"status": "failed", "returncode": exc.returncode, "command": exc.cmd, "out_dir": str(out_dir)}
    summary_payload = try_read_json(out_dir / "scale_compression_summary.json")
    selected = summary_payload.get("selected")
    selected_output = summary_payload.get("selected_output")
    widest = summary_payload.get("selected_widest")
    return {
        "status": "accepted" if selected else "diagnostic_only",
        "out_dir": str(out_dir),
        "report": str((out_dir / "REPORT.md").resolve()),
        "candidate_csv": str((out_dir / "scale_compression_candidates.csv").resolve()),
        "selected_compact_run": str((out_dir / "selected_compact_run").resolve())
        if (out_dir / "selected_compact_run").exists()
        else "",
        "selected_strict_run": str((out_dir / "selected_strict_run").resolve())
        if (out_dir / "selected_strict_run").exists()
        else "",
        "selected": selected or {},
        "selected_output": selected_output or {},
        "baseline": summary_payload.get("baseline") or {},
        "selected_information": summary_payload.get("selected_information") or {},
        "selected_widest": widest or {},
    }
