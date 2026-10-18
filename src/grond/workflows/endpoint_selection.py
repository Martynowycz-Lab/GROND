"""Choose a usable, provenance-bound final endpoint and write the publication tables."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support import execution
from grond.cli_support.execution import run_logged
from grond.cli_support.values import try_read_json
from grond.io.config import PrepConfig
from grond.workflows.conventional_baseline import _validated_fresh_conventional_baseline
from grond.workflows.decision import safe_float, write_json
from grond.workflows.endpoint_candidates import (
    _multi_feedback_endpoint_candidates,
    _publication_model_endpoint_candidate,
)
from grond.workflows.endpoint_chooser import write_publication_endpoint_choice
from grond.workflows.report_commands import publication_report_cmd


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
        allow_npd=bool(getattr(args, "publication_allow_npd", False)),
        combined_baseline_r1_all=(
            safe_float(combined_baseline.get("R1_all"), float("nan")) if combined_baseline else None
        ),
        combined_baseline_source=str(combined_baseline.get("baseline_source") or ""),
        require_combined_baseline=bool(result.get("combined")),
        global_publication_blockers=global_publication_blockers,
    )


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
    """The endpoint-calibration record grond finish writes for a multi run; GROND runs no scale replay."""

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
