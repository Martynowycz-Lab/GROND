"""Assemble final statistics and artifact links for the user-facing report."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.io.config import PrepConfig
from grond.progress import format_seconds
from grond.reports.bloch_diagnostics import _write_bloch_diagnostics_reports
from grond.reports.distributions import _quick_residual_z_summary
from grond.reports.endpoints import (
    _chooser_diagnostic_endpoint,
    _chooser_publication_endpoint,
    _quick_endpoint_file_block,
)
from grond.reports.hand import (
    _quick_final_elastic_hand_aggregate,
    _quick_final_elastic_hand_zkp,
    _quick_final_hand_summary,
    _quick_hand_t_style_text,
    _quick_multi_hand_console_rows,
    _quick_multi_hand_diagnostic,
    _quick_multi_hand_evidence_lines,
)
from grond.reports.heldout import heldout_markdown_lines, heldout_report_for_run, terminal_block
from grond.reports.nuisance import _quick_dataset_nuisance_table_lines, _quick_thickness_text
from grond.reports.progress import _quick_progress_finish_info, _quick_progress_source_roots, _quick_progress_timings
from grond.reports.shelxl import (
    _quick_baseline_shelxl,
    _quick_goof_warning,
    _quick_selected_feedback_shelxl,
    _quick_shelxl_details,
    _quick_shelxl_issue_text,
    _quick_shelxl_weight_text,
)
from grond.reports.source_controls import (
    _quick_delta_percent,
    _quick_flat_hand_t_style_text,
    _quick_row_weight_definition,
    _quick_source_shelx_control_text,
    _quick_source_shelx_controls,
    _quick_transform_summary,
)
from grond.reports.table1 import (
    _quick_complete_jana_metrics,
    _quick_final_jana_row_summary,
    _table1_pair,
    _write_final_statistics_table1,
)
from grond.reports.values import (
    _quick_best_endpoint,
    _quick_best_native,
    _quick_born_metric,
    _quick_count_pair,
    _quick_dataset_nuisance_contracts,
    _quick_dyn_metric,
    _quick_final_run,
    _quick_fraction,
    _quick_int,
    _quick_load_run_summary,
    _quick_metric,
    _quick_nuisance,
    _quick_number,
    _quick_percent,
)
from grond.workflows.conventional_baseline import (
    _endpoint_baseline_no_harm_gate,
    _validated_fresh_conventional_baseline,
)
from grond.workflows.endpoint_selection import _recorded_native_row_tables
from grond.workflows.finish_maps import _publication_endpoint_from_result

REPORT_LAYOUTS = ("heldout-first", "legacy")


def resolve_report_layout(report_layout: str | None, result: Mapping[str, object]) -> str:
    """AUTO writes heldout-first; ``grond finish`` reuses the layout the run recorded (legacy when none was)."""

    value = report_layout
    if value is None:
        quick = result.get("quick_stats") if isinstance(result.get("quick_stats"), Mapping) else {}
        console = quick.get("console") if isinstance(quick.get("console"), Mapping) else {}
        value = console.get("report_layout") or "legacy"
    if str(value) not in REPORT_LAYOUTS:
        raise ValueError(f"unknown report layout {value!r}; expected one of {list(REPORT_LAYOUTS)}")
    return str(value)


def hand_decision_authority(heldout_report: Mapping[str, object] | None) -> str:
    """The rule that decided the hand, as recorded in decisions/hand_decision_heldout_<stage>.json.

    A skipped stage (centrosymmetric model, hand control off) evaluated no rule. When every recorded stage was
    skipped the result is ``not_evaluated``, never the rule named by the skipped stage's empty contract.
    """

    stages = ((heldout_report or {}).get("hand") or {}).get("stages") or {}
    recorded = [stages[stage] for stage in ("final", "early") if isinstance(stages.get(stage), Mapping)]
    evaluated = [doc for doc in recorded if not str(doc.get("status", "")).startswith("skipped")]
    for doc in evaluated:
        if doc.get("decision_authority"):
            return str(doc["decision_authority"])
    return "not_evaluated" if recorded and not evaluated else "unrecorded"


def write_quick_stats(
    prep: PrepConfig, result: Mapping[str, object], args: argparse.Namespace, *, report_layout: str | None = None
) -> dict:
    """Write a short human-first statistics summary at the run root."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    quick_path = out_dir / "QUICK_STATS.md"
    final_summary_path = out_dir / "FINAL_SUMMARY.md"
    row_metrics_path = out_dir / "tables" / "row_mr_metrics.json"
    row_metrics = try_read_json(row_metrics_path)
    endpoint_calibration = (
        result.get("endpoint_calibration") if isinstance(result.get("endpoint_calibration"), Mapping) else {}
    )
    endpoint_selection = (
        endpoint_calibration.get("publication_selection", {}) if isinstance(endpoint_calibration, Mapping) else {}
    )
    publication_endpoint = _publication_endpoint_from_result(result)
    diagnostic_endpoint = _chooser_diagnostic_endpoint(result)
    metadata = row_metrics.get("metadata", {}) if isinstance(row_metrics.get("metadata"), Mapping) else {}
    source_roots = _quick_progress_source_roots(out_dir)
    timing_rows = _quick_progress_timings(out_dir)
    finish_info = _quick_progress_finish_info(out_dir)
    sfac_check = try_read_json(out_dir / "sfac_check.json")
    final_run = _quick_final_run(result)
    final_model = Path(str(result.get("model"))) if result.get("model") else None
    final_summary = _quick_load_run_summary(final_run)
    dataset_nuisance_contracts = _quick_dataset_nuisance_contracts(final_summary)
    recorded_row_tables = _recorded_native_row_tables(result)
    missing_row_tables = {
        name: path for name, path in recorded_row_tables.items() if not Path(path).expanduser().is_file()
    }
    if missing_row_tables:
        missing_text = ", ".join(f"{name}={path}" for name, path in missing_row_tables.items())
        raise RuntimeError(
            "cannot write a coherent final report because a recorded native row table is missing; "
            "refusing to mix saved joint metrics with partial diagnostics: " + missing_text
        )
    frozen_row_tables = [(name, Path(path).expanduser()) for name, path in recorded_row_tables.items()]
    if not publication_endpoint:
        rejected_endpoint = _chooser_publication_endpoint(result)
        rejected_gate = _endpoint_baseline_no_harm_gate(
            final_summary,
            rejected_endpoint,
            baseline_override=_validated_fresh_conventional_baseline(result),
            require_baseline=bool(result.get("combined")),
        )
        if rejected_endpoint and str(rejected_gate.get("status")) == "failed":
            diagnostic_endpoint = dict(rejected_endpoint)
            diagnostic_endpoint["baseline_no_harm_gate"] = rejected_gate
            diagnostic_endpoint["rejection_reasons"] = list(rejected_gate.get("reasons") or [])
    best_native = _quick_best_native(final_summary)
    best_endpoint = _quick_best_endpoint(final_summary)
    native_metrics = best_native.get("metrics") if isinstance(best_native.get("metrics"), Mapping) else {}
    nuisance = _quick_nuisance(final_summary)
    (
        hand_summary,
        multi_hand_diagnostic,
        multi_hand_pooled,
        multi_hand_rows,
        multi_hand_zkp,
        promotion_decision,
        screen_summary,
        transform_summary,
    ) = _prepare_quick_stats_hand_context(
        out_dir=out_dir,
        prep=prep,
        result=result,
    )

    z_summary = _quick_residual_z_summary([path for _name, path in frozen_row_tables])
    free_z = z_summary.get("free", {})
    all_z = z_summary.get("all", {})
    hand_blocks = (
        hand_summary.get("diagnostic_blocks") if isinstance(hand_summary.get("diagnostic_blocks"), Mapping) else {}
    )
    hand_all = hand_blocks.get("all", {}) if isinstance(hand_blocks.get("all"), Mapping) else {}
    hand_work = hand_blocks.get("work", {}) if isinstance(hand_blocks.get("work"), Mapping) else {}
    hand_free = hand_blocks.get("free", {}) if isinstance(hand_blocks.get("free"), Mapping) else {}
    file_block = _quick_endpoint_file_block(publication_endpoint, final_model)
    all_data = result.get("final_all_data_refinement") if isinstance(result.get("final_all_data_refinement"), Mapping) else {}
    if all_data.get("status") == "ok":
        # --free-set off: the result is the published endpoint refined against work and free reflections
        artifacts = all_data.get("artifacts") or {}
        file_block.update({f"all_data_{suffix}": str(artifacts.get(suffix, "")) for suffix in ("res", "hkl", "lst")})
    bloch_diagnostics_path = _write_bloch_diagnostics_reports(
        out_dir,
        frozen_row_tables,
        final_run,
        final_summary,
        row_metrics,
        file_block,
    )
    file_block["bloch_diagnostics"] = str(bloch_diagnostics_path)
    input_shelxl = _quick_baseline_shelxl(prep, final_summary, result)
    file_block["baseline_res"] = str(input_shelxl.get("res") or "")
    file_block["baseline_hkl"] = str(input_shelxl.get("hkl") or "")
    file_block["baseline_lst"] = str(input_shelxl.get("lst") or "")
    source_structure = Path(prep.structure).expanduser() if prep.structure else None
    source_controls = _quick_source_shelx_controls(source_structure)
    input_res_text = str(input_shelxl.get("res") or "").strip()
    input_res = Path(input_res_text).expanduser() if input_res_text else None
    input_controls = (
        _quick_source_shelx_controls(input_res) if input_res is not None and input_res.is_file() else source_controls
    )
    best_endpoint_shelxl = best_endpoint.get("shelxl") if isinstance(best_endpoint.get("shelxl"), Mapping) else {}
    final_feedback_shelxl = _quick_selected_feedback_shelxl(result, final_model, best_endpoint_shelxl)
    publication_shelxl = _quick_shelxl_details(
        {
            "R1_gt": publication_endpoint.get("shelxl_R1gt"),
            "R1_all": publication_endpoint.get("shelxl_R1all"),
            "wR2_all": publication_endpoint.get("shelxl_wR2all"),
            "GooF": publication_endpoint.get("shelxl_GooF"),
            "N_gt": publication_endpoint.get("shelxl_Ngt"),
            "N_all": publication_endpoint.get("shelxl_Nall"),
            "parameters": publication_endpoint.get("shelxl_parameters"),
            "restraints": publication_endpoint.get("shelxl_restraints"),
            "npd_count": publication_endpoint.get("shelxl_npd_count"),
            "npd_atoms": publication_endpoint.get("shelxl_npd_atoms"),
            "split_count": publication_endpoint.get("shelxl_split_count"),
            "has_npd": publication_endpoint.get("shelxl_has_npd"),
            "has_split_warnings": publication_endpoint.get("shelxl_has_split_warnings"),
            "pathology_status": publication_endpoint.get("pathology_status"),
            "active_wght_a": publication_endpoint.get("shelxl_active_wght_a"),
            "active_wght_b": publication_endpoint.get("shelxl_active_wght_b"),
            "recommended_wght_a": publication_endpoint.get("shelxl_recommended_wght_a"),
            "recommended_wght_b": publication_endpoint.get("shelxl_recommended_wght_b"),
            "lst": publication_endpoint.get("endpoint_lst"),
        },
        Path(file_block["publication_lst"]) if file_block.get("publication_lst") else None,
    )
    diagnostic_shelxl = _quick_shelxl_details(
        {
            "R1_gt": diagnostic_endpoint.get("shelxl_R1gt"),
            "R1_all": diagnostic_endpoint.get("shelxl_R1all"),
            "wR2_all": diagnostic_endpoint.get("shelxl_wR2all"),
            "GooF": diagnostic_endpoint.get("shelxl_GooF"),
            "parameters": diagnostic_endpoint.get("shelxl_parameters"),
            "restraints": diagnostic_endpoint.get("shelxl_restraints"),
            "npd_count": diagnostic_endpoint.get("shelxl_npd_count"),
            "split_count": diagnostic_endpoint.get("shelxl_split_count"),
            "lst": diagnostic_endpoint.get("endpoint_lst"),
        },
        Path(str(diagnostic_endpoint.get("endpoint_lst"))) if diagnostic_endpoint.get("endpoint_lst") else None,
    )
    report_layout = resolve_report_layout(report_layout, result)
    heldout_report = heldout_report_for_run(
        out_dir, result, frozen_row_tables, publication_shelxl=publication_shelxl, report_layout=report_layout
    )
    # Under the held-out orbit rule the all-row ZKP is a diagnostic; the labels follow the recorded rule.
    authority = hand_decision_authority(heldout_report)
    zkp_role = (
        "diagnostic only; the held-out orbit-sign rule decides the hand"
        if authority == "heldout_orbit_sign"
        else "no hand decision was evaluated (the hand check was skipped)"
        if authority == "not_evaluated" and report_layout == "heldout-first"
        else "the all-row value controls the 3.5 inversion decision"
    )
    endpoints = []
    resolution_block = row_metrics.get("row_resolution_scale_contracts")
    if not isinstance(resolution_block, Mapping) or not isinstance(resolution_block.get("contracts"), list):
        resolution_block = row_metrics.get("resolution_scale_endpoints", {})
    if isinstance(resolution_block, Mapping):
        raw_endpoints = resolution_block.get("contracts", resolution_block.get("endpoints", []))
        endpoints = raw_endpoints if isinstance(raw_endpoints, list) else []
    row_count_text = str(metadata.get("row_count", "n/a"))
    source_control_text = _quick_source_shelx_control_text(input_controls)
    endpoint_status_text = (
        "corrected endpoint accepted"
        if publication_endpoint
        else "no corrected endpoint passed the fresh conventional-baseline no-harm gate"
    )
    frozen_row_file_lines = [f"- Frozen Bloch row table ({name}): `{path}`" for name, path in frozen_row_tables] or [
        "- Frozen Bloch row tables: `none found`"
    ]
    frozen_row_console_text = "; ".join(f"{name}={path}" for name, path in frozen_row_tables)
    final_row_summary_path = (
        final_run / ("combined_summary.json" if result.get("combined") else "summary.json") if final_run else None
    )
    final_row_table_sources = "; ".join(f"{name}={path}" for name, path in frozen_row_tables)
    table1 = _write_final_statistics_table1(
        out_dir,
        row_metrics=row_metrics,
        row_metrics_path=row_metrics_path,
        input_shelxl=input_shelxl,
        publication_shelxl=publication_shelxl,
        publication_endpoint=publication_endpoint,
        file_block=file_block,
        input_controls=input_controls,
        final_row_table=frozen_row_console_text,
        heldout=heldout_report,
        all_data=all_data,
        unmerged_endpoint=(
            try_read_json(Path(str(result["combined"])) / "combined_summary.json").get("combined_unmerged_endpoint")
            if result.get("combined")
            else None
        ),
    )
    jana_summary = _quick_final_jana_row_summary(row_metrics)
    jana_dynamic = jana_summary.get("dynamic") if isinstance(jana_summary.get("dynamic"), Mapping) else {}
    jana_born = jana_summary.get("born") if isinstance(jana_summary.get("born"), Mapping) else {}
    if not jana_born and isinstance(jana_summary.get("kinematic"), Mapping):
        jana_born = jana_summary.get("kinematic", {})
    if not _quick_complete_jana_metrics(jana_dynamic):
        jana_dynamic = {}
    if not _quick_complete_jana_metrics(jana_born):
        jana_born = {}
    file_block["table1_markdown"] = str(table1["markdown"])
    file_block["table1_json"] = str(table1["json"])
    jana_md_path = row_metrics_path.parent / "JANA_ROW_STATISTICS.md"
    jana_json_path = row_metrics_path.parent / "JANA_ROW_STATISTICS.json"
    per_dataset_jana_md_path = row_metrics_path.parent / "JANA_ROW_STATISTICS_PER_DATASET.md"
    per_dataset_jana_json_path = row_metrics_path.parent / "JANA_ROW_STATISTICS_PER_DATASET.json"
    file_block["jana_row_statistics"] = str(jana_md_path if jana_md_path.is_file() else table1["markdown"])
    file_block["jana_row_statistics_json"] = str(jana_json_path if jana_json_path.is_file() else row_metrics_path)
    file_block["jana_row_statistics_per_dataset"] = (
        str(per_dataset_jana_md_path) if per_dataset_jana_md_path.is_file() else "not available for this run"
    )
    file_block["jana_row_statistics_per_dataset_json"] = (
        str(per_dataset_jana_json_path)
        if per_dataset_jana_json_path.is_file()
        else "not available for this run"
    )
    if result.get("combined") and multi_hand_zkp:
        hand_zkp_report_line = (
            "| final-hand adjusted ZKP, all/work/free | "
            f"`{_quick_number((multi_hand_zkp.get('all') or {}).get('z'), 3)}` / "
            f"`{_quick_number((multi_hand_zkp.get('work') or {}).get('z'), 3)}` / "
            f"`{_quick_number((multi_hand_zkp.get('free') or {}).get('z'), 3)}`; "
            "combined across physical datasets and oriented positive for the final selected hand; "
            f"{zkp_role} |"
        )
    elif result.get("combined"):
        hand_zkp_report_line = "| final-hand adjusted ZKP, all/work/free | `not available for this run` |"
    else:
        hand_zkp_report_line = (
            "| final-hand adjusted ZKP, all/work/free | "
            f"`{_quick_number(hand_all.get('adjusted_zkp_positive_favors_final'), 3)}` / "
            f"`{_quick_number(hand_work.get('adjusted_zkp_positive_favors_final'), 3)}` / "
            f"`{_quick_number(hand_free.get('adjusted_zkp_positive_favors_final'), 3)}`; "
            f"oriented positive for the final selected hand; {zkp_role} |"
        )

    if result.get("combined") and multi_hand_pooled:
        hand_t_report_line = (
            "| final hand paired t-style, all/work/free | "
            f"`{_quick_number((multi_hand_pooled.get('all') or {}).get('t'), 3)}` / "
            f"`{_quick_number((multi_hand_pooled.get('work') or {}).get('t'), 3)}` / "
            f"`{_quick_number((multi_hand_pooled.get('free') or {}).get('t'), 3)}`; "
            "exact pooled rows, positive favors final selected hand; n and df are reported below; "
            "not a calibrated normal Z score |"
        )
        hand_delta_report_line = (
            "| final hand mean delta chi2, all/work/free | "
            f"`{_quick_number((multi_hand_pooled.get('all') or {}).get('mean_delta_chi2_rejected_minus_final'), 3)}` / "
            f"`{_quick_number((multi_hand_pooled.get('work') or {}).get('mean_delta_chi2_rejected_minus_final'), 3)}` / "
            f"`{_quick_number((multi_hand_pooled.get('free') or {}).get('mean_delta_chi2_rejected_minus_final'), 3)}`; "
            "positive favors final selected hand |"
        )
    elif result.get("combined"):
        hand_t_report_line = (
            "| final hand paired t-style, all/work/free | "
            "`legacy per-dataset evidence with n and df below; no pooled statistic was recorded` |"
        )
        hand_delta_report_line = (
            "| final hand sqrt(delta chi2), all/work/free | "
            "`legacy per-dataset score margins below; no pooled statistic was recorded` |"
        )
    else:
        hand_t_report_line = (
            f"| final hand paired t-style, all/work/free | `{_quick_hand_t_style_text(hand_all)}` / "
            f"`{_quick_hand_t_style_text(hand_work)}` / "
            f"`{_quick_hand_t_style_text(hand_free)}`; positive favors final selected hand; "
            "not a calibrated normal Z score |"
        )
        hand_delta_report_line = (
            f"| final hand sqrt(delta chi2), all/work/free | `{_quick_number(hand_all.get('sqrt_positive_delta_chi2'), 3)}` / "
            f"`{_quick_number(hand_work.get('sqrt_positive_delta_chi2'), 3)}` / "
            f"`{_quick_number(hand_free.get('sqrt_positive_delta_chi2'), 3)}`; oriented as rejected-minus-final |"
        )

    lines = _start_quick_stats_markdown(
        heldout_lines=heldout_markdown_lines(heldout_report) if report_layout == "heldout-first" else [],
        dataset_nuisance_contracts=dataset_nuisance_contracts,
        endpoint_status_text=endpoint_status_text,
        file_block=file_block,
        finish_info=finish_info,
        metadata=metadata,
        nuisance=nuisance,
        out_dir=out_dir,
        prep=prep,
        publication_endpoint=publication_endpoint,
        result=result,
        source_roots=source_roots,
        table1=table1,
        timing_rows=timing_rows,
    )

    _append_quick_stats_model_panels(
        all_z=all_z,
        best_endpoint=best_endpoint,
        dataset_nuisance_contracts=dataset_nuisance_contracts,
        diagnostic_endpoint=diagnostic_endpoint,
        diagnostic_shelxl=diagnostic_shelxl,
        file_block=file_block,
        final_feedback_shelxl=final_feedback_shelxl,
        final_row_summary_path=final_row_summary_path,
        final_row_table_sources=final_row_table_sources,
        free_z=free_z,
        hand_delta_report_line=hand_delta_report_line,
        hand_summary=hand_summary,
        hand_t_report_line=hand_t_report_line,
        hand_zkp_report_line=hand_zkp_report_line,
        input_shelxl=input_shelxl,
        jana_born=jana_born,
        jana_dynamic=jana_dynamic,
        lines=lines,
        metadata=metadata,
        multi_hand_diagnostic=multi_hand_diagnostic,
        native_metrics=native_metrics,
        nuisance=nuisance,
        publication_endpoint=publication_endpoint,
        publication_shelxl=publication_shelxl,
        result=result,
        row_count_text=row_count_text,
        screen_summary=screen_summary,
        sfac_check=sfac_check,
        source_control_text=source_control_text,
        transform_summary=transform_summary,
    )

    if report_layout != "heldout-first":
        lines.extend(heldout_markdown_lines(heldout_report))

    _append_quick_stats_resolution_rows(
        endpoints=endpoints,
        lines=lines,
    )

    _append_quick_stats_endpoint_selection(
        bloch_diagnostics_path=bloch_diagnostics_path,
        endpoint_calibration=endpoint_calibration,
        endpoint_selection=endpoint_selection,
        file_block=file_block,
        frozen_row_file_lines=frozen_row_file_lines,
        lines=lines,
        out_dir=out_dir,
        publication_endpoint=publication_endpoint,
        publication_shelxl=publication_shelxl,
        row_metrics_path=row_metrics_path,
        table1=table1,
    )

    quick_path.write_text("\n".join(lines))

    console, selected_row = _prepare_quick_stats_console(
        diagnostic_endpoint=diagnostic_endpoint,
        endpoint_selection=endpoint_selection,
        file_block=file_block,
        final_feedback_shelxl=final_feedback_shelxl,
        final_summary=final_summary,
        final_summary_path=final_summary_path,
        frozen_row_console_text=frozen_row_console_text,
        frozen_row_tables=frozen_row_tables,
        hand_all=hand_all,
        hand_free=hand_free,
        hand_summary=hand_summary,
        input_shelxl=input_shelxl,
        jana_born=jana_born,
        jana_dynamic=jana_dynamic,
        jana_summary=jana_summary,
        multi_hand_diagnostic=multi_hand_diagnostic,
        multi_hand_pooled=multi_hand_pooled,
        multi_hand_rows=multi_hand_rows,
        multi_hand_zkp=multi_hand_zkp,
        native_metrics=native_metrics,
        out_dir=out_dir,
        promotion_decision=promotion_decision,
        publication_endpoint=publication_endpoint,
        publication_shelxl=publication_shelxl,
        quick_path=quick_path,
        result=result,
        row_metrics_path=row_metrics_path,
        screen_summary=screen_summary,
        table1=table1,
        transform_summary=transform_summary,
    )
    console["report_layout"] = report_layout
    console["hand_decision_authority"] = hand_decision_authority(heldout_report)
    console["heldout_validation"] = terminal_block(heldout_report)

    _write_quick_stats_final_summary(
        bloch_diagnostics_path=bloch_diagnostics_path,
        console=console,
        dataset_nuisance_contracts=dataset_nuisance_contracts,
        file_block=file_block,
        final_summary_path=final_summary_path,
        frozen_row_file_lines=frozen_row_file_lines,
        heldout_report=heldout_report,
        input_shelxl=input_shelxl,
        jana_born=jana_born,
        jana_dynamic=jana_dynamic,
        metadata=metadata,
        multi_hand_diagnostic=multi_hand_diagnostic,
        multi_hand_pooled=multi_hand_pooled,
        nuisance=nuisance,
        out_dir=out_dir,
        publication_shelxl=publication_shelxl,
        quick_path=quick_path,
        result=result,
        selected_row=selected_row,
        table1=table1,
    )

    return {"status": "ok", "path": str(quick_path), "final_summary": str(final_summary_path), "console": console}


def _prepare_quick_stats_hand_context(
    *,
    out_dir,
    prep,
    result,
):
    """Load transform and hand decisions before constructing any report tables."""

    decision_tree_path = Path(str(result.get("decision_tree", ""))) if result.get("decision_tree") else None
    if result.get("combined"):
        shared_path_text = str(result.get("shared_absolute_structure_decision") or "").strip()
        shared_decision = try_read_json(Path(shared_path_text)) if shared_path_text else {}
        decision_tree = {
            "selected_hand": str(result.get("selected_hand") or shared_decision.get("selected_hand") or "n/a"),
            "absolute_structure_decision": shared_decision,
        }
    elif decision_tree_path is not None:
        decision_tree = try_read_json(decision_tree_path)
    elif prep.datasets:
        decision_tree = try_read_json(out_dir / "decisions" / prep.datasets[0].name / "decision_tree.json")
    else:
        decision_tree = {}
    transform_decision = (
        decision_tree.get("transform_decision") if isinstance(decision_tree.get("transform_decision"), Mapping) else {}
    )
    if not transform_decision and prep.datasets:
        transform_decision = try_read_json(out_dir / "decisions" / prep.datasets[0].name / "transform_decision.json")
    transform_summary = _quick_transform_summary(transform_decision)
    per_dataset_transforms: dict[str, dict[str, object]] = {}
    if result.get("combined"):
        for ds in prep.datasets:
            transform_payload = try_read_json(out_dir / "decisions" / ds.name / "transform_decision.json")
            per_dataset_transforms[ds.name] = _quick_transform_summary(transform_payload)
        selected_transform_text = "; ".join(
            f"{name}={row.get('selected_transform', 'n/a')}" for name, row in per_dataset_transforms.items()
        )
        per_dataset_parity_text = "; ".join(
            (
                f"{name}={row.get('selected_transform', 'n/a')} "
                f"(det {row.get('transform_determinant', 'n/a')}, "
                f"{row.get('transform_parity', 'n/a')}, "
                f"source {row.get('source_orientation_status', 'missing')})"
            )
            for name, row in per_dataset_transforms.items()
        )
        unique_transforms = {
            str(row.get("selected_transform"))
            for row in per_dataset_transforms.values()
            if str(row.get("selected_transform") or "n/a") != "n/a"
        }
        transform_summary = {
            **transform_summary,
            "selected_transform": next(iter(unique_transforms))
            if len(unique_transforms) == 1
            else selected_transform_text,
            "per_dataset": per_dataset_transforms,
            "per_dataset_text": selected_transform_text,
            "per_dataset_parity_text": per_dataset_parity_text,
            "configuration_status": (
                "parity_unresolved"
                if any(
                    row.get("configuration_status") == "parity_unresolved" for row in per_dataset_transforms.values()
                )
                else (
                    "absolute_configuration_calibrated"
                    if per_dataset_transforms
                    and all(
                        row.get("configuration_status") == "absolute_configuration_calibrated"
                        for row in per_dataset_transforms.values()
                    )
                    else "conditional_inversion_preference"
                )
            ),
            "absolute_configuration_claim_allowed": bool(per_dataset_transforms)
            and all(bool(row.get("absolute_configuration_claim_allowed")) for row in per_dataset_transforms.values()),
            "transform_parity": (
                next(iter({str(row.get("transform_parity", "n/a")) for row in per_dataset_transforms.values()}))
                if len({str(row.get("transform_parity", "n/a")) for row in per_dataset_transforms.values()}) == 1
                else "mixed"
            ),
            "transform_determinant": (
                next(iter({str(row.get("transform_determinant", "n/a")) for row in per_dataset_transforms.values()}))
                if len({str(row.get("transform_determinant", "n/a")) for row in per_dataset_transforms.values()}) == 1
                else "per-dataset"
            ),
            "source_orientation_status": (
                next(
                    iter(
                        {
                            str(row.get("source_orientation_status", "missing"))
                            for row in per_dataset_transforms.values()
                        }
                    )
                )
                if len(
                    {str(row.get("source_orientation_status", "missing")) for row in per_dataset_transforms.values()}
                )
                == 1
                else "mixed"
            ),
            "configuration_status_reason": ("see per-dataset transform/source-orientation contracts"),
        }
    screen_run = (
        Path(str(decision_tree.get("screen_run", ""))) if isinstance(decision_tree.get("screen_run"), str) else None
    )
    screen_summary = try_read_json(screen_run / "summary.json") if screen_run else {}
    promotion = decision_tree.get("post_feedback_hand_promotion")
    promotion_decision = promotion if isinstance(promotion, Mapping) else {}
    diagnostic_path_text = (
        promotion_decision.get("diagnostic")
        or result.get("absolute_structure_diagnostic")
        or decision_tree.get("absolute_structure_diagnostic")
        or ""
    )
    hand_diagnostic = try_read_json(Path(str(diagnostic_path_text))) if diagnostic_path_text else {}
    hand_summary = _quick_final_hand_summary(decision_tree, hand_diagnostic)
    multi_hand_diagnostic = _quick_multi_hand_diagnostic(result) if result.get("combined") else {}
    if result.get("combined") and not multi_hand_diagnostic:
        multi_hand_diagnostic = {
            "status": "warning_missing_diagnostic",
            "artifact": str(out_dir / "decisions" / "shared_post_feedback_hand_diagnostic.json"),
            "datasets": {},
            "consistency_gate": {
                "status": "warning",
                "passed": False,
                "warnings": ["the shared post-feedback multi-dataset hand diagnostic is missing"],
            },
        }
    multi_hand_rows = _quick_multi_hand_console_rows(multi_hand_diagnostic)
    multi_hand_pooled = _quick_final_elastic_hand_aggregate(multi_hand_diagnostic)
    multi_hand_zkp = _quick_final_elastic_hand_zkp(multi_hand_diagnostic)
    if multi_hand_diagnostic:
        hand_summary["promotion_status"] = multi_hand_diagnostic.get("status", "missing")

    return (
        hand_summary,
        multi_hand_diagnostic,
        multi_hand_pooled,
        multi_hand_rows,
        multi_hand_zkp,
        promotion_decision,
        screen_summary,
        transform_summary,
    )


def _start_quick_stats_markdown(
    *,
    heldout_lines=(),
    dataset_nuisance_contracts,
    endpoint_status_text,
    file_block,
    finish_info,
    metadata,
    nuisance,
    out_dir,
    prep,
    publication_endpoint,
    result,
    source_roots,
    table1,
    timing_rows,
):
    """Start the report with run provenance and correctly labeled stage timings."""

    lines = [
        "# GROND Quick Stats",
        "",
        "This is the short run summary. Use the detailed reports for diagnostics and audit trails.",
        "",
        # --report-layout heldout-first: the held-out validation opens the report.
        *heldout_lines,
        "## Canonical Final Statistics",
        "",
        f"- Table 1, human-readable: `{table1['markdown']}`",
        f"- Table 1, machine-readable: `{table1['json']}`",
        f"- JANA/PETS-style row report: `{file_block['jana_row_statistics']}`",
        f"- Per-dataset JANA/PETS-style row report: `{file_block['jana_row_statistics_per_dataset']}`",
        "- Table 1 panel A contains only SHELXL endpoint statistics. Panel B contains only GROND-computed JANA/PETS-style row and post-refinement merged-row factors. Missing comparator classes remain explicit `n/a`; values are never substituted across panels.",
        "",
        "## Run",
        "",
        f"- Output directory: `{out_dir}`",
        f"- Datasets: `{', '.join(ds.name for ds in prep.datasets) if prep.datasets else 'n/a'}`",
        f"- Final model: `{result.get('model', '')}`",
        f"- Publication endpoint model: `{publication_endpoint.get('endpoint_res', '')}`",
        f"- Publication corrected HKL: `{publication_endpoint.get('endpoint_hkl', '')}`",
        f"- Publication endpoint source: `{publication_endpoint.get('selection_source', 'n/a')}` / `{publication_endpoint.get('source_kind', 'n/a')}`",
        f"- Endpoint decision: `{endpoint_status_text}`",
        f"- Rows in final row table: `{metadata.get('row_count', 'n/a')}`",
        f"- Final N/Q: `{metadata.get('n_basis', 'n/a')}/{metadata.get('n_points', 'n/a')}`",
        f"- Effective thickness: `{_quick_thickness_text(nuisance, metadata, dataset_nuisance_contracts)}`",
        f"- Sg window: `{metadata.get('sg_window', 'n/a')}`; absorption: `{metadata.get('absorption_mode', 'n/a')}`; Sg jitter: `{metadata.get('sg_jitter_mode', 'n/a')}`",
        "",
    ]
    if len(source_roots) > 1:
        lines.extend(
            [
                "## Resume Warning",
                "",
                "This output folder contains completed stages from more than one GROND source tree. Treat it as a mixed-code resume, not a clean benchmark. Start a new output directory for final numbers.",
                "",
                "| source root | completed stages |",
                "|---|---:|",
            ]
        )
        for root, count in source_roots:
            lines.append(f"| `{root}` | {count} |")
        lines.append("")
    elif source_roots:
        root, count = source_roots[0]
        lines.extend(["## Source", "", f"- Completed stages were produced from `{root}` ({count} stages).", ""])

    if timing_rows:
        total_elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in timing_rows if row.get("returncode") == 0)
        timing_summary_lines = []
        if finish_info.get("invocation_elapsed_seconds") is not None:
            timing_summary_lines.append(
                f"- Recorded workflow wall time: `{format_seconds(finish_info.get('invocation_elapsed_seconds'))}`"
            )
        timing_summary_lines.extend(
            [
                f"- Cumulative completed-stage runtime: `{format_seconds(total_elapsed)}`",
                "- The cumulative value sums saved stage runtimes. Concurrent stages overlap, and resumed stages may come from earlier invocations, so this is not workflow wall time.",
            ]
        )
        lines.extend(
            [
                "## Stage Timings",
                "",
                *timing_summary_lines,
                "",
                "| stage | returncode | elapsed | log |",
                "|---|---:|---:|---|",
            ]
        )
        for row in timing_rows:
            log = str(row.get("log", ""))
            lines.append(
                f"| `{row.get('stage', 'n/a')}` | {row.get('returncode', 'n/a')} | `{row.get('elapsed', 'n/a')}` | `{log}` |"
            )
        lines.append("")

    return lines


def _append_quick_stats_model_panels(
    *,
    all_z,
    best_endpoint,
    dataset_nuisance_contracts,
    diagnostic_endpoint,
    diagnostic_shelxl,
    file_block,
    final_feedback_shelxl,
    final_row_summary_path,
    final_row_table_sources,
    free_z,
    hand_delta_report_line,
    hand_summary,
    hand_t_report_line,
    hand_zkp_report_line,
    input_shelxl,
    jana_born,
    jana_dynamic,
    lines,
    metadata,
    multi_hand_diagnostic,
    native_metrics,
    nuisance,
    publication_endpoint,
    publication_shelxl,
    result,
    row_count_text,
    screen_summary,
    sfac_check,
    source_control_text,
    transform_summary,
):
    """Append endpoint, hand, native-objective and nuisance panels without mixing metric families."""

    lines.extend(
        [
            "## Starting SHELXL Reference",
            "",
            "GROND first reports the isolated one-cycle replay of the supplied starting RES/HKL, with inherited EXTI/WGHT and hash-bound inputs and outputs. Historical runs fall back to a healthy final-stage `baseline_exti` control or, last, the supplied model's sibling `.lst`/`.log`; those fallbacks are labeled in the table and do not satisfy the required multi-dataset publication no-harm gate.",
            "",
            "| RES | HKL | stats file found | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | parameters | restraints | pathology | WGHT |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
            "| "
            + " | ".join(
                [
                    f"`{input_shelxl.get('res', '')}`",
                    f"`{input_shelxl.get('hkl', '')}`",
                    "`yes`" if input_shelxl.get("lst_exists") else "`no`",
                    _quick_percent(input_shelxl.get("R1_gt")),
                    _quick_percent(input_shelxl.get("R1_all")),
                    _quick_percent(input_shelxl.get("wR2_all")),
                    _quick_number(input_shelxl.get("GooF"), 3),
                    str(input_shelxl.get("parameters", "n/a")),
                    str(input_shelxl.get("restraints", "n/a")),
                    _quick_shelxl_issue_text(input_shelxl),
                    _quick_shelxl_weight_text(input_shelxl),
                ]
            )
            + " |",
            "",
        ]
    )

    lines.extend(
        [
            "## Input Contract, Hand, And Paired T-Style Diagnostics",
            "",
            "XDS, DIALS, and other integrators can produce different signed-row contracts from the same frames. Row filtering, partiality/Lorentz handling, scan-varying geometry, sigmas, and scale conventions can move nuisance fits such as thickness, frame offset, absorption, and scale counts. GROND therefore treats the selected HKL transform, final absolute-structure hand, and raw row-level Bloch/Born statistics as the most portable cross-pipeline checks; nuisance values are fit-contract parameters, not direct physical measurements unless the row contracts agree.",
            "",
            "| item | value |",
            "|---|---|",
            f"| SFAC / endpoint wavelength check | status `{sfac_check.get('status', 'n/a') if isinstance(sfac_check, Mapping) else 'n/a'}`; endpoint wavelength `{_quick_number((sfac_check.get('cell') or {}).get('endpoint_wavelength_A') if isinstance(sfac_check.get('cell'), Mapping) else None, 6)}` A |",
            f"| selected HKL transform | `{transform_summary.get('selected_transform', 'n/a')}` |",
            *(
                [f"| per-dataset HKL transforms | `{transform_summary.get('per_dataset_text', '')}` |"]
                if transform_summary.get("per_dataset_text")
                else []
            ),
            *(
                [
                    f"| per-dataset parity/source orientation | `{transform_summary.get('per_dataset_parity_text', '')}` |"
                ]
                if transform_summary.get("per_dataset_parity_text")
                else []
            ),
            f"| transform scout status | `{transform_summary.get('selected_status', 'n/a')}` |",
            f"| transform parity | `{transform_summary.get('transform_parity', 'n/a')}`; determinant `{transform_summary.get('transform_determinant', 'per-dataset')}`; scout policy `{transform_summary.get('transform_parity_policy', 'n/a')}` |",
            f"| source orientation | `{transform_summary.get('source_orientation_status', 'missing')}`; raw determinant `{_quick_number(transform_summary.get('source_basis_determinant'), 6)}`; transformed determinant `{_quick_number(transform_summary.get('transformed_basis_determinant'), 6)}` |",
            f"| configuration claim status | `{transform_summary.get('configuration_status', 'parity_unresolved')}`; named absolute configuration allowed `{bool(transform_summary.get('absolute_configuration_claim_allowed', False))}` |",
            f"| configuration claim reason | {transform_summary.get('configuration_status_reason', 'see per-dataset transform contracts')} |",
            f"| metadata transform check | `{transform_summary.get('metadata_status', 'n/a')}`; log-I corr `{_quick_number(transform_summary.get('metadata_log_intensity_corr'), 4)}`; gap `{_quick_number(transform_summary.get('metadata_correlation_gap'), 4)}`; pairs `{transform_summary.get('metadata_paired_rows', 'n/a')}`; unique overlap `{transform_summary.get('metadata_unique_overlap', 'n/a')}` |",
            f"| initial hand scout | `{hand_summary.get('initial_hand', 'n/a')}`; delta original-minus-alternate `{_quick_number(hand_summary.get('initial_score_delta_original_minus_alternate'), 4)}` |",
            f"| final hand | `{hand_summary.get('final_hand', 'n/a')}`; post-feedback status `{hand_summary.get('promotion_status', 'n/a')}` |",
            hand_t_report_line,
            hand_zkp_report_line,
            hand_delta_report_line,
            f"| free-row I residual Z | n `{free_z.get('n', 'n/a')}`; RMSZ `{_quick_number(free_z.get('rms_z'), 3)}`; meanZ `{_quick_number(free_z.get('mean_z'), 3)}`; median absZ `{_quick_number(free_z.get('median_abs_z'), 3)}` |",
            f"| all-row I residual Z | n `{all_z.get('n', 'n/a')}`; RMSZ `{_quick_number(all_z.get('rms_z'), 3)}`; meanZ `{_quick_number(all_z.get('mean_z'), 3)}`; median absZ `{_quick_number(all_z.get('median_abs_z'), 3)}` |",
            "",
            *(_quick_multi_hand_evidence_lines(multi_hand_diagnostic) if result.get("combined") else []),
            "## Input Model Row Statistics",
            "",
            "These are the first adaptive-screen row-space statistics for the supplied start model after GROND has selected the row contract used by the run. They are the practical input-model comparator, not final endpoint statistics.",
            "",
            "| model state | dyn R(I)all | dyn wR(I)all | dyn free wR(I)all | matched Born R(I)all | matched Born wR(I)all |",
            "|---|---:|---:|---:|---:|---:|",
            f"| input adaptive screen | `{_quick_percent(_quick_dyn_metric(screen_summary, 'all', 'R1_all'))}` | `{_quick_percent(_quick_dyn_metric(screen_summary, 'all', 'wR2_all'))}` | `{_quick_percent(_quick_dyn_metric(screen_summary, 'free', 'wR2_all'))}` | `{_quick_percent(_quick_born_metric(screen_summary, 'all', 'R1_all'))}` | `{_quick_percent(_quick_born_metric(screen_summary, 'all', 'wR2_all'))}` |",
            "",
            "## Final JANA/PETS-Style Row Statistics",
            "",
            "These are the publication-facing row factors computed from the final frozen signed-row tables. `Robs`/`Rall` and `wRobs`/`wRall` are unmerged-row statistics; `MRobs`/`MRall` and `MwRobs`/`MwRall` are recomputed after arithmetic symmetry merging. `wR`/`MwR` are the amplitude form, Jana2020's `wR(all)` for refinement on F (`sqrt(sum w (Fo - Fc)^2 / sum w Fo^2)`, `w = 1/[sigma(Fo)^2 + (u Fo)^2]`); `MR`/`MwR` follow Klar et al. (2023). The observed subset is `Iobs > 3 sigma(Iobs)`. These values are not SHELXL endpoint R factors.",
            "",
            "| model | Ninput/excluded | Nobs/Nall | Nmerge used/excluded | MNobs/MNall | Robs | Rall | wRobs | wRall | MRobs | MRall | MwRobs | MwRall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| matched Born | {_table1_pair(jana_born, 'Ninput', 'Nexcluded')} | {_quick_count_pair(jana_born)} | {_table1_pair(jana_born, 'Nmerge_used', 'Nmerge_excluded')} | {_table1_pair(jana_born, 'MNobs', 'MNall')} | {_quick_percent(jana_born.get('Robs'))} | {_quick_percent(jana_born.get('Rall'))} | {_quick_percent(jana_born.get('wRobs'))} | {_quick_percent(jana_born.get('wRall'))} | {_quick_percent(jana_born.get('MRobs'))} | {_quick_percent(jana_born.get('MRall'))} | {_quick_percent(jana_born.get('MwRobs'))} | {_quick_percent(jana_born.get('MwRall'))} |",
            f"| Bloch dynamical | {_table1_pair(jana_dynamic, 'Ninput', 'Nexcluded')} | {_quick_count_pair(jana_dynamic)} | {_table1_pair(jana_dynamic, 'Nmerge_used', 'Nmerge_excluded')} | {_table1_pair(jana_dynamic, 'MNobs', 'MNall')} | {_quick_percent(jana_dynamic.get('Robs'))} | {_quick_percent(jana_dynamic.get('Rall'))} | {_quick_percent(jana_dynamic.get('wRobs'))} | {_quick_percent(jana_dynamic.get('wRall'))} | {_quick_percent(jana_dynamic.get('MRobs'))} | {_quick_percent(jana_dynamic.get('MRall'))} | {_quick_percent(jana_dynamic.get('MwRobs'))} | {_quick_percent(jana_dynamic.get('MwRall'))} |",
            "",
            "## Legacy Native-Objective Audit Matrix (Not Table 1)",
            "",
            "This retained engineering matrix contains GROND's historical native-objective `R1_all`/`wR2_all` diagnostics alongside endpoint diagnostics. Those internal row-objective names are not JANA/PETS publication factors and this matrix is not the canonical final table. Use `TABLE1_FINAL_STATISTICS.md`: its SHELXL and JANA/PETS panels have authoritative labels, targets, and denominators.",
            "",
            "| object | target class | Rgt | Rall | weighted R | free/held-out weighted R | GooF | N/reflections | parameters/scales | source and notes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
            "| "
            + " | ".join(
                [
                    "input conventional SHELXL",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_percent(input_shelxl.get("R1_gt")),
                    _quick_percent(input_shelxl.get("R1_all")),
                    _quick_percent(input_shelxl.get("wR2_all")),
                    "n/a",
                    _quick_number(input_shelxl.get("GooF"), 3),
                    str(input_shelxl.get("N_all", "n/a")),
                    f"{input_shelxl.get('parameters', 'n/a')} params; {input_shelxl.get('restraints', 'n/a')} restraints",
                    f"{source_control_text}; {_quick_shelxl_issue_text(input_shelxl)}; `{input_shelxl.get('lst', '')}`",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "input matched Born rows",
                    "raw signed rows (I), matched Born",
                    "n/a",
                    _quick_percent(_quick_born_metric(screen_summary, "all", "R1_all")),
                    _quick_percent(_quick_born_metric(screen_summary, "all", "wR2_all")),
                    _quick_percent(_quick_born_metric(screen_summary, "free", "wR2_all")),
                    "n/a",
                    row_count_text,
                    "screen nuisance/scales",
                    "`adaptive_screen summary.json`; same signed rows as input dynamical comparator",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "input dynamical rows",
                    "raw signed rows (I), Bloch dynamical",
                    "n/a",
                    _quick_percent(_quick_dyn_metric(screen_summary, "all", "R1_all")),
                    _quick_percent(_quick_dyn_metric(screen_summary, "all", "wR2_all")),
                    _quick_percent(_quick_dyn_metric(screen_summary, "free", "wR2_all")),
                    "n/a",
                    row_count_text,
                    "screen nuisance/scales",
                    "`adaptive_screen summary.json`; supplied model under first accepted GROND row contract",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "final matched Born rows",
                    "raw signed rows (I), matched Born",
                    "n/a",
                    _quick_percent(_quick_metric(native_metrics, "all", "born", "R1_all")),
                    _quick_percent(_quick_metric(native_metrics, "all", "born", "wR2_all")),
                    _quick_percent(_quick_metric(native_metrics, "free", "born", "wR2_all")),
                    "n/a",
                    row_count_text,
                    "final nuisance/scales",
                    f"`{final_row_summary_path or ''}`; Born comparator at final model",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "final dynamical rows",
                    "raw signed rows (I), Bloch dynamical",
                    "n/a",
                    _quick_percent(_quick_metric(native_metrics, "all", "dyn", "R1_all")),
                    _quick_percent(_quick_metric(native_metrics, "all", "dyn", "wR2_all")),
                    _quick_percent(_quick_metric(native_metrics, "free", "dyn", "wR2_all")),
                    "n/a",
                    row_count_text,
                    "final nuisance/scales",
                    f"`{final_row_table_sources}`; final raw-row statistic",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "direct LS1 corrected-HKL candidate",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_percent(final_feedback_shelxl.get("R1_gt")),
                    _quick_percent(final_feedback_shelxl.get("R1_all")),
                    _quick_percent(final_feedback_shelxl.get("wR2_all")),
                    "n/a",
                    _quick_number(final_feedback_shelxl.get("GooF"), 3),
                    str(final_feedback_shelxl.get("N_all", "n/a")),
                    f"{final_feedback_shelxl.get('parameters', 'n/a')} params; {final_feedback_shelxl.get('restraints', 'n/a')} restraints",
                    f"{_quick_shelxl_issue_text(final_feedback_shelxl)}; `{file_block['final_feedback_lst']}`",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "best diagnostic endpoint, not promoted",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_percent(diagnostic_shelxl.get("R1_gt")),
                    _quick_percent(diagnostic_shelxl.get("R1_all")),
                    _quick_percent(diagnostic_shelxl.get("wR2_all")),
                    "n/a",
                    _quick_number(diagnostic_shelxl.get("GooF"), 3),
                    str(diagnostic_shelxl.get("N_all", "n/a")),
                    f"{diagnostic_shelxl.get('parameters', 'n/a')} params; {diagnostic_shelxl.get('restraints', 'n/a')} restraints",
                    f"{_quick_shelxl_issue_text(diagnostic_shelxl)}; `{diagnostic_endpoint.get('endpoint_lst', '')}`",
                ]
            )
            + " |"
            if diagnostic_endpoint
            else "",
            "| "
            + " | ".join(
                [
                    "publication corrected-HKL endpoint",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_percent(publication_shelxl.get("R1_gt")),
                    _quick_percent(publication_shelxl.get("R1_all")),
                    _quick_percent(publication_shelxl.get("wR2_all")),
                    "n/a",
                    _quick_number(publication_shelxl.get("GooF"), 3),
                    str(publication_shelxl.get("N_all", "n/a")),
                    f"{publication_endpoint.get('spec') or publication_endpoint.get('source_kind') or 'n/a'}; {publication_endpoint.get('scale_params', 'n/a')} scales; {publication_shelxl.get('parameters', 'n/a')} params",
                    f"{_quick_shelxl_issue_text(publication_shelxl)}; `{file_block['publication_lst']}`",
                ]
            )
            + " |",
            "",
            "## Legacy Native-Objective Delta Matrix (Not Table 1)",
            "",
            "Negative values are improvements for R factors. Deltas are only meaningful within the same target class.",
            "",
            "| comparison | target class | dRall | d weighted R | d free/held-out weighted R | interpretation |",
            "|---|---|---:|---:|---:|---|",
            "| "
            + " | ".join(
                [
                    "input dynamical rows minus input matched Born rows",
                    "raw signed rows",
                    _quick_delta_percent(
                        _quick_born_metric(screen_summary, "all", "R1_all"),
                        _quick_dyn_metric(screen_summary, "all", "R1_all"),
                    ),
                    _quick_delta_percent(
                        _quick_born_metric(screen_summary, "all", "wR2_all"),
                        _quick_dyn_metric(screen_summary, "all", "wR2_all"),
                    ),
                    _quick_delta_percent(
                        _quick_born_metric(screen_summary, "free", "wR2_all"),
                        _quick_dyn_metric(screen_summary, "free", "wR2_all"),
                    ),
                    "starting-model dynamical correction benefit over the matched Born control",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "final dynamical rows minus final matched Born rows",
                    "raw signed rows",
                    _quick_delta_percent(
                        _quick_metric(native_metrics, "all", "born", "R1_all"),
                        _quick_metric(native_metrics, "all", "dyn", "R1_all"),
                    ),
                    _quick_delta_percent(
                        _quick_metric(native_metrics, "all", "born", "wR2_all"),
                        _quick_metric(native_metrics, "all", "dyn", "wR2_all"),
                    ),
                    _quick_delta_percent(
                        _quick_metric(native_metrics, "free", "born", "wR2_all"),
                        _quick_metric(native_metrics, "free", "dyn", "wR2_all"),
                    ),
                    "final model dynamical correction benefit over the final Born control",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "final dynamical rows minus input dynamical rows",
                    "raw signed rows",
                    _quick_delta_percent(
                        _quick_dyn_metric(screen_summary, "all", "R1_all"),
                        _quick_metric(native_metrics, "all", "dyn", "R1_all"),
                    ),
                    _quick_delta_percent(
                        _quick_dyn_metric(screen_summary, "all", "wR2_all"),
                        _quick_metric(native_metrics, "all", "dyn", "wR2_all"),
                    ),
                    _quick_delta_percent(
                        _quick_dyn_metric(screen_summary, "free", "wR2_all"),
                        _quick_metric(native_metrics, "free", "dyn", "wR2_all"),
                    ),
                    "net row-level movement from the supplied model to the final model",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "publication endpoint minus input conventional SHELXL",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_delta_percent(input_shelxl.get("R1_all"), publication_shelxl.get("R1_all")),
                    _quick_delta_percent(input_shelxl.get("wR2_all"), publication_shelxl.get("wR2_all")),
                    "n/a",
                    "headline conventional endpoint change; compare only with matching reflection coverage caveats",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "direct LS1 endpoint minus input conventional SHELXL",
                    "merged SHELXL endpoint (F/F^2)",
                    _quick_delta_percent(input_shelxl.get("R1_all"), final_feedback_shelxl.get("R1_all")),
                    _quick_delta_percent(input_shelxl.get("wR2_all"), final_feedback_shelxl.get("wR2_all")),
                    "n/a",
                    "unreplayed corrected-HKL endpoint change before publication scale selection",
                ]
            )
            + " |",
            "| "
            + " | ".join(
                [
                    "publication endpoint minus direct LS1 endpoint",
                    "merged SHELXL endpoint",
                    _quick_delta_percent(final_feedback_shelxl.get("R1_all"), publication_shelxl.get("R1_all")),
                    _quick_delta_percent(final_feedback_shelxl.get("wR2_all"), publication_shelxl.get("wR2_all")),
                    "n/a",
                    "cost or gain of the publication endpoint replay relative to the direct LS1 candidate",
                ]
            )
            + " |",
            "",
            "## Row Intensity R-Factor Definitions",
            "",
            _quick_row_weight_definition(nuisance, dataset_nuisance_contracts),
            "",
            "## Final Nuisance Contract",
            "",
            "| field | value |",
            "|---|---|",
            *(
                [
                    f"| thickness | `{_quick_thickness_text(nuisance, metadata, dataset_nuisance_contracts)}` |",
                    "| native row-weight mode | `per_dataset_native`; no active joint a/b pair; exact per-dataset values follow |",
                ]
                if dataset_nuisance_contracts
                else [
                    f"| thickness | `{_quick_thickness_text(nuisance, metadata)}` |",
                    f"| frame and rocking | frame offset `{_quick_number(nuisance.get('frame_offset', nuisance.get('frame_offset_grid')), 3)}`; Sg window `{metadata.get('sg_window', 'n/a')}`; matrix `{nuisance.get('matrix_convention', 'n/a')}` |",
                    f"| absorption and jitter | absorption `{nuisance.get('absorption_mode', nuisance.get('absorption_mode_grid', 'n/a'))}`; frac `{_quick_number(nuisance.get('absorption_frac', nuisance.get('absorption_frac_grid')), 4)}`; jitter `{nuisance.get('sg_jitter_mode', nuisance.get('sg_jitter_mode_grid', 'n/a'))}` |",
                    f"| native row weights | a `{_quick_number(nuisance.get('native_wght_a', nuisance.get('wght_a_grid')), 6)}`; b `{_quick_number(nuisance.get('native_wght_b', nuisance.get('wght_b_grid')), 6)}` |",
                ]
            ),
            f"| row dyn R(I)all / wR(I)all / free wR(I)all | `{_quick_percent(_quick_metric(native_metrics, 'all', 'dyn', 'R1_all'))}` / `{_quick_percent(_quick_metric(native_metrics, 'all', 'dyn', 'wR2_all'))}` / `{_quick_percent(_quick_metric(native_metrics, 'free', 'dyn', 'wR2_all'))}` |",
            f"| row Born R(I)all / wR(I)all / free wR(I)all | `{_quick_percent(_quick_metric(native_metrics, 'all', 'born', 'R1_all'))}` / `{_quick_percent(_quick_metric(native_metrics, 'all', 'born', 'wR2_all'))}` / `{_quick_percent(_quick_metric(native_metrics, 'free', 'born', 'wR2_all'))}` |",
            f"| q p05 / median / p95 | `{_quick_number(best_endpoint.get('q_p05'), 3)}` / `{_quick_number(best_endpoint.get('q_median'), 3)}` / `{_quick_number(best_endpoint.get('q_p95'), 3)}` |",
            "",
            *_quick_dataset_nuisance_table_lines(dataset_nuisance_contracts),
        ]
    )


def _append_quick_stats_resolution_rows(
    *,
    endpoints,
    lines,
):
    """Append the separate JANA-style resolution and scale-contract table."""

    lines.extend(
        [
            "## JANA/PETS-Style Row And Post-Refinement Merged Statistics By Resolution/Scale Contract",
            "",
            "These GROND-computed factors follow the JANA/PETS row-statistics family and remain separate from endpoint refinement. `Nobs/Nall` is the accepted observed-row coverage at that resolution gate; `Rall` and `wRall` use unmerged signed rows, while `MRall` and `MwRall` are recomputed after post-refinement symmetry merging; `wRall`/`MwRall` are the amplitude form (Jana2020 `wR(all)` for refinement on F). They are not SHELXL least-squares statistics.",
            "",
            "| endpoint | dmin A | scales | Nobs/Nall | dyn Rall | dyn wRall | dyn MRall | dyn MwRall | kin Rall | kin wRall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if endpoints:
        for endpoint in endpoints:
            if not isinstance(endpoint, Mapping):
                continue
            row_mr = endpoint.get("row_mr", {}) if isinstance(endpoint.get("row_mr"), Mapping) else {}
            dyn = row_mr.get("dynamic", {}) if isinstance(row_mr.get("dynamic"), Mapping) else {}
            kin = row_mr.get("born", {}) if isinstance(row_mr.get("born"), Mapping) else {}
            if not kin and isinstance(row_mr.get("kinematic"), Mapping):
                kin = row_mr.get("kinematic", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(endpoint.get("label", "n/a")),
                        _quick_number(endpoint.get("dmin_A"), 3),
                        str(endpoint.get("scale_params", "n/a")),
                        _quick_count_pair(dyn),
                        _quick_percent(dyn.get("Rall")),
                        _quick_percent(dyn.get("wRall")),
                        _quick_percent(dyn.get("MRall")),
                        _quick_percent(dyn.get("MwRall")),
                        _quick_percent(kin.get("Rall")),
                        _quick_percent(kin.get("wRall")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")


def _append_quick_stats_endpoint_selection(
    *,
    bloch_diagnostics_path,
    endpoint_calibration,
    endpoint_selection,
    file_block,
    frozen_row_file_lines,
    lines,
    out_dir,
    publication_endpoint,
    publication_shelxl,
    row_metrics_path,
    table1,
):
    """Append model-handoff selection, calibration branches and exact artifact links."""

    selection_policy = (
        str(endpoint_selection.get("selection_policy", "n/a")) if isinstance(endpoint_selection, Mapping) else "n/a"
    )
    information_criterion = (
        str(endpoint_selection.get("information_criterion", "n/a"))
        if isinstance(endpoint_selection, Mapping)
        else "n/a"
    )
    lines.extend(
        [
            "",
            "## Recommended Model-Handoff Endpoint",
            "",
            "This is the endpoint GROND recommends opening, inspecting, and continuing in the user's modeling program. Warnings remain attached; they do not hide an otherwise usable model. The endpoint-calibration table below remains the conservative scale-replay audit.",
            "",
            "| source | kind/spec | WGHT | scales | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | params/restraints | pathology |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
            "| "
            + " | ".join(
                [
                    str(publication_endpoint.get("selection_source", "n/a")),
                    str(publication_endpoint.get("spec") or publication_endpoint.get("source_kind") or "n/a"),
                    str(publication_endpoint.get("wght_mode", "n/a")),
                    str(publication_endpoint.get("scale_params", "n/a")),
                    _quick_percent(publication_shelxl.get("R1_gt")),
                    _quick_percent(publication_shelxl.get("R1_all")),
                    _quick_percent(publication_shelxl.get("wR2_all")),
                    _quick_number(publication_shelxl.get("GooF"), 3),
                    f"{publication_shelxl.get('parameters', 'n/a')} / {publication_shelxl.get('restraints', 'n/a')}",
                    _quick_shelxl_issue_text(publication_shelxl),
                ]
            )
            + " |",
            "",
            (
                "**Endpoint warning:** " + _quick_goof_warning(publication_shelxl.get("GooF")) + "."
                if _quick_goof_warning(publication_shelxl.get("GooF"))
                else ""
            ),
            *["**Endpoint warning:** " + str(warning) for warning in
              (publication_endpoint.get("baseline_no_harm_gate") or {}).get("warnings", [])],
            "## Endpoint Calibration",
            "",
            f"Status: `{endpoint_calibration.get('status', 'n/a') if isinstance(endpoint_calibration, Mapping) else 'n/a'}`. {endpoint_calibration.get('reason', '') if isinstance(endpoint_calibration, Mapping) else ''}",
            "",
            f"Selection policy: `{selection_policy}`; information criterion: `{information_criterion}`.",
            "",
            "| branch | spec | WGHT | scales | free row Rall | free row wRall | work row Rall | R1(F)all report | wR2(F^2)all report | GooF |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    branch_rows = [
        (
            "selected_scale_replay",
            endpoint_selection.get("publication_endpoint") if isinstance(endpoint_selection, Mapping) else None,
        ),
        (
            "low_R_diagnostic",
            endpoint_selection.get("low_r_diagnostic") if isinstance(endpoint_selection, Mapping) else None,
        ),
        (
            "best_heldout_row",
            endpoint_selection.get("best_heldout_row") if isinstance(endpoint_selection, Mapping) else None,
        ),
    ]
    wrote_branch = False
    for label, row in branch_rows:
        if not isinstance(row, Mapping):
            continue
        wrote_branch = True
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row.get("spec", "n/a")),
                    str(row.get("wght_mode", "n/a")),
                    str(row.get("scale_params", "n/a")),
                    _quick_percent(row.get("row_free_Rall")),
                    _quick_percent(row.get("row_free_wRall")),
                    _quick_percent(row.get("row_work_Rall")),
                    _quick_percent(row.get("shelxl_R1all")),
                    _quick_percent(row.get("shelxl_wR2all")),
                    _quick_number(row.get("shelxl_GooF"), 3),
                ]
            )
            + " |"
        )
    if not wrote_branch:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Files To Open Next",
            "",
            f"- Canonical final Table 1: `{table1['markdown']}`",
            f"- Machine-readable final Table 1: `{table1['json']}`",
            f"- Combined JANA/PETS-style row factors: `{file_block['jana_row_statistics']}`",
            f"- Per-dataset JANA/PETS-style row factors: `{file_block['jana_row_statistics_per_dataset']}`",
            f"- Fresh conventional baseline RES to retain when no corrected endpoint passes: `{file_block['baseline_res']}`",
            f"- Fresh conventional baseline HKL: `{file_block['baseline_hkl']}`",
            f"- Fresh conventional baseline SHELXL LST: `{file_block['baseline_lst']}`",
            f"- Final LS1 feedback endpoint RES: `{file_block['final_feedback_res']}`",
            f"- Final LS1 feedback corrected HKL: `{file_block['final_feedback_hkl']}`",
            f"- Final LS1 feedback SHELXL LST: `{file_block['final_feedback_lst']}`",
            f"- Final LS1 feedback manifest: `{file_block['final_feedback_manifest']}`",
            f"- Publication endpoint RES to open in SHELXLE/Olex2: `{file_block['publication_res']}`",
            f"- Publication endpoint INS used for the reported run: `{file_block['publication_ins']}`",
            f"- Publication corrected HKL used by that endpoint: `{file_block['publication_hkl']}`",
            f"- Publication SHELXL LST with the reported R/GooF values: `{file_block['publication_lst']}`",
            f"- Publication endpoint manifest: `{file_block['publication_manifest']}`",
            *frozen_row_file_lines,
            f"- Endpoint replay row table after compact scale selection: `{file_block['publication_row_table']}`",
            f"- Bloch matrix and row-contract diagnostics: `{bloch_diagnostics_path}`",
            f"- Detailed row/MR table: `{row_metrics_path}`",
            f"- Endpoint calibration report: `{endpoint_calibration.get('report', '') if isinstance(endpoint_calibration, Mapping) else ''}`",
            f"- Publication endpoint chooser: `{out_dir / 'endpoint_chooser' / 'PUBLICATION_ENDPOINT_CHOOSER.md'}`",
            "",
        ]
    )


def _prepare_quick_stats_console(
    *,
    diagnostic_endpoint,
    endpoint_selection,
    file_block,
    final_feedback_shelxl,
    final_summary,
    final_summary_path,
    frozen_row_console_text,
    frozen_row_tables,
    hand_all,
    hand_free,
    hand_summary,
    input_shelxl,
    jana_born,
    jana_dynamic,
    jana_summary,
    multi_hand_diagnostic,
    multi_hand_pooled,
    multi_hand_rows,
    multi_hand_zkp,
    native_metrics,
    out_dir,
    promotion_decision,
    publication_endpoint,
    publication_shelxl,
    quick_path,
    result,
    row_metrics_path,
    screen_summary,
    table1,
    transform_summary,
):
    """Build the machine-readable console payload from the already resolved report inputs."""

    selected_row = publication_endpoint if publication_endpoint else {}
    final_row_r1 = _quick_dyn_metric(final_summary, "all", "R1_all")
    final_row_wr2 = _quick_dyn_metric(final_summary, "all", "wR2_all")
    final_free_wr2 = _quick_dyn_metric(final_summary, "free", "wR2_all")
    input_row_r1 = _quick_dyn_metric(screen_summary, "all", "R1_all")
    input_row_wr2 = _quick_dyn_metric(screen_summary, "all", "wR2_all")
    console = {
        "out_dir": str(out_dir),
        "final_summary": str(final_summary_path),
        "quick_stats": str(quick_path),
        "full_json": str(out_dir / "grond_summary.json"),
        "final_hand": hand_summary.get("final_hand", "n/a"),
        "final_hand_preferred": (
            multi_hand_diagnostic.get("preferred_hand", "unavailable")
            if result.get("combined")
            else promotion_decision.get("preferred_hand", "unavailable")
        ),
        "final_hand_confidence": (
            multi_hand_diagnostic.get("hand_confidence", "unavailable")
            if result.get("combined")
            else promotion_decision.get("hand_confidence", "unavailable")
        ),
        "final_hand_calculation_errors": (
            multi_hand_diagnostic.get("calculation_errors", {})
            if result.get("combined")
            else promotion_decision.get("calculation_errors", [])
        ),
        "final_hand_t_style_all": (
            _quick_fraction((multi_hand_pooled.get("all") or {}).get("t"))
            if result.get("combined") and multi_hand_pooled
            else None
            if result.get("combined")
            else _quick_fraction(hand_all.get("paired_t_style_positive_favors_final"))
        ),
        "final_hand_t_style_all_n": (
            _quick_int((multi_hand_pooled.get("all") or {}).get("n"))
            if result.get("combined") and multi_hand_pooled
            else None
            if result.get("combined")
            else _quick_int(hand_all.get("n"))
        ),
        "final_hand_t_style_all_df": (
            _quick_int((multi_hand_pooled.get("all") or {}).get("df"))
            if result.get("combined") and multi_hand_pooled
            else None
            if result.get("combined")
            else _quick_int(hand_all.get("df"))
        ),
        "final_hand_t_style_free": (
            _quick_fraction((multi_hand_pooled.get("free") or {}).get("t"))
            if result.get("combined") and multi_hand_pooled
            else None
            if result.get("combined")
            else _quick_fraction(hand_free.get("paired_t_style_positive_favors_final"))
        ),
        "final_hand_t_style_free_n": (
            _quick_int((multi_hand_pooled.get("free") or {}).get("n"))
            if result.get("combined") and multi_hand_pooled
            else None
            if result.get("combined")
            else _quick_int(hand_free.get("n"))
        ),
        "final_hand_t_style_free_df": (
            _quick_int((multi_hand_pooled.get("free") or {}).get("df"))
            if result.get("combined") and multi_hand_pooled
            else None
            if result.get("combined")
            else _quick_int(hand_free.get("df"))
        ),
        "final_hand_t_style_scope": (
            "pooled_exact_rows"
            if result.get("combined") and multi_hand_pooled
            else "per_dataset"
            if result.get("combined")
            else "single_dataset"
        ),
        "final_hand_zkp_all": (
            _quick_fraction((multi_hand_zkp.get("all") or {}).get("z"))
            if result.get("combined") and multi_hand_zkp
            else None
            if result.get("combined")
            else _quick_fraction(hand_all.get("adjusted_zkp_positive_favors_final"))
        ),
        "final_hand_zkp_free": (
            _quick_fraction((multi_hand_zkp.get("free") or {}).get("z"))
            if result.get("combined") and multi_hand_zkp
            else None
            if result.get("combined")
            else _quick_fraction(hand_free.get("adjusted_zkp_positive_favors_final"))
        ),
        "final_hand_zkp_scope": (
            "combined_independent_datasets"
            if result.get("combined") and multi_hand_zkp
            else "per_dataset"
            if result.get("combined")
            else "single_dataset"
        ),
        "final_hand_warnings": (
            list(multi_hand_diagnostic.get("warnings") or [])
            if result.get("combined")
            else list(promotion_decision.get("warnings") or [])
        ),
        "final_hand_zkp_threshold": (
            multi_hand_diagnostic.get("hand_flip_zkp_all_threshold", 3.5)
            if result.get("combined")
            else promotion_decision.get("hand_flip_zkp_all_threshold", 3.5)
        ),
        "final_hand_z_all": (
            None if result.get("combined") else _quick_fraction(hand_all.get("paired_t_style_positive_favors_final"))
        ),
        "final_hand_z_free": (
            None if result.get("combined") else _quick_fraction(hand_free.get("paired_t_style_positive_favors_final"))
        ),
        "final_hand_z_scope": (
            "pooled_exact_rows"
            if result.get("combined") and multi_hand_pooled
            else "per_dataset"
            if result.get("combined")
            else "single_dataset"
        ),
        "final_hand_z_compatibility_note": (
            "legacy field names only; values are paired Student-t-style contrasts, not calibrated normal Z scores"
        ),
        "final_hand_per_dataset": multi_hand_rows,
        "final_hand_consistency_status": multi_hand_diagnostic.get("status", "") if multi_hand_diagnostic else "",
        "selected_transform": transform_summary.get("selected_transform", "n/a"),
        "selected_transform_parity": transform_summary.get("transform_parity", "n/a"),
        "selected_transform_determinant": transform_summary.get("transform_determinant"),
        "configuration_claim_status": transform_summary.get("configuration_status", "parity_unresolved"),
        "absolute_configuration_claim_allowed": bool(
            transform_summary.get("absolute_configuration_claim_allowed", False)
        ),
        "configuration_claim_reason": transform_summary.get(
            "configuration_status_reason", "transform parity contract is missing"
        ),
        "input_shelxl": {
            "R1gt": _quick_fraction(input_shelxl.get("R1_gt")),
            "R1all": _quick_fraction(input_shelxl.get("R1_all")),
            "wR2all": _quick_fraction(input_shelxl.get("wR2_all")),
            "GooF": _quick_fraction(input_shelxl.get("GooF")),
            "parameters": input_shelxl.get("parameters"),
            "restraints": input_shelxl.get("restraints"),
            "npd_count": input_shelxl.get("npd_count"),
            "split_count": input_shelxl.get("split_count"),
            "lst": input_shelxl.get("lst", ""),
        },
        "input_model": {
            "row_R1all": _quick_fraction(input_row_r1),
            "row_wR2all": _quick_fraction(input_row_wr2),
            "born_R1all": _quick_fraction(_quick_born_metric(screen_summary, "all", "R1_all")),
            "born_wR2all": _quick_fraction(_quick_born_metric(screen_summary, "all", "wR2_all")),
        },
        "final_row_model": {
            "row_R1all": _quick_fraction(final_row_r1),
            "row_wR2all": _quick_fraction(final_row_wr2),
            "row_free_wR2all": _quick_fraction(final_free_wr2),
            "born_R1all": _quick_fraction(_quick_metric(native_metrics, "all", "born", "R1_all")),
            "born_wR2all": _quick_fraction(_quick_metric(native_metrics, "all", "born", "wR2_all")),
            "born_free_wR2all": _quick_fraction(_quick_metric(native_metrics, "free", "born", "wR2_all")),
            "row_table": frozen_row_console_text,
            "row_tables": {name: str(path) for name, path in frozen_row_tables},
        },
        "final_feedback_endpoint": {
            "R1gt": _quick_fraction(final_feedback_shelxl.get("R1_gt")),
            "R1all": _quick_fraction(final_feedback_shelxl.get("R1_all")),
            "wR2all": _quick_fraction(final_feedback_shelxl.get("wR2_all")),
            "GooF": _quick_fraction(final_feedback_shelxl.get("GooF")),
            "parameters": final_feedback_shelxl.get("parameters"),
            "restraints": final_feedback_shelxl.get("restraints"),
            "pathology_status": final_feedback_shelxl.get("pathology_status", "unverified"),
            "npd_count": final_feedback_shelxl.get("npd_count"),
            "npd_atoms": final_feedback_shelxl.get("npd_atoms", []),
            "split_count": final_feedback_shelxl.get("split_count"),
            "has_npd": final_feedback_shelxl.get("has_npd"),
            "has_split_warnings": final_feedback_shelxl.get("has_split_warnings"),
            "lst": final_feedback_shelxl.get("lst", file_block["final_feedback_lst"]),
        },
        "publication_endpoint": {
            "baseline_comparison": selected_row.get("baseline_no_harm_gate", {}),
            "R1gt": _quick_fraction(selected_row.get("shelxl_R1gt")),
            "R1all": _quick_fraction(selected_row.get("shelxl_R1all")),
            "wR2all": _quick_fraction(selected_row.get("shelxl_wR2all")),
            "GooF": _quick_fraction(selected_row.get("shelxl_GooF")),
            "scale_spec": selected_row.get("spec", ""),
            "scale_params": selected_row.get("scale_params", ""),
            "wght_mode": selected_row.get("wght_mode") or "n/a",
            "parameters": publication_shelxl.get("parameters"),
            "restraints": publication_shelxl.get("restraints"),
            "pathology_status": selected_row.get("pathology_status")
            or publication_shelxl.get("pathology_status", "unverified"),
            "npd_count": publication_shelxl.get("npd_count"),
            "npd_atoms": publication_shelxl.get("npd_atoms", []),
            "split_count": publication_shelxl.get("split_count"),
            "has_npd": publication_shelxl.get("has_npd"),
            "has_split_warnings": publication_shelxl.get("has_split_warnings"),
            "active_wght_a": publication_shelxl.get("active_wght_a"),
            "active_wght_b": publication_shelxl.get("active_wght_b"),
            "source": selected_row.get("selection_source", ""),
            "source_kind": selected_row.get("source_kind", ""),
            "selection_policy": endpoint_selection.get("selection_policy", "")
            if isinstance(endpoint_selection, Mapping)
            else "",
            "information_criterion": endpoint_selection.get("information_criterion", "")
            if isinstance(endpoint_selection, Mapping)
            else "",
            "information_value": endpoint_selection.get("selected_information_criterion", "")
            if isinstance(endpoint_selection, Mapping)
            else "",
        },
        "diagnostic_endpoint": {
            "R1gt": _quick_fraction(diagnostic_endpoint.get("shelxl_R1gt")),
            "R1all": _quick_fraction(diagnostic_endpoint.get("shelxl_R1all")),
            "wR2all": _quick_fraction(diagnostic_endpoint.get("shelxl_wR2all")),
            "GooF": _quick_fraction(diagnostic_endpoint.get("shelxl_GooF")),
            "source": diagnostic_endpoint.get("selection_source", ""),
            "source_kind": diagnostic_endpoint.get("source_kind", ""),
            "id": diagnostic_endpoint.get("id", ""),
            "rejection_reasons": diagnostic_endpoint.get("rejection_reasons", []),
        },
        "jana_pets_row_statistics": {
            "source": jana_summary.get("source", str(row_metrics_path)),
            "born": dict(jana_born),
            "dynamic": dict(jana_dynamic),
        },
        "table1": str(table1["markdown"]),
        "table1_json": str(table1["json"]),
        "files": file_block,
    }

    return console, selected_row


def _write_quick_stats_final_summary(
    *,
    bloch_diagnostics_path,
    console,
    dataset_nuisance_contracts,
    file_block,
    final_summary_path,
    frozen_row_file_lines,
    input_shelxl,
    jana_born,
    jana_dynamic,
    metadata,
    multi_hand_diagnostic,
    multi_hand_pooled,
    nuisance,
    out_dir,
    publication_shelxl,
    quick_path,
    result,
    selected_row,
    table1,
    heldout_report=None,
):
    """Render and write the short final summary after the complete quick report exists."""

    policy = str(console["publication_endpoint"].get("selection_policy") or "").strip()
    criterion = str(console["publication_endpoint"].get("information_criterion") or "").strip()
    criterion_value = console["publication_endpoint"].get("information_value")
    if not selected_row:
        endpoint_note = "No endpoint chooser-approved physical endpoint was selected. Endpoint-calibration scale replays are diagnostic only for this run."
    elif policy or criterion:
        endpoint_note = (
            f"Endpoint calibration used `{policy or 'n/a'}` with `{criterion or 'n/a'}` = "
            f"`{_quick_number(criterion_value, 3)}`. If the endpoint chooser selected another endpoint, "
            "that endpoint is reported above and the calibration replay remains an audit branch."
        )
    else:
        endpoint_note = (
            "The endpoint chooser selected the publication endpoint. Endpoint-calibration replay details, when present, "
            "are listed in `QUICK_STATS.md` as an audit branch."
        )
    endpoint_warning = _quick_goof_warning(publication_shelxl.get("GooF"))
    input_endpoint_available = _quick_fraction(input_shelxl.get("R1_all")) is not None
    publication_endpoint_available = _quick_fraction(publication_shelxl.get("R1_all")) is not None
    jana_born_available = _quick_complete_jana_metrics(jana_born)
    jana_dynamic_available = _quick_complete_jana_metrics(jana_dynamic)

    if result.get("combined") and multi_hand_pooled:
        final_hand_t_line = (
            f"| final hand paired t-style, all/free | "
            f"`{_quick_flat_hand_t_style_text(console['final_hand_t_style_all'], console['final_hand_t_style_all_n'], console['final_hand_t_style_all_df'])}` / "
            f"`{_quick_flat_hand_t_style_text(console['final_hand_t_style_free'], console['final_hand_t_style_free_n'], console['final_hand_t_style_free_df'])}`; "
            f"exact pooled rows, positive favors the selected hand; selection `{multi_hand_diagnostic.get('status', 'missing')}`; "
            "not a calibrated normal Z score |"
        )
    elif result.get("combined"):
        final_hand_t_line = (
            f"| final hand paired t-style, all/free | `legacy per-dataset only with n and df; no pooled statistic recorded`; consistency "
            f"`{multi_hand_diagnostic.get('status', 'missing')}` |"
        )
    else:
        final_hand_t_line = (
            f"| final hand paired t-style, all/free | "
            f"`{_quick_flat_hand_t_style_text(console['final_hand_t_style_all'], console['final_hand_t_style_all_n'], console['final_hand_t_style_all_df'])}` / "
            f"`{_quick_flat_hand_t_style_text(console['final_hand_t_style_free'], console['final_hand_t_style_free_n'], console['final_hand_t_style_free_df'])}`; "
            "positive favors the selected hand; not a calibrated normal Z score |"
        )

    heldout_rule = console.get("hand_decision_authority") == "heldout_orbit_sign"
    heldout_first = console.get("report_layout") == "heldout-first"
    final_hand_zkp_line = (
        "| final-hand adjusted ZKP, all/free | "
        f"`{_quick_number(console.get('final_hand_zkp_all'), 3)}` / "
        f"`{_quick_number(console.get('final_hand_zkp_free'), 3)}`; "
        f"oriented positive for the final selected hand; "
        + (
            "legacy all-row statistic, diagnostic only: the held-out orbit-sign rule decided the hand |"
            if heldout_rule
            else "no hand decision was evaluated (the hand check was skipped) |"
            if heldout_first and console.get("hand_decision_authority") == "not_evaluated"
            else f"all-row threshold `{_quick_number(console.get('final_hand_zkp_threshold'), 3)}`; "
            "held-out evidence warns but does not veto |"
        )
    )
    hand_summary = list((console.get("heldout_validation") or {}).get("hand_summary") or [])
    heldout_hand_rows = (
        [f"| held-out hand evidence | {'**' + text + '**' if text.startswith('WARNING') else text} |" for text in hand_summary]
        or ["| held-out hand evidence | `not recorded` |"]
        if heldout_first
        else []
    )

    final_lines = [
        "# GROND Final Summary",
        "",
        "This is the compact user-facing summary. `QUICK_STATS.md` and the JSON reports keep the full audit trail.",
        "",
        # --report-layout heldout-first: the held-out validation opens the summary.
        *(heldout_markdown_lines(heldout_report) if heldout_first else []),
        "## Decision",
        "",
        "| item | value |",
        "|---|---|",
        f"| final hand representation | `{console['final_hand']}` |",
        *heldout_hand_rows,
        f"| {'held-out' if heldout_rule else 'all-row'} preferred hand | `{console.get('final_hand_preferred', 'unavailable')}`; `{console.get('final_hand_confidence', 'unavailable')}`; conditional statistic, not calibrated sigma confidence |",
        final_hand_zkp_line,
        final_hand_t_line,
        f"| selected HKL transform | `{console['selected_transform']}` |",
        f"| transform parity / determinant | `{console.get('selected_transform_parity', 'n/a')}` / `{console.get('selected_transform_determinant', 'per-dataset')}` |",
        f"| configuration claim status | `{console.get('configuration_claim_status', 'parity_unresolved')}`; named absolute configuration allowed `{bool(console.get('absolute_configuration_claim_allowed', False))}` |",
        f"| final N/Q | `{metadata.get('n_basis', 'n/a')}/{metadata.get('n_points', 'n/a')}` |",
        f"| effective thickness model | `{_quick_thickness_text(nuisance, metadata, dataset_nuisance_contracts)}`; Sg window `{metadata.get('sg_window', 'n/a')}` |",
        f"| publication endpoint source | `{console['publication_endpoint'].get('source') or 'none'}` / `{console['publication_endpoint'].get('source_kind') or 'no_physical_endpoint'}` |",
        "",
        *(_quick_multi_hand_evidence_lines(multi_hand_diagnostic, compact=True) if result.get("combined") else []),
        "## Canonical Statistics",
        "",
        f"The authoritative human-readable separation is in `{table1['markdown']}` and the machine-readable form is `{table1['json']}`. SHELXL endpoint factors and JANA/PETS-style row factors are never combined into generic R columns.",
        "",
        "### A. SHELXL Endpoint Statistics",
        "",
        "| model | status | Ngt/Nall | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | parameters/restraints | NPD/split |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| conventional kinematic reference | {'available' if input_endpoint_available else 'not available'} | {_table1_pair({'Ngt': input_shelxl.get('N_gt'), 'Nall': input_shelxl.get('N_all')}, 'Ngt', 'Nall')} | {_quick_percent(input_shelxl.get('R1_gt'))} | {_quick_percent(input_shelxl.get('R1_all'))} | {_quick_percent(input_shelxl.get('wR2_all'))} | {_quick_number(input_shelxl.get('GooF'), 3)} | {_table1_pair({'parameters': input_shelxl.get('parameters'), 'restraints': input_shelxl.get('restraints')}, 'parameters', 'restraints')} | {_table1_pair({'NPD': input_shelxl.get('npd_count'), 'split': input_shelxl.get('split_count')}, 'NPD', 'split')} |",
        "| GROND Born corrected-HKL endpoint | not available | n/a | n/a | n/a | n/a | n/a | n/a | n/a |",
        f"| GROND Bloch dynamical model-handoff endpoint | {'available' if publication_endpoint_available else 'not available'} | {_table1_pair({'Ngt': publication_shelxl.get('N_gt'), 'Nall': publication_shelxl.get('N_all')}, 'Ngt', 'Nall')} | {_quick_percent(publication_shelxl.get('R1_gt'))} | {_quick_percent(publication_shelxl.get('R1_all'))} | {_quick_percent(publication_shelxl.get('wR2_all'))} | {_quick_number(publication_shelxl.get('GooF'), 3)} | {_table1_pair({'parameters': publication_shelxl.get('parameters'), 'restraints': publication_shelxl.get('restraints')}, 'parameters', 'restraints')} | {_table1_pair({'NPD': publication_shelxl.get('npd_count'), 'split': publication_shelxl.get('split_count')}, 'NPD', 'split')} |",
        "",
        "### B. JANA/PETS-Style Row And Post-Refinement Merged Statistics",
        "",
        "| model | status | Ninput/excluded | Nobs/Nall | Nmerge used/excluded | MNobs/MNall | Robs | Rall | wRall | MRobs | MRall | MwRall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| conventional kinematic refinement | not available | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |",
        f"| GROND matched Born | {'available' if jana_born_available else 'not available'} | {_table1_pair(jana_born, 'Ninput', 'Nexcluded')} | {_quick_count_pair(jana_born)} | {_table1_pair(jana_born, 'Nmerge_used', 'Nmerge_excluded')} | {_table1_pair(jana_born, 'MNobs', 'MNall')} | {_quick_percent(jana_born.get('Robs'))} | {_quick_percent(jana_born.get('Rall'))} | {_quick_percent(jana_born.get('wRall'))} | {_quick_percent(jana_born.get('MRobs'))} | {_quick_percent(jana_born.get('MRall'))} | {_quick_percent(jana_born.get('MwRall'))} |",
        f"| GROND Bloch dynamical | {'available' if jana_dynamic_available else 'not available'} | {_table1_pair(jana_dynamic, 'Ninput', 'Nexcluded')} | {_quick_count_pair(jana_dynamic)} | {_table1_pair(jana_dynamic, 'Nmerge_used', 'Nmerge_excluded')} | {_table1_pair(jana_dynamic, 'MNobs', 'MNall')} | {_quick_percent(jana_dynamic.get('Robs'))} | {_quick_percent(jana_dynamic.get('Rall'))} | {_quick_percent(jana_dynamic.get('wRall'))} | {_quick_percent(jana_dynamic.get('MRobs'))} | {_quick_percent(jana_dynamic.get('MRall'))} | {_quick_percent(jana_dynamic.get('MwRall'))} |",
        "",
        *([] if heldout_first else heldout_markdown_lines(heldout_report)),
        "## Within-Family Deltas",
        "",
        "| comparison | dRall | d weighted R | dMRall | dMwRall |",
        "|---|---:|---:|---:|---:|",
        f"| Bloch minus matched Born, JANA/PETS-style rows | {_quick_delta_percent(jana_born.get('Rall'), jana_dynamic.get('Rall'))} | {_quick_delta_percent(jana_born.get('wRall'), jana_dynamic.get('wRall'))} | {_quick_delta_percent(jana_born.get('MRall'), jana_dynamic.get('MRall'))} | {_quick_delta_percent(jana_born.get('MwRall'), jana_dynamic.get('MwRall'))} |",
        f"| GROND publication minus conventional reference, SHELXL endpoints | {_quick_delta_percent(input_shelxl.get('R1_all'), publication_shelxl.get('R1_all'))} | {_quick_delta_percent(input_shelxl.get('wR2_all'), publication_shelxl.get('wR2_all'))} | n/a | n/a |",
        "",
        f"**Endpoint warning:** {endpoint_warning}." if endpoint_warning else "",
        *["**Endpoint warning:** " + str(warning) for warning in
          (console["publication_endpoint"].get("baseline_comparison") or {}).get("warnings", [])],
        "",
        endpoint_note,
        "",
        "## Files",
        "",
        f"- Canonical final Table 1: `{table1['markdown']}`",
        f"- Machine-readable final Table 1: `{table1['json']}`",
        f"- Combined JANA/PETS-style row report: `{file_block['jana_row_statistics']}`",
        f"- Per-dataset JANA/PETS-style row report: `{file_block['jana_row_statistics_per_dataset']}`",
        *([f"- Final model, refined against all reflections (--free-set off): `{file_block['all_data_res']}`",
           f"- Its corrected HKL (work and free reflections): `{file_block['all_data_hkl']}`",
           f"- Its SHELXL listing: `{file_block['all_data_lst']}`"] if file_block.get("all_data_res") else []),
        f"- Open this publication endpoint model: `{file_block['publication_res']}`",
        f"- Use this corrected HKL with that model: `{file_block['publication_hkl']}`",
        f"- The endpoint statistics above come from: `{file_block['publication_lst']}`",
        *frozen_row_file_lines,
        f"- Bloch matrix and row-contract diagnostics: `{bloch_diagnostics_path}`",
        *([f"- Held-out validation: `{(heldout_report.get('artifacts') or {}).get('markdown')}`"] if heldout_report else []),
        f"- Full quick report: `{quick_path}`",
        f"- Full machine-readable run summary: `{out_dir / 'grond_summary.json'}`",
        "",
        "## Continue",
        "",
        f"- Checkpoint: `{out_dir / 'grond_continue_checkpoint.json'}`",
        f"- To push from an edited model: `GROND push-endpoint --checkpoint {out_dir / 'grond_continue_checkpoint.json'} --start-res edited.res --out-dir {out_dir / 'endpoint_push_edited'}`",
        f"- To regenerate this package: `GROND finish --run-dir {out_dir}`",
        "",
        "## Notes",
        "",
        "Native maps made by `GROND finish` use the frozen per-dataset GROND row table or tables, the final model phase, and dynamically inferred electron amplitudes. Use `--map-mode shelx2map` only for a fast corrected-HKL endpoint-map check.",
    ]
    final_summary_path.write_text("\n".join(final_lines) + "\n")
