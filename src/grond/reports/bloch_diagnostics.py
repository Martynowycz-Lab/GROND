"""Report row-level Bloch fit diagnostics with the source tables attached."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.reports.distributions import _quick_csv_numeric_audit, _quick_distribution_row
from grond.reports.values import _quick_best_native, _quick_nuisance, _quick_number
from grond.workflows.decision import safe_label


def _write_bloch_diagnostics_report(
    out_dir: Path,
    final_run: Path | None,
    final_summary: Mapping[str, object],
    row_metrics: Mapping[str, object],
    file_block: Mapping[str, str],
    *,
    row_table_override: Path | None = None,
    report_name: str = "BLOCH_DIAGNOSTICS.md",
) -> Path:
    """Write a human-readable Bloch-matrix/row-contract audit report."""

    report = out_dir / report_name
    row_table = (
        row_table_override
        if row_table_override is not None
        else final_run / "native_dyn_calc.csv"
        if final_run
        else Path(str(file_block.get("publication_row_table") or ""))
    )
    config = final_summary.get("config") if isinstance(final_summary.get("config"), Mapping) else {}
    best_native = _quick_best_native(final_summary)
    nuisance = _quick_nuisance(final_summary)
    native_weight = best_native.get("native_weight") if isinstance(best_native.get("native_weight"), Mapping) else {}
    scale_model = best_native.get("scale_model") if isinstance(best_native.get("scale_model"), Mapping) else {}
    scale_text = (
        f"P-spline in scan angle; n_basis `{scale_model.get('n_basis', 'n/a')}`; lambda `{scale_model.get('lambda', 'n/a')}`; "
        f"EDF `{_quick_number(scale_model.get('edf'), 2)}`"
        if scale_model.get("kind") == "pspline"
        else f"group width `{_quick_number(nuisance.get('group_width_deg'), 6)}` deg; detector bins `{nuisance.get('detector_scale_bins', 'n/a')}`; scale count `{best_native.get('scale_count', 'n/a')}`"
    )
    row_selection = final_summary.get("row_selection") if isinstance(final_summary.get("row_selection"), Mapping) else {}
    filters = final_summary.get("filters") if isinstance(final_summary.get("filters"), Mapping) else {}
    diagnostics = final_summary.get("diagnostics") if isinstance(final_summary.get("diagnostics"), Mapping) else {}
    metadata = row_metrics.get("metadata", {}) if isinstance(row_metrics.get("metadata"), Mapping) else {}
    columns = [
        "d_A",
        "basis_size",
        "q_dyn_over_born",
        "center_sg_inv_A",
        "max_sg_solve_residual",
        "sg_window_inv_A",
        "sg_window_raw_inv_A",
        "sg_window_half_frames",
        "sg_dSg_domega_inv_A_per_deg",
        "sg_dSg_dframe_inv_A",
        "sg_jitter_sigma_inv_A",
        "sg_profile_points_total",
        "sg_profile_weight_sum",
        "omega_bragg_minus_center_deg",
        "row_resolution_factor",
        "dyn_residual_over_sigma",
        "born_residual_over_sigma",
        "sigma_eff",
        "peak",
        "corr",
        "rlp",
    ]
    audit = _quick_csv_numeric_audit(row_table, columns)
    column_stats = audit.get("columns") if isinstance(audit.get("columns"), Mapping) else {}
    row_plan = diagnostics.get("row_plan_cache") if isinstance(diagnostics.get("row_plan_cache"), Mapping) else {}
    ug_cache = diagnostics.get("ug_cache") if isinstance(diagnostics.get("ug_cache"), Mapping) else {}
    geometry = row_selection.get("geometry_filter") if isinstance(row_selection.get("geometry_filter"), Mapping) else {}
    row_count = audit.get("rows") or row_selection.get("selected_rows") or filters.get("selected_rows") or metadata.get("row_count", "n/a")
    lines = [
        "# GROND Bloch Diagnostics",
        "",
        "This report audits the forward Bloch calculation and row contract used by the final GROND row table. It is meant for later method review, not for endpoint promotion by itself.",
        "",
        "## Inputs",
        "",
        "| item | value |",
        "|---|---|",
        f"| final row table | `{row_table}` |",
        f"| row table exists | `{audit.get('exists')}` |",
        f"| rows in audit table | `{row_count}` |",
        f"| split counts | `{audit.get('split_counts', {})}` |",
        f"| scale groups / detector bins | `{audit.get('group_count', 'n/a')}` / `{audit.get('detector_bin_count', 'n/a')}` |",
        f"| final model | `{file_block.get('final_feedback_res', '')}` |",
        "",
        "## Bloch Kernel Contract",
        "",
        "| item | value |",
        "|---|---|",
        f"| N / Q | `{config.get('n_basis', metadata.get('n_basis', 'n/a'))}` / `{config.get('n_points', metadata.get('n_points', 'n/a'))}` |",
        f"| matrix convention | `{config.get('matrix_convention', nuisance.get('matrix_convention', 'n/a'))}` |",
        f"| profile quadrature / accumulation | `{config.get('profile_quadrature', 'n/a')}` / `{config.get('profile_accumulation', 'n/a')}` |",
        f"| max support Sg | `{_quick_number(config.get('max_support_sg'), 6)}` 1/A |",
        f"| top-up dmin | `{_quick_number(config.get('topup_d_min'), 3)}` A |",
        f"| row-plan cache | `{row_plan.get('status', row_plan or config.get('row_plan_cache', 'n/a'))}`; unique diffs `{diagnostics.get('row_plan_cache_unique_diffs', 0)}` |",
        f"| U_g cache | `{ug_cache.get('status', ug_cache or 'n/a')}` |",
        "",
        "## Sg, Rocking, And Quadrature",
        "",
        "| quantity | units | n | min | p05 | median | mean | p95 | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        _quick_distribution_row(column_stats.get("center_sg_inv_A", {}), "center excitation error Sg", "1/A"),
        _quick_distribution_row(column_stats.get("max_sg_solve_residual", {}), "max target-Sg solve residual", "1/A"),
        _quick_distribution_row(column_stats.get("sg_window_inv_A", {}), "accepted Sg window", "1/A"),
        _quick_distribution_row(column_stats.get("sg_window_raw_inv_A", {}), "raw Sg window before clipping", "1/A"),
        _quick_distribution_row(column_stats.get("sg_window_half_frames", {}), "Sg window half frames", "frames"),
        _quick_distribution_row(column_stats.get("sg_dSg_domega_inv_A_per_deg", {}), "dSg/domega", "1/A/deg"),
        _quick_distribution_row(column_stats.get("sg_dSg_dframe_inv_A", {}), "dSg/dframe", "1/A/frame"),
        _quick_distribution_row(column_stats.get("sg_jitter_sigma_inv_A", {}), "Sg jitter sigma", "1/A"),
        _quick_distribution_row(column_stats.get("sg_profile_points_total", {}), "profile points after jitter", "points"),
        _quick_distribution_row(column_stats.get("sg_profile_weight_sum", {}), "profile weight sum", "weight"),
        _quick_distribution_row(column_stats.get("omega_bragg_minus_center_deg", {}), "omega Bragg minus row center", "deg"),
        f"| Sg-window clipped rows | count | {audit.get('sg_window_clipped_count', 0)} | n/a | n/a | n/a | n/a | n/a | n/a |",
        "",
        "## Matrix Size And Dynamical Strength",
        "",
        "| quantity | units | n | min | p05 | median | mean | p95 | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        _quick_distribution_row(column_stats.get("basis_size", {}), "basis size", "beams"),
        _quick_distribution_row(column_stats.get("q_dyn_over_born", {}), "q = Idyn/Iborn", "ratio"),
        _quick_distribution_row(column_stats.get("row_resolution_factor", {}), "row resolution factor", "ratio"),
        _quick_distribution_row(column_stats.get("dyn_residual_over_sigma", {}), "dynamical residual/sigma", "sigma"),
        _quick_distribution_row(column_stats.get("born_residual_over_sigma", {}), "Born residual/sigma", "sigma"),
        "",
        "## Row Contract And Geometry Proxies",
        "",
        "| quantity | units | n | min | p05 | median | mean | p95 | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        _quick_distribution_row(column_stats.get("d_A", {}), "d spacing", "A"),
        _quick_distribution_row(column_stats.get("sigma_eff", {}), "effective sigma", "I units"),
        _quick_distribution_row(column_stats.get("peak", {}), "PEAK", "XDS percent"),
        _quick_distribution_row(column_stats.get("corr", {}), "CORR", "XDS percent"),
        _quick_distribution_row(column_stats.get("rlp", {}), "RLP", "XDS units"),
        "",
        "## Geometry And Nuisance Settings",
        "",
        "| item | value |",
        "|---|---|",
        f"| row selection min d / min peak | `{config.get('min_d_spacing', 'n/a')}` A / `{config.get('min_peak', 'n/a')}` |",
        f"| geometry percentile gate | `{config.get('geometry_percentile', 'n/a')}` |",
        f"| geometry filter summary | `{geometry}` |",
        f"| thickness model | `{nuisance.get('thickness_model', 'n/a')}`; center `{_quick_number(nuisance.get('thickness_A', nuisance.get('thickness_grid')), 3)}` A; gradient `{_quick_number(nuisance.get('thickness_grad_frac', nuisance.get('thickness_grad_grid')), 6)}` |",
        f"| Sg jitter | `{nuisance.get('sg_jitter_mode', 'n/a')}`; sigma0 `{_quick_number(nuisance.get('sg_jitter_sigma0'), 6)}`; sigma d^-1 `{_quick_number(nuisance.get('sg_jitter_sigma_d_inv'), 6)}`; sigma frame `{_quick_number(nuisance.get('sg_jitter_sigma_frame'), 6)}`; points `{nuisance.get('sg_jitter_points', 'n/a')}` |",
        f"| absorption | `{nuisance.get('absorption_mode', 'n/a')}`; mu `{_quick_number(nuisance.get('absorption_mu'), 6)}` |",
        f"| scale model | {scale_text} |",
        f"| native row weights | a `{_quick_number(native_weight.get('a'), 6)}`; b `{_quick_number(native_weight.get('b'), 6)}` |",
        "",
        "## Availability Notes",
        "",
        "Rows from older GROND versions may not contain the Sg/quadrature columns. In that case this report still records the high-level config and row residual distributions, but per-row excitation-error distributions require rerunning with this version or using a row-plan cache that records Sg profiles.",
        "",
    ]
    report.write_text("\n".join(lines))
    return report


def _write_bloch_diagnostics_reports(
    out_dir: Path,
    row_tables: list[tuple[str, Path]],
    final_run: Path | None,
    final_summary: Mapping[str, object],
    row_metrics: Mapping[str, object],
    file_block: Mapping[str, str],
) -> Path:
    """Write one exact Bloch audit per dataset and a multi-run index."""

    if len(row_tables) <= 1:
        override = row_tables[0][1] if row_tables else None
        summary = try_read_json(override.parent / "summary.json") if override is not None else final_summary
        return _write_bloch_diagnostics_report(
            out_dir,
            final_run,
            summary if isinstance(summary, Mapping) and summary else final_summary,
            row_metrics,
            file_block,
            row_table_override=override,
        )

    entries: list[tuple[str, Path, int]] = []
    for label, row_table in row_tables:
        summary = try_read_json(row_table.parent / "summary.json")
        report_name = f"BLOCH_DIAGNOSTICS_{safe_label(label)}.md"
        report = _write_bloch_diagnostics_report(
            out_dir,
            row_table.parent,
            summary if isinstance(summary, Mapping) and summary else {},
            row_metrics,
            file_block,
            row_table_override=row_table,
            report_name=report_name,
        )
        audit = _quick_csv_numeric_audit(row_table, [])
        entries.append((label, report, int(audit.get("rows", 0) or 0)))
    index = out_dir / "BLOCH_DIAGNOSTICS.md"
    lines = [
        "# GROND Multi-Dataset Bloch Diagnostics",
        "",
        "Each crystal retains its own frozen row table, nuisance model, scale field, and Bloch audit. No synthetic combined row table is implied by this index.",
        "",
        "| dataset | frozen rows | diagnostics | row table |",
        "|---|---:|---|---|",
    ]
    table_by_label = {label: path for label, path in row_tables}
    for label, report, rows in entries:
        lines.append(f"| {label} | {rows} | `{report}` | `{table_by_label[label]}` |")
    index.write_text("\n".join(lines) + "\n")
    return index
