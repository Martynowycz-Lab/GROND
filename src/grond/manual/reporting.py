"""Render the manual workflow with row factors, endpoint factors and hand evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from grond.manual.state import _read_json, _stage


def _finite_float(value: object) -> float | None:
    """Return ``value`` as a float, or None when it is missing or not finite."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percent(value: object) -> str:
    """Format a fraction as a percentage with two decimals; ``n/a`` when not finite."""

    number = _finite_float(value)
    return "n/a" if number is None else f"{100.0 * number:.2f}%"


def _number(value: object, digits: int = 2, *, signed: bool = False) -> str:
    """Format a number with ``digits`` decimals (with a sign when ``signed``); ``n/a`` when not finite."""

    number = _finite_float(value)
    if number is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}"


def _selected_hand_summary(workflow: Mapping[str, object]) -> dict[str, object]:
    """Summarize the hand stage's adjusted ZKP per crystal and combined, signed for the selected coordinates."""

    hand = _stage(workflow, "hand")
    outputs = hand.get("outputs") if isinstance(hand.get("outputs"), Mapping) else {}
    choice = str(outputs.get("choice") or "unknown")
    orientation = -1.0 if choice == "inverse" else 1.0
    dataset_paths = outputs.get("dataset_diagnostics")
    datasets: dict[str, object] = {}
    combined: dict[str, object] = {}
    numerators = {"all": 0.0, "free": 0.0}
    variances = {"all": 0.0, "free": 0.0}
    usable = {"all": 0, "free": 0}
    if isinstance(dataset_paths, Mapping):
        for name, path_text in dataset_paths.items():
            path = Path(str(path_text))
            if not path.is_file():
                continue
            diagnostic = _read_json(path)
            row: dict[str, object] = {"diagnostic": str(path)}
            for support in ("all", "free"):
                support_block = diagnostic.get(support)
                support_block = support_block if isinstance(support_block, Mapping) else {}
                kp = support_block.get("klar_palatinus_adjusted_hand_z")
                kp = kp if isinstance(kp, Mapping) else {}
                z_input = _finite_float(kp.get("z"))
                n = int(kp.get("n", 0) or 0)
                k_effective = _finite_float(
                    kp.get(
                        "k_effective_selected_wins_plus_half_ties",
                        kp.get("k_selected_lower_abs_intensity_residual"),
                    )
                )
                w = _finite_float(kp.get("w"))
                z_selected = orientation * z_input if z_input is not None else None
                row[support] = {
                    "n": n,
                    "z_input_coordinates": z_input,
                    "z_selected_coordinates": z_selected,
                    "paired_row_t_input_coordinates": _finite_float(support_block.get("paired_row_t_delta_chi2")),
                }
                if k_effective is not None and w is not None and n > w:
                    numerators[support] += orientation * (2.0 * k_effective - n)
                    variances[support] += n - w
                    usable[support] += 1
            datasets[str(name)] = row
    for support in ("all", "free"):
        combined[support] = {
            "z_selected_coordinates": (
                numerators[support] / math.sqrt(variances[support])
                if usable[support] > 0 and variances[support] > 0.0
                else None
            ),
            "dataset_count": usable[support],
            "combination": "independent sum-score diagnostic: sum(2k_eff-N)/sqrt(sum(N-w))",
        }
    return {
        "scope": "pre_feedback_thickness_stage_model",
        "choice": choice,
        "recommendation": outputs.get("recommendation", "not calculated"),
        "selected_model": outputs.get("chosen_model", ""),
        "known_chemistry_used": False,
        "sign_definition": "positive Z favors the coordinates selected before feedback; no final post-feedback hand recheck is implied",
        "datasets": datasets,
        "combined": combined,
    }


def _manual_row_statistics(row_runs: Mapping[str, Path], model: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Return the combined and per-crystal row R statistics of the selected rows for ``model``."""

    from grond.engine.bloch_engine import parse_model
    from grond.engine.row_mr_report import (
        compute_per_dataset_row_mr,
        compute_row_mr_summary,
        read_csv_rows,
        standard_json_data,
    )

    rows: list[dict] = []
    for name, run in row_runs.items():
        rows.extend(read_csv_rows(run / "native_dyn_calc.csv", name))
    parsed_model = parse_model(model)
    combined = compute_row_mr_summary(rows, parsed_model, instability_u=0.01)
    per_dataset = compute_per_dataset_row_mr(rows, parsed_model, instability_u=0.01)
    return standard_json_data(combined), standard_json_data(per_dataset)


def _manual_scale_summary(
    workflow: Mapping[str, object],
    row_runs: Mapping[str, Path],
) -> dict[str, object]:
    """Summarize the applied full scale parameters per crystal and the compact scale audit (not applied)."""

    datasets: dict[str, object] = {}
    frame_total = 0
    detector_total = 0
    row_total = 0
    for name, run in row_runs.items():
        summary = _read_json(run / "summary.json")
        best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
        frame = int(best.get("frame_scale_count", 0) or 0)
        detector = int(best.get("detector_scale_count", 0) or 0)
        n_rows = int(best.get("valid_rows", 0) or 0)
        frame_total += frame
        detector_total += detector
        row_total += n_rows
        datasets[name] = {
            "rows": n_rows,
            "frame_parameters": frame,
            "detector_parameters": detector,
            "rows_per_scale_parameter": n_rows / max(frame + detector, 1),
            "fit_split": best.get("scale_fit_split", ""),
        }
    scales_stage = _stage(workflow, "scales")
    scale_outputs = scales_stage.get("outputs") if isinstance(scales_stage.get("outputs"), Mapping) else {}
    audit_datasets = scale_outputs.get("datasets") if isinstance(scale_outputs.get("datasets"), Mapping) else {}
    compact: dict[str, object] = {}
    compact_frame_total = 0
    compact_detector_total = 0
    for name, audit in audit_datasets.items():
        selected = audit.get("selected") if isinstance(audit, Mapping) else {}
        selected = selected if isinstance(selected, Mapping) else {}
        frame = int(selected.get("group_scale_params", 0) or 0)
        detector = int(selected.get("detector_scale_params", 0) or 0)
        compact_frame_total += frame
        compact_detector_total += detector
        compact[str(name)] = dict(selected)
    return {
        "applied_contract": "full pinned frame grouping",
        "application_note": scale_outputs.get("application", "scale audit not run; full pinned scales remain active"),
        "applied": {
            "rows": row_total,
            "frame_parameters": frame_total,
            "detector_parameters": detector_total,
            "rows_per_scale_parameter": row_total / max(frame_total + detector_total, 1),
            "datasets": datasets,
        },
        "compact_audit": {
            "applied_to_endpoint": False,
            "frame_parameters": compact_frame_total,
            "detector_parameters": compact_detector_total,
            "datasets": compact,
        },
    }


def _manual_markdown_summary(payload: Mapping[str, object]) -> str:
    """Render the final manual-workflow summary payload as Markdown."""

    endpoint = payload.get("endpoint") if isinstance(payload.get("endpoint"), Mapping) else {}
    baseline = payload.get("starting_endpoint") if isinstance(payload.get("starting_endpoint"), Mapping) else {}
    row_stats = payload.get("row_statistics") if isinstance(payload.get("row_statistics"), Mapping) else {}
    combined_rows = row_stats.get("combined") if isinstance(row_stats.get("combined"), Mapping) else {}
    born = combined_rows.get("born") if isinstance(combined_rows.get("born"), Mapping) else {}
    dynamic = combined_rows.get("dynamic") if isinstance(combined_rows.get("dynamic"), Mapping) else {}
    hand = payload.get("hand") if isinstance(payload.get("hand"), Mapping) else {}
    scales = payload.get("scales") if isinstance(payload.get("scales"), Mapping) else {}
    applied_scales = scales.get("applied") if isinstance(scales.get("applied"), Mapping) else {}
    starting_note = (
        ""
        if baseline.get("comparable_to_selected_support", False)
        else f"Starting endpoint note: {baseline.get('reason', 'not available')}."
    )
    lines = [
        "# GROND Manual Workflow Summary",
        "",
        f"Selected model: `{payload.get('model', '')}`",
        "",
        "## Headline result",
        "",
        "| Endpoint | R1gt | R1all | wR2all | GooF | NPD/split |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Starting conventional | {_percent(baseline.get('R1_gt'))} | {_percent(baseline.get('R1_all'))} | {_percent(baseline.get('wR2_all'))} | {baseline.get('GooF', 'n/a')} | {baseline.get('npd_count', 'n/a')}/{baseline.get('split_count', 'n/a')} |",
        f"| Selected GROND | {_percent(endpoint.get('R1_gt'))} | {_percent(endpoint.get('R1_all'))} | {_percent(endpoint.get('wR2_all'))} | {endpoint.get('GooF', 'n/a')} | {endpoint.get('npd_count', 'n/a')}/{endpoint.get('split_count', 'n/a')} |",
        "",
        starting_note,
        "" if starting_note else "",
        f"Selected feedback cycle: **{payload.get('selected_cycle', 'n/a')}**. "
        f"{(payload.get('feedback_architecture') or {}).get('summary', '')}",
        "",
        "## JANA/PETS-style matched row statistics",
        "",
        "| Model | Robs | Rall | wRall | MRobs | MRall | MwRall |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Born | {_percent(born.get('Robs'))} | {_percent(born.get('Rall'))} | {_percent(born.get('wRall'))} | {_percent(born.get('MRobs'))} | {_percent(born.get('MRall'))} | {_percent(born.get('MwRall'))} |",
        f"| Bloch | {_percent(dynamic.get('Robs'))} | {_percent(dynamic.get('Rall'))} | {_percent(dynamic.get('wRall'))} | {_percent(dynamic.get('MRobs'))} | {_percent(dynamic.get('MRall'))} | {_percent(dynamic.get('MwRall'))} |",
        "",
        "| Dataset | Bloch Rall | Bloch wRall | Bloch MRall | Bloch MwRall | Born to Bloch Rall gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    per_dataset_stats = row_stats.get("per_dataset") if isinstance(row_stats.get("per_dataset"), Mapping) else {}
    for name, row in per_dataset_stats.items():
        dataset_born = row.get("born") if isinstance(row, Mapping) and isinstance(row.get("born"), Mapping) else {}
        dataset_dynamic = (
            row.get("dynamic") if isinstance(row, Mapping) and isinstance(row.get("dynamic"), Mapping) else {}
        )
        born_rall = _finite_float(dataset_born.get("Rall"))
        dyn_rall = _finite_float(dataset_dynamic.get("Rall"))
        gain = born_rall - dyn_rall if born_rall is not None and dyn_rall is not None else None
        lines.append(
            f"| {name} | {_percent(dataset_dynamic.get('Rall'))} | {_percent(dataset_dynamic.get('wRall'))} | "
            f"{_percent(dataset_dynamic.get('MRall'))} | {_percent(dataset_dynamic.get('MwRall'))} | {_percent(gain)} |"
        )
    lines.extend(
        [
            "",
            "## Pre-feedback hand decision",
            "",
            f"Coordinate choice before feedback: **{hand.get('choice', 'n/a')}**; calculation recommendation: **{hand.get('recommendation', 'n/a')}**. Positive values below favor those pre-feedback coordinates. This simplified workflow does not claim auto's final post-feedback shared-hand selection.",
            "",
            "| Dataset | ZKP all | ZKP free |",
            "|---|---:|---:|",
        ]
    )
    for name, row in (hand.get("datasets") or {}).items():
        all_block = row.get("all") or {}
        free_block = row.get("free") or {}
        lines.append(
            f"| {name} | {_number(all_block.get('z_selected_coordinates'), signed=True)} | "
            f"{_number(free_block.get('z_selected_coordinates'), signed=True)} |"
        )
    combined_hand = hand.get("combined") if isinstance(hand.get("combined"), Mapping) else {}
    lines.append(
        f"| Explicit combined score | {_number((combined_hand.get('all') or {}).get('z_selected_coordinates'), signed=True)} | "
        f"{_number((combined_hand.get('free') or {}).get('z_selected_coordinates'), signed=True)} |"
    )
    lines.extend(
        [
            "",
            "## Per-dataset physics and applied scales",
            "",
            "| Dataset | Transform | Thickness | Gradient | Frame offset | Frame scales | Detector scales |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in (payload.get("datasets") or {}).items():
        nuisance = row.get("nuisance") or {}
        scale_row = (applied_scales.get("datasets") or {}).get(name, {})
        thickness_A = _finite_float(nuisance.get("thickness_A"))
        thickness = "n/a" if thickness_A is None else f"{thickness_A / 10.0:.1f} nm"
        lines.append(
            f"| {name} | `{row.get('hkl_transform', '')}` | {thickness} | "
            f"{nuisance.get('thickness_grad_frac', 'n/a')} | {nuisance.get('frame_offset', 'n/a')} | "
            f"{scale_row.get('frame_parameters', 'n/a')} | {scale_row.get('detector_parameters', 'n/a')} |"
        )
    compact = scales.get("compact_audit") if isinstance(scales.get("compact_audit"), Mapping) else {}
    lines.extend(
        [
            "",
            f"The selected endpoint uses **{applied_scales.get('frame_parameters', 'n/a')} frame** and **{applied_scales.get('detector_parameters', 'n/a')} detector** parameters across {applied_scales.get('rows', 'n/a')} rows. The compact audit found {compact.get('frame_parameters', 'n/a')} frame and {compact.get('detector_parameters', 'n/a')} detector parameters, but **did not apply them to this endpoint**.",
            "",
            "## Feedback-cycle audit",
            "",
            "| Cycle | Lagged native row R1all | Lagged native row wR2all | Endpoint R1all | Endpoint wR2all | GooF | NPD/split | Selected |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for cycle in payload.get("feedback_cycles") or []:
        lines.append(
            f"| {cycle.get('cycle', '')} | {_percent(cycle.get('dyn_R1_all'))} | {_percent(cycle.get('dyn_wR2_all'))} | "
            f"{_percent(cycle.get('endpoint_R1_all'))} | {_percent(cycle.get('endpoint_wR2_all'))} | "
            f"{cycle.get('endpoint_GooF', 'n/a')} | {cycle.get('npd_count', 'n/a')}/{cycle.get('split_count', 'n/a')} | "
            f"{'yes' if cycle.get('selected') else ''} |"
        )
    lines.extend(
        [
            "",
            "The cycle table contains the lagged row objective used for cycle selection. The JANA/PETS-style table above is recomputed once from the selected endpoint coordinates, without another coordinate update, and is the row table to report for the final model.",
        ]
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"RES: `{endpoint.get('res', '')}`  ",
            f"HKL: `{endpoint.get('hkl', '')}`  ",
            f"LST: `{endpoint.get('lst', '')}`  ",
            f"JANA row statistics: `{row_stats.get('json', '')}`  ",
            f"Map manifest: `{payload.get('map_manifest', '')}`",
            "",
        ]
    )
    return "\n".join(lines)
