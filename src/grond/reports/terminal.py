"""Render the concise terminal handoff from the recorded final result."""

from __future__ import annotations

from collections.abc import Mapping

from grond.progress import format_seconds
from grond.reports.shelxl import _quick_endpoint_scale_text, _quick_goof_warning, _quick_shelxl_issue_text
from grond.reports.source_controls import _quick_delta_percent, _quick_flat_hand_t_style_text
from grond.reports.table1 import _quick_complete_jana_metrics
from grond.reports.values import _quick_count_pair, _quick_fraction, _quick_number, _quick_percent


def _plain_number(value: object, digits: int = 2, *, signed: bool = False) -> str:
    """Format a number with fixed digits (signed when asked), n/a when it is not a finite number."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number or number in {float("inf"), float("-inf")}:
        return "n/a"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _plain_ci(values: object) -> str:
    """Format a two-value confidence interval as [lo, hi], [n/a] when it is not a pair."""

    items = list(values) if isinstance(values, (list, tuple)) else []
    return f"[{_plain_number(items[0])}, {_plain_number(items[1])}]" if len(items) == 2 else "[n/a]"


def _heldout_validation_lines(block: Mapping[str, object]) -> list[str]:
    """Terminal lines of the held-out validation, copied from HELDOUT_VALIDATION.json (heldout.terminal_block)."""

    lines = ["Held-out validation (free rows; the leak flags state any place they were used)"]
    preset = block.get("preset") if isinstance(block.get("preset"), Mapping) else {}
    if preset:
        lines.append(f"  Preset {preset.get('name')}: {' '.join(str(t) for t in preset.get('equivalent_flags') or [])}")
    if block.get("status") != "available":
        return lines + [f"  Not available: {block.get('reason') or block.get('status') or 'not recorded'}"]
    for row in block.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"  {row.get('crystal')}: R_obs {_plain_number(row.get('r_obs'))} {_plain_ci(row.get('r_obs_ci95'))} %, "
            f"R_all {_plain_number(row.get('r_all'))} {_plain_ci(row.get('r_all_ci95'))} % on {row.get('free_rows')} "
            f"free rows / {row.get('free_orbits')} orbits; in-sample R_obs {_plain_number(row.get('in_sample_r_obs'))}; "
            f"overfitting gap R_obs/R_all {_plain_number(row.get('gap_r_obs'), signed=True)} / "
            f"{_plain_number(row.get('gap_r_all'), signed=True)} pp"
        )
    leaks = [item for item in block.get("leaks") or [] if isinstance(item, Mapping)]
    if leaks:
        text = "; ".join(
            f"{item.get('label')} {'n/a' if item.get('value') is None else 'yes' if item.get('value') else 'no'}"
            for item in leaks
        )
        lines.append(f"  Leak flags (free rows used?): {text}")
    if block.get("shelxl_endpoint_free_line"):
        lines.append(str(block["shelxl_endpoint_free_line"]))
    loop = block.get("shelxl_loop") if isinstance(block.get("shelxl_loop"), Mapping) else {}
    if loop:
        last = loop.get("last") if isinstance(loop.get("last"), Mapping) else {}
        text = lambda value: "n/a" if value is None else str(value)  # noqa: E731  (unrecorded, as the console prints it)
        lines.append(
            f"  SHELXL loop: {text(loop.get('status'))}, stop {text(loop.get('stop_reason'))}, kept cycle "
            f"{text(loop.get('kept_cycle'))} of {text(loop.get('cycles'))}"
            + (
                f"; last cycle entering free R_obs {_plain_number(last.get('free_r_obs'))} %, work R_obs "
                f"{_plain_number(last.get('work_r_obs'))} %, max shift {_plain_number(last.get('max_xyz_shift_A'), 4)} A, "
                f"NPD {text(last.get('n_npd'))}, ISOR {text(last.get('n_isor'))}"
                if last
                else "; no per-cycle trajectory recorded"
            )
        )
    timings = block.get("phase_timings") if isinstance(block.get("phase_timings"), Mapping) else {}
    if timings:
        phases = ", ".join(
            f"{row.get('phase')} {_plain_number(row.get('wall_seconds'), 0)} s"
            for row in timings.get("phases") or []
            if isinstance(row, Mapping)
        )
        lines.append(f"  Phase wall time: {phases}; total {_plain_number(timings.get('total_wall_seconds'), 0)} s")
    if block.get("artifact"):
        lines.append(f"  Details: {block.get('artifact')}")
    return lines


def _format_terminal_finish_summary(payload: Mapping[str, object]) -> str:
    """Return the short human-facing completion report printed by auto/run/finish."""

    status = str(payload.get("status", "ok"))
    out_dir = str(payload.get("out_dir", ""))
    if status == "dry_run":
        return "\n".join(
            [
                "",
                "GROND dry run completed successfully.",
                f"Output directory: {out_dir}",
                "Commands were planned, but no scientific stages ran and no endpoint statistics were generated.",
            ]
        )
    final_hand = str(payload.get("final_hand", "n/a"))
    transform = str(payload.get("selected_transform", "n/a"))
    transform_parity = str(payload.get("selected_transform_parity", "n/a"))
    transform_determinant = payload.get("selected_transform_determinant", "per-dataset")
    configuration_status = str(
        payload.get("configuration_claim_status", "parity_unresolved")
    )
    configuration_claim_allowed = bool(
        payload.get("absolute_configuration_claim_allowed", False)
    )
    t_all = payload.get("final_hand_t_style_all", payload.get("final_hand_z_all"))
    t_free = payload.get("final_hand_t_style_free", payload.get("final_hand_z_free"))
    t_all_n = payload.get("final_hand_t_style_all_n")
    t_free_n = payload.get("final_hand_t_style_free_n")
    t_all_df = payload.get("final_hand_t_style_all_df")
    t_free_df = payload.get("final_hand_t_style_free_df")
    zkp_all = payload.get("final_hand_zkp_all")
    zkp_free = payload.get("final_hand_zkp_free")
    per_dataset_hand = (
        payload.get("final_hand_per_dataset")
        if isinstance(payload.get("final_hand_per_dataset"), Mapping)
        else {}
    )
    input_model = payload.get("input_model") if isinstance(payload.get("input_model"), Mapping) else {}
    input_shelxl = payload.get("input_shelxl") if isinstance(payload.get("input_shelxl"), Mapping) else {}
    final_row = payload.get("final_row_model") if isinstance(payload.get("final_row_model"), Mapping) else {}
    jana = (
        payload.get("jana_pets_row_statistics")
        if isinstance(payload.get("jana_pets_row_statistics"), Mapping)
        else {}
    )
    jana_dynamic = jana.get("dynamic") if isinstance(jana.get("dynamic"), Mapping) else {}
    jana_born = jana.get("born") if isinstance(jana.get("born"), Mapping) else {}
    if not _quick_complete_jana_metrics(jana_dynamic):
        jana_dynamic = {}
    if not _quick_complete_jana_metrics(jana_born):
        jana_born = {}
    feedback = payload.get("final_feedback_endpoint") if isinstance(payload.get("final_feedback_endpoint"), Mapping) else {}
    endpoint = payload.get("publication_endpoint") if isinstance(payload.get("publication_endpoint"), Mapping) else {}
    diagnostic = payload.get("diagnostic_endpoint") if isinstance(payload.get("diagnostic_endpoint"), Mapping) else {}
    endpoint_available = _quick_fraction(endpoint.get("R1all")) is not None
    endpoint_warning = _quick_goof_warning(endpoint.get("GooF")) if endpoint_available else ""
    files = payload.get("files") if isinstance(payload.get("files"), Mapping) else {}
    finished = payload.get("finished") if isinstance(payload.get("finished"), Mapping) else {}
    timing_context = payload.get("timing_context")
    timing_context = timing_context if isinstance(timing_context, Mapping) else {}
    timing_kind = str(timing_context.get("kind", "workflow_invocation"))
    finish_elapsed = (
        format_seconds(timing_context.get("finish_invocation_elapsed_seconds"))
        if timing_context.get("finish_invocation_elapsed_seconds") is not None
        else ""
    )
    invocation_elapsed = (
        str(finished.get("invocation_elapsed") or format_seconds(finished.get("invocation_elapsed_seconds")))
        if finished.get("invocation_elapsed_seconds") is not None
        else ""
    )
    cumulative_elapsed = str(
        finished.get("cumulative_stage_elapsed")
        or finished.get("stage_elapsed")
        or format_seconds(
            finished.get(
                "cumulative_stage_elapsed_seconds",
                finished.get("stage_elapsed_seconds", 0.0),
            )
        )
    )
    completed = finished.get("completed_stages", "n/a")
    estimated = finished.get("estimated_stages")
    stage_note = f"{completed} stages"
    if estimated not in {None, "", "n/a"}:
        try:
            if int(estimated) != int(completed):
                stage_note += f" completed; initial estimate was {estimated}"
            else:
                stage_note += " completed"
        except (TypeError, ValueError):
            stage_note += " completed"
    else:
        stage_note += " completed"
    if finished.get("failed_stages", 0):
        stage_note += f"; {finished.get('failed_stages')} failed stage records in progress file"
    if finished.get("completed_this_invocation") is not None:
        invocation_label = "recorded workflow invocation" if timing_kind == "finish_replay" else "this invocation"
        stage_note += f"; {finished.get('completed_this_invocation')} ran in {invocation_label}"
        if finished.get("reused_completed_stages", 0):
            prior_label = "prior workflow invocations" if timing_kind == "finish_replay" else "prior invocations"
            stage_note += f"; {finished.get('reused_completed_stages')} reused from {prior_label}"

    if timing_kind == "finish_replay":
        if status == "ok":
            headline = (
                f"GROND finish completed successfully in {finish_elapsed} wall time."
                if finish_elapsed
                else "GROND finish completed successfully."
            )
        else:
            headline = (
                f"GROND finish completed with status {status} after {finish_elapsed} wall time."
                if finish_elapsed
                else f"GROND finish completed with status {status}."
            )
    elif status == "ok":
        headline = (
            f"GROND finished successfully in {invocation_elapsed} wall time for this invocation."
            if invocation_elapsed
            else "GROND finished successfully."
        )
    else:
        headline = (
            f"GROND finished with status {status} after {invocation_elapsed} wall time for this invocation."
            if invocation_elapsed
            else f"GROND finished with status {status}."
        )
    if timing_kind == "finish_replay":
        recorded_wall = (
            f"recorded workflow wall time {invocation_elapsed}"
            if invocation_elapsed
            else "recorded workflow wall time unavailable"
        )
        timing_line = (
            f"Timing: {recorded_wall}; cumulative completed-stage runtime {cumulative_elapsed}. "
            "The cumulative value sums saved stage runtimes; concurrent stages overlap and resumed "
            "stages may be from earlier invocations, so it is not workflow wall time."
        )
    else:
        timing_line = (
            f"Timing: cumulative completed-stage runtime {cumulative_elapsed}. "
            "This sums saved stage runtimes; concurrent stages overlap and resumed stages may be from "
            "earlier invocations, so it is not workflow wall time."
        )
    publication_is_direct_ls1 = bool(files.get("publication_res") and files.get("publication_res") == files.get("final_feedback_res"))
    endpoint_lines = ["SHELXL endpoint" if publication_is_direct_ls1 else "SHELXL endpoints"]
    if not endpoint_available:
        endpoint_lines.extend(
            [
                "  No corrected endpoint passed the hard artifact, support, endpoint-health, and no-harm checks.",
                f"  Retain conventional baseline: R1(F)all {_quick_percent(input_shelxl.get('R1all'))}; RES {files.get('baseline_res', '')}; HKL {files.get('baseline_hkl', '')}",
            ]
        )
    elif publication_is_direct_ls1:
        endpoint_lines.append(
            f"  Recommended model-handoff endpoint (SHELXL feedback, corrected HKL): R1(F)gt {_quick_percent(endpoint.get('R1gt'))}, R1(F)all {_quick_percent(endpoint.get('R1all'))}, wR2(F^2)all {_quick_percent(endpoint.get('wR2all'))}, GooF {_quick_number(endpoint.get('GooF'), 3)}; {_quick_shelxl_issue_text(endpoint)}"
        )
    else:
        endpoint_lines.extend(
            [
                f"  SHELXL feedback corrected-HKL candidate: R1(F)gt {_quick_percent(feedback.get('R1gt'))}, R1(F)all {_quick_percent(feedback.get('R1all'))}, wR2(F^2)all {_quick_percent(feedback.get('wR2all'))}, GooF {_quick_number(feedback.get('GooF'), 3)}; {_quick_shelxl_issue_text(feedback)}",
                f"  Recommended model-handoff endpoint: R1(F)gt {_quick_percent(endpoint.get('R1gt'))}, R1(F)all {_quick_percent(endpoint.get('R1all'))}, wR2(F^2)all {_quick_percent(endpoint.get('wR2all'))}, GooF {_quick_number(endpoint.get('GooF'), 3)}; {_quick_shelxl_issue_text(endpoint)}",
            ]
        )
    if diagnostic and diagnostic.get("R1all") is not None:
        reasons = diagnostic.get("rejection_reasons")
        first_reason = str(reasons[0]) if isinstance(reasons, list) and reasons else "diagnostic-only"
        endpoint_lines.append(
            f"  Diagnostic endpoint not promoted: {diagnostic.get('source_kind', 'n/a')} R1(F)all {_quick_percent(diagnostic.get('R1all'))}, wR2(F^2)all {_quick_percent(diagnostic.get('wR2all'))}, GooF {_quick_number(diagnostic.get('GooF'), 3)}; reason: {first_reason}"
        )
    if endpoint_available:
        endpoint_lines.append(
            f"  Endpoint source/scales: {_quick_endpoint_scale_text(endpoint)}, WGHT {endpoint.get('wght_mode', 'n/a')}"
        )
    if endpoint_warning:
        endpoint_lines.append(f"  Warning: {endpoint_warning}")
    for warning in (endpoint.get("baseline_comparison") or {}).get("warnings", []):
        endpoint_lines.append(f"  Warning: {warning}")
    hand_scope = str(payload.get("final_hand_t_style_scope") or payload.get("final_hand_z_scope") or "")
    if hand_scope == "per_dataset":
        evidence = "; ".join(
            f"{name}=ZKP {_quick_number((row or {}).get('zkp_all'), 3)}/{_quick_number((row or {}).get('zkp_free'), 3)}; "
            f"t {_quick_number((row or {}).get('t_all'), 3)}/{_quick_number((row or {}).get('t_free'), 3)} "
            f"(df {(row or {}).get('df_all', 'n/a')}/{(row or {}).get('df_free', 'n/a')})"
            for name, row in per_dataset_hand.items()
            if isinstance(row, Mapping)
        )
        hand_line = (
            f"  Hand: {final_hand} (per-dataset paired t-style all/free: "
            f"{evidence or 'diagnostic missing'}; no pooled statistic; "
            f"consistency {payload.get('final_hand_consistency_status') or 'missing'})"
        )
    else:
        hand_line = (
            f"  Hand: {final_hand} (adjusted ZKP all/free = "
            f"{_quick_number(zkp_all, 3)} / {_quick_number(zkp_free, 3)}; "
            "positive favors the selected hand; paired t-style all/free = "
            f"{_quick_flat_hand_t_style_text(t_all, t_all_n, t_all_df)} / "
            f"{_quick_flat_hand_t_style_text(t_free, t_free_n, t_free_df)}; "
            "t-style is not a calibrated normal Z score)"
        )
    heldout_first = str(payload.get("report_layout") or "legacy") == "heldout-first"
    authority = str(payload.get("hand_decision_authority") or "")
    heldout_rule = authority == "heldout_orbit_sign"
    heldout_block = payload.get("heldout_validation") if isinstance(payload.get("heldout_validation"), Mapping) else {}
    if heldout_first:
        # The held-out hand evidence leads; the all-row ZKP line stays as a labelled diagnostic.
        sentences = [str(item) for item in heldout_block.get("hand_summary") or []]
        hand_line = "\n".join(
            [
                f"  Hand: {final_hand} ("
                + (
                    "decided by the pooled held-out orbit-sign rule"
                    if heldout_rule
                    else "decided by the legacy all-row ZKP rule"
                    if authority == "diagnostic_only"
                    else "no hand decision was evaluated; the hand check was skipped"
                    if authority == "not_evaluated"
                    else "decision rule not recorded in decisions/hand_decision_heldout_*.json"
                )
                + ")",
                *[f"    {text}" for text in sentences],
                *([] if sentences else ["    held-out hand evidence: not recorded"]),
                (
                    "    Diagnostic only, not the decision rule: "
                    if heldout_rule
                    else "    Decision statistic: "
                    if authority == "diagnostic_only"
                    else "    All-row statistics: "
                )
                + hand_line.strip().removeprefix(f"Hand: {final_hand} ").strip("()"),
            ]
        )
    if payload.get("final_hand_preferred") is not None:
        hand_line += (
            f"\n  {'Held-out' if heldout_rule else 'All-row'} preferred hand: {payload['final_hand_preferred']}; "
            f"evidence {payload.get('final_hand_confidence', 'unavailable')}; "
            f"exported representation {final_hand}. "
            + ("Orbit sign z is" if heldout_rule else "ZKP is")
            + " not calibrated sigma confidence."
        )
    hand_warnings = [str(item) for item in payload.get("final_hand_warnings", [])]
    if hand_warnings:
        hand_line += "\n  Hand warning: " + "; ".join(hand_warnings)
    if jana_dynamic or jana_born:
        row_stat_lines = [
            "Final JANA/PETS-style row statistics (computed by GROND)",
            (
                "  Matched Born: "
                f"Nobs/Nall {_quick_count_pair(jana_born)}, "
                f"Robs {_quick_percent(jana_born.get('Robs'))}, Rall {_quick_percent(jana_born.get('Rall'))}, "
                f"wRall {_quick_percent(jana_born.get('wRall'))}; "
                f"MRobs {_quick_percent(jana_born.get('MRobs'))}, MRall {_quick_percent(jana_born.get('MRall'))}, "
                f"MwRall {_quick_percent(jana_born.get('MwRall'))}"
            ),
            (
                "  Bloch dynamical: "
                f"Nobs/Nall {_quick_count_pair(jana_dynamic)}, "
                f"Robs {_quick_percent(jana_dynamic.get('Robs'))}, Rall {_quick_percent(jana_dynamic.get('Rall'))}, "
                f"wRall {_quick_percent(jana_dynamic.get('wRall'))}; "
                f"MRobs {_quick_percent(jana_dynamic.get('MRobs'))}, MRall {_quick_percent(jana_dynamic.get('MRall'))}, "
                f"MwRall {_quick_percent(jana_dynamic.get('MwRall'))}"
            ),
        ]
    else:
        row_stat_lines = [
            "Legacy native-objective row diagnostics (not JANA/PETS publication factors)",
            f"  Input dynamical rows: R(I)all {_quick_percent(input_model.get('row_R1all'))}, wR(I)all {_quick_percent(input_model.get('row_wR2all'))}",
            f"  Input matched Born rows: R(I)all {_quick_percent(input_model.get('born_R1all'))}, wR(I)all {_quick_percent(input_model.get('born_wR2all'))}",
            f"  Final dynamical rows: R(I)all {_quick_percent(final_row.get('row_R1all'))}, wR(I)all {_quick_percent(final_row.get('row_wR2all'))}, free wR(I)all {_quick_percent(final_row.get('row_free_wR2all'))}",
            f"  Final matched Born rows: R(I)all {_quick_percent(final_row.get('born_R1all'))}, wR(I)all {_quick_percent(final_row.get('born_wR2all'))}, free wR(I)all {_quick_percent(final_row.get('born_free_wR2all'))}",
            f"  Change: R(I)all {_quick_delta_percent(input_model.get('row_R1all'), final_row.get('row_R1all'))}, wR(I)all {_quick_delta_percent(input_model.get('row_wR2all'), final_row.get('row_wR2all'))}",
        ]
    lines = [
        "",
        headline,
        f"Output directory: {out_dir}",
        f"Progress: {stage_note}. Optional gates can make the final stage count smaller than the starting ETA denominator.",
        timing_line,
        "",
        *([*_heldout_validation_lines(heldout_block), ""] if heldout_first else []),
        "Decision summary",
        hand_line,
        f"  HKL transform: {transform}; parity {transform_parity}; determinant {transform_determinant}",
        (
            f"  Configuration claim: {configuration_status}; named absolute configuration "
            f"allowed = {configuration_claim_allowed}"
        ),
        "",
        "Starting SHELXL reference",
        f"  R1(F)gt {_quick_percent(input_shelxl.get('R1gt'))}, R1(F)all {_quick_percent(input_shelxl.get('R1all'))}, wR2(F^2)all {_quick_percent(input_shelxl.get('wR2all'))}, GooF {_quick_number(input_shelxl.get('GooF'), 3)}",
        f"  Parameters/restraints {input_shelxl.get('parameters', 'n/a')} / {input_shelxl.get('restraints', 'n/a')}; NPD {input_shelxl.get('npd_count', 'n/a')}; split {input_shelxl.get('split_count', 'n/a')}",
        "",
        *row_stat_lines,
        "",
        *endpoint_lines,
        # the held-out SHELXL scores print once: in the held-out block (heldout-first) or here (legacy layout)
        *([str(heldout_block["shelxl_endpoint_free_line"])]
          if heldout_block.get("shelxl_endpoint_free_line") and not heldout_first else []),
        "",
        "Open these first",
        f"  Canonical Table 1: {payload.get('table1', '')}",
        f"  Table 1 JSON: {payload.get('table1_json', '')}",
        f"  Final summary: {payload.get('final_summary', '')}",
        f"  Quick stats: {payload.get('quick_stats', '')}",
        f"  Full JSON: {payload.get('full_json', '')}",
        *([f"  Final model (all reflections, --free-set off): {files.get('all_data_res', '')}",
           f"  Final HKL (work and free reflections): {files.get('all_data_hkl', '')}",
           f"  Endpoint chosen with the free set held out: {files.get('publication_res', '')}"]
          if files.get("all_data_res") else
          [f"  Publication model: {files.get('publication_res', '')}",
           f"  Publication HKL: {files.get('publication_hkl', '')}"]),
        f"  Row table(s): {final_row.get('row_table', '')}",
    ]
    return "\n".join(lines)
