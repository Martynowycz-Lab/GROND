"""Render matched hand evidence, its Born control and any interpretation warnings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.reports.values import _quick_float, _quick_int, _quick_number


def _quick_hand_block_t_style(block: Mapping[str, object], *, final_is_other: bool) -> dict[str, object]:
    """Orient a paired row-level t-style contrast so positive favors the final hand."""

    sign = -1.0 if final_is_other else 1.0
    n = _quick_int(block.get("n"))
    df = n - 1 if n is not None and n >= 1 else None
    raw_delta = _quick_float(block.get("delta_chi2_other_minus_best"))
    raw_t = _quick_float(block.get("paired_row_t_delta_chi2"))
    oriented_delta = sign * raw_delta if math.isfinite(raw_delta) else float("nan")
    oriented_t = sign * raw_t if math.isfinite(raw_t) else float("nan")
    adjusted = block.get("klar_palatinus_adjusted_hand_z")
    adjusted = adjusted if isinstance(adjusted, Mapping) else {}
    raw_zkp = _quick_float(adjusted.get("z"))
    oriented_zkp = sign * raw_zkp if math.isfinite(raw_zkp) else float("nan")
    return {
        "n": n,
        "df": df,
        "delta_chi2_rejected_minus_final": oriented_delta,
        "paired_t_style_positive_favors_final": oriented_t,
        "adjusted_zkp_positive_favors_final": oriented_zkp,
        "paired_t_style_definition": (
            "paired Student-t-style contrast of per-row delta_chi2; positive favors the final selected hand; "
            "not a calibrated normal Z score"
        ),
        "sqrt_positive_delta_chi2": math.sqrt(max(oriented_delta, 0.0)) if math.isfinite(oriented_delta) else float("nan"),
    }


def _quick_hand_t_style_text(block: Mapping[str, object]) -> str:
    """Format one paired hand contrast with its sample count and degrees of freedom."""

    return (
        f"t {_quick_number(block.get('paired_t_style_positive_favors_final'), 3)} "
        f"(n {block.get('n', 'n/a')}, df {block.get('df', 'n/a')})"
    )


def _quick_final_hand_summary(decision_tree: Mapping[str, object], diagnostic: Mapping[str, object]) -> dict[str, object]:
    """Summarize initial and final hand decisions from the decision-tree artifacts."""

    initial = decision_tree.get("absolute_structure_decision")
    initial_decision = initial if isinstance(initial, Mapping) else {}
    promotion = decision_tree.get("post_feedback_hand_promotion")
    promotion_decision = promotion if isinstance(promotion, Mapping) else {}
    final_hand = str(decision_tree.get("selected_hand") or initial_decision.get("selected_hand") or initial_decision.get("selected") or "n/a")
    initial_hand = str(initial_decision.get("selected_hand") or initial_decision.get("selected") or "n/a")
    final_is_other = final_hand == str(promotion_decision.get("opposite_hand"))
    blocks: dict[str, dict[str, object]] = {}
    for split in ("all", "work", "free"):
        block = diagnostic.get(split)
        if isinstance(block, Mapping):
            blocks[split] = _quick_hand_block_t_style(block, final_is_other=final_is_other)
    return {
        "initial_hand": initial_hand,
        "final_hand": final_hand,
        "promotion_status": promotion_decision.get("status", "n/a") if promotion_decision else "n/a",
        "initial_score_delta_original_minus_alternate": initial_decision.get("score_delta_original_minus_alternate"),
        "post_feedback_score_delta_final_minus_opposite": promotion_decision.get("score_delta_final_minus_opposite"),
        "diagnostic_blocks": blocks,
    }


def _quick_multi_hand_diagnostic(result: Mapping[str, object]) -> dict[str, object]:
    """Load the shared post-feedback multi-crystal hand audit."""

    value = result.get("post_feedback_multi_hand_diagnostic")
    if isinstance(value, Mapping):
        return dict(value)
    if value:
        return try_read_json(Path(str(value)))
    path = result.get("post_feedback_multi_hand_diagnostic_path")
    return try_read_json(Path(str(path))) if path else {}


def _quick_final_elastic_hand_aggregate(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Orient the final elastic pooled hand contrast so positive favors the retained model."""

    if str(payload.get("method") or "") != "nakane_negative_thickness_elastic_pair_v1":
        return {}
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), Mapping) else {}
    splits = aggregate.get("splits") if isinstance(aggregate.get("splits"), Mapping) else {}
    sign = -1.0 if bool(payload.get("hand_switched", False)) else 1.0
    oriented: dict[str, dict[str, object]] = {}
    for split in ("all", "work", "free"):
        block = splits.get(split) if isinstance(splits.get(split), Mapping) else {}
        raw_t = _quick_float(block.get("paired_row_t_delta_chi2"))
        n = _quick_int(block.get("n"))
        oriented[split] = {
            "t": sign * raw_t if math.isfinite(raw_t) else None,
            "n": n,
            "df": n - 1 if n is not None and n > 0 else None,
            "mean_delta_chi2_rejected_minus_final": (
                sign * _quick_float(block.get("mean_delta_chi2_inverse_minus_selected"))
                if math.isfinite(_quick_float(block.get("mean_delta_chi2_inverse_minus_selected")))
                else None
            ),
        }
    return oriented


def _quick_final_elastic_hand_zkp(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Return combined adjusted ZKP values already oriented to the final hand."""

    if str(payload.get("method") or "") != "nakane_negative_thickness_elastic_pair_v1":
        return {}
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), Mapping) else {}
    adjusted = (
        aggregate.get("adjusted_z_final_hand_favored")
        if isinstance(aggregate.get("adjusted_z_final_hand_favored"), Mapping)
        else {}
    )
    result: dict[str, dict[str, object]] = {}
    for split in ("all", "work", "free"):
        block = adjusted.get(split) if isinstance(adjusted.get(split), Mapping) else {}
        result[split] = {
            "z": _quick_float(block.get("z")) if block else None,
            "n": _quick_int(block.get("n")) if block else None,
            "datasets": _quick_int(block.get("datasets")) if block else None,
        }
    return result


def _quick_multi_hand_console_rows(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Return compact per-dataset hand evidence for console/report consumers."""

    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), Mapping) else {}
    rows: dict[str, dict[str, object]] = {}
    final_elastic = str(payload.get("method") or "") == "nakane_negative_thickness_elastic_pair_v1"
    sign = -1.0 if bool(payload.get("hand_switched", False)) else 1.0
    for name, raw in sorted(datasets.items(), key=lambda item: str(item[0])):
        record = raw if isinstance(raw, Mapping) else {}
        paired = (
            record.get("paired_t_style_positive_favors_selected")
            if isinstance(record.get("paired_t_style_positive_favors_selected"), Mapping)
            else {}
        )
        paired_compatibility = (
            record.get("paired_t_style_z_positive_favors_selected")
            if isinstance(record.get("paired_t_style_z_positive_favors_selected"), Mapping)
            else {}
        )
        counts = record.get("row_counts") if isinstance(record.get("row_counts"), Mapping) else {}
        margins = (
            record.get("score_margins_positive_favors_selected")
            if isinstance(record.get("score_margins_positive_favors_selected"), Mapping)
            else {}
        )
        diagnostic = try_read_json(Path(str(record.get("diagnostic") or ""))) if final_elastic and record.get("diagnostic") else {}
        diagnostic_blocks = {
            split: diagnostic.get(split) if isinstance(diagnostic.get(split), Mapping) else {}
            for split in ("all", "work", "free")
        }
        if final_elastic:
            counts = {split: _quick_int(block.get("n")) or 0 for split, block in diagnostic_blocks.items()}
            paired = {
                split: {
                    "t": (
                        sign * _quick_float(block.get("paired_row_t_delta_chi2"))
                        if math.isfinite(_quick_float(block.get("paired_row_t_delta_chi2")))
                        else None
                    ),
                    "df": (int(block.get("n")) - 1 if int(block.get("n", 0) or 0) > 0 else None),
                }
                for split, block in diagnostic_blocks.items()
            }
            adjusted = {
                split: (
                    diagnostic_blocks[split].get("klar_palatinus_adjusted_hand_z")
                    if isinstance(
                        diagnostic_blocks[split].get("klar_palatinus_adjusted_hand_z"),
                        Mapping,
                    )
                    else {}
                )
                for split in ("all", "work", "free")
            }
            selected_minus_inverse = _quick_float(record.get("score_delta_selected_minus_inverse"))
            final_margin = sign * (-selected_minus_inverse) if math.isfinite(selected_minus_inverse) else None
            margins = {}
            selected_row_table = record.get("selected_row_table", "")
            opposite_row_table = record.get("inverse_row_table", "")
        else:
            final_margin = record.get(
                "composite_score_margin_opposite_minus_selected_positive_favors_selected"
            )
            selected_row_table = record.get("selected_row_table", "")
            opposite_row_table = record.get("opposite_row_table", "")
        rows[str(name)] = {
            "status": record.get("status", "missing"),
            "rows_all": counts.get("all", 0),
            "rows_work": counts.get("work", 0),
            "rows_free": counts.get("free", 0),
            "t_all": (paired.get("all") or {}).get("t") if isinstance(paired.get("all"), Mapping) else paired_compatibility.get("all"),
            "t_work": (paired.get("work") or {}).get("t") if isinstance(paired.get("work"), Mapping) else paired_compatibility.get("work"),
            "t_free": (paired.get("free") or {}).get("t") if isinstance(paired.get("free"), Mapping) else paired_compatibility.get("free"),
            "zkp_all": (
                sign * _quick_float((adjusted.get("all") or {}).get("z"))
                if final_elastic and math.isfinite(_quick_float((adjusted.get("all") or {}).get("z")))
                else None
            ),
            "zkp_work": (
                sign * _quick_float((adjusted.get("work") or {}).get("z"))
                if final_elastic and math.isfinite(_quick_float((adjusted.get("work") or {}).get("z")))
                else None
            ),
            "zkp_free": (
                sign * _quick_float((adjusted.get("free") or {}).get("z"))
                if final_elastic and math.isfinite(_quick_float((adjusted.get("free") or {}).get("z")))
                else None
            ),
            "df_all": (paired.get("all") or {}).get("df") if isinstance(paired.get("all"), Mapping) else None,
            "df_work": (paired.get("work") or {}).get("df") if isinstance(paired.get("work"), Mapping) else None,
            "df_free": (paired.get("free") or {}).get("df") if isinstance(paired.get("free"), Mapping) else None,
            "chi2_margin_all": (
                (margins.get("all") or {}).get("delta_chi2_opposite_minus_selected")
                if isinstance(margins.get("all"), Mapping)
                else None
            ),
            "chi2_margin_work": (
                (margins.get("work") or {}).get("delta_chi2_opposite_minus_selected")
                if isinstance(margins.get("work"), Mapping)
                else None
            ),
            "chi2_margin_free": (
                (margins.get("free") or {}).get("delta_chi2_opposite_minus_selected")
                if isinstance(margins.get("free"), Mapping)
                else None
            ),
            "composite_margin": final_margin,
            "selected_row_table": selected_row_table,
            "opposite_row_table": opposite_row_table,
            "diagnostic": record.get("diagnostic", ""),
        }
    return rows


def _quick_multi_hand_evidence_lines(
    payload: Mapping[str, object],
    *,
    compact: bool = False,
) -> list[str]:
    """Render final exact-row hand evidence for old and final-elastic schemas."""

    rows = _quick_multi_hand_console_rows(payload)
    pooled = _quick_final_elastic_hand_aggregate(payload)
    pooled_zkp = _quick_final_elastic_hand_zkp(payload)
    final_elastic = bool(pooled)
    lines = [
        "## Final Hand Evidence",
        "",
        (
            "The final elastic negative-thickness comparison uses each crystal's exact frozen rows, nuisance, scales, "
            "N/Q, jitter, and proper transform, with absorption disabled only for the Hermitian hand identity. "
            "The combined all-row Klar--Palatinus adjusted Z is the hand-selection statistic; held-out Z is a "
            "warning diagnostic and each crystal remains visible separately. Displayed ZKP and t-style values are "
            "oriented so positive favors the final retained hand; the t-style values are not normal Z scores."
            if final_elastic
            else
            "Each row below compares one crystal's frozen final row table with an opposite-hand recomputation at "
            "the identical pinned nuisance and N/Q. Positive paired t-style statistics and score margins favor the "
            "selected shared hand. Each t-style statistic is reported with n and df=n-1; it is not a calibrated "
            "normal Z score. No pooled statistic is computed for this legacy diagnostic schema."
        ),
        "",
    ]
    if final_elastic:
        lines.extend(
            [
                f"Selection status: `{payload.get('status', 'missing')}`; hand switched: `{bool(payload.get('hand_switched', False))}`.",
                "",
                "| pooled adjusted ZKP | ZKP favoring final hand | n | datasets |",
                "|---|---:|---:|---:|",
                *[
                    f"| {split} | {_quick_number((pooled_zkp.get(split) or {}).get('z'), 3)} | "
                    f"{(pooled_zkp.get(split) or {}).get('n', 'n/a')} | "
                    f"{(pooled_zkp.get(split) or {}).get('datasets', 'n/a')} |"
                    for split in ("all", "work", "free")
                ],
                "",
                "| pooled exact rows | paired t-style | n | df |",
                "|---|---:|---:|---:|",
                *[
                    f"| {split} | {_quick_number((pooled.get(split) or {}).get('t'), 3)} | "
                    f"{(pooled.get(split) or {}).get('n', 'n/a')} | {(pooled.get(split) or {}).get('df', 'n/a')} |"
                    for split in ("all", "work", "free")
                ],
                "",
            ]
        )
    if not rows:
        lines.extend(
            [
                f"Consistency gate: `{payload.get('status', 'warning_missing_diagnostic')}`. No per-dataset "
                "diagnostic rows were recorded; treat final hand evidence as incomplete.",
                "",
            ]
        )
        return lines
    if compact:
        lines.extend(
            [
                f"Consistency gate: `{payload.get('status', 'missing')}`.",
                "",
                "| dataset | status | rows all/work/free | ZKP all/work/free | paired t-style all/work/free | df all/work/free |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for name, row in rows.items():
            lines.append(
                f"| {name} | {row.get('status', 'missing')} | "
                f"{row.get('rows_all', 0)}/{row.get('rows_work', 0)}/{row.get('rows_free', 0)} | "
                f"{_quick_number(row.get('zkp_all'), 3)}/{_quick_number(row.get('zkp_work'), 3)}/"
                f"{_quick_number(row.get('zkp_free'), 3)} | "
                f"{_quick_number(row.get('t_all'), 3)}/{_quick_number(row.get('t_work'), 3)}/"
                f"{_quick_number(row.get('t_free'), 3)} | "
                f"{row.get('df_all', 'n/a')}/{row.get('df_work', 'n/a')}/{row.get('df_free', 'n/a')} |"
            )
    else:
        lines.extend(
            [
                f"Consistency gate: `{payload.get('status', 'missing')}`; artifact `{payload.get('artifact', '')}`.",
                "",
                "| dataset | status | rows all/work/free | ZKP all/work/free | paired t-style all/work/free | df all/work/free | chi2 margin all/work/free | composite score margin | selected/opposite row tables |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for name, row in rows.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        str(row.get("status", "missing")),
                        f"{row.get('rows_all', 0)}/{row.get('rows_work', 0)}/{row.get('rows_free', 0)}",
                        f"{_quick_number(row.get('zkp_all'), 3)}/{_quick_number(row.get('zkp_work'), 3)}/{_quick_number(row.get('zkp_free'), 3)}",
                        f"{_quick_number(row.get('t_all'), 3)}/{_quick_number(row.get('t_work'), 3)}/{_quick_number(row.get('t_free'), 3)}",
                        f"{row.get('df_all', 'n/a')}/{row.get('df_work', 'n/a')}/{row.get('df_free', 'n/a')}",
                        f"{_quick_number(row.get('chi2_margin_all'), 3)}/{_quick_number(row.get('chi2_margin_work'), 3)}/{_quick_number(row.get('chi2_margin_free'), 3)}",
                        _quick_number(row.get("composite_margin"), 4),
                        f"`{row.get('selected_row_table', '')}` / `{row.get('opposite_row_table', '')}`",
                    ]
                )
                + " |"
            )
    lines.append("")
    return lines
