"""Recheck both hands after shared-model feedback without changing the row contract."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import try_read_json
from grond.io.config import PrepConfig
from grond.io.structure_contract import is_shelx_model, make_alternate_hand_model
from grond.paths import engine_script
from grond.workflows.decision import safe_float, safe_label, score_native_summary, write_json
from grond.workflows.hand_bindings import (
    _native_physics_config_contract,
    _policy_selected_table_binding,
    _summary_hkl_binding,
    _summary_model_binding,
    _summary_native_row_table_binding,
)
from grond.workflows.hand_row_contracts import (
    _args_with_recorded_native_contract,
    _native_split_dyn_wr2,
    _paired_row_split_contract,
    _pinned_nuisance_mismatches,
    _positive_summary_int,
    _row_contract_nullable_numeric_fields,
)
from grond.workflows.hand_statistics import hand_diagnostic_supports_other
from grond.workflows.model_selection import dataset_baseline_hkl
from grond.workflows.native_commands import native_stage_cmd


def run_multi_post_feedback_hand_diagnostic(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    final_model: str,
    selected_hand: str,
    final_runs: Mapping[str, Path],
    publication_runs: Mapping[str, Path],
    shared_hand_decision: Mapping[str, object],
) -> dict[str, object]:
    """Audit the frozen shared hand independently on each publication crystal.

    This is deliberately diagnostic-only.  It inverts the final shared model,
    recomputes every publication-eligible crystal with that crystal's exact
    recorded final nuisance and N/Q contract, and compares matched row tables.
    The initial shared hand gate remains authoritative; this function can only
    pass the consistency audit or emit a fail-closed warning.
    """

    artifact = root / "decisions" / "shared_post_feedback_hand_diagnostic.json"
    opposite_hand = "alternate" if str(selected_hand) == "original" else "original"
    eligible = [ds.name for ds in prep.datasets if ds.name in publication_runs]
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "not_run",
        "selection_scope": "shared_multi_dataset_model",
        "selected_hand_authoritative": str(selected_hand),
        "opposite_hand_diagnostic": opposite_hand,
        "selected_model": str(final_model),
        "shared_hand_decision": str(root / "decisions" / "shared_absolute_structure_decision.json"),
        "artifact": str(artifact),
        "publication_eligible_datasets": eligible,
        "selected_row_table_policy": (
            "use each policy-active publication_runs/native_dyn_calc.csv table exactly; policy annotation columns are "
            "retained, while the paired residual diagnostic uses dyn_residual_over_sigma and ignores raw_row_weight"
        ),
        "datasets": {},
        "merged_t_style_statistic": None,
        "merged_t_style_policy": (
            "not computed: crystals retain independent row, sigma, scale, and free-set contracts"
        ),
        "merged_z": None,
        "merged_z_policy": (
            "compatibility alias only; the paired Student-t-style contrast is not a calibrated normal Z score, "
            "and no pooled statistic is computed"
        ),
        "decision_rule": (
            "diagnostic only; retain the initial shared hand after final refinement and warn if a "
            "per-dataset diagnostic is incomplete or materially favors the opposite hand"
        ),
        "hand_switched": False,
    }
    symmetry_policy = (
        shared_hand_decision.get("symmetry_policy")
        if isinstance(shared_hand_decision.get("symmetry_policy"), Mapping)
        else {}
    )
    if symmetry_policy and not bool(symmetry_policy.get("run_hand_control", True)):
        payload.update(
            {
                "status": "skipped_not_applicable",
                "consistency_gate": {
                    "status": "not_applicable",
                    "passed": None,
                    "warnings": [str(symmetry_policy.get("reason") or "hand control is not applicable")],
                },
            }
        )
        write_json(artifact, payload)
        return payload
    if not bool(getattr(args, "hand_control", True)):
        payload.update(
            {
                "status": "skipped_disabled",
                "consistency_gate": {
                    "status": "not_applicable",
                    "passed": None,
                    "warnings": ["absolute-structure hand control was disabled"],
                },
            }
        )
        write_json(artifact, payload)
        return payload
    if bool(getattr(args, "dry_run", False)):
        payload.update(
            {
                "status": "skipped_dry_run",
                "consistency_gate": {
                    "status": "not_run",
                    "passed": None,
                    "warnings": ["dry-run mode does not materialize final row diagnostics"],
                },
            }
        )
        write_json(artifact, payload)
        return payload

    datasets_by_name = {ds.name: ds for ds in prep.datasets}
    records: dict[str, dict[str, object]] = {}
    runnable: dict[str, dict[str, object]] = {}
    for name in eligible:
        ds = datasets_by_name[name]
        selected_run = Path(publication_runs[name])
        source_run = Path(final_runs.get(name, selected_run))
        selected_table = selected_run / "native_dyn_calc.csv"
        source_table = source_run / "native_dyn_calc.csv"
        source_summary_path = source_run / "summary.json"
        summary = try_read_json(source_summary_path)
        config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
        nuisance = (
            (summary.get("best_native") or {}).get("nuisance")
            if isinstance(summary.get("best_native"), Mapping)
            else {}
        )
        nuisance = dict(nuisance) if isinstance(nuisance, Mapping) else {}
        n_basis = _positive_summary_int(config, "n_basis")
        n_points = _positive_summary_int(config, "n_points")
        scale_fit_split = str(config.get("scale_fit_split") or "")
        selected_model_binding = _summary_model_binding(
            summary,
            Path(final_model),
            semantic_hand=str(selected_hand),
        )
        selected_hkl_binding = _summary_hkl_binding(summary, Path(ds.hkl))
        source_row_table_binding = _summary_native_row_table_binding(summary, source_table)
        policy_selected_table_binding = _policy_selected_table_binding(source_table, selected_table)
        missing: list[str] = []
        if not selected_table.is_file():
            missing.append(f"selected row table is missing: {selected_table}")
        if not source_summary_path.is_file():
            missing.append(f"source summary is missing: {source_summary_path}")
        if n_basis is None:
            missing.append("final summary does not record a positive n_basis")
        if n_points is None:
            missing.append("final summary does not record a positive n_points")
        if not nuisance:
            missing.append("final summary does not record the selected nuisance contract")
        if scale_fit_split not in {"work", "all"}:
            missing.append("final summary does not record a valid scale_fit_split")
        missing.extend(f"selected-model provenance: {error}" for error in selected_model_binding.get("errors", []))
        missing.extend(f"selected-HKL provenance: {error}" for error in selected_hkl_binding.get("errors", []))
        missing.extend(f"source-row provenance: {error}" for error in source_row_table_binding.get("errors", []))
        missing.extend(
            f"policy-selected row table: {error}" for error in policy_selected_table_binding.get("errors", [])
        )
        record: dict[str, object] = {
            "dataset": name,
            "status": "warning_incomplete" if missing else "ready",
            "selected_hand": str(selected_hand),
            "opposite_hand": opposite_hand,
            "selected_run": str(selected_run),
            "selected_source_run": str(source_run),
            "selected_source_row_table": str(source_table),
            "selected_row_table": str(selected_table),
            "selected_summary": str(source_summary_path),
            "comparison_contract": {
                "n_basis": n_basis,
                "n_points": n_points,
                "nuisance": nuisance,
                "scale_fit_split": scale_fit_split or None,
                "sg_window": config.get("sg_window"),
                "sg_window_mode": config.get("sg_window_mode"),
                "max_rows": config.get("max_rows"),
                "same_final_shared_model": bool(selected_model_binding.get("passed", False)),
                "selected_model_binding": selected_model_binding,
                "selected_hkl_binding": selected_hkl_binding,
                "source_row_table_binding": source_row_table_binding,
                "policy_selected_table_binding": policy_selected_table_binding,
                "endpoint_refinement": False,
                "maps": False,
            },
            "selected_row_table_choice": (
                "publication_runs policy-active frozen row table; it may carry policy annotations, but the paired "
                "diagnostic consumes dyn_residual_over_sigma and does not merge or apply raw_row_weight"
            ),
            "warnings": missing,
        }
        records[name] = record
        if not missing:
            runnable[name] = {
                "dataset": ds,
                "summary": summary,
                "config": config,
                "nuisance": nuisance,
                "n_basis": n_basis,
                "n_points": n_points,
                "scale_fit_split": scale_fit_split,
                "selected_table": selected_table,
            }

    if not eligible:
        payload.update(
            {
                "status": "warning_incomplete",
                "datasets": records,
                "consistency_gate": {
                    "status": "warning",
                    "passed": False,
                    "warnings": ["no publication-eligible crystal was available for a final hand diagnostic"],
                },
            }
        )
        write_json(artifact, payload)
        return payload

    opposite_model: Path | None = None
    materialization_error = ""
    if runnable:
        if not is_shelx_model(final_model):
            materialization_error = "the final shared model is not a materializable SHELX model"
        else:
            try:
                opposite_model = make_alternate_hand_model(python, final_model, root, logs, False)
            except Exception as exc:
                materialization_error = f"opposite-hand model materialization failed: {type(exc).__name__}: {exc}"
    payload["opposite_model"] = str(opposite_model or "")

    for name, context in runnable.items():
        record = records[name]
        if opposite_model is None:
            record["status"] = "warning_incomplete"
            record["warnings"] = [materialization_error or "opposite-hand model is unavailable"]
            continue
        ds = context["dataset"]
        config = context["config"]
        opposite_run = root / "runs" / str(name) / "multi_post_feedback_opposite_hand"
        diagnostic_path = root / "decisions" / "post_feedback_multi_hand" / f"{safe_label(str(name))}.json"
        record.update(
            {
                "opposite_model": str(opposite_model),
                "opposite_run": str(opposite_run),
                "opposite_row_table": str(opposite_run / "native_dyn_calc.csv"),
                "diagnostic": str(diagnostic_path),
            }
        )
        try:
            sg_window = safe_float(config.get("sg_window"), float("nan"))
            sg_window_override = sg_window if math.isfinite(sg_window) else None
            diagnostic_args = _args_with_recorded_native_contract(args, config)
            max_rows = config.get("max_rows")
            try:
                max_rows_override = int(max_rows) if max_rows not in {None, ""} else None
            except (TypeError, ValueError, OverflowError):
                max_rows_override = None
            if max_rows_override is not None and max_rows_override <= 0:
                max_rows_override = None
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    str(opposite_model),
                    dataset_baseline_hkl(prep, ds),
                    opposite_run,
                    diagnostic_args,
                    n_basis=int(context["n_basis"]),
                    n_points=int(context["n_points"]),
                    stage="multi_post_feedback_opposite_hand_final",
                    pinned=dict(context["nuisance"]),
                    make_maps=False,
                    endpoint=False,
                    max_rows_override=max_rows_override,
                    sg_window_override=sg_window_override,
                    sg_window_mode_override=str(config.get("sg_window_mode") or "") or None,
                    scale_fit_split_override=str(context["scale_fit_split"]),
                    strict_pinned_contract=True,
                ),
                logs,
                f"{name}_multi_post_feedback_opposite_hand",
                dry_run=False,
            )
            run_logged(
                [
                    python,
                    str(engine_script("absolute_structure_diagnostic.py")),
                    "--best",
                    str(context["selected_table"]),
                    "--other",
                    str(opposite_run / "native_dyn_calc.csv"),
                    "--out",
                    str(diagnostic_path),
                ],
                logs,
                f"{name}_multi_post_feedback_absolute_structure_diagnostic",
                dry_run=False,
            )
            stats = try_read_json(diagnostic_path)
            opposite_summary = try_read_json(opposite_run / "summary.json")
            row_pair_contract = _paired_row_split_contract(
                Path(context["selected_table"]),
                opposite_run / "native_dyn_calc.csv",
                nullable_numeric_fields=_row_contract_nullable_numeric_fields(
                    context["config"],
                    context["nuisance"],
                ),
            )
            row_join = stats.get("row_join") if isinstance(stats.get("row_join"), Mapping) else {}
            row_counts = {
                "selected": row_join.get("n_best", 0),
                "opposite": row_join.get("n_other", 0),
                "joined": row_join.get("n_joined", 0),
            }
            paired_t_style: dict[str, object] = {}
            paired_z_compatibility: dict[str, object] = {}
            score_margins: dict[str, object] = {}
            incomplete_reasons: list[str] = [
                f"row-pair contract: {error}" for error in row_pair_contract.get("errors", [])
            ]
            diagnostic_warnings: list[str] = []
            for split in ("all", "work", "free"):
                block = stats.get(split) if isinstance(stats.get(split), Mapping) else {}
                n_rows = int(block.get("n", 0) or 0)
                paired = safe_float(block.get("paired_row_t_delta_chi2"), float("nan"))
                chi2_margin = safe_float(block.get("delta_chi2_other_minus_best"), float("nan"))
                selected_wr2 = _native_split_dyn_wr2(context["summary"], split)
                opposite_wr2 = _native_split_dyn_wr2(opposite_summary, split)
                paired_t_style[split] = {
                    "t": paired,
                    "n": n_rows,
                    "df": max(0, n_rows - 1),
                }
                paired_z_compatibility[split] = paired
                row_counts[split] = n_rows
                score_margins[split] = {
                    "delta_chi2_opposite_minus_selected": chi2_margin,
                    "dyn_wR2_opposite_minus_selected": (
                        opposite_wr2 - selected_wr2
                        if math.isfinite(selected_wr2) and math.isfinite(opposite_wr2)
                        else float("nan")
                    ),
                    "positive_favors_selected_hand": True,
                }
                if split == "all" and (n_rows <= 0 or not math.isfinite(chi2_margin)):
                    incomplete_reasons.append("all split lacks a finite matched residual calculation")
                elif n_rows <= 1 or not math.isfinite(paired) or not math.isfinite(chi2_margin):
                    diagnostic_warnings.append(
                        f"{split} paired t-style contrast is unavailable or underdetermined; it does not veto the all-row hand result"
                    )
            if str(stats.get("status")) != "ok":
                incomplete_reasons.append(f"diagnostic status is {stats.get('status', 'missing')}")
            n_best = int(row_join.get("n_best", 0) or 0)
            n_other = int(row_join.get("n_other", 0) or 0)
            n_joined = int(row_join.get("n_joined", 0) or 0)
            if n_joined <= 0 or n_joined != n_best or n_joined != n_other:
                incomplete_reasons.append(
                    f"row join is not exact (selected={n_best}, opposite={n_other}, joined={n_joined})"
                )
            diagnostic_duplicates_best = int(row_join.get("duplicate_keys_best", 0) or 0)
            diagnostic_duplicates_other = int(row_join.get("duplicate_keys_other", 0) or 0)
            if diagnostic_duplicates_best or diagnostic_duplicates_other:
                incomplete_reasons.append(
                    "diagnostic row keys are ambiguous "
                    f"(selected duplicates={diagnostic_duplicates_best}, "
                    f"opposite duplicates={diagnostic_duplicates_other})"
                )
            if (
                int(row_pair_contract.get("joined_rows", 0) or 0) != n_joined
                or int(row_pair_contract.get("selected_rows", 0) or 0) != n_best
                or int(row_pair_contract.get("opposite_rows", 0) or 0) != n_other
            ):
                incomplete_reasons.append("independent row-pair audit disagrees with the diagnostic row-join counts")
            selected_score = score_native_summary(dict(context["summary"]))
            opposite_score = score_native_summary(opposite_summary)
            score_delta = safe_float(selected_score.get("score"), float("nan")) - safe_float(
                opposite_score.get("score"), float("nan")
            )
            opposite_gate_ok, opposite_gate_reasons = hand_diagnostic_supports_other(stats, args)
            materially_opposite = bool(not incomplete_reasons and opposite_gate_ok)
            opposite_config = (
                opposite_summary.get("config") if isinstance(opposite_summary.get("config"), Mapping) else {}
            )
            opposite_best = (
                opposite_summary.get("best_native") if isinstance(opposite_summary.get("best_native"), Mapping) else {}
            )
            opposite_nuisance = (
                opposite_best.get("nuisance") if isinstance(opposite_best.get("nuisance"), Mapping) else {}
            )
            nuisance_mismatches = _pinned_nuisance_mismatches(
                context["nuisance"],
                opposite_nuisance,
            )
            opposite_model_binding = _summary_model_binding(
                opposite_summary,
                opposite_model,
                semantic_hand=opposite_hand,
            )
            opposite_hkl_binding = _summary_hkl_binding(
                opposite_summary,
                Path(ds.hkl),
            )
            opposite_row_table_binding = _summary_native_row_table_binding(
                opposite_summary,
                opposite_run / "native_dyn_calc.csv",
            )
            physics_config_contract = _native_physics_config_contract(
                context["summary"],
                opposite_summary,
            )
            opposite_n_basis = _positive_summary_int(opposite_config, "n_basis")
            opposite_n_points = _positive_summary_int(opposite_config, "n_points")
            if opposite_n_basis != int(context["n_basis"]) or opposite_n_points != int(context["n_points"]):
                incomplete_reasons.append("opposite-hand recomputation did not preserve the selected N/Q contract")
                materially_opposite = False
            if str(opposite_config.get("scale_fit_split") or "") != str(context["scale_fit_split"]):
                incomplete_reasons.append("opposite-hand recomputation did not preserve the selected scale-fit split")
                materially_opposite = False
            if nuisance_mismatches:
                incomplete_reasons.extend(f"pinned nuisance mismatch: {mismatch}" for mismatch in nuisance_mismatches)
                materially_opposite = False
            if not bool(opposite_model_binding.get("passed", False)):
                incomplete_reasons.extend(
                    f"opposite-model provenance: {error}" for error in opposite_model_binding.get("errors", [])
                )
                materially_opposite = False
            if not bool(opposite_hkl_binding.get("passed", False)):
                incomplete_reasons.extend(
                    f"opposite-HKL provenance: {error}" for error in opposite_hkl_binding.get("errors", [])
                )
                materially_opposite = False
            if not bool(opposite_row_table_binding.get("passed", False)):
                incomplete_reasons.extend(
                    f"opposite-row provenance: {error}" for error in opposite_row_table_binding.get("errors", [])
                )
                materially_opposite = False
            if not bool(physics_config_contract.get("passed", False)):
                incomplete_reasons.extend(
                    f"physics config: {error}" for error in physics_config_contract.get("errors", [])
                )
                materially_opposite = False
            comparison_contract = record.get("comparison_contract")
            if isinstance(comparison_contract, dict):
                selected_binding = comparison_contract.get("selected_model_binding")
                selected_binding_passed = bool(
                    isinstance(selected_binding, Mapping) and selected_binding.get("passed", False)
                )
                comparison_contract.update(
                    {
                        "opposite_n_basis": opposite_n_basis,
                        "opposite_n_points": opposite_n_points,
                        "opposite_scale_fit_split": opposite_config.get("scale_fit_split"),
                        "opposite_selected_nuisance": dict(opposite_nuisance),
                        "pinned_nuisance_match": not nuisance_mismatches,
                        "pinned_nuisance_mismatches": nuisance_mismatches,
                        "opposite_model_binding": opposite_model_binding,
                        "opposite_hkl_binding": opposite_hkl_binding,
                        "opposite_row_table_binding": opposite_row_table_binding,
                        "physics_config_contract": physics_config_contract,
                        "model_hand_semantics_aligned": bool(
                            selected_binding_passed and opposite_model_binding.get("passed", False)
                        ),
                    }
                )
            record.update(
                {
                    "status": (
                        "warning_incomplete"
                        if incomplete_reasons
                        else "warning_material_opposite_support"
                        if materially_opposite
                        else "consistent_no_material_opposite_support"
                    ),
                    "row_counts": row_counts,
                    "row_pair_contract": row_pair_contract,
                    "paired_t_style_positive_favors_selected": paired_t_style,
                    "paired_t_style_z_positive_favors_selected": paired_z_compatibility,
                    "paired_t_style_z_compatibility_note": (
                        "legacy field name only; values are paired Student-t-style statistics with df=n-1, "
                        "not calibrated normal Z scores"
                    ),
                    "score_margins_positive_favors_selected": score_margins,
                    "selected_score": selected_score,
                    "opposite_score": opposite_score,
                    "score_delta_selected_minus_opposite_positive_favors_opposite": score_delta,
                    "composite_score_margin_opposite_minus_selected_positive_favors_selected": -score_delta,
                    "materially_favors_opposite": materially_opposite,
                    "opposite_support_gate": {
                        "passed": opposite_gate_ok,
                        "reasons": opposite_gate_reasons,
                        "minimum_z_all": float(getattr(args, "hand_min_z_all", 2.0)),
                        "minimum_z_free": float(getattr(args, "hand_min_z_free", 0.5)),
                        "minimum_free_rows": int(getattr(args, "hand_min_free_rows", 10)),
                        "minimum_score_delta": float(getattr(args, "hand_accept_delta", 0.0)),
                    },
                    "warnings": [*incomplete_reasons, *diagnostic_warnings],
                    "calculation_errors": incomplete_reasons,
                    "primary_evidence": "all_rows",
                    "free_evidence_role": "diagnostic_warning_only",
                }
            )
        except Exception as exc:
            record["status"] = "warning_incomplete"
            record["warnings"] = [f"diagnostic execution failed: {type(exc).__name__}: {exc}"]

    warnings: list[str] = []
    for name in eligible:
        record = records[name]
        if str(record.get("status")) == "warning_incomplete":
            warnings.append(f"{name}: final opposite-hand diagnostic is incomplete")
        elif bool(record.get("materially_favors_opposite", False)):
            warnings.append(f"{name}: final row diagnostic materially favors the opposite hand")
    if warnings:
        status = (
            "warning_incomplete_and_opposite_support"
            if any("incomplete" in warning for warning in warnings)
            and any("opposite hand" in warning for warning in warnings)
            else "warning_incomplete"
            if any("incomplete" in warning for warning in warnings)
            else "warning_opposite_hand_support"
        )
    else:
        status = "consistent"
    payload.update(
        {
            "status": status,
            "datasets": records,
            "consistency_gate": {
                "status": "passed" if not warnings else "warning",
                "passed": not warnings,
                "warnings": warnings,
            },
            "selected_hand_after_diagnostic": str(selected_hand),
        }
    )
    write_json(artifact, payload)
    return payload
