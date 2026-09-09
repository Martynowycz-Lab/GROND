"""Select the final elastic hand with a common row and nuisance contract."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import read_json, try_read_json
from grond.io.config import PrepConfig
from grond.io.structure_contract import absolute_structure_policy, is_shelx_model, make_alternate_hand_model
from grond.paths import engine_script
from grond.workflows.decision import safe_float, safe_label, write_json
from grond.workflows.hand_row_contracts import (
    _args_with_recorded_native_contract,
    _paired_row_split_contract,
    _positive_summary_int,
    _row_contract_nullable_numeric_fields,
    elastic_hand_nuisance,
    score_native_metrics,
)
from grond.workflows.hand_statistics import (
    _adjusted_hand_z_block,
    _combine_adjusted_hand_z,
    _hand_evidence_warnings,
    _pooled_hand_pair_statistics,
    hand_preference_summary,
)
from grond.workflows.model_selection import best_nuisance, dataset_baseline_hkl
from grond.workflows.native_commands import native_stage_cmd


def run_multi_final_elastic_hand_selection(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    final_model: str,
    selected_hand: str,
    final_runs: Mapping[str, Path],
    shared_hand_decision: Mapping[str, object],
) -> dict[str, object]:
    """Select one shared hand at the end from exact elastic negative-thickness pairs.

    Every dataset retains its final N/Q, row support, scale split, thickness,
    jitter, and other nuisance values.  Absorption alone is zeroed because the
    negative-thickness identity is exact only for the Hermitian elastic
    Hamiltonian.  The paired calculation shares each eigendecomposition.
    """

    hand_flip_threshold = float(getattr(args, "hand_flip_zkp_all", 3.5))
    if not math.isfinite(hand_flip_threshold) or hand_flip_threshold <= 0.0:
        raise ValueError("hand inversion threshold must be finite and positive")

    artifact = root / "decisions" / "shared_final_elastic_hand_selection.json"
    symmetry_policy = (
        shared_hand_decision.get("symmetry_policy")
        if isinstance(shared_hand_decision.get("symmetry_policy"), Mapping)
        else absolute_structure_policy(final_model)
    )
    base: dict[str, object] = {
        "schema_version": 1,
        "method": "nakane_negative_thickness_elastic_pair_v1",
        "selection_stage": "after_final_shared_structure_and_nuisance_selection",
        "selected_hand_before": selected_hand,
        "selected_model_before": final_model,
        "absorption_policy": "disabled_for_hand_decision_only",
        "shared_eigendecomposition": True,
        "nuisance_refit": False,
        "jitter_refit": False,
        "known_configuration_used_for_selection": False,
        "datasets": {},
        "artifact": str(artifact),
        "symmetry_policy": dict(symmetry_policy),
    }
    if not bool(getattr(args, "hand_control", True)) or not bool(symmetry_policy.get("run_hand_control", True)):
        result = {
            **base,
            "status": "skipped_not_applicable",
            "hand_switched": False,
            "selected_hand_after": selected_hand,
            "selected_model_after": final_model,
            "reason": str(symmetry_policy.get("reason") or "hand control disabled"),
            "consistency_gate": {"status": "not_applicable", "passed": None, "warnings": []},
            "preferred_hand": "not_applicable",
            "hand_confidence": "not_applicable",
        }
        write_json(artifact, result)
        return result
    if bool(getattr(args, "dry_run", False)):
        result = {
            **base,
            "status": "skipped_dry_run",
            "hand_switched": False,
            "selected_hand_after": selected_hand,
            "selected_model_after": final_model,
        }
        write_json(artifact, result)
        return result

    records: dict[str, dict[str, object]] = {}
    for ds in prep.datasets:
        source_run = Path(final_runs[ds.name])
        source_summary = read_json(source_run / "summary.json")
        config = source_summary.get("config") if isinstance(source_summary.get("config"), Mapping) else {}
        nuisance = best_nuisance(source_run)
        elastic, source_absorption = elastic_hand_nuisance(nuisance)
        n_basis = _positive_summary_int(config, "n_basis")
        n_points = _positive_summary_int(config, "n_points")
        scale_fit_split = str(config.get("scale_fit_split") or "work")
        record: dict[str, object] = {
            "dataset": ds.name,
            "source_run": str(source_run),
            "source_absorption_nuisance": source_absorption,
            "n_basis": n_basis,
            "n_points": n_points,
            "scale_fit_split": scale_fit_split,
        }
        if not nuisance or n_basis is None or n_points is None or scale_fit_split not in {"work", "all"}:
            record.update(
                status="incomplete",
                reason="final run lacks a complete nuisance, N/Q, or scale-fit-split contract",
            )
            records[ds.name] = record
            continue
        pair_run = root / "runs" / ds.name / "multi_final_elastic_hand_pair"
        diagnostic = root / "decisions" / "final_elastic_hand" / f"{safe_label(ds.name)}.json"
        hand_args = _args_with_recorded_native_contract(args, config)
        hand_args.compute_backend = "scipy"
        try:
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    final_model,
                    dataset_baseline_hkl(prep, ds),
                    pair_run,
                    hand_args,
                    n_basis=n_basis,
                    n_points=n_points,
                    stage="hand",
                    pinned=elastic,
                    make_maps=False,
                    endpoint=False,
                    scale_fit_split_override=scale_fit_split,
                    strict_pinned_contract=True,
                    allow_pinned_jitter_fallback=False,
                    paired_inverse_hand=True,
                ),
                logs,
                f"{ds.name}_multi_final_elastic_hand_pair",
                dry_run=False,
            )
            run_logged(
                [
                    python,
                    str(engine_script("absolute_structure_diagnostic.py")),
                    "--best",
                    str(pair_run / "native_dyn_calc.csv"),
                    "--other",
                    str(pair_run / "native_inverse_hand_dyn_calc.csv"),
                    "--out",
                    str(diagnostic),
                ],
                logs,
                f"{ds.name}_multi_final_elastic_hand_diagnostic",
                dry_run=False,
            )
            pair_summary = read_json(pair_run / "summary.json")
            paired = (
                pair_summary.get("paired_inverse_hand")
                if isinstance(pair_summary.get("paired_inverse_hand"), Mapping)
                else {}
            )
            selected_score = score_native_metrics(
                pair_summary,
                paired.get("selected_metrics") if isinstance(paired.get("selected_metrics"), Mapping) else {},
            )
            inverse_score = score_native_metrics(
                pair_summary,
                paired.get("inverse_metrics") if isinstance(paired.get("inverse_metrics"), Mapping) else {},
            )
            stats = read_json(diagnostic)
            all_block = stats.get("all") if isinstance(stats.get("all"), Mapping) else {}
            free_block = stats.get("free") if isinstance(stats.get("free"), Mapping) else {}
            exact_rows = _paired_row_split_contract(
                pair_run / "native_dyn_calc.csv",
                pair_run / "native_inverse_hand_dyn_calc.csv",
                nullable_numeric_fields=_row_contract_nullable_numeric_fields(config, elastic),
            )
            record.update(
                {
                    "status": (
                        "complete"
                        if exact_rows.get("passed")
                        and stats.get("status") == "ok"
                        and _adjusted_hand_z_block(stats, "all").get("status") == "ok"
                        else "calculation_error"
                    ),
                    "calculation_errors": [
                        *list(exact_rows.get("errors") or []),
                        *list(stats.get("calculation_errors") or []),
                        *(
                            [f"all-row adjusted statistic: {_adjusted_hand_z_block(stats, 'all').get('status')}"]
                            if _adjusted_hand_z_block(stats, "all").get("status") != "ok"
                            else []
                        ),
                    ],
                    "pair_run": str(pair_run),
                    "selected_row_table": str(pair_run / "native_dyn_calc.csv"),
                    "inverse_row_table": str(pair_run / "native_inverse_hand_dyn_calc.csv"),
                    "diagnostic": str(diagnostic),
                    "row_pair_contract": exact_rows,
                    "selected_score": selected_score,
                    "inverse_score": inverse_score,
                    "matched_all_row_metrics": all_block.get("matched_prediction_metrics", {}),
                    "matched_born_control": stats.get("matched_born_control", {}),
                    "score_delta_selected_minus_inverse": safe_float(selected_score.get("score"), float("nan"))
                    - safe_float(inverse_score.get("score"), float("nan")),
                    "all_t": safe_float(all_block.get("paired_row_t_delta_chi2"), float("nan")),
                    "all_n": int(all_block.get("n", 0) or 0),
                    "free_t": safe_float(free_block.get("paired_row_t_delta_chi2"), float("nan")),
                    "free_n": int(free_block.get("n", 0) or 0),
                    "positive_t_favors_selected": True,
                }
            )
        except Exception as exc:
            record.update(
                status="calculation_error",
                reason=f"{type(exc).__name__}: {exc}",
                calculation_errors=[f"{type(exc).__name__}: {exc}"],
            )
        records[ds.name] = record

    complete = [record for record in records.values() if record.get("status") == "complete"]
    complete_diagnostics = [try_read_json(Path(str(record.get("diagnostic") or ""))) for record in complete]
    pooled = (
        _pooled_hand_pair_statistics(
            [
                (
                    str(record["dataset"]),
                    Path(str(record["selected_row_table"])),
                    Path(str(record["inverse_row_table"])),
                )
                for record in complete
            ]
        )
        if complete
        else {}
    )
    aggregate_all_t = safe_float((pooled.get("all") or {}).get("paired_row_t_delta_chi2"), float("nan"))
    aggregate_free_t = safe_float((pooled.get("free") or {}).get("paired_row_t_delta_chi2"), float("nan"))
    total_free_rows = int((pooled.get("free") or {}).get("n", 0) or 0)
    total_score_delta = sum(safe_float(record.get("score_delta_selected_minus_inverse"), 0.0) for record in complete)
    incomplete = not records or len(complete) != len(prep.datasets)
    adjusted_z_before = {
        split: _combine_adjusted_hand_z(complete_diagnostics, split) for split in ("all", "work", "free")
    }
    aggregate_all_zkp = safe_float(adjusted_z_before["all"].get("z"), float("nan"))
    aggregate_free_zkp = safe_float(adjusted_z_before["free"].get("z"), float("nan"))
    switch = bool(
        complete
        and not incomplete
        and math.isfinite(aggregate_all_zkp)
        and aggregate_all_zkp <= -hand_flip_threshold
        and is_shelx_model(final_model)
    )
    selected_model_after = final_model
    selected_hand_after = selected_hand
    if switch:
        selected_model_after = str(make_alternate_hand_model(python, final_model, root, logs, False))
        selected_hand_after = "alternate" if selected_hand == "original" else "original"
    orientation = -1.0 if switch else 1.0
    adjusted_z_final = {
        split: {
            **block,
            # Keep Z = numerator / sqrt(variance) in the final hand convention.
            # The original counts remain in adjusted_z_before_selection.
            "signed_numerator": orientation * safe_float(block.get("signed_numerator"), float("nan")),
            "z": (
                orientation * safe_float(block.get("z"), float("nan"))
                if math.isfinite(safe_float(block.get("z"), float("nan")))
                else float("nan")
            ),
            "positive_favors_final_selected_hand": True,
        }
        for split, block in adjusted_z_before.items()
    }
    hand_warnings = _hand_evidence_warnings(
        all_z=aggregate_all_zkp,
        free_z=aggregate_free_zkp,
        switched=switch,
        threshold=hand_flip_threshold,
    )
    if incomplete:
        hand_warnings.append(
            "one or more datasets lacked a complete exact hand-pair contract; no automatic inversion was made"
        )
        hand_warnings.extend(
            f"{name}: hand calculation error: {error}"
            for name, record in records.items()
            if record.get("status") != "complete"
            for error in (
                record.get("calculation_errors") or [record.get("reason", "incomplete physical row comparison")]
            )
        )
    result = {
        **base,
        **hand_preference_summary(
            aggregate_all_zkp,
            active_hand=selected_hand,
            threshold=hand_flip_threshold,
            calculation_valid=not incomplete,
        ),
        "status": "switched" if switch else "kept" if not incomplete else "incomplete_kept",
        "datasets": records,
        "aggregate": {
            "method": "exact pooled per-row inverse-minus-selected squared normalized-residual contrast",
            "all_t_style": aggregate_all_t,
            "free_t_style": aggregate_free_t,
            "free_rows": total_free_rows,
            "score_delta_selected_minus_inverse": total_score_delta,
            "negative_favors_inverse": True,
            "splits": pooled,
            "adjusted_z_before_selection": adjusted_z_before,
            "adjusted_z_final_hand_favored": adjusted_z_final,
        },
        "hand_switched": switch,
        "selected_hand_after": selected_hand_after,
        "selected_model_after": selected_model_after,
        "hand_flip_zkp_all_threshold": hand_flip_threshold,
        "warnings": hand_warnings,
        "calculation_errors": {
            name: record.get("calculation_errors") or [record.get("reason", "incomplete hand calculation")]
            for name, record in records.items()
            if record.get("status") != "complete"
        },
        "decision_rule": (
            "switch once when every dataset has a complete exact row pair and the combined all-row "
            "Klar-Palatinus adjusted Z favors the inverse by at least the declared threshold; held-out "
            "evidence is reported and may warn but does not veto the all-row decision"
        ),
    }
    write_json(artifact, result)
    return result
