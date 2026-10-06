"""Select the elastic hand with a common row and nuisance contract (early and final stages)."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from grond.cli_support.execution import run_logged
from grond.cli_support.values import read_json, try_read_json
from grond.io.config import PrepConfig
from grond.io.structure_contract import absolute_structure_policy, is_shelx_model, make_alternate_hand_model
from grond.macromolecule import sha256_file
from grond.paths import engine_script
from grond.reports.console import console_enabled, report_hand_decision
from grond.workflows.crystal_parameters import crystal_arguments
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
    heldout_hand_evidence,
    heldout_hand_rule,
    heldout_mirrored_rule,
    matched_born_delta,
    per_crystal_hand_agreement,
    read_hand_pair_tables,
)
from grond.workflows.model_selection import best_nuisance, dataset_baseline_hkl
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nuisance_policy import resolve_final_n_basis, stage_n_points

HAND_DECISION_EVIDENCE = ("heldout-orbit-sign", "all-row-zkp")
# final-fidelity: an inconclusive early pair at screen fidelity is recomputed at the final N/Q before any refinement.
HAND_EARLY_ESCALATION = ("final-fidelity", "off")
HAND_BLOCK_KEYS = ("orbit_sign_z", "orbits_model_better", "orbits_inverse_better", "n_orbits", "n_rows")


def _grid_values(text: object, default: str) -> list[str]:
    """Split a comma-separated P-spline grid (``default`` when empty) into its values; raise when none remain."""

    values = [part.strip() for part in str(text if text not in (None, "") else default).split(",") if part.strip()]
    if not values:
        raise ValueError("empty P-spline grid for the held-out hand refit")
    return values


def hand_decision_settings(args: argparse.Namespace) -> dict[str, object]:
    """Resolve the hand-decision policy; namespaces without the options keep the legacy rule."""

    evidence = str(getattr(args, "hand_decision_evidence", "all-row-zkp"))
    if evidence not in HAND_DECISION_EVIDENCE:
        raise ValueError(f"unknown hand decision evidence {evidence!r}; expected one of {list(HAND_DECISION_EVIDENCE)}")
    zkp_threshold = float(getattr(args, "hand_flip_zkp_all", 3.5))
    if not math.isfinite(zkp_threshold) or zkp_threshold <= 0.0:
        raise ValueError("hand inversion threshold must be finite and positive")
    orbit_threshold = float(getattr(args, "hand_flip_orbit_z", 3.0))
    if not math.isfinite(orbit_threshold) or orbit_threshold <= 0.0:
        raise ValueError("held-out orbit sign threshold must be finite and positive")
    pair_split = str(getattr(args, "hand_pair_scale_fit_split", "inherit"))
    if pair_split not in {"work", "inherit"}:
        raise ValueError(f"unknown hand-pair scale-fit split {pair_split!r}")
    absorption = str(getattr(args, "hand_pair_absorption", "strip"))
    if absorption not in {"uniform", "strip"}:
        raise ValueError(f"unknown hand-pair absorption policy {absorption!r}")
    scale_model = str(getattr(args, "scale_model", "per-frame") or "per-frame")
    if scale_model not in {"pspline", "per-frame"}:
        raise ValueError(f"unknown scale model {scale_model!r} for the held-out hand refit")
    escalation = str(getattr(args, "hand_early_escalation", "off"))
    if escalation not in HAND_EARLY_ESCALATION:
        raise ValueError(f"unknown early hand escalation {escalation!r}; expected one of {list(HAND_EARLY_ESCALATION)}")
    return {
        "evidence": evidence,
        "hand_flip_zkp_all": zkp_threshold,
        "hand_flip_orbit_z": orbit_threshold,
        "pair_scale_fit_split": pair_split,
        "pair_absorption": absorption,
        "early_escalation": escalation,
        "scale_model": scale_model,
        "scale_refit_config": (
            {
                # The run's own P-spline grids; (n_basis, lambda) is re-chosen by inner orbit CV on each
                # training set, exactly as the engine chooses it per crystal in the nested selection.
                "n_basis_grid": [int(v) for v in _grid_values(getattr(args, "scale_spline_n_basis_grid", None), "32")],
                "lambda_grid": [float(v) for v in _grid_values(getattr(args, "scale_spline_lambda_grid", None), "1")],
                "objective": str(getattr(args, "scale_spline_objective", None) or "row_mr"),
            }
            if scale_model == "pspline"
            else {}
        ),
    }


def _row_rotation_angles(pair_summary: Mapping[str, object], tables: Mapping[str, np.ndarray]) -> np.ndarray:
    """Rotation angle omega(ZCAL) of every pair row, bound to the exact source HKL."""

    from grond.io.xds import XDSData

    inputs = pair_summary.get("inputs") if isinstance(pair_summary.get("inputs"), Mapping) else {}
    hkl_path = Path(str(inputs.get("hkl") or "")).expanduser()
    if not hkl_path.is_file():
        raise ValueError(f"hand pair source HKL is missing: {hkl_path}")
    recorded = str(inputs.get("hkl_sha256") or "")
    if recorded and sha256_file(hkl_path) != recorded:
        raise ValueError("hand pair source HKL changed since the pair was computed")
    data = XDSData(hkl_path, hkl_transform=str(inputs.get("hkl_transform") or "identity"))
    index = np.asarray(tables["row_index"], dtype=int)
    if len(index) and (
        int(index.max()) >= len(data.hkl) or not np.array_equal(np.asarray(data.hkl)[index], tables["hkl"])
    ):
        raise ValueError("hand pair row indices do not bind to the source HKL rows")
    return np.asarray([float(data.omega_from_z(z)) for z in np.asarray(data.zcal, dtype=float)[index]], dtype=float)


def heldout_contract_base(stage: str, model: str, settings: Mapping[str, object]) -> dict[str, object]:
    """Every key of the hand-decision contract, with empty evidence (also for skipped or failed checks)."""

    rule = {
        "orbit_sign_z_min": float(settings["hand_flip_orbit_z"]),
        "require_delta_ci_lower_gt0": True,
        "delta_ci_rule": (
            "improvement R_obs(selected) - R_obs(inverse) 95% CI lower bound > 0, i.e. the recorded "
            "inverse-minus-selected delta_r_obs.ci95 upper bound < 0"
        ),
        "orbit_sign_rule": "pooled orbit_sign_z >= orbit_sign_z_min; positive favours the inverse",
        "require_matched_born_delta_zero": True,
        "require_complete_pairs": True,
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": stage,
        "evidence": str(settings["evidence"]),
        "decision_authority": (
            "heldout_orbit_sign" if settings["evidence"] == "heldout-orbit-sign" else "diagnostic_only"
        ),
        "rule": rule,
        "model": str(model),
        "scale_model": str(settings["scale_model"]),
        "scale_fit_split": "work",
        "pair_scale_fit_split": str(settings["pair_scale_fit_split"]),
        "pooled": {},
        "per_crystal": {},
        "heldout_scheme": {},
        "born_control": {},
        "supports_switch": False,
        "rule_met": False,
        "reasons": [],
        "errors": [],
    }
    return payload


def heldout_hand_decision(
    records: Mapping[str, Mapping[str, object]],
    args: argparse.Namespace,
    *,
    stage: str,
    model: str,
    settings: Mapping[str, object],
    expected_datasets: int,
) -> dict[str, object]:
    """Held-out evidence and rule outcome from the pair row tables (no files written)."""

    payload = heldout_contract_base(stage, model, settings)
    complete = {name: record for name, record in records.items() if record.get("status") == "complete"}
    if not complete or len(complete) != int(expected_datasets):
        payload["errors"].append("one or more datasets lack a complete exact hand pair")
    try:
        from grond.bloch.model import parse_model
        from grond.bloch.splits import load_split_exclusion

        exclusion_path = str(getattr(args, "exclude_orbits_json", "") or "").strip()
        exclusion = load_split_exclusion(exclusion_path) if exclusion_path else None
        crystals = []
        for name, record in complete.items():
            tables = read_hand_pair_tables(
                Path(str(record["selected_row_table"])), Path(str(record["inverse_row_table"]))
            )
            payload["born_control"][name] = matched_born_delta(tables)
            if not payload["born_control"][name]["passed"]:
                payload["errors"].append(f"{name}: matched Born predictions differ between hands")
            pair_summary = read_json(Path(str(record["pair_run"])) / "summary.json")
            pair_config = pair_summary.get("config") if isinstance(pair_summary.get("config"), Mapping) else {}
            if str(pair_config.get("scale_fit_split") or "") != "work":
                # The pair's own row-table scales saw free rows (legacy inherit); record it.
                payload["scale_fit_split"] = str(pair_config.get("scale_fit_split") or "unknown")
            crystals.append(
                {
                    "dataset": name,
                    "tables": tables,
                    "ang": _row_rotation_angles(pair_summary, tables) if settings["scale_model"] == "pspline" else None,
                }
            )
        if crystals:
            evidence = heldout_hand_evidence(
                crystals,
                symmetry_ops=parse_model(Path(model)).get("symmetry_ops"),
                scale_model=str(settings["scale_model"]),
                scale_cfg=settings["scale_refit_config"],
                shell_dmin=exclusion["shell_dmin"] if exclusion is not None else None,
                excluded_orbits=exclusion["orbits"] if exclusion is not None else None,
            )
            payload.update({key: evidence[key] for key in ("pooled", "per_crystal", "heldout_scheme", "scale_refit")})
            rule_met, reasons = heldout_hand_rule(
                evidence["pooled"], orbit_sign_z_min=float(settings["hand_flip_orbit_z"])
            )
            payload["reasons"] = reasons
            payload["rule_met"] = bool(rule_met)
    except Exception as exc:  # recorded; missing evidence never switches the hand
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
    payload["status"] = "complete" if not payload["errors"] else "error"
    payload["supports_switch"] = bool(payload["status"] == "complete" and payload["rule_met"])
    return payload


def _heldout_preference(payload: Mapping[str, object], *, active_hand: str) -> dict[str, object]:
    """Report the held-out preference in the fields of the legacy ZKP summary."""

    pooled = payload.get("pooled") if isinstance(payload.get("pooled"), Mapping) else {}
    z = safe_float(pooled.get("orbit_sign_z"), float("nan"))
    threshold = safe_float((payload.get("rule") or {}).get("orbit_sign_z_min"), float("nan"))
    valid = payload.get("status") == "complete" and math.isfinite(z)
    opposite = "alternate" if active_hand == "original" else "original"
    return {
        "preferred_hand": (
            "unavailable" if not valid else opposite if z > 0.0 else active_hand if z < 0.0 else "no_preference"
        ),
        "hand_confidence": (
            "calculation_error"
            if not valid
            else "no_sensitivity"
            if z == 0.0
            else "strong_heldout"
            if abs(z) >= threshold
            else "weak_heldout"
        ),
        "heldout_orbit_sign_z_positive_favors_inverse": z if valid else None,
        "active_hand_before_selection": active_hand,
        "primary_evidence": "heldout_orbits",
        "all_row_zkp_role": "reported_diagnostic_only",
        "statistic_is_calibrated_probability": False,
    }


def _early_final_agreement(
    early: Mapping[str, object] | None,
    final_payload: Mapping[str, object],
    *,
    switched: bool,
) -> dict[str, object]:
    """Compare the confirmatory final decision with the early decision."""

    if not isinstance(early, Mapping) or not early:
        return {"status": "no_early_decision"}
    pooled = final_payload.get("pooled") if isinstance(final_payload.get("pooled"), Mapping) else {}
    z = safe_float(pooled.get("orbit_sign_z"), float("nan"))
    ci = list((pooled.get("delta_r_obs") or {}).get("ci95") or [float("nan"), float("nan")])
    lower = safe_float(ci[0], float("nan"))
    threshold = safe_float((final_payload.get("rule") or {}).get("orbit_sign_z_min"), float("nan"))
    confirmed = bool(
        math.isfinite(threshold)
        and threshold > 0.0
        and heldout_mirrored_rule(pooled, orbit_sign_z_min=threshold)[0]
    )
    if final_payload.get("status") != "complete":
        status = "final_check_unavailable"
    elif switched:
        status = "conflict_flipped_back"
    elif confirmed:
        status = "confirmed"
    elif math.isfinite(z) and z > 0.0:
        status = "conflict_weak_final_prefers_inverse"
    else:
        status = "not_confirmed_weak"
    delta = safe_float((pooled.get("delta_r_obs") or {}).get("delta"), float("nan"))
    upper = safe_float(ci[1] if len(ci) > 1 else float("nan"), float("nan"))
    agreement = {
        "status": status,
        "early_selected_hand": early.get("selected_hand"),
        "early_switched": bool(early.get("switched")),
        "final_switched": bool(switched),
        "final_orbit_sign_z_positive_favors_inverse": z,
        "final_delta_r_obs_ci95_lower": lower,
        "confirmed_rule": (
            "the final check keeps the refined hand with orbit_sign_z <= -threshold and an inverse-minus-selected "
            "delta_r_obs CI lower bound > 0 (the mirrored switch rule)"
        ),
    }
    for key in ("escalation_status", "decided_on"):
        if early.get(key) is not None:
            agreement[f"early_{key}"] = early.get(key)
    if status == "conflict_flipped_back":
        agreement["message"] = (
            f"the confirmatory final check reversed the early hand decision under the same pooled held-out rule: "
            f"final orbit sign z {z:+.2f} (threshold {threshold:g}), dR_obs(inverse - refined) {delta:+.2f} "
            f"[{lower:+.2f}, {upper:+.2f}] pp; the early decision had selected {early.get('selected_hand')}"
        )
    return agreement


def _elastic_pair_records(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    stage: str,
    model: str,
    source_runs: Mapping[str, Path],
    settings: Mapping[str, object],
    run_prefix: str,
    fidelity: Mapping[str, Mapping[str, int]] | None = None,
    tag: str = "",
) -> dict[str, dict[str, object]]:
    """Compute one exact elastic pair per dataset at its source run's nuisance.

    The pair keeps the source run's N/Q unless ``fidelity`` gives a dataset's ``n_basis``, ``n_points`` and
    ``thickness_points`` (the early escalation at final fidelity).  ``tag`` keeps those runs apart.
    """

    records: dict[str, dict[str, object]] = {}
    for ds in prep.datasets:
        source_run = Path(source_runs[ds.name])
        source_summary = read_json(source_run / "summary.json")
        config = source_summary.get("config") if isinstance(source_summary.get("config"), Mapping) else {}
        nuisance = best_nuisance(source_run)
        override = (fidelity or {}).get(ds.name) or {}
        n_basis = int(override["n_basis"]) if override else _positive_summary_int(config, "n_basis")
        n_points = int(override["n_points"]) if override else _positive_summary_int(config, "n_points")
        source_split = str(config.get("scale_fit_split") or "work")
        # Both hands' pair scales are fitted on work rows unless the legacy split is requested.
        scale_fit_split = "work" if settings["pair_scale_fit_split"] == "work" else source_split
        hand_args = _args_with_recorded_native_contract(args, config)
        hand_args.compute_backend = "scipy"
        if override:
            hand_args.final_thickness_points = int(override["thickness_points"])
        # The values the pair command uses: native_stage_cmd applies the crystal's fixed N/Q and settings.
        effective = crystal_arguments(hand_args, ds.name)
        record: dict[str, object] = {
            "dataset": ds.name,
            "source_run": str(source_run),
            "n_basis": n_basis,
            "n_points": n_points,
            "scale_fit_split": scale_fit_split,
            "source_scale_fit_split": source_split,
            "fidelity": {
                "n_basis": getattr(effective, "fixed_n_basis", None) or n_basis,
                "n_points": getattr(effective, "fixed_n_points", None) or n_points,
                "thickness_points": int(getattr(effective, "final_thickness_points", 0) or 0),
                "source": "final_fidelity_escalation" if override else "source_run",
            },
        }
        try:
            elastic, source_absorption = elastic_hand_nuisance(
                nuisance, absorption_policy=str(settings["pair_absorption"])
            )
        except ValueError as exc:
            # Non-uniform absorption: fail closed for this dataset, never approximate.
            record.update(
                status="calculation_error",
                reason=f"ValueError: {exc}",
                calculation_errors=[f"ValueError: {exc}"],
                source_absorption_nuisance={
                    key: value for key, value in nuisance.items() if str(key).startswith("absorption")
                },
            )
            records[ds.name] = record
            continue
        record["source_absorption_nuisance"] = source_absorption
        if not nuisance or n_basis is None or n_points is None or source_split not in {"work", "all"}:
            record.update(
                status="incomplete",
                reason=f"{stage} run lacks a complete nuisance, N/Q, or scale-fit-split contract",
            )
            records[ds.name] = record
            continue
        pair_run = root / "runs" / ds.name / f"{run_prefix}_{stage}{tag}_elastic_hand_pair"
        diagnostic = root / "decisions" / f"{stage}{tag}_elastic_hand" / f"{safe_label(ds.name)}.json"
        try:
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    model,
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
                f"{ds.name}_{run_prefix}_{stage}{tag}_elastic_hand_pair",
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
                f"{ds.name}_{run_prefix}_{stage}{tag}_elastic_hand_diagnostic",
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
    return records


def _evaluate_pair_records(
    records: Mapping[str, Mapping[str, object]],
    args: argparse.Namespace,
    *,
    stage: str,
    model: str,
    settings: Mapping[str, object],
    expected_datasets: int,
) -> dict[str, object]:
    """Pooled statistics, the held-out payload and both rules' outcomes for one set of pair records."""

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
    incomplete = not records or len(complete) != int(expected_datasets)
    adjusted_z_before = {
        split: _combine_adjusted_hand_z(complete_diagnostics, split) for split in ("all", "work", "free")
    }
    aggregate_all_zkp = safe_float(adjusted_z_before["all"].get("z"), float("nan"))
    heldout_payload = heldout_hand_decision(
        records, args, stage=stage, model=model, settings=settings, expected_datasets=int(expected_datasets)
    )
    zkp_switch = bool(
        complete
        and not incomplete
        and math.isfinite(aggregate_all_zkp)
        and aggregate_all_zkp <= -float(settings["hand_flip_zkp_all"])
        and is_shelx_model(model)
    )
    heldout_switch = bool(
        complete and not incomplete and heldout_payload.get("supports_switch") and is_shelx_model(model)
    )
    return {
        "records": dict(records),
        "complete": complete,
        "incomplete": incomplete,
        "pooled": pooled,
        "aggregate_all_t": safe_float((pooled.get("all") or {}).get("paired_row_t_delta_chi2"), float("nan")),
        "aggregate_free_t": safe_float((pooled.get("free") or {}).get("paired_row_t_delta_chi2"), float("nan")),
        "total_free_rows": int((pooled.get("free") or {}).get("n", 0) or 0),
        "total_score_delta": sum(
            safe_float(record.get("score_delta_selected_minus_inverse"), 0.0) for record in complete
        ),
        "adjusted_z_before": adjusted_z_before,
        "aggregate_all_zkp": aggregate_all_zkp,
        "aggregate_free_zkp": safe_float(adjusted_z_before["free"].get("z"), float("nan")),
        "heldout_payload": heldout_payload,
        "zkp_switch": zkp_switch,
        "heldout_switch": heldout_switch,
        "switch": heldout_switch if settings["evidence"] == "heldout-orbit-sign" else zkp_switch,
    }


def _compact_hand_block(block: Mapping[str, object] | None) -> dict[str, object]:
    """The numbers of one held-out hand block that the console and the reports print."""

    block = block if isinstance(block, Mapping) else {}
    out: dict[str, object] = {key: block.get(key) for key in HAND_BLOCK_KEYS if key in block}
    for name in ("delta_r_obs", "delta_r_all"):
        if isinstance(block.get(name), Mapping):
            out[name] = {key: block[name].get(key) for key in ("delta", "ci95")}
    return out


def _evidence_summary(evaluation: Mapping[str, object]) -> dict[str, object]:
    """Compact record of one evaluation (the screen-fidelity evidence kept beside the escalated decision)."""

    payload = evaluation["heldout_payload"]
    return {
        "status": payload.get("status"),
        "rule_met": bool(payload.get("rule_met")),
        "supports_switch": bool(payload.get("supports_switch")),
        "reasons": list(payload.get("reasons") or []),
        "errors": list(payload.get("errors") or []),
        "pooled": _compact_hand_block(payload.get("pooled")),
        "per_crystal": {
            name: _compact_hand_block(block) for name, block in (payload.get("per_crystal") or {}).items()
        },
        "legacy_all_row_zkp_positive_favors_selected": evaluation["aggregate_all_zkp"],
    }


def early_escalation_trigger(evaluation: Mapping[str, object], settings: Mapping[str, object]) -> tuple[bool, str]:
    """Escalate when the early rule is not met at screen fidelity and the evidence is not clearly against switching.

    Returns (escalate, status).  Missing or failed evidence is not escalated: it never switches the hand, and the
    final check decides with the same rule.  "Clearly against" is the mirrored rule of the deciding evidence.
    """

    heldout_mode = settings["evidence"] == "heldout-orbit-sign"
    payload = evaluation["heldout_payload"]
    if evaluation["incomplete"] or (heldout_mode and payload.get("status") != "complete"):
        return False, "not_escalated_incomplete_evidence"
    if heldout_mode:
        threshold = float(settings["hand_flip_orbit_z"])
        if payload.get("rule_met"):
            return False, "not_needed_rule_met"
        if heldout_mirrored_rule(payload.get("pooled") or {}, orbit_sign_z_min=threshold)[0]:
            return False, "not_needed_clearly_against"
        return True, "escalated"
    zkp, threshold = float(evaluation["aggregate_all_zkp"]), float(settings["hand_flip_zkp_all"])
    if not math.isfinite(zkp):
        return False, "not_escalated_incomplete_evidence"
    if zkp <= -threshold:
        return False, "not_needed_rule_met"
    if zkp >= threshold:
        return False, "not_needed_clearly_against"
    return True, "escalated"


ESCALATION_REASONS = {
    "off": "--hand-early-escalation off (legacy): the early decision uses the screen-fidelity pair only",
    "not_needed_rule_met": "the rule is met at screen fidelity",
    "not_needed_clearly_against": "the mirrored rule holds at screen fidelity: the evidence is clearly against switching",
    "not_escalated_incomplete_evidence": (
        "the screen-fidelity evidence is incomplete or failed; it never switches the hand and the final check decides"
    ),
    "skipped_fast": "--fast: the screen and final fidelity coincide",
    "skipped_same_fidelity": "the final N/Q and thickness points equal the screen ones for every crystal",
}


def _run_early_escalation(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    model: str,
    source_runs: Mapping[str, Path],
    settings: Mapping[str, object],
    run_prefix: str,
    screen: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Decide whether the early pair is recomputed at final fidelity; returns the record and the new evaluation."""

    block: dict[str, object] = {
        "policy": str(settings["early_escalation"]),
        "rule": (
            "escalate when the early rule is not met at screen fidelity and the mirrored rule does not hold "
            "(orbit_sign_z <= -threshold with a dR_obs CI lower bound > 0); rerun the exact pair at the final "
            "N/Q and final thickness points with the screen nuisance before any refinement, and decide on that"
        ),
        "screen_fidelity": {
            name: dict(record.get("fidelity") or {}) for name, record in screen["records"].items()
        },
        "screen_evidence": _evidence_summary(screen),
        "decided_on": "screen_fidelity",
    }

    def done(status: str, **extra) -> tuple[dict[str, object], None]:
        """Return the escalation record with ``status`` and its reason, and no new evaluation."""
        return {**block, "status": status, "reason": ESCALATION_REASONS.get(status, status), **extra}, None

    if settings["early_escalation"] == "off":
        return done("off")
    escalate, status = early_escalation_trigger(screen, settings)
    if not escalate:
        return done(status)
    if bool(getattr(args, "fast", False)):
        return done("skipped_fast")
    try:
        fidelity = {}
        for ds in prep.datasets:
            crystal = crystal_arguments(args, ds.name)
            fidelity[ds.name] = {
                "n_basis": int(
                    getattr(crystal, "fixed_n_basis", None)
                    or resolve_final_n_basis(prep, ds, args, python, logs, root, model, "early_hand_escalation")
                ),
                "n_points": int(getattr(crystal, "fixed_n_points", None) or stage_n_points(args, ds, screen=False)),
                "thickness_points": int(crystal.final_thickness_points),
            }
    except Exception as exc:  # recorded; the screen evidence did not support a switch, so the input hand is kept
        return done("failed", error=f"final-fidelity N/Q: {type(exc).__name__}: {exc}")
    screen_fidelity = block["screen_fidelity"]
    if all(
        all(fidelity[name][key] == (screen_fidelity.get(name) or {}).get(key) for key in fidelity[name])
        for name in fidelity
    ):
        return done("skipped_same_fidelity", final_fidelity=fidelity)
    text = ", ".join(
        f"{name} N/Q {(screen_fidelity.get(name) or {}).get('n_basis')}/{(screen_fidelity.get(name) or {}).get('n_points')}"
        f" -> {values['n_basis']}/{values['n_points']}"
        for name, values in fidelity.items()
    )
    print(
        "hand: the early held-out rule is not met at screen fidelity and the evidence is not clearly against "
        f"switching; recomputing the early exact pair at final fidelity before any refinement ({text})",
        flush=True,
    )
    records = _elastic_pair_records(
        prep,
        args,
        python,
        logs,
        root,
        stage="early",
        model=model,
        source_runs=source_runs,
        settings=settings,
        run_prefix=run_prefix,
        fidelity=fidelity,
        tag="_escalated",
    )
    escalated = _evaluate_pair_records(
        records, args, stage="early", model=model, settings=settings, expected_datasets=len(prep.datasets)
    )
    return (
        {
            **block,
            "status": "escalated",
            "reason": "the rule is not met at screen fidelity and the evidence is not clearly against switching",
            "final_fidelity": fidelity,
            "final_fidelity_evidence": _evidence_summary(escalated),
            "decided_on": "final_fidelity",
        },
        escalated,
    )


def run_elastic_hand_selection(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    stage: str,
    model: str,
    selected_hand: str,
    source_runs: Mapping[str, Path],
    shared_hand_decision: Mapping[str, object],
    run_prefix: str = "multi",
    early_decision: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Select one shared hand from exact elastic negative-thickness pairs.

    Every dataset retains its source run's N/Q, row support, thickness,
    jitter, and other nuisance values.  The paired calculation shares each
    eigendecomposition.  A uniform absorptive potential is kept exactly
    (``--hand-pair-absorption uniform``) or every absorption term is zeroed
    (legacy ``strip``).  With ``--hand-decision-evidence heldout-orbit-sign``
    the hand switches only on the pooled held-out orbit sign test plus the
    paired orbit-bootstrap R_obs interval; the all-row ZKP is reported only.
    The evidence is pooled over every dataset of ``prep``: a crystal whose own
    evidence disagrees is warned about and never vetoes.  At the early stage
    an inconclusive screen-fidelity pair is recomputed at final fidelity
    (``--hand-early-escalation final-fidelity``) and the decision uses it.
    """

    if stage not in {"early", "final"}:
        raise ValueError(f"unknown hand decision stage {stage!r}")
    settings = hand_decision_settings(args)
    hand_flip_threshold = float(settings["hand_flip_zkp_all"])
    heldout_mode = settings["evidence"] == "heldout-orbit-sign"

    artifact = root / "decisions" / f"shared_{stage}_elastic_hand_selection.json"
    heldout_artifact = root / "decisions" / f"hand_decision_heldout_{stage}.json"
    symmetry_policy = (
        shared_hand_decision.get("symmetry_policy")
        if isinstance(shared_hand_decision.get("symmetry_policy"), Mapping)
        else absolute_structure_policy(model)
    )
    base: dict[str, object] = {
        "schema_version": 1,
        "method": "nakane_negative_thickness_elastic_pair_v1",
        "stage": stage,
        "selection_stage": (
            "after_final_shared_structure_and_nuisance_selection"
            if stage == "final"
            else "after_screen_on_hand_blind_start_model"
        ),
        "selected_hand_before": selected_hand,
        "selected_model_before": model,
        "absorption_policy": (
            "uniform_absorption_kept_exact_pair"
            if settings["pair_absorption"] == "uniform"
            else "disabled_for_hand_decision_only"
        ),
        "decision_evidence": settings["evidence"],
        "hand_pair_scale_fit_split": settings["pair_scale_fit_split"],
        "shared_eigendecomposition": True,
        "nuisance_refit": False,
        "jitter_refit": False,
        "known_configuration_used_for_selection": False,
        "datasets": {},
        "artifact": str(artifact),
        "heldout_decision_artifact": str(heldout_artifact),
        "symmetry_policy": dict(symmetry_policy),
    }

    def skipped(status: str, extra: Mapping[str, object]) -> dict[str, object]:
        """Write the unchanged-hand result and its held-out decision artifact for ``status``; return the result."""
        result = {
            **base,
            "status": status,
            "hand_switched": False,
            "selected_hand_after": selected_hand,
            "selected_model_after": model,
            **extra,
        }
        write_json(artifact, result)
        write_json(
            heldout_artifact,
            {
                **heldout_contract_base(stage, model, settings),
                "status": status,
                "selected_hand": selected_hand,
                "switched": False,
                "reason": str(extra.get("reason") or status),
            },
        )
        return result

    if not bool(getattr(args, "hand_control", True)) or not bool(symmetry_policy.get("run_hand_control", True)):
        return skipped(
            "skipped_not_applicable",
            {
                "reason": str(symmetry_policy.get("reason") or "hand control disabled"),
                "consistency_gate": {"status": "not_applicable", "passed": None, "warnings": []},
                "preferred_hand": "not_applicable",
                "hand_confidence": "not_applicable",
            },
        )
    if bool(getattr(args, "dry_run", False)):
        return skipped("skipped_dry_run", {})

    records = _elastic_pair_records(
        prep,
        args,
        python,
        logs,
        root,
        stage=stage,
        model=model,
        source_runs=source_runs,
        settings=settings,
        run_prefix=run_prefix,
    )
    evaluation = _evaluate_pair_records(
        records, args, stage=stage, model=model, settings=settings, expected_datasets=len(prep.datasets)
    )
    escalation: dict[str, object] | None = None
    if stage == "early":
        screen_evaluation = evaluation
        escalation, escalated = _run_early_escalation(
            prep,
            args,
            python,
            logs,
            root,
            model=model,
            source_runs=source_runs,
            settings=settings,
            run_prefix=run_prefix,
            screen=screen_evaluation,
        )
        if escalated is not None:
            screen_artifact = root / "decisions" / "hand_decision_heldout_early_screen_fidelity.json"
            # The complete screen-fidelity evidence stays on disk beside the escalated decision.
            write_json(
                screen_artifact,
                {
                    **screen_evaluation["heldout_payload"],
                    "fidelity": escalation["screen_fidelity"],
                    "decision_authority": "superseded_by_final_fidelity_escalation",
                    "artifact": str(screen_artifact),
                },
            )
            escalation["screen_fidelity_artifact"] = str(screen_artifact)
            escalation["screen_fidelity_datasets"] = screen_evaluation["records"]
            evaluation = escalated
    records = evaluation["records"]
    incomplete = bool(evaluation["incomplete"])
    pooled = evaluation["pooled"]
    aggregate_all_t = evaluation["aggregate_all_t"]
    aggregate_free_t = evaluation["aggregate_free_t"]
    adjusted_z_before = evaluation["adjusted_z_before"]
    aggregate_all_zkp = evaluation["aggregate_all_zkp"]
    aggregate_free_zkp = evaluation["aggregate_free_zkp"]
    heldout_payload = evaluation["heldout_payload"]
    zkp_switch = bool(evaluation["zkp_switch"])
    heldout_switch = bool(evaluation["heldout_switch"])
    switch = bool(evaluation["switch"])
    selected_model_after = model
    selected_hand_after = selected_hand
    if switch:
        selected_model_after = str(make_alternate_hand_model(python, model, root, logs, False))
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
    agreement_by_crystal = None
    if heldout_mode:
        hand_warnings.extend(f"held-out hand evidence: {error}" for error in heldout_payload.get("errors") or [])
        if heldout_payload.get("status") == "complete" and zkp_switch != heldout_switch:
            hand_warnings.append(
                "the legacy all-row ZKP rule and the held-out orbit rule disagree; the held-out rule decides"
            )
        if heldout_payload.get("status") == "complete" and len(heldout_payload.get("per_crystal") or {}) > 1:
            agreement_by_crystal = per_crystal_hand_agreement(
                heldout_payload.get("per_crystal") or {},
                pooled_switches=switch,
                orbit_sign_z_min=float(settings["hand_flip_orbit_z"]),
            )
            # Recorded always; warned with the validation console (--validation-console off keeps the legacy warnings).
            for message in agreement_by_crystal["warnings"] if console_enabled(args) else []:
                hand_warnings.append(f"per-crystal disagreement: {message}")
                print(f"WARN hand ({stage}): {message}", flush=True)
    if escalation is not None and escalation.get("status") == "failed":
        message = f"early hand escalation failed: {escalation.get('error')}; the input hand is kept"
        hand_warnings.append(message)
        print(f"WARN hand: {message}", flush=True)
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
    status = "switched" if switch else "kept" if not incomplete else "incomplete_kept"
    if heldout_mode and not incomplete and heldout_payload.get("status") != "complete":
        status = "heldout_evidence_error_kept"
    if escalation is not None and escalation.get("status") == "failed":
        status = "escalation_failed_kept"
    agreement = _early_final_agreement(early_decision, heldout_payload, switched=switch) if stage == "final" else None
    if agreement is not None and agreement.get("status") not in {"confirmed", "no_early_decision"}:
        message = (
            f"early/final hand decisions: {agreement['status']} (early selected "
            f"{agreement.get('early_selected_hand')}; final held-out orbit sign z "
            f"{safe_float(agreement.get('final_orbit_sign_z_positive_favors_inverse'), float('nan')):+.2f}, "
            "positive favours the inverse of the refined hand)"
        )
        hand_warnings.append(message)
        print(f"WARN hand: {message}", flush=True)
    elif agreement is not None and agreement.get("status") == "confirmed":
        print("hand: the final held-out check confirms the early hand decision", flush=True)
    preference = (
        _heldout_preference(heldout_payload, active_hand=selected_hand)
        if heldout_mode
        else hand_preference_summary(
            aggregate_all_zkp,
            active_hand=selected_hand,
            threshold=hand_flip_threshold,
            calculation_valid=not incomplete,
        )
    )
    result = {
        **base,
        **preference,
        "status": status,
        "datasets": records,
        "aggregate": {
            "method": "exact pooled per-row inverse-minus-selected squared normalized-residual contrast",
            "all_t_style": aggregate_all_t,
            "free_t_style": aggregate_free_t,
            "free_rows": evaluation["total_free_rows"],
            "score_delta_selected_minus_inverse": evaluation["total_score_delta"],
            "negative_favors_inverse": True,
            "splits": pooled,
            "adjusted_z_before_selection": adjusted_z_before,
            "adjusted_z_final_hand_favored": adjusted_z_final,
        },
        "heldout": {
            key: heldout_payload.get(key)
            for key in ("status", "pooled", "rule", "reasons", "errors", "supports_switch", "heldout_scheme")
        },
        "legacy_all_row_zkp_rule_would_switch": zkp_switch,
        "hand_switched": switch,
        "selected_hand_after": selected_hand_after,
        "selected_model_after": selected_model_after,
        "hand_flip_zkp_all_threshold": hand_flip_threshold,
        "hand_flip_orbit_z_threshold": float(settings["hand_flip_orbit_z"]),
        "warnings": hand_warnings,
        "calculation_errors": {
            name: record.get("calculation_errors") or [record.get("reason", "incomplete hand calculation")]
            for name, record in records.items()
            if record.get("status") != "complete"
        },
        "decision_rule": (
            "switch once when every dataset has a complete exact row pair, the matched Born predictions of the "
            "two hands are identical, the pooled held-out orbit sign z favours the inverse by at least the "
            "declared threshold and the paired orbit-bootstrap R_obs(inverse) - R_obs(selected) 95% CI lies "
            "below zero; held-out rows are out-of-fold work orbits plus free orbits with scales refitted on "
            "training orbits only; the all-row ZKP is reported only"
            if heldout_mode
            else "switch once when every dataset has a complete exact row pair and the combined all-row "
            "Klar-Palatinus adjusted Z favors the inverse by at least the declared threshold; held-out "
            "evidence is reported and may warn but does not veto the all-row decision"
        ),
        "evidence_datasets": sorted(str(ds.name) for ds in prep.datasets),
    }
    if isinstance(prep.metadata, Mapping) and isinstance(prep.metadata.get("hand_evidence_datasets"), Mapping):
        result["evidence_dataset_scope"] = dict(prep.metadata["hand_evidence_datasets"])
    if escalation is not None:
        result["escalation"] = escalation
    if agreement_by_crystal is not None:
        result["per_crystal_agreement"] = agreement_by_crystal
    if agreement is not None:
        result["early_final_agreement"] = agreement
    write_json(artifact, result)
    write_heldout_hand_decision(
        heldout_artifact,
        {
            **heldout_payload,
            **({"escalation": {k: v for k, v in escalation.items() if k != "screen_fidelity_datasets"}}
               if escalation is not None else {}),
            **({"per_crystal_agreement": agreement_by_crystal} if agreement_by_crystal is not None else {}),
            "evidence_datasets": result["evidence_datasets"],
            **({"evidence_dataset_scope": result["evidence_dataset_scope"]}
               if "evidence_dataset_scope" in result else {}),
        },
        selected_hand_before=selected_hand,
        selected_hand_after=selected_hand_after,
        switched=switch,
        selected_model_after=selected_model_after,
        legacy_all_row_zkp={
            "z_positive_favors_selected": aggregate_all_zkp,
            "threshold": hand_flip_threshold,
            "would_switch": zkp_switch,
            "paired_row_t_all": aggregate_all_t,
            "paired_row_t_free": aggregate_free_t,
        },
        agreement=agreement,
        selection_artifact=artifact,
    )
    report_hand_decision(heldout_artifact, args)
    return result


def write_heldout_hand_decision(
    path: Path,
    payload: Mapping[str, object],
    *,
    selected_hand_before: str,
    selected_hand_after: str,
    switched: bool,
    selected_model_after: str,
    legacy_all_row_zkp: Mapping[str, object],
    agreement: Mapping[str, object] | None = None,
    selection_artifact: Path | None = None,
) -> dict[str, object]:
    """Write the shared hand-decision contract decisions/hand_decision_heldout_<stage>.json."""

    document = {
        **dict(payload),
        "selected_hand_before": selected_hand_before,
        "selected_hand": selected_hand_after,
        "switched": bool(switched),
        "selected_model_after": str(selected_model_after),
        "legacy_all_row_zkp": dict(legacy_all_row_zkp),
        **({"early_final_agreement": dict(agreement)} if agreement is not None else {}),
        "artifact": str(path),
        **({"selection_artifact": str(selection_artifact)} if selection_artifact is not None else {}),
    }
    write_json(path, document)
    return document


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
    early_decision: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Select (or, after an early decision, confirm) one shared hand after all refinement."""

    return run_elastic_hand_selection(
        prep,
        args,
        python,
        logs,
        root,
        stage="final",
        model=final_model,
        selected_hand=selected_hand,
        source_runs=final_runs,
        shared_hand_decision=shared_hand_decision,
        early_decision=early_decision,
    )


def run_early_elastic_hand_selection(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    start_model: str,
    screen_runs: Mapping[str, Path],
    shared_hand_decision: Mapping[str, object],
    run_prefix: str = "multi",
) -> dict[str, object]:
    """Decide the hand right after the screen, on the hand-blind start model."""

    return run_elastic_hand_selection(
        prep,
        args,
        python,
        logs,
        root,
        stage="early",
        model=start_model,
        selected_hand="original",
        source_runs=screen_runs,
        shared_hand_decision=shared_hand_decision,
        run_prefix=run_prefix,
    )
