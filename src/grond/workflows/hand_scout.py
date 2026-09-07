"""Compare coordinate hands on the same scout contract and combine per-crystal evidence."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import read_json, try_read_json
from grond.io.config import DatasetSpec, PrepConfig
from grond.io.structure_contract import absolute_structure_policy, is_shelx_model
from grond.macromolecule import sha256_file
from grond.workflows.decision import safe_float, score_native_summary, write_json
from grond.workflows.hand_row_contracts import _validated_row_selection_counts
from grond.workflows.hand_statistics import _combine_adjusted_hand_z, hand_preference_summary
from grond.workflows.model_selection import dataset_baseline_hkl
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nuisance_policy import compact_base_nuisance
from grond.workflows.parity_evidence import build_common_sigma_hand_evidence, select_branch_symmetric_nuisance


def raw_free_dyn_wr2(summary: Mapping[str, object]) -> float:
    """Return the recorded free-row dynamical wR2 without score fallbacks."""

    best = summary.get("best_native")
    best = best if isinstance(best, Mapping) else {}
    metrics = best.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    free = metrics.get("free")
    free = free if isinstance(free, Mapping) else {}
    dyn = free.get("dyn")
    dyn = dyn if isinstance(dyn, Mapping) else {}
    return safe_float(dyn.get("wR2_all"), float("nan"))


def _dataset_decision_identity(prep: PrepConfig, ds: DatasetSpec) -> dict[str, object]:
    """Bind transform and hand decisions to one exact dataset/model/split."""

    hkl_path = Path(ds.hkl).expanduser().resolve()
    model_path = Path(prep.structure).expanduser().resolve()
    split_path = Path(ds.split_contract).expanduser().resolve() if ds.split_contract else None
    split_payload = try_read_json(split_path) if split_path is not None and split_path.is_file() else {}
    split_payload = split_payload if isinstance(split_payload, Mapping) else {}
    return {
        "dataset": ds.name,
        "setup_id": str(prep.parity_setup_id).strip(),
        "setup_manifest_sha256": str(prep.parity_setup_manifest_sha256).strip().lower(),
        "physical_crystal_id": str(ds.physical_crystal_id).strip(),
        "acquisition_fingerprint": str(ds.acquisition_fingerprint).strip().lower(),
        "source_hkl": str(hkl_path),
        "source_hkl_sha256": sha256_file(hkl_path) if hkl_path.is_file() else "",
        "source_model": str(model_path),
        "source_model_sha256": (sha256_file(model_path) if model_path.is_file() else ""),
        "split_contract_path": str(split_path) if split_path is not None else "",
        "split_contract_sha256": (sha256_file(split_path) if split_path is not None and split_path.is_file() else ""),
        "split_membership_sha256": str(split_payload.get("membership_sha256") or ""),
    }


def choose_hand_by_scout(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    active_run: Path,
    *,
    scout_max_rows: int | None = None,
) -> tuple[str, dict, Path]:
    """Compare matched original- and alternate-hand scouts.

    ``active_run`` is retained only as a production diagnostic.  Nuisance
    selection is deliberately redone from the exact same predeclared grid for
    both coordinate hands.  One common nuisance is chosen by a hand-label-
    invariant paired work-row objective and then replayed in two singleton
    evidence runs.  Input-coordinate inversion can therefore exchange branch
    labels but cannot change the nuisance state or physical conclusion.
    """

    from grond.workflows.transform_selection import _selected_transform_claim_contract

    production_active_summary = read_json(active_run / "summary.json")
    production_active_score = score_native_summary(production_active_summary)
    hand_root = root / "decisions" / ds.name
    original_grid_dir = hand_root / "hand_nuisance_grid_original"
    alternate_grid_dir = hand_root / "hand_nuisance_grid_alt"
    original_dir = hand_root / "hand_scout_original"
    alternate_dir = hand_root / "hand_scout_alt"
    n_basis = 20 if args.fast else int(args.hand_n_basis)
    n_points = 5 if args.fast else int(args.hand_n_points)
    max_rows = int(args.scout_max_rows if scout_max_rows is None else scout_max_rows)
    hand_nuisance_grid = compact_base_nuisance(args)
    comparison_contract: dict[str, object] = {
        "dataset": ds.name,
        "structure": prep.structure,
        "hkl_transform": ds.hkl_transform,
        "n_basis": n_basis,
        "n_points": n_points,
        "max_rows": max_rows,
        "split_seed": int(args.split_seed),
        "scale_fit_split": "work",
        "production_scale_preference_ignored_for_holdout": bool(args.production_scales),
        "endpoint": False,
        "nuisance_selection_rule": ("min_paired_mean_work_dyn_wR2_then_nuisance_sha256_v1"),
        "nuisance_selection_hand_label_invariant": True,
        "nuisance_selection_uses_free_rows": False,
        "nuisance_selection_uses_known_configuration": False,
        "predeclared_compact_nuisance_grid": hand_nuisance_grid,
        "only_physical_difference": "alternate-hand inversion",
        "dataset_identity": _dataset_decision_identity(prep, ds),
        "transform_claim_contract": _selected_transform_claim_contract(
            ds.hkl_transform,
            selection_source="fixed_matched_hand_comparison",
            parity_policy=str(getattr(args, "transform_parity_policy", "proper-only")),
            source_hkl=ds.hkl,
        ),
    }
    original_score: dict = {}
    alternate_score: dict = {}
    original_summary: dict = {}
    alternate_summary: dict = {}
    symmetric_nuisance_selection: dict[str, object] = {}
    try:
        run_logged(
            native_stage_cmd(
                python,
                ds,
                prep.structure,
                dataset_baseline_hkl(prep, ds),
                original_grid_dir,
                args,
                n_basis=n_basis,
                n_points=n_points,
                stage="scout_hand_nuisance_grid_original",
                pinned=hand_nuisance_grid,
                alternate_hand=False,
                make_maps=False,
                endpoint=False,
                max_rows_override=max_rows,
                scale_fit_split_override="work",
                selection_metric_override="work_wR2",
            ),
            logs,
            f"{ds.name}_scout_hand_nuisance_grid_original",
            dry_run=args.dry_run,
        )
        run_logged(
            native_stage_cmd(
                python,
                ds,
                prep.structure,
                dataset_baseline_hkl(prep, ds),
                alternate_grid_dir,
                args,
                n_basis=n_basis,
                n_points=n_points,
                stage="scout_hand_nuisance_grid_alternate",
                pinned=hand_nuisance_grid,
                alternate_hand=True,
                make_maps=False,
                endpoint=False,
                max_rows_override=max_rows,
                scale_fit_split_override="work",
                selection_metric_override="work_wR2",
            ),
            logs,
            f"{ds.name}_scout_hand_nuisance_grid_alt",
            dry_run=args.dry_run,
        )
        symmetric_nuisance_selection = select_branch_symmetric_nuisance(
            original_grid_dir / "summary.json",
            alternate_grid_dir / "summary.json",
        )
        nui = dict(symmetric_nuisance_selection.get("selected_nuisance") or {})
        if not nui:
            raise ValueError("branch-symmetric nuisance selection returned no selected state")
        comparison_contract["branch_symmetric_nuisance_selection"] = symmetric_nuisance_selection
        comparison_contract["pinned_nuisance"] = nui
        for alternate_hand, out_dir, label in (
            (False, original_dir, "original"),
            (True, alternate_dir, "alt"),
        ):
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    prep.structure,
                    dataset_baseline_hkl(prep, ds),
                    out_dir,
                    args,
                    n_basis=n_basis,
                    n_points=n_points,
                    stage=f"scout_hand_{label}",
                    pinned=nui,
                    alternate_hand=alternate_hand,
                    make_maps=False,
                    endpoint=False,
                    max_rows_override=max_rows,
                    scale_fit_split_override="work",
                    selection_metric_override="work_wR2",
                    strict_pinned_contract=True,
                    allow_pinned_jitter_fallback=False,
                ),
                logs,
                f"{ds.name}_scout_hand_{label}",
                dry_run=args.dry_run,
            )
        original_summary = read_json(original_dir / "summary.json")
        original_score = score_native_summary(original_summary)
        alternate_summary = read_json(alternate_dir / "summary.json")
        alternate_score = score_native_summary(alternate_summary)
    except Exception as exc:
        payload = {
            "selected": "original",
            "selected_hand": "original",
            "scout_selected_before_quality_gate": "original",
            "quality_gate_status": "scout_failed",
            "quality_gate_reasons": [f"matched hand scout failed: {exc}"],
            "active_score": original_score,
            "matched_original_score": original_score,
            "alternate_score": alternate_score,
            "production_active_score": production_active_score,
            "score_delta_original_minus_alternate": float("-inf"),
            "score_delta_original_minus_alternate_hand": float("-inf"),
            "accept_delta": float(args.hand_accept_delta),
            "production_active_run": str(active_run),
            "original_nuisance_grid_run": str(original_grid_dir),
            "alternate_nuisance_grid_run": str(alternate_grid_dir),
            "original_run": str(original_dir),
            "alternate_run": str(alternate_dir),
            "comparison_contract": comparison_contract,
            "transform_claim_contract": comparison_contract["transform_claim_contract"],
            "warning": f"matched hand scout failed: {exc}",
        }
        write_json(root / "decisions" / ds.name / "absolute_structure_decision.json", payload)
        return "original", payload, alternate_dir

    try:
        common_sigma_evidence = build_common_sigma_hand_evidence(
            original_dir / "summary.json",
            alternate_dir / "summary.json",
            minimum_abs_adjusted_z=float(getattr(args, "hand_min_adjusted_z", 2.0)),
        )
        original_score = dict(common_sigma_evidence["original_score"])
        alternate_score = dict(common_sigma_evidence["alternate_score"])
    except Exception as exc:
        common_sigma_evidence = {
            "schema_version": 1,
            "status": "failed",
            "error": str(exc),
            "minimum_abs_adjusted_z": float(getattr(args, "hand_min_adjusted_z", 2.0)),
            "known_configuration_used": False,
        }

    original_rows = original_summary.get("row_selection")
    original_rows = dict(original_rows) if isinstance(original_rows, Mapping) else {}
    alternate_rows = alternate_summary.get("row_selection")
    alternate_rows = dict(alternate_rows) if isinstance(alternate_rows, Mapping) else {}
    row_contract_keys = ("selected_rows", "work_rows", "free_rows")
    original_row_counts = _validated_row_selection_counts(original_rows)
    alternate_row_counts = _validated_row_selection_counts(alternate_rows)
    row_selection_match = bool(
        original_row_counts is not None
        and alternate_row_counts is not None
        and original_row_counts == alternate_row_counts
    )
    original_free_rows = original_row_counts[2] if original_row_counts is not None else 0
    alternate_free_rows = alternate_row_counts[2] if alternate_row_counts is not None else 0
    positive_matched_free_rows = bool(
        original_free_rows > 0 and alternate_free_rows > 0 and original_free_rows == alternate_free_rows
    )
    original_raw_free_wr2 = raw_free_dyn_wr2(original_summary)
    alternate_raw_free_wr2 = raw_free_dyn_wr2(alternate_summary)
    finite_raw_free_metrics = math.isfinite(original_raw_free_wr2) and math.isfinite(alternate_raw_free_wr2)
    if common_sigma_evidence.get("status") == "replayed":
        original_raw_free_wr2 = safe_float(original_score.get("free_dyn_wR2_all"), float("nan"))
        alternate_raw_free_wr2 = safe_float(alternate_score.get("free_dyn_wR2_all"), float("nan"))
        finite_raw_free_metrics = math.isfinite(original_raw_free_wr2) and math.isfinite(alternate_raw_free_wr2)
    comparison_contract.update(
        {
            "original_row_selection": {key: original_rows.get(key) for key in row_contract_keys},
            "alternate_row_selection": {key: alternate_rows.get(key) for key in row_contract_keys},
            "original_row_selection_valid": original_row_counts is not None,
            "alternate_row_selection_valid": alternate_row_counts is not None,
            "row_selection_match": row_selection_match,
            "positive_matched_free_rows": positive_matched_free_rows,
            "original_raw_free_dyn_wR2_all": original_raw_free_wr2 if math.isfinite(original_raw_free_wr2) else None,
            "alternate_raw_free_dyn_wR2_all": alternate_raw_free_wr2 if math.isfinite(alternate_raw_free_wr2) else None,
            "finite_raw_free_metrics": finite_raw_free_metrics,
            "common_sigma_hand_evidence": common_sigma_evidence,
        }
    )
    delta = (
        float(common_sigma_evidence["score_delta_original_minus_alternate"])
        if common_sigma_evidence.get("status") == "replayed"
        else float(original_score["score"] - alternate_score["score"])
    )
    scout_selected = "alternate" if delta > float(args.hand_accept_delta) else "original"
    selected = scout_selected
    gate_reasons: list[str] = []
    evidence_warnings: list[str] = []
    if not row_selection_match:
        gate_reasons.append("matched hand scouts did not report identical selected/work/free row counts")
    if not positive_matched_free_rows:
        evidence_warnings.append("held-out hand evidence is absent; the matched all-row comparison remains primary")
    if not finite_raw_free_metrics:
        evidence_warnings.append("held-out dynamical wR2 is unavailable; it does not veto the all-row hand comparison")
    if common_sigma_evidence.get("status") != "replayed":
        gate_reasons.append(
            "matched hand scouts could not be replayed with one immutable source-sigma vector and scale-design contract"
        )
    if scout_selected == "alternate":
        max_all = float(getattr(args, "hand_max_wR2_all", 0.45))
        max_free = float(getattr(args, "hand_max_free_wR2_all", 0.50))
        min_free_gain = float(getattr(args, "hand_min_free_wR2_gain", 0.0))
        alt_all = safe_float(alternate_score.get("dyn_wR2_all"))
        alt_free = alternate_raw_free_wr2
        active_free = original_raw_free_wr2
        if not math.isfinite(alt_all):
            gate_reasons.append("alternate all-row wR2 is nonfinite")
        elif alt_all > max_all:
            evidence_warnings.append(f"alternate all-row wR2 {alt_all:.4g} exceeds the warning level {max_all:.4g}")
        if not math.isfinite(alt_free) or alt_free > max_free:
            evidence_warnings.append(f"alternate free-row wR2 {alt_free:.4g} exceeds the warning level {max_free:.4g}")
        if math.isfinite(active_free) and math.isfinite(alt_free):
            free_gain = active_free - alt_free
            if free_gain < min_free_gain:
                evidence_warnings.append(
                    f"free-row wR2 gain {free_gain:.4g} is below the warning level {min_free_gain:.4g}"
                )
        else:
            evidence_warnings.append("free-row wR2 values were not finite")
    if gate_reasons:
        selected = "original"
    scout_all_z = safe_float(
        (common_sigma_evidence.get("adjusted_hand_z_original_favored", {}).get("all") or {}).get("z"),
        float("nan"),
    )
    payload = {
        **hand_preference_summary(
            scout_all_z,
            active_hand="original",
            threshold=float(getattr(args, "hand_min_adjusted_z", 2.0)),
            calculation_valid=not gate_reasons,
        ),
        "selected": selected,
        "selected_hand": selected,
        "scout_selected_before_quality_gate": scout_selected,
        "quality_gate_status": "rejected_switch" if gate_reasons else "passed",
        "quality_gate_reasons": gate_reasons,
        "warnings": evidence_warnings,
        "active_score": original_score,
        "matched_original_score": original_score,
        "alternate_score": alternate_score,
        "production_active_score": production_active_score,
        "score_delta_original_minus_alternate": delta,
        "score_delta_original_minus_alternate_hand": delta,
        "accept_delta": float(args.hand_accept_delta),
        "hand_max_wR2_all": float(getattr(args, "hand_max_wR2_all", 0.45)),
        "hand_max_free_wR2_all": float(getattr(args, "hand_max_free_wR2_all", 0.50)),
        "hand_min_free_wR2_gain": float(getattr(args, "hand_min_free_wR2_gain", 0.0)),
        "hand_min_adjusted_z": float(getattr(args, "hand_min_adjusted_z", 2.0)),
        "calibration_evidence_status": (
            "adjusted_z_passed"
            if common_sigma_evidence.get("adjusted_z_threshold_passed") is True
            else "residual_preference_only"
        ),
        "production_active_run": str(active_run),
        "original_nuisance_grid_run": str(original_grid_dir),
        "alternate_nuisance_grid_run": str(alternate_grid_dir),
        "original_run": str(original_dir),
        "alternate_run": str(alternate_dir),
        "comparison_contract": comparison_contract,
        "transform_claim_contract": comparison_contract["transform_claim_contract"],
    }
    write_json(root / "decisions" / ds.name / "absolute_structure_decision.json", payload)
    return selected, payload, alternate_dir


def aggregate_shared_hand_decisions(
    decisions: Mapping[str, Mapping[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Report pooled all-row hand preference; individual free subsets only warn."""

    threshold = float(getattr(args, "hand_min_adjusted_z", 2.0))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("hand adjusted-Z threshold must be finite and positive")
    accept_delta = float(args.hand_accept_delta)
    evidence: list[dict[str, object]] = []
    errors: list[str] = []
    warnings: list[str] = []
    deltas: list[float] = []
    diagnostics: list[dict[str, object]] = []
    for name, decision in sorted(decisions.items()):
        delta = safe_float(decision.get("score_delta_original_minus_alternate"), float("nan"))
        gate_status = str(decision.get("quality_gate_status") or "missing")
        reasons = [str(reason) for reason in decision.get("quality_gate_reasons", [])]
        if gate_status not in {"passed", "passed_with_warnings"} or not math.isfinite(delta):
            errors.append(f"{name}: invalid matched hand calculation ({gate_status}); " + "; ".join(reasons))
        else:
            deltas.append(delta)
        warnings.extend(f"{name}: {warning}" for warning in decision.get("warnings", []))
        if str(decision.get("warning") or ""):
            errors.append(f"{name}: {decision['warning']}")
        comparison = decision.get("comparison_contract") or {}
        common = comparison.get("common_sigma_hand_evidence") or {}
        adjusted = (common.get("adjusted_hand_z_original_favored") or {}).get("all") or {}
        if adjusted.get("status") == "ok":
            diagnostics.append({"all": {"klar_palatinus_adjusted_hand_z": adjusted}})
        active = decision.get("active_score") or {}
        alternate = decision.get("alternate_score") or {}
        evidence.append(
            {
                "dataset": name,
                "dataset_scout_selected_hand": decision.get("selected_hand", decision.get("selected", "original")),
                "quality_gate_status": gate_status,
                "score_delta_original_minus_alternate": delta if math.isfinite(delta) else None,
                "active_dyn_wR2_all": active.get("dyn_wR2_all"),
                "active_free_dyn_wR2_all": active.get("free_dyn_wR2_all"),
                "alternate_dyn_wR2_all": alternate.get("dyn_wR2_all"),
                "alternate_free_dyn_wR2_all": alternate.get("free_dyn_wR2_all"),
                "supports_shared_alternate": bool(math.isfinite(delta) and delta > accept_delta),
                "blocking_reasons": reasons if gate_status != "passed" else [],
            }
        )
    mean_delta = sum(deltas) / len(deltas) if deltas else float("nan")
    pooled = _combine_adjusted_hand_z(diagnostics, "all")
    use_count = len(diagnostics) == len(evidence) and bool(evidence)
    z = safe_float(pooled.get("z"), float("nan"))
    valid = bool(evidence) and not errors and len(deltas) == len(evidence)
    if use_count:
        preference = hand_preference_summary(z, active_hand="original", threshold=threshold, calculation_valid=valid)
        switch = bool(valid and z <= -threshold)
    else:
        preference = {
            "preferred_hand": "alternate"
            if valid and mean_delta > 0.0
            else "original"
            if valid and mean_delta < 0.0
            else "no_preference"
            if valid
            else "unavailable",
            "hand_confidence": "residual_preference_only" if valid else "calculation_error",
            "primary_evidence": "all_rows",
            "free_evidence_role": "diagnostic_warning_only",
            "statistic_is_calibrated_probability": False,
        }
        switch = bool(valid and mean_delta > accept_delta)
        warnings.append(
            "Adjusted reflection-count evidence is unavailable in this historical scout; only the all-row residual preference is reported."
        )
    if deltas and min(deltas) < 0.0 < max(deltas):
        warnings.append("Individual datasets disagree about the hand; the pooled all-row preference is conditional.")
    selected = "alternate" if switch else "original"
    return {
        **preference,
        "status": "calculation_error" if errors else "selected_alternate" if switch else "retained_original",
        "selected": selected,
        "selected_hand": selected,
        "accept_delta": accept_delta,
        "dataset_count": len(evidence),
        "alternate_support_count": sum(bool(row["supports_shared_alternate"]) for row in evidence),
        "aggregate_score_delta_original_minus_alternate": mean_delta if math.isfinite(mean_delta) else None,
        "aggregate_adjusted_z_original_favored": pooled,
        "quality_gate_status": "calculation_error" if errors else "passed",
        "quality_gate_reasons": errors,
        "calculation_errors": errors,
        "warnings": warnings,
        "dataset_evidence": evidence,
        "rule": "use pooled all-row matched evidence; free-set weakness and between-crystal disagreement warn rather than veto; invalid physical calculations do not select a hand",
    }


def choose_shared_hand_by_scout(
    prep: PrepConfig,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    active_runs: Mapping[str, Path],
    scout_rows_by_name: Mapping[str, int],
) -> tuple[str, dict[str, object], dict[str, Path]]:
    """Run per-dataset hand scouts and persist one shared model decision."""

    shared_path = root / "decisions" / "shared_absolute_structure_decision.json"
    policy = absolute_structure_policy(prep.structure)
    skipped_status = ""
    skipped_reason = ""
    if not getattr(args, "auto_decide", True):
        skipped_status = "skipped_auto_decide_disabled"
        skipped_reason = "automatic decisions were disabled"
    elif not getattr(args, "hand_control", True):
        skipped_status = "skipped_hand_control_disabled"
        skipped_reason = "alternate-hand control was disabled"
    elif not policy.get("run_hand_control", True):
        skipped_status = "skipped_not_applicable"
        skipped_reason = str(policy.get("reason") or "alternate-hand control is not applicable")
    elif getattr(args, "dry_run", False):
        skipped_status = "skipped_dry_run"
        skipped_reason = "dry-run mode does not execute hand scouts"

    if skipped_status:
        shared: dict[str, object] = {
            "status": skipped_status,
            "selected": "original",
            "selected_hand": "original",
            "quality_gate_status": "not_run",
            "quality_gate_reasons": [skipped_reason],
            "dataset_count": len(prep.datasets),
            "dataset_evidence": [],
            "symmetry_policy": policy,
            "selected_model": prep.structure,
            "rule": "retain the input hand when the shared multi-dataset hand gate is not run",
        }
        write_json(shared_path, shared)
        for ds in prep.datasets:
            dataset_payload = {
                **policy,
                "status": skipped_status,
                "selected": "original",
                "selected_hand": "original",
                "dataset_scout_selected_hand": "not_run",
                "shared_selected_hand": "original",
                "shared_decision": str(shared_path),
                "reason": skipped_reason,
                "selection_scope": "shared_multi_dataset_model",
            }
            write_json(root / "decisions" / ds.name / "absolute_structure_decision.json", dataset_payload)
        return "original", shared, {}

    per_dataset: dict[str, dict[str, object]] = {}
    alternate_runs: dict[str, Path] = {}
    for ds in prep.datasets:
        _selected, decision, alternate_run = choose_hand_by_scout(
            prep,
            ds,
            args,
            python,
            logs,
            root,
            active_runs[ds.name],
            scout_max_rows=scout_rows_by_name.get(ds.name, int(getattr(args, "scout_max_rows", 1000))),
        )
        per_dataset[ds.name] = {
            **decision,
            "active_run": str(active_runs[ds.name]),
            "scout_max_rows": int(scout_rows_by_name.get(ds.name, int(getattr(args, "scout_max_rows", 1000)))),
        }
        alternate_runs[ds.name] = alternate_run

    shared = aggregate_shared_hand_decisions(per_dataset, args)
    shared["symmetry_policy"] = policy
    shared["per_dataset_decisions"] = {
        ds.name: str(root / "decisions" / ds.name / "absolute_structure_decision.json") for ds in prep.datasets
    }
    shared["active_runs"] = {ds.name: str(active_runs[ds.name]) for ds in prep.datasets}
    shared["alternate_runs"] = {ds.name: str(alternate_runs[ds.name]) for ds in prep.datasets}
    shared["scout_max_rows_by_dataset"] = {
        ds.name: int(scout_rows_by_name.get(ds.name, int(getattr(args, "scout_max_rows", 1000))))
        for ds in prep.datasets
    }
    if shared.get("selected_hand") == "alternate" and not is_shelx_model(prep.structure):
        shared["status"] = "retained_original_unmaterializable_model"
        shared["selected"] = "original"
        shared["selected_hand"] = "original"
        shared["quality_gate_status"] = "retained_original"
        reasons = list(shared.get("quality_gate_reasons") or [])
        reasons.append("the alternate shared model can currently be materialized only from a SHELX .res/.ins input")
        shared["quality_gate_reasons"] = reasons
    shared["selected_model"] = prep.structure
    write_json(shared_path, shared)

    shared_selected = str(shared["selected_hand"])
    for ds in prep.datasets:
        decision = dict(per_dataset[ds.name])
        dataset_selected = str(decision.get("selected_hand") or decision.get("selected") or "original")
        decision.update(
            {
                "selected": shared_selected,
                "selected_hand": shared_selected,
                "dataset_scout_selected_hand": dataset_selected,
                "shared_selected_hand": shared_selected,
                "shared_quality_gate_status": shared.get("quality_gate_status"),
                "shared_quality_gate_reasons": shared.get("quality_gate_reasons", []),
                "shared_decision": str(shared_path),
                "selection_scope": "shared_multi_dataset_model",
            }
        )
        write_json(root / "decisions" / ds.name / "absolute_structure_decision.json", decision)
    return shared_selected, shared, alternate_runs
