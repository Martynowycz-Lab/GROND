"""Calculate both hands, retain their evidence and record the explicit user choice."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from grond.io.config import load_prep
from grond.manual.execution import _child_base, _run_stage_command, _tool_args, _verbose_child_output
from grond.manual.state import (
    _load_workflow,
    _next_command,
    _print_handoff,
    _read_json,
    _record_stage,
    _require_stage,
    _sha256,
    _stage,
    _write_json,
)


def _hand_evidence(
    diagnostics: Mapping[str, Mapping[str, object]],
    *,
    min_z_all: float = 3.5,
    min_z_free: float = 3.5,
    min_free_rows: int = 10,
) -> dict[str, object]:
    min_z_all = float(min_z_all)
    if not math.isfinite(min_z_all) or min_z_all <= 0.0:
        raise ValueError("all-row hand threshold must be finite and positive")
    all_delta = 0.0
    free_delta = 0.0
    all_n = 0
    free_n = 0
    rows: dict[str, object] = {}
    z_numerators = {"all": 0.0, "free": 0.0}
    z_variances = {"all": 0.0, "free": 0.0}
    z_datasets = {"all": 0, "free": 0}
    calculation_errors: list[str] = []
    for name, diagnostic in diagnostics.items():
        if diagnostic.get("calculation_status") == "error":
            calculation_errors.extend(f"{name}: {error}" for error in diagnostic.get("calculation_errors", []))
        all_block = diagnostic.get("all") if isinstance(diagnostic.get("all"), Mapping) else {}
        free_block = diagnostic.get("free") if isinstance(diagnostic.get("free"), Mapping) else {}
        delta_all = float(all_block.get("delta_chi2_other_minus_best", 0.0) or 0.0)
        delta_free = float(free_block.get("delta_chi2_other_minus_best", 0.0) or 0.0)
        all_delta += delta_all
        free_delta += delta_free
        all_n += int(all_block.get("n", 0) or 0)
        free_n += int(free_block.get("n", 0) or 0)
        rows[name] = {
            "all_delta_chi2_inverse_minus_input": delta_all,
            "free_delta_chi2_inverse_minus_input": delta_free,
            "all": all_block,
            "free": free_block,
        }
        for support, block in (("all", all_block), ("free", free_block)):
            adjusted = block.get("klar_palatinus_adjusted_hand_z")
            adjusted = adjusted if isinstance(adjusted, Mapping) else {}
            n = int(adjusted.get("n", 0) or 0)
            k_effective = float(
                adjusted.get(
                    "k_effective_selected_wins_plus_half_ties",
                    adjusted.get("k_selected_lower_abs_intensity_residual", float("nan")),
                )
            )
            w = float(adjusted.get("w", float("nan")))
            if (
                str(adjusted.get("status") or "") == "ok"
                and n > 0
                and math.isfinite(k_effective)
                and math.isfinite(w)
                and n - w > 0.0
            ):
                z_numerators[support] += 2.0 * k_effective - n
                z_variances[support] += n - w
                z_datasets[support] += 1
    combined_z = {
        support: (
            z_numerators[support] / math.sqrt(z_variances[support])
            if z_datasets[support] > 0 and z_variances[support] > 0.0
            else float("nan")
        )
        for support in ("all", "free")
    }
    complete = bool(diagnostics) and z_datasets["all"] == len(diagnostics) and not calculation_errors
    if not complete and not calculation_errors:
        calculation_errors.append("one or more datasets lack a complete all-row adjusted hand calculation")
    decisive_support = bool(
        complete and math.isfinite(combined_z["all"]) and abs(combined_z["all"]) >= float(min_z_all)
    )
    if decisive_support and combined_z["all"] >= float(min_z_all):
        recommendation = "input"
    elif decisive_support and combined_z["all"] <= -float(min_z_all):
        recommendation = "inverse"
    else:
        recommendation = "ambiguous"
    selected_sign = -1.0 if recommendation == "inverse" else 1.0
    oriented_free_z = selected_sign * combined_z["free"]
    warnings: list[str] = []
    preferred = (
        "input"
        if complete and combined_z["all"] > 0.0
        else "inverse"
        if complete and combined_z["all"] < 0.0
        else "no_preference"
        if complete
        else "unavailable"
    )
    if complete and not decisive_support:
        warnings.append(
            f"All-row hand preference is weak or tied (ZKP={combined_z['all']:.3g}); the calculated preference is retained in the report."
        )
    warnings.append(
        "Adjusted ZKP is a conditional count contrast, not calibrated sigma confidence under correlated rows."
    )
    if not math.isfinite(combined_z["free"]):
        warnings.append("Held-out Klar-Palatinus adjusted hand Z is unavailable.")
    elif oriented_free_z < 0.0:
        warnings.append(f"Held-out ZKP {oriented_free_z:.3g} conflicts with the all-row recommended hand.")
    elif oriented_free_z < float(min_z_free):
        warnings.append(
            f"Held-out ZKP {oriented_free_z:.3g} supports the recommended hand only weakly (< {float(min_z_free):.3g})."
        )
    if free_n < int(min_free_rows):
        warnings.append(
            f"Held-out hand evidence contains {free_n} rows, below the reporting target {int(min_free_rows)}."
        )
    return {
        "definition": "positive delta chi2 means the input coordinate hand has lower residual; no known chemistry is used",
        "datasets": rows,
        "combined_all_delta_chi2_inverse_minus_input": all_delta,
        "combined_free_delta_chi2_inverse_minus_input": free_delta,
        "combined_all_rows": all_n,
        "combined_free_rows": free_n,
        "combined_adjusted_z_input_favored": combined_z,
        "minimum_adjusted_z_all": float(min_z_all),
        "minimum_adjusted_z_free": float(min_z_free),
        "minimum_free_rows": int(min_free_rows),
        "decisive_support": decisive_support,
        "recommendation": recommendation,
        "preferred_hand": preferred,
        "calculation_status": "complete" if complete else "error",
        "calculation_errors": calculation_errors,
        "hand_confidence": "strong_conditional"
        if decisive_support
        else "weak_conditional"
        if complete and combined_z["all"]
        else "no_sensitivity"
        if complete
        else "calculation_error",
        "primary_evidence": "all_rows",
        "free_evidence_role": "diagnostic_warning_only",
        "warnings": warnings,
        "decision_rule": (
            "recommend inversion when combined all-row adjusted ZKP <= -threshold; held-out evidence "
            "is reported and may warn but does not veto the recommendation"
        ),
    }


def manual_hand(args) -> int:
    workflow_path, workflow = _load_workflow(args)
    thickness = _require_stage(workflow, "thickness")
    root = Path(workflow["root"])
    out = root / "03_hand"
    out.mkdir(parents=True, exist_ok=True)
    prep = load_prep(Path(workflow["prep"]))
    choice = "skip" if bool(args.skip) else str(args.choice or "report")
    selected_summaries = dict(thickness["outputs"]["selected_summaries"])
    if choice == "skip":
        workflow["chosen_model"] = prep.structure
        handoff = _record_stage(
            workflow_path,
            workflow,
            name="hand",
            output_dir=out,
            status="skipped",
            outputs={
                "choice": "input",
                "chosen_model": prep.structure,
                "diagnostics": {},
                "headline": {
                    "recommendation": "not calculated",
                    "recorded_choice": "input coordinates retained",
                },
            },
            command={"status": "skipped_user_choice"},
            warnings=["Absolute-structure checking was explicitly skipped."],
            next_command=_next_command("feedback", workflow_path, workflow=workflow),
        )
        _print_handoff(handoff)
        return 0
    diagnostics: dict[str, dict[str, object]] = {}
    handoffs: dict[str, dict[str, object]] = {}
    commands: dict[str, object] = {}
    existing = _stage(workflow, "hand")
    reuse_report = bool(existing.get("status") == "needs_choice" and choice in {"input", "inverse"})
    if reuse_report:
        existing_outputs = existing.get("outputs") if isinstance(existing.get("outputs"), Mapping) else {}
        diagnostic_paths = (
            existing_outputs.get("dataset_diagnostics")
            if isinstance(existing_outputs.get("dataset_diagnostics"), Mapping)
            else {}
        )
        handoff_paths = (
            existing_outputs.get("dataset_handoffs")
            if isinstance(existing_outputs.get("dataset_handoffs"), Mapping)
            else {}
        )
        if set(diagnostic_paths) != {dataset.name for dataset in prep.datasets} or set(handoff_paths) != {
            dataset.name for dataset in prep.datasets
        }:
            reuse_report = False
        else:
            diagnostics = {name: _read_json(Path(path)) for name, path in diagnostic_paths.items()}
            handoffs = {name: _read_json(Path(path)) for name, path in handoff_paths.items()}
            commands = {
                "selection": {
                    "status": "reused_existing_hand_report_without_recalculation",
                    "elapsed_seconds": 0.0,
                },
                "evidence_calculation": dict(existing.get("command") or {}),
            }
    if not reuse_report:
        for dataset in prep.datasets:
            dataset_out = out / dataset.name
            command_line = [
                *_child_base(args),
                "hand-check",
                "--source-summary",
                selected_summaries[dataset.name],
                "--out-dir",
                str(dataset_out),
                "--workers",
                str(args.workers),
                "--hand-check-method",
                "negative-thickness",
                "--hand-check-absorption",
                "elastic",
                *_tool_args(args),
            ]
            commands[dataset.name] = _run_stage_command(
                command_line,
                verbose=_verbose_child_output(args),
            )
            diagnostics[dataset.name] = _read_json(dataset_out / "absolute_structure_diagnostic.json")
            handoffs[dataset.name] = _read_json(dataset_out / "hand_handoff.json")
    evidence = _hand_evidence(
        diagnostics,
        min_z_all=float(args.min_adjusted_z_all),
        min_z_free=float(args.min_adjusted_z_free),
        min_free_rows=int(args.min_free_rows),
    )
    evidence_path = out / "combined_hand_evidence.json"
    _write_json(evidence_path, evidence)
    if choice == "report":
        status = "needs_choice"
        chosen_model = ""
        workflow["chosen_model"] = ""
        recommended_choice = str(evidence["recommendation"])
        if recommended_choice not in {"input", "inverse"}:
            recommended_choice = "input"
        next_command = _next_command("hand", workflow_path, f"--choice {recommended_choice}", workflow=workflow)
    elif choice == "input":
        status = "ready"
        chosen_model = prep.structure
        next_command = _next_command("feedback", workflow_path, workflow=workflow)
    elif choice == "inverse":
        inverse_models = [Path(str(handoffs[dataset.name]["inverse_model"])) for dataset in prep.datasets]
        hashes = {_sha256(path) for path in inverse_models}
        if len(hashes) != 1:
            raise RuntimeError("Per-dataset hand checks did not materialize one identical inverse-coordinate model")
        status = "ready"
        chosen_model = str(inverse_models[0].resolve())
        next_command = _next_command("feedback", workflow_path, workflow=workflow)
    else:
        raise SystemExit("--choice must be report, input, inverse, or skip")
    if chosen_model:
        workflow["chosen_model"] = chosen_model
    selected_sign = -1.0 if choice == "inverse" else 1.0
    headline_coordinate_label = "selected" if choice in {"input", "inverse"} else "input"
    hand_headline: dict[str, object] = {
        "recommendation": evidence["recommendation"],
        "recorded_choice": choice if choice != "report" else "awaiting explicit choice",
    }
    for dataset_name, diagnostic in diagnostics.items():
        all_block = diagnostic.get("all") if isinstance(diagnostic.get("all"), Mapping) else {}
        free_block = diagnostic.get("free") if isinstance(diagnostic.get("free"), Mapping) else {}
        all_z = all_block.get("klar_palatinus_adjusted_hand_z", {})
        free_z = free_block.get("klar_palatinus_adjusted_hand_z", {})
        hand_headline[f"{dataset_name}_ZKP_{headline_coordinate_label}_all/free"] = (
            f"{selected_sign * float(all_z.get('z', float('nan'))):+.2f} / "
            f"{selected_sign * float(free_z.get('z', float('nan'))):+.2f}"
        )
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="hand",
        output_dir=out,
        status=status,
        outputs={
            "choice": choice,
            "chosen_model": chosen_model,
            "combined_evidence": str(evidence_path),
            "recommendation": evidence["recommendation"],
            "dataset_handoffs": {name: str(out / name / "hand_handoff.json") for name in handoffs},
            "dataset_diagnostics": {
                name: str(out / name / "absolute_structure_diagnostic.json") for name in diagnostics
            },
            "headline": hand_headline,
        },
        command={"datasets": commands},
        warnings=list(evidence.get("warnings") or [])
        + (
            ["All-row hand evidence is below the decision threshold; inspect both hands before choosing."]
            if evidence["recommendation"] == "ambiguous"
            else []
        ),
        next_command=next_command,
    )
    _print_handoff(handoff)
    return 0
