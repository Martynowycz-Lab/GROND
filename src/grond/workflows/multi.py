"""Orchestrate shared-model refinement across the selected datasets."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from grond.cli_support.execution import run_logged, transform_scout_job_count
from grond.cli_support.values import read_json, try_read_json
from grond.io.config import DatasetSpec, PrepConfig, write_datasets_csv
from grond.io.structure_contract import absolute_structure_policy
from grond.reports.console import report_feature_decisions, report_macro_cycles, report_split, report_stage_heldout
from grond.workflows.dataset_gates import (
    apply_dataset_policy_to_nuisance,
    policy_active_runs,
    require_materialized_dataset_policy,
    require_policy_active_runs,
    run_dataset_policy_gate,
)
from grond.workflows.dataset_policy import policy_by_name, write_policy_weighted_runs
from grond.workflows.decision import apply_preflight, preflight_dataset, thickness_polish_nuisance, write_json
from grond.workflows.final_hand_selection import (
    hand_decision_settings,
    run_early_elastic_hand_selection,
    run_multi_final_elastic_hand_selection,
)
from grond.workflows.forensic_gates import run_resolution_limit_gate
from grond.workflows.hand_row_contracts import _args_with_recorded_native_contract, _positive_summary_int
from grond.workflows.model_selection import (
    best_nuisance,
    dataset_baseline_hkl,
    resolve_structure_update_strategy,
    resolved_frame_offset_candidates,
)
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nested_nuisance import (
    NESTED_GATE_FAMILIES,
    feedback_final_nuisance,
    nested_gate_skip_record,
    nested_selection_enabled,
    run_nested_nuisance_stage,
)
from grond.workflows.nuisance_policy import (
    FEATURE_GATE_FAMILIES,
    compact_base_nuisance,
    configure_auto_physics,
    configure_cell_esd,
    resolve_final_n_basis,
    stage_n_points,
)
from grond.workflows.report_commands import combine_cmd
from grond.workflows.sg_window_selection import run_sg_window_scout
from grond.workflows.shared_refinement import run_multi_shelxl_feedback_loop
from grond.workflows.single import run_feature_gate
from grond.workflows.thickness_materialization import (
    discrete_thickness_baseline_nuisance,
    discrete_thickness_promotion_record,
)
from grond.workflows.transform_selection import (
    SCOUT_MAX_ROWS,
    choose_transform_by_scout,
    exclude_metric_incompatible_datasets,
    plan_split_candidates,
    run_cross_crystal_split_check,
)


def final_hand_evidence_datasets(
    prep: PrepConfig,
    final_dataset_policy: Mapping[str, object],
) -> dict[str, object]:
    """Crystals pooled in the final hand check.

    Every crystal except one the final dataset policy excluded, and that exclusion must carry its recorded reason;
    a downweighted or completeness-only crystal is demoted for the endpoint, not for the hand, and stays in the
    pooled evidence.
    """

    names = [dataset.name for dataset in prep.datasets]
    rows = policy_by_name(final_dataset_policy)
    roles = {name: str((rows.get(name) or {}).get("action") or "primary") for name in names}
    excluded = {}
    for name in names:
        if roles[name] != "exclude":
            continue
        reasons = [str(reason) for reason in (rows.get(name) or {}).get("reasons") or [] if str(reason).strip()]
        if not reasons:
            raise RuntimeError(
                f"the final dataset policy excluded {name} without a recorded reason; refusing to drop it "
                "from the pooled hand evidence"
            )
        excluded[name] = reasons
    included = [name for name in names if name not in excluded]
    if not included:
        raise RuntimeError("no crystal is left for the final hand check; refusing to decide the hand on no evidence")
    return {
        "scope": "not-excluded",
        "policy_label": "every_dataset_except_policy_excluded_with_recorded_reason",
        "included": included,
        "excluded": excluded,
        "roles": roles,
        "demoted_included": {name: roles[name] for name in included if roles[name] not in {"primary"}},
        "policy_status": str(final_dataset_policy.get("status") or ""),
    }


def run_multi(prep: PrepConfig, args: argparse.Namespace, python: str, logs: Path) -> dict:
    """Run one structure against multiple datasets and combine endpoints."""

    root = Path(args.out_dir)
    configure_auto_physics(prep, args)
    # Every crystal's split contract comes from one candidate set fixed by cell metrics before any scout, so an
    # orbit is free (or excluded) in every crystal or in none (trusted transforms write no contract).
    if not getattr(args, "trust_hkl_transform", False) and len(prep.datasets) > 1:
        prep, _metric_exclusions = exclude_metric_incompatible_datasets(prep, args, root)
    split_plan = None if getattr(args, "trust_hkl_transform", False) else plan_split_candidates(prep, args)

    def decide_dataset(ds0: DatasetSpec) -> tuple[DatasetSpec, int]:
        """Preflight one crystal and choose its HKL transform by scout; return the dataset and its scout row cap."""
        preflight = preflight_dataset(
            ds0,
            requested_min_peak=ds0.min_peak,
            requested_geometry_percentile=ds0.geometry_percentile,
            requested_scout_rows=SCOUT_MAX_ROWS,
        )
        write_json(root / "decisions" / ds0.name / "preflight.json", preflight)
        for warning in preflight.warnings:
            print(f"WARN preflight {ds0.name}: {warning}", flush=True)
        if preflight.status == "fatal":
            raise RuntimeError(f"fatal preflight failure for {ds0.name}: {preflight.warnings}")
        ds = apply_preflight(ds0, preflight)
        scout_rows = preflight.scout_max_rows if preflight.scout_max_rows > 0 else SCOUT_MAX_ROWS
        ds, _transform_decision = choose_transform_by_scout(
            prep, ds, args, python, logs, root, scout_max_rows=scout_rows, split_plan=split_plan
        )
        return ds, int(scout_rows)

    if len(prep.datasets) > 1 and not args.dry_run:
        decided_by_name: dict[str, DatasetSpec] = {}
        max_workers = min(len(prep.datasets), transform_scout_job_count(args))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="grond-dataset-transform") as pool:
            futures = {pool.submit(decide_dataset, ds0): ds0.name for ds0 in prep.datasets}
            for future in as_completed(futures):
                decided, _scout_rows = future.result()
                decided_by_name[decided.name] = decided
        decided = [decided_by_name[ds0.name] for ds0 in prep.datasets]
    else:
        decided_rows = [decide_dataset(ds0) for ds0 in prep.datasets]
        decided = [item[0] for item in decided_rows]
    prep = PrepConfig(
        structure=prep.structure,
        datasets=decided,
        baseline_hkl=prep.baseline_hkl,
        notes=prep.notes,
        default_hkl_transform=prep.default_hkl_transform,
        geometry_percentile=prep.geometry_percentile,
        parity_setup_id=prep.parity_setup_id,
        parity_setup_manifest_sha256=prep.parity_setup_manifest_sha256,
        metadata=prep.metadata,
    )
    configure_cell_esd(prep.structure, prep.datasets, args)
    screen_runs: dict[str, Path] = {}
    initial_runs: dict[str, Path] = {}
    nuisance: dict[str, dict] = {}
    for ds in prep.datasets:
        run_dir = root / "runs" / ds.name / "screen"
        run_logged(
            native_stage_cmd(
                python,
                ds,
                prep.structure,
                dataset_baseline_hkl(prep, ds),
                run_dir,
                args,
                n_basis=args.screen_n_basis,
                n_points=stage_n_points(args, ds, screen=True),
                stage="screen",
                pinned=compact_base_nuisance(args),
                make_maps=False,
            ),
            logs,
            f"{ds.name}_screen",
            dry_run=args.dry_run,
        )
        screen_runs[ds.name] = run_dir
        initial_runs[ds.name] = run_dir
        nuisance[ds.name] = best_nuisance(run_dir)
    # Every transform is selected and every crystal has screen rows: before any hand, thickness or structure
    # work, refuse a run whose held-out orbits differ between crystals (orbit-hash only; legacy is unchanged).
    # With the shared candidate closure this assertion passes by construction.
    run_cross_crystal_split_check(prep, args, screen_runs, root)
    # Validation console: the split and the screen's held-out R, printed from their recorded JSON.
    report_split(prep, args, screen_runs, root)
    report_stage_heldout(args, screen_runs, root, label="screen", model=prep.structure)

    selected_hand = "original"
    symmetry_policy = absolute_structure_policy(prep.structure)
    hand_settings = hand_decision_settings(args)
    hand_applicable = bool(args.hand_control and symmetry_policy.get("run_hand_control", True))
    early_hand = hand_applicable
    shared_model = prep.structure
    early_hand_decision: dict = {}
    if early_hand:
        # Decide on the hand-blind start model before thickness, gates and the SHELXL loop,
        # so every later stage refines the chosen hand.
        early_hand_decision = run_early_elastic_hand_selection(
            prep,
            args,
            python,
            logs,
            root,
            start_model=prep.structure,
            screen_runs=screen_runs,
            shared_hand_decision={"symmetry_policy": symmetry_policy},
        )
        if str(early_hand_decision.get("status") or "") not in {"kept", "switched", "skipped_dry_run"}:
            print(
                f"WARN hand: early held-out hand decision {early_hand_decision.get('status')!r}; continuing in the "
                "input hand, the final elastic pair decides with the same rule",
                flush=True,
            )
        if bool(early_hand_decision.get("hand_switched", False)):
            selected_hand = str(early_hand_decision["selected_hand_after"])
            shared_model = str(early_hand_decision["selected_model_after"])
            for ds in prep.datasets:
                # Re-screen in the chosen hand so the thickness seed is not chosen in the rejected one.
                run_dir = root / "runs" / ds.name / "screen_selected_hand"
                run_logged(
                    native_stage_cmd(
                        python,
                        ds,
                        shared_model,
                        dataset_baseline_hkl(prep, ds),
                        run_dir,
                        args,
                        n_basis=args.screen_n_basis,
                        n_points=stage_n_points(args, ds, screen=True),
                        stage="screen",
                        pinned=compact_base_nuisance(args),
                        make_maps=False,
                    ),
                    logs,
                    f"{ds.name}_screen_selected_hand",
                    dry_run=args.dry_run,
                )
                screen_runs[ds.name] = run_dir
                initial_runs[ds.name] = run_dir
                nuisance[ds.name] = best_nuisance(run_dir)
            report_stage_heldout(args, screen_runs, root, label="screen_selected_hand", model=shared_model)
    shared_hand_decision = {
        "schema_version": 2,
        "status": (
            "early_selected_final_confirmation_pending"
            if early_hand
            else "deferred_until_final_model"
            if hand_applicable
            else "skipped_not_applicable"
        ),
        "selection_scope": "shared_multi_dataset_model",
        "selected": selected_hand,
        "symmetry_policy": symmetry_policy,
        "selection_stage": (
            "early_elastic_pair_after_screen_then_final_confirmation"
            if early_hand
            else "final_elastic_negative_thickness_pair"
        ),
        "method": "nakane_negative_thickness_elastic_pair_v1",
        "decision_evidence": hand_settings["evidence"],
        "absorption_policy": "uniform_absorption_kept_exact_pair",
        "nuisance_refit": False,
        "jitter_refit": False,
        "known_configuration_used_for_selection": False,
        "reason": (
            "hand selected on held-out evidence right after the screen on the hand-blind start model; "
            "the final elastic pair confirms it with the same rule"
            if early_hand
            else "hand selection is deferred until all thickness, geometry, nuisance, and shared-structure work is complete"
        ),
    }
    if early_hand:
        shared_hand_decision["early_elastic_selection"] = early_hand_decision
    shared_hand_decision["selected_model"] = shared_model
    write_json(root / "decisions" / "shared_absolute_structure_decision.json", shared_hand_decision)
    for ds in prep.datasets:
        decision_path = root / "decisions" / ds.name / "absolute_structure_decision.json"
        decision = try_read_json(decision_path)
        decision["selected_model"] = shared_model
        write_json(decision_path, decision)

    for ds in prep.datasets:
        active_run = initial_runs[ds.name]
        active_nui = nuisance[ds.name]
        screen_frame_offsets = resolved_frame_offset_candidates(screen_runs[ds.name])
        if nested_selection_enabled(args):
            # Cone x dose x thickness distribution x lambda jointly by inner orbit CV on work rows; the in-sample
            # gates then run on the pinned pick (--nuisance-selection sequential: the stages below).
            active_nui, active_run, nested_record = run_nested_nuisance_stage(
                prep, ds, args, python, logs, root, active_model=shared_model, active_nui=active_nui,
                run_prefix="multi",
            )
            feature_records = [nested_record]
            for family in FEATURE_GATE_FAMILIES:
                if family in NESTED_GATE_FAMILIES:
                    feature_records.append(nested_gate_skip_record(family))
                    continue
                active_nui, active_run, record = run_feature_gate(
                    prep,
                    ds,
                    args,
                    python,
                    logs,
                    root,
                    family=family,
                    active_model=shared_model,
                    active_run=active_run,
                    active_nui=active_nui,
                )
                feature_records.append(record)
            write_json(root / "decisions" / ds.name / "feature_decisions.json", feature_records)
        else:
            thick = root / "runs" / ds.name / "multi_thickness_polish"
            thickness_baseline = discrete_thickness_baseline_nuisance(args, active_nui)
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    shared_model,
                    dataset_baseline_hkl(prep, ds),
                    thick,
                    args,
                    n_basis=args.screen_n_basis,
                    n_points=stage_n_points(args, ds, screen=True),
                    stage="multi_thickness_polish",
                    pinned=thickness_polish_nuisance(thickness_baseline, screen_frame_offsets),
                    promotion_baseline=thickness_baseline,
                    scale_fit_split_override="work",
                    selection_metric_override="work_wR2",
                    make_maps=False,
                    endpoint=False,
                ),
                logs,
                f"{ds.name}_multi_thickness_polish",
                dry_run=args.dry_run,
            )
            active_run = thick
            active_nui = best_nuisance(thick) or active_nui
            feature_records: list[dict] = [discrete_thickness_promotion_record(thick, dry_run=bool(args.dry_run))]
            for family in FEATURE_GATE_FAMILIES:
                active_nui, active_run, record = run_feature_gate(
                    prep,
                    ds,
                    args,
                    python,
                    logs,
                    root,
                    family=family,
                    active_model=shared_model,
                    active_run=active_run,
                    active_nui=active_nui,
                )
                feature_records.append(record)
            write_json(root / "decisions" / ds.name / "feature_decisions.json", feature_records)
            report_feature_decisions(
                ds.name, root / "decisions" / ds.name / "feature_decisions.json", args, nuisance=active_nui
            )
        initial_runs[ds.name] = active_run
        nuisance[ds.name] = active_nui

    structure_dataset_policy = run_dataset_policy_gate(
        initial_runs,
        root / "dataset_policy" / "structure_update",
        args,
        label="structure_update",
    )
    require_materialized_dataset_policy(
        structure_dataset_policy,
        context="structure-update",
        dry_run=bool(args.dry_run),
    )
    datasets_csv = root / "prep_datasets_with_nuisance.csv"
    write_datasets_csv(datasets_csv, prep, apply_dataset_policy_to_nuisance(nuisance, structure_dataset_policy))
    final_model = shared_model
    structure_strategy = resolve_structure_update_strategy(args, multi_dataset=True)
    structure_feedback: dict = {"status": "not_run", "strategy": structure_strategy}
    if structure_strategy != "off":
        if structure_strategy == "shelxl_feedback":
            final_model, structure_feedback = run_multi_shelxl_feedback_loop(
                prep,
                args,
                python,
                logs,
                root,
                start_model=shared_model,
                nuisance=nuisance,
                structure_dataset_policy=structure_dataset_policy,
            )
            report_macro_cycles(root / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json", args)
            # The final, hand-pair and publication stages use the thickness selected for the kept model.
            nuisance = feedback_final_nuisance(nuisance, structure_feedback)
        else:
            structure_feedback = {"status": "unsupported_strategy", "strategy": structure_strategy}
    final_runs: dict[str, Path] = {}
    resolution_limits: dict[str, dict] = {}
    sg_window_scout_runs: dict[str, dict] = {}
    for ds in prep.datasets:
        run_dir = root / "runs" / ds.name / "final"
        final_n_basis = resolve_final_n_basis(prep, ds, args, python, logs, root, final_model, "final")
        run_logged(
            native_stage_cmd(
                python,
                ds,
                final_model,
                dataset_baseline_hkl(prep, ds),
                run_dir,
                args,
                n_basis=final_n_basis,
                n_points=stage_n_points(args, ds, screen=False),
                stage="final",
                pinned=nuisance[ds.name],
                make_maps=args.make_maps,
            ),
            logs,
            f"{ds.name}_final",
            dry_run=args.dry_run,
        )
        final_runs[ds.name] = run_dir
        resolution_limits[ds.name] = run_resolution_limit_gate(
            python,
            {ds.name: run_dir},
            final_model,
            root / "resolution_limits" / ds.name,
            logs,
            ds.name,
            args,
            title=f"GROND resolution-limit promotion: {ds.name}",
            scope="dataset",
        )
    final_dataset_policy = run_dataset_policy_gate(
        final_runs,
        root / "dataset_policy" / "final",
        args,
        label="final",
    )
    require_materialized_dataset_policy(
        final_dataset_policy,
        context="final",
        dry_run=bool(args.dry_run),
    )
    if args.dry_run or final_dataset_policy.get("status") in {
        "disabled",
        "single_dataset",
        "dry_run",
        "missing_row_table",
    }:
        full_input_runs = final_runs
        full_publication_runs = final_runs
    else:
        weighted_runs = write_policy_weighted_runs(
            final_runs,
            final_dataset_policy,
            root / "dataset_policy" / "final" / "weighted_runs",
            include_endpoint_only=True,
        )
        if not weighted_runs:
            raise RuntimeError(
                "multi-dataset final policy selected no endpoint-contributing dataset; "
                "refusing to fall back to all final runs"
            )
        full_input_runs = weighted_runs
        full_publication_runs = policy_active_runs(weighted_runs, final_dataset_policy)
        require_policy_active_runs(full_publication_runs, context="final publication")
    hand_scope = final_hand_evidence_datasets(prep, final_dataset_policy)
    hand_dataset_names = set(hand_scope["included"])
    hand_prep = PrepConfig(
        structure=prep.structure,
        datasets=[dataset for dataset in prep.datasets if dataset.name in hand_dataset_names],
        baseline_hkl=prep.baseline_hkl,
        notes=prep.notes,
        default_hkl_transform=prep.default_hkl_transform,
        geometry_percentile=prep.geometry_percentile,
        parity_setup_id=prep.parity_setup_id,
        parity_setup_manifest_sha256=prep.parity_setup_manifest_sha256,
        metadata={
            **dict(prep.metadata or {}),
            "final_hand_dataset_policy": hand_scope["policy_label"],
            "excluded_from_final_hand_selection": sorted(hand_scope["excluded"]),
            "hand_evidence_datasets": hand_scope,
        },
    )
    post_feedback_multi_hand_diagnostic = run_multi_final_elastic_hand_selection(
        hand_prep,
        args,
        python,
        logs,
        root,
        final_model=final_model,
        selected_hand=selected_hand,
        final_runs={name: final_runs[name] for name in hand_dataset_names},
        shared_hand_decision=shared_hand_decision,
        early_decision=(
            {
                "selected_hand": selected_hand,
                "switched": bool(early_hand_decision.get("hand_switched", False)),
                "escalation_status": (early_hand_decision.get("escalation") or {}).get("status"),
                "decided_on": (early_hand_decision.get("escalation") or {}).get("decided_on"),
            }
            if early_hand
            else None
        ),
    )
    if bool(post_feedback_multi_hand_diagnostic.get("hand_switched", False)):
        final_model = str(post_feedback_multi_hand_diagnostic["selected_model_after"])
        selected_hand = str(post_feedback_multi_hand_diagnostic["selected_hand_after"])
        switched_runs: dict[str, Path] = {}
        for ds in prep.datasets:
            source_run = final_runs[ds.name]
            source_summary = read_json(source_run / "summary.json")
            source_config = source_summary.get("config") if isinstance(source_summary.get("config"), Mapping) else {}
            source_nuisance = best_nuisance(source_run) or nuisance[ds.name]
            selected_run = root / "runs" / ds.name / "final_selected_hand"
            selected_args = _args_with_recorded_native_contract(args, source_config)
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    final_model,
                    dataset_baseline_hkl(prep, ds),
                    selected_run,
                    selected_args,
                    n_basis=_positive_summary_int(source_config, "n_basis") or int(args.final_basis_fallback),
                    n_points=_positive_summary_int(source_config, "n_points") or int(args.n_points),
                    stage="final",
                    pinned=source_nuisance,
                    make_maps=args.make_maps,
                    endpoint=True,
                    scale_fit_split_override=str(source_config.get("scale_fit_split") or "work"),
                    strict_pinned_contract=True,
                    allow_pinned_jitter_fallback=False,
                ),
                logs,
                f"{ds.name}_final_selected_hand",
                dry_run=False,
            )
            switched_runs[ds.name] = selected_run
        final_runs = switched_runs
        resolution_limits = {}
        for ds in prep.datasets:
            run_dir = final_runs[ds.name]
            resolution_limits[ds.name] = run_resolution_limit_gate(
                python,
                {ds.name: run_dir},
                final_model,
                root / "resolution_limits_selected_hand" / ds.name,
                logs,
                ds.name,
                args,
                title=f"GROND resolution-limit promotion: {ds.name}",
                scope="dataset",
            )
        final_dataset_policy = run_dataset_policy_gate(
            final_runs,
            root / "dataset_policy" / "final_selected_hand",
            args,
            label="final_selected_hand",
        )
        require_materialized_dataset_policy(
            final_dataset_policy,
            context="final selected hand",
            dry_run=False,
        )
        if final_dataset_policy.get("status") in {"disabled", "single_dataset", "missing_row_table"}:
            full_input_runs = final_runs
            full_publication_runs = final_runs
        else:
            weighted_runs = write_policy_weighted_runs(
                final_runs,
                final_dataset_policy,
                root / "dataset_policy" / "final_selected_hand" / "weighted_runs",
                include_endpoint_only=True,
            )
            full_input_runs = weighted_runs
            full_publication_runs = policy_active_runs(weighted_runs, final_dataset_policy)
            require_policy_active_runs(full_publication_runs, context="final selected-hand publication")
        shared_hand_decision = {
            **shared_hand_decision,
            "status": "complete",
            "selected": selected_hand,
            "selected_model": final_model,
            "final_elastic_selection": post_feedback_multi_hand_diagnostic,
        }
        write_json(root / "decisions" / "shared_absolute_structure_decision.json", shared_hand_decision)
    else:
        shared_hand_decision = {
            **shared_hand_decision,
            "status": "complete"
            if post_feedback_multi_hand_diagnostic.get("status") in {"kept", "skipped_not_applicable"}
            else str(post_feedback_multi_hand_diagnostic.get("status") or "incomplete"),
            "selected": selected_hand,
            "selected_model": final_model,
            "final_elastic_selection": post_feedback_multi_hand_diagnostic,
        }
        write_json(root / "decisions" / "shared_absolute_structure_decision.json", shared_hand_decision)
    # The combined endpoint uses every crystal's own full P-spline scales.
    combined_input_runs = {str(name): Path(path) for name, path in full_input_runs.items()}
    publication_runs = {str(name): Path(path) for name, path in full_publication_runs.items()}
    combined = root / "runs" / "combined"
    run_logged(
        combine_cmd(python, combined_input_runs, final_model, combined, args, free_validation=True),
        logs,
        "combine_final",
        dry_run=args.dry_run,
    )
    merged_resolution_limit = run_resolution_limit_gate(
        python,
        combined_input_runs,
        final_model,
        root / "resolution_limits" / "combined",
        logs,
        "combined",
        args,
        title=f"GROND merged resolution-limit promotion: {', '.join(ds.name for ds in prep.datasets)}",
        scope="merged",
    )
    for ds in prep.datasets:
        sg_window_scout_runs[ds.name] = run_sg_window_scout(
            python,
            prep,
            ds,
            final_model,
            final_runs[ds.name],
            root / "sg_window_scout" / ds.name,
            logs,
            f"{ds.name}_final",
            args,
        )
    return {
        "combined": str(combined),
        "model": final_model,
        "selected_hand": selected_hand,
        "shared_absolute_structure_decision": str(root / "decisions" / "shared_absolute_structure_decision.json"),
        "runs": {k: str(v) for k, v in final_runs.items()},
        "combined_input_runs": {k: str(v) for k, v in combined_input_runs.items()},
        "publication_runs": {k: str(v) for k, v in publication_runs.items()},
        "publication_full_runs": {k: str(v) for k, v in full_publication_runs.items()},
        "post_feedback_multi_hand_diagnostic": post_feedback_multi_hand_diagnostic,
        "early_hand_decision": early_hand_decision,
        "hand_decision_heldout": {
            stage: str(root / "decisions" / f"hand_decision_heldout_{stage}.json")
            for stage in ("early", "final")
            if (root / "decisions" / f"hand_decision_heldout_{stage}.json").is_file()
        },
        "structure_update_strategy": structure_strategy,
        "structure_feedback": structure_feedback,
        "resolution_limits": resolution_limits,
        "merged_resolution_limit": merged_resolution_limit,
        "sg_window_scout": sg_window_scout_runs,
        "dataset_policy": {
            "structure_update": structure_dataset_policy,
            "final": final_dataset_policy,
        },
    }
