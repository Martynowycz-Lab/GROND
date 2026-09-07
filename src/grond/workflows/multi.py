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
from grond.paths import engine_script
from grond.workflows.dataset_gates import (
    apply_dataset_policy_to_nuisance,
    policy_active_runs,
    require_materialized_dataset_policy,
    require_policy_active_runs,
    run_dataset_policy_gate,
)
from grond.workflows.dataset_policy import write_policy_weighted_runs
from grond.workflows.decision import apply_preflight, preflight_dataset, thickness_polish_nuisance, write_json
from grond.workflows.endpoint_selection import run_scale_window_polish
from grond.workflows.final_hand_selection import run_multi_final_elastic_hand_selection
from grond.workflows.forensic_gates import run_forensic_gate, run_resolution_limit_gate
from grond.workflows.hand_row_contracts import _args_with_recorded_native_contract, _positive_summary_int
from grond.workflows.model_selection import (
    best_nuisance,
    dataset_baseline_hkl,
    resolve_structure_update_strategy,
    resolved_frame_offset_candidates,
)
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nuisance_policy import (
    compact_base_nuisance,
    effective_topup_d_min,
    resolve_final_n_basis,
    resolved_row_plan_cache,
)
from grond.workflows.report_commands import combine_cmd
from grond.workflows.scale_selection import run_multi_scale_contract_selection
from grond.workflows.sg_window_selection import run_sg_window_scout
from grond.workflows.shared_refinement import run_multi_shelxl_feedback_loop
from grond.workflows.single import run_feature_gate
from grond.workflows.thickness_materialization import (
    continuous_thickness_cmd,
    continuous_thickness_enabled,
    continuous_thickness_seed_nuisance,
    discrete_thickness_baseline_nuisance,
    discrete_thickness_promotion_record,
    materialize_continuous_nuisance_run,
)
from grond.workflows.transform_selection import choose_transform_by_scout


def run_multi(prep: PrepConfig, args: argparse.Namespace, python: str, logs: Path) -> dict:
    """Run one structure against multiple datasets and combine endpoints."""

    root = Path(args.out_dir)
    if args.auto_decide:

        def decide_dataset(ds0: DatasetSpec) -> tuple[DatasetSpec, int]:
            preflight = preflight_dataset(
                ds0,
                requested_min_peak=ds0.min_peak,
                requested_geometry_percentile=ds0.geometry_percentile,
                requested_scout_rows=args.scout_max_rows,
            )
            write_json(root / "decisions" / ds0.name / "preflight.json", preflight)
            for warning in preflight.warnings:
                print(f"WARN preflight {ds0.name}: {warning}", flush=True)
            if preflight.status == "fatal":
                raise RuntimeError(f"fatal preflight failure for {ds0.name}: {preflight.warnings}")
            ds = apply_preflight(ds0, preflight)
            scout_rows = preflight.scout_max_rows if preflight.scout_max_rows > 0 else args.scout_max_rows
            ds, _transform_decision = choose_transform_by_scout(
                prep, ds, args, python, logs, root, scout_max_rows=scout_rows
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
    screen_runs: dict[str, Path] = {}
    initial_runs: dict[str, Path] = {}
    nuisance: dict[str, dict] = {}
    for ds in prep.datasets:
        run_dir = root / "runs" / ds.name / ("fast_N20_Q5" if args.fast else "screen")
        run_logged(
            native_stage_cmd(
                python,
                ds,
                prep.structure,
                dataset_baseline_hkl(prep, ds),
                run_dir,
                args,
                n_basis=20 if args.fast else args.screen_n_basis,
                n_points=5 if args.fast else args.screen_n_points,
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

    selected_hand = "original"
    symmetry_policy = absolute_structure_policy(prep.structure)
    shared_hand_decision = {
        "schema_version": 2,
        "status": (
            "deferred_until_final_model"
            if args.hand_control and symmetry_policy.get("run_hand_control", True)
            else "skipped_not_applicable"
        ),
        "selection_scope": "shared_multi_dataset_model",
        "selected": selected_hand,
        "symmetry_policy": symmetry_policy,
        "selection_stage": "final_elastic_negative_thickness_pair",
        "method": "nakane_negative_thickness_elastic_pair_v1",
        "absorption_policy": "disabled_for_hand_decision_only",
        "nuisance_refit": False,
        "jitter_refit": False,
        "known_configuration_used_for_selection": False,
        "reason": (
            "hand selection is deferred until all thickness, geometry, nuisance, and shared-structure work is complete"
        ),
    }
    shared_model = prep.structure
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
        if not args.fast:
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
                    n_points=args.screen_n_points,
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
            continuous_seed = continuous_thickness_seed_nuisance(args, active_nui)
            if continuous_thickness_enabled(args) and continuous_seed:
                continuous_baseline_run = active_run
                continuous = root / "runs" / ds.name / "multi_thickness_continuous"
                run_logged(
                    continuous_thickness_cmd(
                        python,
                        ds,
                        shared_model,
                        continuous,
                        args,
                        continuous_seed,
                        n_basis=args.screen_n_basis,
                        n_points=args.screen_n_points,
                        stage="multi_thickness_continuous",
                    ),
                    logs,
                    f"{ds.name}_multi_thickness_continuous",
                    dry_run=args.dry_run,
                )
                active_nui = best_nuisance(continuous) or active_nui or continuous_seed
                active_nui, active_run, materialization_record = materialize_continuous_nuisance_run(
                    prep,
                    ds,
                    args,
                    python,
                    logs,
                    active_model=shared_model,
                    baseline_run=continuous_baseline_run,
                    continuous_run=continuous,
                    selected_nuisance=active_nui,
                    n_basis=args.screen_n_basis,
                    n_points=args.screen_n_points,
                    stage="multi_thickness_continuous_materialized",
                )
            else:
                materialization_record = None
            if materialization_record is not None:
                feature_records.append(materialization_record)
            for family in ["row_resolution", "absorption", "sg_jitter", "sigma_error", "dose"]:
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
    if args.mode in {"regularize", "auto"} and not args.fast and structure_strategy != "off":
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
        elif structure_strategy == "native_gradient":
            shared = root / "runs" / "shared_structure_update"
            cmd = [
                python,
                str(engine_script("native_multi_structure_update.py")),
                "--datasets-csv",
                str(datasets_csv),
                "--res",
                shared_model,
                "--out-dir",
                str(shared),
                "--n-basis",
                str(args.structure_n_basis),
                "--n-points",
                str(args.structure_n_points),
                "--topup-d-min",
                f"{effective_topup_d_min(args, min_d_spacing=min(ds.min_d_spacing for ds in prep.datasets)):g}",
                "--min-d-spacing",
                "0.70",
                "--min-peak",
                "90",
                "--geometry-percentile",
                f"{prep.geometry_percentile:g}",
                "--block-schedule",
                args.structure_blocks,
                "--atom-selection",
                args.structure_atom_selection,
                "--max-nfev",
                str(args.structure_max_nfev),
                "--compute-backend",
                args.compute_backend,
                "--row-plan-cache",
                resolved_row_plan_cache(args, fixed_model=False),
                "--jax-batch-size",
                str(args.jax_batch_size),
                "--jax-matrix-bucket-size",
                str(args.jax_matrix_bucket_size),
                "--bethe-strong-count",
                str(args.bethe_strong_count),
                "--bethe-variant",
                args.bethe_variant,
                "--bethe-promote-denom",
                f"{args.bethe_promote_denom:g}",
                "--bethe-denom-eps",
                f"{args.bethe_denom_eps:g}",
                "--profile-quadrature",
                args.profile_quadrature,
                "--profile-accumulation",
                args.profile_accumulation,
                "--early-stop-window",
                str(args.structure_early_stop_window),
                "--early-stop-min-evals",
                str(args.structure_early_stop_min_evals),
                "--early-stop-min-improvement",
                f"{args.structure_early_stop_min_improvement:g}",
                "--early-stop-metric",
                args.structure_early_stop_metric,
                "--restraints",
                args.restraints,
                "--scale-fit-split",
                "work",
                "--workers",
                str(args.workers),
                "--chunksize",
                str(args.chunksize),
            ]
            run_logged(cmd, logs, "shared_structure_update", dry_run=args.dry_run)
            accepted = shared / "accepted_model.res"
            structure_feedback = {
                "status": "ok" if accepted.exists() else "no_accepted_model",
                "strategy": structure_strategy,
                "run": str(shared),
            }
            if accepted.exists():
                final_model = str(accepted)
        else:
            structure_feedback = {"status": "unsupported_strategy", "strategy": structure_strategy}
    final_runs: dict[str, Path] = {}
    forensic_runs: dict[str, dict] = {}
    scale_polish_runs: dict[str, dict] = {}
    resolution_limits: dict[str, dict] = {}
    sg_window_scout_runs: dict[str, dict] = {}
    for ds in prep.datasets:
        run_dir = initial_runs[ds.name] if args.fast else root / "runs" / ds.name / "final"
        if not args.fast:
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
                    n_points=args.n_points,
                    stage="final",
                    pinned=nuisance[ds.name],
                    make_maps=args.make_maps,
                ),
                logs,
                f"{ds.name}_final",
                dry_run=args.dry_run,
            )
        final_runs[ds.name] = run_dir
        scale_polish_runs[ds.name] = run_scale_window_polish(
            python,
            run_dir,
            root / "scale_polish" / ds.name,
            logs,
            f"{ds.name}_final",
            args,
        )
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
        forensic_runs[ds.name] = run_forensic_gate(
            python,
            run_dir / "native_dyn_calc.csv",
            ds.hkl,
            final_model,
            root / "forensics" / ds.name,
            logs,
            ds.name,
            args,
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
    hand_dataset_names = set(full_publication_runs)
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
            "final_hand_dataset_policy": "primary_and_downweighted_policy_active_datasets_only",
            "excluded_from_final_hand_selection": sorted(
                {dataset.name for dataset in prep.datasets} - hand_dataset_names
            ),
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
        scale_polish_runs = {}
        resolution_limits = {}
        forensic_runs = {}
        for ds in prep.datasets:
            run_dir = final_runs[ds.name]
            scale_polish_runs[ds.name] = run_scale_window_polish(
                python,
                run_dir,
                root / "scale_polish_selected_hand" / ds.name,
                logs,
                f"{ds.name}_final_selected_hand",
                args,
            )
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
            forensic_runs[ds.name] = run_forensic_gate(
                python,
                run_dir / "native_dyn_calc.csv",
                ds.hkl,
                final_model,
                root / "forensics_selected_hand" / ds.name,
                logs,
                ds.name,
                args,
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
    combined_input_runs, publication_runs, scale_contract_selection = run_multi_scale_contract_selection(
        python=python,
        full_runs=final_runs,
        full_input_runs=full_input_runs,
        full_publication_runs=full_publication_runs,
        scale_polish=scale_polish_runs,
        final_dataset_policy=final_dataset_policy,
        final_model=final_model,
        root=root,
        logs=logs,
        args=args,
    )
    combined = root / "runs" / "combined"
    run_logged(
        combine_cmd(python, combined_input_runs, final_model, combined, args),
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
        "structure_update_strategy": structure_strategy,
        "structure_feedback": structure_feedback,
        "forensics": forensic_runs,
        "scale_polish": scale_polish_runs,
        "scale_contract_selection": scale_contract_selection,
        "resolution_limits": resolution_limits,
        "merged_resolution_limit": merged_resolution_limit,
        "sg_window_scout": sg_window_scout_runs,
        "dataset_policy": {
            "structure_update": structure_dataset_policy,
            "final": final_dataset_policy,
        },
    }
