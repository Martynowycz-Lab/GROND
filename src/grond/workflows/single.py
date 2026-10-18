"""Orchestrate one dataset; stage builders and selection rules live beside this module."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import read_json, try_read_json
from grond.hashing import sha256_file
from grond.io.config import DatasetSpec, PrepConfig
from grond.io.structure_contract import (
    absolute_structure_policy,
    is_shelx_model,
    make_alternate_hand_model,
    make_noh_model,
)
from grond.paths import engine_script
from grond.reports.console import (
    console_enabled,
    report_feature_decisions,
    report_hand_decision,
    report_macro_cycles,
    report_split,
    report_stage_heldout,
)
from grond.workflows.crystal_parameters import crystal_arguments
from grond.workflows.decision import (
    apply_preflight,
    nuisance_score_rows,
    preflight_dataset,
    safe_float,
    score_native_summary,
    should_stop_for_bad_data,
    thickness_polish_nuisance,
    write_json,
)
from grond.workflows.endpoint_commands import shelxl_feedback_cmd
from grond.workflows.endpoint_selection import run_scale_window_polish
from grond.workflows.final_hand_selection import (
    _early_final_agreement,
    _heldout_preference,
    hand_decision_settings,
    heldout_contract_base,
    heldout_hand_decision,
    run_early_elastic_hand_selection,
    write_heldout_hand_decision,
)
from grond.workflows.forensic_gates import run_resolution_limit_gate
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
    _hand_evidence_warnings,
    hand_diagnostic_supports_other,
    hand_preference_summary,
)
from grond.workflows.heldout_gates import run_heldout_feature_gate, uses_heldout_gate
from grond.workflows.model_selection import (
    best_endpoint_model,
    best_nuisance,
    dataset_baseline_hkl,
    resolve_structure_update_strategy,
    resolved_frame_offset_candidates,
    shelxl_feedback_current_model,
    shelxl_feedback_current_run,
    should_run_endpoint_feedback,
)
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nested_nuisance import (
    NESTED_GATE_FAMILIES,
    nested_feedback_args,
    nested_gate_skip_record,
    nested_selection_enabled,
    run_nested_nuisance_stage,
)
from grond.workflows.nuisance_policy import (
    FEATURE_GATE_FAMILIES,
    compact_base_nuisance,
    configure_auto_physics,
    configure_cell_esd,
    feature_grid,
    feature_is_enabled,
    resolve_final_n_basis,
    resolved_final_n_points,
    stage_n_points,
)
from grond.workflows.sg_window_selection import run_sg_window_scout
from grond.workflows.thickness_materialization import (
    continuous_thickness_cmd,
    continuous_thickness_enabled,
    continuous_thickness_seed_nuisance,
    discrete_thickness_baseline_nuisance,
    discrete_thickness_promotion_record,
    materialize_continuous_nuisance_run,
)
from grond.workflows.transform_selection import _selected_transform_claim_contract, choose_transform_by_scout

# Loop stops that reject a cycle's model but leave the model entering it valid (shelxl_feedback_loop). A failed
# cycle (non-zero return) is not one of them: it still raises.
FEEDBACK_SAFE_STOPS = frozenset({"stopped_adp_pathology", "stopped_unsafe_endpoint"})

def feedback_loop_outcome(struct_dir: Path, dataset: str, entering_model: str) -> tuple[Path | None, dict]:
    """The run holding the loop's usable model, or None plus a fallback record when a safe stop kept the entering model.

    A loop whose first cycle ends NPD/unsafe adopts nothing; like the multi-crystal loop, the clean model that entered
    it is kept and the ordinary final stage evaluates it. Any other loop without a usable model raises.
    """

    feedback_summary = try_read_json(struct_dir / "shelxl_feedback_loop_summary.json")
    run = shelxl_feedback_current_run(struct_dir)
    if run is not None:
        return run, {}
    loop_status = str(feedback_summary.get("status") or "")
    if loop_status in FEEDBACK_SAFE_STOPS:
        print(
            f"WARN {dataset}: SHELXL feedback stopped ({loop_status}) before adopting a cycle; the pre-loop model "
            "is kept and evaluated by the final stage",
            flush=True,
        )
        return None, {
            "status": loop_status,
            "kept": "pre_loop_model",
            "model": str(entering_model),
            "stopped_reason": feedback_summary.get("stopped_reason"),
            "rejected_model": feedback_summary.get("rejected_model"),
        }
    runs = feedback_summary.get("runs") or []
    failed = runs[-1] if runs else {}
    raise RuntimeError(
        "SHELXL feedback did not produce a usable RES model. "
        f"Last cycle status: returncode={failed.get('returncode', 'n/a')}; "
        f"endpoint={failed.get('output_model', 'n/a')}; "
        f"health={failed.get('endpoint_res_health', {})}. "
        f"See {struct_dir / 'shelxl_feedback_loop_summary.json'} and the cycle run.log."
    )



def run_feature_gate(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    family: str,
    active_model: str,
    active_run: Path,
    active_nui: dict,
) -> tuple[dict, Path, dict]:
    """Test one optional physics/nuisance family and promote it only if useful.

    The mosaic-cone and dose families are scored on orbit cross-validated work rows
    (grond.workflows.heldout_gates); the other families keep the in-sample score.
    """

    if not feature_is_enabled(args, family):
        return active_nui, active_run, {"family": family, "status": "skipped", "reason": "no varied grid values"}
    if uses_heldout_gate(args, family):
        return run_heldout_feature_gate(
            prep,
            ds,
            args,
            python,
            logs,
            root,
            family=family,
            active_model=active_model,
            active_run=active_run,
            active_nui=active_nui,
        )
    baseline_summary = read_json(active_run / "summary.json")
    baseline_score = score_native_summary(baseline_summary, nuisance_score_rows(args))
    gate_dir = root / "runs" / ds.name / f"adaptive_{family}_gate"
    # Score the gate on the scale-fit split of its baseline so the free/all terms compare like with like.
    matched_split = (
        str((baseline_summary.get("config") or {}).get("scale_fit_split") or "") or None
        if str(getattr(args, "thickness_gate_split", "legacy")) == "matched"
        else None
    )
    try:
        run_logged(
            native_stage_cmd(
                python,
                ds,
                active_model,
                dataset_baseline_hkl(prep, ds),
                gate_dir,
                args,
                n_basis=args.screen_n_basis,
                n_points=stage_n_points(args, ds, screen=True),
                stage=f"adaptive_{family}_gate",
                pinned=feature_grid(args, family, active_nui, ds=ds),
                make_maps=False,
                endpoint=False,
                scale_fit_split_override=matched_split,
            ),
            logs,
            f"{ds.name}_adaptive_{family}_gate",
            dry_run=args.dry_run,
        )
        gate_summary = read_json(gate_dir / "summary.json")
        gate_split = str((gate_summary.get("config") or {}).get("scale_fit_split") or "")
        if matched_split and gate_split != matched_split:
            raise RuntimeError(f"gate scale-fit split {gate_split!r} differs from its baseline {matched_split!r}")
        gate_score = score_native_summary(gate_summary, nuisance_score_rows(args))
    except Exception as exc:
        return active_nui, active_run, {"family": family, "status": "failed", "error": str(exc), "run": str(gate_dir)}
    improvement = safe_float(baseline_score.get("score")) - safe_float(gate_score.get("score"))
    accepted = improvement > float(args.feature_accept_delta)
    scale_fit_split = matched_split or ("all" if getattr(args, "production_scales", False) else "work")
    record = {
        "family": family,
        "status": "accepted" if accepted else "rejected",
        "run": str(gate_dir),
        "baseline_score": baseline_score,
        "gate_score": gate_score,
        "score_improvement": improvement,
        "accept_delta": float(args.feature_accept_delta),
        "selection": "in_sample_score_native_summary",
        "scale_fit_split": scale_fit_split,
        # The gate stage ranks its grid on --selection-metric, then this score compares it with the baseline.
        "selection_metric": str(args.selection_metric),
        "free_rows_in_selection": not (
            nuisance_score_rows(args) == "work"
            and str(args.selection_metric) == "work_wR2"
            and scale_fit_split == "work"
        ),
    }
    if accepted:
        return best_nuisance(gate_dir) or active_nui, gate_dir, record
    return active_nui, active_run, record


def ensure_single_final_row_model_contract(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    source_run: Path,
    final_model: str,
    fallback_nuisance: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    """Bind final single-dataset rows to the exact accepted endpoint model.

    A native endpoint pass predicts rows from its input model and then writes an
    LS1-updated output model.  If that output is accepted, downstream row
    statistics, scale checks, calibration, and native maps require one
    endpoint-free recomputation from the accepted coordinates.
    """

    model = Path(final_model).expanduser().resolve()
    if not model.is_file():
        raise RuntimeError(f"accepted final model does not exist: {model}")
    model_sha = sha256_file(model)
    source_summary = read_json(source_run / "summary.json")
    source_inputs = source_summary.get("inputs") if isinstance(source_summary.get("inputs"), Mapping) else {}
    source_row = source_run / "native_dyn_calc.csv"
    source_model_sha = str(source_inputs.get("res_sha256") or "")
    if source_model_sha == model_sha and source_row.is_file():
        return source_run, {
            "status": "already_exact",
            "model": str(model),
            "model_sha256": model_sha,
            "row_run": str(source_run),
            "row_table": str(source_row),
            "definition": "native rows already bind the exact accepted model; no recomputation was needed",
        }

    source_config = source_summary.get("config") if isinstance(source_summary.get("config"), Mapping) else {}
    nuisance = best_nuisance(source_run) or dict(fallback_nuisance)
    n_basis = _positive_summary_int(source_config, "n_basis")
    n_points = _positive_summary_int(source_config, "n_points")
    scale_fit_split = str(source_config.get("scale_fit_split") or "work")
    if not nuisance or n_basis is None or n_points is None or scale_fit_split not in {"work", "all"}:
        raise RuntimeError(
            "cannot recompute final-model rows: source run lacks a complete nuisance, N/Q, or scale-fit-split contract"
        )
    recompute = root / "runs" / ds.name / "adaptive_final_row_recompute"
    recompute_args = _args_with_recorded_native_contract(args, source_config)
    run_logged(
        native_stage_cmd(
            python,
            ds,
            str(model),
            dataset_baseline_hkl(prep, ds),
            recompute,
            recompute_args,
            n_basis=n_basis,
            n_points=n_points,
            stage="adaptive_final_row_recompute",
            pinned=nuisance,
            make_maps=bool(args.make_maps),
            endpoint=False,
            scale_fit_split_override=scale_fit_split,
            strict_pinned_contract=True,
            allow_pinned_jitter_fallback=False,
        ),
        logs,
        f"{ds.name}_adaptive_final_row_recompute",
        dry_run=False,
    )
    recompute_summary = read_json(recompute / "summary.json")
    recompute_inputs = recompute_summary.get("inputs") if isinstance(recompute_summary.get("inputs"), Mapping) else {}
    recompute_row = recompute / "native_dyn_calc.csv"
    if str(recompute_inputs.get("res_sha256") or "") != model_sha or not recompute_row.is_file():
        raise RuntimeError(
            "final row recomputation did not bind the accepted model hash and materialize native_dyn_calc.csv"
        )
    return recompute, {
        "status": "recomputed",
        "model": str(model),
        "model_sha256": model_sha,
        "source_run": str(source_run),
        "source_model_sha256": source_model_sha,
        "row_run": str(recompute),
        "row_table": str(recompute_row),
        "definition": "fixed-nuisance native rows for the accepted LS1 output model; no additional coordinate step",
    }


def run_single_adaptive(prep: PrepConfig, args: argparse.Namespace, python: str, logs: Path) -> dict:
    """Run the full automatic staged path for one dataset."""

    root = Path(args.out_dir)
    ds0 = prep.datasets[0]
    args = crystal_arguments(args, ds0.name)
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
    if preflight.status == "bad" and args.stop_on_bad_preflight:
        return {"status": "stopped_bad_preflight", "preflight": str(root / "decisions" / ds0.name / "preflight.json")}

    ds, transform_decision = choose_transform_by_scout(prep, ds, args, python, logs, root, scout_max_rows=scout_rows)
    configure_cell_esd(prep.structure, [ds], args)
    screen = root / "runs" / ds.name / "adaptive_screen"
    run_logged(
        native_stage_cmd(
            python,
            ds,
            prep.structure,
            dataset_baseline_hkl(prep, ds),
            screen,
            args,
            n_basis=20 if args.fast else args.screen_n_basis,
            n_points=5 if args.fast else args.screen_n_points,
            stage="adaptive_screen",
            pinned=compact_base_nuisance(args),
            make_maps=False,
            endpoint=False,
        ),
        logs,
        f"{ds.name}_adaptive_screen",
        dry_run=args.dry_run,
    )
    screen_summary = read_json(screen / "summary.json")
    screen_frame_offsets = resolved_frame_offset_candidates(screen)
    stop, reasons = should_stop_for_bad_data(
        screen_summary, max_wR2=args.stop_if_wR2_above, min_improvement=args.min_bloch_wR2_gain
    )
    if stop and args.stop_on_bad_fit:
        write_json(
            root / "decisions" / ds.name / "stop_decision.json",
            {"stop": True, "reasons": reasons, "screen": str(screen)},
        )
        return {"status": "stopped_bad_fit", "primary_run": str(screen), "reasons": reasons}
    # Validation console: the split and the screen's held-out R, printed from their recorded JSON.
    single_prep = PrepConfig(structure=prep.structure, datasets=[ds], baseline_hkl=prep.baseline_hkl)
    report_split(single_prep, args, {ds.name: screen}, root)
    report_stage_heldout(args, {ds.name: screen}, root, label="screen", model=prep.structure)

    active_model = prep.structure
    active_nui = best_nuisance(screen)
    absolute_structure_decision = absolute_structure_policy(prep.structure)
    active_hand = "original"
    hand_settings = hand_decision_settings(args)
    hand_absorption_label = (
        "uniform_absorption_kept_exact_pair"
        if hand_settings["pair_absorption"] == "uniform"
        else "disabled_for_hand_decision_only"
    )
    early_hand = bool(
        args.hand_control
        and absolute_structure_decision.get("run_hand_control", True)
        and str(getattr(args, "hand_decision_stage", "final")) == "early"
    )
    early_hand_decision: dict = {}
    if early_hand:
        # Held-out hand decision on the hand-blind start model, before thickness, gates and refinement.
        early_hand_decision = run_early_elastic_hand_selection(
            PrepConfig(
                structure=prep.structure,
                datasets=[ds],
                baseline_hkl=prep.baseline_hkl,
                notes=prep.notes,
                default_hkl_transform=prep.default_hkl_transform,
                geometry_percentile=prep.geometry_percentile,
                parity_setup_id=prep.parity_setup_id,
                parity_setup_manifest_sha256=prep.parity_setup_manifest_sha256,
                metadata=prep.metadata,
            ),
            args,
            python,
            logs,
            root,
            start_model=prep.structure,
            screen_runs={ds.name: screen},
            shared_hand_decision={"symmetry_policy": absolute_structure_decision},
            run_prefix="adaptive",
        )
        if str(early_hand_decision.get("status") or "") not in {"kept", "switched", "skipped_dry_run"}:
            print(
                f"WARN hand {ds.name}: early held-out hand decision {early_hand_decision.get('status')!r}; continuing "
                "in the input hand, the final elastic pair decides with the same rule",
                flush=True,
            )
        if bool(early_hand_decision.get("hand_switched", False)):
            active_model = str(early_hand_decision["selected_model_after"])
            active_hand = str(early_hand_decision["selected_hand_after"])
            # Re-screen in the chosen hand so the thickness seed is not chosen in the rejected one.
            screen = root / "runs" / ds.name / "adaptive_screen_selected_hand"
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    active_model,
                    dataset_baseline_hkl(prep, ds),
                    screen,
                    args,
                    n_basis=20 if args.fast else args.screen_n_basis,
                    n_points=5 if args.fast else args.screen_n_points,
                    stage="adaptive_screen",
                    pinned=compact_base_nuisance(args),
                    make_maps=False,
                    endpoint=False,
                ),
                logs,
                f"{ds.name}_adaptive_screen_selected_hand",
                dry_run=args.dry_run,
            )
            screen_frame_offsets = resolved_frame_offset_candidates(screen)
            active_nui = best_nuisance(screen)
            report_stage_heldout(args, {ds.name: screen}, root, label="screen_selected_hand", model=active_model)
    if early_hand:
        absolute_structure_decision.update(
            {
                "status": "early_selected_final_confirmation_pending",
                "selected": active_hand,
                "selection_stage": "early_elastic_pair_after_screen_then_final_confirmation",
                "reason": (
                    "hand selected on held-out evidence right after the screen on the hand-blind start model; "
                    "the final elastic pair confirms it with the same rule"
                ),
                "absorption_policy": hand_absorption_label,
                "decision_evidence": hand_settings["evidence"],
                "early_elastic_selection": early_hand_decision,
            }
        )
    elif args.hand_control and absolute_structure_decision.get("run_hand_control", True):
        absolute_structure_decision.update(
            {
                "status": "deferred_until_final_model",
                "selected": "original",
                "selection_stage": "final_elastic_negative_thickness_pair",
                "reason": (
                    "hand selection is deferred until thickness, geometry, nuisance, and structure refinement are "
                    "complete; the final elastic pair shares one eigendecomposition and does not rerun nuisance grids"
                ),
                "absorption_policy": hand_absorption_label,
                "decision_evidence": hand_settings["evidence"],
            }
        )

    regularized_nui = active_nui
    thickness_run = screen
    feature_records: list[dict] = []
    if nested_selection_enabled(args):
        # Cone x dose x thickness distribution x lambda jointly by inner orbit CV on work rows; the in-sample
        # gates then run on the pinned pick (--nuisance-selection sequential: the stages below).
        regularized_nui, feature_source, nested_record = run_nested_nuisance_stage(
            prep, ds, args, python, logs, root, active_model=active_model, active_nui=active_nui, run_prefix="adaptive"
        )
        feature_records.append(nested_record)
        for family in FEATURE_GATE_FAMILIES:
            if family in NESTED_GATE_FAMILIES:
                feature_records.append(nested_gate_skip_record(family))
                continue
            regularized_nui, feature_source, record = run_feature_gate(
                prep,
                ds,
                args,
                python,
                logs,
                root,
                family=family,
                active_model=active_model,
                active_run=feature_source,
                active_nui=regularized_nui,
            )
            feature_records.append(record)
        thickness_run = feature_source
    elif not args.fast:
        thickness_run = root / "runs" / ds.name / "adaptive_thickness_polish"
        thickness_baseline = discrete_thickness_baseline_nuisance(args, active_nui)
        thickness_candidates = thickness_polish_nuisance(thickness_baseline, screen_frame_offsets)
        if bool(getattr(args, "pets_static_auto", False)):
            thickness_candidates["frame_offset"] = "0"
        run_logged(
            native_stage_cmd(
                python,
                ds,
                active_model,
                dataset_baseline_hkl(prep, ds),
                thickness_run,
                args,
                n_basis=args.screen_n_basis,
                n_points=args.screen_n_points,
                stage="adaptive_thickness_polish",
                pinned=thickness_candidates,
                promotion_baseline=thickness_baseline,
                scale_fit_split_override="work",
                selection_metric_override="work_wR2",
                make_maps=False,
                endpoint=False,
            ),
            logs,
            f"{ds.name}_adaptive_thickness_polish",
            dry_run=args.dry_run,
        )
        regularized_nui = best_nuisance(thickness_run) or active_nui
        feature_source = thickness_run
        feature_records.append(discrete_thickness_promotion_record(thickness_run, dry_run=bool(args.dry_run)))
        continuous_seed = continuous_thickness_seed_nuisance(args, regularized_nui, active_nui)
        if continuous_thickness_enabled(args) and continuous_seed:
            continuous_baseline_run = feature_source
            continuous_run = root / "runs" / ds.name / "adaptive_thickness_continuous"
            run_logged(
                continuous_thickness_cmd(
                    python,
                    ds,
                    active_model,
                    continuous_run,
                    args,
                    continuous_seed,
                    n_basis=args.screen_n_basis,
                    n_points=args.screen_n_points,
                    stage="adaptive_thickness_continuous",
                ),
                logs,
                f"{ds.name}_adaptive_thickness_continuous",
                dry_run=args.dry_run,
            )
            regularized_nui = best_nuisance(continuous_run) or regularized_nui or continuous_seed
            regularized_nui, feature_source, materialization_record = materialize_continuous_nuisance_run(
                prep,
                ds,
                args,
                python,
                logs,
                active_model=active_model,
                baseline_run=continuous_baseline_run,
                continuous_run=continuous_run,
                selected_nuisance=regularized_nui,
                n_basis=args.screen_n_basis,
                n_points=args.screen_n_points,
                stage="adaptive_thickness_continuous_materialized",
            )
            feature_records.append(materialization_record)
        for family in FEATURE_GATE_FAMILIES:
            regularized_nui, feature_source, record = run_feature_gate(
                prep,
                ds,
                args,
                python,
                logs,
                root,
                family=family,
                active_model=active_model,
                active_run=feature_source,
                active_nui=regularized_nui,
            )
            feature_records.append(record)
        thickness_run = feature_source
        if console_enabled(args):
            # The same records the decision tree carries, written now so the console prints the recorded values.
            write_json(root / "decisions" / ds.name / "feature_decisions.json", feature_records)
            report_feature_decisions(
                ds.name, root / "decisions" / ds.name / "feature_decisions.json", args, nuisance=regularized_nui
            )

    final_model = active_model
    final_n_basis = 20 if args.fast else int(args.final_basis_fallback)
    structure_strategy = resolve_structure_update_strategy(args, multi_dataset=False)
    structure_update_run = None
    structure_feedback_fallback: dict = {}
    feedback_final_run: Path | None = None
    if args.mode in {"regularize", "auto"} and not args.fast and structure_strategy == "shelxl_feedback":
        final_n_basis = resolve_final_n_basis(
            prep, ds, args, python, logs, root, active_model, "adaptive_shelxl_feedback"
        )
        struct_dir = root / "runs" / ds.name / "adaptive_shelxl_feedback"
        source_summary = thickness_run / "summary.json"
        run_logged(
            shelxl_feedback_cmd(
                python,
                ds,
                active_model,
                source_summary,
                dataset_baseline_hkl(prep, ds),
                struct_dir,
                args,
                n_basis=final_n_basis,
            )
            + nested_feedback_args(args, root, ds),
            logs,
            f"{ds.name}_adaptive_shelxl_feedback",
            dry_run=args.dry_run,
        )
        structure_update_run = struct_dir
        report_macro_cycles(struct_dir / "shelxl_feedback_loop_summary.json", args)
        if args.dry_run:
            feedback_final_run = (
                struct_dir
                / f"cycle_{int(args.shelxl_feedback_cycles):02d}_N{int(final_n_basis)}Q{resolved_final_n_points(args, ds)}"
            )
        else:
            feedback_final_run, structure_feedback_fallback = feedback_loop_outcome(struct_dir, ds.name, final_model)
            feedback_model = shelxl_feedback_current_model(struct_dir)
            if feedback_model is not None:
                final_model = str(feedback_model)

    if feedback_final_run is not None:
        final = feedback_final_run
    else:
        final = root / "runs" / ds.name / ("adaptive_fast_final" if args.fast else "adaptive_final")
        final_n_basis = resolve_final_n_basis(prep, ds, args, python, logs, root, final_model, "adaptive_final")
        run_logged(
            native_stage_cmd(
                python,
                ds,
                final_model,
                dataset_baseline_hkl(prep, ds),
                final,
                args,
                n_basis=final_n_basis,
                n_points=5 if args.fast else args.n_points,
                stage="adaptive_final",
                pinned=regularized_nui,
                make_maps=args.make_maps,
                endpoint=True,
            ),
            logs,
            f"{ds.name}_adaptive_final",
            dry_run=args.dry_run,
        )
    endpoint_feedback_decision: dict = {}
    endpoint_feedback_run = None
    if args.mode in {"regularize", "auto"} and not args.fast and not args.dry_run and feedback_final_run is None:
        run_feedback, endpoint_feedback_decision = should_run_endpoint_feedback(final, args)
        endpoint_model = best_endpoint_model(final)
        if run_feedback and endpoint_model is not None:
            endpoint_feedback_run = root / "runs" / ds.name / "adaptive_endpoint_feedback"
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    str(endpoint_model),
                    dataset_baseline_hkl(prep, ds),
                    endpoint_feedback_run,
                    args,
                    n_basis=final_n_basis,
                    n_points=args.n_points,
                    stage="adaptive_endpoint_feedback",
                    pinned=best_nuisance(final) or regularized_nui,
                    make_maps=args.make_maps,
                    endpoint=True,
                ),
                logs,
                f"{ds.name}_adaptive_endpoint_feedback",
                dry_run=args.dry_run,
            )
            final = endpoint_feedback_run
            final_model = str(best_endpoint_model(endpoint_feedback_run) or endpoint_model)
        elif run_feedback:
            endpoint_feedback_decision = {
                **endpoint_feedback_decision,
                "status": "promoted_but_no_endpoint_model_found",
            }
    absolute_structure_diagnostic = ""
    post_feedback_hand_promotion: dict = {}
    if args.hand_control and absolute_structure_decision.get("run_hand_control", True) and not args.dry_run:
        hand_input_run, hand_input_model, hand_input_representation = final, final_model, active_hand
        try:
            hand_pair = root / "runs" / ds.name / "adaptive_final_elastic_hand_pair"
            final_summary = read_json(final / "summary.json")
            final_config = final_summary.get("config") if isinstance(final_summary.get("config"), Mapping) else {}
            hand_nuisance, source_absorption = elastic_hand_nuisance(
                best_nuisance(final) or regularized_nui, absorption_policy=str(hand_settings["pair_absorption"])
            )
            hand_args = _args_with_recorded_native_contract(args, final_config)
            hand_n_basis = _positive_summary_int(final_config, "n_basis") or (20 if args.fast else final_n_basis)
            hand_n_points = _positive_summary_int(final_config, "n_points") or (5 if args.fast else args.n_points)
            scale_fit_split = str(final_config.get("scale_fit_split") or "work")
            if scale_fit_split not in {"work", "all"}:
                scale_fit_split = "work"
            if hand_settings["pair_scale_fit_split"] == "work":
                # Both hands' pair scales exclude the free orbits.
                scale_fit_split = "work"
            run_logged(
                native_stage_cmd(
                    python,
                    ds,
                    final_model,
                    dataset_baseline_hkl(prep, ds),
                    hand_pair,
                    hand_args,
                    n_basis=hand_n_basis,
                    n_points=hand_n_points,
                    stage="hand",
                    pinned=hand_nuisance,
                    make_maps=False,
                    endpoint=False,
                    scale_fit_split_override=scale_fit_split,
                    strict_pinned_contract=True,
                    allow_pinned_jitter_fallback=False,
                    paired_inverse_hand=True,
                ),
                logs,
                f"{ds.name}_adaptive_final_elastic_hand_pair",
                dry_run=args.dry_run,
            )
            stats_path = root / "runs" / ds.name / "adaptive_absolute_structure_diagnostic.json"
            run_logged(
                [
                    python,
                    str(engine_script("absolute_structure_diagnostic.py")),
                    "--best",
                    str(hand_pair / "native_dyn_calc.csv"),
                    "--other",
                    str(hand_pair / "native_inverse_hand_dyn_calc.csv"),
                    "--out",
                    str(stats_path),
                ],
                logs,
                f"{ds.name}_adaptive_absolute_structure_diagnostic",
                dry_run=args.dry_run,
            )
            absolute_structure_diagnostic = str(stats_path)
            pair_summary = read_json(hand_pair / "summary.json")
            pair_contract = (
                pair_summary.get("paired_inverse_hand")
                if isinstance(pair_summary.get("paired_inverse_hand"), Mapping)
                else {}
            )
            final_score = score_native_metrics(
                pair_summary,
                pair_contract.get("selected_metrics")
                if isinstance(pair_contract.get("selected_metrics"), Mapping)
                else {},
            )
            other_score = score_native_metrics(
                pair_summary,
                pair_contract.get("inverse_metrics")
                if isinstance(pair_contract.get("inverse_metrics"), Mapping)
                else {},
            )
            post_feedback_delta = final_score.get("score", float("inf")) - other_score.get("score", float("inf"))
            hand_stats = try_read_json(stats_path) or {}
            hand_row_contract = _paired_row_split_contract(
                hand_pair / "native_dyn_calc.csv",
                hand_pair / "native_inverse_hand_dyn_calc.csv",
                nullable_numeric_fields=_row_contract_nullable_numeric_fields(final_config, hand_nuisance),
            )
            hand_calculation_errors = [
                *list(hand_row_contract.get("errors") or []),
                *list(hand_stats.get("calculation_errors") or []),
            ]
            if not hand_row_contract.get("passed"):
                hand_stats = {
                    **hand_stats,
                    "calculation_status": "error",
                    "calculation_errors": hand_calculation_errors,
                }
            raw_all_zkp = safe_float(
                _adjusted_hand_z_block(hand_stats, "all").get("z"),
                float("nan"),
            )
            raw_free_zkp = safe_float(
                _adjusted_hand_z_block(hand_stats, "free").get("z"),
                float("nan"),
            )
            hand_flip_threshold = float(getattr(args, "hand_flip_zkp_all", 3.5))
            heldout_mode = hand_settings["evidence"] == "heldout-orbit-sign"
            hand_active_before = active_hand
            heldout_payload = heldout_hand_decision(
                {
                    ds.name: {
                        "status": (
                            "complete"
                            if hand_row_contract.get("passed") and hand_stats.get("status") == "ok"
                            else "calculation_error"
                        ),
                        "selected_row_table": str(hand_pair / "native_dyn_calc.csv"),
                        "inverse_row_table": str(hand_pair / "native_inverse_hand_dyn_calc.csv"),
                        "pair_run": str(hand_pair),
                    }
                },
                args,
                stage="final",
                model=final_model,
                settings=hand_settings,
                expected_datasets=1,
            )
            post_feedback_hand_promotion = {
                **(
                    _heldout_preference(heldout_payload, active_hand=active_hand)
                    if heldout_mode
                    else hand_preference_summary(
                        raw_all_zkp,
                        active_hand=active_hand,
                        threshold=hand_flip_threshold,
                        calculation_valid=not hand_calculation_errors and hand_stats.get("status") == "ok",
                    )
                ),
                "status": "not_promoted",
                "method": "nakane_negative_thickness_elastic_pair_v1",
                "active_hand": active_hand,
                "opposite_hand": "alternate" if active_hand == "original" else "original",
                "final_score": final_score,
                "opposite_score": other_score,
                "score_delta_final_minus_opposite": post_feedback_delta,
                "accept_delta": float(args.hand_accept_delta),
                "diagnostic": str(stats_path),
                "paired_run": str(hand_pair),
                "source_absorption_nuisance": source_absorption,
                "hand_decision_absorption": (
                    "uniform_kept" if hand_settings["pair_absorption"] == "uniform" else "disabled"
                ),
                "hand_pair_scale_fit_split": scale_fit_split,
                "decision_evidence": hand_settings["evidence"],
                "heldout": {
                    key: heldout_payload.get(key)
                    for key in ("status", "pooled", "rule", "reasons", "errors", "supports_switch", "heldout_scheme")
                },
                "heldout_decision_artifact": str(root / "decisions" / "hand_decision_heldout_final.json"),
                "shared_eigendecomposition": True,
                "nuisance_refit": False,
                "jitter_refit": False,
                "adjusted_z_before_selection": {
                    "all": raw_all_zkp,
                    "free": raw_free_zkp,
                    "positive_favors_active_hand": True,
                },
                "hand_flip_zkp_all_threshold": hand_flip_threshold,
                "row_pair_contract": hand_row_contract,
                "calculation_errors": hand_calculation_errors,
                "matched_all_row_metrics": (hand_stats.get("all") or {}).get("matched_prediction_metrics", {}),
                "matched_born_control": hand_stats.get("matched_born_control", {}),
            }
            legacy_gate_ok, legacy_gate_reasons = hand_diagnostic_supports_other(
                hand_stats,
                args,
            )
            if heldout_mode:
                diagnostic_gate_ok = bool(heldout_payload.get("supports_switch") and not hand_calculation_errors)
                diagnostic_gate_reasons = [
                    *list(heldout_payload.get("reasons") or []),
                    *list(heldout_payload.get("errors") or []),
                ]
            else:
                diagnostic_gate_ok, diagnostic_gate_reasons = legacy_gate_ok, legacy_gate_reasons
            post_feedback_hand_promotion["legacy_all_row_zkp_rule_would_switch"] = bool(legacy_gate_ok)
            post_feedback_hand_promotion["diagnostic_quality_gate_status"] = (
                "switch_supported" if diagnostic_gate_ok else "keep_current"
            )
            post_feedback_hand_promotion["diagnostic_quality_gate_reasons"] = diagnostic_gate_reasons
            if diagnostic_gate_ok and is_shelx_model(final_model):
                promoted_model = make_alternate_hand_model(python, final_model, root, logs, args.dry_run)
                promoted = root / "runs" / ds.name / "adaptive_promoted_hand_final"
                run_logged(
                    native_stage_cmd(
                        python,
                        ds,
                        str(promoted_model),
                        dataset_baseline_hkl(prep, ds),
                        promoted,
                        args,
                        n_basis=20 if args.fast else final_n_basis,
                        n_points=5 if args.fast else args.n_points,
                        stage="adaptive_promoted_hand_final",
                        pinned=best_nuisance(final) or regularized_nui,
                        make_maps=args.make_maps,
                        endpoint=True,
                        strict_pinned_contract=True,
                        allow_pinned_jitter_fallback=False,
                    ),
                    logs,
                    f"{ds.name}_adaptive_promoted_hand_final",
                    dry_run=args.dry_run,
                )
                post_feedback_hand_promotion.update(
                    {
                        "status": "promoted",
                        "promoted_run": str(promoted),
                        "promoted_model_input": str(promoted_model),
                        "promotion_policy": (
                            "the matched final elastic hand decision is authoritative; the selected coordinate hand is "
                            "materialized once under the already selected full nuisance contract"
                        ),
                    }
                )
                final = promoted
                final_model = str(best_endpoint_model(promoted) or promoted_model)
                active_hand = post_feedback_hand_promotion["opposite_hand"]
                post_feedback_hand_promotion["adjusted_z_final_hand_favored"] = {
                    "all": -raw_all_zkp if math.isfinite(raw_all_zkp) else float("nan"),
                    "free": -raw_free_zkp if math.isfinite(raw_free_zkp) else float("nan"),
                    "positive_favors_final_selected_hand": True,
                }
                post_feedback_hand_promotion["warnings"] = _hand_evidence_warnings(
                    all_z=raw_all_zkp,
                    free_z=raw_free_zkp,
                    switched=True,
                    threshold=hand_flip_threshold,
                )
                absolute_structure_decision = {
                    **absolute_structure_decision,
                    "status": "complete",
                    "post_feedback_selected": active_hand,
                    "post_feedback_selected_hand": active_hand,
                    "post_feedback_promotion": post_feedback_hand_promotion,
                }
            else:
                post_feedback_hand_promotion["adjusted_z_final_hand_favored"] = {
                    "all": raw_all_zkp,
                    "free": raw_free_zkp,
                    "positive_favors_final_selected_hand": True,
                }
                post_feedback_hand_promotion["warnings"] = _hand_evidence_warnings(
                    all_z=raw_all_zkp,
                    free_z=raw_free_zkp,
                    switched=False,
                    threshold=hand_flip_threshold,
                )
                absolute_structure_decision = {
                    **absolute_structure_decision,
                    "status": "complete",
                    "post_feedback_selected": active_hand,
                    "post_feedback_selected_hand": active_hand,
                    "post_feedback_promotion": post_feedback_hand_promotion,
                }

            if hand_calculation_errors:
                post_feedback_hand_promotion["status"] = "calculation_error"
                post_feedback_hand_promotion["warnings"].extend(
                    f"Hand calculation error: {error}" for error in hand_calculation_errors
                )
                absolute_structure_decision["status"] = "hand_calculation_error"
            elif heldout_mode and heldout_payload.get("status") != "complete":
                post_feedback_hand_promotion["status"] = "heldout_evidence_error"
                post_feedback_hand_promotion["warnings"].extend(
                    f"held-out hand evidence: {error}" for error in heldout_payload.get("errors") or []
                )
                absolute_structure_decision["status"] = "hand_heldout_evidence_error"
            if heldout_mode and heldout_payload.get("status") == "complete" and legacy_gate_ok != diagnostic_gate_ok:
                post_feedback_hand_promotion["warnings"].append(
                    "the legacy all-row ZKP rule and the held-out orbit rule disagree; the held-out rule decides"
                )
            hand_agreement = (
                _early_final_agreement(
                    {
                        "selected_hand": hand_active_before,
                        "switched": bool(early_hand_decision.get("hand_switched")),
                        "escalation_status": (early_hand_decision.get("escalation") or {}).get("status"),
                        "decided_on": (early_hand_decision.get("escalation") or {}).get("decided_on"),
                    },
                    heldout_payload,
                    switched=active_hand != hand_active_before,
                )
                if early_hand
                else None
            )
            if hand_agreement is not None:
                post_feedback_hand_promotion["early_final_agreement"] = hand_agreement
                if hand_agreement.get("status") != "confirmed":
                    message = f"early/final hand decisions: {hand_agreement['status']}"
                    post_feedback_hand_promotion["warnings"].append(message)
                    print(f"WARN hand {ds.name}: {message}", flush=True)
            write_heldout_hand_decision(
                root / "decisions" / "hand_decision_heldout_final.json",
                heldout_payload,
                selected_hand_before=hand_active_before,
                selected_hand_after=active_hand,
                switched=active_hand != hand_active_before,
                selected_model_after=final_model,
                legacy_all_row_zkp={
                    "z_positive_favors_selected": raw_all_zkp,
                    "threshold": hand_flip_threshold,
                    "would_switch": bool(legacy_gate_ok),
                },
                agreement=hand_agreement,
            )
            report_hand_decision(root / "decisions" / "hand_decision_heldout_final.json", args)
            post_feedback_hand_promotion["selected_hand_after"] = active_hand
        except Exception as exc:
            final, final_model, active_hand = hand_input_run, hand_input_model, hand_input_representation
            error = f"{type(exc).__name__}: {exc}"
            post_feedback_hand_promotion = {
                "status": "calculation_error",
                "stage": "final_elastic_hand",
                "preferred_hand": "unavailable",
                "hand_confidence": "calculation_error",
                "selected_hand_after": active_hand,
                "selected_model_after": final_model,
                "calculation_errors": [error],
                "warnings": [
                    f"Final hand calculation failed: {error}; the preceding usable structural endpoint is retained."
                ],
                "source_run": str(final),
                "logs": str(logs),
                "primary_evidence": "all_rows",
                "free_evidence_role": "diagnostic_warning_only",
            }
            absolute_structure_decision = {
                **absolute_structure_decision,
                "status": "hand_calculation_error",
                "post_feedback_selected_hand": active_hand,
                "post_feedback_promotion": post_feedback_hand_promotion,
            }
            write_json(root / "runs" / ds.name / "adaptive_hand_calculation_error.json", post_feedback_hand_promotion)
            # The final held-out contract is written even when the check could not run.
            failed_final = {**heldout_contract_base("final", final_model, hand_settings), "status": "error", "errors": [error]}
            hand_agreement = (
                _early_final_agreement(
                    {
                        "selected_hand": hand_input_representation,
                        "switched": bool(early_hand_decision.get("hand_switched")),
                        "escalation_status": (early_hand_decision.get("escalation") or {}).get("status"),
                        "decided_on": (early_hand_decision.get("escalation") or {}).get("decided_on"),
                    },
                    failed_final,
                    switched=False,
                )
                if early_hand
                else None
            )
            if hand_agreement is not None:
                post_feedback_hand_promotion["early_final_agreement"] = hand_agreement
                print(f"WARN hand {ds.name}: early/final hand decisions: {hand_agreement['status']}", flush=True)
            write_heldout_hand_decision(
                root / "decisions" / "hand_decision_heldout_final.json",
                failed_final,
                selected_hand_before=hand_input_representation,
                selected_hand_after=active_hand,
                switched=False,
                selected_model_after=final_model,
                legacy_all_row_zkp={},
                agreement=hand_agreement,
            )
            report_hand_decision(root / "decisions" / "hand_decision_heldout_final.json", args)

    if args.dry_run:
        final_row_model_contract = {
            "status": "skipped_dry_run",
            "model": str(final_model),
            "row_run": str(final),
        }
    else:
        final, final_row_model_contract = ensure_single_final_row_model_contract(
            prep,
            ds,
            args,
            python,
            logs,
            root,
            source_run=final,
            final_model=final_model,
            fallback_nuisance=regularized_nui,
        )
    if post_feedback_hand_promotion.get("status") == "promoted":
        post_feedback_hand_promotion["final_row_model_contract"] = final_row_model_contract

    no_h_run = None
    if args.h_omit in {"on", "both", "only"} and is_shelx_model(final_model):
        no_h = make_noh_model(python, final_model, root, logs, args.dry_run)
        no_h_run = root / "runs" / ds.name / "adaptive_final_noH"
        run_logged(
            native_stage_cmd(
                python,
                ds,
                str(no_h),
                dataset_baseline_hkl(prep, ds),
                no_h_run,
                args,
                n_basis=final_n_basis,
                n_points=5 if args.fast else args.n_points,
                stage="adaptive_noH",
                pinned=best_nuisance(final) or regularized_nui,
                make_maps=args.make_maps,
                endpoint=True,
            ),
            logs,
            f"{ds.name}_adaptive_final_noH",
            dry_run=args.dry_run,
        )

    scale_polish = run_scale_window_polish(
        python,
        final,
        root / "scale_polish" / ds.name,
        logs,
        f"{ds.name}_adaptive_final",
        args,
    )
    resolution_limit = run_resolution_limit_gate(
        python,
        {ds.name: final},
        final_model,
        root / "resolution_limits" / ds.name,
        logs,
        ds.name,
        args,
        title=f"GROND resolution-limit promotion: {ds.name}",
        scope="dataset",
    )
    sg_window_scout = run_sg_window_scout(
        python,
        prep,
        ds,
        final_model,
        final,
        root / "sg_window_scout" / ds.name,
        logs,
        f"{ds.name}_adaptive_final",
        args,
    )
    decision_summary = {
        "status": "ok",
        "dataset": ds.name,
        "selected_transform": ds.hkl_transform,
        "selected_transform_claim_contract": (
            transform_decision.get("selected_transform_claim_contract")
            if isinstance(transform_decision, Mapping)
            else _selected_transform_claim_contract(
                ds.hkl_transform,
                selection_source="decision_tree_fallback",
                parity_policy=str(getattr(args, "transform_parity_policy", "proper-only")),
                source_hkl=ds.hkl,
            )
        ),
        "selected_hand": active_hand,
        "structure_update_strategy": structure_strategy,
        "structure_update_run": str(structure_update_run) if structure_update_run else "",
        "transform_decision": transform_decision,
        "absolute_structure_decision": absolute_structure_decision,
        "feature_decisions": feature_records,
        "endpoint_feedback_decision": endpoint_feedback_decision,
        "structure_feedback_fallback": structure_feedback_fallback,
        "endpoint_feedback_run": str(endpoint_feedback_run) if endpoint_feedback_run else "",
        "post_feedback_hand_promotion": post_feedback_hand_promotion,
        "final_row_model_contract": final_row_model_contract,
        "scale_polish": scale_polish,
        "resolution_limit": resolution_limit,
        "sg_window_scout": sg_window_scout,
        "screen_run": str(screen),
        "thickness_run": str(thickness_run),
        "final_run": str(final),
        "final_model": final_model,
        "preflight": str(root / "decisions" / ds0.name / "preflight.json"),
    }
    write_json(root / "decisions" / ds.name / "decision_tree.json", decision_summary)
    return {
        "primary_run": str(final),
        "no_h_run": str(no_h_run) if no_h_run else "",
        "model": final_model,
        "absolute_structure_diagnostic": absolute_structure_diagnostic,
        "post_feedback_hand_promotion": post_feedback_hand_promotion,
        "final_row_model_contract": final_row_model_contract,
        "scale_polish": scale_polish,
        "resolution_limits": {ds.name: resolution_limit},
        "sg_window_scout": {ds.name: sg_window_scout},
        "decision_tree": str(root / "decisions" / ds.name / "decision_tree.json"),
    }


def run_single(prep: PrepConfig, args: argparse.Namespace, python: str, logs: Path) -> dict:
    """Run either the automatic or fixed single-dataset workflow."""

    configure_auto_physics(prep, args)
    args = crystal_arguments(args, prep.datasets[0].name)
    if args.auto_decide:
        return run_single_adaptive(prep, args, python, logs)
    ds = prep.datasets[0]
    root = Path(args.out_dir)
    make_maps = args.make_maps
    n_basis = 20 if args.fast else args.screen_n_basis
    n_points = 5 if args.fast else args.screen_n_points
    configure_cell_esd(prep.structure, [ds], args)
    screen = root / "runs" / ds.name / ("fast_N20_Q5" if args.fast else "screen")
    run_logged(
        native_stage_cmd(
            python,
            ds,
            prep.structure,
            dataset_baseline_hkl(prep, ds),
            screen,
            args,
            n_basis=n_basis,
            n_points=n_points,
            stage="screen",
            make_maps=make_maps and args.mode == "bloch",
        ),
        logs,
        f"{ds.name}_bloch",
        dry_run=args.dry_run,
    )
    nui = best_nuisance(screen)
    if args.mode in {"regularize", "auto"} and not args.fast and not args.dry_run and not nui:
        raise RuntimeError(f"No best nuisance result found in {screen}; refusing to launch an unpinned final grid.")
    final_model = prep.structure
    final = screen
    absolute_structure_diagnostic = ""
    if args.mode in {"regularize", "auto"} and not args.fast:
        final = root / "runs" / ds.name / "final"
        final_n_basis = resolve_final_n_basis(prep, ds, args, python, logs, root, final_model, "final")
        run_logged(
            native_stage_cmd(
                python,
                ds,
                final_model,
                dataset_baseline_hkl(prep, ds),
                final,
                args,
                n_basis=final_n_basis,
                n_points=args.n_points,
                stage="final",
                pinned=nui,
                make_maps=make_maps,
            ),
            logs,
            f"{ds.name}_final",
            dry_run=args.dry_run,
        )
    hand_policy = absolute_structure_policy(final_model)
    if args.hand_control and hand_policy.get("run_hand_control", True) and not args.dry_run:
        hand_run = root / "runs" / ds.name / "hand_alt"
        run_logged(
            native_stage_cmd(
                python,
                ds,
                final_model,
                dataset_baseline_hkl(prep, ds),
                hand_run,
                args,
                n_basis=20 if args.fast else args.hand_n_basis,
                n_points=5 if args.fast else args.hand_n_points,
                stage="hand",
                pinned=best_nuisance(final) or nui,
                alternate_hand=True,
                make_maps=False,
            ),
            logs,
            f"{ds.name}_hand_alt",
            dry_run=args.dry_run,
        )
        stats_path = root / "runs" / ds.name / "absolute_structure_diagnostic.json"
        run_logged(
            [
                python,
                str(engine_script("absolute_structure_diagnostic.py")),
                "--best",
                str(final / "native_dyn_calc.csv"),
                "--other",
                str(hand_run / "native_dyn_calc.csv"),
                "--out",
                str(stats_path),
            ],
            logs,
            f"{ds.name}_absolute_structure_diagnostic",
            dry_run=args.dry_run,
        )
        absolute_structure_diagnostic = str(stats_path)
    no_h_run = None
    if args.h_omit in {"on", "both", "only"} and is_shelx_model(final_model):
        no_h = make_noh_model(python, final_model, root, logs, args.dry_run)
        no_h_run = root / "runs" / ds.name / "final_noH"
        no_h_n_basis = resolve_final_n_basis(prep, ds, args, python, logs, root, str(no_h), "final_noH")
        run_logged(
            native_stage_cmd(
                python,
                ds,
                str(no_h),
                dataset_baseline_hkl(prep, ds),
                no_h_run,
                args,
                n_basis=20 if args.fast else no_h_n_basis,
                n_points=5 if args.fast else args.n_points,
                stage="noH",
                pinned=best_nuisance(final) or nui,
                make_maps=make_maps,
            ),
            logs,
            f"{ds.name}_final_noH",
            dry_run=args.dry_run,
        )
    scale_polish = run_scale_window_polish(
        python,
        final,
        root / "scale_polish" / ds.name,
        logs,
        f"{ds.name}_final",
        args,
    )
    resolution_limit = run_resolution_limit_gate(
        python,
        {ds.name: final},
        final_model,
        root / "resolution_limits" / ds.name,
        logs,
        ds.name,
        args,
        title=f"GROND resolution-limit promotion: {ds.name}",
        scope="dataset",
    )
    sg_window_scout = run_sg_window_scout(
        python,
        prep,
        ds,
        final_model,
        final,
        root / "sg_window_scout" / ds.name,
        logs,
        f"{ds.name}_final",
        args,
    )
    return {
        "primary_run": str(final),
        "no_h_run": str(no_h_run) if no_h_run else "",
        "model": final_model,
        "absolute_structure_diagnostic": absolute_structure_diagnostic,
        "scale_polish": scale_polish,
        "resolution_limits": {ds.name: resolution_limit},
        "sg_window_scout": {ds.name: sg_window_scout},
    }
