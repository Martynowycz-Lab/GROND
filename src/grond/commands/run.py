"""Prepare, dispatch and finish automatic or fixed-model refinement runs."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

from grond.accelerators import probe_accelerators
from grond.cli_support import execution
from grond.cli_support.execution import configure_progress, estimate_run_steps
from grond.cli_support.tools import choose_python, external_tool_statuses, raise_for_missing_external_tools
from grond.commands.preparation import prep_from_args
from grond.io.config import PrepConfig, write_prep
from grond.io.structure_contract import check_sfac_contract, prepare_structure_for_legacy
from grond.reports.progress import _quick_progress_finish_info
from grond.reports.quick_stats import write_quick_stats
from grond.reports.terminal import _format_terminal_finish_summary
from grond.runtime import auto_chunksize, auto_worker_count
from grond.workflows.all_data_refinement import all_data_console_line, run_all_data_refinement
from grond.workflows.checkpoints import write_continue_checkpoint
from grond.workflows.conventional_baseline import run_fresh_conventional_baseline
from grond.workflows.endpoint_selection import run_publication_endpoint_chooser, run_publication_report
from grond.workflows.leak_free_start import prep_with_leak_free_start, run_leak_free_start
from grond.workflows.model_selection import resolve_structure_update_strategy
from grond.workflows.multi import run_multi
from grond.workflows.nested_nuisance import validate_nested_nuisance_policy
from grond.workflows.nuisance_policy import (
    refuse_sigma_weight_startup_conflicts,
    resolved_dataset_sigma_policy,
    resolved_merge_sigma_policy,
    resolved_post_bloch_sigma_policy,
    warn_missing_cell_esd_source,
)
from grond.workflows.run_presets import apply_run_preset, preset_console_lines
from grond.workflows.scale_model_policy import validate_scale_model_policy
from grond.workflows.single import run_single
from grond.workflows.split_record import announce_run_split
from grond.workflows.transform_selection import exclude_metric_incompatible_datasets_at_start


def _finalize_quick_stats(prep: PrepConfig, result: Mapping[str, object], args: argparse.Namespace) -> dict:
    """Write real-run statistics, or describe an intentionally unmaterialized dry run."""

    if bool(getattr(args, "dry_run", False)):
        return {
            "status": "dry_run",
            "out_dir": str(Path(args.out_dir).expanduser().resolve()),
            "note": (
                "Command planning completed without executing scientific stages. "
                "No endpoint or row-level statistics were materialized."
            ),
        }
    return write_quick_stats(prep, result, args, report_layout="heldout-first")


def _write_run_completion_artifacts(
    prep: PrepConfig,
    result: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Write reports and the run summary before completion is persisted."""

    result.pop("finished", None)
    result.pop("finish_preview", None)
    quick = _finalize_quick_stats(prep, result, args)
    result["quick_stats"] = quick
    (Path(args.out_dir) / "grond_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return quick


def _validated_h_omit_mode(value: object, dataset_count: int) -> str:
    """Normalize the retrospective H-omit request and reject unsupported multi use."""

    mode = str(value or "off").strip().lower()
    if mode not in {"off", "on", "only", "both"}:
        raise ValueError("--h-omit must be off, on, only, or both")
    if int(dataset_count) > 1 and mode != "off":
        raise ValueError(
            "--h-omit is not implemented for multi-dataset auto and was not run. "
            "For prospective hydrogen discovery, refine a separate model that has never contained H atoms; "
            "for a retrospective single-dataset sensitivity control, run that dataset separately with --h-omit on."
        )
    return "on" if mode in {"on", "only", "both"} else "off"


def _refuse_static_pets_auto(prep: PrepConfig) -> None:
    """Refuse a PETS2 import: its exact rows have no continuous scan angle for the P-spline frame scales."""

    if isinstance((prep.metadata or {}).get("pets2_bridge"), Mapping):
        raise SystemExit(
            "PETS2 input is not supported: GROND fits P-spline frame scales in the continuous rotation-scan "
            "angle, which PETS exact rows do not have"
        )


def cmd_run(args: argparse.Namespace) -> int:
    """CLI entrypoint for the configurable `GROND run` workflow."""

    invocation_started_monotonic = time.perf_counter()
    invocation_started_at = time.time()
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logs = Path(args.out_dir) / "logs"
    python = choose_python(args.python)
    # A named preset fills the options the user did not give, before anything reads them.
    preset = apply_run_preset(args)
    for line in preset_console_lines(preset):
        print(line, flush=True)
    from grond.bloch import shared_grid

    # Every stage copies os.environ, so the evaluator options are set once here, before any stage starts.
    args.bloch_evaluator_settings = shared_grid.apply_cli_options(args)
    if args.bloch_evaluator_settings["mode"] != "off" or args.bloch_evaluator_settings["basis_centring_coset"]:
        print(f"Bloch evaluator settings: {json.dumps(args.bloch_evaluator_settings, sort_keys=True)}", flush=True)
    prep = prep_from_args(args)
    _refuse_static_pets_auto(prep)
    try:
        requested_h_omit = str(getattr(args, "h_omit", "off") or "off")
        args.h_omit = _validated_h_omit_mode(requested_h_omit, len(prep.datasets))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if requested_h_omit in {"only", "both"}:
        print(
            f"WARN --h-omit {requested_h_omit} is a deprecated alias for --h-omit on; "
            "the calculation is a retrospective strip-H sensitivity control, not a prospective never-H experiment.",
            flush=True,
        )
    args.workers = auto_worker_count(args.workers, args.max_core_fraction)
    args.chunksize = auto_chunksize(args.chunksize, args.workers)
    original_structure = prep.structure
    prep.structure, structure_prep = prepare_structure_for_legacy(prep.structure, Path(args.out_dir))
    sfac_check = check_sfac_contract(prep.structure)
    sfac_check_path = Path(args.out_dir) / "sfac_check.json"
    sfac_check_path.write_text(json.dumps(sfac_check, indent=2) + "\n")
    if not sfac_check.get("ok", False):
        error_text = "; ".join(
            str(item.get("message", "")) for item in sfac_check.get("messages", []) if item.get("severity") == "error"
        )
        raise SystemExit(
            "GROND model SFAC check failed. "
            + (error_text or "See sfac_check.json for details.")
            + f" Report: {sfac_check_path}"
        )
    raise_for_missing_external_tools(args, structure=prep.structure, multi_dataset=len(prep.datasets) > 1)
    try:
        scale_model_policy = validate_scale_model_policy(args)
        refuse_sigma_weight_startup_conflicts(args)
        warn_missing_cell_esd_source(prep.structure, prep.datasets, args)
        nuisance_selection_policy = validate_nested_nuisance_policy(args, prep)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    configure_progress(
        logs,
        resume=bool(args.resume),
        total=estimate_run_steps(prep, args),
        invocation_started_monotonic=invocation_started_monotonic,
        invocation_started_at=invocation_started_at,
        begin_invocation=True,
    )
    prep.metadata = dict(prep.metadata or {})
    if structure_prep.get("prepared"):
        prep.metadata["original_structure"] = original_structure
        prep.metadata["structure_preparation"] = structure_prep
    write_prep(Path(args.out_dir) / "run_prep.json", prep)
    tools = external_tool_statuses(args)
    config = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": python,
        "tool_config": {
            "path": getattr(args, "tool_config_path", ""),
            "configured_tools": getattr(args, "configured_tools", {}),
            "external_tools": tools,
            "allow_missing_tools": bool(getattr(args, "allow_missing_tools", False)),
        },
        "accelerators": probe_accelerators(python),
        "mode": "auto",
        "preset": preset,
        "datasets": [ds.__dict__ for ds in prep.datasets],
        "structure": prep.structure,
        "original_structure": original_structure,
        "structure_preparation": structure_prep,
        "structure_sfac_check": sfac_check,
        "baseline_hkl": prep.baseline_hkl,
        "workers": args.workers,
        "chunksize": args.chunksize,
        "max_core_fraction": args.max_core_fraction,
        "n_basis": args.n_basis,
        "final_basis_scan_limit": args.final_basis_scan_limit,
        "final_basis_margin": args.final_basis_margin,
        "final_basis_min": args.final_basis_min,
        "final_basis_max": args.final_basis_max,
        "prediction_cache_scope": args.prediction_cache_scope,
        "prediction_cache_compression": args.prediction_cache_compression,
        "bloch_evaluator": args.bloch_evaluator_settings,
        "matrix_convention": args.matrix_convention,
        "structure_update": {
            "resolved": resolve_structure_update_strategy(args, multi_dataset=len(prep.datasets) > 1),
            "shelxl": args.shelxl,
            "shelxl_feedback_cycles": args.shelxl_feedback_cycles,
            "shelxl_feedback_shelxl_iterations": args.shelxl_feedback_shelxl_iterations,
        },
        "post_bloch_sigma": {
            "policy": args.post_bloch_sigma_policy,
            "resolved_final_endpoint": resolved_post_bloch_sigma_policy(args, final_endpoint=True),
            "resolved_feedback_loop": resolved_post_bloch_sigma_policy(args, feedback_loop=True),
            "holdout_fraction": args.post_bloch_sigma_holdout_fraction,
            "min_scale": args.post_bloch_sigma_min_scale,
            "max_scale": args.post_bloch_sigma_max_scale,
        },
        "multi_dataset_sigma": {
            "dataset_policy": args.dataset_sigma_policy,
            "resolved_dataset_policy": resolved_dataset_sigma_policy(args, multi_dataset=len(prep.datasets) > 1),
            "dataset_holdout_fraction": args.dataset_sigma_holdout_fraction,
            "dataset_min_scale": args.dataset_sigma_min_scale,
            "dataset_max_scale": args.dataset_sigma_max_scale,
            "dataset_holdout_floor": args.dataset_sigma_holdout_floor,
            "merge_policy": args.merge_sigma_policy,
            "resolved_merge_policy": resolved_merge_sigma_policy(args, multi_dataset=len(prep.datasets) > 1),
        },
        "scale_model": scale_model_policy,
        "nuisance_selection": nuisance_selection_policy,
        "dataset_policy": {
            "mode": args.dataset_policy,
        },
        "free_set": str(getattr(args, "free_set", "on")),
        "publication_endpoint_chooser": {
            "enabled": args.publication_endpoint_chooser,
            "min_hkl_safe_factor": args.publication_endpoint_min_hkl_safe_factor,
            "max_modulation_p95": args.publication_endpoint_max_modulation_p95,
            "allow_npd": args.publication_allow_npd,
            "feedback_cycles": str(getattr(args, "publication_feedback_cycles", "all") or "all"),
        },
        # the command line, so that `grond refine` can repeat this run's own options
        "argv": list(getattr(args, "argv", []) or []),
    }
    (Path(args.out_dir) / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    # a crystal whose cell no transform reconciles with the model is left out, recorded, before any split is planned
    prep = exclude_metric_incompatible_datasets_at_start(prep, args, Path(args.out_dir))
    # the free set, the excluded (lockbox) orbits and their counts, before any fit (decisions/split_summary.json)
    split_plan = announce_run_split(prep, args, Path(args.out_dir))
    conventional_baseline = run_fresh_conventional_baseline(prep, args, python, logs)
    # every stage starts from the work-orbit refinement of the supplied model (free rows never shaped it)
    leak_free_start = run_leak_free_start(prep, args, logs, split_plan=split_plan)
    prep = prep_with_leak_free_start(prep, leak_free_start)
    result = run_single(prep, args, python, logs) if len(prep.datasets) == 1 else run_multi(prep, args, python, logs)
    result["conventional_baseline"] = conventional_baseline
    result["leak_free_start"] = leak_free_start
    result["publication_endpoint_chooser"] = run_publication_endpoint_chooser(prep, result, args)
    # --free-set off: the published endpoint refined once more against work and free reflections
    result["final_all_data_refinement"] = run_all_data_refinement(prep, result, args)
    print(all_data_console_line(result["final_all_data_refinement"]), flush=True)
    result["publication_tables"] = run_publication_report(python, prep, result, args, logs)
    result["continue_checkpoint"] = write_continue_checkpoint(prep, result, args)
    if bool(getattr(args, "dry_run", False)):
        result["status"] = "dry_run"
    quick = _write_run_completion_artifacts(prep, result, args)
    finished = (
        execution._PROGRESS.mark_finished(str(result.get("status", "ok")))
        if execution._PROGRESS
        else _quick_progress_finish_info(Path(args.out_dir))
    )
    console = quick.get("console") if isinstance(quick.get("console"), Mapping) else {}
    payload = (
        {"status": result.get("status", "ok"), "finished": finished, **console}
        if console
        else {
            "status": result.get("status", "ok"),
            "finished": finished,
            "out_dir": str(Path(args.out_dir).resolve()),
            "quick_stats": quick,
        }
    )
    print(_format_terminal_finish_summary(payload), flush=True)
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """Run the ordinary one-command GROND auto workflow."""

    return cmd_run(args)
