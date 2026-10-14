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
from grond.hkl_transform import parse_hkl_transform_info
from grond.io.config import PrepConfig, write_prep
from grond.io.structure_contract import check_sfac_contract, prepare_structure_for_legacy
from grond.macromolecule import sha256_file
from grond.reports.progress import _quick_progress_finish_info
from grond.reports.quick_stats import write_quick_stats
from grond.reports.terminal import _format_terminal_finish_summary
from grond.runtime import auto_chunksize, auto_worker_count
from grond.workflows.all_data_refinement import all_data_console_line, run_all_data_refinement
from grond.workflows.auto_crystal_parameters import configure_auto_crystals
from grond.workflows.checkpoints import write_continue_checkpoint
from grond.workflows.conventional_baseline import run_fresh_conventional_baseline
from grond.workflows.endpoint_selection import (
    run_endpoint_calibration,
    run_publication_endpoint_chooser,
    run_publication_report,
)
from grond.workflows.leak_free_start import prep_with_leak_free_start, run_leak_free_start
from grond.workflows.model_selection import resolve_structure_update_strategy
from grond.workflows.multi import run_multi
from grond.workflows.nested_nuisance import validate_nested_nuisance_policy
from grond.workflows.nuisance_policy import (
    apply_speed_profile,
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
    return write_quick_stats(prep, result, args)


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


def _configure_static_pets_auto(
    prep: PrepConfig,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    """Apply source-safe AUTO defaults for a row-matched ordinary PETS import."""

    bridge = (prep.metadata or {}).get("pets2_bridge")
    if not isinstance(bridge, Mapping):
        return None
    if len(prep.datasets) != 1:
        raise SystemExit("PETS2 AUTO currently supports exactly one ordinary integration")
    if bool(bridge.get("virtual_frames")):
        raise SystemExit(
            "PETS *_dyn.cif_pets virtual-frame occurrences are correlated row-space data, "
            "not a static SHELXL endpoint. Import the ordinary cif_pets plus its companion "
            ".hkl and PETS-exported SHELX HKLF4 instead."
        )
    static_contract = bridge.get("static_endpoint_contract")
    if not isinstance(static_contract, Mapping) or static_contract.get("status") != "validated":
        raise SystemExit(
            "PETS2 AUTO requires a row-for-row validated PETS static HKLF4. Re-import "
            "the ordinary cif_pets with its 13-column .hkl, starting model, and the "
            "matching PETS-exported SHELX HKL as --baseline-hkl."
        )
    baseline = Path(prep.baseline_hkl).expanduser().resolve()
    if not baseline.is_file():
        raise SystemExit(f"validated PETS static HKLF4 is not readable: {baseline}")
    expected_hash = str(static_contract.get("sha256") or "")
    actual_hash = sha256_file(baseline)
    if not expected_hash or actual_hash != expected_hash:
        raise SystemExit(
            "PETS static HKLF4 hash changed after import; refusing the AUTO run "
            f"(expected {expected_hash or 'missing'}, observed {actual_hash})"
        )
    dataset = prep.datasets[0]
    transform = parse_hkl_transform_info(dataset.hkl_transform or "identity")
    if transform.canonical != "identity" or not transform.proper:
        raise SystemExit("PETS2 AUTO currently requires the importer-validated proper identity setting")

    changes: dict[str, dict[str, object]] = {}

    def force(name: str, value: object) -> None:
        """Set one option on ``args`` and record the old and new value when it changes."""
        old = getattr(args, name)
        setattr(args, name, value)
        if old != value:
            changes[name] = {"old": old, "new": value}

    force("trust_hkl_transform", True)
    force("frame_offset_grid", "0")
    force("group_width_grid", "999,10,5,3,2,1")
    force("detector_scale_bins", "0")
    force("scale_polish", "off")
    force("scale_polish_detector_bins", "0")
    # PETS exact rows have no continuous scan angle for P-spline scales; keep compact angular groups.
    force("scale_model", "per-frame")
    force("row_resolution_b_grid", "0.0")
    force("sigma_c_peak_grid", "0.0")
    force("sg_jitter_mode_grid", "none")
    force("sg_jitter_sigma0_grid", "0.0")
    force("sg_jitter_sigma_d_inv_grid", "0.0")
    force("sg_jitter_sigma_frame_grid", "0.0")
    # PETS exact rows have no mosaic-cone implementation; the engine refuses it.
    force("mosaicity_mode", "none")
    # The nested nuisance stage scores with P-spline scales in scan angle, which PETS exact rows do not have.
    force("nuisance_selection", "sequential")
    force("feedback_thickness_update", "off")
    force("thickness_model_grid", "single,uniform_gradient")
    force("thickness_grad_grid", "0.0,0.4,1.0")
    force("profile_quadrature", "uniform")
    force("profile_accumulation", "sum")
    force("sg_window_scout", "off")
    force("observation_polish", "off")
    force("forensic_depth", "quick")
    force(
        "endpoint_calibration_specs",
        "baseline,window:1.0:det0,window:3.0:det0,window:5.0:det0",
    )
    args.pets_static_auto = True
    payload = {
        "schema": "grond.pets2-static-auto-contract.v1",
        "status": "validated",
        "source": bridge.get("source"),
        "rows": bridge.get("rows"),
        "virtual_frames": False,
        "static_hkl": str(baseline),
        "static_hkl_sha256": actual_hash,
        "transform": transform.to_dict(),
        "automatic_overrides": changes,
        "selection": (
            "AUTO chooses screen/final beam bases, quadrature count, thickness, compact "
            "angular frame scales, hand, and the SHELXL rounds; detector scales remain disabled "
            "because cif_pets has no detector coordinates"
        ),
    }
    contract_path = Path(args.out_dir) / "PETS_STATIC_AUTO_CONTRACT.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


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
    _configure_static_pets_auto(prep, args)
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
    if len(prep.datasets) > 1 and str(getattr(args, "endpoint_calibration", "auto")) == "on":
        raise SystemExit(
            "--endpoint-calibration on cannot be forced for a multi-dataset run. "
            "Each crystal has an independent scale and sigma contract; a single concatenated replay would mix them. "
            "Leave the option at auto for the safe combined-endpoint workflow."
        )
    profile = apply_speed_profile(args, prep.structure)
    configure_auto_crystals(prep, args)
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
        scale_model_policy = validate_scale_model_policy(args, multi_dataset=len(prep.datasets) > 1, prep=prep)
        refuse_sigma_weight_startup_conflicts(args)
        warn_missing_cell_esd_source(prep.structure, prep.datasets, args)
        nuisance_selection_policy = validate_nested_nuisance_policy(args, prep)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    from grond.commands.final_polish import refuse_unused_polish_options, validate_final_polish

    refuse_unused_polish_options(args)
    if str(getattr(args, "final_polish", "off") or "off") != "off":
        # Refuse bad options and models the polish cannot represent before the run, not after it.
        validate_final_polish(args, structure=prep.structure)
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
        "crystal_parameters": getattr(args, "crystal_parameters", {}),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": python,
        "tool_config": {
            "path": getattr(args, "tool_config_path", ""),
            "configured_tools": getattr(args, "configured_tools", {}),
            "external_tools": tools,
            "allow_missing_tools": bool(getattr(args, "allow_missing_tools", False)),
        },
        "accelerators": probe_accelerators(python),
        "mode": args.mode,
        "fast": args.fast,
        "full_fast": args.full_fast,
        "speed_profile": profile,
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
        "final_basis_fallback": args.final_basis_fallback,
        "prediction_cache_scope": args.prediction_cache_scope,
        "prediction_cache_compression": args.prediction_cache_compression,
        "compute_backend": args.compute_backend,
        "bloch_evaluator": args.bloch_evaluator_settings,
        "jax_batch_size": args.jax_batch_size,
        "jax_matrix_bucket_size": args.jax_matrix_bucket_size,
        "matrix_convention": args.matrix_convention,
        "structure_update": {
            "requested": args.structure_update,
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
        "scale_polish": {
            "mode": args.scale_polish,
            "frame_models": args.scale_polish_frame_models,
            "adaptive_counts": args.scale_polish_adaptive_counts,
            "window_deg": args.scale_polish_window_deg,
            "detector_bins": args.scale_polish_detector_bins,
            "detector_ridge_to_unity": args.detector_scale_ridge,
            "detector_spatial_smoothness": args.detector_scale_smoothness,
            "detector_minimum_rows_per_tile": args.detector_scale_min_rows,
            "detector_minimum_holdout_improvement": args.detector_scale_min_holdout_improvement,
            "detector_required_improved_splits": args.detector_scale_required_improved_splits,
            "detector_maximum_bloch_gain_loss": args.detector_scale_max_bloch_gain_loss,
            "detector_minimum_field_correlation": args.detector_scale_min_field_correlation,
            "detector_maximum_field_log_rms": args.detector_scale_max_field_log_rms,
            "detector_independent_calibration": args.detector_scale_independent_calibration,
            "objective": args.scale_polish_objective,
            "tolerance": args.scale_polish_tolerance,
            "selection_metrics": args.scale_polish_selection_metrics,
            "scan_offsets": args.scale_polish_scan_offsets,
        },
        "multi_scale_contract_selection": {
            "mode": args.multi_scale_contract_selection,
            "minimum_scale_reduction": args.multi_scale_min_reduction,
            "require_bic_improvement_diagnostic_only": args.multi_scale_require_bic_improvement,
            "maximum_row_rall_cost": args.multi_scale_max_row_rall_cost,
            "maximum_endpoint_r1_cost": args.multi_scale_max_endpoint_r1_cost,
            "maximum_endpoint_wr2_cost": args.multi_scale_max_endpoint_wr2_cost,
            "goof_range": [args.multi_scale_goof_min, args.multi_scale_goof_max],
            "maximum_input_row_loss_fraction": args.multi_scale_max_input_row_loss_fraction,
            "minimum_bloch_rall_gain": args.multi_scale_min_bloch_rall_gain,
            "maximum_dataset_row_rall_cost": args.multi_scale_max_dataset_row_rall_cost,
            "maximum_preexisting_bloch_gain_loss": args.multi_scale_max_preexisting_bloch_gain_loss,
        },
        "resolution_limit": {
            "enabled": args.resolution_limit_gate,
            "shells": args.resolution_shells,
            "strict_split_cc": args.resolution_strict_split_cc,
            "strict_mean_i_over_sigma": args.resolution_strict_mean_i_over_sigma,
            "exploratory_split_cc": args.resolution_exploratory_split_cc,
            "exploratory_mean_i_over_sigma": args.resolution_exploratory_mean_i_over_sigma,
            "min_common_pairs": args.resolution_min_common_pairs,
            "min_shell_rows": args.resolution_min_shell_rows,
            "min_strong_fraction": args.resolution_min_strong_fraction,
            "max_shell_mwrall": args.resolution_max_shell_mwrall,
            "max_free_work_mwr_gap": args.resolution_max_free_work_mwr_gap,
            "min_free_rows": args.resolution_min_free_rows,
            "max_cumulative_mwr_penalty": args.resolution_max_cumulative_mwr_penalty,
        },
        "sg_window_scout": {
            "mode": args.sg_window_scout,
            "grid": args.sg_window_scout_grid,
            "n_basis": args.sg_window_scout_n_basis,
            "n_points": args.sg_window_scout_n_points,
            "max_rows": args.sg_window_scout_max_rows,
            "min_wR2_gain": args.sg_window_scout_min_wR2_gain,
            "free_wR2_tolerance": args.sg_window_scout_free_wR2_tolerance,
        },
        "dataset_policy": {
            "mode": args.dataset_policy,
            "min_rows": args.dataset_policy_min_rows,
            "min_weight": args.dataset_policy_min_weight,
            "primary_min_quality": args.dataset_policy_primary_min_quality,
            "min_raw_quality": args.dataset_policy_min_raw_quality,
            "min_cc": args.dataset_policy_min_cc,
            "max_wr2": args.dataset_policy_max_wr2,
            "max_r1": args.dataset_policy_max_r1,
            "unique_gain_fraction": args.dataset_policy_unique_gain_fraction,
            "wr2_scale": args.dataset_policy_wr2_scale,
            "r1_scale": args.dataset_policy_r1_scale,
            "cc_scale": args.dataset_policy_cc_scale,
        },
        "free_set": str(getattr(args, "free_set", "on")),
        "final_polish": {
            "mode": str(getattr(args, "final_polish", "off")),
            "max_iterations": getattr(args, "final_polish_max_iterations", None),
            "thickness": getattr(args, "final_polish_thickness", None),
            "shelxl": getattr(args, "final_polish_shelxl", None),
        },
        "publication_endpoint_chooser": {
            "enabled": args.publication_endpoint_chooser,
            "min_hkl_safe_factor": args.publication_endpoint_min_hkl_safe_factor,
            "max_modulation_p95": args.publication_endpoint_max_modulation_p95,
            "allow_npd": args.publication_allow_npd,
            "feedback_cycles": str(getattr(args, "publication_feedback_cycles", "all") or "all"),
        },
        "endpoint_calibration": {
            "mode": args.endpoint_calibration,
            "specs": args.endpoint_calibration_specs,
            "lambda_q": args.endpoint_calibration_lambda_q,
            "wght_modes": args.endpoint_calibration_wght_modes,
            "preferred_wght_mode": args.endpoint_calibration_prefer_wght_mode,
            "selection_policy": args.endpoint_calibration_selection_policy,
            "information_criterion": args.endpoint_calibration_information_criterion,
            "goof_min": args.endpoint_calibration_goof_min,
            "goof_max": args.endpoint_calibration_goof_max,
            "max_r1_cost": args.endpoint_calibration_max_r1_cost,
            "free_wr2_tolerance": args.endpoint_calibration_free_wr2_tolerance,
            "sigma_min_scale": args.endpoint_calibration_sigma_min_scale,
            "sigma_max_scale": args.endpoint_calibration_sigma_max_scale,
            "allow_npd": args.publication_allow_npd,
        },
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
    result["endpoint_calibration"] = run_endpoint_calibration(prep, result, args, python, logs)
    result["publication_endpoint_chooser"] = run_publication_endpoint_chooser(prep, result, args)
    # --free-set off: the published endpoint refined once more against work and free reflections
    result["final_all_data_refinement"] = run_all_data_refinement(prep, result, args)
    print(all_data_console_line(result["final_all_data_refinement"]), flush=True)
    result["publication_tables"] = run_publication_report(python, prep, result, args, logs)
    result["continue_checkpoint"] = write_continue_checkpoint(prep, result, args)
    if bool(getattr(args, "dry_run", False)):
        result["status"] = "dry_run"
    quick = _write_run_completion_artifacts(prep, result, args)
    if str(getattr(args, "final_polish", "off") or "off") != "off":
        # Optional last stage on the finished run; its outputs are additional and fail closed.
        from grond.commands.final_polish import run_final_polish_stage

        result["final_polish"] = run_final_polish_stage(args, python, logs, result)
        (Path(args.out_dir) / "grond_summary.json").write_text(json.dumps(result, indent=2) + "\n")
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
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, allow_nan=True))
    else:
        print(_format_terminal_finish_summary(payload), flush=True)
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """Run the ordinary one-command GROND auto workflow."""

    args.mode = "auto"
    return cmd_run(args)
