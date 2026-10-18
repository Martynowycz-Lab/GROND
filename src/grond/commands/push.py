"""Push an existing selected endpoint through an explicitly configured refinement step."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from grond.cli_support.tools import choose_python, external_tool_statuses
from grond.cli_support.values import read_json
from grond.io.config import write_datasets_csv, write_prep
from grond.paths import engine_script
from grond.reports.push import write_multi_push_endpoint_report, write_push_endpoint_report
from grond.workflows.crystal_parameters import resolve_crystal_parameters
from grond.workflows.dataset_gates import all_primary_policy, apply_dataset_policy_to_nuisance
from grond.workflows.push_contracts import (
    apply_push_checkpoint,
    continuation_contract_record,
    continuation_source_summaries,
    dataset_policy_from_row_contracts,
    hydrate_push_args_for_multi,
    override_push_native_weights,
    prep_from_multi_checkpoint,
    refuse_single_push_split_scale_overrides,
    resolve_dataset_row_contracts,
    resolve_dataset_scale_specs,
    resolve_push_continuation_contract,
    single_push_handoff_args,
    warn_historical_continuation_defaults,
)
from grond.workflows.push_models import prepare_push_endpoint_model
from grond.workflows.shared_refinement import run_multi_shelxl_feedback_loop


def cmd_push_endpoint_multi(
    args: argparse.Namespace,
    checkpoint_info: Mapping[str, object],
    start_model: Path,
    prep_audit: Path | None,
    python: str,
    out_dir: Path,
) -> int:
    """Continue a multi-dataset endpoint from a finished GROND checkpoint."""

    checkpoint = checkpoint_info.get("checkpoint_data")
    if not isinstance(checkpoint, Mapping):
        raise SystemExit("Internal error: multi-dataset continuation missing checkpoint data.")
    prep, nuisance = prep_from_multi_checkpoint(
        checkpoint,
        start_model,
        min_d_spacing=float(getattr(args, "min_d_spacing", 0.0) or 0.0),
    )
    if args.final_n_basis:
        raise SystemExit("For multi-dataset continuation, use --dataset-n-basis DATASET=N or --n-basis N instead of --final-n-basis.")
    args.crystal_parameters, nuisance = resolve_crystal_parameters(
        checkpoint, prep.datasets, nuisance, args,
        checkpoint_dir=Path(str(checkpoint_info["checkpoint"])).parent,
    )
    # The split, scale and SHELXL handoff contract replays the checkpoint's source summaries, not AUTO defaults.
    args = hydrate_push_args_for_multi(
        args, continuation_source_summaries(checkpoint, Path(str(checkpoint_info["checkpoint"])).parent)
    )
    nuisance = override_push_native_weights(nuisance, args)
    args.dataset_row_contracts = resolve_dataset_row_contracts(
        getattr(args, "dataset_row_contract", None), prep.datasets
    )
    args.dataset_scale_specs = resolve_dataset_scale_specs(getattr(args, "dataset_scale_spec", None), prep.datasets)
    row_contract_policy = (
        dataset_policy_from_row_contracts(args.dataset_row_contracts, prep.datasets)
        if args.dataset_row_contracts
        else {}
    )
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (out_dir / "crystal_parameters.json").write_text(json.dumps(args.crystal_parameters, indent=2) + "\n")
    write_prep(out_dir / "push_prep.json", prep)
    policy = checkpoint.get("structure_dataset_policy")
    if not isinstance(policy, Mapping) or not policy:
        policy = all_primary_policy(
            {ds.name: Path(str(ds.hkl)).parent for ds in prep.datasets}, status="checkpoint_no_policy"
        )
    if row_contract_policy:
        policy = row_contract_policy
    # As AUTO's prep_datasets_with_nuisance.csv: the role and weight columns are the dataset policy the loop applies.
    write_datasets_csv(out_dir / "push_datasets.csv", prep, apply_dataset_policy_to_nuisance(nuisance, dict(policy)))
    if args.dry_run:
        plan = {
            "mode": "multi_dataset",
            "checkpoint": checkpoint_info.get("checkpoint", ""),
            "prepared_start_res": str(start_model),
            "prepared_audit": str(prep_audit) if prep_audit else "",
            "cycles": int(args.shelxl_feedback_cycles),
            "datasets": [asdict(ds) for ds in prep.datasets],
            "nuisance_datasets": sorted(nuisance),
            "crystal_parameters": args.crystal_parameters,
            "dataset_row_contracts": dict(args.dataset_row_contracts),
            "dataset_scale_specs": dict(args.dataset_scale_specs),
            "structure_dataset_policy_source": "immutable_row_contract" if row_contract_policy else "checkpoint",
            "multi_endpoint_merge_key": args.multi_endpoint_merge_key,
            "continuation_contract": args.continuation_contract,
            "shared_loop": "per-dataset Bloch rows -> GROND-side corrected-HKL merge -> one shared SHELXL LS step -> repeat",
            "compact_scale_reselection_performed": False,
            "tool_config": getattr(args, "tool_config_path", ""),
            "external_tools": external_tool_statuses(args),
        }
        (out_dir / "push_endpoint_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
        return 0

    final_model, feedback = run_multi_shelxl_feedback_loop(
        prep,
        args,
        python,
        logs,
        out_dir,
        start_model=str(start_model),
        nuisance=nuisance,
        structure_dataset_policy=dict(policy),
    )
    result = {
        "status": "ok",
        "mode": "multi_dataset",
        "checkpoint": checkpoint_info.get("checkpoint", ""),
        "start_model": str(start_model),
        "model": final_model,
        "structure_feedback": feedback,
        "crystal_parameters": args.crystal_parameters,
        "dataset_row_contracts": dict(args.dataset_row_contracts),
        "dataset_scale_specs": dict(args.dataset_scale_specs),
        "structure_dataset_policy": dict(policy),
        "continuation_contract": args.continuation_contract,
        "compact_scale_reselection_performed": False,
        "compact_scale_contract_replayed": bool(args.dataset_scale_specs),
        "scale_contract_note": (
            "The push replays the declared compact scale topology after each Bloch calculation; it does not select a new topology."
            if args.dataset_scale_specs
            else "The push refit the pinned full grouping. Run compact-scale selection separately on the selected safe cycle."
        ),
    }
    (out_dir / "push_endpoint_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    report = write_multi_push_endpoint_report(out_dir, best_cycle_tolerance=float(args.best_cycle_tolerance))
    if report:
        print(f"Multi-dataset endpoint push report: {report}")
    return 0


def cmd_push_endpoint(args: argparse.Namespace) -> int:
    """Continue an existing GROND endpoint with pinned nuisance settings."""

    python = choose_python(args.python)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = apply_push_checkpoint(args)
    start_model, prep_audit = prepare_push_endpoint_model(args, out_dir)
    if int(checkpoint_info.get("dataset_count") or 1) > 1:
        return cmd_push_endpoint_multi(args, checkpoint_info, start_model, prep_audit, python, out_dir)
    if args.dataset_n_basis or args.dataset_thickness_nm:
        raise SystemExit("--dataset-n-basis and --dataset-thickness-nm require a multi-dataset --checkpoint.")
    args.n_basis = 108 if args.n_basis is None else args.n_basis
    args.n_points = 20 if args.n_points is None else args.n_points
    args.topup_d_min = 0.7 if args.topup_d_min is None else args.topup_d_min
    source_summary = Path(str(args.source_summary)).expanduser()
    if not source_summary.is_file():
        raise SystemExit(f"--source-summary {source_summary} does not exist")
    # The loop replays the split and scale model from the source summary; the SHELXL handoff is forwarded
    # explicitly from the same summary (historical values when it predates a key) or from an override.
    contract = resolve_push_continuation_contract(args, {"source_summary": read_json(source_summary)})
    refuse_single_push_split_scale_overrides(contract)
    warn_historical_continuation_defaults(contract)
    args.rescale = str(contract["rescale"]["value"])
    cmd = [
        python,
        str(engine_script("shelxl_feedback_loop.py")),
        "--source-summary",
        args.source_summary,
        "--start-res",
        str(start_model),
        "--out-dir",
        str(out_dir),
        "--python",
        python,
        "--cycles",
        str(args.cycles),
        "--n-basis",
        str(args.n_basis),
        "--n-points",
        str(args.n_points),
        "--final-n-basis",
        str(args.final_n_basis),
        "--final-n-points",
        str(args.final_n_points),
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
        "--shelxl",
        args.shelxl,
        "--shelxl-iterations",
        str(args.shelxl_iterations),
        "--endpoint-ls-cycles",
        str(args.endpoint_ls_cycles),
        "--shelxl-feedback-xyz-tol",
        f"{float(args.shelxl_feedback_xyz_tol):g}",
        "--shelxl-feedback-keep",
        str(args.shelxl_feedback_keep),
        "--post-bloch-sigma-policy",
        args.post_bloch_sigma_policy,
        "--post-bloch-sigma-holdout-fraction",
        f"{args.post_bloch_sigma_holdout_fraction:g}",
        "--post-bloch-sigma-min-scale",
        f"{args.post_bloch_sigma_min_scale:g}",
        "--post-bloch-sigma-max-scale",
        f"{args.post_bloch_sigma_max_scale:g}",
        "--lambda-grid",
        args.lambda_grid,
        "--rescale",
        args.rescale,
        *single_push_handoff_args(contract),
        "--cache-compression",
        args.prediction_cache_compression,
        "--compute-backend",
        args.compute_backend,
        "--row-plan-cache",
        args.row_plan_cache,
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
        "--min-d-spacing",
        f"{args.min_d_spacing:g}",
        "--topup-d-min",
        f"{args.topup_d_min:g}",
        "--min-peak",
        f"{args.min_peak:g}",
        "--geometry-percentile",
        f"{args.geometry_percentile:g}",
        "--max-rows",
        str(args.max_rows),
    ]
    for field in ("profile_coordinate", "profile_normalization", "profile_angle_max_half_width_deg",
                  "profile_basis_mode", "profile_quadrature", "profile_accumulation"):
        value = getattr(args, field, None)
        if value is not None:
            cmd += ["--" + field.replace("_", "-"), str(value)]
    if args.hkl:
        cmd += ["--hkl", args.hkl]
    if args.baseline_hkl:
        cmd += ["--baseline-hkl", args.baseline_hkl]
    if bool(getattr(args, "ignore_source_baseline_hkl", False)):
        cmd.append("--ignore-source-baseline-hkl")
    if args.hkl_transform:
        cmd += [f"--hkl-transform={args.hkl_transform}"]
    if args.make_maps_on_final:
        cmd.append("--make-maps-on-final")
    if args.dry_run:
        plan = {
            "command": cmd,
            "prepared_start_res": str(start_model),
            "prepared_audit": str(prep_audit) if prep_audit else "",
            "continuation_contract": continuation_contract_record(contract),
            **checkpoint_info,
            "tool_config": getattr(args, "tool_config_path", ""),
            "external_tools": external_tool_statuses(args),
        }
        (out_dir / "push_endpoint_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
        return 0
    # The loop summary records the values; this records where each one came from (summary, override, historical).
    (out_dir / "push_endpoint_continuation_contract.json").write_text(
        json.dumps({"source_summary": str(args.source_summary), **continuation_contract_record(contract)}, indent=2)
        + "\n"
    )
    subprocess.run(cmd, check=True)
    report = write_push_endpoint_report(out_dir, best_cycle_tolerance=float(args.best_cycle_tolerance))
    if report:
        print(f"Endpoint push report: {report}")
    return 0
