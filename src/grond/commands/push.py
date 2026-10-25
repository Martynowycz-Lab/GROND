"""Push an existing selected endpoint through an explicitly configured refinement step."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from grond.cli_support.tools import choose_python, external_tool_statuses
from grond.cli_support.values import read_json, try_read_json
from grond.io.config import DatasetSpec, PrepConfig, write_datasets_csv, write_prep
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
    prep_from_multi_checkpoint,
    resolve_dataset_row_contracts,
    resolve_push_continuation_contract,
    single_push_handoff_args,
    warn_historical_continuation_defaults,
)
from grond.workflows.push_models import prepare_push_endpoint_model
from grond.workflows.shared_refinement import run_multi_shelxl_feedback_loop

# Safety report: cycles within this of the best row wR(I)all are listed as equivalent.
BEST_CYCLE_TOLERANCE = 0.001


# Options that describe where a run's inputs and outputs are; `grond refine` sets its own.
RUN_LOCATION_OPTIONS = {
    "--prep", "--out-dir", "--prediction-cache-dir", "--resume", "--no-resume", "--hkl", "--xds-dir", "--structure",
    "--baseline-hkl", "--dials-refl", "--dials-expt", "--hkl-transform", "--trust-hkl-transform", "--hand-control",
    "--no-hand-control",
}
# Options of `grond refine` itself; every other option on its command line goes to `grond auto`.
REFINE_OPTIONS = {"--checkpoint", "--start-res", "--out-dir", "--cycles", "--reuse-settings"}


def _auto_option_actions() -> dict[str, argparse.Action]:
    """The options of `grond auto`, by flag."""

    from grond.cli_options.parser import build_parser

    sub = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    return {flag: action for action in sub.choices["auto"]._actions for flag in action.option_strings}


def _option_groups(argv: list[str], actions: Mapping[str, argparse.Action]) -> list[tuple[str, list[str]]]:
    """Split a command line into (flag, tokens) with the arity of each `grond auto` or `grond refine` option."""

    groups, i = [], 0
    while i < len(argv):
        token = argv[i]
        flag = token.split("=", 1)[0]
        action = actions.get(flag)
        takes_value = "=" not in token and (
            (action is not None and action.nargs != 0) or (action is None and flag in REFINE_OPTIONS - {"--reuse-settings"})
        )
        n = 2 if takes_value and i + 1 < len(argv) else 1
        groups.append((flag, argv[i:i + n]))
        i += n
    return groups


def cmd_refine_new_settings(args: argparse.Namespace) -> int:
    """Run `grond auto` on the new model with the run's data, indexing, hand, held-out set and options.

    The new model is not re-refined against the kinematic data (no baseline HKL) and its hand is kept
    (--no-hand-control). The split contracts of the run keep the held-out set unchanged. Beams, rocking points,
    thickness, mosaic spread and damage are chosen again, as `grond auto` chooses them.
    """

    from grond.cli_support.entrypoint import main as grond_main

    if not str(getattr(args, "checkpoint", "") or ""):
        raise SystemExit("grond refine needs --checkpoint of a grond auto run (--source-summary works with --reuse-settings).")
    checkpoint_info = apply_push_checkpoint(args)
    checkpoint = checkpoint_info["checkpoint_data"]
    run_dir = Path(str(checkpoint.get("run_dir") or Path(str(checkpoint_info["checkpoint"])).parent))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for row in checkpoint.get("datasets") or []:
        values = {key: row[key] for key in DatasetSpec.__dataclass_fields__ if row.get(key) is not None}
        values["baseline_hkl"] = ""
        datasets.append(DatasetSpec(**values))
    if not datasets:
        raise SystemExit(f"{checkpoint_info['checkpoint']} records no datasets.")
    prep = PrepConfig(
        structure=str(Path(args.start_res).expanduser().resolve()),
        datasets=datasets,
        default_hkl_transform=str(checkpoint.get("hkl_transform") or "identity"),
        metadata={"refined_from": str(run_dir)},
    )
    prep_path = out_dir / "refine_prep.json"
    write_prep(prep_path, prep)

    actions = _auto_option_actions()
    recorded = try_read_json(run_dir / "run_config.json").get("argv")
    carried: list[str] = []
    if isinstance(recorded, list) and recorded and recorded[0] == "auto":
        for flag, tokens in _option_groups([str(v) for v in recorded[1:]], actions):
            if flag not in RUN_LOCATION_OPTIONS:
                carried += tokens
    else:
        print("NOTE the run did not record its command line; refine uses the grond auto defaults and the options "
              "given here.", flush=True)
    given: list[str] = []
    argv = list(getattr(args, "argv", []) or [])
    for flag, tokens in _option_groups(argv[argv.index("refine") + 1:] if "refine" in argv else [], actions):
        if flag == "--cycles":
            given += ["--shelxl-feedback-cycles", tokens[-1]]
        elif flag in REFINE_OPTIONS:
            continue
        elif flag not in actions and flag not in ("-v", "--verbose"):
            raise SystemExit(f"grond refine: {flag} works only with --reuse-settings.")
        elif flag not in ("-v", "--verbose"):
            given += tokens
    auto_argv = ["auto", *carried, *given, "--prep", str(prep_path), "--out-dir", str(out_dir),
                 "--trust-hkl-transform", "--no-hand-control"]
    record = {"checkpoint": str(checkpoint_info["checkpoint"]), "start_res": str(prep.structure),
              "options_of_the_run": carried, "options_given": given, "grond": auto_argv}
    (out_dir / "refine_command.json").write_text(json.dumps(record, indent=2) + "\n")
    print("grond refine: grond " + " ".join(auto_argv), flush=True)
    return grond_main(auto_argv)


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
    args.dataset_row_contracts = resolve_dataset_row_contracts(
        getattr(args, "dataset_row_contract", None), prep.datasets
    )
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
            "structure_dataset_policy_source": "immutable_row_contract" if row_contract_policy else "checkpoint",
            "multi_endpoint_merge_key": args.multi_endpoint_merge_key,
            "continuation_contract": args.continuation_contract,
            "shared_loop": "per-dataset Bloch rows -> GROND-side corrected-HKL merge -> one shared SHELXL LS step -> repeat",
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
        "structure_dataset_policy": dict(policy),
        "continuation_contract": args.continuation_contract,
    }
    (out_dir / "push_endpoint_summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    report = write_multi_push_endpoint_report(out_dir, best_cycle_tolerance=BEST_CYCLE_TOLERANCE)
    if report:
        print(f"Multi-dataset endpoint push report: {report}")
    return 0


def cmd_push_endpoint(args: argparse.Namespace) -> int:
    """`grond refine`: new settings by default, or the run's settings with --reuse-settings."""

    if not args.reuse_settings:
        return cmd_refine_new_settings(args)
    args.cycles = 3 if args.cycles is None else args.cycles
    python = choose_python(args.python)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = apply_push_checkpoint(args)
    start_model, prep_audit = prepare_push_endpoint_model(args, out_dir)
    if int(checkpoint_info.get("dataset_count") or 1) > 1:
        return cmd_push_endpoint_multi(args, checkpoint_info, start_model, prep_audit, python, out_dir)
    if args.dataset_n_basis or args.dataset_thickness_nm:
        raise SystemExit("--dataset-n-basis and --dataset-thickness-nm require a multi-dataset --checkpoint.")
    source_summary = Path(str(args.source_summary)).expanduser()
    if not source_summary.is_file():
        raise SystemExit(f"--source-summary {source_summary} does not exist")
    # The run's own beam count and rocking points, as the multi-crystal path replays them.
    saved = read_json(source_summary).get("config") or {}
    for key in ("n_basis", "n_points"):
        if getattr(args, key) is None:
            if saved.get(key) in (None, ""):
                raise SystemExit(f"{source_summary} records no {key}; give --{key.replace('_', '-')}.")
            setattr(args, key, int(saved[key]))
    # 0 replays the beam-pool depth recorded by the run, min(0.8 Å, d_min), as the multi-crystal path does.
    args.topup_d_min = 0.0 if args.topup_d_min is None else args.topup_d_min
    # The loop replays the split and scale model from the source summary; the SHELXL handoff is forwarded
    # explicitly from the same summary (historical values when it predates a key) or from an override.
    contract = resolve_push_continuation_contract(args, {"source_summary": read_json(source_summary)})
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
        "--row-plan-cache",
        args.row_plan_cache,
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
    for field in ("profile_quadrature", "profile_accumulation"):
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
    report = write_push_endpoint_report(out_dir, best_cycle_tolerance=BEST_CYCLE_TOLERANCE)
    if report:
        print(f"Endpoint push report: {report}")
    return 0
