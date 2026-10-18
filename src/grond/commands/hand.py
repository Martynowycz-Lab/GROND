"""Run an explicit hand comparison with source-summary physics and row bindings."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.tools import choose_python
from grond.cli_support.values import read_json, try_read_json
from grond.commands.scouting import _fill_common_run_defaults
from grond.hashing import sha256_file
from grond.io.config import DatasetSpec
from grond.io.structure_contract import make_alternate_hand_model
from grond.paths import engine_script
from grond.runtime import auto_chunksize, auto_worker_count
from grond.workflows.native_commands import native_stage_cmd


def _source_summary_hand_dataset(
    source_summary: Path,
) -> tuple[dict, DatasetSpec, str, str, dict]:
    """Resolve one immutable native-row summary into a hand-check contract."""

    summary = read_json(source_summary)
    inputs = summary.get("inputs") if isinstance(summary.get("inputs"), Mapping) else {}
    config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
    nuisance = best.get("nuisance") if isinstance(best.get("nuisance"), Mapping) else {}
    hkl = str(inputs.get("hkl") or "").strip()
    structure = str(inputs.get("res") or "").strip()
    row_table = str((summary.get("artifacts") or {}).get("native_row_table") or "").strip()
    if not hkl or not Path(hkl).is_file():
        raise SystemExit(f"Source summary does not bind a readable HKL file: {source_summary}")
    if not structure or not Path(structure).is_file():
        raise SystemExit(f"Source summary does not bind a readable model: {source_summary}")
    if not row_table or not Path(row_table).is_file():
        fallback = source_summary.parent / "native_dyn_calc.csv"
        if not fallback.is_file():
            raise SystemExit(f"Source summary does not bind a readable native row table: {source_summary}")
        row_table = str(fallback.resolve())
    raw_name = str(config.get("dataset_name") or source_summary.parent.parent.name or "dataset")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("_")[:64] or "dataset"
    dataset = DatasetSpec(
        name=name,
        hkl=str(Path(hkl).resolve()),
        xds_dir=str(Path(hkl).resolve().parent),
        baseline_hkl=str(inputs.get("baseline_hkl") or ""),
        hkl_transform=str(inputs.get("hkl_transform") or "identity"),
        min_d_spacing=float(config.get("min_d_spacing") or 0.70),
        min_peak=float(config.get("min_peak") or 0.0),
        target_mode=str(config.get("target_mode") or "iobs"),
        include_nonpositive=not bool(config.get("positive_only", True)),
        geometry_percentile=float(config.get("geometry_percentile") or 100.0),
        split_contract=str(config.get("split_contract") or ""),
    )
    return summary, dataset, structure, row_table, dict(nuisance)


def _apply_source_physics_to_hand_args(
    args: argparse.Namespace,
    source_config: Mapping[str, object],
) -> None:
    """Replay the exact source row-physics settings needed by hand checking."""

    keys = (
        "sg_window",
        "sg_window_mode",
        "sg_window_min",
        "sg_window_max",
        "sg_window_scale",
        "sg_window_frame_floor",
        "profile_coordinate",
        "profile_normalization",
        "profile_angle_max_half_width_deg",
        "profile_quadrature",
        "profile_accumulation",
        "topup_d_min",
        "max_support_sg",
        "free_fraction",
        "split_seed",
        "matrix_convention",
        "thickness_points",
    )
    for key in keys:
        if key in source_config and hasattr(args, key):
            setattr(args, key, source_config[key])
    # The free-split scheme belongs to the source row contract; summaries that
    # predate the option used the legacy per-dataset split.
    args.split_scheme = str(source_config.get("split_scheme") or "legacy")
    if str(source_config.get("split_salt") or ""):
        args.split_salt = str(source_config["split_salt"])
    args.exclude_orbits_json = str(source_config.get("exclude_orbits_json") or "")
    args.fast = False
    args.native_only = True
    args.make_maps = False
    args.production_scales = str(source_config.get("scale_fit_split") or "work") == "all"
    # Replay the source frame-scale family; summaries that predate the option used per-frame scales.
    args.scale_model = str(source_config.get("scale_model") or "per-frame")
    for key in ("scale_spline_n_basis_grid", "scale_spline_lambda_grid"):
        if key in source_config:
            setattr(args, key, source_config[key])


def cmd_hand_check(args: argparse.Namespace) -> int:
    """Run a fixed-nuisance matched hand check from one completed native stage."""

    _fill_common_run_defaults(args)
    source_path = Path(args.source_summary).expanduser().resolve()
    summary, dataset, structure, row_table, nuisance = _source_summary_hand_dataset(source_path)
    source_config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    _apply_source_physics_to_hand_args(args, source_config)
    out = Path(args.out_dir).expanduser().resolve()
    args.out_dir = str(out)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    python = choose_python(args.python)
    args.workers = auto_worker_count(args.workers, args.max_core_fraction)
    args.chunksize = auto_chunksize(args.chunksize, args.workers)
    args.dataset_row_contracts = {dataset.name: row_table}
    n_basis = int(args.hand_check_n_basis or source_config.get("n_basis") or 20)
    n_points = int(args.hand_check_n_points or source_config.get("n_points") or 10)
    inverse_structure = make_alternate_hand_model(
        python,
        structure,
        out / "inverse_model",
        logs,
        bool(args.dry_run),
    )
    elastic_nuisance = dict(nuisance)
    source_absorption = {
        key: elastic_nuisance.get(key)
        for key in (
            "absorption_mode",
            "absorption_mu",
            "absorption_frac",
            "absorption_mu_g2",
            "absorption_mu_theta2",
            "absorption_tds_b",
        )
    }
    source_absorption_active = str(source_absorption.get("absorption_mode") or "none").lower() != "none" or any(
        abs(float(source_absorption.get(key) or 0.0)) > 0.0
        for key in (
            "absorption_mu",
            "absorption_frac",
            "absorption_mu_g2",
            "absorption_mu_theta2",
            "absorption_tds_b",
        )
    )
    elastic_only = args.hand_check_method == "negative-thickness" or args.hand_check_absorption == "elastic"
    if args.hand_check_method == "negative-thickness" and args.hand_check_absorption == "preserve":
        raise SystemExit(
            "negative-thickness hand checking is exact only for the elastic Bloch "
            "Hamiltonian. Use --hand-check-absorption elastic, or choose "
            "--hand-check-method explicit to preserve absorption."
        )
    if elastic_only:
        elastic_nuisance.update(
            {
                "absorption_mode": "none",
                "absorption_mu": 0.0,
                "absorption_frac": 0.0,
                "absorption_mu_g2": 0.0,
                "absorption_mu_theta2": 0.0,
                "absorption_tds_b": 0.0,
            }
        )
    selected_run = out / "selected"
    selected_cmd = native_stage_cmd(
        python,
        dataset,
        structure,
        dataset.baseline_hkl,
        selected_run,
        args,
        n_basis=n_basis,
        n_points=n_points,
        stage="expert_hand_selected",
        pinned=elastic_nuisance,
        make_maps=False,
        endpoint=False,
        strict_pinned_contract=True,
        allow_pinned_jitter_fallback=False,
        scale_fit_split_override=str(source_config.get("scale_fit_split") or "work"),
        paired_inverse_hand=args.hand_check_method == "negative-thickness",
    )
    run_logged(selected_cmd, logs, "expert_hand_selected", dry_run=args.dry_run)
    if args.hand_check_method == "negative-thickness":
        other_table = selected_run / "native_inverse_hand_dyn_calc.csv"
        other_run = selected_run
    else:
        other_run = out / "inverse"
        other_cmd = native_stage_cmd(
            python,
            dataset,
            structure,
            dataset.baseline_hkl,
            other_run,
            args,
            n_basis=n_basis,
            n_points=n_points,
            stage="expert_hand_inverse_explicit",
            pinned=elastic_nuisance,
            alternate_hand=True,
            make_maps=False,
            endpoint=False,
            strict_pinned_contract=True,
            allow_pinned_jitter_fallback=False,
            scale_fit_split_override=str(source_config.get("scale_fit_split") or "work"),
        )
        run_logged(
            other_cmd,
            logs,
            "expert_hand_inverse_explicit",
            dry_run=args.dry_run,
        )
        other_table = other_run / "native_dyn_calc.csv"
    diagnostic = out / "absolute_structure_diagnostic.json"
    run_logged(
        [
            python,
            str(engine_script("absolute_structure_diagnostic.py")),
            "--best",
            str(selected_run / "native_dyn_calc.csv"),
            "--other",
            str(other_table),
            "--out",
            str(diagnostic),
        ],
        logs,
        "expert_absolute_structure_diagnostic",
        dry_run=args.dry_run,
    )
    manifest = {
        "schema_version": 1,
        "kind": "grond_expert_hand_handoff",
        "status": "dry_run" if args.dry_run else "ready",
        "method": args.hand_check_method,
        "source_summary": str(source_path),
        "source_summary_sha256": sha256_file(source_path),
        "source_row_table": row_table,
        "source_row_table_sha256": sha256_file(Path(row_table)),
        "model": structure,
        "model_sha256": sha256_file(Path(structure)),
        "inverse_model": str(inverse_structure.resolve()),
        "inverse_model_sha256": (sha256_file(inverse_structure) if inverse_structure.is_file() else ""),
        "hkl": dataset.hkl,
        "hkl_sha256": sha256_file(Path(dataset.hkl)),
        "hkl_transform": dataset.hkl_transform,
        "n_basis": n_basis,
        "n_points": n_points,
        "source_absorption": source_absorption,
        "hand_check_absorption": (
            "disabled_for_matched_elastic_hand_contract" if elastic_only else "preserved_from_source_nuisance"
        ),
        "source_absorption_changed": bool(elastic_only and source_absorption_active),
        "known_configuration_used": False,
        "same_transform_rows_split_and_nonabsorptive_nuisance": True,
        "selected_row_table": str((selected_run / "native_dyn_calc.csv").resolve()),
        "other_row_table": str(other_table.resolve()),
        "diagnostic": str(diagnostic.resolve()),
        "next": {
            "refine_input_hand": (
                f"grond refine --source-summary {shlex.quote(str(source_path))} "
                f"--start-res {shlex.quote(structure)} --out-dir "
                f"{shlex.quote(str((out / 'refine_input_hand').resolve()))}"
            ),
            "refine_inverse_hand": (
                f"grond refine --source-summary {shlex.quote(str(source_path))} "
                f"--start-res {shlex.quote(str(inverse_structure.resolve()))} --out-dir "
                f"{shlex.quote(str((out / 'refine_inverse_hand').resolve()))}"
            ),
        },
    }
    manifest_path = out / "hand_handoff.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    payload = try_read_json(diagnostic)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "method": args.hand_check_method,
                "diagnostic": str(diagnostic.resolve()),
                "handoff": str(manifest_path.resolve()),
                "all": payload.get("all"),
                "free": payload.get("free"),
            },
            indent=2,
            allow_nan=True,
        )
    )
    return 0
