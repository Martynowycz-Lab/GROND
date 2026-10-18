"""Run a focused transform scout using the same defaults as the complete workflow."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Mapping
from pathlib import Path

from grond.cli_options.common import add_common_run_args
from grond.cli_support.execution import configure_progress
from grond.cli_support.tools import choose_python
from grond.commands.preparation import prep_from_args
from grond.io.config import DatasetSpec, PrepConfig, write_prep
from grond.io.structure_contract import check_sfac_contract, prepare_structure_for_legacy
from grond.runtime import auto_chunksize, auto_worker_count
from grond.workflows.decision import ALL_PROPER_SIGNED_PERMUTATIONS, apply_preflight, preflight_dataset, write_json
from grond.workflows.transform_selection import SCOUT_MAX_ROWS, choose_transform_by_scout, plan_split_candidates


def _fill_common_run_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill hidden run defaults for focused expert-stage subcommands."""

    parser = argparse.ArgumentParser(add_help=False)
    add_common_run_args(parser, expose_lambda_controls=False)
    defaults = parser.parse_args(["--out-dir", str(args.out_dir)])
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def cmd_scout(args: argparse.Namespace) -> int:
    """Run only preflight and coordinate-hand-invariant HKL transform scouting."""

    _fill_common_run_defaults(args)
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    python = choose_python(args.python)
    prep = prep_from_args(args)
    if isinstance((prep.metadata or {}).get("pets2_bridge"), Mapping):
        raise SystemExit(
            "Generic scout/thickness handoffs are not convergence-safe for native PETS2. "
            "Use GROND pets-evaluate with explicit converged N, Q, support, and thickness grid."
        )
    args.workers = auto_worker_count(args.workers, args.max_core_fraction)
    args.chunksize = auto_chunksize(args.chunksize, args.workers)
    write_prep(root / "source_prep.json", prep)
    original_structure = prep.structure
    prep.structure, structure_prep = prepare_structure_for_legacy(prep.structure, root)
    sfac_check = check_sfac_contract(prep.structure)
    (root / "sfac_check.json").write_text(json.dumps(sfac_check, indent=2) + "\n")
    if not sfac_check.get("ok", False):
        errors = "; ".join(
            str(item.get("message", "")) for item in sfac_check.get("messages", []) if item.get("severity") == "error"
        )
        raise SystemExit(
            "GROND model SFAC check failed before transform scouting. " + (errors or "See sfac_check.json for details.")
        )
    prep.metadata = dict(prep.metadata or {})
    if structure_prep.get("prepared"):
        prep.metadata["original_structure"] = original_structure
        prep.metadata["structure_preparation"] = structure_prep
    write_prep(root / "input_prep.json", prep)
    configure_progress(
        logs,
        resume=bool(args.resume),
        total=max(1, len(prep.datasets) * max(1, len(ALL_PROPER_SIGNED_PERMUTATIONS))),
    )
    selected: list[DatasetSpec] = []
    records: dict[str, object] = {}
    # One split candidate set for every crystal (see GROND auto); trusted transforms write no split contract.
    split_plan = None if getattr(args, "trust_hkl_transform", False) else plan_split_candidates(prep, args)
    for source in prep.datasets:
        preflight = preflight_dataset(
            source,
            requested_min_peak=source.min_peak,
            requested_geometry_percentile=source.geometry_percentile,
            requested_scout_rows=SCOUT_MAX_ROWS,
        )
        write_json(root / "decisions" / source.name / "preflight.json", preflight)
        if preflight.status == "fatal":
            raise RuntimeError(f"fatal preflight failure for {source.name}: {preflight.warnings}")
        prepared = apply_preflight(source, preflight)
        scout_rows = preflight.scout_max_rows if preflight.scout_max_rows > 0 else SCOUT_MAX_ROWS
        decided, decision = choose_transform_by_scout(
            prep,
            prepared,
            args,
            python,
            logs,
            root,
            scout_max_rows=scout_rows,
            split_plan=split_plan,
        )
        selected.append(decided)
        records[source.name] = {
            "preflight": str((root / "decisions" / source.name / "preflight.json").resolve()),
            "decision": str((root / "decisions" / source.name / "transform_decision.json").resolve()),
            "selected_transform": decided.hkl_transform,
            "selected_transform_claim_contract": decision.get("selected_transform_claim_contract"),
        }
    scouted = PrepConfig(
        structure=prep.structure,
        datasets=selected,
        baseline_hkl=prep.baseline_hkl,
        notes=prep.notes,
        default_hkl_transform=prep.default_hkl_transform,
        geometry_percentile=prep.geometry_percentile,
        parity_setup_id=prep.parity_setup_id,
        parity_setup_manifest_sha256=prep.parity_setup_manifest_sha256,
        metadata={
            **dict(prep.metadata or {}),
            "scouted_from": str((root / "input_prep.json").resolve()),
            "scout_root": str(root.resolve()),
        },
    )
    scouted_path = root / "scouted_prep.json"
    write_prep(scouted_path, scouted)
    handoff = {
        "schema_version": 1,
        "kind": "grond_expert_transform_handoff",
        "status": "dry_run" if args.dry_run else "ready",
        "input_prep": str((root / "input_prep.json").resolve()),
        "source_prep": str((root / "source_prep.json").resolve()),
        "original_structure": original_structure,
        "scout_structure": prep.structure,
        "structure_preparation": structure_prep,
        "sfac_check": str((root / "sfac_check.json").resolve()),
        "output_prep": str(scouted_path.resolve()),
        "datasets": records,
        "known_configuration_used": False,
        "next": {
            "thickness": (
                f"grond thickness --prep {shlex.quote(str(scouted_path.resolve()))} "
                f"--out-dir {shlex.quote(str((root / '02_thickness').resolve()))}"
            ),
            "fixed_model": (
                f"grond run --prep {shlex.quote(str(scouted_path.resolve()))} "
                f"--out-dir {shlex.quote(str((root / '03_fixed_model').resolve()))} "
                "--mode bloch --no-auto-decide --no-hand-control --no-maps --native-only"
            ),
        },
    }
    handoff_path = root / "scout_handoff.json"
    handoff_path.write_text(json.dumps(handoff, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": handoff["status"],
                "scouted_prep": str(scouted_path.resolve()),
                "handoff": str(handoff_path.resolve()),
                "selected_transforms": {dataset.name: dataset.hkl_transform for dataset in selected},
            },
            indent=2,
        )
    )
    return 0
