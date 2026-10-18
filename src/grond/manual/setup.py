"""Bind starting model and ordinary integration inputs before expert-stage decisions."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from grond.io.config import PrepConfig, parse_dataset_spec, write_prep
from grond.manual.execution import _run_stage_command
from grond.manual.state import (
    SCHEMA_VERSION,
    WORKFLOW_KIND,
    _next_command,
    _print_handoff,
    _read_json,
    _record_stage,
    _sha256,
    _write_json,
)
from grond.paths import PACKAGE_ROOT


def manual_setup(args) -> int:
    """Run the ``setup`` stage: check the model and datasets and write prep.json and manual_workflow.json.

    A fresh conventional SHELXL baseline is run only when a baseline HKL and a non-default ``--shelxl`` are given.
    """

    if not args.out_dir:
        raise SystemExit("manual setup requires --out-dir")
    if not args.structure:
        raise SystemExit("manual setup requires --structure")
    if not args.dataset:
        raise SystemExit("manual setup requires at least one --dataset NAME=PATH")
    root = Path(args.out_dir).expanduser().resolve()
    workflow_path = root / "manual_workflow.json"
    setup_dir = root / "00_setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    structure = Path(args.structure).expanduser().resolve()
    if not structure.is_file():
        raise SystemExit(f"Structure is missing: {structure}")
    baseline = Path(args.baseline_hkl).expanduser().resolve() if args.baseline_hkl else None
    if baseline is not None and not baseline.is_file():
        raise SystemExit(f"Baseline HKL is missing: {baseline}")
    from grond.io.structure_contract import check_sfac_contract

    sfac_check = check_sfac_contract(structure)
    _write_json(setup_dir / "sfac_check.json", sfac_check)
    if not sfac_check.get("ok", False):
        messages = [
            str(row.get("message") or "") for row in sfac_check.get("messages", []) if row.get("severity") == "error"
        ]
        raise SystemExit(
            "Manual workflow model SFAC check failed: "
            + ("; ".join(messages) or f"see {setup_dir / 'sfac_check.json'}")
        )
    datasets = [
        parse_dataset_spec(
            item,
            default_hkl_transform=str(args.hkl_transform or "identity"),
            geometry_percentile=float(args.geometry_percentile),
        )
        for item in args.dataset
    ]
    for dataset in datasets:
        if not Path(dataset.hkl).is_file():
            raise SystemExit(f"Dataset {dataset.name!r} is missing its row file: {dataset.hkl}")
    prep = PrepConfig(
        structure=str(structure),
        datasets=datasets,
        baseline_hkl=str(baseline) if baseline else "",
        notes=str(args.notes or ""),
        default_hkl_transform=str(args.hkl_transform or "identity"),
        geometry_percentile=float(args.geometry_percentile),
        parity_setup_id=str(getattr(args, "parity_setup_id", "") or ""),
        parity_setup_manifest_sha256=str(getattr(args, "parity_setup_manifest_sha256", "") or ""),
    )
    prep_path = setup_dir / "prep.json"
    write_prep(prep_path, prep)
    conventional_baseline: dict[str, object] = {}
    conventional_command: dict[str, object] = {}
    if baseline is not None and str(args.shelxl or "shelxl") != "shelxl":
        baseline_dir = setup_dir / "fresh_conventional_baseline"
        command_line = [
            str(Path(args.python).expanduser().absolute())  # keep a venv symlink
            if str(getattr(args, "python", "") or "").strip()
            else sys.executable,
            str(PACKAGE_ROOT / "engine" / "run_shelxl_baseline.py"),
            "--res",
            str(structure),
            "--hkl",
            str(baseline),
            "--out-dir",
            str(baseline_dir),
            "--shelxl",
            str(args.shelxl),
        ]
        conventional_command = _run_stage_command(command_line)
        conventional_baseline = _read_json(baseline_dir / "summary.json")
    workflow: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": WORKFLOW_KIND,
        "root": str(root),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prep": str(prep_path),
        "source_model": str(structure),
        "chosen_model": "",
        "datasets": [asdict(dataset) for dataset in datasets],
        "runtime_defaults": {
            "workers": int(getattr(args, "workers", 0) or 0),
            "python": str(Path(args.python).expanduser().absolute())  # keep a venv symlink
            if str(getattr(args, "python", "") or "").strip()
            else "",
            "config": str(Path(args.config).expanduser().resolve())
            if str(getattr(args, "config", "") or "").strip()
            else "",
            "shelxl": str(getattr(args, "shelxl", "shelxl") or "shelxl"),
        },
        "conventional_baseline": (
            str(setup_dir / "fresh_conventional_baseline" / "summary.json") if conventional_baseline else ""
        ),
        "stages": {},
    }
    outputs = {
        "prep": str(prep_path),
        "structure": str(structure),
        "structure_sha256": _sha256(structure),
        "sfac_check": str(setup_dir / "sfac_check.json"),
        "baseline_hkl": str(baseline) if baseline else "",
        "conventional_baseline": workflow["conventional_baseline"],
        "datasets": {dataset.name: dataset.hkl for dataset in datasets},
        "headline": {
            "datasets": len(datasets),
            "starting_model": str(structure),
            "baseline_hkl": str(baseline) if baseline else "not supplied",
        },
    }
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="setup",
        output_dir=setup_dir,
        status="ready",
        outputs=outputs,
        command={
            "setup": {"elapsed_seconds": 0.0, "status": "setup_only"},
            "fresh_conventional_baseline": conventional_command,
        },
        next_command=_next_command("scout", workflow_path, workflow=workflow),
    )
    _print_handoff(handoff)
    return 0
