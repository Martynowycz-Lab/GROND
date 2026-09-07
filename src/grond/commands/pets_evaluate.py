"""Evaluate the optional PETS comparison under its recorded geometry and model contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.execution import configure_progress, run_logged
from grond.cli_support.tools import choose_python
from grond.cli_support.values import read_json
from grond.hkl_transform import parse_hkl_transform_info
from grond.io.config import load_prep
from grond.macromolecule import sha256_file
from grond.paths import engine_script
from grond.runtime import auto_chunksize, auto_worker_count
from grond.workflows.decision import safe_float


def cmd_pets_evaluate(args: argparse.Namespace) -> int:
    """Run a bounded native PETS2 forward/thickness/hand calculation."""

    prep = load_prep(Path(args.prep).expanduser().resolve())
    bridge = (prep.metadata or {}).get("pets2_bridge")
    if not isinstance(bridge, Mapping):
        raise SystemExit("--prep must be written by GROND import-pets")
    if len(prep.datasets) != 1:
        raise SystemExit("PETS native evaluation currently requires exactly one dataset")
    dataset = prep.datasets[0]
    structure = str(
        Path(args.structure).expanduser().resolve()
        if str(args.structure).strip()
        else Path(prep.structure).expanduser().resolve()
    )
    if not Path(structure).is_file():
        raise SystemExit(f"PETS evaluation structure does not exist: {structure}")
    transform = parse_hkl_transform_info(dataset.hkl_transform)
    if transform.canonical != "identity":
        raise SystemExit(
            "PETS native evaluation currently requires identity indexing; nonidentity "
            "setting conjugation is not yet implemented, and coordinate hand is evaluated "
            "separately by negative-thickness propagation"
        )
    if int(args.n_basis) < 2 or int(args.n_points) < 1:
        raise SystemExit("--n-basis must be >=2 and --n-points must be >=1")
    if float(args.topup_d_min) <= 0.0 or float(args.max_support_sg) <= 0.0:
        raise SystemExit("--topup-d-min and --max-support-sg must be positive")
    if not str(args.thickness_grid).strip():
        raise SystemExit("--thickness-grid must name at least one positive thickness in Angstrom")

    out = Path(args.out_dir).expanduser().resolve()
    logs = out / "logs"
    native = out / "native"
    jana = out / "jana_rows"
    python = choose_python(args.python)
    workers = auto_worker_count(int(args.workers), 0.85)
    chunksize = auto_chunksize(int(args.chunksize), workers)
    configure_progress(
        logs,
        resume=bool(args.resume),
        total=3 if bool(args.paired_inverse_hand) else 2,
        begin_invocation=True,
    )
    baseline_hkl = str(prep.baseline_hkl or dataset.baseline_hkl or dataset.hkl)
    command = [
        python,
        str(engine_script("native_row_regularize.py")),
        "--hkl",
        str(dataset.hkl),
        "--res",
        structure,
        "--baseline-hkl",
        baseline_hkl,
        "--hkl-transform",
        str(dataset.hkl_transform),
        "--out-dir",
        str(native),
        "--n-basis",
        str(int(args.n_basis)),
        "--n-points",
        str(int(args.n_points)),
        "--min-d-spacing",
        f"{float(dataset.min_d_spacing):.12g}",
        "--min-peak",
        "0",
        "--geometry-percentile",
        "100",
        "--target-mode",
        "iobs",
        "--include-nonpositive",
        "--thickness-grid",
        str(args.thickness_grid),
        "--thickness-model-grid",
        "single",
        "--thickness-grad-grid",
        "0",
        "--thickness-sigma-frac-grid",
        "0",
        "--frame-offset-grid",
        "0",
        "--absorption-mode-grid",
        "none",
        "--sg-jitter-mode-grid",
        "none",
        "--mosaicity-mode",
        "none",
        "--group-width-grid",
        "1",
        "--detector-scale-bins",
        "0",
        "--sigma-a-grid",
        "0",
        "--sigma-b-corr-grid",
        "0",
        "--sigma-c-peak-grid",
        "0",
        "--wght-a-grid",
        "0",
        "--wght-b-grid",
        "0",
        "--thickness-points",
        "1",
        "--topup-d-min",
        f"{float(args.topup_d_min):.12g}",
        "--max-support-sg",
        f"{float(args.max_support_sg):.12g}",
        "--free-fraction",
        f"{float(args.free_fraction):.12g}",
        "--split-seed",
        str(int(args.split_seed)),
        "--scale-fit-split",
        "work",
        "--selection-metric",
        "free_wR2",
        "--lambda-grid",
        "1.0",
        "--profile-quadrature",
        "uniform",
        "--profile-accumulation",
        "sum",
        "--compute-backend",
        "scipy",
        "--row-plan-cache",
        "off",
        "--prediction-cache-dir",
        str(out / "cache"),
        "--ug-cache-dir",
        str(out / "cache" / "ug"),
        "--workers",
        str(workers),
        "--chunksize",
        str(chunksize),
        "--skip-endpoint-refinement",
    ]
    if bool(args.paired_inverse_hand):
        command.append("--paired-inverse-hand")
    run_logged(command, logs, "pets_native_evaluate", dry_run=bool(args.dry_run))

    hand_diagnostic = native / "absolute_structure_diagnostic.json"
    hand_command: list[str] = []
    if bool(args.paired_inverse_hand):
        hand_command = [
            python,
            str(engine_script("absolute_structure_diagnostic.py")),
            "--best",
            str(native / "native_dyn_calc.csv"),
            "--other",
            str(native / "native_inverse_hand_dyn_calc.csv"),
            "--out",
            str(hand_diagnostic),
        ]
        run_logged(
            hand_command,
            logs,
            "pets_absolute_structure_diagnostic",
            dry_run=bool(args.dry_run),
        )

    report_command = [
        python,
        str(engine_script("row_mr_report.py")),
        "--row-table",
        f"{dataset.name}={native / 'native_dyn_calc.csv'}",
        "--res",
        structure,
        "--out-dir",
        str(jana),
        "--title",
        f"PETS2 native row statistics: {dataset.name}",
        "--instability-u",
        f"{float(args.instability_u):.12g}",
        "--no-pdf",
    ]
    run_logged(
        report_command,
        logs,
        "pets_jana_row_report",
        dry_run=bool(args.dry_run),
    )
    if bool(args.dry_run):
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "native_command": command,
                    "hand_command": hand_command or None,
                    "row_report_command": report_command,
                },
                indent=2,
            )
        )
        return 0
    native_summary_path = native / "summary.json"
    native_rows_path = native / "native_dyn_calc.csv"
    jana_path = jana / "JANA_ROW_STATISTICS.json"
    native_summary = read_json(native_summary_path)
    jana_summary = read_json(jana_path)
    hand_summary = read_json(hand_diagnostic) if bool(args.paired_inverse_hand) else {}
    best_native = native_summary.get("best_native")
    best_native = best_native if isinstance(best_native, Mapping) else {}
    nuisance = best_native.get("nuisance")
    nuisance = nuisance if isinstance(nuisance, Mapping) else {}
    row_mr = jana_summary.get("row_mr")
    row_mr = row_mr if isinstance(row_mr, Mapping) else {}
    payload = {
        "schema": "grond.pets2-native-evaluation.v1",
        "status": "complete",
        "prep": str(Path(args.prep).expanduser().resolve()),
        "structure": structure,
        "structure_sha256": sha256_file(Path(structure)),
        "native_summary": str(native_summary_path.resolve()),
        "native_summary_sha256": sha256_file(native_summary_path),
        "native_rows": str(native_rows_path.resolve()),
        "native_rows_sha256": sha256_file(native_rows_path),
        "hand_diagnostic": (str(hand_diagnostic.resolve()) if bool(args.paired_inverse_hand) else None),
        "hand_diagnostic_sha256": (sha256_file(hand_diagnostic) if bool(args.paired_inverse_hand) else None),
        "jana_row_statistics": str(jana_path.resolve()),
        "jana_row_statistics_sha256": sha256_file(jana_path),
        "selected_thickness_A": safe_float(nuisance.get("thickness_A")),
        "selected_thickness_nm": safe_float(nuisance.get("thickness_A")) / 10.0,
        "frame_scale_count": int(best_native.get("frame_scale_count") or 0),
        "detector_scale_count": int(best_native.get("detector_scale_count") or 0),
        "applied_frame_groups": len(best_native.get("dyn_scales") or {}),
        "applied_detector_groups": len(best_native.get("dyn_detector_scales") or {}),
        "row_statistics": {
            "born": row_mr.get("born"),
            "bloch": row_mr.get("dynamic"),
        },
        "hand_statistics": {
            "all": hand_summary.get("all"),
            "work": hand_summary.get("work"),
            "free": hand_summary.get("free"),
        }
        if hand_summary
        else None,
        "scientific_scope": (
            "native PETS row-space Born/Bloch, thickness, and elastic hand evaluation; no corrected SHELXL endpoint"
        ),
        "hkl_transform": transform.to_dict(),
    }
    (out / "PETS_NATIVE_EVALUATION.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0
