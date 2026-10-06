"""Scout thickness and gradient candidates, then write the exact selected handoff."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from grond.cli_support.execution import configure_progress, run_logged
from grond.cli_support.tools import choose_python
from grond.cli_support.values import (
    centered_grid,
    format_numeric_grid,
    parse_int_grid,
    parse_name_grid,
    parse_numeric_grid,
)
from grond.commands.preparation import prep_from_args
from grond.io.config import write_prep
from grond.io.structure_contract import prepare_structure_for_legacy
from grond.runtime import auto_chunksize, auto_worker_count
from grond.workflows.decision import safe_label
from grond.workflows.model_selection import dataset_baseline_hkl
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.thickness_scout import (
    _float_from_mapping,
    _q_field_stability,
    _read_csv_dicts,
    _thickness_scout_variant_args,
    _thickness_summary_row,
    _write_csv,
    _write_thickness_scout_report,
)


def cmd_thickness_scout(args: argparse.Namespace) -> int:
    """Run a diagnostic thickness contract-sensitivity scout."""

    # --preset comes with the shared run options, but only grond auto and grond run apply it; refuse it here
    # rather than silently running with other settings than the ones asked for.
    if str(getattr(args, "preset", "standard") or "standard") != "standard":
        raise SystemExit(f"--preset {args.preset} applies to grond auto and grond run only, not to thickness-scout")
    args.out_dir = str(Path(args.out_dir).expanduser().resolve())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    python = choose_python(args.python)
    prep = prep_from_args(args)
    if isinstance((prep.metadata or {}).get("pets2_bridge"), Mapping):
        raise SystemExit(
            "Generic thickness scouting is not convergence-safe for native PETS2. "
            "Use grond pets-evaluate with explicit converged N, Q, reciprocal support, "
            "and thickness grid."
        )
    args.workers = auto_worker_count(args.workers, args.max_core_fraction)
    args.chunksize = auto_chunksize(args.chunksize, args.workers)
    original_structure = prep.structure
    prep.structure, structure_prep = prepare_structure_for_legacy(prep.structure, out_dir)
    prep.metadata = dict(prep.metadata or {})
    if structure_prep.get("prepared"):
        prep.metadata["original_structure"] = original_structure
        prep.metadata["structure_preparation"] = structure_prep
    write_prep(out_dir / "thickness_scout_prep.json", prep)

    basis_grid = parse_int_grid(args.n_basis_grid)
    if not basis_grid:
        raise SystemExit("Provide at least one integer in --n-basis-grid.")
    n_points = int(args.n_points)
    thickness_nm_grid = parse_numeric_grid(args.thickness_grid_nm)
    if not thickness_nm_grid:
        raise SystemExit("Provide at least one value in --thickness-grid-nm.")
    gradient_grid = parse_numeric_grid(args.gradient_grid)
    if not gradient_grid:
        raise SystemExit("Provide at least one value in --gradient-grid.")
    scale_modes = parse_name_grid(args.scale_modes)
    feature_modes = parse_name_grid(args.feature_modes)
    total = len(prep.datasets) * len(scale_modes) * len(feature_modes) * len(basis_grid)
    configure_progress(logs, resume=bool(args.resume), total=total)

    rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    full_grid_A = [10.0 * v for v in thickness_nm_grid]
    promoted_reference: dict[str, object] = {}

    for ds in prep.datasets:
        for scale_mode in scale_modes:
            for feature_mode in feature_modes:
                center_A = float("nan")
                center_grad = float("nan")
                for index, n_basis in enumerate(basis_grid):
                    if index == 0 or not args.local_confirmation:
                        thick_values_A = full_grid_A
                        grad_values = gradient_grid
                    else:
                        thick_values_A = (
                            centered_grid(
                                center_A,
                                window=10.0 * float(args.local_window_nm),
                                step=10.0 * float(args.local_step_nm),
                                lower=10.0 * float(args.thickness_min_nm),
                                upper=10.0 * float(args.thickness_max_nm),
                            )
                            or full_grid_A
                        )
                        grad_values = (
                            centered_grid(
                                center_grad,
                                window=float(args.local_gradient_window),
                                step=float(args.local_gradient_step),
                                lower=float(args.gradient_min),
                                upper=float(args.gradient_max),
                            )
                            or gradient_grid
                        )
                    thickness_grid_A = format_numeric_grid(thick_values_A)
                    gradient_grid_text = format_numeric_grid(grad_values)
                    variant = _thickness_scout_variant_args(
                        args,
                        scale_mode=scale_mode,
                        feature_mode=feature_mode,
                        thickness_grid_A=thickness_grid_A,
                        gradient_grid=gradient_grid_text,
                    )
                    label = f"thickness_{safe_label(ds.name)}_{safe_label(scale_mode)}_{safe_label(feature_mode)}_N{int(n_basis)}Q{n_points}"
                    run_dir = (
                        out_dir
                        / "runs"
                        / ds.name
                        / f"{safe_label(scale_mode)}_{safe_label(feature_mode)}_N{int(n_basis)}Q{n_points}"
                    )
                    cmd = native_stage_cmd(
                        python,
                        ds,
                        prep.structure,
                        dataset_baseline_hkl(prep, ds),
                        run_dir,
                        variant,
                        n_basis=int(n_basis),
                        n_points=n_points,
                        stage=label,
                        make_maps=False,
                        endpoint=False,
                    )
                    run_logged(cmd, logs, label, dry_run=bool(args.dry_run))
                    row = _thickness_summary_row(
                        dataset=ds.name,
                        scale_mode=scale_mode,
                        feature_mode=feature_mode,
                        n_basis=int(n_basis),
                        n_points=n_points,
                        run_dir=run_dir,
                        thickness_grid=thickness_grid_A,
                        gradient_grid=gradient_grid_text,
                    )
                    rows.append(row)
                    if not promoted_reference and row.get("status") == "ok":
                        promoted_reference = row
                    if math.isfinite(_float_from_mapping(row, "thickness_A")):
                        center_A = _float_from_mapping(row, "thickness_A")
                    if math.isfinite(_float_from_mapping(row, "thickness_grad_frac")):
                        center_grad = _float_from_mapping(row, "thickness_grad_frac")
                    for trial in _read_csv_dicts(run_dir / "nuisance_trials.csv"):
                        trial_rows.append(
                            {
                                "dataset": ds.name,
                                "scale_mode": scale_mode,
                                "feature_mode": feature_mode,
                                "n_basis": int(n_basis),
                                "n_points": n_points,
                                "run_dir": str(run_dir.resolve()),
                                **trial,
                            }
                        )

    q_rows: list[dict[str, object]] = []
    ref_csv = Path(str(promoted_reference.get("row_table", ""))) if promoted_reference else Path()
    if promoted_reference and ref_csv.exists():
        for row in rows:
            cand_csv = Path(str(row.get("row_table", "")))
            if cand_csv.exists():
                q_rows.append(
                    {
                        **{k: row.get(k, "") for k in ("dataset", "scale_mode", "feature_mode", "n_basis", "n_points")},
                        **_q_field_stability(ref_csv, cand_csv),
                    }
                )
    if rows:
        reference_thickness = _float_from_mapping(promoted_reference, "thickness_nm")
        for row in rows:
            thickness_nm = _float_from_mapping(row, "thickness_nm")
            if math.isfinite(reference_thickness) and math.isfinite(thickness_nm):
                delta_nm = abs(thickness_nm - reference_thickness)
                if delta_nm >= float(args.warning_shift_nm):
                    warnings.append(
                        f"{row.get('dataset')} {row.get('scale_mode')}/{row.get('feature_mode')} N{row.get('n_basis')} selected {thickness_nm:.1f} nm, {delta_nm:.1f} nm away from the promoted reference. Treat this as row-contract sensitivity, not a new physical thickness."
                    )

    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": "thickness-scout",
        "inputs": {
            "structure": prep.structure,
            "original_structure": original_structure,
            "datasets": [asdict(ds) for ds in prep.datasets],
            "n_basis_grid": basis_grid,
            "n_points": n_points,
            "thickness_grid_nm": thickness_nm_grid,
            "gradient_grid": gradient_grid,
            "scale_modes": scale_modes,
            "feature_modes": feature_modes,
        },
        "policy": {
            "interpretation": "GROND thickness is row-contract effective thickness, not a direct microscope thickness unless independently calibrated.",
            "auto_recommendation": "Use the early bare production-scale thickness for auto, pin it through LS1, and treat final-feature movement as a warning.",
            "warning_shift_nm": float(args.warning_shift_nm),
        },
        "promoted_reference": promoted_reference,
        "runs": rows,
        "q_field_stability": q_rows,
        "warnings": warnings,
    }
    _write_csv(out_dir / "thickness_scout_table.csv", rows)
    _write_csv(out_dir / "thickness_scout_trials.csv", trial_rows)
    _write_csv(out_dir / "q_field_stability.csv", q_rows)
    (out_dir / "thickness_scout_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    selected_by_dataset: dict[str, dict[str, object]] = {}
    for row in rows:
        dataset_name = str(row.get("dataset") or "dataset")
        if dataset_name not in selected_by_dataset and row.get("status") == "ok":
            selected_by_dataset[dataset_name] = row
    thickness_handoff = {
        "schema_version": 1,
        "kind": "grond_expert_thickness_handoff",
        "status": "dry_run" if args.dry_run else "ready",
        "source_prep": str((out_dir / "thickness_scout_prep.json").resolve()),
        "summary": str((out_dir / "thickness_scout_summary.json").resolve()),
        "selection_policy": (
            "first successful production-order contract per dataset; inspect the full "
            "table before accepting a different sensitivity branch"
        ),
        "datasets": {},
    }
    for dataset_name, row in selected_by_dataset.items():
        run_dir = Path(str(row.get("run_dir") or "")).expanduser().resolve()
        source_summary = run_dir / "summary.json"
        hand_out = out_dir / "03_hand" / dataset_name
        thickness_handoff["datasets"][dataset_name] = {
            "selected_run": str(run_dir),
            "selected_summary": str(source_summary),
            "selected_n_basis": row.get("n_basis"),
            "selected_n_points": row.get("n_points"),
            "selected_thickness_A": row.get("thickness_A"),
            "selected_thickness_grad_frac": row.get("thickness_grad_frac"),
            "next_hand_check": (
                f"grond hand-check --source-summary {shlex.quote(str(source_summary))} "
                f"--out-dir {shlex.quote(str(hand_out.resolve()))}"
            ),
        }
    (out_dir / "thickness_handoff.json").write_text(json.dumps(thickness_handoff, indent=2, allow_nan=True) + "\n")
    _write_thickness_scout_report(out_dir, summary)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "report": str((out_dir / "REPORT.md").resolve()),
                "handoff": str((out_dir / "thickness_handoff.json").resolve()),
                "warnings": warnings,
            },
            indent=2,
            allow_nan=True,
        )
    )
    return 0
