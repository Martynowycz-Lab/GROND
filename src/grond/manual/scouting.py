"""Record transform and thickness choices against the actual input hashes."""

from __future__ import annotations

import math
import re
import shutil
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from grond.hkl_transform import parse_hkl_transform_info
from grond.io.config import load_prep
from grond.manual.execution import _child_base, _run_stage_command, _tool_args, _verbose_child_output
from grond.manual.state import (
    _load_workflow,
    _next_command,
    _print_handoff,
    _read_json,
    _record_stage,
    _require_stage,
    _sha256,
    _summary_nuisance,
    _write_json,
)


def manual_scout(args) -> int:
    workflow_path, workflow = _load_workflow(args)
    _require_stage(workflow, "setup")
    root = Path(workflow["root"])
    out = root / "01_scout"
    out.mkdir(parents=True, exist_ok=True)
    source_prep = Path(workflow["prep"])
    warnings: list[str] = []
    if args.skip:
        from grond.workflows.decision import preflight_dataset

        asserted_prep = load_prep(source_prep)
        preflight_records: dict[str, object] = {}
        for dataset in asserted_prep.datasets:
            transform = parse_hkl_transform_info(dataset.hkl_transform)
            if not transform.proper:
                raise SystemExit(
                    f"manual scout --skip refuses improper transform {transform.canonical!r} for "
                    f"{dataset.name!r}; supply its proper sign-paired representation or run the scout"
                )
            decision = preflight_dataset(
                dataset,
                requested_min_peak=dataset.min_peak,
                requested_geometry_percentile=dataset.geometry_percentile,
                requested_scout_rows=0,
            )
            decision_path = out / "decisions" / dataset.name / "preflight.json"
            _write_json(decision_path, asdict(decision))
            preflight_records[dataset.name] = str(decision_path)
            if decision.status == "fatal":
                raise SystemExit(
                    f"manual scout --skip preflight failed for {dataset.name!r}: " + "; ".join(decision.warnings)
                )
            warnings.extend(f"{dataset.name}: {warning}" for warning in decision.warnings)
        scouted_prep = out / "scouted_prep.json"
        shutil.copy2(source_prep, scouted_prep)
        command = {
            "status": "skipped_user_asserted_proper_transforms",
            "preflight": preflight_records,
        }
        warnings.append("Transform scouting was explicitly skipped; prep-file transforms are a user assertion.")
        status = "skipped"
    else:
        command_line = [
            *_child_base(args),
            "scout",
            "--prep",
            str(source_prep),
            "--out-dir",
            str(out),
            "--workers",
            str(args.workers),
            "--transform-parity-policy",
            "proper-only",
            *_tool_args(args),
        ]
        command = _run_stage_command(command_line, verbose=_verbose_child_output(args))
        scouted_prep = out / "scouted_prep.json"
        status = "ready"
    if not scouted_prep.is_file():
        raise RuntimeError(f"Scout did not write {scouted_prep}")
    workflow["prep"] = str(scouted_prep)
    workflow["datasets"] = [asdict(dataset) for dataset in load_prep(scouted_prep).datasets]
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="scout",
        output_dir=out,
        status=status,
        outputs={
            "prep": str(scouted_prep),
            "scout_handoff": str(out / "scout_handoff.json") if not args.skip else "",
            "selected_transforms": {
                dataset.name: dataset.hkl_transform for dataset in load_prep(scouted_prep).datasets
            },
            "headline": {
                "proper_transforms": ", ".join(
                    f"{dataset.name}={dataset.hkl_transform}" for dataset in load_prep(scouted_prep).datasets
                ),
                "selection": "user asserted" if args.skip else "chemistry-blind scout",
            },
        },
        command=command,
        warnings=warnings,
        next_command=_next_command("thickness", workflow_path, workflow=workflow),
    )
    _print_handoff(handoff)
    return 0


def _summary_bindings(values: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit("--summary must be DATASET=/path/to/summary.json")
        name, path = value.split("=", 1)
        result[name.strip()] = str(Path(path).expanduser().resolve())
    return result


def _validate_reused_summary_binding(dataset, summary_path: Path) -> dict[str, object]:
    """Bind a reused thickness summary to the exact dataset and row policy."""

    summary, _nuisance, config = _summary_nuisance(summary_path)
    inputs = summary.get("inputs") if isinstance(summary.get("inputs"), Mapping) else {}
    hkl_input = inputs.get("hkl")
    hkl_record = hkl_input if isinstance(hkl_input, Mapping) else {}
    recorded_hkl_text = str(
        (hkl_record.get("path") if hkl_record else hkl_input) or config.get("hkl") or ""
    ).strip()
    if not recorded_hkl_text:
        raise SystemExit(f"Reused thickness summary does not record its source HKL: {summary_path}")
    recorded_hkl = Path(recorded_hkl_text).expanduser().resolve()
    expected_hkl = Path(dataset.hkl).expanduser().resolve()
    if not recorded_hkl.is_file() or not expected_hkl.is_file():
        raise SystemExit(
            f"Reused thickness summary is not bound to dataset {dataset.name!r}: "
            f"source HKL {recorded_hkl} or dataset HKL {expected_hkl} is unavailable"
        )
    current_source_hash = _sha256(recorded_hkl)
    current_dataset_hash = _sha256(expected_hkl)
    recorded_hashes: dict[str, str] = {}
    for field, value in (
        ("inputs.hkl_sha256", inputs.get("hkl_sha256")),
        ("inputs.hkl.sha256", hkl_record.get("sha256")),
    ):
        if value is None or value == "":
            continue
        recorded_hash = str(value).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
            raise SystemExit(
                f"Reused thickness summary has an invalid recorded source-HKL SHA256 in {field}: "
                f"summary={summary_path}, original_path={recorded_hkl}, recorded_hash={value!r}"
            )
        recorded_hashes[field] = recorded_hash
        if recorded_hash != current_source_hash or recorded_hash != current_dataset_hash:
            raise SystemExit(
                f"Reused thickness summary source HKL changed for dataset {dataset.name!r}: "
                f"summary={summary_path}; original_path={recorded_hkl}; "
                f"recorded_sha256={recorded_hash} ({field}); "
                f"current_source_path={recorded_hkl}, current_source_sha256={current_source_hash}; "
                f"current_dataset_path={expected_hkl}, current_dataset_sha256={current_dataset_hash}"
            )
    if current_source_hash != current_dataset_hash:
        raise SystemExit(
            f"Reused thickness summary is not bound to dataset {dataset.name!r}: "
            f"source HKL {recorded_hkl} ({current_source_hash}) does not match "
            f"{expected_hkl} ({current_dataset_hash})"
        )
    historical_source_verified = bool(recorded_hashes)
    verification_warnings = [] if historical_source_verified else [
        f"{dataset.name}: reused summary {summary_path} has no recorded source-HKL SHA256; "
        "current source and dataset files match, but identity with the historical calculation is unverified."
    ]
    recorded_transform = parse_hkl_transform_info(
        str(inputs.get("hkl_transform") or config.get("hkl_transform") or "identity")
    ).canonical
    expected_transform = parse_hkl_transform_info(dataset.hkl_transform).canonical
    if recorded_transform != expected_transform:
        raise SystemExit(
            f"Reused thickness summary transform mismatch for {dataset.name!r}: "
            f"{recorded_transform!r} != {expected_transform!r}"
        )
    for field, expected in (
        ("min_d_spacing", float(dataset.min_d_spacing)),
        ("min_peak", float(dataset.min_peak)),
        ("geometry_percentile", float(dataset.geometry_percentile)),
    ):
        recorded = config.get(field)
        if recorded is None or not math.isclose(float(recorded), expected, rel_tol=0.0, abs_tol=1e-9):
            raise SystemExit(
                f"Reused thickness summary {field} mismatch for {dataset.name!r}: {recorded!r} != {expected!r}"
            )
    return {
        "dataset": dataset.name,
        "summary": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "source_hkl": str(recorded_hkl),
        "source_hkl_sha256": current_source_hash,
        "recorded_source_hkl_hashes": recorded_hashes,
        "historical_source_verified": historical_source_verified,
        "verification_status": (
            "verified_recorded_source_hash"
            if historical_source_verified
            else "unverified_legacy_missing_source_hash"
        ),
        "warnings": verification_warnings,
        "hkl_transform": recorded_transform,
        "min_d_spacing": float(dataset.min_d_spacing),
        "min_peak": float(dataset.min_peak),
        "geometry_percentile": float(dataset.geometry_percentile),
    }


def manual_thickness(args) -> int:
    workflow_path, workflow = _load_workflow(args)
    _require_stage(workflow, "scout")
    root = Path(workflow["root"])
    out = root / "02_thickness"
    out.mkdir(parents=True, exist_ok=True)
    prep = load_prep(Path(workflow["prep"]))
    warnings: list[str] = []
    if args.skip:
        selected = _summary_bindings(args.summary)
        missing = sorted({dataset.name for dataset in prep.datasets} - set(selected))
        if missing:
            raise SystemExit(
                "Skipping thickness requires one --summary DATASET=summary.json for every dataset; "
                f"missing: {', '.join(missing)}"
            )
        bindings = {
            dataset.name: _validate_reused_summary_binding(dataset, Path(selected[dataset.name]))
            for dataset in prep.datasets
        }
        historical_sources_verified = all(binding["historical_source_verified"] for binding in bindings.values())
        command = {
            "status": (
                "skipped_with_hash_bound_supplied_summaries"
                if historical_sources_verified
                else "skipped_with_supplied_summaries_historical_hash_unavailable"
            ),
            "bindings": bindings,
        }
        warnings.extend(warning for binding in bindings.values() for warning in binding["warnings"])
        status = "skipped"
        warnings.append("Thickness calculation skipped; supplied summaries define the pinned nuisance contracts.")
        report = ""
    else:
        scale_modes = "production,no-detector" if args.expanded else "no-detector"
        feature_modes = "bare,final-diagnostic" if args.expanded else "bare"
        command_line = [
            *_child_base(args),
            "thickness",
            "--prep",
            str(workflow["prep"]),
            "--out-dir",
            str(out),
            "--workers",
            str(args.workers),
            "--n-basis-grid",
            args.n_basis_grid,
            "--n-points",
            str(min(int(args.n_points), 10)),
            "--thickness-grid-nm",
            args.thickness_grid_nm,
            "--gradient-grid",
            args.gradient_grid,
            "--scale-modes",
            scale_modes,
            "--feature-modes",
            feature_modes,
            "--detector-scale-bins",
            "0",
            *_tool_args(args),
        ]
        command = _run_stage_command(command_line, verbose=_verbose_child_output(args))
        source_handoff = _read_json(out / "thickness_handoff.json")
        selected = {name: str(row["selected_summary"]) for name, row in source_handoff.get("datasets", {}).items()}
        status = "ready"
        report = str(out / "REPORT.md")
    if set(selected) != {dataset.name for dataset in prep.datasets}:
        raise RuntimeError("Thickness handoff does not contain exactly one selected summary per dataset")
    for path in selected.values():
        _summary_nuisance(Path(path))
    thickness_headline: dict[str, object] = {}
    for name, path in selected.items():
        _summary, nuisance, _config = _summary_nuisance(Path(path))
        thickness_headline[name] = (
            f"{float(nuisance.get('thickness_A', float('nan'))) / 10.0:.1f} nm; "
            f"gradient={nuisance.get('thickness_grad_frac', 'n/a')}; "
            f"frame offset={nuisance.get('frame_offset', 'n/a')}"
        )
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="thickness",
        output_dir=out,
        status=status,
        outputs={"selected_summaries": selected, "report": report, "headline": thickness_headline},
        command=command,
        warnings=warnings,
        next_command=_next_command("hand", workflow_path, workflow=workflow),
    )
    _print_handoff(handoff)
    return 0
