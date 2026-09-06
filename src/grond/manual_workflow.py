from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .hkl_transform import parse_hkl_transform_info
from .io.config import PrepConfig, load_prep, parse_dataset_spec, write_prep
from .legacy.engine.shelx_diagnostics import parse_shelx_lst_metrics, parse_shelx_lst_pathologies
from .maps.native_dynamic import NativeDynamicMapOptions, write_native_dynamic_map_package

SCHEMA_VERSION = 1
WORKFLOW_KIND = "grond_manual_workflow"
STAGE_ORDER = ("setup", "scout", "thickness", "hand", "feedback", "scales", "finish")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    temporary.replace(path)


def _workflow_path(args) -> Path:
    text = str(getattr(args, "workflow", "") or "").strip()
    path = Path(text).expanduser().resolve() if text else (Path.cwd() / "manual_workflow.json").resolve()
    if not path.is_file():
        raise SystemExit(
            f"Manual workflow state not found: {path}\n"
            "Start with `grond manual setup --out-dir RUN ...`, then pass "
            "`--workflow RUN/manual_workflow.json`."
        )
    return path


def _load_workflow(args) -> tuple[Path, dict[str, Any]]:
    path = _workflow_path(args)
    workflow = _read_json(path)
    if workflow.get("kind") != WORKFLOW_KIND:
        raise SystemExit(f"Not a GROND manual workflow: {path}")
    defaults = workflow.get("runtime_defaults") if isinstance(workflow.get("runtime_defaults"), Mapping) else {}
    if hasattr(args, "workers") and int(getattr(args, "workers", 0) or 0) == 0 and defaults.get("workers"):
        args.workers = int(defaults["workers"])
    if hasattr(args, "python") and not str(getattr(args, "python", "") or "").strip() and defaults.get("python"):
        args.python = str(defaults["python"])
    if hasattr(args, "config") and not str(getattr(args, "config", "") or "").strip() and defaults.get("config"):
        args.config = str(defaults["config"])
    if hasattr(args, "shelxl") and str(getattr(args, "shelxl", "") or "shelxl") == "shelxl" and defaults.get("shelxl"):
        args.shelxl = str(defaults["shelxl"])
    return path, workflow


def _save_workflow(path: Path, workflow: dict[str, Any]) -> None:
    workflow["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_json(path, workflow)


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def _manual_stream_line(line: str, *, verbose: bool) -> bool:
    """Return whether a child line belongs on the concise controller stream."""

    if verbose:
        return True
    stripped = line.lstrip()
    return stripped.startswith(("STEP", "DONE", "FAILED", "WARN", "WAIT", "RERUN", "ERROR"))


def _run_stage_command(command: list[str], *, verbose: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    env = dict(os.environ)
    package_src = str(Path(__file__).resolve().parents[1])
    inherited = str(env.get("PYTHONPATH", "") or "")
    env["PYTHONPATH"] = package_src + (f"{os.pathsep}{inherited}" if inherited else "")
    env["PYTHONUNBUFFERED"] = "1"
    heartbeat_seconds = max(
        0.01,
        float(env.get("GROND_MANUAL_HEARTBEAT_SECONDS", "30") or 30.0),
    )
    out_dir: Path | None = None
    if "--out-dir" in command:
        out_dir = Path(command[command.index("--out-dir") + 1]).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        env["GROND_CACHE_DIR"] = str(out_dir.parent / "cache")
    if len(command) > 3 and command[1:3] == ["-m", "grond.cli"]:
        child_name = command[3]
    elif len(command) > 1:
        child_name = Path(command[1]).stem
    else:
        child_name = Path(command[0]).name
    log_path = (out_dir or Path.cwd()) / f"manual_{child_name.replace('-', '_')}.log"
    print(f"RUN  {child_name} -> {out_dir or Path.cwd()}", flush=True)
    continuing_log = log_path.exists() and log_path.stat().st_size > 0
    with log_path.open("a" if continuing_log else "w") as log:
        if continuing_log:
            log.write("\n[manual_resume]\n")
        log.write(f"$ {_command_text(command)}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            text=True,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                output_queue.put(line)
            output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        while True:
            try:
                line = output_queue.get(timeout=heartbeat_seconds)
            except queue.Empty:
                running_seconds = time.perf_counter() - started
                print(
                    f"WAIT {child_name} still running ({running_seconds:.0f}s); "
                    f"controller log {log_path}",
                    flush=True,
                )
                continue
            if line is None:
                break
            if _manual_stream_line(line, verbose=verbose):
                print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
        reader.join(timeout=1.0)
    elapsed = time.perf_counter() - started
    record = {
        "command": command,
        "command_text": _command_text(command),
        "returncode": int(returncode),
        "elapsed_seconds": float(elapsed),
        "log": str(log_path.resolve()),
    }
    if returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(
            f"Manual stage command failed with return code {returncode}: "
            f"{record['command_text']}\nLog: {log_path}\nLast log lines:\n{tail}"
        )
    print(f"DONE {child_name} in {elapsed:.1f}s; log {log_path}", flush=True)
    return record


def _command_elapsed(command: Mapping[str, object]) -> float | None:
    elapsed_values: list[float] = []

    def collect(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        if value.get("elapsed_seconds") is not None:
            elapsed_values.append(float(value["elapsed_seconds"]))
            return
        for nested in value.values():
            collect(nested)

    collect(command)
    return sum(elapsed_values) if elapsed_values else None


def _child_base(args) -> list[str]:
    python = str(getattr(args, "python", "") or sys.executable)
    return [python, "-m", "grond.cli"]


def _tool_args(args) -> list[str]:
    out: list[str] = []
    config = str(getattr(args, "config", "") or "").strip()
    if config:
        out += ["--config", str(Path(config).expanduser().resolve())]
    if bool(getattr(args, "allow_missing_tools", False)):
        out.append("--allow-missing-tools")
    return out


def _verbose_child_output(args) -> bool:
    return bool(getattr(args, "verbose_child_output", False))


def _next_command(
    stage: str,
    workflow_path: Path,
    extra: str = "",
    *,
    workflow: Mapping[str, object] | None = None,
) -> str:
    parts = ["grond", "manual", stage, "--workflow", str(workflow_path)]
    defaults = workflow.get("runtime_defaults") if isinstance(workflow, Mapping) and isinstance(workflow.get("runtime_defaults"), Mapping) else {}
    if stage in {"scout", "thickness", "hand", "feedback", "scales"}:
        if defaults.get("workers"):
            parts += ["--workers", str(defaults["workers"])]
        if defaults.get("python"):
            parts += ["--python", str(defaults["python"])]
        if defaults.get("config"):
            parts += ["--config", str(defaults["config"])]
    if stage == "feedback" and defaults.get("shelxl"):
        parts += ["--shelxl", str(defaults["shelxl"])]
    text = shlex.join(parts)
    return text + (f" {extra.strip()}" if extra.strip() else "")


def _record_stage(
    workflow_path: Path,
    workflow: dict[str, Any],
    *,
    name: str,
    output_dir: Path,
    status: str,
    outputs: Mapping[str, object],
    command: Mapping[str, object] | None = None,
    next_command: str = "",
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    stages = workflow.setdefault("stages", {})
    invalidated: list[str] = []
    if name in STAGE_ORDER:
        current_index = STAGE_ORDER.index(name)
        for later_name in STAGE_ORDER[current_index + 1 :]:
            if later_name in stages:
                invalidated.append(later_name)
                stages.pop(later_name)
    if name in {"scout", "thickness"}:
        workflow["chosen_model"] = ""
    handoff = {
        "schema_version": SCHEMA_VERSION,
        "kind": "grond_manual_stage_handoff",
        "stage": name,
        "status": status,
        "workflow": str(workflow_path),
        "output_dir": str(output_dir),
        "outputs": dict(outputs),
        "command": dict(command or {}),
        "warnings": list(warnings or []),
        "invalidated_downstream_stages": invalidated,
        "next_command": next_command,
    }
    handoff_path = output_dir / "handoff.json"
    handoff["handoff"] = str(handoff_path)
    _write_json(handoff_path, handoff)
    stages[name] = handoff
    workflow["current_stage"] = name
    workflow["next_command"] = next_command
    _save_workflow(workflow_path, workflow)
    return handoff


def _print_handoff(handoff: Mapping[str, object]) -> None:
    outputs = handoff.get("outputs") if isinstance(handoff.get("outputs"), Mapping) else {}
    headline = outputs.get("headline") if isinstance(outputs.get("headline"), Mapping) else {}
    print(f"\nRESULT {handoff.get('stage')}: {handoff.get('status')}")
    for label, value in headline.items():
        print(f"  {str(label).replace('_', ' ')}: {value}")
    for warning in handoff.get("warnings") or []:
        print(f"  warning: {warning}")
    if handoff.get("next_command"):
        print(f"  next: {handoff['next_command']}")
    print()
    payload: dict[str, object] = {
        "stage": handoff.get("stage"),
        "status": handoff.get("status"),
        "workflow": handoff.get("workflow"),
        "handoff": handoff.get("handoff"),
    }
    for key in (
        "prep",
        "recommendation",
        "combined_evidence",
        "selected_model",
        "report",
        "summary_json",
        "summary_markdown",
        "map_manifest",
        "headline",
    ):
        if outputs.get(key):
            payload[key] = outputs[key]
    if handoff.get("warnings"):
        payload["warnings"] = handoff["warnings"]
    if handoff.get("invalidated_downstream_stages"):
        payload["invalidated_downstream_stages"] = handoff["invalidated_downstream_stages"]
    if handoff.get("next_command"):
        payload["next_command"] = handoff["next_command"]
    print(json.dumps(payload, indent=2, allow_nan=True))


def _stage(workflow: Mapping[str, object], name: str) -> dict[str, Any]:
    stages = workflow.get("stages")
    value = stages.get(name) if isinstance(stages, Mapping) else None
    return dict(value) if isinstance(value, Mapping) else {}


def _require_stage(workflow: Mapping[str, object], name: str) -> dict[str, Any]:
    stage = _stage(workflow, name)
    if not stage or stage.get("status") not in {"ready", "complete", "skipped", "needs_choice"}:
        raise SystemExit(f"Manual stage {name!r} is not ready. Current next command: {workflow.get('next_command', '')}")
    return stage


def _summary_nuisance(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary = _read_json(path)
    best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
    nuisance = best.get("nuisance") if isinstance(best.get("nuisance"), Mapping) else {}
    config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    if not nuisance:
        raise SystemExit(f"Selected summary has no best-native nuisance contract: {path}")
    return summary, dict(nuisance), dict(config)


def manual_setup(args) -> int:
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
    from .cli import check_sfac_contract

    sfac_check = check_sfac_contract(structure)
    _write_json(setup_dir / "sfac_check.json", sfac_check)
    if not sfac_check.get("ok", False):
        messages = [
            str(row.get("message") or "")
            for row in sfac_check.get("messages", [])
            if row.get("severity") == "error"
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
        parity_setup_manifest_sha256=str(
            getattr(args, "parity_setup_manifest_sha256", "") or ""
        ),
    )
    prep_path = setup_dir / "prep.json"
    write_prep(prep_path, prep)
    conventional_baseline: dict[str, object] = {}
    conventional_command: dict[str, object] = {}
    if baseline is not None and str(args.shelxl or "shelxl") != "shelxl":
        baseline_dir = setup_dir / "fresh_conventional_baseline"
        command_line = [
            str(Path(args.python).expanduser().resolve()) if str(getattr(args, "python", "") or "").strip() else sys.executable,
            str(Path(__file__).resolve().parent / "legacy" / "engine" / "run_shelxl_baseline.py"),
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
            "python": str(Path(args.python).expanduser().resolve()) if str(getattr(args, "python", "") or "").strip() else "",
            "config": str(Path(args.config).expanduser().resolve()) if str(getattr(args, "config", "") or "").strip() else "",
            "shelxl": str(getattr(args, "shelxl", "shelxl") or "shelxl"),
        },
        "conventional_baseline": (
            str(setup_dir / "fresh_conventional_baseline" / "summary.json")
            if conventional_baseline
            else ""
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


def manual_scout(args) -> int:
    workflow_path, workflow = _load_workflow(args)
    _require_stage(workflow, "setup")
    root = Path(workflow["root"])
    out = root / "01_scout"
    out.mkdir(parents=True, exist_ok=True)
    source_prep = Path(workflow["prep"])
    warnings: list[str] = []
    if args.skip:
        from .workflows.decision import preflight_dataset

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
                    f"manual scout --skip preflight failed for {dataset.name!r}: "
                    + "; ".join(decision.warnings)
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
                    f"{dataset.name}={dataset.hkl_transform}"
                    for dataset in load_prep(scouted_prep).datasets
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
    recorded_hkl_text = str(inputs.get("hkl") or config.get("hkl") or "").strip()
    if not recorded_hkl_text:
        raise SystemExit(f"Reused thickness summary does not record its source HKL: {summary_path}")
    recorded_hkl = Path(recorded_hkl_text).expanduser().resolve()
    expected_hkl = Path(dataset.hkl).expanduser().resolve()
    if not recorded_hkl.is_file() or _sha256(recorded_hkl) != _sha256(expected_hkl):
        raise SystemExit(
            f"Reused thickness summary is not bound to dataset {dataset.name!r}: "
            f"source HKL {recorded_hkl} does not match {expected_hkl}"
        )
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
                f"Reused thickness summary {field} mismatch for {dataset.name!r}: "
                f"{recorded!r} != {expected!r}"
            )
    return {
        "dataset": dataset.name,
        "summary": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "source_hkl": str(recorded_hkl),
        "source_hkl_sha256": _sha256(recorded_hkl),
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
        command = {"status": "skipped_with_hash_bound_supplied_summaries", "bindings": bindings}
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
        selected = {
            name: str(row["selected_summary"])
            for name, row in source_handoff.get("datasets", {}).items()
        }
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


def _hand_evidence(
    diagnostics: Mapping[str, Mapping[str, object]],
    *,
    min_z_all: float = 3.5,
    min_z_free: float = 3.5,
    min_free_rows: int = 10,
) -> dict[str, object]:
    min_z_all = float(min_z_all)
    if not math.isfinite(min_z_all) or min_z_all <= 0.0:
        raise ValueError("all-row hand threshold must be finite and positive")
    all_delta = 0.0
    free_delta = 0.0
    all_n = 0
    free_n = 0
    rows: dict[str, object] = {}
    z_numerators = {"all": 0.0, "free": 0.0}
    z_variances = {"all": 0.0, "free": 0.0}
    z_datasets = {"all": 0, "free": 0}
    calculation_errors: list[str] = []
    for name, diagnostic in diagnostics.items():
        if diagnostic.get("calculation_status") == "error":
            calculation_errors.extend(f"{name}: {error}" for error in diagnostic.get("calculation_errors", []))
        all_block = diagnostic.get("all") if isinstance(diagnostic.get("all"), Mapping) else {}
        free_block = diagnostic.get("free") if isinstance(diagnostic.get("free"), Mapping) else {}
        delta_all = float(all_block.get("delta_chi2_other_minus_best", 0.0) or 0.0)
        delta_free = float(free_block.get("delta_chi2_other_minus_best", 0.0) or 0.0)
        all_delta += delta_all
        free_delta += delta_free
        all_n += int(all_block.get("n", 0) or 0)
        free_n += int(free_block.get("n", 0) or 0)
        rows[name] = {
            "all_delta_chi2_inverse_minus_input": delta_all,
            "free_delta_chi2_inverse_minus_input": delta_free,
            "all": all_block,
            "free": free_block,
        }
        for support, block in (("all", all_block), ("free", free_block)):
            adjusted = block.get("klar_palatinus_adjusted_hand_z")
            adjusted = adjusted if isinstance(adjusted, Mapping) else {}
            n = int(adjusted.get("n", 0) or 0)
            k_effective = float(
                adjusted.get(
                    "k_effective_selected_wins_plus_half_ties",
                    adjusted.get("k_selected_lower_abs_intensity_residual", float("nan")),
                )
            )
            w = float(adjusted.get("w", float("nan")))
            if (
                str(adjusted.get("status") or "") == "ok"
                and n > 0
                and math.isfinite(k_effective)
                and math.isfinite(w)
                and n - w > 0.0
            ):
                z_numerators[support] += 2.0 * k_effective - n
                z_variances[support] += n - w
                z_datasets[support] += 1
    combined_z = {
        support: (
            z_numerators[support] / math.sqrt(z_variances[support])
            if z_datasets[support] > 0 and z_variances[support] > 0.0
            else float("nan")
        )
        for support in ("all", "free")
    }
    complete = bool(diagnostics) and z_datasets["all"] == len(diagnostics) and not calculation_errors
    if not complete and not calculation_errors:
        calculation_errors.append("one or more datasets lack a complete all-row adjusted hand calculation")
    decisive_support = bool(complete and math.isfinite(combined_z["all"]) and abs(combined_z["all"]) >= float(min_z_all))
    if (
        decisive_support
        and combined_z["all"] >= float(min_z_all)
    ):
        recommendation = "input"
    elif (
        decisive_support
        and combined_z["all"] <= -float(min_z_all)
    ):
        recommendation = "inverse"
    else:
        recommendation = "ambiguous"
    selected_sign = -1.0 if recommendation == "inverse" else 1.0
    oriented_free_z = selected_sign * combined_z["free"]
    warnings: list[str] = []
    preferred = (
        "input" if complete and combined_z["all"] > 0.0 else "inverse"
        if complete and combined_z["all"] < 0.0 else "no_preference"
        if complete else "unavailable"
    )
    if complete and not decisive_support:
        warnings.append(f"All-row hand preference is weak or tied (ZKP={combined_z['all']:.3g}); the calculated preference is retained in the report.")
    warnings.append("Adjusted ZKP is a conditional count contrast, not calibrated sigma confidence under correlated rows.")
    if not math.isfinite(combined_z["free"]):
        warnings.append("Held-out Klar-Palatinus adjusted hand Z is unavailable.")
    elif oriented_free_z < 0.0:
        warnings.append(
            f"Held-out ZKP {oriented_free_z:.3g} conflicts with the all-row recommended hand."
        )
    elif oriented_free_z < float(min_z_free):
        warnings.append(
            f"Held-out ZKP {oriented_free_z:.3g} supports the recommended hand only weakly "
            f"(< {float(min_z_free):.3g})."
        )
    if free_n < int(min_free_rows):
        warnings.append(
            f"Held-out hand evidence contains {free_n} rows, below the reporting target {int(min_free_rows)}."
        )
    return {
        "definition": "positive delta chi2 means the input coordinate hand has lower residual; no known chemistry is used",
        "datasets": rows,
        "combined_all_delta_chi2_inverse_minus_input": all_delta,
        "combined_free_delta_chi2_inverse_minus_input": free_delta,
        "combined_all_rows": all_n,
        "combined_free_rows": free_n,
        "combined_adjusted_z_input_favored": combined_z,
        "minimum_adjusted_z_all": float(min_z_all),
        "minimum_adjusted_z_free": float(min_z_free),
        "minimum_free_rows": int(min_free_rows),
        "decisive_support": decisive_support,
        "recommendation": recommendation,
        "preferred_hand": preferred,
        "calculation_status": "complete" if complete else "error",
        "calculation_errors": calculation_errors,
        "hand_confidence": "strong_conditional" if decisive_support else "weak_conditional" if complete and combined_z["all"] else "no_sensitivity" if complete else "calculation_error",
        "primary_evidence": "all_rows",
        "free_evidence_role": "diagnostic_warning_only",
        "warnings": warnings,
        "decision_rule": (
            "recommend inversion when combined all-row adjusted ZKP <= -threshold; held-out evidence "
            "is reported and may warn but does not veto the recommendation"
        ),
    }


def manual_hand(args) -> int:
    workflow_path, workflow = _load_workflow(args)
    thickness = _require_stage(workflow, "thickness")
    root = Path(workflow["root"])
    out = root / "03_hand"
    out.mkdir(parents=True, exist_ok=True)
    prep = load_prep(Path(workflow["prep"]))
    choice = "skip" if bool(args.skip) else str(args.choice or "report")
    selected_summaries = dict(thickness["outputs"]["selected_summaries"])
    if choice == "skip":
        workflow["chosen_model"] = prep.structure
        handoff = _record_stage(
            workflow_path,
            workflow,
            name="hand",
            output_dir=out,
            status="skipped",
            outputs={
                "choice": "input",
                "chosen_model": prep.structure,
                "diagnostics": {},
                "headline": {
                    "recommendation": "not calculated",
                    "recorded_choice": "input coordinates retained",
                },
            },
            command={"status": "skipped_user_choice"},
            warnings=["Absolute-structure checking was explicitly skipped."],
            next_command=_next_command("feedback", workflow_path, workflow=workflow),
        )
        _print_handoff(handoff)
        return 0
    diagnostics: dict[str, dict[str, object]] = {}
    handoffs: dict[str, dict[str, object]] = {}
    commands: dict[str, object] = {}
    existing = _stage(workflow, "hand")
    reuse_report = bool(existing.get("status") == "needs_choice" and choice in {"input", "inverse"})
    if reuse_report:
        existing_outputs = existing.get("outputs") if isinstance(existing.get("outputs"), Mapping) else {}
        diagnostic_paths = existing_outputs.get("dataset_diagnostics") if isinstance(existing_outputs.get("dataset_diagnostics"), Mapping) else {}
        handoff_paths = existing_outputs.get("dataset_handoffs") if isinstance(existing_outputs.get("dataset_handoffs"), Mapping) else {}
        if set(diagnostic_paths) != {dataset.name for dataset in prep.datasets} or set(handoff_paths) != {
            dataset.name for dataset in prep.datasets
        }:
            reuse_report = False
        else:
            diagnostics = {name: _read_json(Path(path)) for name, path in diagnostic_paths.items()}
            handoffs = {name: _read_json(Path(path)) for name, path in handoff_paths.items()}
            commands = {
                "selection": {
                    "status": "reused_existing_hand_report_without_recalculation",
                    "elapsed_seconds": 0.0,
                },
                "evidence_calculation": dict(existing.get("command") or {}),
            }
    if not reuse_report:
        for dataset in prep.datasets:
            dataset_out = out / dataset.name
            command_line = [
                *_child_base(args),
                "hand-check",
                "--source-summary",
                selected_summaries[dataset.name],
                "--out-dir",
                str(dataset_out),
                "--workers",
                str(args.workers),
                "--hand-check-method",
                "negative-thickness",
                "--hand-check-absorption",
                "elastic",
                *_tool_args(args),
            ]
            commands[dataset.name] = _run_stage_command(
                command_line,
                verbose=_verbose_child_output(args),
            )
            diagnostics[dataset.name] = _read_json(dataset_out / "absolute_structure_diagnostic.json")
            handoffs[dataset.name] = _read_json(dataset_out / "hand_handoff.json")
    evidence = _hand_evidence(
        diagnostics,
        min_z_all=float(args.min_adjusted_z_all),
        min_z_free=float(args.min_adjusted_z_free),
        min_free_rows=int(args.min_free_rows),
    )
    evidence_path = out / "combined_hand_evidence.json"
    _write_json(evidence_path, evidence)
    if choice == "report":
        status = "needs_choice"
        chosen_model = ""
        workflow["chosen_model"] = ""
        recommended_choice = str(evidence["recommendation"])
        if recommended_choice not in {"input", "inverse"}:
            recommended_choice = "input"
        next_command = _next_command("hand", workflow_path, f"--choice {recommended_choice}", workflow=workflow)
    elif choice == "input":
        status = "ready"
        chosen_model = prep.structure
        next_command = _next_command("feedback", workflow_path, workflow=workflow)
    elif choice == "inverse":
        inverse_models = [Path(str(handoffs[dataset.name]["inverse_model"])) for dataset in prep.datasets]
        hashes = {_sha256(path) for path in inverse_models}
        if len(hashes) != 1:
            raise RuntimeError("Per-dataset hand checks did not materialize one identical inverse-coordinate model")
        status = "ready"
        chosen_model = str(inverse_models[0].resolve())
        next_command = _next_command("feedback", workflow_path, workflow=workflow)
    else:
        raise SystemExit("--choice must be report, input, inverse, or skip")
    if chosen_model:
        workflow["chosen_model"] = chosen_model
    selected_sign = -1.0 if choice == "inverse" else 1.0
    headline_coordinate_label = "selected" if choice in {"input", "inverse"} else "input"
    hand_headline: dict[str, object] = {
        "recommendation": evidence["recommendation"],
        "recorded_choice": choice if choice != "report" else "awaiting explicit choice",
    }
    for dataset_name, diagnostic in diagnostics.items():
        all_block = diagnostic.get("all") if isinstance(diagnostic.get("all"), Mapping) else {}
        free_block = diagnostic.get("free") if isinstance(diagnostic.get("free"), Mapping) else {}
        all_z = all_block.get("klar_palatinus_adjusted_hand_z", {})
        free_z = free_block.get("klar_palatinus_adjusted_hand_z", {})
        hand_headline[f"{dataset_name}_ZKP_{headline_coordinate_label}_all/free"] = (
            f"{selected_sign * float(all_z.get('z', float('nan'))):+.2f} / "
            f"{selected_sign * float(free_z.get('z', float('nan'))):+.2f}"
        )
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="hand",
        output_dir=out,
        status=status,
        outputs={
            "choice": choice,
            "chosen_model": chosen_model,
            "combined_evidence": str(evidence_path),
            "recommendation": evidence["recommendation"],
            "dataset_handoffs": {name: str(out / name / "hand_handoff.json") for name in handoffs},
            "dataset_diagnostics": {
                name: str(out / name / "absolute_structure_diagnostic.json") for name in diagnostics
            },
            "headline": hand_headline,
        },
        command={"datasets": commands},
        warnings=list(evidence.get("warnings") or [])
        + (
            ["All-row hand evidence is below the decision threshold; inspect both hands before choosing."]
            if evidence["recommendation"] == "ambiguous"
            else []
        ),
        next_command=next_command,
    )
    _print_handoff(handoff)
    return 0


def _manual_checkpoint(workflow: Mapping[str, object], output: Path) -> dict[str, object]:
    prep = load_prep(Path(str(workflow["prep"])))
    thickness = _stage(workflow, "thickness")
    selected = dict(thickness["outputs"]["selected_summaries"])
    datasets = []
    for dataset in prep.datasets:
        summary_path = Path(selected[dataset.name]).resolve()
        summary, nuisance, config = _summary_nuisance(summary_path)
        final_config = {
            key: config[key]
            for key in (
                "hkl_transform",
                "n_basis",
                "n_points",
                "topup_d_min",
                "compute_backend",
                "row_plan_cache",
                "chunksize",
                "post_bloch_sigma_policy",
                "min_d_spacing",
                "min_peak",
                "geometry_percentile",
            )
            if key in config and config[key] not in (None, "")
        }
        datasets.append(
            {
                **asdict(dataset),
                "source_run": str(summary_path.parent),
                "source_summary": str(summary_path),
                "nuisance": nuisance,
                "final_config": final_config,
                "source_summary_sha256": _sha256(summary_path),
                "row_table": str(summary_path.parent / "native_dyn_calc.csv"),
            }
        )
    model = Path(str(workflow.get("chosen_model") or prep.structure)).resolve()
    single_dataset = datasets[0] if len(datasets) == 1 else None
    checkpoint = {
        "schema_version": 1,
        "kind": "grond_continue_checkpoint",
        "status": "ready",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dir": str(output.parent),
        "dataset_count": len(datasets),
        "dataset": datasets[0]["name"],
        "source_run": str(single_dataset["source_run"]) if single_dataset else "",
        "source_summary": str(single_dataset["source_summary"]) if single_dataset else "",
        "recommended_start_res": str(model),
        "hkl": datasets[0]["hkl"],
        "baseline_hkl": prep.baseline_hkl,
        "hkl_transform": datasets[0]["hkl_transform"],
        "datasets": datasets,
        "structure_dataset_policy": {},
        "final_dataset_policy": {},
        "scale_contract_selection": {},
        "multi_endpoint_merge_key": "model_symmetry",
        "manual_workflow": str(workflow.get("root", "")),
    }
    _write_json(output, checkpoint)
    return checkpoint


def manual_feedback(args) -> int:
    workflow_path, workflow = _load_workflow(args)
    hand = _require_stage(workflow, "hand")
    if hand.get("status") == "needs_choice" or not workflow.get("chosen_model"):
        raise SystemExit(
            "Hand stage needs an explicit choice. Run the next command printed by `grond manual hand`, "
            "or use `grond manual hand --workflow ... --choice skip`."
        )
    root = Path(workflow["root"])
    out = root / "04_feedback"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "input_checkpoint.json"
    checkpoint = _manual_checkpoint(workflow, checkpoint_path)
    single_dataset = int(checkpoint.get("dataset_count", 0) or 0) == 1
    ignore_source_baseline = single_dataset and not str(checkpoint.get("baseline_hkl") or "").strip()
    command_line = [
        *_child_base(args),
        "refine",
        "--checkpoint",
        str(checkpoint_path),
        "--start-res",
        str(workflow["chosen_model"]),
        "--out-dir",
        str(out),
        "--cycles",
        str(args.cycles),
        "--n-points",
        str(args.n_points),
        "--final-n-points",
        str(args.n_points),
        "--workers",
        str(args.workers),
        "--shelxl",
        args.shelxl,
        "--shelxl-iterations",
        "1",
        "--endpoint-ls-cycles",
        "1",
        "--post-bloch-sigma-policy",
        "uniform_z_rms",
        "--lambda-grid",
        "1.0",
        "--row-plan-cache",
        "off",
        *_tool_args(args),
    ]
    if int(args.n_basis) > 0:
        command_line += ["--n-basis", str(args.n_basis)]
    if ignore_source_baseline:
        command_line.append("--ignore-source-baseline-hkl")
    command = _run_stage_command(command_line, verbose=_verbose_child_output(args))
    if single_dataset:
        push_summary_path = out / "shelxl_feedback_loop_summary.json"
        decision_path = out / "endpoint_push_decision.json"
        feedback_summary_path = push_summary_path
        report_path = out / "ENDPOINT_PUSH_REPORT.md"
        push_summary = _read_json(push_summary_path)
        decision = _read_json(decision_path)
        selected = decision.get("selected_safe_cycle")
    else:
        push_summary_path = out / "push_endpoint_summary.json"
        decision_path = out / "multi_endpoint_push_decision.json"
        feedback_summary_path = out / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"
        report_path = out / "MULTI_ENDPOINT_PUSH_REPORT.md"
        push_summary = _read_json(push_summary_path)
        decision = _read_json(decision_path)
        selected = decision.get("selected_cycle")
    if not isinstance(selected, Mapping) or not selected.get("model"):
        raise RuntimeError(
            "Feedback produced no clean selected model; diagnostic best-native/NPD cycles are not eligible for "
            f"manual handoff. Inspect {report_path}."
        )
    selected_model = str(Path(str(selected["model"])).resolve())
    selected_npd_count = int(selected.get("npd_count", len(selected.get("npd_atoms") or [])) or 0)
    selected_health = selected.get("endpoint_res_health") if isinstance(selected.get("endpoint_res_health"), Mapping) else {}
    if selected_npd_count > 0 or selected.get("has_npd") is True or selected_health.get("safe_for_feedback") is False:
        raise RuntimeError(
            f"Feedback selected only an unsafe diagnostic model (NPD={selected_npd_count}); "
            f"manual handoff requires a clean cycle. Inspect {report_path}."
        )
    feedback_summary = _read_json(feedback_summary_path)
    cycle_number = int(selected.get("cycle", 0) or 0)
    if single_dataset:
        cycle_record = next(
            (
                row
                for row in feedback_summary.get("runs", [])
                if str(row.get("label") or "") == f"cycle_{cycle_number:02d}"
            ),
            {},
        )
    else:
        cycle_record = next(
            (row for row in feedback_summary.get("cycles", []) if int(row.get("cycle", 0) or 0) == cycle_number),
            {},
        )
    selected_rows_out = out / "selected_rows"
    recompute_command = [
        *_child_base(args),
        "refine",
        "--checkpoint",
        str(checkpoint_path),
        "--start-res",
        selected_model,
        "--out-dir",
        str(selected_rows_out),
        "--cycles",
        "0",
        "--recompute-final-rows",
        "--n-points",
        str(args.n_points),
        "--final-n-points",
        str(args.n_points),
        "--workers",
        str(args.workers),
        "--shelxl",
        args.shelxl,
        "--row-plan-cache",
        "off",
        *_tool_args(args),
    ]
    if int(args.n_basis) > 0:
        recompute_command += ["--n-basis", str(args.n_basis)]
    if ignore_source_baseline:
        recompute_command.append("--ignore-source-baseline-hkl")
    recompute = _run_stage_command(recompute_command, verbose=_verbose_child_output(args))
    selected_feedback_path = (
        selected_rows_out / "shelxl_feedback_loop_summary.json"
        if single_dataset
        else selected_rows_out / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"
    )
    selected_feedback = _read_json(selected_feedback_path)
    if single_dataset:
        final_run_text = str(selected_feedback.get("final_row_stats_run") or "")
        final_run = Path(final_run_text)
        if not final_run.is_dir() or not (final_run / "native_dyn_calc.csv").is_file():
            raise RuntimeError("Selected-model single-dataset row recomputation did not complete")
        raw_runs = {str(checkpoint.get("dataset") or "dataset"): str(final_run.resolve())}
    else:
        final_rows = selected_feedback.get("final_row_recompute")
        if not isinstance(final_rows, Mapping) or final_rows.get("status") != "ok":
            raise RuntimeError("Selected-model row recomputation did not complete")
        raw_runs = dict(final_rows.get("raw_runs") or {})
    workflow["chosen_model"] = selected_model
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="feedback",
        output_dir=out,
        status="ready",
        outputs={
            "checkpoint": str(checkpoint_path),
            "selected_model": selected_model,
            "selected_cycle": cycle_number,
            "selected_endpoint_combined_dir": str(
                cycle_record.get("combined") or Path(selected_model).parent
            ),
            "selected_row_runs": raw_runs,
            "feedback_summary": str(feedback_summary_path),
            "selected_row_recompute_summary": str(selected_feedback_path),
            "report": str(report_path),
            "push_summary": str(push_summary_path),
            "feedback_status": push_summary.get("status") or ("ready" if selected_model else ""),
            "decision": str(decision_path),
            "headline": {
                "selected_cycle": cycle_number,
                "endpoint_R1all": f"{100.0 * float(selected.get('endpoint_R1_all', float('nan'))):.2f}%",
                "endpoint_wR2all": f"{100.0 * float(selected.get('endpoint_wR2_all', float('nan'))):.2f}%",
                "endpoint_GooF": selected.get("endpoint_GooF", "n/a"),
                "NPD/split": f"{selected.get('npd_count', len(selected.get('npd_atoms') or []))}/{selected.get('split_count', 'n/a')}",
                "coordinate_updates": (
                    "one single-crystal SHELXL LS1 per macrocycle"
                    if single_dataset
                    else "one joint inverse-variance SHELXL LS1 per macrocycle"
                ),
            },
        },
        command={"feedback": command, "selected_row_recompute": recompute},
        next_command=_next_command("scales", workflow_path, workflow=workflow),
    )
    _print_handoff(handoff)
    return 0


def manual_scales(args) -> int:
    workflow_path, workflow = _load_workflow(args)
    feedback = _require_stage(workflow, "feedback")
    root = Path(workflow["root"])
    out = root / "05_scales"
    out.mkdir(parents=True, exist_ok=True)
    if args.skip:
        handoff = _record_stage(
            workflow_path,
            workflow,
            name="scales",
            output_dir=out,
            status="skipped",
            outputs={
                "datasets": {},
                "application": "not run; full pinned scales remain active",
                "headline": {"application": "full pinned frame scales retained; compact audit skipped"},
            },
            command={"status": "skipped_user_choice"},
            next_command=_next_command("finish", workflow_path, workflow=workflow),
        )
        _print_handoff(handoff)
        return 0
    commands: dict[str, object] = {}
    outputs: dict[str, object] = {}
    for name, run_text in feedback["outputs"]["selected_row_runs"].items():
        run = Path(run_text)
        dataset_out = out / name
        command_line = [
            *_child_base(args),
            "scale-polish",
            "--row-table",
            str(run / "native_dyn_calc.csv"),
            "--summary",
            str(run / "summary.json"),
            "--out-dir",
            str(dataset_out),
            *_tool_args(args),
        ]
        commands[name] = _run_stage_command(command_line, verbose=_verbose_child_output(args))
        summary_path = dataset_out / "scale_compression_summary.json"
        summary = _read_json(summary_path)
        selected = summary.get("selected") if isinstance(summary.get("selected"), Mapping) else {}
        outputs[name] = {
            "summary": str(summary_path),
            "report": str(dataset_out / "REPORT.md"),
            "selected": {
                key: selected.get(key)
                for key in (
                    "name",
                    "frame_model",
                    "detector_bins",
                    "scale_params",
                    "group_scale_params",
                    "detector_scale_params",
                    "Robs",
                    "Rall",
                    "wRall",
                    "within_strict",
                )
            },
        }
    compact_frame_total = sum(
        int((row.get("selected") or {}).get("group_scale_params") or 0)
        for row in outputs.values()
        if isinstance(row, Mapping)
    )
    compact_detector_total = sum(
        int((row.get("selected") or {}).get("detector_scale_params") or 0)
        for row in outputs.values()
        if isinstance(row, Mapping)
    )
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="scales",
        output_dir=out,
        status="ready",
        outputs={
            "datasets": outputs,
            "application": "audit_only; full pinned scales remain active",
            "headline": {
                "compact_audit": f"{compact_frame_total} frame + {compact_detector_total} detector parameters",
                "application": "audit only; endpoint still uses the full pinned scale grouping",
            },
        },
        command={"datasets": commands},
        warnings=["Scale compression is audited but not silently applied to the shared endpoint."],
        next_command=_next_command("finish", workflow_path, workflow=workflow),
    )
    _print_handoff(handoff)
    return 0


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percent(value: object) -> str:
    number = _finite_float(value)
    return "n/a" if number is None else f"{100.0 * number:.2f}%"


def _number(value: object, digits: int = 2, *, signed: bool = False) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}"


def _selected_hand_summary(workflow: Mapping[str, object]) -> dict[str, object]:
    hand = _stage(workflow, "hand")
    outputs = hand.get("outputs") if isinstance(hand.get("outputs"), Mapping) else {}
    choice = str(outputs.get("choice") or "unknown")
    orientation = -1.0 if choice == "inverse" else 1.0
    dataset_paths = outputs.get("dataset_diagnostics")
    datasets: dict[str, object] = {}
    combined: dict[str, object] = {}
    numerators = {"all": 0.0, "free": 0.0}
    variances = {"all": 0.0, "free": 0.0}
    usable = {"all": 0, "free": 0}
    if isinstance(dataset_paths, Mapping):
        for name, path_text in dataset_paths.items():
            path = Path(str(path_text))
            if not path.is_file():
                continue
            diagnostic = _read_json(path)
            row: dict[str, object] = {"diagnostic": str(path)}
            for support in ("all", "free"):
                support_block = diagnostic.get(support)
                support_block = support_block if isinstance(support_block, Mapping) else {}
                kp = support_block.get("klar_palatinus_adjusted_hand_z")
                kp = kp if isinstance(kp, Mapping) else {}
                z_input = _finite_float(kp.get("z"))
                n = int(kp.get("n", 0) or 0)
                k_effective = _finite_float(
                    kp.get(
                        "k_selected_lower_abs_intensity_residual",
                        kp.get("k_effective_selected_wins_plus_half_ties"),
                    )
                )
                w = _finite_float(kp.get("w"))
                z_selected = orientation * z_input if z_input is not None else None
                row[support] = {
                    "n": n,
                    "z_input_coordinates": z_input,
                    "z_selected_coordinates": z_selected,
                    "paired_row_t_input_coordinates": _finite_float(
                        support_block.get("paired_row_t_delta_chi2")
                    ),
                }
                if k_effective is not None and w is not None and n > w:
                    numerators[support] += orientation * (2.0 * k_effective - n)
                    variances[support] += n - w
                    usable[support] += 1
            datasets[str(name)] = row
    for support in ("all", "free"):
        combined[support] = {
            "z_selected_coordinates": (
                numerators[support] / math.sqrt(variances[support])
                if usable[support] > 0 and variances[support] > 0.0
                else None
            ),
            "dataset_count": usable[support],
            "combination": "independent sum-score diagnostic: sum(2k_eff-N)/sqrt(sum(N-w))",
        }
    return {
        "scope": "pre_feedback_thickness_stage_model",
        "choice": choice,
        "recommendation": outputs.get("recommendation", "not calculated"),
        "selected_model": outputs.get("chosen_model", ""),
        "known_chemistry_used": False,
        "sign_definition": "positive Z favors the coordinates selected before feedback; no final post-feedback hand recheck is implied",
        "datasets": datasets,
        "combined": combined,
    }


def _manual_row_statistics(row_runs: Mapping[str, Path], model: Path) -> tuple[dict[str, object], dict[str, object]]:
    from .legacy.engine.bloch_engine import parse_model
    from .legacy.engine.row_mr_report import (
        compute_per_dataset_row_mr,
        compute_row_mr_summary,
        read_csv_rows,
        standard_json_data,
    )

    rows: list[dict] = []
    for name, run in row_runs.items():
        rows.extend(read_csv_rows(run / "native_dyn_calc.csv", name))
    parsed_model = parse_model(model)
    combined = compute_row_mr_summary(rows, parsed_model, instability_u=0.01)
    per_dataset = compute_per_dataset_row_mr(rows, parsed_model, instability_u=0.01)
    return standard_json_data(combined), standard_json_data(per_dataset)


def _manual_scale_summary(
    workflow: Mapping[str, object],
    row_runs: Mapping[str, Path],
) -> dict[str, object]:
    datasets: dict[str, object] = {}
    frame_total = 0
    detector_total = 0
    row_total = 0
    for name, run in row_runs.items():
        summary = _read_json(run / "summary.json")
        best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
        frame = int(best.get("frame_scale_count", 0) or 0)
        detector = int(best.get("detector_scale_count", 0) or 0)
        n_rows = int(best.get("valid_rows", 0) or 0)
        frame_total += frame
        detector_total += detector
        row_total += n_rows
        datasets[name] = {
            "rows": n_rows,
            "frame_parameters": frame,
            "detector_parameters": detector,
            "rows_per_scale_parameter": n_rows / max(frame + detector, 1),
            "fit_split": best.get("scale_fit_split", ""),
        }
    scales_stage = _stage(workflow, "scales")
    scale_outputs = scales_stage.get("outputs") if isinstance(scales_stage.get("outputs"), Mapping) else {}
    audit_datasets = scale_outputs.get("datasets") if isinstance(scale_outputs.get("datasets"), Mapping) else {}
    compact: dict[str, object] = {}
    compact_frame_total = 0
    compact_detector_total = 0
    for name, audit in audit_datasets.items():
        selected = audit.get("selected") if isinstance(audit, Mapping) else {}
        selected = selected if isinstance(selected, Mapping) else {}
        frame = int(selected.get("group_scale_params", 0) or 0)
        detector = int(selected.get("detector_scale_params", 0) or 0)
        compact_frame_total += frame
        compact_detector_total += detector
        compact[str(name)] = dict(selected)
    return {
        "applied_contract": "full pinned frame grouping",
        "application_note": scale_outputs.get(
            "application", "scale audit not run; full pinned scales remain active"
        ),
        "applied": {
            "rows": row_total,
            "frame_parameters": frame_total,
            "detector_parameters": detector_total,
            "rows_per_scale_parameter": row_total / max(frame_total + detector_total, 1),
            "datasets": datasets,
        },
        "compact_audit": {
            "applied_to_endpoint": False,
            "frame_parameters": compact_frame_total,
            "detector_parameters": compact_detector_total,
            "datasets": compact,
        },
    }


def _manual_markdown_summary(payload: Mapping[str, object]) -> str:
    endpoint = payload.get("endpoint") if isinstance(payload.get("endpoint"), Mapping) else {}
    baseline = payload.get("starting_endpoint") if isinstance(payload.get("starting_endpoint"), Mapping) else {}
    row_stats = payload.get("row_statistics") if isinstance(payload.get("row_statistics"), Mapping) else {}
    combined_rows = row_stats.get("combined") if isinstance(row_stats.get("combined"), Mapping) else {}
    born = combined_rows.get("born") if isinstance(combined_rows.get("born"), Mapping) else {}
    dynamic = combined_rows.get("dynamic") if isinstance(combined_rows.get("dynamic"), Mapping) else {}
    hand = payload.get("hand") if isinstance(payload.get("hand"), Mapping) else {}
    scales = payload.get("scales") if isinstance(payload.get("scales"), Mapping) else {}
    applied_scales = scales.get("applied") if isinstance(scales.get("applied"), Mapping) else {}
    starting_note = (
        ""
        if baseline.get("comparable_to_selected_support", False)
        else f"Starting endpoint note: {baseline.get('reason', 'not available')}."
    )
    lines = [
        "# GROND Manual Workflow Summary",
        "",
        f"Selected model: `{payload.get('model', '')}`",
        "",
        "## Headline result",
        "",
        "| Endpoint | R1gt | R1all | wR2all | GooF | NPD/split |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Starting conventional | {_percent(baseline.get('R1_gt'))} | {_percent(baseline.get('R1_all'))} | {_percent(baseline.get('wR2_all'))} | {baseline.get('GooF', 'n/a')} | {baseline.get('npd_count', 'n/a')}/{baseline.get('split_count', 'n/a')} |",
        f"| Selected GROND | {_percent(endpoint.get('R1_gt'))} | {_percent(endpoint.get('R1_all'))} | {_percent(endpoint.get('wR2_all'))} | {endpoint.get('GooF', 'n/a')} | {endpoint.get('npd_count', 'n/a')}/{endpoint.get('split_count', 'n/a')} |",
        "",
        starting_note,
        "" if starting_note else "",
        f"Selected feedback cycle: **{payload.get('selected_cycle', 'n/a')}**. "
        f"{(payload.get('feedback_architecture') or {}).get('summary', '')}",
        "",
        "## JANA/PETS-style matched row statistics",
        "",
        "| Model | Robs | Rall | wRall | MRobs | MRall | MwRall |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Born | {_percent(born.get('Robs'))} | {_percent(born.get('Rall'))} | {_percent(born.get('wRall'))} | {_percent(born.get('MRobs'))} | {_percent(born.get('MRall'))} | {_percent(born.get('MwRall'))} |",
        f"| Bloch | {_percent(dynamic.get('Robs'))} | {_percent(dynamic.get('Rall'))} | {_percent(dynamic.get('wRall'))} | {_percent(dynamic.get('MRobs'))} | {_percent(dynamic.get('MRall'))} | {_percent(dynamic.get('MwRall'))} |",
        "",
        "| Dataset | Bloch Rall | Bloch wRall | Bloch MRall | Bloch MwRall | Born to Bloch Rall gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    per_dataset_stats = row_stats.get("per_dataset") if isinstance(row_stats.get("per_dataset"), Mapping) else {}
    for name, row in per_dataset_stats.items():
        dataset_born = row.get("born") if isinstance(row, Mapping) and isinstance(row.get("born"), Mapping) else {}
        dataset_dynamic = row.get("dynamic") if isinstance(row, Mapping) and isinstance(row.get("dynamic"), Mapping) else {}
        born_rall = _finite_float(dataset_born.get("Rall"))
        dyn_rall = _finite_float(dataset_dynamic.get("Rall"))
        gain = born_rall - dyn_rall if born_rall is not None and dyn_rall is not None else None
        lines.append(
            f"| {name} | {_percent(dataset_dynamic.get('Rall'))} | {_percent(dataset_dynamic.get('wRall'))} | "
            f"{_percent(dataset_dynamic.get('MRall'))} | {_percent(dataset_dynamic.get('MwRall'))} | {_percent(gain)} |"
        )
    lines.extend(
        [
        "",
        "## Pre-feedback hand decision",
        "",
        f"Coordinate choice before feedback: **{hand.get('choice', 'n/a')}**; calculation recommendation: **{hand.get('recommendation', 'n/a')}**. Positive values below favor those pre-feedback coordinates. This simplified workflow does not claim auto's final post-feedback shared-hand selection.",
        "",
        "| Dataset | ZKP all | ZKP free |",
        "|---|---:|---:|",
        ]
    )
    for name, row in (hand.get("datasets") or {}).items():
        all_block = row.get("all") or {}
        free_block = row.get("free") or {}
        lines.append(
            f"| {name} | {_number(all_block.get('z_selected_coordinates'), signed=True)} | "
            f"{_number(free_block.get('z_selected_coordinates'), signed=True)} |"
        )
    combined_hand = hand.get("combined") if isinstance(hand.get("combined"), Mapping) else {}
    lines.append(
        f"| Explicit combined score | {_number((combined_hand.get('all') or {}).get('z_selected_coordinates'), signed=True)} | "
        f"{_number((combined_hand.get('free') or {}).get('z_selected_coordinates'), signed=True)} |"
    )
    lines.extend(
        [
            "",
            "## Per-dataset physics and applied scales",
            "",
            "| Dataset | Transform | Thickness | Gradient | Frame offset | Frame scales | Detector scales |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in (payload.get("datasets") or {}).items():
        nuisance = row.get("nuisance") or {}
        scale_row = (applied_scales.get("datasets") or {}).get(name, {})
        thickness_A = _finite_float(nuisance.get("thickness_A"))
        thickness = "n/a" if thickness_A is None else f"{thickness_A / 10.0:.1f} nm"
        lines.append(
            f"| {name} | `{row.get('hkl_transform', '')}` | {thickness} | "
            f"{nuisance.get('thickness_grad_frac', 'n/a')} | {nuisance.get('frame_offset', 'n/a')} | "
            f"{scale_row.get('frame_parameters', 'n/a')} | {scale_row.get('detector_parameters', 'n/a')} |"
        )
    compact = scales.get("compact_audit") if isinstance(scales.get("compact_audit"), Mapping) else {}
    lines.extend(
        [
            "",
            f"The selected endpoint uses **{applied_scales.get('frame_parameters', 'n/a')} frame** and **{applied_scales.get('detector_parameters', 'n/a')} detector** parameters across {applied_scales.get('rows', 'n/a')} rows. The compact audit found {compact.get('frame_parameters', 'n/a')} frame and {compact.get('detector_parameters', 'n/a')} detector parameters, but **did not apply them to this endpoint**.",
            "",
            "## Feedback-cycle audit",
            "",
            "| Cycle | Lagged native row R1all | Lagged native row wR2all | Endpoint R1all | Endpoint wR2all | GooF | NPD/split | Selected |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for cycle in payload.get("feedback_cycles") or []:
        lines.append(
            f"| {cycle.get('cycle', '')} | {_percent(cycle.get('dyn_R1_all'))} | {_percent(cycle.get('dyn_wR2_all'))} | "
            f"{_percent(cycle.get('endpoint_R1_all'))} | {_percent(cycle.get('endpoint_wR2_all'))} | "
            f"{cycle.get('endpoint_GooF', 'n/a')} | {cycle.get('npd_count', 'n/a')}/{cycle.get('split_count', 'n/a')} | "
            f"{'yes' if cycle.get('selected') else ''} |"
        )
    lines.extend(
        [
            "",
            "The cycle table contains the lagged row objective used for cycle selection. The JANA/PETS-style table above is recomputed once from the selected endpoint coordinates, without another coordinate update, and is the row table to report for the final model.",
        ]
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"RES: `{endpoint.get('res', '')}`  ",
            f"HKL: `{endpoint.get('hkl', '')}`  ",
            f"LST: `{endpoint.get('lst', '')}`  ",
            f"JANA row statistics: `{row_stats.get('json', '')}`  ",
            f"Map manifest: `{payload.get('map_manifest', '')}`",
            "",
        ]
    )
    return "\n".join(lines)


def manual_finish(args) -> int:
    workflow_path, workflow = _load_workflow(args)
    feedback = _require_stage(workflow, "feedback")
    root = Path(workflow["root"])
    out = root / "06_finish"
    out.mkdir(parents=True, exist_ok=True)
    model = Path(str(feedback["outputs"]["selected_model"])).resolve()
    row_runs = {name: Path(path) for name, path in feedback["outputs"]["selected_row_runs"].items()}
    single_dataset_endpoint = len(row_runs) == 1
    maps: dict[str, object]
    started = time.perf_counter()
    if args.skip or args.map_mode == "off":
        maps = {"status": "skipped"}
    else:
        maps = write_native_dynamic_map_package(
            row_table=[run / "native_dyn_calc.csv" for run in row_runs.values()],
            row_table_labels=list(row_runs),
            model_path=model,
            out_dir=out / "native_dynamic_maps",
            options=NativeDynamicMapOptions(
                grid_spacing_A=float(args.native_map_spacing),
                d_min_A=None,
                min_i_over_sigma=0.0,
                max_grid_points=8_000_000,
            ),
        )
    elapsed = time.perf_counter() - started
    endpoint_res = model
    endpoint_hkl = endpoint_res.with_suffix(".hkl")
    endpoint_lst = endpoint_res.with_suffix(".lst")
    metrics = parse_shelx_lst_metrics(endpoint_lst) if endpoint_lst.is_file() else {}
    pathology = parse_shelx_lst_pathologies(endpoint_lst) if endpoint_lst.is_file() else {}
    source_model_text = str(workflow.get("source_model") or "").strip()
    source_model = Path(source_model_text) if source_model_text else None
    source_lst = source_model.with_suffix(".lst") if source_model is not None else None
    source_hkl = source_model.with_suffix(".hkl") if source_model is not None else None
    setup_stage = _stage(workflow, "setup")
    setup_outputs = setup_stage.get("outputs") if isinstance(setup_stage.get("outputs"), Mapping) else {}
    baseline_hkl_text = str(setup_outputs.get("baseline_hkl") or "").strip()
    baseline_hkl = Path(baseline_hkl_text) if baseline_hkl_text else None
    baseline_summary_text = str(workflow.get("conventional_baseline") or "").strip()
    baseline_summary_path = Path(baseline_summary_text) if baseline_summary_text else None
    fresh_baseline = _read_json(baseline_summary_path) if baseline_summary_path is not None and baseline_summary_path.is_file() else {}
    fresh_inputs = fresh_baseline.get("inputs") if isinstance(fresh_baseline.get("inputs"), Mapping) else {}
    fresh_shelxl = fresh_baseline.get("shelxl") if isinstance(fresh_baseline.get("shelxl"), Mapping) else {}
    fresh_res_input = fresh_inputs.get("res") if isinstance(fresh_inputs.get("res"), Mapping) else {}
    fresh_hkl_input = fresh_inputs.get("hkl") if isinstance(fresh_inputs.get("hkl"), Mapping) else {}
    fresh_baseline_current = bool(
        source_model is not None
        and source_model.is_file()
        and baseline_hkl is not None
        and baseline_hkl.is_file()
        and str(fresh_res_input.get("sha256") or "") == _sha256(source_model)
        and str(fresh_hkl_input.get("sha256") or "") == _sha256(baseline_hkl)
        and str(fresh_shelxl.get("status") or "") == "ok"
    )
    sibling_baseline_current = bool(
        source_lst is not None
        and source_lst.is_file()
        and source_hkl is not None
        and source_hkl.is_file()
        and baseline_hkl is not None
        and baseline_hkl.is_file()
        and _sha256(source_hkl) == _sha256(baseline_hkl)
    )
    starting_endpoint_comparable = fresh_baseline_current or sibling_baseline_current
    if fresh_baseline_current:
        source_lst = Path(str(fresh_shelxl.get("lst") or ""))
        starting_metrics = dict(fresh_shelxl)
        starting_pathology = {
            key: fresh_shelxl.get(key)
            for key in (
                "pathology_status",
                "npd_count",
                "split_count",
                "npd_atoms",
                "has_npd",
                "has_split_warnings",
            )
        }
        starting_reason = "fresh one-cycle SHELXL control on the supplied source model and baseline HKL"
    elif sibling_baseline_current:
        starting_metrics = parse_shelx_lst_metrics(source_lst)
        starting_pathology = parse_shelx_lst_pathologies(source_lst)
        starting_reason = "source-model LST and supplied baseline HKL are hash-matched"
    else:
        starting_metrics = {}
        starting_pathology = {}
        starting_reason = "starting SHELXL statistics omitted because no hash-bound source-model/baseline control is available"
    prep_path = Path(str(workflow.get("prep") or ""))
    transforms: dict[str, str] = {}
    if prep_path.is_file():
        prep = load_prep(prep_path)
        transforms = {dataset.name: dataset.hkl_transform for dataset in prep.datasets}
    dataset_payload: dict[str, object] = {}
    for name, run in row_runs.items():
        summary = _read_json(run / "summary.json")
        best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
        dataset_payload[name] = {
            "row_table": str(run / "native_dyn_calc.csv"),
            "summary": str(run / "summary.json"),
            "metrics": best.get("metrics", {}),
            "nuisance": best.get("nuisance", {}),
            "hkl_transform": transforms.get(name, ""),
        }
    try:
        combined_row_stats, per_dataset_row_stats = _manual_row_statistics(row_runs, model)
        row_statistics_status = "complete"
        row_statistics_reason = ""
    except (OSError, ValueError, KeyError) as exc:
        combined_row_stats = {}
        per_dataset_row_stats = {}
        row_statistics_status = "unavailable"
        row_statistics_reason = str(exc)
    jana_json = out / "JANA_ROW_STATISTICS.json"
    jana_per_dataset_json = out / "JANA_ROW_STATISTICS_PER_DATASET.json"
    _write_json(jana_json, combined_row_stats)
    _write_json(jana_per_dataset_json, per_dataset_row_stats)
    hand_summary = _selected_hand_summary(workflow)
    scale_summary = _manual_scale_summary(workflow, row_runs)
    feedback_outputs = feedback.get("outputs") if isinstance(feedback.get("outputs"), Mapping) else {}
    decision_path = Path(str(feedback_outputs.get("decision") or root / "04_feedback" / "multi_endpoint_push_decision.json"))
    decision = _read_json(decision_path) if decision_path.is_file() else {}
    selected_cycle = int(feedback_outputs.get("selected_cycle", 0) or 0)
    feedback_summary_path = Path(str(feedback_outputs.get("feedback_summary") or ""))
    feedback_summary = _read_json(feedback_summary_path) if feedback_summary_path.is_file() else {}
    single_run_metrics = {
        int(str(run.get("label") or "0").split("_")[-1]): run.get("metrics", {})
        for run in feedback_summary.get("runs", [])
        if str(run.get("label") or "").startswith("cycle_")
    }
    feedback_cycles = []
    for row in decision.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        cycle = int(row.get("cycle", 0) or 0)
        run_metrics = single_run_metrics.get(cycle, {})
        feedback_cycles.append(
            {
                **dict(row),
                "dyn_R1_all": row.get("dyn_R1_all", run_metrics.get("native_dyn_R1_all")),
                "dyn_wR2_all": row.get(
                    "dyn_wR2_all",
                    run_metrics.get("native_dyn_wR2_all", row.get("native_wR2_all")),
                ),
                "npd_count": row.get("npd_count", len(row.get("npd_atoms") or [])),
                "selected": cycle == selected_cycle,
            }
        )
    selected_decision_raw = decision.get("selected_cycle") or decision.get("selected_safe_cycle")
    selected_decision = selected_decision_raw if isinstance(selected_decision_raw, Mapping) else {}
    stage_timings = {}
    for stage_name, stage in workflow.get("stages", {}).items():
        command_record = stage.get("command") if isinstance(stage.get("command"), Mapping) else {}
        stage_timings[stage_name] = _command_elapsed(command_record)
    map_outputs = maps.get("outputs") if isinstance(maps.get("outputs"), Mapping) else {}
    payload = {
        "schema_version": 1,
        "kind": "grond_manual_final_summary",
        "workflow": str(workflow_path),
        "model": str(model),
        "selected_cycle": selected_cycle,
        "selection_reason": decision.get("selection_reason", ""),
        "feedback_architecture": {
            "per_dataset_step": "Bloch/Born row recomputation and pinned nuisance-scale fit; no separate coordinate refinement",
            "joint_step": (
                "not applicable for one crystal"
                if single_dataset_endpoint
                else "model-symmetry merge with cross-dataset overlap scaling and inverse-variance weights"
            ),
            "coordinate_step": (
                "one single-crystal SHELXL LS1 update per macrocycle"
                if single_dataset_endpoint
                else "one shared SHELXL LS1 update per macrocycle"
            ),
            "summary": (
                "Each macrocycle recomputes the crystal rows/scales and executes one SHELXL LS1 coordinate update."
                if single_dataset_endpoint
                else "Each macrocycle recomputes rows separately, performs one inverse-variance/model-symmetry merge, and executes one shared SHELXL LS1 coordinate update."
            ),
            "rows_recomputed_after_selection": True,
        },
        "starting_endpoint": {
            "model": str(source_model) if source_model is not None else "",
            "lst": str(source_lst) if source_lst is not None and source_lst.is_file() else "",
            "baseline_hkl": str(baseline_hkl) if baseline_hkl is not None else "",
            "comparable_to_selected_support": starting_endpoint_comparable,
            "reason": (
                starting_reason
            ),
            **starting_metrics,
            **starting_pathology,
        },
        "endpoint": {
            "res": str(endpoint_res),
            "hkl": str(endpoint_hkl) if endpoint_hkl.exists() else "",
            "lst": str(endpoint_lst) if endpoint_lst.exists() else "",
            **metrics,
            **pathology,
            "parameters": selected_decision.get("endpoint_parameters", selected_decision.get("parameters")),
            "restraints": selected_decision.get("endpoint_restraints", selected_decision.get("restraints")),
        },
        "datasets": dataset_payload,
        "hand": hand_summary,
        "row_statistics": {
            "definition": "JANA/PETS-style GROND calculation; distinct from SHELXL endpoint statistics",
            "status": row_statistics_status,
            "reason": row_statistics_reason,
            "instability_u": 0.01,
            "combined": combined_row_stats,
            "per_dataset": per_dataset_row_stats,
            "json": str(jana_json),
            "per_dataset_json": str(jana_per_dataset_json),
        },
        "scales": scale_summary,
        "feedback_cycles": feedback_cycles,
        "timings_seconds": stage_timings,
        "map_manifest": str(map_outputs.get("manifest", "")),
        "maps": maps,
        "elapsed_seconds": elapsed,
    }
    summary_json = out / "manual_final_summary.json"
    summary_md = out / "MANUAL_FINAL_SUMMARY.md"
    _write_json(summary_json, payload)
    summary_md.write_text(_manual_markdown_summary(payload))
    root_summary_json = root / "MANUAL_TOP_LEVEL_SUMMARY.json"
    root_summary_md = root / "MANUAL_TOP_LEVEL_SUMMARY.md"
    shutil.copy2(summary_json, root_summary_json)
    shutil.copy2(summary_md, root_summary_md)
    applied_scales = scale_summary.get("applied") if isinstance(scale_summary.get("applied"), Mapping) else {}
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="finish",
        output_dir=out,
        status="complete",
        outputs={
            "summary_json": str(summary_json),
            "summary_markdown": str(summary_md),
            "top_level_summary_json": str(root_summary_json),
            "top_level_summary_markdown": str(root_summary_md),
            "model": str(model),
            "endpoint_hkl": str(endpoint_hkl) if endpoint_hkl.exists() else "",
            "map_status": maps.get("status", ""),
            "map_manifest": str(map_outputs.get("manifest", "")),
            "headline": {
                "endpoint": (
                    f"R1gt {_percent(metrics.get('R1_gt'))}; R1all {_percent(metrics.get('R1_all'))}; "
                    f"wR2all {_percent(metrics.get('wR2_all'))}; GooF {metrics.get('GooF', 'n/a')}"
                ),
                "hand": (
                    f"pre-feedback {hand_summary.get('choice', 'n/a')} selected; "
                    f"recommendation {hand_summary.get('recommendation', 'n/a')}"
                ),
                "scales": (
                    f"{applied_scales.get('frame_parameters', 'n/a')} frame + "
                    f"{applied_scales.get('detector_parameters', 'n/a')} detector; full pinned contract applied"
                ),
                "summary": str(root_summary_md),
            },
        },
        command={"elapsed_seconds": elapsed, "map_mode": args.map_mode},
        next_command="",
    )
    _print_handoff(handoff)
    return 0


def manual_status(args) -> int:
    workflow_path, workflow = _load_workflow(args)
    rows = []
    for name, stage in workflow.get("stages", {}).items():
        command = stage.get("command") if isinstance(stage.get("command"), Mapping) else {}
        elapsed = _command_elapsed(command)
        if elapsed is None and stage.get("status") == "skipped":
            elapsed = 0.0
        rows.append(
            {
                "stage": name,
                "status": stage.get("status"),
                "elapsed_seconds": elapsed,
                "handoff": stage.get("handoff"),
            }
        )
    payload = {
        "workflow": str(workflow_path),
        "current_stage": workflow.get("current_stage"),
        "chosen_model": workflow.get("chosen_model"),
        "next_command": workflow.get("next_command"),
        "stages": rows,
    }
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2, allow_nan=True))
        return 0
    print("GROND manual workflow")
    print(f"  state: {workflow_path}")
    print(f"  current stage: {workflow.get('current_stage', 'not started')}")
    print("\n  stages:")
    for row in rows:
        elapsed = row.get("elapsed_seconds")
        elapsed_text = "n/a" if elapsed is None else f"{float(elapsed):.1f}s"
        print(f"    {str(row['stage']):10s} {str(row['status']):12s} {elapsed_text:>10s}")
    if workflow.get("chosen_model"):
        print(f"\n  selected model: {workflow['chosen_model']}")
    finish = _stage(workflow, "finish")
    finish_outputs = finish.get("outputs") if isinstance(finish.get("outputs"), Mapping) else {}
    if finish_outputs.get("top_level_summary_markdown"):
        print(f"  final summary: {finish_outputs['top_level_summary_markdown']}")
    if workflow.get("next_command"):
        print(f"\n  next: {workflow['next_command']}")
    elif finish:
        print("\n  workflow complete")
    return 0


def run_manual_stage(args) -> int:
    handlers = {
        "setup": manual_setup,
        "status": manual_status,
        "scout": manual_scout,
        "thickness": manual_thickness,
        "hand": manual_hand,
        "feedback": manual_feedback,
        "scales": manual_scales,
        "finish": manual_finish,
    }
    return handlers[args.stage](args)
