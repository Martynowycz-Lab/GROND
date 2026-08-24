from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    return path, workflow


def _save_workflow(path: Path, workflow: dict[str, Any]) -> None:
    workflow["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_json(path, workflow)


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def _run_stage_command(command: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    env = dict(os.environ)
    package_src = str(Path(__file__).resolve().parents[1])
    inherited = str(env.get("PYTHONPATH", "") or "")
    env["PYTHONPATH"] = package_src + (f"{os.pathsep}{inherited}" if inherited else "")
    env["PYTHONUNBUFFERED"] = "1"
    out_dir: Path | None = None
    if "--out-dir" in command:
        out_dir = Path(command[command.index("--out-dir") + 1]).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        env["GROND_CACHE_DIR"] = str(out_dir.parent / "cache")
    child_name = command[3] if len(command) > 3 else Path(command[0]).name
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
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
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


def _next_command(stage: str, workflow_path: Path, extra: str = "") -> str:
    suffix = f" {extra.strip()}" if extra.strip() else ""
    return f"grond manual {stage} --workflow {shlex.quote(str(workflow_path))}{suffix}"


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
    )
    prep_path = setup_dir / "prep.json"
    write_prep(prep_path, prep)
    workflow: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": WORKFLOW_KIND,
        "root": str(root),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prep": str(prep_path),
        "source_model": str(structure),
        "chosen_model": "",
        "datasets": [asdict(dataset) for dataset in datasets],
        "stages": {},
    }
    outputs = {
        "prep": str(prep_path),
        "structure": str(structure),
        "structure_sha256": _sha256(structure),
        "baseline_hkl": str(baseline) if baseline else "",
        "datasets": {dataset.name: dataset.hkl for dataset in datasets},
    }
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="setup",
        output_dir=setup_dir,
        status="ready",
        outputs=outputs,
        command={"elapsed_seconds": 0.0, "status": "setup_only"},
        next_command=_next_command("scout", workflow_path),
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
        scouted_prep = out / "scouted_prep.json"
        shutil.copy2(source_prep, scouted_prep)
        command = {"status": "skipped_user_asserted_transforms"}
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
        command = _run_stage_command(command_line)
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
        },
        command=command,
        warnings=warnings,
        next_command=_next_command("thickness", workflow_path),
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
        for path in selected.values():
            _summary_nuisance(Path(path))
        command = {"status": "skipped_with_supplied_summaries"}
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
        command = _run_stage_command(command_line)
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
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="thickness",
        output_dir=out,
        status=status,
        outputs={"selected_summaries": selected, "report": report},
        command=command,
        warnings=warnings,
        next_command=_next_command("hand", workflow_path),
    )
    _print_handoff(handoff)
    return 0


def _hand_evidence(diagnostics: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    all_delta = 0.0
    free_delta = 0.0
    all_n = 0
    free_n = 0
    rows: dict[str, object] = {}
    for name, diagnostic in diagnostics.items():
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
    if all_delta > 0.0 and free_delta >= 0.0:
        recommendation = "input"
    elif all_delta < 0.0 and free_delta <= 0.0:
        recommendation = "inverse"
    else:
        recommendation = "ambiguous"
    return {
        "definition": "positive delta chi2 means the input coordinate hand has lower residual; no known chemistry is used",
        "datasets": rows,
        "combined_all_delta_chi2_inverse_minus_input": all_delta,
        "combined_free_delta_chi2_inverse_minus_input": free_delta,
        "combined_all_rows": all_n,
        "combined_free_rows": free_n,
        "recommendation": recommendation,
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
            outputs={"choice": "input", "chosen_model": prep.structure, "diagnostics": {}},
            command={"status": "skipped_user_choice"},
            warnings=["Absolute-structure checking was explicitly skipped."],
            next_command=_next_command("feedback", workflow_path),
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
                "status": "reused_existing_hand_report_without_recalculation",
                "elapsed_seconds": 0.0,
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
            commands[dataset.name] = _run_stage_command(command_line)
            diagnostics[dataset.name] = _read_json(dataset_out / "absolute_structure_diagnostic.json")
            handoffs[dataset.name] = _read_json(dataset_out / "hand_handoff.json")
    evidence = _hand_evidence(diagnostics)
    evidence_path = out / "combined_hand_evidence.json"
    _write_json(evidence_path, evidence)
    if choice == "report":
        status = "needs_choice"
        chosen_model = ""
        workflow["chosen_model"] = ""
        recommended_choice = str(evidence["recommendation"])
        if recommended_choice not in {"input", "inverse"}:
            recommended_choice = "input"
        next_command = _next_command("hand", workflow_path, f"--choice {recommended_choice}")
    elif choice == "input":
        status = "ready"
        chosen_model = prep.structure
        next_command = _next_command("feedback", workflow_path)
    elif choice == "inverse":
        inverse_models = [Path(str(handoffs[dataset.name]["inverse_model"])) for dataset in prep.datasets]
        hashes = {_sha256(path) for path in inverse_models}
        if len(hashes) != 1:
            raise RuntimeError("Per-dataset hand checks did not materialize one identical inverse-coordinate model")
        status = "ready"
        chosen_model = str(inverse_models[0].resolve())
        next_command = _next_command("feedback", workflow_path)
    else:
        raise SystemExit("--choice must be report, input, inverse, or skip")
    if chosen_model:
        workflow["chosen_model"] = chosen_model
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
        },
        command={"datasets": commands},
        warnings=(
            ["Hand evidence is mixed across all/free rows; inspect both datasets before choosing."]
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
    checkpoint = {
        "schema_version": 1,
        "kind": "grond_continue_checkpoint",
        "status": "ready",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dir": str(output.parent),
        "dataset_count": len(datasets),
        "dataset": datasets[0]["name"],
        "source_run": "",
        "source_summary": "",
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
    _manual_checkpoint(workflow, checkpoint_path)
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
        command_line += ["--n-basis", str(args.n_basis), "--final-n-basis", str(args.n_basis)]
    command = _run_stage_command(command_line)
    push_summary = _read_json(out / "push_endpoint_summary.json")
    decision = _read_json(out / "multi_endpoint_push_decision.json")
    selected = decision.get("selected_cycle") or decision.get("best_native_cycle")
    if not isinstance(selected, Mapping) or not selected.get("model"):
        raise RuntimeError("Feedback produced no clean selected model; inspect ENDPOINT_PUSH_REPORT.md")
    selected_model = str(Path(str(selected["model"])).resolve())
    feedback_summary_path = out / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"
    feedback_summary = _read_json(feedback_summary_path)
    cycle_number = int(selected.get("cycle", 0) or 0)
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
        recompute_command += ["--n-basis", str(args.n_basis), "--final-n-basis", str(args.n_basis)]
    recompute = _run_stage_command(recompute_command)
    selected_feedback_path = (
        selected_rows_out / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"
    )
    selected_feedback = _read_json(selected_feedback_path)
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
            "selected_endpoint_combined_dir": str(cycle_record.get("combined", "")),
            "selected_row_runs": raw_runs,
            "feedback_summary": str(feedback_summary_path),
            "selected_row_recompute_summary": str(selected_feedback_path),
            "report": str(out / "MULTI_ENDPOINT_PUSH_REPORT.md"),
            "push_summary": str(out / "push_endpoint_summary.json"),
            "feedback_status": push_summary.get("status", ""),
        },
        command={"feedback": command, "selected_row_recompute": recompute},
        next_command=_next_command("scales", workflow_path),
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
            outputs={"datasets": {}},
            command={"status": "skipped_user_choice"},
            next_command=_next_command("finish", workflow_path),
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
        commands[name] = _run_stage_command(command_line)
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
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="scales",
        output_dir=out,
        status="ready",
        outputs={"datasets": outputs, "application": "audit_only; full pinned scales remain active"},
        command={"datasets": commands},
        warnings=["Scale compression is audited but not silently applied to the shared endpoint."],
        next_command=_next_command("finish", workflow_path),
    )
    _print_handoff(handoff)
    return 0


def _manual_markdown_summary(payload: Mapping[str, object]) -> str:
    endpoint = payload.get("endpoint") if isinstance(payload.get("endpoint"), Mapping) else {}
    lines = [
        "# GROND Manual Workflow Summary",
        "",
        f"Selected model: `{payload.get('model', '')}`",
        "",
        "## Endpoint",
        "",
        f"- RES: `{endpoint.get('res', '')}`",
        f"- HKL: `{endpoint.get('hkl', '')}`",
        f"- LST: `{endpoint.get('lst', '')}`",
        f"- R1gt: `{endpoint.get('R1_gt', 'n/a')}`",
        f"- R1all: `{endpoint.get('R1_all', 'n/a')}`",
        f"- wR2all: `{endpoint.get('wR2_all', 'n/a')}`",
        f"- GooF: `{endpoint.get('GooF', 'n/a')}`",
        f"- NPD/split: `{endpoint.get('npd_count', 'n/a')}/{endpoint.get('split_count', 'n/a')}`",
        "",
        "## Dataset row tables",
        "",
    ]
    for name, row in (payload.get("datasets") or {}).items():
        lines.append(f"- {name}: `{row.get('row_table', '')}`; summary `{row.get('summary', '')}`")
    lines.extend(["", "## Maps", "", f"Manifest: `{payload.get('map_manifest', '')}`", ""])
    return "\n".join(lines)


def manual_finish(args) -> int:
    workflow_path, workflow = _load_workflow(args)
    feedback = _require_stage(workflow, "feedback")
    root = Path(workflow["root"])
    out = root / "06_finish"
    out.mkdir(parents=True, exist_ok=True)
    model = Path(str(feedback["outputs"]["selected_model"])).resolve()
    row_runs = {name: Path(path) for name, path in feedback["outputs"]["selected_row_runs"].items()}
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
    dataset_payload: dict[str, object] = {}
    for name, run in row_runs.items():
        summary = _read_json(run / "summary.json")
        best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
        dataset_payload[name] = {
            "row_table": str(run / "native_dyn_calc.csv"),
            "summary": str(run / "summary.json"),
            "metrics": best.get("metrics", {}),
            "nuisance": best.get("nuisance", {}),
        }
    map_outputs = maps.get("outputs") if isinstance(maps.get("outputs"), Mapping) else {}
    payload = {
        "schema_version": 1,
        "kind": "grond_manual_final_summary",
        "workflow": str(workflow_path),
        "model": str(model),
        "endpoint": {
            "res": str(endpoint_res),
            "hkl": str(endpoint_hkl) if endpoint_hkl.exists() else "",
            "lst": str(endpoint_lst) if endpoint_lst.exists() else "",
            **metrics,
            **pathology,
        },
        "datasets": dataset_payload,
        "map_manifest": str(map_outputs.get("manifest", "")),
        "maps": maps,
        "elapsed_seconds": elapsed,
    }
    summary_json = out / "manual_final_summary.json"
    summary_md = out / "MANUAL_FINAL_SUMMARY.md"
    _write_json(summary_json, payload)
    summary_md.write_text(_manual_markdown_summary(payload))
    handoff = _record_stage(
        workflow_path,
        workflow,
        name="finish",
        output_dir=out,
        status="complete",
        outputs={
            "summary_json": str(summary_json),
            "summary_markdown": str(summary_md),
            "model": str(model),
            "endpoint_hkl": str(endpoint_hkl) if endpoint_hkl.exists() else "",
            "map_status": maps.get("status", ""),
            "map_manifest": str(map_outputs.get("manifest", "")),
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
    print(
        json.dumps(
            {
                "workflow": str(workflow_path),
                "current_stage": workflow.get("current_stage"),
                "chosen_model": workflow.get("chosen_model"),
                "next_command": workflow.get("next_command"),
                "stages": rows,
            },
            indent=2,
            allow_nan=True,
        )
    )
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
