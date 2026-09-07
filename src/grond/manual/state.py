"""Read, bind and atomically record the ordered manual-workflow state."""

from __future__ import annotations

import hashlib
import json
import shlex
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


def _next_command(
    stage: str,
    workflow_path: Path,
    extra: str = "",
    *,
    workflow: Mapping[str, object] | None = None,
) -> str:
    parts = ["grond", "manual", stage, "--workflow", str(workflow_path)]
    defaults = (
        workflow.get("runtime_defaults")
        if isinstance(workflow, Mapping) and isinstance(workflow.get("runtime_defaults"), Mapping)
        else {}
    )
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
        raise SystemExit(
            f"Manual stage {name!r} is not ready. Current next command: {workflow.get('next_command', '')}"
        )
    return stage


def _summary_nuisance(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary = _read_json(path)
    best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
    nuisance = best.get("nuisance") if isinstance(best.get("nuisance"), Mapping) else {}
    config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    if not nuisance:
        raise SystemExit(f"Selected summary has no best-native nuisance contract: {path}")
    return summary, dict(nuisance), dict(config)
