"""Dispatch a manual stage and report the recorded workflow status."""

from __future__ import annotations

import json
from collections.abc import Mapping

from grond.manual.execution import _command_elapsed
from grond.manual.feedback import manual_feedback, manual_scales
from grond.manual.finish import manual_finish
from grond.manual.hand import manual_hand
from grond.manual.scouting import manual_scout, manual_thickness
from grond.manual.setup import manual_setup
from grond.manual.state import _load_workflow, _stage


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
