"""Read stage timing and resume history for the final human-readable report."""

from __future__ import annotations

import datetime
import math
import re
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.progress import format_seconds
from grond.reports.values import _quick_float


def _quick_progress_source_roots(out_dir: Path) -> list[tuple[str, int]]:
    """Return GROND source roots recorded in completed stage commands."""

    progress = try_read_json(out_dir / "grond_progress.json")
    completed = progress.get("completed", {}) if isinstance(progress.get("completed"), Mapping) else {}
    counts: dict[str, int] = {}
    marker = "/src/grond/engine/"
    for record in completed.values():
        if not isinstance(record, Mapping):
            continue
        command = record.get("command", [])
        if not isinstance(command, list):
            continue
        text = " ".join(str(part) for part in command)
        if marker not in text:
            continue
        root = text.split(marker, 1)[0].split()[-1]
        counts[root] = counts.get(root, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _quick_progress_timings(out_dir: Path) -> list[dict[str, object]]:
    """Return completed stage timings from the resumable progress state."""

    progress = try_read_json(out_dir / "grond_progress.json")
    completed = progress.get("completed", {}) if isinstance(progress.get("completed"), Mapping) else {}
    if not completed:
        return []
    event_order: list[str] = []
    events = progress.get("events", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, Mapping) or event.get("event") not in {"completed", "failed"}:
                continue
            label = str(event.get("label", ""))
            if label and label not in event_order:
                event_order.append(label)
    labels = [label for label in event_order if label in completed]
    labels.extend(label for label in completed if label not in labels)
    rows: list[dict[str, object]] = []
    for label in labels:
        record = completed.get(label)
        if not isinstance(record, Mapping):
            continue
        try:
            elapsed = float(record.get("elapsed_seconds", 0.0))
        except (TypeError, ValueError):
            elapsed = 0.0
        rows.append(
            {
                "stage": label,
                "returncode": record.get("returncode", "n/a"),
                "elapsed_seconds": elapsed,
                "elapsed": format_seconds(elapsed),
                "log": record.get("log", ""),
            }
        )
    return rows


def _quick_progress_finish_info(out_dir: Path) -> dict[str, object]:
    """Return a compact completion/timing block from progress state."""

    progress = try_read_json(out_dir / "grond_progress.json")
    finished = progress.get("finished") if isinstance(progress.get("finished"), Mapping) else {}
    completed = progress.get("completed", {}) if isinstance(progress.get("completed"), Mapping) else {}
    completed_ok = len([v for v in completed.values() if isinstance(v, Mapping) and v.get("returncode") == 0])
    failed = len([v for v in completed.values() if isinstance(v, Mapping) and v.get("returncode") not in {0, None}])
    total_elapsed = sum(
        _quick_float(v.get("elapsed_seconds"), 0.0)
        for v in completed.values()
        if isinstance(v, Mapping) and v.get("returncode") == 0
    )
    out = dict(finished)
    out.update(
        {
            "completed_stages": completed_ok,
            "failed_stages": failed,
            "cumulative_stage_elapsed_seconds": total_elapsed,
            "stage_elapsed_seconds": total_elapsed,
            "cumulative_stage_elapsed": format_seconds(total_elapsed),
            "stage_elapsed": format_seconds(total_elapsed),
        }
    )
    if finished:
        if out.get("completed_this_invocation") is not None:
            try:
                completed_here = max(0, min(completed_ok, int(out["completed_this_invocation"])))
            except (TypeError, ValueError):
                completed_here = 0
            out["completed_this_invocation"] = completed_here
            out["reused_completed_stages"] = max(0, completed_ok - completed_here)
        if out.get("invocation_elapsed_seconds") is not None:
            out["invocation_elapsed"] = format_seconds(out.get("invocation_elapsed_seconds"))
    return out


PHASE_ORDER = (
    "prep",
    "transform scout",
    "screen",
    "thickness",
    "nested nuisance",
    "nuisance gates",
    "structure feedback",
    "final",
    "hand",
    "selected-hand rerun",
    "combine endpoints",
    "combined resolution and tables",
    "other",
)


def progress_phase(label: str) -> str | None:
    """Map a stage label to its workflow phase; None means "same phase as the preceding stage".

    Unphased stages (resolution limits, polish) run inside the phase that
    launched them, so the caller attributes them to the previous stage's phase.
    """

    text = str(label)
    if text == "fresh_conventional_baseline":
        return "prep"
    if "scout_transform" in text:
        return "transform scout"
    if "final_selected_hand" in text:
        return "selected-hand rerun"
    if re.search(r"_nested_(rows|nuisance|selected)$", text):
        return "nested nuisance"
    if "_feedback_thickness_" in text:  # per-macro-cycle thickness re-selection inside the SHELXL loop
        return "structure feedback"
    if "hand" in text:
        return "hand"
    if "ls1_cycle" in text or "shelxl_feedback" in text or "structure_update" in text:
        return "structure feedback"
    if "thickness" in text:
        return "thickness"
    if re.search(r"adaptive_[a-z0-9_]+_gate$", text):
        return "nuisance gates"
    if text.endswith("_screen"):
        return "screen"
    if text.startswith("combined_") or text.startswith("publication"):
        return "combined resolution and tables"
    if text.startswith("combine_"):
        return "combine endpoints"
    if "final" in text:
        return "final"
    return None


def _progress_time(value: object) -> float | None:
    """Parse an event timestamp (%Y-%m-%dT%H:%M:%S%z) to epoch seconds, None when it does not parse."""

    try:
        return datetime.datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except (TypeError, ValueError):
        return None


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Return the total length (s) of the union of intervals, so overlapping stages count once."""

    total = 0.0
    end = None
    for lo, hi in sorted(intervals):
        if end is None or lo > end:
            total += hi - lo
            end = hi
        elif hi > end:
            total += hi - end
            end = hi
    return total


def _quick_progress_phase_timings(out_dir: Path) -> dict[str, object]:
    """Per-phase wall time from grond_progress.json event timestamps (union of stage intervals).

    Wall seconds count overlapping parallel stages once; stage seconds are the sum of the
    per-stage elapsed times. Event timestamps have one-second resolution.
    """

    progress = try_read_json(out_dir / "grond_progress.json")
    events = progress.get("events", []) if isinstance(progress.get("events"), list) else []
    open_stages: dict[str, list[tuple[float, str]]] = {}
    intervals: dict[str, list[tuple[float, float]]] = {}
    stage_seconds: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    invocations: list[list[float]] = []
    current = "prep"
    for event in events:
        if not isinstance(event, Mapping):
            continue
        kind = event.get("event")
        stamp = _progress_time(event.get("time"))
        if stamp is None:
            continue
        if kind == "invocation_started":
            invocations.append([stamp, stamp])
            continue
        if invocations:
            invocations[-1][1] = max(invocations[-1][1], stamp)
        label = str(event.get("label", ""))
        if kind == "started" and label:
            current = progress_phase(label) or current
            open_stages.setdefault(label, []).append((stamp, current))
        elif kind in {"completed", "failed"} and label and open_stages.get(label):
            start, phase = open_stages[label].pop(0)
            intervals.setdefault(phase, []).append((start, stamp))
            stage_seconds[phase] = stage_seconds.get(phase, 0.0) + _quick_float(event.get("elapsed_seconds"), stamp - start)
            stage_counts[phase] = stage_counts.get(phase, 0) + 1
    if not intervals:
        return {"status": "not_available", "reason": "no timestamped stage events in grond_progress.json", "phases": []}
    rows = []
    for phase in sorted(intervals, key=lambda name: PHASE_ORDER.index(name) if name in PHASE_ORDER else len(PHASE_ORDER)):
        wall = _union_seconds(intervals[phase])
        rows.append(
            {
                "phase": phase,
                "wall_seconds": wall,
                "wall": format_seconds(wall),
                "stage_seconds": stage_seconds.get(phase, 0.0),
                "stages": stage_counts.get(phase, 0),
            }
        )
    all_intervals = [item for values in intervals.values() for item in values]
    in_stages = _union_seconds(all_intervals)
    finished = progress.get("finished") if isinstance(progress.get("finished"), Mapping) else {}
    invocation_wall = sum(hi - lo for lo, hi in invocations)
    total = _quick_float(finished.get("invocation_elapsed_seconds"), float("nan")) if len(invocations) <= 1 else float("nan")
    if math.isnan(total):  # several invocations (resume) or no finish record
        total = invocation_wall if invocations else in_stages
    return {
        "status": "available",
        "source": str(out_dir / "grond_progress.json"),
        "phases": rows,
        "invocations": len(invocations),
        "total_wall_seconds": float(total),
        "in_stage_wall_seconds": float(in_stages),
        "outside_stage_wall_seconds": float(max(0.0, total - in_stages)),
        "cumulative_stage_seconds": float(sum(stage_seconds.values())),
        "phase_rule": "reports.progress.progress_phase; unphased stages inherit the preceding stage's phase",
    }

