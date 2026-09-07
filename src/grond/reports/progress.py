"""Read stage timing and resume history for the final human-readable report."""

from __future__ import annotations

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
    marker = "/src/grond/legacy/engine/"
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
