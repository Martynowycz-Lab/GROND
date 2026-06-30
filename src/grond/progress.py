from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def format_seconds(seconds: float | int | None) -> str:
    """Format a duration as plain wall-clock time for CLI status lines."""

    try:
        value = max(0, int(float(seconds or 0)))
    except (TypeError, ValueError):
        value = 0
    minutes, sec = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    return f"{minutes:d}m{sec:02d}s"


def progress_bar(done: int, total: int, width: int = 24) -> str:
    """Return a compact ASCII progress bar that is safe in plain terminals."""

    if total <= 0:
        return "[" + "." * width + "]"
    done = max(0, min(done, total))
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


@dataclass
class ProgressState:
    """Small resumable state record for one GROND output directory."""

    root: Path
    resume: bool = True
    total: int = 0
    started_at: float = field(default_factory=time.time)
    done_this_session: int = 0

    @property
    def state_path(self) -> Path:
        """Location of the JSON state file used for resume-aware reruns."""

        return self.root / "grond_progress.json"

    def load(self) -> dict[str, Any]:
        """Read the saved progress record, returning an empty record if absent."""

        if not self.state_path.exists():
            return {"completed": {}, "events": []}
        try:
            return json.loads(self.state_path.read_text())
        except Exception:
            return {"completed": {}, "events": [], "warning": "previous progress file could not be parsed"}

    def save(self, payload: dict[str, Any]) -> None:
        """Write the progress record atomically enough for normal CLI use."""

        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
        tmp.replace(self.state_path)

    def should_skip(self, label: str, command: list[str] | None = None) -> bool:
        """Return true when a completed stage can be reused on a resume run."""

        if not self.resume:
            return False
        record = self.load().get("completed", {}).get(label, {})
        if not record or record.get("returncode") != 0:
            return False
        return command is None or record.get("command") == command

    def completed_record(self, label: str) -> dict[str, Any]:
        """Return a completed stage record if one exists."""

        record = self.load().get("completed", {}).get(label, {})
        return record if isinstance(record, dict) else {}

    def mark_started(self, label: str, command: list[str]) -> None:
        """Record the currently running stage so an interruption is explainable."""

        payload = self.load()
        payload["current"] = {
            "label": label,
            "command": command,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        payload.setdefault("events", []).append({"event": "started", "label": label, "time": payload["current"]["started_at"]})
        self.save(payload)

    def mark_completed(self, label: str, command: list[str], elapsed: float, returncode: int, log: Path) -> None:
        """Remember that a stage completed successfully or failed."""

        payload = self.load()
        payload.pop("current", None)
        payload.setdefault("completed", {})[label] = {
            "returncode": int(returncode),
            "elapsed_seconds": float(elapsed),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": command,
            "log": str(log),
        }
        payload.setdefault("events", []).append(
            {
                "event": "completed" if returncode == 0 else "failed",
                "label": label,
                "returncode": int(returncode),
                "elapsed_seconds": float(elapsed),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
        self.save(payload)

    def mark_interrupted(self, label: str | None = None) -> None:
        """Record that the current process was interrupted by the user."""

        payload = self.load()
        payload["interrupted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if label:
            payload["interrupted_stage"] = label
        payload.setdefault("events", []).append({"event": "interrupted", "label": label or "", "time": payload["interrupted_at"]})
        self.save(payload)

    def mark_finished(self, status: str = "ok") -> dict[str, Any]:
        """Record that the full workflow reached its final packaging step."""

        payload = self.load()
        payload.pop("current", None)
        completed = payload.get("completed", {})
        completed_ok = len([v for v in completed.values() if isinstance(v, dict) and v.get("returncode") == 0])
        failed = len([v for v in completed.values() if isinstance(v, dict) and v.get("returncode") not in {0, None}])
        total_elapsed = sum(
            float(v.get("elapsed_seconds", 0.0))
            for v in completed.values()
            if isinstance(v, dict) and v.get("returncode") == 0
        )
        finished = {
            "status": status,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "completed_stages": completed_ok,
            "failed_stages": failed,
            "estimated_stages": int(self.total),
            "stage_elapsed_seconds": float(total_elapsed),
        }
        payload["finished"] = finished
        payload.setdefault("events", []).append({"event": "finished", **finished})
        self.save(payload)
        return finished

    def line_prefix(self, label: str, skipped: bool = False) -> str:
        """Build the common prefix used before every stage status line."""

        completed = len([v for v in self.load().get("completed", {}).values() if v.get("returncode") == 0])
        current_done = completed if skipped else completed + 1
        total = max(self.total, current_done)
        bar = progress_bar(min(current_done, total), total)
        avg = self.average_completed_seconds()
        remaining = max(0, total - current_done)
        eta = format_seconds(avg * remaining) if avg and math.isfinite(avg) else "unknown"
        state = "SKIP" if skipped else "STEP"
        return f"{state} {bar} {current_done}/{total} {label} ETA {eta}"

    def average_completed_seconds(self) -> float:
        """Average successful stage runtime from the saved state."""

        vals = [
            float(item.get("elapsed_seconds", 0.0))
            for item in self.load().get("completed", {}).values()
            if item.get("returncode") == 0 and float(item.get("elapsed_seconds", 0.0)) > 0.0
        ]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))
