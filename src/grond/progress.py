"""Resume state and progress display for multi-stage GROND runs.

``grond_progress.json`` in the output directory records each completed stage with its
command, a hash of the GROND source, fingerprints of its input paths (taken from known
command-line flags) and of its outputs. A rerun skips a stage only when all of these still
match.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INPUT_PATH_FLAGS = {
    "--baseline-hkl",
    "--config",
    "--datasets-csv",
    "--expt",
    "--hkl",
    "--input",
    "--model",
    "--prep",
    "--refl",
    "--resolution-limit",
    "--res",
    "--row-table",
    "--rows",
    "--run",
    "--scale-polish",
    "--source-summary",
    "--start-res",
    "--structure",
    "--summary",
    "--shelxl",
    "--xds-dir",
}
OUTPUT_PATH_FLAGS = {"--out", "--out-dir", "--out-mtz", "--output", "--write-rows"}
ASSIGNMENT_PATH_FLAGS = {"--resolution-limit", "--run", "--scale-polish"}


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    """Return (device, inode, size, mtime_ns, ctime_ns) for a file."""

    stat = path.stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _hashed_file_fingerprint(
    path: Path,
    cache: dict[str, tuple[tuple[int, int, int, int, int], dict[str, Any]]] | None,
) -> dict[str, Any]:
    """Return the path, size and SHA-256 of a file, reusing the cache while its stat signature is unchanged."""

    path = path.expanduser().resolve()
    signature = _stat_signature(path)
    key = str(path)
    if cache is not None and key in cache and cache[key][0] == signature:
        return dict(cache[key][1])
    fingerprint = {
        "path": key,
        "status": "file",
        "size": signature[2],
        "sha256": _sha256_file(path),
    }
    if cache is not None:
        cache[key] = (signature, dict(fingerprint))
    return fingerprint


def _manifest_entry(
    child: Path,
    base: Path,
    cache: dict[str, tuple[tuple[int, int, int, int, int], dict[str, Any]]] | None,
) -> dict[str, Any]:
    """Describe one file of a directory by its relative name, size and SHA-256."""

    try:
        name = str(child.relative_to(base))
    except ValueError:
        name = str(child)
    fingerprint = _hashed_file_fingerprint(child, cache)
    return {
        "name": name,
        "size": fingerprint["size"],
        "sha256": fingerprint["sha256"],
    }


def _path_fingerprint(
    path: Path,
    cache: dict[str, tuple[tuple[int, int, int, int, int], dict[str, Any]]] | None = None,
    *,
    prefer_xds_inputs: bool = True,
) -> dict[str, Any]:
    """Fingerprint an input path: a missing marker, a file digest or a directory manifest.

    For an XDS directory only INTEGRATE.HKL, XDS_ASCII.HKL, XPARM.XDS, GXPARM.XDS and a
    DIALS scan-point matrix file named in the INTEGRATE.HKL header are used; otherwise the
    directory is walked, skipping caches, logs and symlinks.
    """

    path = path.expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    if path.is_file():
        return _hashed_file_fingerprint(path, cache)
    preferred = ("INTEGRATE.HKL", "XDS_ASCII.HKL", "XPARM.XDS", "GXPARM.XDS")
    files = [path / name for name in preferred if (path / name).is_file()] if prefer_xds_inputs else []
    integrate = path / "INTEGRATE.HKL"
    if prefer_xds_inputs and integrate.is_file():
        for line in integrate.read_text(errors="replace").splitlines():
            if not line.startswith("!"):
                break
            if line.startswith("!DIALS_A_AT_SCAN_POINTS_FILE="):
                value = line.split("=", 1)[1].strip().split()[0]
                scan_matrix = Path(value)
                if not scan_matrix.is_absolute():
                    scan_matrix = path / scan_matrix
                if scan_matrix.is_file() and scan_matrix not in files:
                    files.append(scan_matrix)
    if not files:
        files = []
        for root, directories, names in os.walk(path):
            root_path = Path(root)
            directories[:] = sorted(
                name
                for name in directories
                if "cache" not in name.lower()
                and name not in {".git", "__pycache__", "logs"}
                and not (root_path / name).is_symlink()
            )
            for name in sorted(names):
                child = root_path / name
                if name == "grond_progress.json" or child.is_symlink():
                    continue
                files.append(child)
    return {
        "path": str(path),
        "status": "directory",
        "files": [
            _manifest_entry(
                child,
                path,
                cache,
            )
            for child in files
        ],
    }


def _path_from_flag_value(flag: str, value: str) -> Path:
    """Return the resolved path given to a flag, dropping a ``name=`` prefix where the flag takes one."""

    if flag in ASSIGNMENT_PATH_FLAGS and "=" in value:
        value = value.split("=", 1)[1]
    return Path(value).expanduser().resolve()


def _flag_values(command: list[str], flags: set[str]) -> list[tuple[str, str]]:
    """Return (flag, value) pairs for the given flags, in either ``--flag value`` or ``--flag=value`` form."""

    values: list[tuple[str, str]] = []
    for index, token in enumerate(command):
        if token in flags and index + 1 < len(command) and command[index + 1]:
            values.append((token, command[index + 1]))
            continue
        for flag in flags:
            prefix = flag + "="
            if token.startswith(prefix) and token[len(prefix) :]:
                values.append((flag, token[len(prefix) :]))
                break
    return values


def _flag_paths(command: list[str], flags: set[str]) -> list[Path]:
    """Return the resolved paths passed to the given flags in a command."""

    return [_path_from_flag_value(flag, value) for flag, value in _flag_values(command, flags)]


def _input_fingerprints(
    command: list[str],
    cache: dict[str, tuple[tuple[int, int, int, int, int], dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Fingerprint every input path named in a stage command."""

    fingerprints = []
    for flag, value in _flag_values(command, INPUT_PATH_FLAGS):
        fingerprints.append(
            _path_fingerprint(
                _path_from_flag_value(flag, value),
                cache,
                prefer_xds_inputs=True,
            )
        )
    return fingerprints


def _matches_recorded_output(
    snapshot: dict[str, Any],
    cache: dict[str, tuple[tuple[int, int, int, int, int], dict[str, Any]]] | None,
) -> bool:
    """Check recorded output contents while allowing later files to be added."""

    path = Path(str(snapshot.get("path", "")))
    status = snapshot.get("status")
    if status == "file":
        if not path.is_file():
            return False
        return (
            path.stat().st_size == int(snapshot.get("size", -1))
            and _hashed_file_fingerprint(path, cache).get("sha256") == snapshot.get("sha256")
        )
    if status != "directory" or not path.is_dir():
        return False
    files = snapshot.get("files")
    if not isinstance(files, list) or not files:
        return False
    for record in files:
        child = path / str(record.get("name", ""))
        if (
            not child.is_file()
            or child.stat().st_size != int(record.get("size", -1))
            or _hashed_file_fingerprint(child, cache).get("sha256") != record.get("sha256")
        ):
            return False
    return True


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
    invocation_started_monotonic: float = field(default_factory=time.perf_counter, repr=False)
    done_this_session: int = 0
    _source_hash: str = field(default="", init=False, repr=False)
    _fingerprint_cache: dict[
        str,
        tuple[tuple[int, int, int, int, int], dict[str, Any]],
    ] = field(default_factory=dict, init=False, repr=False)

    @property
    def state_path(self) -> Path:
        """Location of the JSON state file used for resume-aware reruns."""

        return self.root / "grond_progress.json"

    def load(self) -> dict[str, Any]:
        """Read the saved progress record, returning an empty record if absent."""

        if not self.state_path.exists():
            return {"completed": {}, "events": []}
        try:
            payload = json.loads(self.state_path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("progress record must be a JSON object")
            if "completed" in payload and not isinstance(payload["completed"], dict):
                raise ValueError("progress record completed field must be a JSON object")
            if "events" in payload and not isinstance(payload["events"], list):
                raise ValueError("progress record events field must be a JSON array")
            return payload
        except Exception as exc:
            stamp = f"{time.strftime('%Y%m%dT%H%M%S')}-{time.time_ns()}"
            quarantine = self.root / f"grond_progress.corrupt-{stamp}.json"
            try:
                self.state_path.replace(quarantine)
                detail = f"; preserved as {quarantine}"
            except OSError:
                quarantine = None
                detail = "; the unreadable file could not be quarantined"
            message = f"Previous GROND progress file could not be parsed or validated ({type(exc).__name__}){detail}. Starting a fresh progress record."
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return {
                "completed": {},
                "events": [],
                "warning": message,
                "quarantined_path": str(quarantine) if quarantine is not None else "",
            }

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
        if command is not None:
            if record.get("command") != command:
                return False
            if record.get("fingerprint") != self.command_fingerprint(command):
                return False
        log = Path(str(record.get("log", "")))
        if not log.is_file() or log.stat().st_size == 0:
            return False
        output_fingerprints = record.get("output_fingerprints", [])
        if record.get("output_paths") and not output_fingerprints:
            return False
        return all(
            _matches_recorded_output(snapshot, self._fingerprint_cache)
            for snapshot in output_fingerprints
        )

    def source_hash(self) -> str:
        """Fingerprint the installed GROND Python source used by resumed stages."""

        if self._source_hash:
            return self._source_hash
        digest = hashlib.sha256()
        source_root = Path(__file__).resolve().parent
        for path in sorted(source_root.rglob("*.py")):
            digest.update(str(path.relative_to(source_root)).encode())
            digest.update(path.read_bytes())
        self._source_hash = digest.hexdigest()
        return self._source_hash

    def command_fingerprint(self, command: list[str]) -> dict[str, Any]:
        """Fingerprint code and declared input paths for one resumable stage."""

        return {
            "source_sha256": self.source_hash(),
            "inputs": _input_fingerprints(command, self._fingerprint_cache),
        }

    def completed_record(self, label: str) -> dict[str, Any]:
        """Return a completed stage record if one exists."""

        record = self.load().get("completed", {}).get(label, {})
        return record if isinstance(record, dict) else {}

    def refresh_output_fingerprints(self, label: str, *, record_timestamp: bool = True) -> bool:
        """Refresh saved outputs after intentional postprocessing of a completed stage.

        The stage command and its input fingerprint remain immutable.  This is
        deliberately narrower than marking the stage complete again: it does
        not add an event, change elapsed time, or increment session counts.
        """

        payload = self.load()
        completed = payload.get("completed", {})
        if not isinstance(completed, dict):
            return False
        record = completed.get(label)
        if not isinstance(record, dict) or record.get("returncode") != 0:
            return False
        output_paths = record.get("output_paths")
        if (
            not isinstance(output_paths, list)
            or not output_paths
            or not all(isinstance(value, str) and value for value in output_paths)
        ):
            return False
        paths = [Path(value) for value in output_paths]
        try:
            log = Path(str(record.get("log", "")))
            if not log.is_file() or log.stat().st_size == 0:
                return False
            if not all(path.is_file() or path.is_dir() for path in paths):
                return False
            refreshed = [
                _path_fingerprint(
                    path,
                    self._fingerprint_cache,
                    prefer_xds_inputs=False,
                )
                for path in paths
            ]
            if not all(_matches_recorded_output(snapshot, self._fingerprint_cache) for snapshot in refreshed):
                return False
        except (OSError, ValueError):
            return False

        refreshed_record = dict(record)
        refreshed_record["output_fingerprints"] = refreshed
        if record_timestamp:
            refreshed_record["output_fingerprints_refreshed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        refreshed_completed = dict(completed)
        refreshed_completed[label] = refreshed_record
        payload["completed"] = refreshed_completed
        self.save(payload)
        return True

    def begin_invocation(self) -> None:
        """Clear stale completion state before this invocation does any work."""

        payload = self.load()
        payload.pop("finished", None)
        payload.pop("current", None)
        started_at = time.strftime(
            "%Y-%m-%dT%H:%M:%S%z",
            time.localtime(self.started_at),
        )
        payload["invocation"] = {
            "status": "running",
            "started_at": started_at,
        }
        payload.setdefault("events", []).append({"event": "invocation_started", "time": started_at})
        self.save(payload)

    def mark_started(self, label: str, command: list[str]) -> None:
        """Record the currently running stage so an interruption is explainable."""

        payload = self.load()
        # A new stage means any earlier workflow-finished record is historical.
        # Remove it so an interrupted or failed resume cannot look successful.
        payload.pop("finished", None)
        payload["current"] = {
            "label": label,
            "command": command,
            "fingerprint": self.command_fingerprint(command),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        payload.setdefault("events", []).append({"event": "started", "label": label, "time": payload["current"]["started_at"]})
        self.save(payload)

    def mark_completed(self, label: str, command: list[str], elapsed: float, returncode: int, log: Path) -> None:
        """Remember that a stage completed successfully or failed."""

        payload = self.load()
        current = payload.get("current", {})
        fingerprint = (
            current.get("fingerprint")
            if current.get("command") == command and isinstance(current.get("fingerprint"), dict)
            else self.command_fingerprint(command)
        )
        output_paths = _flag_paths(command, OUTPUT_PATH_FLAGS)
        payload.pop("current", None)
        payload.setdefault("completed", {})[label] = {
            "returncode": int(returncode),
            "elapsed_seconds": float(elapsed),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": command,
            "log": str(log),
            "fingerprint": fingerprint,
            "output_paths": [str(path) for path in output_paths],
            "output_fingerprints": [
                _path_fingerprint(
                    path,
                    self._fingerprint_cache,
                    prefer_xds_inputs=False,
                )
                for path in output_paths
            ],
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
        if returncode == 0:
            self.done_this_session += 1
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
        payload.pop("invocation", None)
        finished = self._finish_info_from_payload(
            payload,
            status,
            snapshot_kind="completed_invocation",
        )
        payload["finished"] = finished
        payload.setdefault("events", []).append({"event": "finished", **finished})
        self.save(payload)
        return finished

    def finish_info(
        self,
        status: str = "ok",
        *,
        snapshot_kind: str = "pre_packaging_preview",
    ) -> dict[str, Any]:
        """Return completion timing and counts without persisting finished state."""

        return self._finish_info_from_payload(
            self.load(),
            status,
            snapshot_kind=snapshot_kind,
        )

    def _finish_info_from_payload(
        self,
        payload: dict[str, Any],
        status: str,
        *,
        snapshot_kind: str,
    ) -> dict[str, Any]:
        """Build one immutable timing snapshot from an already loaded record."""

        completed = payload.get("completed", {})
        completed_ok = len([v for v in completed.values() if isinstance(v, dict) and v.get("returncode") == 0])
        failed = len([v for v in completed.values() if isinstance(v, dict) and v.get("returncode") not in {0, None}])
        cumulative_stage_elapsed = sum(
            float(v.get("elapsed_seconds", 0.0))
            for v in completed.values()
            if isinstance(v, dict) and v.get("returncode") == 0
        )
        invocation_elapsed = max(0.0, time.perf_counter() - self.invocation_started_monotonic)
        finished = {
            "status": status,
            "snapshot_kind": snapshot_kind,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "invocation_started_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z",
                time.localtime(self.started_at),
            ),
            "invocation_elapsed_seconds": float(invocation_elapsed),
            "completed_stages": completed_ok,
            "completed_this_invocation": int(self.done_this_session),
            "reused_completed_stages": max(0, completed_ok - int(self.done_this_session)),
            "failed_stages": failed,
            "estimated_stages": int(self.total),
            "cumulative_stage_elapsed_seconds": float(cumulative_stage_elapsed),
            # Backward-compatible alias. This is a sum of saved stage runtimes,
            # not workflow wall-clock time when stages overlap or are resumed.
            "stage_elapsed_seconds": float(cumulative_stage_elapsed),
        }
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
