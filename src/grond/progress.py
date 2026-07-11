from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INPUT_PATH_FLAGS = {
    "--baseline-hkl",
    "--cbf-dir",
    "--config",
    "--datasets-csv",
    "--expt",
    "--hkl",
    "--images",
    "--integrate-hkl",
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
    "--xds-dir",
}
OUTPUT_PATH_FLAGS = {"--out", "--out-dir", "--out-mtz", "--write-rows"}
ASSIGNMENT_PATH_FLAGS = {"--resolution-limit", "--run", "--scale-polish"}
MANIFEST_ONLY_FLAGS = {"--cbf-dir", "--images"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
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
    *,
    hash_contents: bool,
) -> dict[str, Any]:
    try:
        name = str(child.relative_to(base))
    except ValueError:
        name = str(child)
    if hash_contents:
        fingerprint = _hashed_file_fingerprint(child, cache)
        return {
            "name": name,
            "size": fingerprint["size"],
            "sha256": fingerprint["sha256"],
        }
    signature = _stat_signature(child)
    return {
        "name": name,
        "size": signature[2],
        "mtime_ns": signature[3],
        "ctime_ns": signature[4],
    }


def _path_fingerprint(
    path: Path,
    cache: dict[str, tuple[tuple[int, int, int, int, int], dict[str, Any]]] | None = None,
    *,
    metadata_only: bool = False,
    prefer_xds_inputs: bool = True,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    if path.is_file():
        if not metadata_only:
            return _hashed_file_fingerprint(path, cache)
        signature = _stat_signature(path)
        return {
            "path": str(path),
            "status": "file_manifest",
            "size": signature[2],
            "mtime_ns": signature[3],
            "ctime_ns": signature[4],
        }
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
                hash_contents=not metadata_only,
            )
            for child in files
        ],
    }


def _glob_fingerprint(pattern: str) -> dict[str, Any]:
    expanded = str(Path(pattern).expanduser())
    matches = sorted(
        Path(value).expanduser().resolve()
        for value in glob.glob(expanded, recursive=True)
        if Path(value).is_file()
    )
    files = []
    for path in matches:
        signature = _stat_signature(path)
        files.append(
            {
                "name": str(path),
                "size": signature[2],
                "mtime_ns": signature[3],
                "ctime_ns": signature[4],
            }
        )
    return {
        "path": expanded,
        "status": "glob_manifest" if files else "missing_glob",
        "files": files,
    }


def _path_from_flag_value(flag: str, value: str) -> Path:
    if flag in ASSIGNMENT_PATH_FLAGS and "=" in value:
        value = value.split("=", 1)[1]
    return Path(value).expanduser().resolve()


def _flag_values(command: list[str], flags: set[str]) -> list[tuple[str, str]]:
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
    return [_path_from_flag_value(flag, value) for flag, value in _flag_values(command, flags)]


def _input_fingerprints(
    command: list[str],
    cache: dict[str, tuple[tuple[int, int, int, int, int], dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    fingerprints = []
    for flag, value in _flag_values(command, INPUT_PATH_FLAGS):
        if flag == "--images":
            fingerprints.append(_glob_fingerprint(value))
            continue
        fingerprints.append(
            _path_fingerprint(
                _path_from_flag_value(flag, value),
                cache,
                metadata_only=flag in MANIFEST_ONLY_FLAGS,
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

    def mark_started(self, label: str, command: list[str]) -> None:
        """Record the currently running stage so an interruption is explainable."""

        payload = self.load()
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
