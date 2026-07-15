#!/usr/bin/env python3
"""Run one conventional SHELXL baseline with explicit provenance.

This is a thin command-line wrapper around the baseline implementation in
``native_row_regularize``.  It intentionally keeps the supplied conventional
EXTI and WGHT policy while normalizing the copied model to one LS/CGLS cycle.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

if __package__:
    from . import native_row_regularize as nrf
else:  # pragma: no cover - direct script execution from the engine directory
    import native_row_regularize as nrf


def file_provenance(path: Path) -> dict[str, object]:
    """Return path, size, and SHA256 provenance for one file."""

    resolved = path.expanduser().resolve()
    record: dict[str, object] = {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "size": None,
        "sha256": "",
    }
    if not resolved.is_file():
        return record
    try:
        record["size"] = int(resolved.stat().st_size)
        record["sha256"] = nrf.sha256(resolved)
    except OSError as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def resolve_shelxl_executable(value: str) -> Path | None:
    """Resolve an explicit SHELXL path or command name without substitution."""

    requested = str(value or "").strip()
    if not requested:
        return None
    has_separator = os.sep in requested or bool(os.altsep and os.altsep in requested)
    if has_separator:
        candidate = Path(requested).expanduser().resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        return None
    found = shutil.which(requested)
    if not found:
        return None
    candidate = Path(found).expanduser().resolve()
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def write_summary(out_dir: Path, payload: dict[str, object]) -> Path:
    """Write the machine-readable baseline result."""

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    return path


def output_provenance(out_dir: Path) -> dict[str, dict[str, object]]:
    """Return provenance for canonical files written by ``copy_baseline``."""

    return {
        label: file_provenance(out_dir / f"temp_a.{suffix}")
        for label, suffix in (
            ("ins", "ins"),
            ("hkl", "hkl"),
            ("res", "res"),
            ("lst", "lst"),
            ("fcf", "fcf"),
        )
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res", required=True, help="Conventional SHELX RES/INS model.")
    parser.add_argument("--hkl", required=True, help="Conventional HKLF reflection file.")
    parser.add_argument("--out-dir", required=True, help="Directory for the isolated baseline replay.")
    parser.add_argument("--shelxl", required=True, help="Explicit SHELXL executable path or command name.")
    args = parser.parse_args(argv)

    res_path = Path(args.res).expanduser().resolve()
    hkl_path = Path(args.hkl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "res": file_provenance(res_path),
        "hkl": file_provenance(hkl_path),
    }
    base: dict[str, object] = {
        "schema_version": 1,
        "definition": "fresh conventional SHELXL baseline; inherited EXTI/WGHT; one LS/CGLS cycle",
        "inputs": inputs,
        "requested_shelxl": str(args.shelxl),
        "outputs": {},
        "shelxl": {},
        "safe_for_feedback": False,
    }

    missing_inputs = [name for name, record in inputs.items() if not record.get("exists")]
    unreadable_inputs = [name for name, record in inputs.items() if record.get("exists") and not record.get("sha256")]
    if missing_inputs or unreadable_inputs:
        status = "missing_input" if missing_inputs else "unreadable_input"
        write_summary(
            out_dir,
            {
                **base,
                "status": status,
                "missing_inputs": missing_inputs,
                "unreadable_inputs": unreadable_inputs,
            },
        )
        return 2

    executable = resolve_shelxl_executable(args.shelxl)
    if executable is None:
        write_summary(
            out_dir,
            {
                **base,
                "status": "missing_shelxl",
                "executable": {
                    "requested": str(args.shelxl),
                    "resolved": "",
                    "exists": False,
                    "executable": False,
                },
            },
        )
        return 2

    executable_record = file_provenance(executable)
    executable_record.update(
        {
            "requested": str(args.shelxl),
            "resolved": str(executable),
            "executable": True,
        }
    )
    previous_shelxl = nrf.SHELXL
    try:
        nrf.SHELXL = str(executable)
        result = nrf.copy_baseline(out_dir, res_path, hkl_path)
    except (OSError, RuntimeError, ValueError) as exc:
        write_summary(
            out_dir,
            {
                **base,
                "status": "process_error",
                "executable": executable_record,
                "outputs": output_provenance(out_dir),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 3
    finally:
        nrf.SHELXL = previous_shelxl

    result = dict(result or {})
    returncode = result.get("returncode")
    health = result.get("res_health") if isinstance(result.get("res_health"), dict) else {}
    status = str(result.get("status") or ("ok" if returncode == 0 else "unknown"))
    explicit_npd = bool(
        result.get("has_npd")
        or health.get("has_npd")
        or int(result.get("npd_count", health.get("npd_count", 0)) or 0) > 0
    )
    if returncode == 0 and explicit_npd:
        status = "adp_pathology"
    safe_for_feedback = bool(
        returncode == 0
        and status == "ok"
        and health.get("ok") is True
        and health.get("safe_for_feedback") is not False
    )
    payload = {
        **base,
        "status": status,
        "executable": executable_record,
        "outputs": output_provenance(out_dir),
        "shelxl": result,
        "safe_for_feedback": safe_for_feedback,
    }
    write_summary(out_dir, payload)

    if status == "missing_shelxl" or returncode is None:
        return 2
    if returncode != 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
