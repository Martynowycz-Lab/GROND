"""Run the held-out resolution-limit gate on final row tables and retain its decision."""

from __future__ import annotations

import argparse
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import try_read_json
from grond.workflows.report_commands import resolution_limit_cmd


def run_resolution_limit_gate(
    python: str,
    runs: dict[str, Path],
    res: str,
    out_dir: Path,
    logs: Path,
    label: str,
    args: argparse.Namespace,
    *,
    title: str,
    scope: str,
) -> dict:
    """Run held-out resolution-limit promotion on final row tables."""

    if args.dry_run:
        return {"status": "dry_run", "out_dir": str(out_dir), "runs": {k: str(v) for k, v in runs.items()}}
    missing = [
        str(Path(run) / "native_dyn_calc.csv")
        for run in runs.values()
        if not (Path(run) / "native_dyn_calc.csv").exists()
    ]
    if missing:
        return {"status": "missing_row_table", "missing": missing}
    cmd = resolution_limit_cmd(python, runs, res, out_dir, title=title, scope=scope)
    run_logged(cmd, logs, f"{label}_resolution_limit", dry_run=False)
    summary = try_read_json(out_dir / "summary.json")
    if summary:
        return {"status": "ok", **summary}
    return {
        "status": "missing_artifact_summary",
        "summary": str((out_dir / "summary.json").resolve()),
        "report": str((out_dir / "REPORT.md").resolve()),
        "shells_csv": str((out_dir / "resolution_shells.csv").resolve()),
    }
