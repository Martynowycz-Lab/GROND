"""Launch residual forensics or a focused endpoint-calibration audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grond.cli_support.execution import _estimated_forensic_steps, configure_progress
from grond.cli_support.tools import choose_python
from grond.commands.scale import cmd_scale_replay
from grond.workflows.forensic_gates import run_forensic_gate


def cmd_forensics(args: argparse.Namespace) -> int:
    """CLI entrypoint for standalone row-space forensic audits."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    python = choose_python(args.python)
    configure_progress(logs, resume=bool(args.resume), total=_estimated_forensic_steps(args))
    args.fast = False
    args.forensic_gate = True
    summary = run_forensic_gate(
        python,
        Path(args.row_table).expanduser().resolve(),
        str(Path(args.hkl).expanduser().resolve()),
        str(Path(args.res).expanduser().resolve()),
        out_dir,
        logs,
        "standalone",
        args,
    )
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2, allow_nan=True))
    return 0


def cmd_calibrate_endpoint(args: argparse.Namespace) -> int:
    """Production-facing alias for endpoint calibration."""

    return cmd_scale_replay(args)
