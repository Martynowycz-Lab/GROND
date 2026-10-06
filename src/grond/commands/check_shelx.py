"""Check that one SHELXL endpoint is ready for Olex2/SHELXLE (``grond check-shelx DIR``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grond.refine.shelx_check import check_shelx_endpoint, format_check_report


def cmd_check_shelx(args: argparse.Namespace) -> int:
    """Run SHELXL once on a copy of an endpoint and report weights, GooF, NPD, BASF and file consistency."""

    try:
        report = check_shelx_endpoint(
            Path(args.path),
            shelxl=str(args.shelxl or ""),
            stem=str(args.stem or ""),
            out_dir=Path(args.out_dir) if args.out_dir else None,
            acta=bool(args.acta),
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(f"check-shelx: {exc}") from exc
    print(json.dumps(report, indent=2, allow_nan=True) if args.json else format_check_report(report))
    return 0 if report["passed"] else 1
