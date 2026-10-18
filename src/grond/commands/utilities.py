"""Expose configuration, row-template and model-check commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grond.io.structure_contract import check_sfac_contract
from grond.io.tool_config import write_config_template


def cmd_init_config(args: argparse.Namespace) -> int:
    """CLI entrypoint that writes an external-tool config template."""

    out = write_config_template(
        args.out,
        fmt=args.format,
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "config": str(out),
                "format": args.format,
                "profile": "small_molecule",
            },
            indent=2,
        )
    )
    return 0


def cmd_manual(args: argparse.Namespace) -> int:
    """Run one stateful, inspectable manual-workflow stage."""

    from grond.manual.entrypoint import run_manual_stage

    if bool(getattr(args, "skip", False)) and args.stage == "hand":
        args.choice = "skip"
    return run_manual_stage(args)


def cmd_sfac_check(args: argparse.Namespace) -> int:
    """CLI entrypoint that checks model SFAC and endpoint wavelength readiness."""

    payload = check_sfac_contract(args.structure)
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        payload["out"] = str(out.resolve())
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 2
