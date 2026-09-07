"""Expose configuration, row-template, model-check and endpoint-discovery commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grond.io.structure_contract import check_sfac_contract
from grond.io.tool_config import write_config_template
from grond.refine.endpoints import endpoint_report


def cmd_init_config(args: argparse.Namespace) -> int:
    """CLI entrypoint that writes an external-tool config template."""

    out = write_config_template(
        args.out,
        fmt=args.format,
        overwrite=bool(args.overwrite),
        include_optional_endpoints=bool(args.with_endpoint_adapters),
    )
    print(
        json.dumps(
            {
                "config": str(out),
                "format": args.format,
                "profile": "all_endpoint_adapters" if args.with_endpoint_adapters else "small_molecule",
            },
            indent=2,
        )
    )
    return 0


def cmd_manual(args: argparse.Namespace) -> int:
    """Run one stateful, inspectable manual-workflow stage."""

    from grond.manual_workflow import run_manual_stage

    if bool(getattr(args, "skip", False)) and args.stage == "hand":
        args.choice = "skip"
    return run_manual_stage(args)


def cmd_row_template(args: argparse.Namespace) -> int:
    """CLI entrypoint that writes portable row-contract template files."""

    from grond.io.row_contract import write_template

    rows_path, metadata_path = write_template(args.rows_out, args.metadata_out, overwrite=bool(args.overwrite))
    print(json.dumps({"rows": str(rows_path), "metadata": str(metadata_path)}, indent=2))
    return 0


def cmd_row_validate(args: argparse.Namespace) -> int:
    """CLI entrypoint that validates portable row-contract files."""

    from dataclasses import asdict

    from grond.io.row_contract import (
        load_row_metadata,
        read_row_table,
        validate_row_contract,
        write_xds_compatibility_dir,
    )

    try:
        rows = read_row_table(args.rows)
        metadata = load_row_metadata(args.metadata)
        summary = validate_row_contract(rows, metadata)
        payload: dict[str, object] = {"status": "ok", "summary": asdict(summary)}
        if args.write_xds_dir:
            hkl_path = write_xds_compatibility_dir(rows, metadata, args.write_xds_dir, overwrite=bool(args.overwrite))
            payload["xds_compatibility"] = {"integrate_hkl": str(hkl_path)}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(payload, indent=2))
    return 0


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


def cmd_endpoints(args: argparse.Namespace) -> int:
    """CLI entrypoint that reports configured model-update endpoint adapters."""

    payload = endpoint_report(getattr(args, "tool_config", {}) or {})
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    for row in payload["endpoints"]:
        marker = "available" if row.get("available") else "missing"
        print(f"{row['name']}: {row['status']} ({marker})")
        print(f"  {row['description']}")
        resolved = ((row.get("executable") or {}).get("resolved") or "").strip()
        if resolved:
            print(f"  executable: {resolved}")
        if row.get("configured_command_template"):
            print("  command: configured from grond.config.json")
        if row.get("notes"):
            print(f"  {row['notes']}")
    return 0
