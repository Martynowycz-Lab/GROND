"""Dispatch the internal comparison of XDS intensity conventions."""

from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path

from grond.cli_support.values import XDS_TARGET_MODE_LABELS
from grond.progress import format_seconds
from grond.reports.source_controls import _quick_flat_hand_t_style_text
from grond.reports.values import _quick_number, _quick_percent
from grond.workflows.target_ablation import (
    _run_target_ablation_variant,
    _target_ablation_auto_command,
    _target_ablation_summary_row,
    _target_ablation_variants,
    _write_target_ablation_report,
)
from grond.workflows.thickness_scout import _write_csv


def cmd_target_ablation(args: argparse.Namespace) -> int:
    """Run a fixed XDS target-transform ablation through ordinary auto."""

    if not args.xds_dir and not args.hkl:
        raise SystemExit("target-ablation requires --xds-dir or --hkl for the XDS row table.")
    if not args.structure:
        raise SystemExit("target-ablation requires --structure.")
    if not args.baseline_hkl:
        raise SystemExit("target-ablation requires --baseline-hkl.")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = _target_ablation_variants(args.variants)
    rows: list[dict[str, object]] = []
    commands: list[dict[str, object]] = []
    started = time.perf_counter()
    for variant in variants:
        variant_dir = out_dir / variant
        log_path = out_dir / "logs" / f"{variant}.terminal.log"
        cmd = _target_ablation_auto_command(args, variant, variant_dir)
        command_text = " ".join(shlex.quote(part) for part in cmd)
        commands.append(
            {
                "target_mode": variant,
                "out_dir": str(variant_dir.resolve()),
                "terminal_log": str(log_path.resolve()),
                "command": cmd,
                "command_text": command_text,
            }
        )
        print(f"\n=== XDS target ablation: {variant} ({XDS_TARGET_MODE_LABELS.get(variant, variant)}) ===", flush=True)
        returncode, elapsed = _run_target_ablation_variant(cmd, log_path, dry_run=bool(args.dry_run))
        row = _target_ablation_summary_row(variant, variant_dir, log_path, returncode, elapsed)
        rows.append(row)
        if returncode != 0 and args.fail_fast:
            break
    total_elapsed = time.perf_counter() - started
    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": "target-ablation",
        "interpretation": "Reduced XDS row-target ablation. This does not invert the data back to raw detector pixels.",
        "inputs": {
            "xds_dir": str(args.xds_dir),
            "hkl": str(args.hkl),
            "structure": str(args.structure),
            "baseline_hkl": str(args.baseline_hkl),
            "variants": variants,
            "shelxl_feedback_cycles": int(args.shelxl_feedback_cycles),
            "hand_control": bool(args.hand_control),
        },
        "elapsed_seconds": total_elapsed,
        "elapsed": format_seconds(total_elapsed),
        "runs": rows,
        "commands": commands,
    }
    _write_csv(out_dir / "target_ablation_summary.csv", rows)
    (out_dir / "target_ablation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    _write_target_ablation_report(out_dir, rows, commands)
    print("")
    print(f"GROND target ablation finished in {format_seconds(total_elapsed)}.")
    print(f"Report: {out_dir / 'TARGET_ABLATION_SUMMARY.md'}")
    print("")
    print("| target | status | hand paired t-style all/free | final dyn R1/wR2 | publication R1/wR2/GooF |")
    print("|---|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.get('target_mode', '')} | {row.get('status', '')} | "
            f"{_quick_flat_hand_t_style_text(row.get('hand_t_style_all'), row.get('hand_t_style_all_n'), row.get('hand_t_style_all_df'))} / "
            f"{_quick_flat_hand_t_style_text(row.get('hand_t_style_free'), row.get('hand_t_style_free_n'), row.get('hand_t_style_free_df'))} | "
            f"{_quick_percent(row.get('final_dyn_R1all'))} / {_quick_percent(row.get('final_dyn_wR2all'))} | "
            f"{_quick_percent(row.get('publication_R1all'))} / {_quick_percent(row.get('publication_wR2all'))} / {_quick_number(row.get('publication_GooF'), 3)} |"
        )
    return 1 if any(int(row.get("returncode", 0) or 0) != 0 for row in rows) else 0
