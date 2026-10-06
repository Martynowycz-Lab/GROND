"""Expose free-flag, corrected-MTZ and protein-refinement endpoint adapters."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from grond.macromolecule import (
    FREE_FLAG_LABEL,
    export_corrected_mtz,
    infer_cell_and_spacegroup,
    make_free_flags,
    propagate_free_flags,
    read_free_flags,
    read_table as read_macromolecule_table,
    render_protein_endpoint_command,
    run_protein_endpoint,
    write_table as write_macromolecule_table,
)
from grond.refine.endpoints import configured_endpoint_blocks


def cmd_make_free_flags(args: argparse.Namespace) -> int:
    """Assign deterministic macromolecular FreeR flags at unique-HKL level."""

    try:
        cell, sg = infer_cell_and_spacegroup(
            cell_text=args.cell, space_group_text=args.space_group, model=args.model or None
        )
        rows = read_macromolecule_table(args.rows)
        flag_rows, summary = make_free_flags(
            rows,
            sg=sg,
            cell=cell,
            fraction=float(args.fraction),
            min_free_count=int(args.min_free_count),
            seed=int(args.seed),
            shells=int(args.shells),
            merge_anomalous=bool(args.merge_anomalous),
            free_value=int(args.free_value),
            work_value=int(args.work_value),
        )
        out = write_macromolecule_table(args.out, flag_rows)
        summary_path = Path(args.summary).expanduser() if args.summary else out.with_suffix(".summary.json")
        payload: dict[str, object] = {
            "status": "ok",
            "flags": str(out),
            "summary": asdict(summary),
        }
        if args.write_rows:
            flags = read_free_flags(out, sg=sg, merge_anomalous=bool(args.merge_anomalous))
            row_out = write_macromolecule_table(
                args.write_rows,
                propagate_free_flags(
                    rows,
                    flags,
                    sg=sg,
                    merge_anomalous=bool(args.merge_anomalous),
                    flag_column=FREE_FLAG_LABEL,
                ),
            )
            payload["rows_with_flags"] = str(row_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2) + "\n")
        payload["summary_file"] = str(summary_path.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(payload, indent=2))
    return 0


def cmd_export_mtz(args: argparse.Namespace) -> int:
    """Export corrected GROND rows to a macromolecular MTZ with fixed FreeR flags."""

    try:
        cell, sg = infer_cell_and_spacegroup(
            cell_text=args.cell, space_group_text=args.space_group, model=args.model or None
        )
        summary = export_corrected_mtz(
            row_table=args.row_table,
            free_flags=args.free_flags,
            out_mtz=args.out_mtz,
            out_dir=args.out_dir or None,
            cell=cell,
            sg=sg,
            dataset=args.dataset,
            merge_anomalous=bool(args.merge_anomalous),
            sigma_policy=args.sigma_policy,
            q_policy=args.q_policy,
            q_off_d_A=float(args.q_off_d_A),
            q_full_d_A=float(args.q_full_d_A),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps({"status": "ok", "summary": asdict(summary)}, indent=2))
    return 0


def _format_protein_output_model_template(
    template: str,
    *,
    model: str | Path,
    mtz: str | Path,
    out_dir: str | Path,
    cycles: int,
    free_label: str,
) -> str:
    """Fill the protein ``output_model`` template's placeholders; raise ValueError on an unknown placeholder."""

    values = {
        "model": str(model),
        "mtz": str(mtz),
        "reflections": str(mtz),
        "out_dir": str(out_dir),
        "cycles": int(cycles),
        "free_label": str(free_label),
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(f"protein output_model template uses unknown placeholder {exc.args[0]!r}") from exc


def cmd_protein_ls1(args: argparse.Namespace) -> int:
    """Run one configured macromolecular endpoint macrocycle against a corrected MTZ."""

    endpoint = str(args.endpoint).strip().lower().replace("-", "_")
    blocks = configured_endpoint_blocks(getattr(args, "tool_config", {}) or {})
    block = blocks.get(endpoint, {})
    command_template = (args.endpoint_command or str(block.get("command", ""))).strip()
    output_template = (args.output_model or str(block.get("output_model", ""))).strip()
    if not command_template or not output_template:
        payload = {
            "status": "error",
            "error": (
                "protein-ls1 needs --command and --output-model, or matching endpoints."
                f"{endpoint}.command/output_model entries in grond.config.json"
            ),
        }
        print(json.dumps(payload, indent=2))
        return 2
    try:
        out_dir = Path(args.out_dir).expanduser()
        output_model = _format_protein_output_model_template(
            output_template,
            model=args.model,
            mtz=args.mtz,
            out_dir=out_dir,
            cycles=int(args.cycles),
            free_label=args.free_label,
        )
        command = render_protein_endpoint_command(
            command_template,
            model=args.model,
            mtz=args.mtz,
            out_dir=out_dir,
            cycles=int(args.cycles),
            free_label=args.free_label,
        )
        payload = run_protein_endpoint(
            command=command,
            out_dir=out_dir,
            output_model=output_model,
            model=args.model,
            mtz=args.mtz,
            cycles=int(args.cycles),
            dry_run=bool(args.dry_run),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(payload, indent=2))
    return 0
