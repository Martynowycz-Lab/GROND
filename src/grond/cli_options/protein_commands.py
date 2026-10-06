"""Register the macromolecular commands: make-free-flags, export-mtz, protein-ls1 and endpoints."""

from __future__ import annotations

import argparse

from grond.macromolecule import FREE_FLAG_LABEL


def register_protein(sub: argparse._SubParsersAction) -> None:
    """Register protein commands."""

    from grond.cli_support.tools import add_tool_config_args
    from grond.commands.protein import cmd_export_mtz, cmd_make_free_flags, cmd_protein_ls1
    from grond.commands.utilities import (
        cmd_endpoints,
    )

    p_free = sub.add_parser("make-free-flags", help="Assign fixed macromolecular FreeR flags at unique-HKL level.")
    p_free.add_argument("--rows", required=True, help="Signed row table used to define the unique-HKL population.")
    p_free.add_argument("--out", required=True, help="Output CSV table with h,k,l,free_flag.")
    p_free.add_argument(
        "--model", default="", help="Optional PDB/mmCIF/RES/CIF model used to read cell and space group."
    )
    p_free.add_argument("--cell", default="", help="Six cell constants: a b c alpha beta gamma.")
    p_free.add_argument("--space-group", default="", help="Space group symbol, for example 'P 21 21 21'.")
    p_free.add_argument("--fraction", type=float, default=0.05, help="Target FreeR fraction on unique HKLs.")
    p_free.add_argument(
        "--min-free-count",
        type=int,
        default=500,
        help="Minimum FreeR reflections when enough unique reflections exist.",
    )
    p_free.add_argument("--seed", type=int, default=20260622)
    p_free.add_argument(
        "--shells",
        type=int,
        default=10,
        help="Balance FreeR flags approximately across resolution shells when d-spacings are known.",
    )
    p_free.add_argument(
        "--merge-anomalous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assign FreeR flags after Friedel merging. Use --no-merge-anomalous to keep anomalous mates separate.",
    )
    p_free.add_argument("--free-value", type=int, default=1)
    p_free.add_argument("--work-value", type=int, default=0)
    p_free.add_argument(
        "--write-rows", default="", help="Optional signed-row CSV output with FreeR_flag propagated to every row."
    )
    p_free.add_argument("--summary", default="", help="Optional JSON summary path. Defaults to OUT.summary.json.")
    p_free.set_defaults(func=cmd_make_free_flags)

    p_mtz = sub.add_parser(
        "export-mtz", help="Merge corrected GROND rows into a macromolecular MTZ with fixed FreeR flags."
    )
    p_mtz.add_argument("--row-table", required=True, help="Corrected or GROND-native row CSV/TSV/JSONL table.")
    p_mtz.add_argument(
        "--free-flags", required=True, help="Unique-HKL FreeR flag table from make-free-flags or an equivalent wrapper."
    )
    p_mtz.add_argument("--out-mtz", required=True)
    p_mtz.add_argument(
        "--out-dir", default="", help="Directory for merged_corrected_rows.csv and export_mtz_summary.json."
    )
    p_mtz.add_argument("--model", default="", help="Optional model used to read cell and space group.")
    p_mtz.add_argument("--cell", default="", help="Six cell constants: a b c alpha beta gamma.")
    p_mtz.add_argument("--space-group", default="", help="Space group symbol, for example 'P 21 21 21'.")
    p_mtz.add_argument("--dataset", default="GROND")
    p_mtz.add_argument("--sigma-policy", choices=["formal", "chisq_overdispersion"], default="formal")
    p_mtz.add_argument("--q-policy", choices=["full", "low_resolution_damped", "off"], default="full")
    p_mtz.add_argument(
        "--q-off-d-A",
        type=float,
        default=5.0,
        help="Low-resolution q damping is off at and below this resolution when using --q-policy low_resolution_damped.",
    )
    p_mtz.add_argument(
        "--q-full-d-A",
        type=float,
        default=3.0,
        help="Low-resolution q damping reaches full correction at and above this resolution.",
    )
    p_mtz.add_argument("--merge-anomalous", action=argparse.BooleanOptionalAction, default=True)
    p_mtz.set_defaults(func=cmd_export_mtz)

    p_protein = sub.add_parser(
        "protein-ls1", help="Run one configured macromolecular endpoint macrocycle against a corrected MTZ."
    )
    add_tool_config_args(p_protein)
    p_protein.add_argument("--model", required=True, help="Input PDB/mmCIF model for the endpoint macrocycle.")
    p_protein.add_argument("--mtz", required=True, help="Corrected MTZ from GROND export-mtz.")
    p_protein.add_argument("--out-dir", required=True)
    p_protein.add_argument("--endpoint", choices=["custom", "phenix_refine", "refmac5", "servalcat"], default="custom")
    p_protein.add_argument(
        "--command",
        dest="endpoint_command",
        default="",
        help="Endpoint command template. Placeholders: {model}, {mtz}, {reflections}, {out_dir}, {cycles}, {free_label}.",
    )
    p_protein.add_argument(
        "--output-model", default="", help="Output model path or template, relative to --out-dir unless absolute."
    )
    p_protein.add_argument("--cycles", type=int, default=1)
    p_protein.add_argument("--free-label", default=FREE_FLAG_LABEL)
    p_protein.add_argument("--dry-run", action="store_true")
    p_protein.set_defaults(func=cmd_protein_ls1)

    p_endpoints = sub.add_parser(
        "endpoints", help="List GROND model-update endpoint adapters and configured tool availability."
    )
    add_tool_config_args(p_endpoints)
    p_endpoints.add_argument("--json", action="store_true")
    p_endpoints.set_defaults(func=cmd_endpoints)
