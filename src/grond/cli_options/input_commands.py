"""Register the input commands: prep, init-config, row-template, row-validate, sfac-check, audit-hydrogen-omit."""

from __future__ import annotations

import argparse


def register_inputs(sub: argparse._SubParsersAction) -> None:
    """Register inputs commands."""

    from grond.cli_support.tools import add_tool_config_args
    from grond.commands.hydrogen_audit import cmd_audit_hydrogen_omit
    from grond.commands.preparation import cmd_prep
    from grond.commands.utilities import (
        cmd_init_config,
        cmd_row_template,
        cmd_row_validate,
        cmd_sfac_check,
    )

    p_prep = sub.add_parser("prep", help="Write a JSON prep file for one structure and one or more XDS datasets.")
    p_prep.add_argument("--structure", required=True)
    p_prep.add_argument("--baseline-hkl", default="")
    p_prep.add_argument("--dataset", action="append", required=True)
    p_prep.add_argument("--hkl-transform", default="identity")
    p_prep.add_argument("--geometry-percentile", type=float, default=99.0)
    p_prep.add_argument(
        "--parity-setup-id",
        default="",
        help="Stable acquisition/import setup identity frozen before selection.",
    )
    p_prep.add_argument(
        "--parity-setup-manifest-sha256",
        default="",
        help="SHA-256 of the immutable acquisition/import setup manifest.",
    )
    p_prep.add_argument("--notes", default="")
    p_prep.add_argument("--out", required=True)
    add_tool_config_args(p_prep)
    p_prep.set_defaults(func=cmd_prep)

    p_cfg = sub.add_parser("init-config", help="Write a GROND JSON/YAML external-tool config template.")
    p_cfg.add_argument("--out", default="grond.config.json")
    p_cfg.add_argument("--format", choices=["json", "yaml"], default="json")
    p_cfg.add_argument(
        "--with-endpoint-adapters",
        action="store_true",
        help="Also include optional Phenix, REFMAC, Servalcat, and custom endpoint-adapter blocks.",
    )
    p_cfg.add_argument("--overwrite", action="store_true")
    p_cfg.set_defaults(func=cmd_init_config)

    p_row_template = sub.add_parser(
        "row-template", help="Write portable GROND row-contract CSV and metadata templates."
    )
    p_row_template.add_argument("--rows-out", default="grond_rows.csv")
    p_row_template.add_argument("--metadata-out", default="grond_rows_metadata.json")
    p_row_template.add_argument("--overwrite", action="store_true")
    p_row_template.set_defaults(func=cmd_row_template)

    p_row_validate = sub.add_parser(
        "row-validate", help="Validate a portable GROND row table and optional geometry sidecar."
    )
    p_row_validate.add_argument("--rows", required=True, help="Portable row CSV, TSV, JSONL, or NDJSON file.")
    p_row_validate.add_argument("--metadata", required=True, help="Portable row metadata JSON file.")
    p_row_validate.add_argument(
        "--write-xds-dir", default="", help="Optional output directory for an XDS-like compatibility INTEGRATE.HKL."
    )
    p_row_validate.add_argument("--overwrite", action="store_true")
    p_row_validate.set_defaults(func=cmd_row_validate)

    p_sfac = sub.add_parser(
        "sfac-check", help="Check a RES/INS model for explicit electron SFAC rows and endpoint wavelength readiness."
    )
    p_sfac.add_argument("--structure", required=True, help="SHELX .res/.ins model to check.")
    p_sfac.add_argument("--out", default="", help="Optional JSON report path.")
    p_sfac.set_defaults(func=cmd_sfac_check)

    p_h_audit = sub.add_parser(
        "audit-hydrogen-omit",
        help="Quantify prospective never-H Fo-Fc peaks at independent full-H reference sites.",
        description=(
            "Audit one or more difference maps produced from a model that has never contained H atoms. The full-H "
            "reference supplies expected H coordinates only. Both models/maps must share the same cell, origin, proper "
            "HKL setting, and selected hand."
        ),
    )
    p_h_audit.add_argument(
        "--reference-res", required=True, help="Independent full-H RES used only for expected H positions."
    )
    p_h_audit.add_argument(
        "--never-h-res", required=True, help="Prospective refined model; the command fails if it contains any H atoms."
    )
    p_h_audit.add_argument(
        "--map",
        action="append",
        default=[],
        help="name=/absolute/path/to/fofc.map; repeat for native, GROND endpoint, and conventional comparators.",
    )
    p_h_audit.add_argument(
        "--radius-A",
        type=float,
        default=0.40,
        help="Maximum distance from each independent H site to the reported local positive peak.",
    )
    p_h_audit.add_argument(
        "--min-common-nonh-fraction",
        type=float,
        default=0.80,
        help="Minimum fraction of the smaller model's non-H atoms that must share unique labels/elements for the alignment audit (at least three atoms are always required).",
    )
    p_h_audit.add_argument(
        "--max-common-nonh-rms-A",
        type=float,
        default=0.35,
        help="Maximum minimum-image RMS displacement of common labeled non-H atoms before the reference is rejected as unaligned.",
    )
    p_h_audit.add_argument(
        "--max-common-nonh-max-A",
        type=float,
        default=0.75,
        help="Maximum single common-atom displacement before the reference is rejected as unaligned.",
    )
    p_h_audit.add_argument("--out-dir", required=True)
    p_h_audit.set_defaults(func=cmd_audit_hydrogen_omit)
