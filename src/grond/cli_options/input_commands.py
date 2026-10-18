"""Options of `grond prep`, `grond init-config` and `grond sfac-check`."""

from __future__ import annotations

import argparse


def register_inputs(sub: argparse._SubParsersAction) -> None:
    """Register inputs commands."""

    from grond.cli_support.tools import add_tool_config_args
    from grond.commands.preparation import cmd_prep
    from grond.commands.utilities import (
        cmd_init_config,
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
    p_cfg.add_argument("--overwrite", action="store_true")
    p_cfg.set_defaults(func=cmd_init_config)

    p_sfac = sub.add_parser(
        "sfac-check", help="Check a RES/INS model for explicit electron SFAC rows and endpoint wavelength readiness."
    )
    p_sfac.add_argument("--structure", required=True, help="SHELX .res/.ins model to check.")
    p_sfac.add_argument("--out", default="", help="Optional JSON report path.")
    p_sfac.set_defaults(func=cmd_sfac_check)

