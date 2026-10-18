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

    p_prep = sub.add_parser("prep", help="Write the input file for one model and one or more crystals.")
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

    p_cfg = sub.add_parser("init-config", help="Write a config file with the paths of SHELXL and shelx2map.")
    p_cfg.add_argument("--out", default="grond.config.json")
    p_cfg.add_argument("--format", choices=["json", "yaml"], default="json")
    p_cfg.add_argument("--overwrite", action="store_true")
    p_cfg.set_defaults(func=cmd_init_config)

    p_sfac = sub.add_parser(
        "sfac-check", help="Check that a SHELX model has numeric electron scattering factors."
    )
    p_sfac.add_argument("--structure", required=True, help="SHELX .res/.ins model to check.")
    p_sfac.add_argument("--out", default="", help="Optional JSON report path.")
    p_sfac.set_defaults(func=cmd_sfac_check)

