"""Options of `grond doctor`."""

from __future__ import annotations

import argparse


def register_runtime(sub: argparse._SubParsersAction) -> None:
    """Register runtime commands."""

    from grond.cli_support.tools import add_tool_config_args
    from grond.commands.doctor import cmd_doctor

    p_doc = sub.add_parser(
        "doctor", help="Report Python packages, accelerator visibility, and the active compute core."
    )
    p_doc.add_argument("--python", default="")
    add_tool_config_args(p_doc)
    p_doc.add_argument(
        "--probe-bloch", action="store_true", help="Also test complex eig/eigh support needed by the Bloch wave kernel."
    )
    p_doc.add_argument(
        "--workflow",
        choices=["core", "small-molecule", "native-maps", "endpoint-maps"],
        default="core",
        help="Readiness scope: core/native-maps need no external program; small-molecule adds SHELXL; endpoint-maps also adds shelx2map.",
    )
    p_doc.add_argument(
        "--all-tools",
        action="store_true",
        help="Also report optional endpoint tools and experimental accelerator backends.",
    )
    p_doc.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when required packages, installed features, or requested Bloch operations are not ready.",
    )
    p_doc.set_defaults(func=cmd_doctor)

