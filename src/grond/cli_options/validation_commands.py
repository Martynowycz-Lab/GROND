"""Options of `grond doctor`."""

from __future__ import annotations

import argparse


def register_runtime(sub: argparse._SubParsersAction) -> None:
    """Register runtime commands."""

    from grond.cli_support.tools import add_tool_config_args
    from grond.commands.doctor import cmd_doctor

    p_doc = sub.add_parser("doctor", help="Check the Python packages, the Bloch-wave kernel and SHELXL.")
    add_tool_config_args(p_doc)
    p_doc.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    p_doc.set_defaults(func=cmd_doctor)
