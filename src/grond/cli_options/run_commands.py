"""Options of `grond auto`."""

from __future__ import annotations

import argparse


def register_runs(sub: argparse._SubParsersAction) -> None:
    """Register runs commands."""

    from grond.cli_options.common import add_common_run_args
    from grond.commands.run import cmd_auto

    p_auto = sub.add_parser("auto", help="Simple one-command full automatic workflow.")
    add_common_run_args(p_auto, expose_lambda_controls=False)
    p_auto.set_defaults(func=cmd_auto)
