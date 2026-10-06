"""Register the run commands: run and auto."""

from __future__ import annotations

import argparse


def register_runs(sub: argparse._SubParsersAction) -> None:
    """Register runs commands."""

    from grond.cli_options.common import add_common_run_args
    from grond.commands.run import cmd_auto, cmd_run

    p_run = sub.add_parser("run", help="Run fixed-structure Bloch, dynamic regularization, or full auto mode.")
    add_common_run_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_auto = sub.add_parser("auto", help="Simple one-command full automatic workflow.")
    add_common_run_args(p_auto, expose_lambda_controls=False)
    p_auto.set_defaults(func=cmd_auto)
