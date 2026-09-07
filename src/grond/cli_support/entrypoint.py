"""Dispatch the public command and report an interrupted run consistently."""

from __future__ import annotations

from grond.cli_options.parser import build_parser
from grond.cli_support.tools import apply_tool_config


def main(argv: list[str] | None = None) -> int:
    """Package entry point used by both `grond` and `GROND` scripts."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if getattr(args, "command", "") != "init-config":
            apply_tool_config(args)
        return int(args.func(args))
    except KeyboardInterrupt:
        print(
            "\nGROND was interrupted. Re-run the same command with the same --out-dir to resume completed stages.",
            flush=True,
        )
        return 130
