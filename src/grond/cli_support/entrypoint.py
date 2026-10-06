"""Dispatch the public command and report an interrupted run consistently."""

from __future__ import annotations

import sys

from grond.cli_options.parser import build_parser
from grond.cli_support.tools import apply_tool_config


def main(argv: list[str] | None = None) -> int:
    """Package entry point used by both `grond` and `GROND` scripts."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if str(getattr(args, "preset", "standard") or "standard") != "standard":
        # A preset fills only the options the user did not give; record which ones were given.
        from grond.workflows.run_presets import explicit_option_dests

        args.explicit_options = sorted(explicit_option_dests(sys.argv[1:] if argv is None else argv))
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
