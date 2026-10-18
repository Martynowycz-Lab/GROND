"""The `grond` entry point."""

from __future__ import annotations

import sys

from grond.cli_options.parser import build_parser, help_level
from grond.cli_support.tools import apply_tool_config


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser(help_level(argv)).parse_args(argv)
    if str(getattr(args, "preset", "standard") or "standard") != "standard":
        # A preset fills only the options the user did not give; record which ones were given.
        from grond.workflows.run_presets import explicit_option_dests

        args.explicit_options = sorted(explicit_option_dests(argv))
    try:
        if args.command != "init-config":
            apply_tool_config(args)
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted. Run the same command with the same --out-dir to resume.", flush=True)
        return 130
