"""Build the `grond` command line.

`grond <command> -h` lists the options most runs need, `-h -v` adds the settings the paper's runs changed and
`-h -vv` lists every option. Unlisted options still work.
"""

from __future__ import annotations

import argparse

from .endpoint_commands import register_finish, register_push
from .input_commands import register_inputs
from .integration_commands import register_integration
from .manual_commands import register_manual
from .run_commands import register_runs
from .scout_commands import register_scouts, register_thickness
from .validation_commands import register_runtime

INTERNAL_COMMANDS = {"scout", "hand-check", "thickness-scout"}

# Help level of each option, per command. Options not listed here are level 2.
HELP_LEVELS = {
    "auto": {
        0: ["--prep", "--out-dir", "--config", "--workers", "--preset", "--free-set", "--make-maps", "--min-d-spacing",
            "--shelxl"],
        1: ["--python", "--resume", "--endpoint-min-d-spacing", "--min-peak", "--h-omit", "--max-support-sg",
            "--n-points", "--nested-n-points", "--nested-n-basis", "--final-basis-max", "--final-basis-scan-limit",
            "--mosaicity-mode", "--nested-response-tables", "--shelxl-feedback-cycles", "--chunksize",
            "--prediction-cache-dir", "--row-plan-cache", "--shared-grid", "--shared-grid-gpus", "--basis-centring-coset"],
    },
    "refine": {
        0: ["--checkpoint", "--start-res", "--out-dir", "--cycles", "--workers", "--config", "--shelxl"],
        1: ["--python", "--resume"],
    },
    "finish": {
        0: ["--run-dir", "--config", "--map-mode", "--native-map-spacing"],
        1: ["--checkpoint", "--model", "--shelxl", "--make-maps", "--publication-feedback-cycles", "--json"],
    },
}


def help_level(argv: list[str]) -> int:
    """Count -v, -vv, -vvv and --verbose on the command line."""

    level = 0
    for arg in argv:
        if arg == "--verbose":
            level += 1
        elif len(arg) > 1 and arg[0] == "-" and set(arg[1:]) == {"v"}:
            level += len(arg) - 1
    return level


def _apply_help_levels(name: str, parser: argparse.ArgumentParser, level: int) -> None:
    """Hide the options of one command above the requested help level."""

    parser.add_argument("-v", "--verbose", action="count", default=0, help="with -h: -v lists more options, -vv all")
    levels = HELP_LEVELS.get(name)
    if not levels:
        return
    shown = {flag: lvl for lvl, flags in levels.items() for flag in flags}
    hidden = 0
    for action in parser._actions:
        flags = [flag for flag in action.option_strings if flag.startswith("--")]
        if not flags or flags[0] in ("--help", "--verbose"):
            continue
        if min(shown.get(flag, 2) for flag in flags) > level:
            action.help = argparse.SUPPRESS
            hidden += 1
    if hidden:
        parser.epilog = f"{hidden} more options not listed; use -h {'-v or -vv' if level == 0 else '-vv'} to see them."


def build_parser(level: int = 0) -> argparse.ArgumentParser:
    """Create the parser; `level` is the help level from help_level()."""

    parser = argparse.ArgumentParser(
        prog="grond",
        description="GROND (Global Regularization Of Non-kinematical Diffraction): Bloch-wave corrections for "
        "electron-diffraction intensities between DIALS/XDS and SHELXL.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    register_runs(sub)
    register_manual(sub)
    register_push(sub)
    register_finish(sub)
    register_inputs(sub)
    register_integration(sub)
    register_runtime(sub)
    # Stage commands that `grond manual` runs; they are not listed in `grond -h`.
    register_scouts(sub)
    register_thickness(sub)
    sub._choices_actions = [a for a in sub._choices_actions if a.dest not in INTERNAL_COMMANDS]
    seen = set()
    for name, subparser in sub.choices.items():
        if id(subparser) not in seen:  # an alias maps to the same parser
            seen.add(id(subparser))
            _apply_help_levels(name, subparser, level)
    return parser
