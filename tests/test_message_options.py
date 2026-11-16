"""Messages and docstrings name only options and commands that exist.

0.4.3 named five options no parser registers (--split-candidate-set, --transform-candidates,
--validation-console, --conventional-fcf, and --allow-improper-space-group-operations in the model-placement
refusal), and the missing-SHELXL hint named `GROND init-config` (the command is lower case; `GROND` is not found on
a case-sensitive file system).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import grond
from grond.cli_options.parser import build_parser

SRC = Path(grond.__file__).resolve().parent
MODULES = (
    "reports/console.py",
    "workflows/finish_maps.py",
    "workflows/split_record.py",
    "workflows/transform_selection.py",
    "cli_support/tools.py",
    "maps/placement.py",
)
OPTION = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*[a-z0-9])")


def _registered_options() -> set[str]:
    found: set[str] = set()

    def walk(parser: argparse.ArgumentParser) -> None:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    walk(child)
            found.update(action.option_strings)

    walk(build_parser(2))
    return found


def test_messages_name_registered_options():
    registered = _registered_options()
    missing = {
        f"{module}: {option}"
        for module in MODULES
        for option in OPTION.findall((SRC / module).read_text())
        if option not in registered
    }
    assert not missing


def test_removed_options_are_not_named():
    text = "\n".join((SRC / module).read_text() for module in MODULES)
    for option in ("--split-candidate-set", "--transform-candidates", "--validation-console", "--conventional-fcf"):
        assert option not in text


def test_missing_tool_hint_names_the_lower_case_command():
    text = (SRC / "cli_support/tools.py").read_text()
    assert "`GROND init-config" not in text
    assert "`grond init-config --out grond.config.json`" in text
