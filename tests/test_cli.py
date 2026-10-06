"""The command line builds, every command documents itself, and the version is stated once."""

import argparse
import contextlib
import io
import re
import tomllib

import pytest

import grond
from grond.cli_options.parser import build_parser
from tests._support import REPO_ROOT


def _commands():
    parser = build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return parser, sorted(action.choices)


@pytest.mark.parametrize("command", _commands()[1])
def test_every_command_prints_help(command):
    parser, _ = _commands()
    with contextlib.redirect_stdout(io.StringIO()) as out, pytest.raises(SystemExit) as stop:
        parser.parse_args([command, "--help"])
    assert stop.value.code == 0 and "usage:" in out.getvalue()


def test_version_is_consistent():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    citation = re.search(r"^version:\s*\"?([^\"\n]+)", (REPO_ROOT / "CITATION.cff").read_text(), re.M).group(1)
    declared = project.get("version") or grond.__version__
    assert grond.__version__ == declared == citation.strip()
