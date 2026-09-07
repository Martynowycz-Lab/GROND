"""CLI runtime tools regression tests."""



import os
from pathlib import Path

import grond.cli as cli
import grond.cli_support.tools as owner_cli_support_tools
from grond.cli import (
    apply_tool_config,
    build_parser,
    required_external_tool_errors,
)
from tests.cli._support import valid_shelx_res_text


def test_tool_config_applies_shelxl_paths(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SHELXL", raising=False)
    monkeypatch.delenv("SHELX2MAP", raising=False)
    fake_shelxl = tmp_path / "shelxl"
    fake_shelx2map = tmp_path / "shelx2map"
    fake_shelxl.write_text("#!/bin/sh\n")
    fake_shelx2map.write_text("#!/bin/sh\n")
    cfg = tmp_path / "grond.config.json"
    cfg.write_text(
        '{\n'
        '  "tools": {\n'
        f'    "shelxl": "{fake_shelxl}",\n'
        f'    "shelx2map": "{fake_shelx2map}"\n'
        "  }\n"
        "}\n"
    )
    args = build_parser().parse_args(
        ["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res", "--config", str(cfg)]
    )
    apply_tool_config(args)
    assert args.shelxl == str(fake_shelxl)
    assert os.environ["SHELXL"] == str(fake_shelxl)
    assert os.environ["SHELX2MAP"] == str(fake_shelx2map)


def test_missing_map_tool_only_errors_when_legacy_maps_are_requested(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("SHELXL", raising=False)
    monkeypatch.delenv("SHELX2MAP", raising=False)
    fake_shelxl = tmp_path / "shelxl"
    fake_shelxl.write_text("#!/bin/sh\n")
    cfg = tmp_path / "grond.config.json"
    cfg.write_text('{"tools": {"shelxl": "' + str(fake_shelxl) + '"}}\n')
    model = tmp_path / "model.res"
    model.write_text("TITL test\nEND\n")
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            str(model),
            "--config",
            str(cfg),
            "--make-maps",
        ]
    )
    apply_tool_config(args)
    errors = required_external_tool_errors(args, structure=str(model))
    assert any("shelx2map" in item for item in errors)
    args.make_maps = False
    assert required_external_tool_errors(args, structure=str(model)) == []


def test_explicit_shelxl_argument_satisfies_preflight_without_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty_path"))
    monkeypatch.delenv("SHELXL", raising=False)
    monkeypatch.delenv("SHELX2MAP", raising=False)
    fake_shelxl = tmp_path / "ccp4" / "shelxl"
    fake_shelxl.parent.mkdir()
    fake_shelxl.write_text("#!/bin/sh\nexit 0\n")
    fake_shelxl.chmod(0o755)
    model = tmp_path / "model.res"
    model.write_text("TITL test\nEND\n")
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            str(model),
            "--shelxl",
            str(fake_shelxl),
            "--no-maps",
        ]
    )
    apply_tool_config(args)

    status = cli.external_tool_statuses(args)["shelxl"]
    assert status["available"] is True
    assert status["resolved"] == str(fake_shelxl.resolve())
    assert status["source"] == "argument:--shelxl"
    assert required_external_tool_errors(args, structure=str(model)) == []


def test_explicit_relative_shelxl_is_frozen_to_launch_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    fake_shelxl = tmp_path / "private-ccp4" / "shelxl"
    fake_shelxl.parent.mkdir()
    fake_shelxl.write_text("#!/bin/sh\nexit 0\n")
    fake_shelxl.chmod(0o755)
    model = tmp_path / "model.res"
    model.write_text(valid_shelx_res_text())
    requested = "./private-ccp4/shelxl"
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            str(tmp_path / "out"),
            "--xds-dir",
            str(tmp_path / "xds"),
            "--structure",
            str(model),
            "--shelxl",
            requested,
        ]
    )

    apply_tool_config(args)

    assert args.shelxl_requested == requested
    assert args.shelxl == str(fake_shelxl.resolve())
    status = cli.external_tool_statuses(args)["shelxl"]
    assert status["requested"] == requested
    assert status["resolved"] == str(fake_shelxl.resolve())


def test_default_shelxl_resolution_falls_back_to_shelxl2019(monkeypatch):
    monkeypatch.delenv("SHELXL", raising=False)

    def fake_resolve(_name: str, *, configured: str = "", **_kwargs):
        if configured:
            return {"available": False, "resolved": ""}
        return {"available": True, "resolved": "/opt/shelx/shelxl2019"}

    monkeypatch.setattr(owner_cli_support_tools, "resolve_executable", fake_resolve)
    args = build_parser().parse_args(
        ["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"]
    )

    apply_tool_config(args)

    assert args.shelxl_requested == "shelxl"
    assert args.shelxl == "/opt/shelx/shelxl2019"
