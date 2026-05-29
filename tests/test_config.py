from pathlib import Path

from grond.io.config import parse_dataset_spec
from grond.io.tool_config import (
    configured_tools,
    load_tool_config,
    resolve_executable,
    tool_env_updates,
    write_config_template,
)


def test_parse_dataset_spec_directory(tmp_path: Path):
    xds = tmp_path / "xds"
    xds.mkdir()
    (xds / "INTEGRATE.HKL").write_text("! test\n")
    ds = parse_dataset_spec(f"crystal_a={xds},dmin=0.8,hkl_transform=l,-k,h,geometry_percentile=99")
    assert ds.name == "crystal_a"
    assert ds.hkl.endswith("INTEGRATE.HKL")
    assert ds.hkl_transform == "l,-k,h"
    assert ds.min_d_spacing == 0.8
    assert ds.geometry_percentile == 99.0


def test_tool_config_json_template_round_trip(tmp_path: Path):
    cfg = write_config_template(tmp_path / "grond.config.json")
    data = load_tool_config(cfg)
    tools = configured_tools(data)
    assert "shelxl" not in tools
    assert tool_env_updates(data) == {}


def test_tool_config_accepts_executables_alias(tmp_path: Path):
    cfg = tmp_path / "tools.json"
    cfg.write_text('{"executables": {"SHELXL": "~/bin/shelxl", "shelx2map": "$HOME/bin/shelx2map"}}\n')
    tools = configured_tools(load_tool_config(cfg))
    assert tools["shelxl"].endswith("/bin/shelxl")
    assert tools["shelx2map"].endswith("/bin/shelx2map")


def test_explicit_bad_tool_path_does_not_fallback_to_path(tmp_path: Path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "shelxl").write_text("#!/bin/sh\n")
    monkeypatch.setenv("PATH", str(fake_bin))
    status = resolve_executable("shelxl", configured=str(tmp_path / "missing_shelxl"), aliases=("shelxl",))
    assert status["available"] is False
    assert status["source"] == "config"
    assert "path does not exist" in status["error"]
