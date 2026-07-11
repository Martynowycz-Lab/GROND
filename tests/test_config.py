from pathlib import Path

import pytest

from grond.io.config import DatasetSpec, PrepConfig, parse_dataset_spec
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "../escape"),
        ("hkl_transform", "h,h,l"),
        ("hkl_transform", "h,k,q"),
        ("target_mode", "unknown"),
        ("min_d_spacing", 0.0),
        ("min_peak", 101.0),
        ("geometry_percentile", float("nan")),
    ],
)
def test_dataset_spec_rejects_unsafe_or_unphysical_values(field, value):
    values = {"name": "dataset", "hkl": "INTEGRATE.HKL"}
    values[field] = value
    with pytest.raises(ValueError):
        DatasetSpec(**values)


def test_prep_config_requires_unique_dataset_names():
    datasets = [
        DatasetSpec(name="same", hkl="one.HKL"),
        DatasetSpec(name="same", hkl="two.HKL"),
    ]
    with pytest.raises(ValueError, match="must be unique"):
        PrepConfig(structure="model.res", datasets=datasets)


def test_prep_config_rejects_case_insensitive_dataset_collisions():
    datasets = [
        DatasetSpec(name="Crystal_A", hkl="one.HKL"),
        DatasetSpec(name="crystal_a", hkl="two.HKL"),
    ]
    with pytest.raises(ValueError, match="must be unique"):
        PrepConfig(structure="model.res", datasets=datasets)


def test_parse_dataset_spec_rejects_unknown_option(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown dataset option.*min_peek"):
        parse_dataset_spec(f"crystal={tmp_path},min_peek=80")


@pytest.mark.parametrize("value", ["maybe", "truthy", "2", ""])
def test_parse_dataset_spec_rejects_invalid_boolean(value: str, tmp_path: Path):
    with pytest.raises(ValueError, match="include_nonpositive must be one of"):
        parse_dataset_spec(f"crystal={tmp_path},include_nonpositive={value}")


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
