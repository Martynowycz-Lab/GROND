"""Map interoperability must preserve coefficients and bound external failures."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from grond.workflows import finish_maps


@pytest.mark.parametrize("title", ["a" * 90, "é" * 40])
def test_long_title_gets_separate_map_copy_without_changing_coefficients(tmp_path, title):
    source = tmp_path / "endpoint.fcf"
    tail = "_cell_length_a 5.957\r\n_refln_phase_calc\r\n2 0 0 187.84 6.75 13.95 180.0\r\n"
    original = "data_test\r\n_shelx_refln_list_code 6\r\n_shelx_title\t'" + title + "'\r\n" + tail
    source.write_bytes(original.encode())
    mapped, record = finish_maps._shelx2map_compatible_fcf(source)
    assert mapped != source
    assert source.read_bytes() == original.encode()
    assert mapped.read_bytes().endswith(tail.encode())
    assert "_shelx_title GROND_endpoint_map\n" in mapped.read_text()
    assert record["status"] == "shortened_descriptive_title"
    assert record["source_sha256"] != record["map_input_sha256"]


def test_compatible_fcf_is_used_unchanged_and_long_physical_header_is_rejected(tmp_path):
    source = tmp_path / "endpoint.fcf"
    source.write_text("data_test\n_shelx_refln_list_code 6\n_shelx_title simple\n_refln_phase_calc\n")
    mapped, record = finish_maps._shelx2map_compatible_fcf(source)
    assert mapped == source
    assert record["status"] == "unchanged"
    original = "data_test\n_cell_length_a " + "1" * 90 + "\n_refln_phase_calc\n"
    source.write_text(original)
    with pytest.raises(ValueError, match="cannot safely read FCF header line 2"):
        finish_maps._shelx2map_compatible_fcf(source)
    assert source.read_text() == original


def test_converter_timeout_reports_cause_and_continues_other_maps(tmp_path, monkeypatch):
    model = tmp_path / "endpoint.res"
    model.write_text("selected model retained\n")
    model.with_suffix(".fcf").write_text("data_test\n_shelx_refln_list_code 6\n_refln_phase_calc\n")
    executable = tmp_path / "shelx2map"
    executable.touch()
    old_map = tmp_path / "grond_corrected_fofc.map"
    old_map.write_text("stale output must not count as success")
    monkeypatch.setattr(finish_maps, "_publication_endpoint_from_result", lambda _result: {"endpoint_res": str(model)})
    monkeypatch.setattr(finish_maps, "_shelx2map_executable", lambda _args: str(executable))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 120
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 120, output=b"input header stalled")
        Path(command[command.index("-o") + 1]).write_text("new map")
        return subprocess.CompletedProcess(command, 0, "created map\n")

    monkeypatch.setattr(finish_maps.subprocess, "run", run)
    result = finish_maps.run_finish_endpoint_maps({}, SimpleNamespace())
    assert len(calls) == 4
    assert result["status"] == "failed"
    assert result["maps"]["fofc"]["status"] == "timed_out"
    assert result["maps"]["fofc"]["exists"] is False
    assert all(value["returncode"] == 0 for key, value in result["maps"].items() if key != "fofc")
    assert "120-second limit" in Path(result["maps"]["fofc"]["log"]).read_text()
    assert model.read_text() == "selected model retained\n"


@pytest.mark.parametrize("header", ["_cell_length_a " + "1" * 90, " " * 90])
def test_long_physical_header_is_reported_without_aborting_finish(tmp_path, monkeypatch, header):
    model = tmp_path / "endpoint.res"
    model.write_text("selected model retained\n")
    source = model.with_suffix(".fcf")
    source.write_text("data_test\n_shelx_refln_list_code 6\n" + header + "\n_refln_phase_calc\n")
    original = source.read_bytes()
    executable = tmp_path / "shelx2map"
    executable.touch()
    monkeypatch.setattr(finish_maps, "_publication_endpoint_from_result", lambda _result: {"endpoint_res": str(model)})
    monkeypatch.setattr(finish_maps, "_shelx2map_executable", lambda _args: str(executable))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an unsafe FCF header reached the converter")

    monkeypatch.setattr(finish_maps.subprocess, "run", forbidden)
    result = finish_maps.run_finish_maps(tmp_path, {}, SimpleNamespace(map_mode="shelx2map", make_maps=True))
    assert result["status"] == "warning"
    assert result["shelx2map"]["status"] == "fcf_header_incompatible"
    assert result["shelx2map"]["fcf"] == str(source)
    assert "cannot safely read FCF header line 3" in result["warnings"][0]
    assert source.read_bytes() == original


def test_timeout_kills_and_reaps_actual_converter_processes(tmp_path, monkeypatch):
    model = tmp_path / "endpoint.res"
    model.write_text("selected model retained\n")
    model.with_suffix(".fcf").write_text("data_test\n_shelx_refln_list_code 6\n_refln_phase_calc\n")
    executable = tmp_path / "stalled_converter"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys, time\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "out.with_suffix('.pid').write_text(str(os.getpid()))\n"
        "out.write_text('incomplete map')\n"
        "time.sleep(30)\n"
    )
    executable.chmod(0o755)
    monkeypatch.setattr(finish_maps, "_publication_endpoint_from_result", lambda _result: {"endpoint_res": str(model)})
    monkeypatch.setattr(finish_maps, "_shelx2map_executable", lambda _args: str(executable))
    actual_run = subprocess.run
    actual_popen = subprocess.Popen
    spawned = []

    def tracked_popen(*args, **kwargs):
        process = actual_popen(*args, **kwargs)
        spawned.append(process.pid)
        return process

    # Track children in the parent; a short timeout may interrupt Python startup.
    monkeypatch.setattr(subprocess, "Popen", tracked_popen)

    def short_deadline(command, **kwargs):
        assert kwargs["timeout"] == 120
        kwargs["timeout"] = 0.5
        return actual_run(command, **kwargs)

    monkeypatch.setattr(finish_maps.subprocess, "run", short_deadline)
    result = finish_maps.run_finish_endpoint_maps({}, SimpleNamespace())
    assert result["status"] == "failed"
    assert len(spawned) == 4
    for pid in spawned:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert not list(tmp_path.glob('grond_corrected_*.map'))
    assert all(item["status"] == "timed_out" for item in result["maps"].values())


def test_successful_converter_with_invalid_map_reports_error(tmp_path, monkeypatch):
    model = tmp_path / "endpoint.res"
    model.write_text("selected model retained\n")
    model.with_suffix(".fcf").write_text("data_test\n_shelx_refln_list_code 6\n_refln_phase_calc\n")
    executable = tmp_path / "invalid_converter"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_text('not a map')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setattr(finish_maps, "_publication_endpoint_from_result", lambda _result: {"endpoint_res": str(model)})
    monkeypatch.setattr(finish_maps, "_shelx2map_executable", lambda _args: str(executable))
    result = finish_maps.run_finish_endpoint_maps({}, SimpleNamespace())
    assert result["status"] == "failed"
    assert all(item["status"] == "invalid_map" and item["error"] for item in result["maps"].values())
    assert model.read_text() == "selected model retained\n"
