import pytest

import grond.progress as progress
from grond.progress import ProgressState


def test_progress_quarantines_malformed_state_before_restarting(tmp_path):
    bad = tmp_path / "grond_progress.json"
    bad.write_text("{not valid json\n")
    state = ProgressState(root=tmp_path, resume=True, total=1)

    with pytest.warns(RuntimeWarning, match="could not be parsed"):
        payload = state.load()

    quarantined = list(tmp_path.glob("grond_progress.corrupt-*.json"))
    assert payload["completed"] == {}
    assert len(quarantined) == 1
    assert quarantined[0].read_text() == "{not valid json\n"
    assert not bad.exists()


@pytest.mark.parametrize(
    "text",
    [
        "[]\n",
        "null\n",
        '{"completed": [], "events": []}\n',
        '{"completed": {}, "events": {}}\n',
    ],
)
def test_progress_quarantines_structurally_invalid_json(tmp_path, text):
    bad = tmp_path / "grond_progress.json"
    bad.write_text(text)
    state = ProgressState(root=tmp_path, resume=True, total=1)

    with pytest.warns(RuntimeWarning, match="could not be parsed or validated"):
        payload = state.load()

    quarantined = list(tmp_path.glob("grond_progress.corrupt-*.json"))
    assert payload["completed"] == {}
    assert payload["events"] == []
    assert len(quarantined) == 1
    assert quarantined[0].read_text() == text
    assert not bad.exists()


def test_progress_quarantine_names_do_not_collide_within_one_second(tmp_path, monkeypatch):
    bad = tmp_path / "grond_progress.json"
    state = ProgressState(root=tmp_path, resume=True, total=1)
    monkeypatch.setattr(progress.time, "strftime", lambda *args, **kwargs: "20260713T120000")
    times = iter((1001, 1002))
    monkeypatch.setattr(progress.time, "time_ns", lambda: next(times))

    with pytest.warns(RuntimeWarning):
        bad.write_text("first invalid record")
        state.load()
    with pytest.warns(RuntimeWarning):
        bad.write_text("second invalid record")
        state.load()

    quarantined = sorted(tmp_path.glob("grond_progress.corrupt-*.json"))
    assert [path.read_text() for path in quarantined] == ["first invalid record", "second invalid record"]


def test_progress_resume_requires_matching_command(tmp_path):
    state = ProgressState(root=tmp_path, resume=True, total=1)
    source = tmp_path / "input.hkl"
    source.write_text("input rows\n")
    output = tmp_path / "run"
    output.mkdir()
    (output / "summary.json").write_text("{}\n")
    log = tmp_path / "stage.log"
    log.write_text("completed\n")
    command = ["python", "old_engine.py", "--hkl", str(source), "--out-dir", str(output)]
    state.mark_completed("stage", command, 1.0, 0, log)

    assert state.should_skip("stage", command)
    assert not state.should_skip("stage", ["python", "new_engine.py", "--out-dir", "run"])
    assert not ProgressState(root=tmp_path, resume=False, total=1).should_skip("stage", command)
    source.write_text("changed rows\n")
    assert not state.should_skip("stage", command)


def test_progress_resume_hashes_explicit_shelxl_executable(tmp_path):
    state = ProgressState(root=tmp_path / "progress", resume=True, total=1)
    executable = tmp_path / "private-ccp4" / "shelxl"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n")
    output = tmp_path / "run"
    output.mkdir()
    (output / "summary.json").write_text("{}\n")
    log = tmp_path / "stage.log"
    log.write_text("completed\n")
    command = [
        "python",
        "native_row_regularize.py",
        "--shelxl",
        str(executable),
        "--out-dir",
        str(output),
    ]
    state.mark_completed("native", command, 1.0, 0, log)

    assert state.should_skip("native", command)
    executable.write_text("#!/bin/sh\nexit 1\n")
    assert not state.should_skip("native", command)


def test_progress_records_finished_workflow(tmp_path):
    state = ProgressState(root=tmp_path, resume=True, total=3)
    (tmp_path / "a.log").write_text("done\n")
    (tmp_path / "b.log").write_text("done\n")
    state.mark_completed("stage_a", ["python", "a.py"], 2.0, 0, tmp_path / "a.log")
    state.mark_completed("stage_b", ["python", "b.py"], 3.0, 0, tmp_path / "b.log")

    finished = state.mark_finished("ok")

    assert finished["status"] == "ok"
    assert finished["completed_stages"] == 2
    assert finished["estimated_stages"] == 3
    assert finished["stage_elapsed_seconds"] == 5.0
    assert state.load()["finished"]["completed_stages"] == 2


def test_progress_invalidates_repeated_named_run_directory_inputs(tmp_path):
    state = ProgressState(root=tmp_path / "progress", resume=True, total=1)
    source_run = tmp_path / "source_run"
    source_run.mkdir()
    nested = source_run / "endpoint" / "native_dyn_calc.csv"
    nested.parent.mkdir()
    nested.write_text("h,k,l\n1,0,0\n")
    output = tmp_path / "combined"
    output.mkdir()
    (output / "summary.json").write_text("{}\n")
    log = tmp_path / "combine.log"
    log.write_text("done\n")
    command = [
        "python",
        "combine.py",
        "--run",
        f"crystal_a={source_run}",
        "--out-dir",
        str(output),
    ]
    state.mark_completed("combine", command, 1.0, 0, log)

    assert state.should_skip("combine", command)
    nested.write_text("h,k,l\n2,0,0\n")
    assert not state.should_skip("combine", command)


def test_progress_invalidates_missing_or_changed_recorded_output(tmp_path):
    state = ProgressState(root=tmp_path / "progress", resume=True, total=1)
    output = tmp_path / "run"
    output.mkdir()
    summary = output / "summary.json"
    summary.write_text('{"status": "ok"}\n')
    result = output / "result.csv"
    result.write_text("value\n1\n")
    log = tmp_path / "stage.log"
    log.write_text("done\n")
    command = ["python", "stage.py", "--out-dir", str(output)]
    state.mark_completed("stage", command, 1.0, 0, log)

    assert state.should_skip("stage", command)
    result.write_text("value\n2\n")
    assert not state.should_skip("stage", command)


def test_progress_tracks_integrate_hkl_and_raw_image_glob(tmp_path):
    state = ProgressState(root=tmp_path / "progress", resume=True, total=1)
    hkl = tmp_path / "INTEGRATE.HKL"
    hkl.write_text("rows\n")
    images = tmp_path / "images"
    images.mkdir()
    frame = images / "image_00001.cbf"
    frame.write_bytes(b"frame one")
    log = tmp_path / "raw.log"
    log.write_text("done\n")
    command = [
        "python",
        "raw.py",
        "--integrate-hkl",
        str(hkl),
        "--images",
        str(images / "*.cbf"),
    ]
    state.mark_completed("raw", command, 1.0, 0, log)

    assert state.should_skip("raw", command)
    frame.write_bytes(b"frame two")
    assert not state.should_skip("raw", command)


def test_progress_reuses_file_digest_until_stat_signature_changes(tmp_path, monkeypatch):
    state = ProgressState(root=tmp_path / "progress", resume=True, total=1)
    source = tmp_path / "input.hkl"
    source.write_text("first\n")
    command = ["python", "stage.py", "--hkl", str(source)]
    original = progress._sha256_file
    calls = 0

    def counted(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(progress, "_sha256_file", counted)

    first = state.command_fingerprint(command)
    second = state.command_fingerprint(command)
    assert first == second
    assert calls == 1

    source.write_text("second\n")
    third = state.command_fingerprint(command)
    assert third != second
    assert calls == 2


def test_progress_output_snapshot_includes_dials_sidecars(tmp_path):
    state = ProgressState(root=tmp_path / "progress", resume=True, total=1)
    output = tmp_path / "dials_bridge"
    output.mkdir()
    (output / "INTEGRATE.HKL").write_text("!header\n")
    summary = output / "dials_bridge_summary.json"
    summary.write_text('{"status": "ok"}\n')
    (output / "DIALS_A_AT_SCAN_POINTS.txt").write_text("1 0 0 0 1 0 0 0 1\n")
    log = tmp_path / "bridge.log"
    log.write_text("done\n")
    command = ["python", "bridge.py", "--out-dir", str(output)]
    state.mark_completed("bridge", command, 1.0, 0, log)

    assert state.should_skip("bridge", command)
    summary.unlink()
    assert not state.should_skip("bridge", command)


def test_progress_xds_input_includes_referenced_dials_scan_matrix(tmp_path):
    state = ProgressState(root=tmp_path / "progress", resume=True, total=1)
    source = tmp_path / "xds"
    source.mkdir()
    (source / "INTEGRATE.HKL").write_text(
        "!DIALS_A_AT_SCAN_POINTS_FILE= DIALS_A_AT_SCAN_POINTS.txt\n"
        " 1 0 0 1 1 0 0 0 1 1 1\n"
    )
    scan_matrix = source / "DIALS_A_AT_SCAN_POINTS.txt"
    scan_matrix.write_text("1 0 0 0 1 0 0 0 1\n")
    log = tmp_path / "stage.log"
    log.write_text("done\n")
    command = ["python", "stage.py", "--xds-dir", str(source)]
    state.mark_completed("stage", command, 1.0, 0, log)

    assert state.should_skip("stage", command)
    scan_matrix.write_text("2 0 0 0 1 0 0 0 1\n")
    assert not state.should_skip("stage", command)
