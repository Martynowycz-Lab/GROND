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


def test_progress_records_finished_workflow(tmp_path, monkeypatch):
    state = ProgressState(
        root=tmp_path,
        resume=True,
        total=3,
        started_at=1_700_000_000.0,
        invocation_started_monotonic=100.0,
    )
    monkeypatch.setattr(progress.time, "perf_counter", lambda: 112.5)
    (tmp_path / "a.log").write_text("done\n")
    (tmp_path / "b.log").write_text("done\n")
    state.mark_completed("stage_a", ["python", "a.py"], 2.0, 0, tmp_path / "a.log")
    state.mark_completed("stage_b", ["python", "b.py"], 3.0, 0, tmp_path / "b.log")

    finished = state.mark_finished("ok")

    assert finished["status"] == "ok"
    assert finished["snapshot_kind"] == "completed_invocation"
    assert finished["completed_stages"] == 2
    assert finished["completed_this_invocation"] == 2
    assert finished["reused_completed_stages"] == 0
    assert finished["estimated_stages"] == 3
    assert finished["invocation_elapsed_seconds"] == 12.5
    assert finished["cumulative_stage_elapsed_seconds"] == 5.0
    assert finished["stage_elapsed_seconds"] == 5.0
    assert state.load()["finished"]["completed_stages"] == 2


def test_progress_separates_resumed_stage_runtime_from_current_wall_time(tmp_path, monkeypatch):
    (tmp_path / "a.log").write_text("done\n")
    (tmp_path / "b.log").write_text("done\n")
    prior = ProgressState(root=tmp_path, resume=True, total=2)
    prior.mark_completed("stage_a", ["python", "a.py"], 40.0, 0, tmp_path / "a.log")

    resumed = ProgressState(
        root=tmp_path,
        resume=True,
        total=2,
        invocation_started_monotonic=200.0,
    )
    resumed.mark_completed("stage_b", ["python", "b.py"], 30.0, 0, tmp_path / "b.log")
    monkeypatch.setattr(progress.time, "perf_counter", lambda: 220.0)

    finished = resumed.mark_finished("ok")

    assert finished["invocation_elapsed_seconds"] == 20.0
    assert finished["cumulative_stage_elapsed_seconds"] == 70.0
    assert finished["stage_elapsed_seconds"] == 70.0
    assert finished["completed_this_invocation"] == 1
    assert finished["reused_completed_stages"] == 1


def test_started_resume_clears_stale_finished_status(tmp_path):
    log = tmp_path / "stage.log"
    log.write_text("done\n")
    prior = ProgressState(root=tmp_path, resume=True, total=1)
    prior.mark_completed("stage", ["python", "stage.py"], 5.0, 0, log)
    prior.mark_finished("ok")
    assert prior.load()["finished"]["status"] == "ok"

    resumed = ProgressState(root=tmp_path, resume=True, total=1)
    resumed.mark_started("stage", ["python", "changed_stage.py"])

    payload = resumed.load()
    assert "finished" not in payload
    assert payload["current"]["label"] == "stage"


def test_begin_invocation_clears_finished_before_fully_skipped_resume(tmp_path):
    log = tmp_path / "stage.log"
    log.write_text("done\n")
    prior = ProgressState(root=tmp_path, resume=True, total=1)
    prior.mark_completed("stage", ["python", "stage.py"], 5.0, 0, log)
    prior.mark_finished("ok")

    resumed = ProgressState(root=tmp_path, resume=True, total=1)
    resumed.begin_invocation()
    preview = resumed.finish_info("ok")

    payload = resumed.load()
    assert "finished" not in payload
    assert payload["invocation"]["status"] == "running"
    assert preview["snapshot_kind"] == "pre_packaging_preview"
    assert "finished" not in resumed.load()


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


def test_progress_refreshes_intentionally_postprocessed_output_without_recounting(tmp_path):
    state = ProgressState(root=tmp_path / "progress", resume=True, total=3)
    source = tmp_path / "input.hkl"
    source.write_text("input rows\n")
    output = tmp_path / "run"
    output.mkdir()
    result = output / "result.csv"
    result.write_text("value\n1\n")
    log = tmp_path / "stage.log"
    log.write_text("done\n")
    command = ["python", "stage.py", "--hkl", str(source), "--out-dir", str(output)]
    state.mark_completed("stage", command, 12.5, 0, log)
    before = state.load()
    before_record = before["completed"]["stage"]
    before_events = list(before["events"])
    before_done_this_session = state.done_this_session

    result.write_text("value\n2\n")
    assert not state.should_skip("stage", command)
    assert state.refresh_output_fingerprints("stage", record_timestamp=False)
    assert state.should_skip("stage", command)
    assert "output_fingerprints_refreshed_at" not in state.completed_record("stage")

    result.write_text("value\n3\n")
    assert not state.should_skip("stage", command)
    assert state.refresh_output_fingerprints("stage")
    assert state.should_skip("stage", command)

    after = state.load()
    after_record = after["completed"]["stage"]
    assert len(after["completed"]) == len(before["completed"]) == 1
    assert after["events"] == before_events
    assert state.done_this_session == before_done_this_session == 1
    assert after_record["elapsed_seconds"] == before_record["elapsed_seconds"] == 12.5
    assert after_record["completed_at"] == before_record["completed_at"]
    assert after_record["command"] == before_record["command"] == command
    assert after_record["fingerprint"] == before_record["fingerprint"]
    assert after_record["output_paths"] == before_record["output_paths"]
    assert after_record["output_fingerprints"] != before_record["output_fingerprints"]
    assert after_record["output_fingerprints_refreshed_at"]


def test_progress_output_refresh_fails_closed_for_missing_failed_or_lost_stage(tmp_path):
    state = ProgressState(root=tmp_path / "progress", resume=True, total=1)
    output = tmp_path / "run"
    output.mkdir()
    result = output / "result.csv"
    result.write_text("value\n1\n")
    log = tmp_path / "stage.log"
    log.write_text("failed\n")
    command = ["python", "stage.py", "--out-dir", str(output)]
    state.mark_completed("failed", command, 2.0, 1, log)
    saved = state.state_path.read_bytes()

    assert not state.refresh_output_fingerprints("missing")
    assert state.state_path.read_bytes() == saved
    assert not state.refresh_output_fingerprints("failed")
    assert state.state_path.read_bytes() == saved

    state.mark_completed("completed", command, 3.0, 0, log)
    result.unlink()
    saved = state.state_path.read_bytes()
    assert not state.refresh_output_fingerprints("completed")
    assert state.state_path.read_bytes() == saved


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
