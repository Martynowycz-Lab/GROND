from grond.progress import ProgressState


def test_progress_resume_requires_matching_command(tmp_path):
    state = ProgressState(root=tmp_path, resume=True, total=1)
    command = ["python", "old_engine.py", "--out-dir", "run"]
    state.mark_completed("stage", command, 1.0, 0, tmp_path / "stage.log")

    assert state.should_skip("stage", command)
    assert not state.should_skip("stage", ["python", "new_engine.py", "--out-dir", "run"])
    assert not ProgressState(root=tmp_path, resume=False, total=1).should_skip("stage", command)


def test_progress_records_finished_workflow(tmp_path):
    state = ProgressState(root=tmp_path, resume=True, total=3)
    state.mark_completed("stage_a", ["python", "a.py"], 2.0, 0, tmp_path / "a.log")
    state.mark_completed("stage_b", ["python", "b.py"], 3.0, 0, tmp_path / "b.log")

    finished = state.mark_finished("ok")

    assert finished["status"] == "ok"
    assert finished["completed_stages"] == 2
    assert finished["estimated_stages"] == 3
    assert finished["stage_elapsed_seconds"] == 5.0
    assert state.load()["finished"]["completed_stages"] == 2
