from grond.progress import ProgressState


def test_progress_resume_requires_matching_command(tmp_path):
    state = ProgressState(root=tmp_path, resume=True, total=1)
    command = ["python", "old_engine.py", "--out-dir", "run"]
    state.mark_completed("stage", command, 1.0, 0, tmp_path / "stage.log")

    assert state.should_skip("stage", command)
    assert not state.should_skip("stage", ["python", "new_engine.py", "--out-dir", "run"])
    assert not ProgressState(root=tmp_path, resume=False, total=1).should_skip("stage", command)

