"""CLI execution regression tests."""

import argparse
import json
from pathlib import Path

import pytest

import grond.cli as cli
import grond.cli_support.execution as owner_cli_support_execution
import grond.commands.run as owner_commands_run
import grond.workflows.endpoint_selection as owner_workflows_endpoint_selection
from grond.io.config import DatasetSpec, PrepConfig
from grond.progress import ProgressState
from tests.cli._support import valid_shelx_res_text


def test_run_logged_writes_concise_failure_artifact(tmp_path: Path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    monkeypatch.setattr(owner_cli_support_execution, "_PROGRESS", None)
    command = [cli.sys.executable, "-c", "print('decisive failure'); raise SystemExit(7)"]

    with pytest.raises(cli.subprocess.CalledProcessError):
        cli.run_logged(command, logs, "broken_stage", cwd=tmp_path)

    failure = json.loads((logs / "failure.json").read_text())
    assert failure["label"] == "broken_stage"
    assert failure["returncode"] == 7
    assert any("decisive failure" in line for line in failure["log_tail"])
    output = capsys.readouterr().out
    assert "FAILED broken_stage" in output
    assert str(logs / "broken_stage.log") in output
    assert "rerun the same GROND command" in output


def test_run_logged_preserves_subprocess_error_when_failure_artifact_write_fails(
    tmp_path: Path, monkeypatch, capsys
):
    logs = tmp_path / "logs"
    monkeypatch.setattr(owner_cli_support_execution, "_PROGRESS", None)
    command = [cli.sys.executable, "-c", "raise SystemExit(9)"]
    original_write_text = cli.Path.write_text

    def write_text(self, *args, **kwargs):
        if self.name.endswith("failure.json"):
            raise OSError("simulated full disk")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(cli.Path, "write_text", write_text)

    with pytest.raises(cli.subprocess.CalledProcessError) as caught:
        cli.run_logged(command, logs, "broken_artifact_stage", cwd=tmp_path)

    assert caught.value.returncode == 9
    output = capsys.readouterr().out
    assert "could not write failure artifact" in output
    assert "FAILED broken_artifact_stage" in output


def test_dry_run_finalization_does_not_require_materialized_row_statistics(tmp_path: Path, monkeypatch):
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="dataset", hkl=str(tmp_path / "INTEGRATE.HKL"))],
    )
    args = argparse.Namespace(out_dir=str(tmp_path / "dry_run"), dry_run=True)

    def unexpected_quick_stats(*_args, **_kwargs):
        raise AssertionError("dry-run finalization must not inspect materialized row tables")

    monkeypatch.setattr(owner_commands_run, "write_quick_stats", unexpected_quick_stats)

    quick = cli._finalize_quick_stats(prep, {}, args)
    text = cli._format_terminal_finish_summary(
        {"status": "dry_run", "out_dir": str(tmp_path / "dry_run"), "quick_stats": quick}
    )

    assert quick["status"] == "dry_run"
    assert "No endpoint or row-level statistics were materialized" in quick["note"]
    assert "GROND dry run completed successfully" in text
    assert "no scientific stages ran" in text


def test_run_completion_report_failure_does_not_persist_finished(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "run"
    state = ProgressState(root=out_dir, resume=True, total=0)
    state.begin_invocation()
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="dataset", hkl=str(tmp_path / "INTEGRATE.HKL"))],
    )
    args = argparse.Namespace(out_dir=str(out_dir), dry_run=False)

    def fail_quick_stats(*_args, **_kwargs):
        raise RuntimeError("quick stats failed")

    monkeypatch.setattr(owner_commands_run, "_finalize_quick_stats", fail_quick_stats)

    with pytest.raises(RuntimeError, match="quick stats failed"):
        cli._write_run_completion_artifacts(prep, {}, args)

    assert "finished" not in state.load()
    assert not (out_dir / "grond_summary.json").exists()


def test_run_completion_summary_write_failure_does_not_persist_finished(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "grond_summary.json").mkdir()
    state = ProgressState(root=out_dir, resume=True, total=0)
    state.begin_invocation()
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="dataset", hkl=str(tmp_path / "INTEGRATE.HKL"))],
    )
    args = argparse.Namespace(out_dir=str(out_dir), dry_run=False)
    monkeypatch.setattr(owner_commands_run, "_finalize_quick_stats", lambda *_args, **_kwargs: {"status": "ok"})

    with pytest.raises(IsADirectoryError):
        cli._write_run_completion_artifacts(prep, {}, args)

    assert "finished" not in state.load()
    assert state.load()["invocation"]["status"] == "running"


def test_publication_report_refreshes_fingerprint_after_parent_enrichment(
    tmp_path: Path,
    monkeypatch,
):
    out_dir = tmp_path / "workflow"
    logs = out_dir / "logs"
    state = ProgressState(root=out_dir, resume=True, total=1)
    structure = tmp_path / "model.res"
    structure.write_text(valid_shelx_res_text())
    primary_run = tmp_path / "primary_run"
    primary_run.mkdir()
    prep = PrepConfig(
        structure=str(structure),
        datasets=[DatasetSpec(name="dataset", hkl=str(tmp_path / "INTEGRATE.HKL"))],
    )
    result = {
        "model": str(structure),
        "primary_run": str(primary_run),
        "resolution_limits": {"dataset": {"status": "accepted", "min_d_spacing": 1.0}},
    }
    args = argparse.Namespace(out_dir=str(out_dir), dry_run=False)
    captured_command: list[str] = []

    def fake_run_logged(command, log_dir, label, *, dry_run=False, cwd=cli.LEGACY_ROOT):
        del dry_run, cwd
        captured_command[:] = command
        table_dir = Path(command[command.index("--out-dir") + 1])
        table_dir.mkdir(parents=True)
        (table_dir / "publication_artifacts.json").write_text(
            json.dumps({"summary": str(table_dir / "row_mr_metrics.json")}) + "\n"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{label}.log"
        log.write_text("completed\n")
        state.mark_completed(label, command, 1.0, 0, log)

    monkeypatch.setattr(owner_cli_support_execution, "_PROGRESS", state)
    monkeypatch.setattr(owner_workflows_endpoint_selection, "run_logged", fake_run_logged)

    report = cli.run_publication_report(cli.sys.executable, prep, result, args, logs)

    artifact = json.loads((out_dir / "tables" / "publication_artifacts.json").read_text())
    record = state.completed_record("publication_row_mr_tables")
    assert report["status"] == "ok"
    assert artifact["resolution_limits"] == result["resolution_limits"]
    assert record["output_fingerprints_refreshed_at"]
    assert state.should_skip("publication_row_mr_tables", captured_command)


def test_configured_run_invocation_clears_finished_before_fully_skipped_resume(
    tmp_path: Path,
    monkeypatch,
):
    out_dir = tmp_path / "run"
    prior = ProgressState(root=out_dir, resume=True, total=0)
    prior.mark_finished("ok")
    assert prior.load()["finished"]["status"] == "ok"
    monkeypatch.setattr(owner_cli_support_execution, "_PROGRESS", None)

    cli.configure_progress(
        out_dir / "logs",
        resume=True,
        total=0,
        begin_invocation=True,
    )

    assert owner_cli_support_execution._PROGRESS is not None
    payload = owner_cli_support_execution._PROGRESS.load()
    assert "finished" not in payload
    assert payload["invocation"]["status"] == "running"


def test_generic_progress_configuration_preserves_finished_without_run_lifecycle(
    tmp_path: Path,
    monkeypatch,
):
    out_dir = tmp_path / "standalone"
    prior = ProgressState(root=out_dir, resume=True, total=0)
    prior.mark_finished("ok")
    monkeypatch.setattr(owner_cli_support_execution, "_PROGRESS", None)

    cli.configure_progress(out_dir / "logs", resume=True, total=1)

    assert owner_cli_support_execution._PROGRESS is not None
    assert owner_cli_support_execution._PROGRESS.load()["finished"]["status"] == "ok"
