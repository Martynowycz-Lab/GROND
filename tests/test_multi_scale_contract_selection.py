from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from grond import cli


def _write_run(path: Path, *, scale_count: int, weak_free_bloch: bool = False) -> None:
    path.mkdir(parents=True)
    rows = [
        {
            "h": "1",
            "k": "0",
            "l": "0",
            "obs": "100",
            "sigma": "5",
            "split": "work",
            "dyn_scaled": "95",
            "born_scaled": "80",
        },
        {
            "h": "0",
            "k": "1",
            "l": "0",
            "obs": "50",
            "sigma": "4",
            "split": "free",
            "dyn_scaled": "48",
            "born_scaled": "40",
        },
    ]
    with (path / "native_dyn_calc.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        split: {
            "dyn": {"R1_all": 0.20, "wR2_all": 0.20},
            "born": {
                "R1_all": 0.201 if weak_free_bloch and split == "free" else 0.30,
                "wR2_all": 0.201 if weak_free_bloch and split == "free" else 0.30,
            },
        }
        for split in ("all", "work", "free")
    }
    (path / "summary.json").write_text(
        json.dumps({"best_native": {"scale_count": scale_count, "metrics": metrics}}) + "\n"
    )


def _candidate(label: str, *, scales: int, bic: float) -> dict[str, object]:
    row_cost = 0.004 if label != "full" else 0.0
    endpoint_cost = 0.003 if label != "full" else 0.0
    return {
        "label": label,
        "scale_params": scales,
        "aggregate_bic": bic,
        "input_endpoint_rows": 1000,
        "row_rall": {"all": 0.20 + row_cost, "work": 0.19 + row_cost, "free": 0.23 + row_cost},
        "bloch_gain_rall": {"all": 0.08, "work": 0.07, "free": 0.09},
        "endpoint": {
            "R1_all": 0.08 + endpoint_cost,
            "R1_gt": 0.06 + endpoint_cost,
            "wR2_all": 0.12 + endpoint_cost,
            "GooF": 1.0,
            "n_reflections": 500,
            "npd_count": 0,
            "split_count": 0,
            "returncode": 0,
            "status": "ok",
            "safe_for_feedback": True,
            "artifacts_exist": True,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 10.0,
        },
        "physics_contract": {"q": 1.0, "wght": [0.0, 0.0]},
    }


def _selection_case(
    tmp_path: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, dict[str, object]]]:
    full_runs: dict[str, Path] = {}
    endpoint_runs: dict[str, Path] = {}
    polish: dict[str, dict[str, object]] = {}
    for name in ("a", "b"):
        full = tmp_path / name / "full"
        bic = tmp_path / name / "bic"
        strict = tmp_path / name / "strict"
        _write_run(full, scale_count=100)
        _write_run(bic, scale_count=80)
        _write_run(strict, scale_count=70)
        full_runs[name] = full
        endpoint_runs[name] = full
        polish_out = tmp_path / name / "polish"
        polish_out.mkdir()
        summary = {
            "selected_run_dir": str(bic),
            "selected_strict_run_dir": str(strict),
            "baseline": {"scale_params": 100, "information": {"bic_fitted_variance": 1000.0}},
            "selected_output": {
                "scale_params": 80,
                "information": {"bic_fitted_variance": 900.0},
            },
            "selected": {"scale_params": 70, "information": {"bic_fitted_variance": 925.0}},
        }
        (polish_out / "scale_compression_summary.json").write_text(json.dumps(summary) + "\n")
        polish[name] = {
            "status": "accepted",
            "out_dir": str(polish_out),
            "selected_compact_run": str(bic),
            "selected_strict_run": str(strict),
            **{key: summary[key] for key in ("baseline", "selected_output", "selected")},
        }
    return full_runs, endpoint_runs, polish


def _selection_args() -> SimpleNamespace:
    return SimpleNamespace(
        dry_run=False,
        multi_scale_contract_selection="auto",
        multi_scale_min_bloch_rall_gain=0.002,
        multi_scale_min_reduction=0.10,
        multi_scale_require_bic_improvement=True,
        multi_scale_max_row_rall_cost=0.01,
        multi_scale_max_endpoint_r1_cost=0.005,
        multi_scale_max_endpoint_wr2_cost=0.01,
        multi_scale_goof_min=0.85,
        multi_scale_goof_max=1.15,
        multi_scale_max_input_row_loss_fraction=0.001,
    )


def test_validate_scale_contract_run_map_allows_only_scale_predictions(tmp_path: Path) -> None:
    full = tmp_path / "full"
    compact = tmp_path / "compact"
    _write_run(full, scale_count=100)
    _write_run(compact, scale_count=70)
    rows = list(csv.DictReader((compact / "native_dyn_calc.csv").open()))
    rows[0]["dyn_scaled"] = "96"
    with (compact / "native_dyn_calc.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = cli.validate_scale_contract_run_map(
        {"a": full},
        {"a": compact},
        min_bloch_rall_gain=0.002,
    )

    assert result["ok"] is True
    assert result["datasets"]["a"]["scale_count"] == 70
    assert result["datasets"]["a"]["bloch_rall_gain"]["free"] == pytest.approx(0.10)


def test_validate_scale_contract_run_map_allows_scale_replay_assignments(tmp_path: Path) -> None:
    full = tmp_path / "full"
    compact = tmp_path / "compact"
    _write_run(full, scale_count=100)
    _write_run(compact, scale_count=70)
    rows = list(csv.DictReader((compact / "native_dyn_calc.csv").open()))
    for index, row in enumerate(rows):
        row["scale_replay_group"] = str(index)
        row["scale_replay_detector_bin"] = "0"
    with (compact / "native_dyn_calc.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = cli.validate_scale_contract_run_map(
        {"a": full},
        {"a": compact},
        min_bloch_rall_gain=0.002,
    )

    assert result["ok"] is True
    assert result["datasets"]["a"]["scale_count"] == 70


def test_validate_scale_contract_run_map_rejects_weak_held_out_bloch_gain(tmp_path: Path) -> None:
    full = tmp_path / "full"
    compact = tmp_path / "compact"
    _write_run(full, scale_count=100)
    _write_run(compact, scale_count=70, weak_free_bloch=True)

    result = cli.validate_scale_contract_run_map(
        {"a": full},
        {"a": compact},
        min_bloch_rall_gain=0.002,
    )

    assert result["ok"] is False
    assert result["reason"] == "bloch_rall_gain_below_minimum"
    assert result["split"] == "free"


@pytest.mark.parametrize(
    ("compact_born_wr2", "expected_ok", "expected_reason"),
    [
        (0.185, True, None),
        (0.179, False, "preexisting_bloch_gain_loss_exceeded"),
    ],
)
def test_preexisting_bloch_loss_is_preserved_without_requiring_a_false_win(
    tmp_path: Path,
    compact_born_wr2: float,
    expected_ok: bool,
    expected_reason: str | None,
) -> None:
    full = tmp_path / "full"
    compact = tmp_path / "compact"
    _write_run(full, scale_count=100)
    _write_run(compact, scale_count=70)
    for path, born_wr2 in ((full, 0.190), (compact, compact_born_wr2)):
        summary_path = path / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["best_native"]["metrics"]["free"]["born"]["R1_all"] = born_wr2
        summary_path.write_text(json.dumps(summary) + "\n")

    result = cli.validate_scale_contract_run_map(
        {"a": full},
        {"a": compact},
        min_bloch_rall_gain=0.002,
        max_preexisting_bloch_gain_loss=0.01,
    )

    assert result["ok"] is expected_ok
    if expected_reason is not None:
        assert result["reason"] == expected_reason
        assert result["split"] == "free"


def test_multi_selection_promotes_the_smallest_passing_materialized_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_runs: dict[str, Path] = {}
    endpoint_input_runs: dict[str, Path] = {}
    polish: dict[str, dict[str, object]] = {}
    for name in ("a", "b"):
        full = tmp_path / name / "full"
        bic = tmp_path / name / "bic"
        strict = tmp_path / name / "strict"
        _write_run(full, scale_count=100)
        _write_run(bic, scale_count=80)
        _write_run(strict, scale_count=70)
        full_runs[name] = full
        endpoint_input_runs[name] = full
        polish_out = tmp_path / name / "polish"
        polish_out.mkdir()
        summary_payload = {
            "selected_run_dir": str(bic),
            "selected_strict_run_dir": str(strict),
            "baseline": {"scale_params": 100, "information": {"bic_fitted_variance": 1000.0}},
            "selected_output": {"scale_params": 80, "information": {"bic_fitted_variance": 900.0}},
            "selected": {"scale_params": 70, "information": {"bic_fitted_variance": 925.0}},
        }
        (polish_out / "scale_compression_summary.json").write_text(json.dumps(summary_payload) + "\n")
        polish[name] = {
            "status": "accepted",
            "out_dir": str(polish_out),
            "selected_compact_run": str(bic),
            "selected_strict_run": str(strict),
            **{key: summary_payload[key] for key in ("baseline", "selected_output", "selected")},
        }
    excluded = tmp_path / "excluded" / "full"
    _write_run(excluded, scale_count=900)
    full_runs["excluded"] = excluded
    polish["excluded"] = {"status": "diagnostic_only"}

    monkeypatch.setattr(cli, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(cli, "run_logged", lambda *args, **kwargs: None)

    def fake_combined(label: str, _path: Path, *, scale_params: int, aggregate_bic: float):
        return _candidate(label, scales=scale_params, bic=aggregate_bic)

    monkeypatch.setattr(cli, "combined_scale_contract_candidate", fake_combined)
    args = SimpleNamespace(
        dry_run=False,
        multi_scale_contract_selection="auto",
        multi_scale_min_bloch_rall_gain=0.002,
        multi_scale_min_reduction=0.10,
        multi_scale_require_bic_improvement=True,
        multi_scale_max_row_rall_cost=0.01,
        multi_scale_max_endpoint_r1_cost=0.005,
        multi_scale_max_endpoint_wr2_cost=0.01,
        multi_scale_goof_min=0.85,
        multi_scale_goof_max=1.15,
        multi_scale_max_input_row_loss_fraction=0.001,
    )

    selected_input, selected_publication, decision = cli.run_multi_scale_contract_selection(
        python="python",
        full_runs=full_runs,
        full_input_runs=endpoint_input_runs,
        full_publication_runs=endpoint_input_runs,
        scale_polish=polish,
        final_dataset_policy={"status": "disabled"},
        final_model="model.res",
        root=tmp_path / "result",
        logs=tmp_path / "logs",
        args=args,
    )

    assert decision["status"] == "selected_compact"
    assert decision["selected_contract"] == "compact_min_strict"
    assert {path.name for path in selected_input.values()} == {"strict"}
    assert selected_publication == selected_input
    saved = json.loads((tmp_path / "result" / "scale_contract_selection" / "decision.json").read_text())
    assert saved["selected_contract"] == "compact_min_strict"
    assert saved["candidates"]["full"]["scale_params"] == 200
    projected_path = Path(saved["publication_scale_polish_summaries"]["a"])
    projected = json.loads(projected_path.read_text())
    assert projected["selected_output"]["scale_params"] == 70
    assert Path(projected["selected_run_dir"]).name == "strict"
    report_args = cli.scale_polish_report_args(
        SimpleNamespace(datasets=[]),
        {
            "combined": "combined",
            "scale_polish": polish,
            "scale_contract_selection": saved,
        },
    )
    assert any(item.endswith(str(projected_path)) for item in report_args)


def test_candidate_run_logged_failure_rejects_only_that_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_runs, endpoint_runs, polish = _selection_case(tmp_path)
    monkeypatch.setattr(cli, "combine_cmd", lambda *args, **kwargs: ["combine"])

    def fake_run_logged(command, _logs, label, **_kwargs):
        if label == "combine_scale_contract_compact_bic":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(cli, "run_logged", fake_run_logged)
    monkeypatch.setattr(
        cli,
        "combined_scale_contract_candidate",
        lambda label, _path, *, scale_params, aggregate_bic: _candidate(
            label,
            scales=scale_params,
            bic=aggregate_bic,
        ),
    )

    selected_input, _selected_publication, decision = cli.run_multi_scale_contract_selection(
        python="python",
        full_runs=full_runs,
        full_input_runs=endpoint_runs,
        full_publication_runs=endpoint_runs,
        scale_polish=polish,
        final_dataset_policy={"status": "disabled"},
        final_model="model.res",
        root=tmp_path / "result",
        logs=tmp_path / "logs",
        args=_selection_args(),
    )

    assert decision["status"] == "selected_compact"
    assert decision["selected_contract"] == "compact_min_strict"
    assert {path.name for path in selected_input.values()} == {"strict"}
    evaluations = {row["label"]: row for row in decision["evaluations"]}
    assert evaluations["compact_bic"]["accepted"] is False
    assert evaluations["compact_bic"]["reasons"] == ["candidate_replay_failed"]
    assert evaluations["compact_bic"]["diagnostics"]["failure_stage"] == "combined_replay"
    assert evaluations["compact_bic"]["diagnostics"]["error_type"] == "CalledProcessError"
    assert evaluations["compact_min_strict"]["accepted"] is True


def test_candidate_summary_parse_failure_preserves_another_viable_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_runs, endpoint_runs, polish = _selection_case(tmp_path)
    monkeypatch.setattr(cli, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(cli, "run_logged", lambda *args, **kwargs: None)

    def fake_candidate(label: str, _path: Path, *, scale_params: int, aggregate_bic: float):
        if label == "compact_min_strict":
            raise ValueError("malformed compact summary")
        return _candidate(label, scales=scale_params, bic=aggregate_bic)

    monkeypatch.setattr(cli, "combined_scale_contract_candidate", fake_candidate)

    selected_input, _selected_publication, decision = cli.run_multi_scale_contract_selection(
        python="python",
        full_runs=full_runs,
        full_input_runs=endpoint_runs,
        full_publication_runs=endpoint_runs,
        scale_polish=polish,
        final_dataset_policy={"status": "disabled"},
        final_model="model.res",
        root=tmp_path / "result",
        logs=tmp_path / "logs",
        args=_selection_args(),
    )

    assert decision["selected_contract"] == "compact_bic"
    assert {path.name for path in selected_input.values()} == {"bic"}
    evaluations = {row["label"]: row for row in decision["evaluations"]}
    assert evaluations["compact_min_strict"]["reasons"] == [
        "candidate_summary_parsing_failed"
    ]
    assert evaluations["compact_min_strict"]["diagnostics"]["error"] == (
        "malformed compact summary"
    )


def test_candidate_policy_failure_keeps_full_when_no_other_candidate_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_runs, endpoint_runs, polish = _selection_case(tmp_path)
    monkeypatch.setattr(cli, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(cli, "run_logged", lambda *args, **kwargs: None)
    original_policy = cli._policy_weight_scale_contract_runs

    def fake_policy(runs, policy, out_dir):
        if {Path(path).name for path in runs.values()} == {"strict"}:
            raise OSError("strict policy output unavailable")
        return original_policy(runs, policy, out_dir)

    monkeypatch.setattr(cli, "_policy_weight_scale_contract_runs", fake_policy)

    def fake_candidate(label: str, _path: Path, *, scale_params: int, aggregate_bic: float):
        candidate = _candidate(label, scales=scale_params, bic=aggregate_bic)
        if label == "compact_bic":
            candidate["endpoint"]["npd_count"] = 1
        return candidate

    monkeypatch.setattr(cli, "combined_scale_contract_candidate", fake_candidate)

    selected_input, selected_publication, decision = cli.run_multi_scale_contract_selection(
        python="python",
        full_runs=full_runs,
        full_input_runs=endpoint_runs,
        full_publication_runs=endpoint_runs,
        scale_polish=polish,
        final_dataset_policy={"status": "disabled"},
        final_model="model.res",
        root=tmp_path / "result",
        logs=tmp_path / "logs",
        args=_selection_args(),
    )

    assert decision["status"] == "kept_full"
    assert decision["selected_contract"] == "full"
    assert selected_input == endpoint_runs
    assert selected_publication == endpoint_runs
    evaluations = {row["label"]: row for row in decision["evaluations"]}
    assert "endpoint_npd_present" in evaluations["compact_bic"]["reasons"]
    assert evaluations["compact_min_strict"]["reasons"] == [
        "candidate_policy_weighting_failed"
    ]
    assert evaluations["compact_min_strict"]["diagnostics"]["failure_stage"] == (
        "policy_weighting"
    )


def test_unexpected_candidate_preflight_failure_is_recorded_per_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_runs, endpoint_runs, polish = _selection_case(tmp_path)
    original_validate = cli.validate_scale_contract_run_map

    def fake_validate(full, candidate, **kwargs):
        if {Path(path).name for path in candidate.values()} == {"strict"}:
            raise ValueError("strict row table could not be parsed")
        return original_validate(full, candidate, **kwargs)

    monkeypatch.setattr(cli, "validate_scale_contract_run_map", fake_validate)
    monkeypatch.setattr(cli, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(cli, "run_logged", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "combined_scale_contract_candidate",
        lambda label, _path, *, scale_params, aggregate_bic: _candidate(
            label,
            scales=scale_params,
            bic=aggregate_bic,
        ),
    )

    selected_input, _selected_publication, decision = cli.run_multi_scale_contract_selection(
        python="python",
        full_runs=full_runs,
        full_input_runs=endpoint_runs,
        full_publication_runs=endpoint_runs,
        scale_polish=polish,
        final_dataset_policy={"status": "disabled"},
        final_model="model.res",
        root=tmp_path / "result",
        logs=tmp_path / "logs",
        args=_selection_args(),
    )

    assert decision["selected_contract"] == "compact_bic"
    assert {path.name for path in selected_input.values()} == {"bic"}
    assert decision["preflight"]["compact_min_strict"] == {
        "ok": False,
        "reason": "candidate_preflight_failed",
        "failure_stage": "preflight",
        "error_type": "ValueError",
        "error": "strict row table could not be parsed",
    }
