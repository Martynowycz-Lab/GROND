"""CLI dataset and endpoint policy regression tests."""

import argparse
from pathlib import Path

import pytest

import grond.cli as cli
from grond.io.config import DatasetSpec, PrepConfig


def test_fresh_baseline_no_harm_gate_warns_but_keeps_worse_endpoint_available():
    summary = {
        "baseline_exti": {
            "status": "ok",
            "R1_all": 0.1377,
            "npd_count": 0,
            "split_count": 0,
            "lst": "fresh.lst",
            "res_health": {"safe_for_feedback": True, "path": "fresh.res"},
        }
    }

    rejected = cli._endpoint_baseline_no_harm_gate(
        summary,
        {"shelxl_R1all": 0.1404},
    )
    accepted = cli._endpoint_baseline_no_harm_gate(
        summary,
        {"shelxl_R1all": 0.1330},
    )

    assert rejected["status"] == "warning_worse_than_baseline"
    assert rejected["passed"] is True
    assert rejected["no_harm"] is False
    assert rejected["reasons"] == []
    assert any("warning only" in warning for warning in rejected["warnings"])
    assert rejected["delta_R1all"] == pytest.approx(0.0027)
    assert rejected["baseline_res"] == "fresh.res"
    assert accepted["status"] == "passed"
    assert accepted["no_harm"] is True


def test_multi_endpoint_calibration_auto_skips_unsafe_single_table_replay(tmp_path: Path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    for run in (run_a, run_b):
        run.mkdir()
        (run / "native_dyn_calc.csv").write_text("h,k,l\n1,0,0\n")
    combined = tmp_path / "combined"
    combined.mkdir()
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="a", hkl="a.hkl"), DatasetSpec(name="b", hkl="b.hkl")],
    )
    result = {
        "combined": str(combined),
        "combined_input_runs": {"a": str(run_a), "b": str(run_b)},
    }
    args = argparse.Namespace(endpoint_calibration="auto", fast=False, dry_run=False, out_dir=str(tmp_path))

    status = cli.run_endpoint_calibration(prep, result, args, "python", tmp_path / "logs")

    assert status["status"] == "not_applicable_multi_dataset"
    assert status["row_tables"] == {
        "a": str(run_a / "native_dyn_calc.csv"),
        "b": str(run_b / "native_dyn_calc.csv"),
    }
    assert status["row_tables_exist"] == {"a": True, "b": True}
    assert "single-table" in status["reason"]


def test_multi_endpoint_calibration_forced_on_fails_closed(tmp_path: Path):
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="a", hkl="a.hkl"), DatasetSpec(name="b", hkl="b.hkl")],
    )
    result = {"combined": str(tmp_path / "combined")}
    args = argparse.Namespace(endpoint_calibration="on", fast=False, dry_run=False, out_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="not supported for multi-dataset"):
        cli.run_endpoint_calibration(prep, result, args, "python", tmp_path / "logs")


def test_missing_materialized_policy_stops_instead_of_falling_back():
    with pytest.raises(RuntimeError, match="refusing to silently fall back"):
        cli.require_materialized_dataset_policy(
            {"status": "missing_row_table", "missing": ["a/native_dyn_calc.csv"]},
            context="structure-update",
            dry_run=False,
        )

    cli.require_materialized_dataset_policy(
        {"status": "missing_row_table", "missing": ["planned/native_dyn_calc.csv"]},
        context="structure-update",
        dry_run=True,
    )


def test_evaluated_dataset_policy_never_resurrects_excluded_runs(tmp_path: Path):
    runs = {"a": tmp_path / "a", "b": tmp_path / "b"}
    policy = {
        "status": "ok",
        "datasets": [
            {"name": "a", "action": "exclude"},
            {"name": "b", "action": "completeness_only"},
        ],
    }

    active = cli.policy_active_runs(runs, policy)

    assert active == {}
    with pytest.raises(RuntimeError, match="no primary or downweighted dataset"):
        cli.require_policy_active_runs(active, context="test")


def test_dataset_policy_returns_only_raw_row_anchor_runs(tmp_path: Path):
    runs = {"a": tmp_path / "a", "b": tmp_path / "b", "c": tmp_path / "c"}
    policy = {
        "status": "ok",
        "datasets": [
            {"name": "a", "action": "primary"},
            {"name": "b", "action": "downweight"},
            {"name": "c", "action": "completeness_only"},
        ],
    }

    active = cli.policy_active_runs(runs, policy)

    assert active == {"a": runs["a"], "b": runs["b"]}
    cli.require_policy_active_runs(active, context="test")
