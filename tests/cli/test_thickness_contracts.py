"""CLI thickness contracts regression tests."""

import argparse
import json
from pathlib import Path

import pytest

import grond.cli as cli
import grond.cli_support.execution as owner_cli_support_execution
import grond.workflows.thickness_materialization as owner_workflows_thickness_materialization
from grond.cli import (
    build_parser,
    resolved_sg_window_mode,
)
from grond.io.config import DatasetSpec, PrepConfig
from tests.cli._support import valid_shelx_res_text


def test_q_field_stability_compares_shared_signed_rows(tmp_path: Path):
    reference = tmp_path / "ref.csv"
    candidate = tmp_path / "cand.csv"
    reference.write_text(
        "row_index,h,k,l,q_dyn_over_born\n"
        "0,1,0,0,2.0\n"
        "1,0,1,0,4.0\n"
        "2,0,0,1,8.0\n"
    )
    candidate.write_text(
        "row_index,h,k,l,q_dyn_over_born\n"
        "0,1,0,0,2.0\n"
        "1,0,1,0,4.0\n"
        "2,0,0,1,16.0\n"
    )

    result = cli._q_field_stability(reference, candidate)

    assert result["shared_rows"] == 3
    assert result["logq_corr"] > 0.98
    assert result["p95_abs_dlogq"] == pytest.approx(0.693147, rel=1e-5)


def test_continuous_nuisance_is_materialized_on_full_row_contract(tmp_path: Path, monkeypatch):
    selected = {
        "thickness_A": 875.0,
        "thickness_grad_frac": 0.8,
        "thickness_model": "uniform_gradient",
        "frame_offset": 0.0,
        "group_width_deg": 1.0,
        "detector_scale_bins": 8,
        "row_resolution_b": 0.0,
        "sg_jitter_mode": "none",
        "sg_jitter_sigma0": 0.0,
        "sg_jitter_sigma_d_inv": 0.0,
        "sg_jitter_sigma_frame": 0.0,
        "absorption_mode": "none",
        "absorption_mu": 0.0,
        "absorption_frac": 0.0,
        "dose_b_mode": "none",
        "dose_b_per_frame": 0.0,
        "dose_b_ref_frame": float("nan"),
        "matrix_convention": "j_minus_i",
    }
    continuous = tmp_path / "continuous"
    continuous.mkdir()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    baseline_nuisance = {**selected, "thickness_A": 900.0}

    def summary(nuisance, *, all_wr2, free_wr2, all_r1):
        return {
            "best_native": {
                "nuisance": nuisance,
                "metrics": {
                    "all": {
                        "dyn": {"wR2_all": all_wr2, "R1_all": all_r1, "R1_gt": all_r1, "CC": 0.9},
                        "born": {"R1_gt": 0.2, "CC": 0.9},
                    },
                    "free": {"dyn": {"wR2_all": free_wr2, "n": 10}},
                },
            }
        }

    (baseline / "summary.json").write_text(
        json.dumps(summary(baseline_nuisance, all_wr2=0.12, free_wr2=0.13, all_r1=0.13)) + "\n"
    )
    captured = {}

    def fake_native_stage_cmd(_python, _ds, _structure, _baseline, out_dir, _args, **kwargs):
        captured["out_dir"] = out_dir
        captured["pinned"] = dict(kwargs["pinned"])
        out_dir.mkdir(parents=True)
        (out_dir / "native_dyn_calc.csv").write_text("h,k,l\n1,0,0\n")
        (out_dir / "summary.json").write_text(
            json.dumps(summary(selected, all_wr2=0.10, free_wr2=0.11, all_r1=0.12)) + "\n"
        )
        return ["fake-native"]

    monkeypatch.setattr(owner_workflows_thickness_materialization, "native_stage_cmd", fake_native_stage_cmd)
    monkeypatch.setattr(owner_cli_support_execution, "run_logged", lambda *_args, **_kwargs: None)
    prep = PrepConfig(structure=str(tmp_path / "model.res"), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    ds = prep.datasets[0]
    args = argparse.Namespace(
        dry_run=False,
        continuous_thickness_min_gain=5e-5,
        continuous_thickness_max_free_wr2_loss=0.002,
        continuous_thickness_max_all_r1_loss=0.001,
    )

    nuisance, run, record = cli.materialize_continuous_nuisance_run(
        prep,
        ds,
        args,
        "python",
        tmp_path / "logs",
        active_model=prep.structure,
        baseline_run=baseline,
        continuous_run=continuous,
        selected_nuisance=selected,
        n_basis=40,
        n_points=10,
        stage="continuous_materialized",
    )

    assert captured["pinned"] == selected
    assert run == continuous / "materialized_full_rows"
    assert (run / "native_dyn_calc.csv").exists()
    assert nuisance["thickness_A"] == 875.0
    assert record["nuisance_contract_verified"] is True
    assert record["status"] == "accepted_full_row"
    assert record["full_row_score_gain"] > 0.0


@pytest.mark.parametrize(
    ("expected", "actual", "matches"),
    [
        (float("nan"), float("nan"), True),
        ("nan", float("nan"), True),
        (float("nan"), 0.0, False),
        (0.0, float("nan"), False),
        (float("inf"), float("inf"), False),
        ("not-a-number", "not-a-number", False),
        (1204.5805095106, 1204.580510, True),
        (1204.5805095106, 1204.6, False),
    ],
)
def test_numeric_nuisance_contract_handles_nan_only_as_matching_na(
    expected: object,
    actual: object,
    matches: bool,
) -> None:
    assert cli.numeric_nuisance_contract_matches(expected, actual) is matches


def test_discrete_thickness_baseline_fails_closed_but_has_dry_run_plan_seed():
    with pytest.raises(RuntimeError, match="no starting nuisance"):
        cli.discrete_thickness_baseline_nuisance(argparse.Namespace(dry_run=False), {})

    assert cli.discrete_thickness_baseline_nuisance(argparse.Namespace(dry_run=True), {}) == {
        "thickness_A": 1200.0,
        "thickness_grad_frac": 0.0,
        "frame_offset": 0.0,
    }


def test_continuous_full_row_gate_restores_coarse_baseline_on_regression(tmp_path: Path, monkeypatch):
    baseline = tmp_path / "coarse"
    continuous = tmp_path / "continuous"
    baseline.mkdir()
    continuous.mkdir()
    baseline_nuisance = {
        "thickness_A": 1200.0,
        "thickness_grad_frac": 0.4,
        "thickness_model": "uniform_gradient",
        "frame_offset": 0.0,
        "group_width_deg": 1.0,
        "detector_scale_bins": 0,
        "matrix_convention": "j_minus_i",
    }
    selected = {**baseline_nuisance, "thickness_A": 700.0, "thickness_grad_frac": 1.0}

    def payload(nuisance, all_wr2, free_wr2, all_r1):
        return {
            "best_native": {
                "nuisance": nuisance,
                "metrics": {
                    "all": {
                        "dyn": {"wR2_all": all_wr2, "R1_all": all_r1, "R1_gt": all_r1, "CC": 0.9},
                        "born": {"R1_gt": 0.2, "CC": 0.9},
                    },
                    "free": {"dyn": {"wR2_all": free_wr2, "n": 10}},
                },
            }
        }

    (baseline / "summary.json").write_text(json.dumps(payload(baseline_nuisance, 0.10, 0.11, 0.12)) + "\n")

    def fake_native_stage_cmd(_python, _ds, _structure, _baseline, out_dir, _args, **_kwargs):
        out_dir.mkdir(parents=True)
        (out_dir / "native_dyn_calc.csv").write_text("h,k,l\n1,0,0\n")
        (out_dir / "summary.json").write_text(json.dumps(payload(selected, 0.11, 0.13, 0.13)) + "\n")
        return ["fake-native"]

    monkeypatch.setattr(owner_workflows_thickness_materialization, "native_stage_cmd", fake_native_stage_cmd)
    monkeypatch.setattr(owner_cli_support_execution, "run_logged", lambda *_args, **_kwargs: None)
    prep = PrepConfig(structure=str(tmp_path / "model.res"), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    args = argparse.Namespace(
        dry_run=False,
        continuous_thickness_min_gain=5e-5,
        continuous_thickness_max_free_wr2_loss=0.002,
        continuous_thickness_max_all_r1_loss=0.001,
    )

    nuisance, run, record = cli.materialize_continuous_nuisance_run(
        prep,
        prep.datasets[0],
        args,
        "python",
        tmp_path / "logs",
        active_model=prep.structure,
        baseline_run=baseline,
        continuous_run=continuous,
        selected_nuisance=selected,
        n_basis=40,
        n_points=10,
        stage="continuous_materialized",
    )

    assert run == baseline
    assert nuisance["thickness_A"] == 1200.0
    assert record["status"] == "rejected_full_row"
    assert "full_row_gain_below_minimum" in record["promotion_reasons"]
    assert "full_row_free_wR2_loss" in record["promotion_reasons"]


def test_continuous_full_row_gate_requires_heldout_rows(tmp_path: Path):
    baseline = tmp_path / "baseline"
    continuous = tmp_path / "continuous"
    baseline.mkdir()
    continuous.mkdir()
    nuisance = {
        "thickness_A": 900.0,
        "thickness_grad_frac": 0.5,
        "thickness_model": "uniform_gradient",
    }

    def payload(*, all_wr2: float, free_wr2: float, free_n: int) -> dict:
        return {
            "best_native": {
                "nuisance": nuisance,
                "metrics": {
                    "all": {
                        "dyn": {"wR2_all": all_wr2, "R1_all": 0.10, "R1_gt": 0.08, "CC": 0.9},
                        "born": {"R1_gt": 0.2, "CC": 0.9},
                    },
                    "free": {"dyn": {"wR2_all": free_wr2, "n": free_n}},
                },
            }
        }

    (baseline / "summary.json").write_text(json.dumps(payload(all_wr2=0.12, free_wr2=0.13, free_n=10)))
    (continuous / "summary.json").write_text(json.dumps(payload(all_wr2=0.08, free_wr2=0.09, free_n=0)))
    (continuous / "native_dyn_calc.csv").write_text("h,k,l\n1,0,0\n")
    prep = PrepConfig(structure=str(tmp_path / "model.res"), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    args = argparse.Namespace(
        dry_run=False,
        continuous_thickness_min_gain=5e-5,
        continuous_thickness_max_free_wr2_loss=0.002,
        continuous_thickness_max_all_r1_loss=0.001,
    )

    _selected, selected_run, record = cli.materialize_continuous_nuisance_run(
        prep,
        prep.datasets[0],
        args,
        "python",
        tmp_path / "logs",
        active_model=prep.structure,
        baseline_run=baseline,
        continuous_run=continuous,
        selected_nuisance=nuisance,
        n_basis=40,
        n_points=10,
        stage="continuous_materialized",
    )

    assert selected_run == baseline
    assert record["candidate_free_rows"] == 0
    assert "candidate_missing_positive_free_rows" in record["promotion_reasons"]


def test_auto_sg_window_mode_keeps_fixed_width_unless_explicit():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"])
    ds = DatasetSpec(name="d", hkl="missing.INTEGRATE.HKL")

    assert resolved_sg_window_mode(args, ds) == "fixed"

    args = build_parser().parse_args(
        ["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res", "--sg-window-mode", "dials_bbox"]
    )
    assert resolved_sg_window_mode(args, ds) == "dials_bbox"


def test_pinned_sg_jitter_gets_explicit_no_jitter_fallback(tmp_path: Path):
    args = build_parser().parse_args(
        ["run", "--out-dir", "out", "--hkl", "rows.hkl", "--structure", "model.res"]
    )
    dataset = DatasetSpec(name="sample", hkl="rows.hkl")
    cmd = cli.native_stage_cmd(
        "python",
        dataset,
        "model.res",
        "baseline.hkl",
        tmp_path / "run",
        args,
        n_basis=20,
        n_points=5,
        stage="multi_ls1_cycle_01",
        pinned={"sg_jitter_mode": "gaussian", "sg_jitter_sigma0": 0.0015},
        endpoint=False,
    )

    assert cmd[cmd.index("--sg-jitter-mode-grid") + 1] == "gaussian,none"


def test_native_stage_cmd_forwards_discrete_thickness_promotion_contract(tmp_path: Path):
    model = tmp_path / "model.res"
    model.write_text(valid_shelx_res_text())
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            str(tmp_path / "out"),
            "--xds-dir",
            str(tmp_path / "xds"),
            "--structure",
            str(model),
        ]
    )
    dataset = DatasetSpec(name="sample", hkl=str(tmp_path / "rows.hkl"))
    args.selection_metric = "free_wR2"
    args.production_scales = True
    baseline = {
        "thickness_A": 1204.5805095106,
        "thickness_grad_frac": 0.8,
        "dose_b_ref_frame": float("nan"),
    }

    command = cli.native_stage_cmd(
        "python",
        dataset,
        str(model),
        "",
        tmp_path / "thickness",
        args,
        n_basis=24,
        n_points=10,
        stage="multi_thickness_polish",
        pinned=baseline,
        promotion_baseline=baseline,
        scale_fit_split_override="work",
        selection_metric_override="work_wR2",
        endpoint=False,
    )

    forwarded = json.loads(command[command.index("--promotion-baseline-json") + 1])
    assert forwarded["thickness_A"] == baseline["thickness_A"]
    assert forwarded["thickness_grad_frac"] == baseline["thickness_grad_frac"]
    assert cli.math.isnan(forwarded["dose_b_ref_frame"])
    assert float(command[command.index("--promotion-min-gain") + 1]) == args.continuous_thickness_min_gain
    assert float(command[command.index("--promotion-max-free-wr2-loss") + 1]) == args.continuous_thickness_max_free_wr2_loss
    assert float(command[command.index("--promotion-max-all-r1-loss") + 1]) == args.continuous_thickness_max_all_r1_loss
    assert command[command.index("--scale-fit-split") + 1] == "work"
    assert command[command.index("--selection-metric") + 1] == "work_wR2"
