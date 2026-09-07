"""CLI stage commands regression tests."""

import argparse
import json
from pathlib import Path

import pytest

import grond.cli as cli
import grond.workflows.scale_materialization as owner_workflows_scale_materialization
from grond.cli import (
    build_parser,
    combine_cmd,
    resolved_dataset_sigma_policy,
    resolved_merge_sigma_policy,
    resolved_post_bloch_sigma_policy,
)
from grond.io.config import DatasetSpec
from tests.cli._support import valid_shelx_res_text


def test_post_bloch_sigma_auto_is_final_endpoint_only():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"])

    assert resolved_post_bloch_sigma_policy(args) == "none"
    assert resolved_post_bloch_sigma_policy(args, final_endpoint=True) == "uniform_z_rms"
    assert resolved_post_bloch_sigma_policy(args, feedback_loop=True) == "uniform_z_rms"

    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--post-bloch-sigma-policy",
            "none",
        ]
    )
    assert resolved_post_bloch_sigma_policy(args, final_endpoint=True) == "none"


def test_multi_dataset_sigma_auto_uses_conservative_combination_defaults():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"])

    assert resolved_dataset_sigma_policy(args, multi_dataset=False) == "none"
    assert resolved_dataset_sigma_policy(args, multi_dataset=True) == "uniform_z_rms"
    assert resolved_merge_sigma_policy(args, multi_dataset=False) == "formal"
    assert resolved_merge_sigma_policy(args, multi_dataset=True) == "chisq_overdispersion"

    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--dataset-sigma-policy",
            "none",
            "--merge-sigma-policy",
            "formal",
        ]
    )
    assert resolved_dataset_sigma_policy(args, multi_dataset=True) == "none"
    assert resolved_merge_sigma_policy(args, multi_dataset=True) == "formal"


def test_combine_cmd_forwards_promoted_multi_dataset_sigma_policies(tmp_path: Path):
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--shelxl",
            "/opt/shelx/shelxl",
            "--shelxl-iterations",
            "3",
        ]
    )

    cmd = combine_cmd(
        "python",
        {"a": tmp_path / "run_a", "b": tmp_path / "run_b"},
        "model.res",
        tmp_path / "combined",
        args,
    )

    assert cmd[cmd.index("--dataset-sigma-policy") + 1] == "uniform_z_rms"
    assert cmd[cmd.index("--merge-sigma-policy") + 1] == "chisq_overdispersion"
    assert cmd[cmd.index("--post-bloch-sigma-policy") + 1] == "uniform_z_rms"
    assert cmd[cmd.index("--shelxl") + 1] == "/opt/shelx/shelxl"
    assert cmd[cmd.index("--shelxl-iterations") + 1] == "3"
    assert cmd[cmd.index("--endpoint-ls-cycles") + 1] == "1"
    assert cmd[cmd.index("--endpoint-wght-mode") + 1] == "freeze-zero"
    assert "--dataset-sigma-holdout-floor" in cmd


def test_combine_cmd_forwards_explicit_endpoint_resolution_cutoff(tmp_path: Path):
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--endpoint-min-d-spacing",
            "0.75",
        ]
    )

    cmd = combine_cmd(
        "python",
        {"a": tmp_path / "run_a", "b": tmp_path / "run_b"},
        "model.res",
        tmp_path / "combined",
        args,
    )

    assert cmd[cmd.index("--min-d-spacing") + 1] == "0.75"


def test_materialize_dataset_scale_contract_uses_endpoint_free_replay(tmp_path: Path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "native_dyn_calc.csv").write_text("h,k,l\n0,0,1\n")
    (raw / "summary.json").write_text("{}\n")
    captured = []

    def fake_run_logged(command, _logs, label, **_kwargs):
        captured.extend([command, label])
        out_dir = Path(command[command.index("--out-dir") + 1])
        materialized = out_dir / "adaptive_2_det_8"
        materialized.mkdir(parents=True)
        (materialized / "native_dyn_calc.csv").write_text("h,k,l\n0,0,1\n")
        (materialized / "summary.json").write_text("{}\n")
        (out_dir / "summary.json").write_text(
            json.dumps({"materialized_runs": {"adaptive_2_det_8": str(materialized)}}) + "\n"
        )

    monkeypatch.setattr(owner_workflows_scale_materialization, "run_logged", fake_run_logged)
    selected = cli.materialize_dataset_scale_contract(
        "python",
        raw,
        argparse.Namespace(dataset_scale_specs={"a": "adaptive:2:det8"}),
        "a",
        tmp_path / "logs",
        label="cycle_01",
    )

    assert selected == raw / "scale_contract_replay" / "adaptive_2_det_8"
    assert "--skip-endpoint-refinement" in captured[0]
    assert captured[1] == "a_cycle_01_scale_contract"


@pytest.mark.parametrize(
    ("stage", "endpoint", "expected_skip"),
    [
        ("screen", True, False),
        ("adaptive_final", True, False),
        ("multi_ls1_cycle_01", False, True),
    ],
)
def test_native_stage_cmd_forwards_explicit_non_path_shelxl(
    tmp_path: Path,
    stage: str,
    endpoint: bool,
    expected_skip: bool,
):
    model = tmp_path / "model.res"
    model.write_text(valid_shelx_res_text())
    explicit_shelxl = tmp_path / "private-ccp4" / "shelxl"
    explicit_shelxl.parent.mkdir()
    explicit_shelxl.write_text("#!/bin/sh\nexit 0\n")
    explicit_shelxl.chmod(0o755)
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            str(tmp_path / "out"),
            "--xds-dir",
            str(tmp_path / "xds"),
            "--structure",
            str(model),
            "--shelxl",
            str(explicit_shelxl),
        ]
    )
    dataset = DatasetSpec(name="sample", hkl=str(tmp_path / "rows.hkl"))

    command = cli.native_stage_cmd(
        "python",
        dataset,
        str(model),
        "",
        tmp_path / stage,
        args,
        n_basis=24,
        n_points=10,
        stage=stage,
        endpoint=endpoint,
    )

    assert command.count("--shelxl") == 1
    assert command[command.index("--shelxl") + 1] == str(explicit_shelxl)
    assert ("--skip-endpoint-refinement" in command) is expected_skip


def test_observation_polish_cmd_forwards_explicit_non_path_shelxl(tmp_path: Path):
    explicit_shelxl = tmp_path / "private-ccp4" / "shelxl"
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            str(tmp_path / "out"),
            "--xds-dir",
            str(tmp_path / "xds"),
            "--structure",
            str(tmp_path / "model.res"),
            "--shelxl",
            str(explicit_shelxl),
        ]
    )

    command = cli.observation_polish_cmd(
        "python",
        tmp_path / "rows.csv",
        str(tmp_path / "rows.hkl"),
        str(tmp_path / "model.res"),
        tmp_path / "forensics",
        args,
    )

    assert command.count("--shelxl") == 1
    assert command[command.index("--shelxl") + 1] == str(explicit_shelxl)
