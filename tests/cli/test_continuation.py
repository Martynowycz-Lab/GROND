"""CLI continuation regression tests."""

import argparse
import json
from pathlib import Path

import pytest

import grond.cli as cli
from grond.cli import (
    build_parser,
)
from grond.io.config import DatasetSpec, PrepConfig


def test_push_endpoint_exposes_diagnostic_endpoint_ls_cycles():
    args = build_parser().parse_args(
        [
            "push-endpoint",
            "--source-summary",
            "summary.json",
            "--start-res",
            "model.res",
            "--out-dir",
            "out",
            "--endpoint-ls-cycles",
            "20",
        ]
    )

    assert args.endpoint_ls_cycles == 20


def test_push_endpoint_native_weight_override_is_global_and_explicit():
    args = build_parser().parse_args(
        [
            "push-endpoint",
            "--source-summary",
            "summary.json",
            "--start-res",
            "model.res",
            "--out-dir",
            "out",
            "--native-wght-a",
            "0.179443",
            "--native-wght-b",
            "0.131843",
        ]
    )
    nuisance = {
        "a": {"native_wght_a": 0.0, "native_wght_b": 0.0},
        "b": {"native_wght_a": 0.2, "native_wght_b": 0.3},
    }

    result = cli.override_push_native_weights(nuisance, args)

    assert result is nuisance
    assert {block["native_wght_a"] for block in nuisance.values()} == {0.179443}
    assert {block["native_wght_b"] for block in nuisance.values()} == {0.131843}


def test_write_continue_checkpoint_records_single_dataset_paths(tmp_path):
    out_dir = tmp_path / "grond_out"
    run_dir = out_dir / "runs" / "dataset" / "final"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}\n")
    model = out_dir / "final_model.res"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("TITL final\nEND\n")
    hkl = tmp_path / "INTEGRATE.HKL"
    hkl.write_text("! rows\n")
    baseline = tmp_path / "baseline.hkl"
    baseline.write_text("0 0 1 1.0 1.0\n")
    prep = PrepConfig(
        structure=str(model),
        baseline_hkl=str(baseline),
        datasets=[DatasetSpec(name="dataset", hkl=str(hkl), hkl_transform="identity")],
    )
    args = argparse.Namespace(out_dir=str(out_dir))

    checkpoint = cli.write_continue_checkpoint(prep, {"primary_run": str(run_dir), "model": str(model)}, args)
    saved = json.loads((out_dir / "grond_continue_checkpoint.json").read_text())

    assert checkpoint["status"] == "ready"
    assert saved["kind"] == "grond_continue_checkpoint"
    assert saved["source_summary"] == str((run_dir / "summary.json").resolve())
    assert saved["recommended_start_res"] == str(model.resolve())
    assert saved["hkl"] == str(hkl.resolve())
    assert saved["baseline_hkl"] == str(baseline.resolve())
    assert saved["hkl_transform"] == "identity"


def test_write_continue_checkpoint_records_multi_dataset_contract(tmp_path):
    out_dir = tmp_path / "grond_multi"
    model = out_dir / "final_model.res"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("TITL final\nCELL 0.025079 5 6 7 90 90 90\nEND\n")
    run_a = out_dir / "runs" / "a" / "final"
    run_b = out_dir / "runs" / "b" / "final"
    for run, thickness, transform in [(run_a, 1000.0, "-h,k,l"), (run_b, 1200.0, "identity")]:
        run.mkdir(parents=True)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "config": {
                        "hkl_transform": transform,
                        "topup_d_min": 0.7,
                        "compute_backend": "scipy",
                        "row_plan_cache": "off",
                        "chunksize": 4,
                        "post_bloch_sigma_policy": "none",
                    },
                    "best_native": {"nuisance": {"thickness_A": thickness}},
                }
            )
            + "\n"
        )
    hkl_a = tmp_path / "a" / "INTEGRATE.HKL"
    hkl_b = tmp_path / "b" / "INTEGRATE.HKL"
    hkl_a.parent.mkdir()
    hkl_b.parent.mkdir()
    hkl_a.write_text("! a\n")
    hkl_b.write_text("! b\n")
    prep = PrepConfig(
        structure=str(model),
        datasets=[
            DatasetSpec(name="a", hkl=str(hkl_a), hkl_transform="identity"),
            DatasetSpec(name="b", hkl=str(hkl_b), hkl_transform="-h,k,l"),
        ],
    )
    args = argparse.Namespace(out_dir=str(out_dir), dry_run=False, multi_endpoint_merge_key="model_symmetry")

    checkpoint = cli.write_continue_checkpoint(
        prep,
        {
            "runs": {"a": str(run_a), "b": str(run_b)},
            "model": str(model),
            "dataset_policy": {"structure_update": {"status": "disabled"}},
        },
        args,
    )
    saved = json.loads((out_dir / "grond_continue_checkpoint.json").read_text())

    assert checkpoint["status"] == "ready"
    assert saved["dataset_count"] == 2
    assert saved["datasets"][0]["source_summary"] == str((run_a / "summary.json").resolve())
    assert saved["datasets"][0]["hkl_transform"] == "-h,k,l"
    assert saved["datasets"][1]["nuisance"]["thickness_A"] == 1200.0
    assert saved["datasets"][1]["final_config"]["topup_d_min"] == 0.7
    assert saved["multi_endpoint_merge_key"] == "model_symmetry"
    assert not any("reselected after" in note for note in saved["notes"])
    assert any("does not automatically rerun compact scale selection" in note for note in saved["notes"])


def test_push_endpoint_dry_run_accepts_continue_checkpoint(tmp_path):
    source_summary = tmp_path / "summary.json"
    source_summary.write_text("{}\n")
    start_res = tmp_path / "edited.res"
    start_res.write_text("TITL edited\nLATT 1\nL.S. 4\nEND\n")
    hkl = tmp_path / "INTEGRATE.HKL"
    hkl.write_text("! rows\n")
    checkpoint = tmp_path / "grond_continue_checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "grond_continue_checkpoint",
                "status": "ready",
                "source_summary": str(source_summary),
                "recommended_start_res": str(start_res),
                "hkl": str(hkl),
                "hkl_transform": "identity",
            }
        )
        + "\n"
    )
    out_dir = tmp_path / "push"
    args = build_parser().parse_args(
        [
            "push-endpoint",
            "--checkpoint",
            str(checkpoint),
            "--start-res",
            str(start_res),
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ]
    )

    assert args.func(args) == 0
    plan = json.loads((out_dir / "push_endpoint_plan.json").read_text())
    command = [str(item) for item in plan["command"]]

    assert "--source-summary" in command
    assert command[command.index("--source-summary") + 1] == str(source_summary.resolve())
    assert "--hkl" in command
    assert command[command.index("--hkl") + 1] == str(hkl.resolve())
    assert "--hkl-transform=identity" in command
    assert plan["checkpoint"] == str(checkpoint.resolve())


def test_push_endpoint_dry_run_accepts_multi_continue_checkpoint(tmp_path):
    out_dir = tmp_path / "push_multi"
    start_res = tmp_path / "edited.res"
    start_res.write_text("TITL edited\nCELL 0.025079 5 6 7 90 90 90\nLATT 1\nL.S. 4\nEND\n")
    run_a = tmp_path / "runs" / "a"
    run_b = tmp_path / "runs" / "b"
    for run, thickness in [(run_a, 1000.0), (run_b, 1200.0)]:
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps({"config": {"n_basis": 24, "n_points": 20}, "best_native": {"nuisance": {"thickness_A": thickness}}}) + "\n")
    hkl_a = tmp_path / "a" / "INTEGRATE.HKL"
    hkl_b = tmp_path / "b" / "INTEGRATE.HKL"
    hkl_a.parent.mkdir()
    hkl_b.parent.mkdir()
    hkl_a.write_text("! a\n")
    hkl_b.write_text("! b\n")
    checkpoint = tmp_path / "grond_continue_checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "grond_continue_checkpoint",
                "status": "ready",
                "dataset_count": 2,
                "recommended_start_res": str(start_res),
                "multi_endpoint_merge_key": "model_symmetry",
                "datasets": [
                    {
                        "name": "a",
                        "hkl": str(hkl_a),
                        "hkl_transform": "identity",
                        "source_run": str(run_a),
                        "source_summary": str(run_a / "summary.json"),
                        "nuisance": {"thickness_A": 1000.0},
                    },
                    {
                        "name": "b",
                        "hkl": str(hkl_b),
                        "hkl_transform": "-h,k,l",
                        "source_run": str(run_b),
                        "source_summary": str(run_b / "summary.json"),
                        "nuisance": {"thickness_A": 1200.0},
                    },
                ],
                "structure_dataset_policy": {"status": "disabled"},
            }
        )
        + "\n"
    )
    args = build_parser().parse_args(
        [
            "push-endpoint",
            "--checkpoint",
            str(checkpoint),
            "--start-res",
            str(start_res),
            "--out-dir",
            str(out_dir),
            "--cycles",
            "10",
            "--dry-run",
        ]
    )

    assert args.func(args) == 0
    plan = json.loads((out_dir / "push_endpoint_plan.json").read_text())

    assert plan["mode"] == "multi_dataset"
    assert plan["cycles"] == 10
    assert plan["multi_endpoint_merge_key"] == "model_symmetry"
    assert [row["name"] for row in plan["datasets"]] == ["a", "b"]
    assert "GROND-side corrected-HKL merge" in plan["shared_loop"]
    assert plan["compact_scale_reselection_performed"] is False


def test_multi_push_report_states_compact_reselection_is_separate(tmp_path: Path):
    summary_dir = tmp_path / "runs" / "shared_shelxl_feedback"
    summary_dir.mkdir(parents=True)
    (summary_dir / "multi_shelxl_feedback_loop_summary.json").write_text(
        json.dumps({"status": "ok", "cycles": []}) + "\n"
    )

    report = cli.write_multi_push_endpoint_report(tmp_path, best_cycle_tolerance=0.001)

    assert report == tmp_path / "MULTI_ENDPOINT_PUSH_REPORT.md"
    decision = json.loads((tmp_path / "multi_endpoint_push_decision.json").read_text())
    assert decision["compact_scale_reselection_performed"] is False
    assert "not performed inside" in report.read_text()


def test_write_push_endpoint_report_handles_missing_output_model(tmp_path):
    out_dir = tmp_path / "push"
    out_dir.mkdir()
    (out_dir / "shelxl_feedback_loop_summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "label": "cycle_01",
                        "output_model": ".",
                        "endpoint_res_health": {
                            "ok": False,
                            "reason": "missing_or_invalid_endpoint_res",
                        },
                        "elapsed_seconds": 12.5,
                        "metrics": {
                            "native_dyn_wR2_all": 0.12,
                            "native_dyn_free_wR2_all": 0.14,
                        },
                    }
                ]
            }
        )
        + "\n"
    )

    report = cli.write_push_endpoint_report(out_dir, best_cycle_tolerance=0.001)

    assert report == out_dir / "ENDPOINT_PUSH_REPORT.md"
    text = report.read_text()
    assert "missing_or_invalid_endpoint_res" in text
    decision = json.loads((out_dir / "endpoint_push_decision.json").read_text())
    assert decision["rows"][0]["model"] == ""
    assert decision["rows"][0]["status"] == "missing_or_invalid_endpoint_res"


def test_write_push_endpoint_report_rejects_counted_npd_without_atom_label(tmp_path):
    out_dir = tmp_path / "push"
    out_dir.mkdir()
    model = out_dir / "cycle.res"
    model.write_text("TITL test\n")
    model.with_suffix(".lst").write_text(
        "0 atoms may be split and 2 atoms NPD\n"
    )
    (out_dir / "shelxl_feedback_loop_summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "label": "cycle_01",
                        "output_model": str(model),
                        "endpoint_res_health": {
                            "ok": True,
                            "npd_count": 2,
                            "has_npd": True,
                            "safe_for_feedback": False,
                        },
                        "elapsed_seconds": 1.0,
                        "metrics": {
                            "native_dyn_wR2_all": 0.10,
                            "native_dyn_free_wR2_all": 0.11,
                        },
                    }
                ]
            }
        )
        + "\n"
    )

    cli.write_push_endpoint_report(out_dir, best_cycle_tolerance=0.001)

    decision = json.loads((out_dir / "endpoint_push_decision.json").read_text())
    assert decision["selected_safe_cycle"] is None
    assert decision["rows"][0]["npd_count"] == 2
    assert decision["rows"][0]["npd_atoms"] == []
    assert decision["rows"][0]["safe_for_feedback"] is False


def test_multi_checkpoint_push_overrides_dataset_resolution_cutoff():
    checkpoint = {
        "datasets": [
            {"name": "a", "hkl": "/tmp/a.hkl", "min_d_spacing": 0.70, "nuisance": {"thickness_A": 1200}},
            {"name": "b", "hkl": "/tmp/b.hkl", "min_d_spacing": 0.70, "nuisance": {"thickness_A": 1300}},
        ]
    }

    prep, _nuisance = cli.prep_from_multi_checkpoint(checkpoint, "/tmp/model.res", min_d_spacing=0.75)

    assert [dataset.min_d_spacing for dataset in prep.datasets] == [0.75, 0.75]


def test_multi_push_row_contracts_are_complete_and_forwarded(tmp_path: Path):
    datasets = [
        DatasetSpec(name="one", hkl=str(tmp_path / "one.hkl")),
        DatasetSpec(name="two", hkl=str(tmp_path / "two.hkl")),
    ]
    one = tmp_path / "one_policy.csv"
    two = tmp_path / "two_policy.csv"
    contracts = cli.resolve_dataset_row_contracts([f"one={one}", f"two={two}"], datasets)
    assert contracts == {"one": str(one.resolve()), "two": str(two.resolve())}
    with pytest.raises(SystemExit, match="missing: two"):
        cli.resolve_dataset_row_contracts([f"one={one}"], datasets)

    args = build_parser().parse_args(
        ["run", "--out-dir", "out", "--hkl", "rows.hkl", "--structure", "model.res"]
    )
    args.dataset_row_contracts = contracts
    command = cli.native_stage_cmd(
        "python",
        datasets[0],
        "model.res",
        "baseline.hkl",
        tmp_path / "run",
        args,
        n_basis=20,
        n_points=5,
        stage="multi_ls1_cycle_01",
        endpoint=False,
    )
    assert command[command.index("--row-contract") + 1] == str(one.resolve())
    assert command[command.index("--row-plan-cache") + 1] == "auto"

    structure_command = cli.structure_cmd(
        "python",
        datasets[0],
        "model.res",
        "baseline.hkl",
        tmp_path / "structure",
        args,
        {"thickness_A": 1200.0},
    )
    assert structure_command[structure_command.index("--row-plan-cache") + 1] == "off"

    for path, name, role, raw_weight, endpoint_weight in (
        (one, "one", "downweight", "0.25", "0.30"),
        (two, "two", "primary", "1.0", "1.0"),
    ):
        path.write_text(
            "dataset,dataset_role,raw_row_weight,endpoint_weight\n"
            f"{name},{role},{raw_weight},{endpoint_weight}\n"
            f"{name},{role},{raw_weight},{endpoint_weight}\n"
        )
    policy = cli.dataset_policy_from_row_contracts(contracts, datasets)
    assert policy["status"] == "immutable_row_contract"
    assert [(row["action"], row["raw_row_weight"], row["endpoint_weight"]) for row in policy["datasets"]] == [
        ("downweight", 0.25, 0.30),
        ("primary", 1.0, 1.0),
    ]


def test_multi_push_scale_specs_are_complete():
    datasets = [DatasetSpec(name="one", hkl="one.hkl"), DatasetSpec(name="two", hkl="two.hkl")]
    specs = cli.resolve_dataset_scale_specs(
        ["one=adaptive:28:det8", "two=stride:6:det8"],
        datasets,
    )
    assert specs == {"one": "adaptive:28:det8", "two": "stride:6:det8"}
    with pytest.raises(SystemExit, match="missing: two"):
        cli.resolve_dataset_scale_specs(["one=adaptive:28:det8"], datasets)
    with pytest.raises(SystemExit, match="empty compact scale specification"):
        cli.resolve_dataset_scale_specs(["one=", "two=stride:6:det8"], datasets)
