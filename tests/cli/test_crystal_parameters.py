"""Per-crystal physics must survive every shared update and the final replay."""

import copy
import json
import sys
from pathlib import Path

import pytest

from grond.cli_options.parser import build_parser
from grond.io.config import DatasetSpec
from grond.workflows import shared_refinement
from grond.workflows.checkpoints import _checkpoint_dataset_contract
from grond.workflows.crystal_parameters import crystal_arguments, resolve_crystal_parameters
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.push_contracts import hydrate_push_args_for_multi, prep_from_multi_checkpoint
from tests.cli._support import combined_endpoint_summary, valid_shelx_res_text


@pytest.fixture
def four_crystals(tmp_path):
    rows = []
    for index in range(4):
        rows.append(
            {
                "name": f"crystal_{index + 1}",
                "hkl": str(tmp_path / f"{index}.hkl"),
                "nuisance": {
                    "thickness_A": 700.0 + 100 * index,
                    "thickness_model": "uniform_gradient",
                    "thickness_grad_frac": 0.2,
                    "frame_offset": 0.1 * index,
                },
                "final_config": {
                    "n_basis": 12 + 3 * index,
                    "n_points": 3 + index,
                    "thickness_points": 5 + index,
                    "sg_window": 0.001 + 0.0001 * index,
                    "max_support_sg": 0.01,
                    "topup_d_min": 0.6 - 0.1 * index,
                },
            }
        )
    return {
        "kind": "grond_continue_checkpoint",
        "status": "ready",
        "dataset_count": 4,
        "datasets": rows,
        "structure_dataset_policy": {"status": "disabled"},
    }


def resolve(checkpoint, tmp_path, *options):
    args = build_parser().parse_args(
        ["refine", "--checkpoint", str(tmp_path / "checkpoint.json"), "--out-dir", str(tmp_path / "out"), *options]
    )
    prep, nuisance = prep_from_multi_checkpoint(checkpoint, tmp_path / "start.res")
    args.crystal_parameters, nuisance = resolve_crystal_parameters(
        checkpoint,
        prep.datasets,
        nuisance,
        args,
        checkpoint_dir=tmp_path,
    )
    return prep, nuisance, hydrate_push_args_for_multi(args)


def value(command, option):
    return command[command.index(option) + 1]


def test_inherit_four_distinct_crystals_and_override_one(four_crystals, tmp_path):
    before = copy.deepcopy(four_crystals)
    prep, nuisance, args = resolve(
        four_crystals, tmp_path, "--dataset-n-basis", "crystal_2=180", "--dataset-thickness-nm", "crystal_3=125.25"
    )
    assert four_crystals == before
    assert [crystal_arguments(args, ds.name).n_basis for ds in prep.datasets] == [12, 180, 18, 21]
    assert [nuisance[ds.name]["thickness_A"] for ds in prep.datasets] == [700, 800, 1252.5, 1000]
    assert all(nuisance[ds.name]["thickness_grad_frac"] == 0.2 for ds in prep.datasets)
    assert args.crystal_parameters["crystal_3"]["thickness_nm"] == 125.25
    assert args.crystal_parameters["crystal_2"]["n_basis_source"] == "dataset_override"
    for i, ds in enumerate(prep.datasets):
        local = crystal_arguments(args, ds.name)
        cmd = native_stage_cmd(
            sys.executable,
            ds,
            prep.structure,
            "",
            tmp_path / ds.name,
            local,
            n_basis=local.n_basis,
            n_points=local.n_points,
            stage="multi_ls1_cycle_01",
            pinned=nuisance[ds.name],
            endpoint=False,
            strict_pinned_contract=True,
        )
        assert int(value(cmd, "--n-basis")) == [12, 180, 18, 21][i]
        assert int(value(cmd, "--n-points")) == 3 + i
        assert int(value(cmd, "--thickness-points")) == 5 + i
        assert float(value(cmd, "--thickness-grid")) == nuisance[ds.name]["thickness_A"]
        assert float(value(cmd, "--topup-d-min")) == pytest.approx(0.6 - 0.1 * i)


def test_explicit_global_overrides_and_dataset_precedence(four_crystals, tmp_path):
    prep, _, args = resolve(
        four_crystals,
        tmp_path,
        "--n-basis",
        "60",
        "--n-points",
        "30",
        "--dataset-n-basis",
        "crystal_4=80",
        "--topup-d-min",
        "0.5",
    )
    assert [crystal_arguments(args, ds.name).n_basis for ds in prep.datasets] == [60, 60, 60, 80]
    assert all(crystal_arguments(args, ds.name).n_points == 30 for ds in prep.datasets)
    assert all(crystal_arguments(args, ds.name).topup_d_min == 0.5 for ds in prep.datasets)


@pytest.mark.parametrize(
    "options, message",
    [
        (("--dataset-n-basis", "wrong=20"), "Available datasets"),
        (("--dataset-n-basis", "crystal_1=20", "--dataset-n-basis", "crystal_1=30"), "more than once"),
        (("--dataset-n-basis", "crystal_1=1"), "at least 2"),
        (("--dataset-n-basis", "crystal_1=20.5"), "at least 2"),
        (("--dataset-n-basis", "crystal_1=nan"), "at least 2"),
        (("--dataset-thickness-nm", "crystal_1=0"), "finite positive"),
        (("--dataset-thickness-nm", "crystal_1=-3"), "finite positive"),
        (("--dataset-thickness-nm", "crystal_1=inf"), "finite positive"),
        (("--dataset-thickness-nm", "crystal_1=nan"), "finite positive"),
        (("--dataset-thickness-nm", "crystal_1=10", "--dataset-thickness-nm", "crystal_1=10"), "more than once"),
        (("--dataset-thickness-nm", "crystal_1"), "expected DATASET"),
        (("--n-basis", "0"), "invalid n_basis"),
    ],
)
def test_invalid_overrides_fail_with_actionable_errors(four_crystals, tmp_path, options, message):
    with pytest.raises(SystemExit, match=message):
        resolve(four_crystals, tmp_path, *options)


def test_old_checkpoint_recovers_sampling_from_its_relative_summary(four_crystals, tmp_path):
    first = four_crystals["datasets"][0]
    expected = first.pop("final_config")
    first["source_summary"] = "old/summary.json"
    (tmp_path / "old").mkdir()
    (tmp_path / "old/summary.json").write_text(json.dumps({"config": expected}))
    prep, _, args = resolve(four_crystals, tmp_path)
    assert crystal_arguments(args, prep.datasets[0].name).thickness_points == 5
    assert crystal_arguments(args, prep.datasets[0].name).n_basis == 12


def test_missing_count_requires_an_explicit_value(four_crystals, tmp_path):
    del four_crystals["datasets"][0]["final_config"]["n_basis"]
    with pytest.raises(SystemExit, match="--dataset-n-basis crystal_1=N"):
        resolve(four_crystals, tmp_path)
    _, _, args = resolve(four_crystals, tmp_path, "--dataset-n-basis", "crystal_1=40")
    assert args.crystal_parameters["crystal_1"]["prediction"]["n_basis"] == 40


def test_checkpoint_writer_keeps_sampling_fields(four_crystals, tmp_path):
    config = four_crystals["datasets"][2]["final_config"]
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"config": config}))
    _, stored = _checkpoint_dataset_contract(DatasetSpec(name="three", hkl="input.hkl"), summary)
    assert stored == config


def test_four_crystals_survive_shared_cycles_and_final_recompute(four_crystals, tmp_path, monkeypatch):
    start = tmp_path / "start.res"
    start.write_text(valid_shelx_res_text("no H"))
    prep, nuisance, args = resolve(
        four_crystals,
        tmp_path,
        "--cycles",
        "2",
        "--dataset-n-basis",
        "crystal_2=45",
        "--dataset-thickness-nm",
        "crystal_4=150",
    )
    calls = []

    def run(command, logs, label, **kwargs):
        directory = Path(value(command, "--out-dir"))
        directory.mkdir(parents=True, exist_ok=True)
        if label.endswith("_combine"):
            model = directory / "combined_endpoint_lam_1p0/multi_dyn_corrected.res"
            model.parent.mkdir()
            model.write_text(valid_shelx_res_text(label))
            (directory / "combined_summary.json").write_text(json.dumps(combined_endpoint_summary(model)))
        else:
            calls.append((label, command))

    monkeypatch.setattr(shared_refinement, "run_logged", run)
    final, result = shared_refinement.run_multi_shelxl_feedback_loop(
        prep,
        args,
        sys.executable,
        tmp_path / "logs",
        tmp_path / "out",
        start_model=str(start),
        nuisance=nuisance,
        structure_dataset_policy={"status": "disabled"},
    )
    assert result["status"] == "ok"
    assert len(calls) == 12  # Four crystals at each of two updates and the final model replay.
    assert len(result["cycles"]) == 2
    for label, command in calls:
        index = int(label.split("_")[1]) - 1
        assert int(value(command, "--n-basis")) == [12, 45, 18, 21][index]
        assert float(value(command, "--thickness-grid")) == [700, 800, 900, 1500][index]
        assert int(value(command, "--n-points")) == index + 3
        assert int(value(command, "--thickness-points")) == index + 5
        if label.endswith("final_row_recompute"):
            assert value(command, "--res") == final


def test_dry_run_records_effective_values_without_launching_bloch(four_crystals, tmp_path):
    start = tmp_path / "start.res"
    start.write_text(valid_shelx_res_text())
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({**four_crystals, "recommended_start_res": str(start)}))
    args = build_parser().parse_args(
        [
            "refine",
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(tmp_path / "plan"),
            "--dataset-thickness-nm",
            "crystal_1=123",
            "--dry-run",
        ]
    )
    assert args.func(args) == 0
    plan = json.loads((tmp_path / "plan/push_endpoint_plan.json").read_text())
    assert plan["crystal_parameters"]["crystal_1"]["thickness_A"] == 1230
    assert plan["crystal_parameters"]["crystal_4"]["prediction"]["n_basis"] == 21
    assert not (tmp_path / "plan/runs").exists()


def test_resolved_settings_reproduce_explicit_bloch_predictions(four_crystals, tmp_path):
    """Compare four resolved worker contracts with explicit independent calculations."""
    from grond.legacy.engine.bloch_engine import Config, XDSBlochPort, XDSData, parse_model, sha256
    from tests.physics.test_continuous_thickness import _write_single_row_inputs

    hkl, model = _write_single_row_inputs(tmp_path)
    for i, row in enumerate(four_crystals["datasets"]):
        row["hkl"] = str(hkl)
        row["final_config"].update(
            n_basis=2 + i, n_points=3, sg_window=0.0001, sg_window_min=0.0001, sg_window_max=0.0001, topup_d_min=0.7
        )
    prep, nuisance, args = resolve(four_crystals, tmp_path, "--dataset-thickness-nm", "crystal_3=12.5")
    observed = []
    for i, ds in enumerate(prep.datasets):
        local = crystal_arguments(args, ds.name)
        command = native_stage_cmd(
            sys.executable,
            ds,
            str(model),
            "",
            tmp_path / ds.name,
            local,
            n_basis=local.n_basis,
            n_points=local.n_points,
            stage="final_row_recompute",
            pinned=nuisance[ds.name],
            endpoint=False,
            strict_pinned_contract=True,
        )
        selected = Config(
            n_basis=int(value(command, "--n-basis")),
            n_points=int(value(command, "--n-points")),
            thickness_A=float(value(command, "--thickness-grid")),
            thickness_points=int(value(command, "--thickness-points")),
            thickness_grad_frac=float(value(command, "--thickness-grad-grid")),
            thickness_model=value(command, "--thickness-model-grid"),
            frame_offset=0.1 * i,
            sg_window=0.0001,
            sg_window_min=0.0001,
            sg_window_max=0.0001,
            topup_d_min=0.7,
            ug_d_min=0.7,
        )
        explicit = Config(
            n_basis=2 + i,
            n_points=3,
            thickness_A=[700, 800, 125, 1000][i],
            thickness_points=5 + i,
            thickness_model="uniform_gradient",
            thickness_grad_frac=0.2,
            frame_offset=0.1 * i,
            sg_window=0.0001,
            sg_window_min=0.0001,
            sg_window_max=0.0001,
            topup_d_min=0.7,
            ug_d_min=0.7,
        )
        data, atoms = XDSData(hkl), parse_model(model)
        resolved = XDSBlochPort(data, atoms, selected, res_hash=sha256(model)).row_prediction(0)
        expected = XDSBlochPort(data, atoms, explicit, res_hash=sha256(model)).row_prediction(0)
        assert resolved["dyn"] == pytest.approx(expected["dyn"], rel=1e-12, abs=1e-14)
        assert resolved["born"] == pytest.approx(expected["born"], rel=1e-12, abs=1e-14)
        assert resolved["support_hkls"] == expected["support_hkls"]
        observed.append(resolved["dyn"])
    assert len({round(x, 12) for x in observed}) == 4


def test_readme_four_crystal_command_runs_as_documented(four_crystals, tmp_path):
    import re
    import shlex

    start = tmp_path / "never_H.res"
    start.write_text(valid_shelx_res_text("never H"))
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({**four_crystals, "recommended_start_res": str(start)}))
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    example = next(
        block for block in re.findall(r"```bash\n(.*?)```", readme, re.S) if "--dataset-n-basis crystal_4=235" in block
    )
    example = example.replace("/path/to/full_H/grond_continue_checkpoint.json", str(checkpoint))
    example = example.replace("/path/to/never_H.res", str(start))
    example = example.replace("/path/to/H_free_refinement", str(tmp_path / "documented"))
    command = shlex.split(example.replace(chr(92) + chr(10), ""))[1:] + ["--dry-run"]
    args = build_parser().parse_args(command)
    assert args.func(args) == 0
    data = json.loads((tmp_path / "documented/crystal_parameters.json").read_text())
    assert [data[f"crystal_{i}"]["prediction"]["n_basis"] for i in range(1, 5)] == [180, 210, 160, 235]
    assert [data[f"crystal_{i}"]["thickness_A"] for i in range(1, 5)] == [1200, 1500, 900, 1000]
