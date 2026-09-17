"""An independent solution needs new indexing, with the requested physics retained."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from grond.cli_options.parser import build_parser
from grond.commands.preparation import prep_from_args
from grond.io.config import DatasetSpec, PrepConfig, write_prep
from grond.workflows.auto_crystal_parameters import configure_auto_crystals
from grond.workflows.crystal_parameters import crystal_arguments
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nuisance_policy import compact_base_nuisance, resolve_final_n_basis
from grond.workflows.thickness_materialization import continuous_thickness_enabled
from tests.cli._support import combined_endpoint_summary, valid_shelx_res_text
from tests.cli.test_crystal_parameters import value
from tests.physics.test_continuous_thickness import _write_single_row_inputs


@pytest.fixture
def independent_run(tmp_path):
    hkl, model = _write_single_row_inputs(tmp_path)
    checkpoint = tmp_path / "reference.json"
    datasets = [DatasetSpec(name=f"crystal_{i}", hkl=str(hkl)) for i in range(1, 5)]
    rows = [
        {
            "name": ds.name, "hkl": str(hkl), "hkl_transform": "-h,-k,l",
            "nuisance": {
                "thickness_A": 100 * i, "thickness_model": "uniform_gradient",
                "thickness_grad_frac": 0.1 * i, "thickness_sigma_frac": 0.0,
                "frame_offset": 99, "sg_jitter_mode": "gaussian", "sg_jitter_sigma0": 0.123,
            },
            "final_config": {"n_basis": i + 2, "n_points": i + 2, "thickness_points": 3,
                             "sg_window": 0.0001, "sg_window_mode": "fixed", "topup_d_min": 0.7},
        }
        for i, ds in enumerate(datasets, 1)
    ]
    checkpoint.write_text(json.dumps({
        "kind": "grond_continue_checkpoint", "status": "ready", "datasets": rows,
        "recommended_start_res": "/must/not/read/full_H.res",
    }))
    prep_path = tmp_path / "prep.json"
    write_prep(prep_path, PrepConfig(structure=str(model), baseline_hkl="old_setting.hkl", datasets=datasets))
    out = tmp_path / "out"
    out.mkdir()
    args = build_parser().parse_args([
        "auto", "--prep", str(prep_path), "--structure", str(model),
        "--baseline-hkl", str(tmp_path / "never_H.hkl"),
        "--crystal-parameters-from", str(checkpoint), "--out-dir", str(out),
        "--workers", "1", "--shelxl-feedback-cycles", "2",
    ])
    return prep_from_args(args), args, checkpoint


def test_new_model_and_baseline_replace_prep_pair(independent_run, tmp_path):
    _, args, _ = independent_run
    args.structure = str(tmp_path / "independent_solution.res")
    prep = prep_from_args(args)
    assert prep.structure == args.structure
    assert prep.baseline_hkl == args.baseline_hkl
    assert all(ds.baseline_hkl == args.baseline_hkl for ds in prep.datasets)
    assert json.loads(Path(args.prep).read_text())["baseline_hkl"] == "old_setting.hkl"


def test_reference_copies_only_requested_physics(independent_run):
    prep, args, checkpoint = independent_run
    before = checkpoint.read_bytes()
    configure_auto_crystals(prep, args)
    assert checkpoint.read_bytes() == before
    assert args.auto_decide and not args.trust_hkl_transform
    assert all(ds.hkl_transform == "identity" for ds in prep.datasets)
    for i, ds in enumerate(prep.datasets, 1):
        local = crystal_arguments(args, ds.name)
        assert local.n_basis == i + 2
        assert local.n_points == i + 2
        assert not continuous_thickness_enabled(local)
        assert local.fixed_thickness["thickness_A"] == 100 * i
        assert set(local.fixed_thickness) == {"thickness_A", "thickness_model", "thickness_grad_frac", "thickness_sigma_frac"}
        command = native_stage_cmd(sys.executable, ds, prep.structure, prep.baseline_hkl, Path(args.out_dir),
                                   args, n_basis=24, n_points=10, stage="screen", pinned=compact_base_nuisance(args))
        assert value(command, "--sg-jitter-mode-grid") == "none"
        assert "--frame-offset-grid=99" not in command


@pytest.mark.parametrize("stage", ["scout_transform_coarse", "screen", "multi_thickness_polish", "adaptive_sg_jitter_gate", "final", "hand"])
def test_every_stage_obeys_fixed_values(independent_run, stage):
    prep, args, _ = independent_run
    configure_auto_crystals(prep, args)
    attempted = {"thickness_A": "300,500,700", "thickness_grad_frac": "0.1,0.5,1", "sg_jitter_mode": "none,gaussian"}
    before = copy.deepcopy(attempted)
    for i, ds in enumerate(prep.datasets, 1):
        command = native_stage_cmd(sys.executable, ds, prep.structure, prep.baseline_hkl, Path(args.out_dir),
                                   args, n_basis=24, n_points=10, stage=stage, pinned=attempted)
        assert int(value(command, "--n-basis")) == i + 2
        assert int(value(command, "--n-points")) == i + 2
        assert float(value(command, "--thickness-grid")) == 100 * i
        assert float(value(command, "--thickness-grad-grid")) == pytest.approx(0.1 * i)
        assert value(command, "--sg-jitter-mode-grid").startswith("none,gaussian")
        assert resolve_final_n_basis(prep, ds, args, "python", Path(args.out_dir), Path(args.out_dir), prep.structure, stage) == i + 2
    assert attempted == before


def test_explicit_flags_without_checkpoint_do_not_pin_other_crystals(independent_run):
    prep, args, _ = independent_run
    args.crystal_parameters_from = ""
    args.dataset_n_basis = ["crystal_1=600"]
    args.dataset_thickness_nm = ["crystal_1=55.5"]
    configure_auto_crystals(prep, args)
    command = native_stage_cmd(sys.executable, prep.datasets[0], prep.structure, "", Path(args.out_dir), args,
                               n_basis=24, n_points=10, stage="screen")
    assert value(command, "--n-basis") == "600"  # No implicit 500-beam clamp.
    assert value(command, "--thickness-grid") == "555"
    assert value(command, "--thickness-grad-grid") == args.thickness_grad_grid
    assert crystal_arguments(args, "crystal_2") is args


@pytest.mark.parametrize("change, message", [
    (lambda c: c.update(status="not_ready"), "ready"),
    (lambda c: c["datasets"].pop(), "same unique"),
    (lambda c: c["datasets"][0]["nuisance"].pop("thickness_grad_frac"), "complete thickness"),
    (lambda c: c["datasets"][0]["nuisance"].update(thickness_sigma_frac=float("nan")), "invalid thickness"),
])
def test_incomplete_reference_fails_before_calculation(independent_run, change, message):
    prep, args, checkpoint = independent_run
    payload = json.loads(checkpoint.read_text())
    change(payload)
    checkpoint.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match=message):
        configure_auto_crystals(prep, args)


def test_different_integrations_cannot_borrow_parameters_by_name(independent_run, tmp_path):
    prep, args, _ = independent_run
    other = tmp_path / "other.hkl"
    other.write_text("different observations")
    prep.datasets[0] = replace(prep.datasets[0], hkl=str(other))
    with pytest.raises(SystemExit, match="integration differs"):
        configure_auto_crystals(prep, args)


def test_reference_allows_named_overrides(independent_run):
    prep, args, _ = independent_run
    args.dataset_n_basis = ["crystal_2=45"]
    args.dataset_thickness_nm = ["crystal_3=37.25"]
    configure_auto_crystals(prep, args)
    assert crystal_arguments(args, "crystal_2").n_basis == 45
    third = crystal_arguments(args, "crystal_3").fixed_thickness
    assert third["thickness_A"] == 372.5
    assert third["thickness_grad_frac"] == pytest.approx(0.3)


def test_four_crystal_auto_rescouts_and_runs_shared_feedback(independent_run, monkeypatch):
    """Run orchestration through final combination; replace expensive external jobs."""
    from grond.workflows import multi, shared_refinement

    prep, args, _ = independent_run
    configure_auto_crystals(prep, args)
    calls, scouted = [], []
    real_preflight = multi.preflight_dataset
    monkeypatch.setattr(multi, "preflight_dataset", lambda *a, **k: replace(real_preflight(*a, **k), status="ok"))

    def scout(current_prep, ds, *unused, **kwargs):
        scouted.append(ds.name)
        assert current_prep.structure == prep.structure
        assert ds.baseline_hkl == args.baseline_hkl
        return replace(ds, hkl_transform="k,-h,l"), {}

    def run(command, logs, label, **kwargs):
        calls.append((label, command))
        out = Path(value(command, "--out-dir"))
        out.mkdir(parents=True, exist_ok=True)
        if "combine" in label:
            model = out / "combined_endpoint_lam_1p0/multi_dyn_corrected.res"
            model.parent.mkdir(exist_ok=True)
            model.write_text(valid_shelx_res_text(label))
            (out / "combined_summary.json").write_text(json.dumps(combined_endpoint_summary(model)))
        else:
            nui = {
                "thickness_A": float(value(command, "--thickness-grid")),
                "thickness_grad_frac": float(value(command, "--thickness-grad-grid")),
                "thickness_model": value(command, "--thickness-model-grid"),
                "thickness_sigma_frac": 0, "frame_offset": 0.2, "sg_jitter_mode": "none",
            }
            (out / "summary.json").write_text(json.dumps({"best_native": {"nuisance": nui}}))

    monkeypatch.setattr(multi, "choose_transform_by_scout", scout)
    monkeypatch.setattr(multi, "run_logged", run)
    monkeypatch.setattr(shared_refinement, "run_logged", run)
    # These independent policies have their own tests; retain the actual native
    # commands, shared update loop and final recomputation in this test.
    for name in ("run_scale_window_polish", "run_resolution_limit_gate", "run_forensic_gate", "run_dataset_policy_gate", "run_multi_final_elastic_hand_selection", "run_sg_window_scout", "discrete_thickness_promotion_record"):
        monkeypatch.setattr(multi, name, lambda *a, **k: {"status": "disabled"})
    monkeypatch.setattr(multi, "run_feature_gate", lambda *a, **k: (k["active_nui"], k["active_run"], {"status": "disabled"}))
    monkeypatch.setattr(multi, "run_multi_scale_contract_selection", lambda **k: (k["full_runs"], k["full_runs"], {"status": "checked"}))
    monkeypatch.setattr(multi, "continuous_thickness_cmd", lambda *a, **k: pytest.fail("fixed thickness was reopened"))
    result = multi.run_multi(prep, args, sys.executable, Path(args.out_dir) / "logs")
    assert set(scouted) == {ds.name for ds in prep.datasets}
    assert len(result["structure_feedback"]["cycles"]) == 2
    assert result["scale_contract_selection"]["status"] == "checked"
    native = [(label, command) for label, command in calls if "combine" not in label]
    assert len(native) == 24  # screen, local polish, two LS1 updates, final replay, final endpoint.
    for label, command in native:
        i = int(label.split("_")[1])
        assert int(value(command, "--n-basis")) == i + 2
        assert float(value(command, "--thickness-grid")) == 100 * i
        assert "--hkl-transform=k,-h,l" in command
        assert "full_H.res" not in str(command)
    assert calls[-1][0] == "combine_final"


def test_actual_bloch_predictions_match_explicit_parameters(independent_run):
    from grond.legacy.engine.bloch_engine import Config, XDSBlochPort, XDSData, parse_model, sha256

    prep, args, _ = independent_run
    configure_auto_crystals(prep, args)
    for i, ds in enumerate(prep.datasets, 1):
        command = native_stage_cmd(sys.executable, ds, prep.structure, "", Path(args.out_dir), args,
                                   n_basis=24, n_points=10, stage="final", pinned=compact_base_nuisance(args), endpoint=False)
        fixed = dict(sg_window=0.0001, sg_window_min=0.0001, sg_window_max=0.0001,
                     topup_d_min=0.7, ug_d_min=0.7, thickness_points=3, thickness_model="uniform_gradient")
        resolved = Config(**fixed, n_basis=int(value(command, "--n-basis")), n_points=int(value(command, "--n-points")),
                          thickness_A=float(value(command, "--thickness-grid")), thickness_grad_frac=float(value(command, "--thickness-grad-grid")))
        explicit = Config(**fixed, n_basis=i + 2, n_points=i + 2, thickness_A=100 * i, thickness_grad_frac=0.1 * i)
        model = Path(prep.structure)
        data, atoms = XDSData(Path(ds.hkl)), parse_model(model)
        actual = XDSBlochPort(data, atoms, resolved, res_hash=sha256(model)).row_prediction(0)
        expected = XDSBlochPort(data, atoms, explicit, res_hash=sha256(model)).row_prediction(0)
        assert actual["dyn"] == pytest.approx(expected["dyn"], rel=1e-12, abs=1e-14)
        assert actual["born"] == pytest.approx(expected["born"], rel=1e-12, abs=1e-14)
        assert actual["support_hkls"] == expected["support_hkls"]


def test_transform_scout_reports_the_parameters_it_actually_uses(independent_run):
    from grond.workflows.transform_selection import transform_scout_records

    prep, args, _ = independent_run
    args.dry_run = True
    configure_auto_crystals(prep, args)
    records = transform_scout_records(prep, prep.datasets[0], args, sys.executable, Path(args.out_dir) / "logs",
                                     Path(args.out_dir) / "scout", ["identity", "k,-h,l"],
                                     scout_stage="coarse", n_basis=24, n_points=10, max_rows=100)
    assert len(records) == 2
    assert all(r["n_basis"] == 3 and r["n_points"] == 3 for r in records)
    assert all(r["frozen_nuisance_contract"]["thickness_A"] == 100 for r in records)


@pytest.mark.parametrize("reference", [True, False])
def test_readme_fresh_commands_resolve_correctly(independent_run, reference):
    import re
    import shlex

    prep, original, checkpoint = independent_run
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    marker = "--crystal-parameters-from" if reference else "--dataset-n-basis crystal_4=235"
    example = next(block for block in re.findall(r"```bash\n(.*?)```", readme, re.S)
                   if "GROND auto" in block and marker in block)
    for old, new in {
        "/path/to/full_H/run_prep.json": original.prep,
        "/path/to/crystals.json": original.prep,
        "/path/to/never_H.res": prep.structure,
        "/path/to/never_H.hkl": original.baseline_hkl,
        "/path/to/full_H/grond_continue_checkpoint.json": str(checkpoint),
        "/path/to/never_H_GROND": original.out_dir,
    }.items():
        example = example.replace(old, new)
    args = build_parser().parse_args(shlex.split(example.replace(chr(92) + chr(10), ""))[1:])
    configure_auto_crystals(prep_from_args(args), args)
    assert [crystal_arguments(args, ds.name).n_basis for ds in prep.datasets] == ([3, 4, 5, 6] if reference else [180, 210, 160, 235])


@pytest.mark.parametrize("option, val", [("fast", True), ("full_fast", True), ("native_only", True), ("structure_update", "native_gradient")])
def test_unsupported_execution_modes_fail_clearly(independent_run, option, val):
    prep, args, _ = independent_run
    setattr(args, option, val)
    with pytest.raises(SystemExit, match="ordinary SHELXL feedback"):
        configure_auto_crystals(prep, args)


def test_single_crystal_uses_the_same_fresh_interface(independent_run, monkeypatch):
    from grond.workflows import single

    prep, args, _ = independent_run
    args.crystal_parameters_from = ""
    args.dataset_n_basis = ["crystal_1=31"]
    args.dataset_thickness_nm = ["crystal_1=55"]
    prep.datasets = prep.datasets[:1]
    configure_auto_crystals(prep, args)
    captured = {}

    def adaptive(p, options, python, logs):
        captured.update(vars(options))
        return {}

    monkeypatch.setattr(single, "run_single_adaptive", adaptive)
    single.run_single(prep, args, sys.executable, Path(args.out_dir) / "logs")
    assert captured["n_basis"] == 31
    assert captured["fixed_thickness"] == {"thickness_A": 550}
    assert captured["continuous_thickness_polish"] == "off"
