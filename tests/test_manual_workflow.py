from __future__ import annotations

import json
import sys
from pathlib import Path

from grond import cli
from grond.manual_workflow import _hand_evidence, _manual_checkpoint, _run_stage_command


def _run(*arguments: str) -> int:
    args = cli.build_parser().parse_args(["manual", *arguments])
    return cli.cmd_manual(args)


def _write_source_summary(path: Path, model: Path, hkl: Path, transform: str, thickness: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_table = path.parent / "native_dyn_calc.csv"
    row_table.write_text("row_index,h,k,l,target_I,xds_sigma,d_A\n0,1,0,0,10,2,5\n")
    path.write_text(
        json.dumps(
            {
                "inputs": {"res": str(model), "hkl": str(hkl), "hkl_transform": transform},
                "config": {
                    "hkl_transform": transform,
                    "n_basis": 48,
                    "n_points": 10,
                    "topup_d_min": 0.7,
                    "compute_backend": "scipy",
                    "row_plan_cache": "off",
                    "chunksize": 2,
                    "post_bloch_sigma_policy": "uniform_z_rms",
                    "min_d_spacing": 0.7,
                    "min_peak": 90.0,
                    "geometry_percentile": 99.0,
                },
                "best_native": {
                    "nuisance": {
                        "thickness_A": thickness,
                        "thickness_grad_frac": 0.4,
                        "frame_offset": 0.0,
                        "group_width_deg": 0.5,
                        "detector_scale_bins": 0,
                        "sigma_a": 0.0,
                        "sigma_b_corr": 0.0,
                        "sigma_c_peak": 0.0,
                    }
                },
                "artifacts": {"native_row_table": str(row_table)},
            }
        )
        + "\n"
    )


def test_manual_child_progress_is_teed_live_and_logged(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setenv("GROND_MANUAL_HEARTBEAT_SECONDS", "0.01")
    record = _run_stage_command(
        [
            sys.executable,
            "-c",
            "import time; print('STEP live progress', flush=True); time.sleep(0.04)",
            "--out-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr().out
    assert "STEP live progress" in captured
    assert "WAIT" in captured
    assert "DONE" in captured
    log = Path(str(record["log"]))
    assert "STEP live progress" in log.read_text()
    assert record["returncode"] == 0

    _run_stage_command(
        [
            sys.executable,
            "-c",
            "print('STEP resumed progress', flush=True)",
            "--out-dir",
            str(tmp_path),
        ]
    )
    resumed_text = log.read_text()
    assert "STEP live progress" in resumed_text
    assert "[manual_resume]" in resumed_text
    assert "STEP resumed progress" in resumed_text


def test_manual_state_handoffs_and_multi_checkpoint(tmp_path: Path):
    model = tmp_path / "model.res"
    model.write_text("TITL model\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nSFAC C\nUNIT 1\nC1 1 0 0 0 1 0.05\nHKLF 4\nEND\n")
    baseline = tmp_path / "model.hkl"
    baseline.write_text("   0   0   0    0.00    0.00\n")
    datasets = []
    summaries = []
    for index, transform in enumerate(("identity", "-h,k,-l"), start=1):
        directory = tmp_path / f"ds{index}"
        directory.mkdir()
        hkl = directory / "INTEGRATE.HKL"
        hkl.write_text("!UNIT_CELL_CONSTANTS= 5 6 7 90 90 90\n!END_OF_HEADER\n!END_OF_DATA\n")
        datasets.append((directory, transform, hkl))
        summary = tmp_path / f"source{index}" / "summary.json"
        _write_source_summary(summary, model, hkl, transform, 900.0 + 100.0 * index)
        summaries.append(summary)

    root = tmp_path / "manual"
    assert (
        _run(
            "setup",
            "--out-dir",
            str(root),
            "--structure",
            str(model),
            "--baseline-hkl",
            str(baseline),
            "--dataset",
            f"one={datasets[0][0]},hkl_transform={datasets[0][1]}",
            "--dataset",
            f"two={datasets[1][0]},hkl_transform={datasets[1][1]}",
        )
        == 0
    )
    workflow = root / "manual_workflow.json"
    assert _run("scout", "--workflow", str(workflow), "--skip") == 0
    assert (
        _run(
            "thickness",
            "--workflow",
            str(workflow),
            "--skip",
            "--summary",
            f"one={summaries[0]}",
            "--summary",
            f"two={summaries[1]}",
        )
        == 0
    )
    assert _run("hand", "--workflow", str(workflow), "--skip") == 0

    state = json.loads(workflow.read_text())
    assert state["current_stage"] == "hand"
    assert state["chosen_model"] == str(model.resolve())
    assert "manual feedback" in state["next_command"]
    for stage in ("00_setup", "01_scout", "02_thickness", "03_hand"):
        assert (root / stage / "handoff.json").exists()

    checkpoint_path = root / "checkpoint.json"
    checkpoint = _manual_checkpoint(state, checkpoint_path)
    assert checkpoint["status"] == "ready"
    assert checkpoint["dataset_count"] == 2
    assert [row["hkl_transform"] for row in checkpoint["datasets"]] == ["identity", "-h,k,-l"]
    assert [row["nuisance"]["thickness_A"] for row in checkpoint["datasets"]] == [1000.0, 1100.0]

    # Recording a choice after a report reuses the stored evidence; it must not
    # launch the expensive hand calculation a second time.
    hand_dir = root / "03_hand"
    diagnostic_paths = {}
    handoff_paths = {}
    for name in ("one", "two"):
        dataset_dir = hand_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        diagnostic = dataset_dir / "absolute_structure_diagnostic.json"
        diagnostic.write_text(
            json.dumps(
                {
                    "all": {"n": 10, "delta_chi2_other_minus_best": 2.0},
                    "free": {"n": 2, "delta_chi2_other_minus_best": 0.5},
                }
            )
            + "\n"
        )
        dataset_handoff = dataset_dir / "hand_handoff.json"
        dataset_handoff.write_text(json.dumps({"inverse_model": str(model)}) + "\n")
        diagnostic_paths[name] = str(diagnostic)
        handoff_paths[name] = str(dataset_handoff)
    state = json.loads(workflow.read_text())
    state["stages"]["hand"] = {
        "status": "needs_choice",
        "outputs": {
            "dataset_diagnostics": diagnostic_paths,
            "dataset_handoffs": handoff_paths,
        },
    }
    workflow.write_text(json.dumps(state) + "\n")
    assert _run("hand", "--workflow", str(workflow), "--choice", "input") == 0
    final_state = json.loads(workflow.read_text())
    assert final_state["stages"]["hand"]["command"]["datasets"]["status"] == (
        "reused_existing_hand_report_without_recalculation"
    )


def test_manual_hand_evidence_is_explicit_and_sign_consistent():
    input_favored = {
        "one": {
            "all": {"n": 100, "delta_chi2_other_minus_best": 12.0},
            "free": {"n": 10, "delta_chi2_other_minus_best": 2.0},
        },
        "two": {
            "all": {"n": 80, "delta_chi2_other_minus_best": 4.0},
            "free": {"n": 8, "delta_chi2_other_minus_best": 1.0},
        },
    }
    evidence = _hand_evidence(input_favored)
    assert evidence["recommendation"] == "input"
    assert evidence["combined_all_delta_chi2_inverse_minus_input"] == 16.0
    assert evidence["combined_free_rows"] == 18

    mixed = dict(input_favored)
    mixed["two"] = {
        "all": {"n": 80, "delta_chi2_other_minus_best": -20.0},
        "free": {"n": 8, "delta_chi2_other_minus_best": -1.0},
    }
    assert _hand_evidence(mixed)["recommendation"] == "ambiguous"


def test_manual_finish_writes_openable_handoff_without_maps(tmp_path: Path):
    model = tmp_path / "endpoint.res"
    model.write_text("TITL endpoint\nCELL 0.025079 5 6 7 90 90 90\nEND\n")
    model.with_suffix(".hkl").write_text("   0   0   0    0.00    0.00\n")
    model.with_suffix(".lst").write_text(
        "wR2 = 0.1065, GooF = S = 0.775 for all data\n"
        "R1 = 0.0578 for 10 Fo > 4sig(Fo) and 0.1116 for all 20 data\n"
        "0 atoms may be split and 0 atoms NPD\n"
    )
    run = tmp_path / "row_run"
    run.mkdir()
    (run / "native_dyn_calc.csv").write_text("h,k,l,target_I,sigma_eff,dyn_scaled\n1,0,0,10,2,9\n")
    (run / "summary.json").write_text(json.dumps({"best_native": {"metrics": {}, "nuisance": {}}}) + "\n")
    root = tmp_path / "manual"
    root.mkdir()
    workflow = root / "manual_workflow.json"
    workflow.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "grond_manual_workflow",
                "root": str(root),
                "prep": "",
                "chosen_model": str(model),
                "stages": {
                    "feedback": {
                        "status": "ready",
                        "outputs": {"selected_model": str(model), "selected_row_runs": {"one": str(run)}},
                    }
                },
            }
        )
        + "\n"
    )
    assert _run("finish", "--workflow", str(workflow), "--skip") == 0
    payload = json.loads((root / "06_finish" / "manual_final_summary.json").read_text())
    assert payload["endpoint"]["R1_all"] == 0.1116
    assert payload["endpoint"]["GooF"] == 0.775
    assert payload["endpoint"]["npd_count"] == 0
    assert (root / "06_finish" / "handoff.json").exists()

    # Going back to an earlier stage keeps its old files but invalidates later
    # entries in the active workflow state.
    assert _run("scales", "--workflow", str(workflow), "--skip") == 0
    state = json.loads(workflow.read_text())
    assert state["current_stage"] == "scales"
    assert "finish" not in state["stages"]
    assert state["stages"]["scales"]["invalidated_downstream_stages"] == ["finish"]
