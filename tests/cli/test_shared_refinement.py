"""CLI shared refinement regression tests."""

import argparse
import json
import sys
from pathlib import Path

import pytest

import grond.cli as cli
import grond.workflows.scale_materialization as owner_workflows_scale_materialization
import grond.workflows.shared_refinement as owner_workflows_shared_refinement
import grond.workflows.single as owner_workflows_single
from grond.cli import (
    build_parser,
)
from grond.io.config import DatasetSpec, PrepConfig
from tests.cli._support import combined_endpoint_summary, valid_shelx_res_text


def test_single_final_row_recompute_binds_accepted_model_without_endpoint(tmp_path, monkeypatch):
    model = tmp_path / "accepted.res"
    model.write_text("TITL accepted\n")
    hkl = tmp_path / "INTEGRATE.HKL"
    hkl.write_text("! rows\n")
    source_run = tmp_path / "source"
    source_run.mkdir()
    (source_run / "native_dyn_calc.csv").write_text("row_index\n0\n")
    (source_run / "summary.json").write_text(
        json.dumps(
            {
                "inputs": {"res_sha256": "stale-model-hash"},
                "config": {"n_basis": 24, "n_points": 10, "scale_fit_split": "work"},
                "best_native": {"nuisance": {"thickness_A": 1000.0}},
            }
        )
        + "\n"
    )
    prep = PrepConfig(
        structure=str(model),
        datasets=[DatasetSpec(name="dataset", hkl=str(hkl), xds_dir=str(tmp_path))],
    )
    args = build_parser().parse_args(
        ["auto", "--prep", str(tmp_path / "prep.json"), "--out-dir", str(tmp_path / "run"), "--no-maps"]
    )
    captured = {}

    def fake_run_logged(cmd, logs, label, *, dry_run=False):
        captured["cmd"] = list(cmd)
        assert "--skip-endpoint-refinement" in cmd
        run_dir = Path(cmd[cmd.index("--out-dir") + 1])
        run_dir.mkdir(parents=True)
        (run_dir / "native_dyn_calc.csv").write_text("row_index\n0\n")
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "inputs": {"res": str(model), "res_sha256": cli.sha256_file(model)},
                    "config": {"n_basis": 24, "n_points": 10, "scale_fit_split": "work"},
                    "best_native": {"nuisance": {"thickness_A": 1000.0}},
                }
            )
            + "\n"
        )

    monkeypatch.setattr(owner_workflows_single, "run_logged", fake_run_logged)
    run_dir, contract = cli.ensure_single_final_row_model_contract(
        prep,
        prep.datasets[0],
        args,
        sys.executable,
        tmp_path / "logs",
        tmp_path / "run",
        source_run=source_run,
        final_model=str(model),
        fallback_nuisance={"thickness_A": 1000.0},
    )

    assert contract["status"] == "recomputed"
    assert Path(contract["row_table"]).is_file()
    assert run_dir.name == "adaptive_final_row_recompute"
    assert captured["cmd"][captured["cmd"].index("--res") + 1] == str(model.resolve())


def test_multi_shelxl_feedback_stops_before_reusing_npd_model(tmp_path: Path, monkeypatch):
    start_model = tmp_path / "start.res"
    start_model.write_text(valid_shelx_res_text("clean"))
    combined = tmp_path / "runs" / "shared_shelxl_feedback" / "cycle_01" / "combined"
    pathological_model = combined / "combined_endpoint_lam_1p0" / "multi_dyn_corrected.res"
    pathological_model.parent.mkdir(parents=True)
    pathological_model.write_text(valid_shelx_res_text("npd"))
    (combined / "combined_summary.json").write_text(
        json.dumps(
            combined_endpoint_summary(
                pathological_model,
                status="adp_pathology",
                safe_for_feedback=False,
                npd_count=1,
                split_count=0,
                npd_atoms=["O001"],
                has_npd=True,
            )
        )
        + "\n"
    )
    labels = []
    monkeypatch.setattr(owner_workflows_shared_refinement, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(owner_workflows_shared_refinement, "native_stage_cmd", lambda *args, **kwargs: ["native"])
    monkeypatch.setattr(owner_workflows_shared_refinement, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(owner_workflows_scale_materialization, "run_logged", lambda cmd, logs, label, **kwargs: labels.append(label))
    monkeypatch.setattr(owner_workflows_shared_refinement, "run_logged", lambda cmd, logs, label, **kwargs: labels.append(label))
    prep = PrepConfig(structure=str(start_model), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    args = argparse.Namespace(
        shelxl_feedback_cycles=3,
        shelxl_feedback_shelxl_iterations=1,
        dry_run=False,
        n_points=10,
    )

    current_model, summary = cli.run_multi_shelxl_feedback_loop(
        prep,
        args,
        "python",
        tmp_path / "logs",
        tmp_path,
        start_model=str(start_model),
        nuisance={"a": {}},
        structure_dataset_policy={"status": "disabled"},
    )

    assert current_model == str(start_model)
    assert summary["current_model"] == str(start_model)
    assert summary["status"] == "stopped_adp_pathology"
    assert summary["pathological_model"] == str(pathological_model)
    assert "NPD atoms=1" in summary["stopped_reason"]
    assert len(summary["cycles"]) == 1
    assert summary["cycles"][0]["adopted_for_feedback"] is False
    assert summary["cycles"][0]["endpoint_res_health"]["safe_for_feedback"] is False
    assert labels == ["a_multi_ls1_cycle_01", "multi_ls1_cycle_01_combine"]
    saved = json.loads(
        (tmp_path / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json").read_text()
    )
    assert saved["status"] == "stopped_adp_pathology"
    assert saved["current_model"] == str(start_model)


def test_multi_shelxl_feedback_allows_split_warning_without_npd(tmp_path: Path, monkeypatch):
    start_model = tmp_path / "start.res"
    start_model.write_text(valid_shelx_res_text("clean"))
    combined = tmp_path / "runs" / "shared_shelxl_feedback" / "cycle_01" / "combined"
    output_model = combined / "combined_endpoint_lam_1p0" / "multi_dyn_corrected.res"
    output_model.parent.mkdir(parents=True)
    output_model.write_text(valid_shelx_res_text("split"))
    (combined / "combined_summary.json").write_text(
        json.dumps(
            combined_endpoint_summary(
                output_model,
                split_count=2,
                has_split_warnings=True,
            )
        )
        + "\n"
    )
    monkeypatch.setattr(owner_workflows_shared_refinement, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(owner_workflows_shared_refinement, "native_stage_cmd", lambda *args, **kwargs: ["native"])
    monkeypatch.setattr(owner_workflows_shared_refinement, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(owner_workflows_scale_materialization, "run_logged", lambda *args, **kwargs: None)
    monkeypatch.setattr(owner_workflows_shared_refinement, "run_logged", lambda *args, **kwargs: None)
    prep = PrepConfig(structure=str(start_model), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    args = argparse.Namespace(
        shelxl_feedback_cycles=1,
        shelxl_feedback_shelxl_iterations=1,
        dry_run=False,
        n_points=10,
    )

    current_model, summary = cli.run_multi_shelxl_feedback_loop(
        prep,
        args,
        "python",
        tmp_path / "logs",
        tmp_path,
        start_model=str(start_model),
        nuisance={"a": {}},
        structure_dataset_policy={"status": "disabled"},
    )

    assert current_model == str(output_model)
    assert summary["current_model"] == str(output_model)
    assert summary["status"] == "ok"
    assert summary["cycles"][0]["adopted_for_feedback"] is True
    assert summary["cycles"][0]["endpoint_res_health"]["split_count"] == 2


def test_multi_shelxl_feedback_recomputes_rows_for_accepted_output_model(tmp_path: Path, monkeypatch):
    start_model = tmp_path / "start.res"
    start_model.write_text(valid_shelx_res_text("clean"))
    combined = tmp_path / "runs" / "shared_shelxl_feedback" / "cycle_01" / "combined"
    output_model = combined / "combined_endpoint_lam_1p0" / "multi_dyn_corrected.res"
    output_model.parent.mkdir(parents=True)
    output_model.write_text(valid_shelx_res_text("accepted_output"))
    (combined / "combined_summary.json").write_text(
        json.dumps(combined_endpoint_summary(output_model)) + "\n"
    )
    labels = []
    captured = []
    monkeypatch.setattr(owner_workflows_shared_refinement, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(owner_workflows_shared_refinement, "native_stage_cmd", lambda _python, _ds, structure, _baseline, out_dir, _args, **kwargs: captured.append(
            (
                str(structure),
                str(out_dir),
                kwargs["stage"],
                kwargs["endpoint"],
                kwargs.get("strict_pinned_contract", False),
            )
        )
        or ["native"])
    monkeypatch.setattr(owner_workflows_shared_refinement, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(owner_workflows_scale_materialization, "run_logged", lambda _cmd, _logs, label, **_kwargs: labels.append(label))
    monkeypatch.setattr(owner_workflows_shared_refinement, "run_logged", lambda _cmd, _logs, label, **_kwargs: labels.append(label))
    prep = PrepConfig(structure=str(start_model), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    args = argparse.Namespace(
        shelxl_feedback_cycles=1,
        shelxl_feedback_shelxl_iterations=1,
        dry_run=False,
        n_points=10,
    )

    current_model, summary = cli.run_multi_shelxl_feedback_loop(
        prep,
        args,
        "python",
        tmp_path / "logs",
        tmp_path,
        start_model=str(start_model),
        nuisance={"a": {}},
        structure_dataset_policy={"status": "disabled"},
    )

    assert current_model == str(output_model)
    assert labels == ["a_multi_ls1_cycle_01", "multi_ls1_cycle_01_combine", "a_final_row_recompute"]
    assert captured[0][-1] is True
    assert captured[-1] == (
        str(output_model),
        str(tmp_path / "runs" / "shared_shelxl_feedback" / "final_row_recompute" / "a"),
        "final_row_recompute",
        False,
        True,
    )
    assert summary["final_row_recompute"]["model"] == str(output_model)
    assert summary["final_row_recompute"]["status"] == "ok"


def test_multi_shelxl_feedback_zero_cycles_can_recompute_rows_without_ls1(tmp_path: Path, monkeypatch):
    start_model = tmp_path / "start.res"
    start_model.write_text(valid_shelx_res_text("clean"))
    labels = []
    captured = []
    monkeypatch.setattr(owner_workflows_shared_refinement, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(owner_workflows_shared_refinement, "native_stage_cmd", lambda _python, _ds, structure, _baseline, out_dir, _args, **kwargs: captured.append(
            (
                str(structure),
                str(out_dir),
                kwargs["stage"],
                kwargs["endpoint"],
                kwargs.get("strict_pinned_contract", False),
            )
        )
        or ["native"])
    monkeypatch.setattr(owner_workflows_scale_materialization, "run_logged", lambda _cmd, _logs, label, **_kwargs: labels.append(label))
    monkeypatch.setattr(owner_workflows_shared_refinement, "run_logged", lambda _cmd, _logs, label, **_kwargs: labels.append(label))
    prep = PrepConfig(structure=str(start_model), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    args = argparse.Namespace(
        shelxl_feedback_cycles=0,
        recompute_final_rows=True,
        dry_run=False,
        n_points=10,
    )

    current_model, summary = cli.run_multi_shelxl_feedback_loop(
        prep,
        args,
        "python",
        tmp_path / "logs",
        tmp_path,
        start_model=str(start_model),
        nuisance={"a": {}},
        structure_dataset_policy={"status": "disabled"},
    )

    assert current_model == str(start_model)
    assert labels == ["a_final_row_recompute"]
    assert captured == [
        (
            str(start_model),
            str(tmp_path / "runs" / "shared_shelxl_feedback" / "final_row_recompute" / "a"),
            "final_row_recompute",
            False,
            True,
        )
    ]
    assert summary["status"] == "ok_recomputed_zero_cycles"
    assert summary["final_row_recompute"]["model"] == str(start_model)


@pytest.mark.parametrize("modulation_p95", [None, 20.1])
def test_multi_shelxl_feedback_warns_but_adopts_missing_or_excessive_modulation(
    tmp_path: Path,
    monkeypatch,
    modulation_p95: float | None,
):
    start_model = tmp_path / "start.res"
    start_model.write_text(valid_shelx_res_text("clean"))
    combined = tmp_path / "runs" / "shared_shelxl_feedback" / "cycle_01" / "combined"
    output_model = combined / "combined_endpoint_lam_1p0" / "multi_dyn_corrected.res"
    output_model.parent.mkdir(parents=True)
    output_model.write_text(valid_shelx_res_text("bounded_modulation"))
    combined_summary = combined_endpoint_summary(output_model)
    if modulation_p95 is None:
        combined_summary["combined_endpoint"].pop("mod_p95")
    else:
        combined_summary["combined_endpoint"]["mod_p95"] = modulation_p95
    (combined / "combined_summary.json").write_text(json.dumps(combined_summary) + "\n")
    monkeypatch.setattr(owner_workflows_shared_refinement, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(owner_workflows_shared_refinement, "native_stage_cmd", lambda *args, **kwargs: ["native"])
    monkeypatch.setattr(owner_workflows_shared_refinement, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(owner_workflows_scale_materialization, "run_logged", lambda *args, **kwargs: None)
    monkeypatch.setattr(owner_workflows_shared_refinement, "run_logged", lambda *args, **kwargs: None)
    prep = PrepConfig(structure=str(start_model), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    args = argparse.Namespace(
        shelxl_feedback_cycles=1,
        shelxl_feedback_shelxl_iterations=1,
        publication_endpoint_max_modulation_p95=20.0,
        dry_run=False,
        n_points=10,
    )

    current_model, summary = cli.run_multi_shelxl_feedback_loop(
        prep,
        args,
        "python",
        tmp_path / "logs",
        tmp_path,
        start_model=str(start_model),
        nuisance={"a": {}},
        structure_dataset_policy={"status": "disabled"},
    )

    assert current_model == str(output_model)
    assert summary["status"] == "ok"
    assert summary["cycles"][0]["adopted_for_feedback"] is True
    assert summary["cycles"][0]["endpoint_modulation_gate"]["passed"] is True
    assert summary["cycles"][0]["endpoint_modulation_gate"]["blocking"] is False
    assert summary["cycles"][0]["endpoint_modulation_gate"]["status"] == "warning"


def test_multi_shelxl_feedback_stops_on_invalid_record_with_usable_res(tmp_path: Path, monkeypatch):
    start_model = tmp_path / "start.res"
    start_model.write_text(valid_shelx_res_text("clean"))
    combined = tmp_path / "runs" / "shared_shelxl_feedback" / "cycle_01" / "combined"
    invalid_model = combined / "combined_endpoint_lam_1p0" / "multi_dyn_corrected.res"
    invalid_model.parent.mkdir(parents=True)
    invalid_model.write_text(valid_shelx_res_text("invalid"))
    invalid_summary = combined_endpoint_summary(invalid_model, status="invalid_res_output")
    invalid_summary["combined_endpoint"]["shelxl"].update(
        {"invalid_res_reason": "missing atom records"}
    )
    invalid_summary["combined_endpoint"]["shelxl"]["res_health"].update(
        {"ok": False, "reason": "missing atom records"}
    )
    (combined / "combined_summary.json").write_text(json.dumps(invalid_summary) + "\n")
    labels = []
    monkeypatch.setattr(owner_workflows_shared_refinement, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(owner_workflows_shared_refinement, "native_stage_cmd", lambda *args, **kwargs: ["native"])
    monkeypatch.setattr(owner_workflows_shared_refinement, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(owner_workflows_scale_materialization, "run_logged", lambda cmd, logs, label, **kwargs: labels.append(label))
    monkeypatch.setattr(owner_workflows_shared_refinement, "run_logged", lambda cmd, logs, label, **kwargs: labels.append(label))
    prep = PrepConfig(structure=str(start_model), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    args = argparse.Namespace(
        shelxl_feedback_cycles=3,
        shelxl_feedback_shelxl_iterations=1,
        dry_run=False,
        n_points=10,
    )

    current_model, summary = cli.run_multi_shelxl_feedback_loop(
        prep,
        args,
        "python",
        tmp_path / "logs",
        tmp_path,
        start_model=str(start_model),
        nuisance={"a": {}},
        structure_dataset_policy={"status": "disabled"},
    )

    assert current_model == str(start_model)
    assert summary["current_model"] == str(start_model)
    assert summary["status"] == "stopped_unsafe_endpoint"
    assert summary["unsafe_model"] == str(invalid_model)
    assert "status=invalid_res_output" in summary["stopped_reason"]
    assert "missing atom records" in summary["stopped_reason"]
    assert len(summary["cycles"]) == 1
    assert summary["cycles"][0]["adopted_for_feedback"] is False
    assert labels == ["a_multi_ls1_cycle_01", "multi_ls1_cycle_01_combine"]
