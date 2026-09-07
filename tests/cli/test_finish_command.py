"""CLI finish command regression tests."""


import json
from pathlib import Path

import pytest

import grond.cli as cli
from grond.cli import (
    build_parser,
)
from grond.io.config import DatasetSpec, PrepConfig, write_prep


def test_finish_regenerates_compact_package(tmp_path: Path, capsys):
    run = tmp_path / "grond"
    final_run = run / "runs" / "dataset" / "final"
    screen_run = run / "runs" / "dataset" / "adaptive_screen"
    endpoint = run / "endpoint_calibration" / "dataset" / "window_1.5deg_det_8" / "endpoint_freeze-zero"
    for path in (final_run, screen_run, endpoint, run / "decisions" / "dataset"):
        path.mkdir(parents=True)
    (final_run / "native_dyn_calc.csv").write_text(
        "split,dyn_residual_over_sigma\n"
        "work,0.5\n"
        "free,1.5\n"
    )
    metrics = {
        "all": {"dyn": {"R1_all": 0.11, "wR2_all": 0.10}, "born": {"R1_all": 0.19, "wR2_all": 0.25}},
        "free": {"dyn": {"wR2_all": 0.15}, "born": {"wR2_all": 0.30}},
    }
    (final_run / "summary.json").write_text(
        json.dumps(
            {
                "best_native": {
                    "metrics": metrics,
                    "nuisance": {"thickness_A": 1200, "thickness_model": "uniform_gradient"},
                },
                "best_endpoint": {"q_p05": 0.8, "q_median": 1.0, "q_p95": 1.4},
            }
        )
    )
    (screen_run / "summary.json").write_text(json.dumps({"best_native": {"metrics": metrics}}))
    endpoint_res = endpoint / "dyn_scale_replay.res"
    endpoint_hkl = endpoint / "dyn_scale_replay.hkl"
    endpoint_lst = endpoint / "dyn_scale_replay.lst"
    endpoint_res.write_text("TITL endpoint\nCELL 0.025079 5 6 7 90 90 90\nEND\n")
    endpoint_hkl.write_text("   0   0   0    0.00    0.00\n")
    endpoint_lst.write_text(
        "wR2 = 0.1115, GooF = S = 0.906 for all data\n"
        "R1 = 0.05 for 1 Fo > 4sig(Fo) and 0.1105 for all 2 data\n"
        "0 atoms may be split and 0 atoms NPD\n"
    )
    (endpoint / "manifest.json").write_text(
        json.dumps(
            {
                "hkl_safe_factor": 1.0,
                "mod_p95": 1.2,
                "endpoint_b": 0.0,
                "keep_exti": False,
                "artifacts": {
                    "res": str(endpoint_res),
                    "res_sha256": cli.sha256_file(endpoint_res),
                    "hkl": str(endpoint_hkl),
                    "hkl_sha256": cli.sha256_file(endpoint_hkl),
                    "lst": str(endpoint_lst),
                    "lst_sha256": cli.sha256_file(endpoint_lst),
                },
            }
        )
        + "\n"
    )
    (run / "decisions" / "dataset" / "decision_tree.json").write_text(
        json.dumps(
            {
                "screen_run": str(screen_run),
                "selected_hand": "original",
                "transform_decision": {"selected": {"transform": "identity", "status": "ok"}},
            }
        )
    )
    write_prep(
        run / "run_prep.json",
        PrepConfig(
            structure=str(tmp_path / "model.res"),
            datasets=[DatasetSpec(name="dataset", hkl=str(tmp_path / "INTEGRATE.HKL"))],
        ),
    )
    (run / "grond_summary.json").write_text(
        json.dumps(
            {
                "primary_run": str(final_run),
                "model": str(endpoint_res),
                "decision_tree": str(run / "decisions" / "dataset" / "decision_tree.json"),
                "endpoint_calibration": {
                    "report": str(run / "endpoint_calibration" / "dataset" / "REPORT.md"),
                    "publication_selection": {
                        "publication_endpoint": {
                            "spec": "window_1.5deg_det_8",
                            "wght_mode": "freeze-zero",
                            "scale_params": 137,
                            "shelxl_R1gt": 0.05,
                            "shelxl_R1all": 0.1105,
                            "shelxl_wR2all": 0.1115,
                            "shelxl_GooF": 0.906,
                            "shelxl_Ngt": 1,
                            "shelxl_Nall": 2,
                            "native_free_wR2_all": 0.1508,
                            "hkl_safe_factor": 1.0,
                            "mod_p95": 1.2,
                            "endpoint_b": 0.0,
                            "keep_exti": False,
                            "endpoint_res": str(endpoint_res),
                            "endpoint_hkl": str(endpoint_hkl),
                            "endpoint_lst": str(endpoint_lst),
                            "endpoint_res_sha256": cli.sha256_file(endpoint_res),
                            "endpoint_hkl_sha256": cli.sha256_file(endpoint_hkl),
                            "endpoint_lst_sha256": cli.sha256_file(endpoint_lst),
                            "endpoint_manifest": str(endpoint / "manifest.json"),
                            "endpoint_row_table": str(endpoint.parent / "native_dyn_calc.csv"),
                        }
                    },
                },
            }
        )
    )
    (run / "grond_progress.json").write_text(
        json.dumps(
            {
                "completed": {},
                "finished": {
                    "status": "ok",
                    "snapshot_kind": "completed_invocation",
                    "invocation_elapsed_seconds": 300.0,
                    "completed_stages": 0,
                    "failed_stages": 0,
                    "estimated_stages": 0,
                    "cumulative_stage_elapsed_seconds": 0.0,
                    "stage_elapsed_seconds": 0.0,
                },
            }
        )
        + "\n"
    )

    args = build_parser().parse_args(["finish", "--run-dir", str(run), "--no-make-maps", "--json"])
    cli.cmd_finish(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["timing_context"]["kind"] == "finish_replay"
    assert payload["finished"]["invocation_elapsed_seconds"] == 300.0
    assert payload["publication_endpoint"]["R1all"] == pytest.approx(0.1105)
    assert (run / "FINAL_SUMMARY.md").exists()
    assert (run / "QUICK_STATS.md").exists()
    assert (run / "finish_manifest.json").exists()
    assert "GROND Final Summary" in (run / "FINAL_SUMMARY.md").read_text()


def test_finish_prefers_endpoint_chooser_physical_endpoint(tmp_path: Path, capsys):
    run = tmp_path / "grond"
    final_run = run / "runs" / "dataset" / "final"
    screen_run = run / "runs" / "dataset" / "adaptive_screen"
    calibration = run / "endpoint_calibration" / "dataset" / "adaptive_80_det_8" / "endpoint_freeze-zero"
    polished = run / "forensics" / "dataset" / "observation_polish" / "endpoint_lam_1p0"
    for path in (final_run, screen_run, calibration, polished, run / "decisions" / "dataset"):
        path.mkdir(parents=True)
    (final_run / "native_dyn_calc.csv").write_text("split,dyn_residual_over_sigma\nwork,0.5\nfree,1.5\n")
    metrics = {
        "all": {"dyn": {"R1_all": 0.21, "wR2_all": 0.33}, "born": {"R1_all": 0.28, "wR2_all": 0.42}},
        "free": {"dyn": {"wR2_all": 0.34}, "born": {"wR2_all": 0.43}},
    }
    (final_run / "summary.json").write_text(json.dumps({"best_native": {"metrics": metrics}, "best_endpoint": {}}))
    (screen_run / "summary.json").write_text(json.dumps({"best_native": {"metrics": metrics}}))
    calibration_res = calibration / "dyn_scale_replay.res"
    calibration_hkl = calibration / "dyn_scale_replay.hkl"
    calibration_lst = calibration / "dyn_scale_replay.lst"
    polished_res = polished / "dyn_observation_polished.res"
    polished_hkl = polished / "dyn_observation_polished.hkl"
    polished_lst = polished / "dyn_observation_polished.lst"
    for res in (calibration_res, polished_res):
        res.write_text("TITL endpoint\nCELL 0.025079 5 6 7 90 90 90\nEND\n")
    for hkl in (calibration_hkl, polished_hkl):
        hkl.write_text("   0   0   0    0.00    0.00\n")
    calibration_lst.write_text(
        "wR2 = 0.2726, GooF = S = 1.073 for all data\n"
        "R1 = 0.08 for 1 Fo > 4sig(Fo) and 0.1843 for all 2 data\n"
        "0 atoms may be split and 0 atoms NPD\n"
    )
    polished_lst.write_text(
        "WGHT 0 0\n"
        "wR2 = 0.1922, GooF = S = 0.935 for all data\n"
        "R1 = 0.0952 for 1 Fo > 4sig(Fo) and 0.1537 for all 2 data\n"
        "0 atoms may be split and 0 atoms NPD\n"
    )
    (run / "decisions" / "dataset" / "decision_tree.json").write_text(
        json.dumps(
            {
                "screen_run": str(screen_run),
                "selected_hand": "original",
                "transform_decision": {"selected": {"transform": "identity", "status": "ok"}},
            }
        )
    )
    write_prep(
        run / "run_prep.json",
        PrepConfig(
            structure=str(tmp_path / "model.res"),
            datasets=[DatasetSpec(name="dataset", hkl=str(tmp_path / "INTEGRATE.HKL"))],
        ),
    )
    (run / "grond_summary.json").write_text(
        json.dumps(
            {
                "primary_run": str(final_run),
                "model": str(calibration_res),
                "decision_tree": str(run / "decisions" / "dataset" / "decision_tree.json"),
                "endpoint_calibration": {
                    "publication_selection": {
                        "publication_endpoint": {
                            "spec": "adaptive_80_det_8",
                            "wght_mode": "freeze-zero",
                            "scale_params": 144,
                            "shelxl_R1all": 0.1843,
                            "shelxl_wR2all": 0.2726,
                            "shelxl_GooF": 1.073,
                            "endpoint_res": str(calibration_res),
                            "endpoint_hkl": str(calibration_hkl),
                            "endpoint_lst": str(calibration_lst),
                        }
                    }
                },
                "publication_endpoint_chooser": {
                    "status": "ok",
                    "best_physical_endpoint": {
                        "id": "dataset:observation_polish:B=0:noexti:candidate=1",
                        "label": "dataset:observation_polish",
                        "source_kind": "observation_polish",
                        "run_dir": str(polished.parent),
                        "lambda_q": 1.0,
                        "endpoint_b": 0.0,
                        "keep_exti": False,
                        "hkl_safe_factor": 1.0,
                        "modulation_p95": 1.0,
                        "R1_gt": 0.0952,
                        "R1_all": 0.1537,
                        "wR2_all": 0.1922,
                        "GooF": 0.935,
                        "N_gt": 1,
                        "N_all": 2,
                        "hkl": str(polished_hkl),
                        "lst": str(polished_lst),
                        "res": str(polished_res),
                        "res_sha256": cli.sha256_file(polished_res),
                        "hkl_sha256": cli.sha256_file(polished_hkl),
                        "lst_sha256": cli.sha256_file(polished_lst),
                        "artifact_provenance_ok": True,
                        "pathology_status": "complete",
                        "shelxl": {
                            "R1_gt": 0.0952,
                            "R1_all": 0.1537,
                            "wR2_all": 0.1922,
                            "GooF": 0.935,
                            "N_gt": 1,
                            "N_all": 2,
                            "lst": str(polished_lst),
                        },
                    },
                },
            }
        )
    )

    args = build_parser().parse_args(["finish", "--run-dir", str(run), "--no-make-maps", "--json"])
    cli.cmd_finish(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["publication_endpoint"]["R1all"] == pytest.approx(0.1537)
    assert payload["publication_endpoint"]["wght_mode"] == "freeze-zero"
    assert payload["publication_endpoint"]["pathology_status"] == "complete"
    assert payload["publication_endpoint"]["npd_count"] == 0
    assert payload["publication_endpoint"]["npd_atoms"] == []
    assert payload["publication_endpoint"]["has_npd"] is False
    assert payload["publication_endpoint"]["has_split_warnings"] is False
    assert payload["final_feedback_endpoint"]["pathology_status"] == "complete"
    assert payload["final_feedback_endpoint"]["npd_count"] == 0
    assert payload["final_feedback_endpoint"]["npd_atoms"] == []
    assert payload["final_feedback_endpoint"]["has_npd"] is False
    assert payload["final_feedback_endpoint"]["has_split_warnings"] is False
    assert payload["files"]["publication_res"] == str(polished_res)
    terminal = cli._format_terminal_finish_summary(payload)
    assert "WGHT freeze-zero" in terminal
    assert "LST pathology `complete`; NPD `0`; split `0`" in terminal
    assert "unverified" not in terminal
    assert "Scale selector:  /  = n/a" not in terminal
    final_text = (run / "FINAL_SUMMARY.md").read_text()
    quick_text = (run / "QUICK_STATS.md").read_text()
    assert "endpoint_chooser" in final_text
    assert "observation_polish" in final_text
    assert "endpoint_chooser | observation_polish" in quick_text
    assert "warning_missing_diagnostic" not in final_text
    assert "warning_missing_diagnostic" not in quick_text


def test_finish_does_not_promote_calibration_when_chooser_rejects_all(tmp_path: Path, capsys):
    run = tmp_path / "grond"
    final_run = run / "runs" / "dataset" / "final"
    screen_run = run / "runs" / "dataset" / "adaptive_screen"
    calibration = run / "endpoint_calibration" / "dataset" / "adaptive_80_det_8" / "endpoint_freeze-zero"
    for path in (final_run, screen_run, calibration, run / "decisions" / "dataset"):
        path.mkdir(parents=True)
    (final_run / "native_dyn_calc.csv").write_text("split,dyn_residual_over_sigma\nwork,0.5\nfree,1.5\n")
    metrics = {
        "all": {"dyn": {"R1_all": 0.18, "wR2_all": 0.45}, "born": {"R1_all": 0.19, "wR2_all": 0.48}},
        "free": {"dyn": {"wR2_all": 0.44}, "born": {"wR2_all": 0.48}},
    }
    (final_run / "summary.json").write_text(json.dumps({"best_native": {"metrics": metrics}, "best_endpoint": {}}))
    (screen_run / "summary.json").write_text(json.dumps({"best_native": {"metrics": metrics}}))
    calibration_res = calibration / "dyn_scale_replay.res"
    calibration_hkl = calibration / "dyn_scale_replay.hkl"
    calibration_lst = calibration / "dyn_scale_replay.lst"
    calibration_res.write_text("TITL endpoint\nCELL 0.025079 5 6 7 90 90 90\nEND\n")
    calibration_hkl.write_text("   0   0   0    0.00    0.00\n")
    calibration_lst.write_text("R1 = 0.05 for 1 Fo > 4sig(Fo) and 0.11 for all 2 data\n")
    (run / "decisions" / "dataset" / "decision_tree.json").write_text(
        json.dumps(
            {
                "screen_run": str(screen_run),
                "selected_hand": "alternate",
                "transform_decision": {"selected": {"transform": "identity", "status": "ok"}},
            }
        )
    )
    write_prep(
        run / "run_prep.json",
        PrepConfig(
            structure=str(tmp_path / "model.res"),
            datasets=[DatasetSpec(name="dataset", hkl=str(tmp_path / "INTEGRATE.HKL"))],
        ),
    )
    (run / "grond_summary.json").write_text(
        json.dumps(
            {
                "primary_run": str(final_run),
                "model": str(calibration_res),
                "decision_tree": str(run / "decisions" / "dataset" / "decision_tree.json"),
                "endpoint_calibration": {
                    "publication_selection": {
                        "publication_endpoint": {
                            "spec": "adaptive_80_det_8",
                            "wght_mode": "freeze-zero",
                            "scale_params": 143,
                            "shelxl_R1all": 0.183,
                            "shelxl_wR2all": 0.2716,
                            "shelxl_GooF": 1.118,
                            "endpoint_res": str(calibration_res),
                            "endpoint_hkl": str(calibration_hkl),
                            "endpoint_lst": str(calibration_lst),
                        }
                    }
                },
                "publication_endpoint_chooser": {"status": "no_endpoint_candidates", "best_physical_endpoint": {}},
            }
        )
    )

    args = build_parser().parse_args(["finish", "--run-dir", str(run), "--no-make-maps", "--json"])
    cli.cmd_finish(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["publication_endpoint"]["R1all"] is None
    assert payload["files"]["publication_res"] == ""
    text = (run / "FINAL_SUMMARY.md").read_text()
    assert "GROND Bloch dynamical model-handoff endpoint | not available" in text


def test_finish_endpoint_resolution_fails_closed_against_fresh_baseline(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "baseline_exti": {
                    "status": "ok",
                    "R1_all": 0.1377,
                    "npd_count": 0,
                    "split_count": 0,
                    "res_health": {"safe_for_feedback": True, "path": "fresh.res"},
                }
            }
        )
    )
    result = {
        "primary_run": str(run),
        "endpoint_calibration": {
            "publication_selection": {
                "publication_endpoint": {
                    "shelxl_R1all": 0.1404,
                    "shelxl_wR2all": 0.1786,
                    "shelxl_GooF": 1.036,
                    "endpoint_res": "corrected.res",
                }
            }
        },
    }

    assert cli._publication_endpoint_from_result(result) == {}
