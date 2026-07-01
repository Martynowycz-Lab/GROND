import argparse
import json
import os
from pathlib import Path

import pytest

import grond.cli as cli
from grond.cli import (
    apply_speed_profile,
    apply_tool_config,
    build_parser,
    check_sfac_contract,
    combine_cmd,
    estimate_model_nonh_atoms,
    parse_numeric_grid,
    publication_report_cmd,
    required_external_tool_errors,
    resolve_structure_update_strategy,
    resolved_dataset_sigma_policy,
    resolved_merge_sigma_policy,
    resolved_post_bloch_sigma_policy,
    resolved_sg_window_mode,
    transform_scout_job_count,
)
from grond.io.config import DatasetSpec, PrepConfig, write_prep

EXPECTED_CLI_COMMANDS = {
    "auto",
    "calibrate-endpoint",
    "compile",
    "doctor",
    "endpoints",
    "export-mtz",
    "finish",
    "forensics",
    "import-dials",
    "init-config",
    "make-free-flags",
    "prep",
    "protein-ls1",
    "publish",
    "push-endpoint",
    "raw-polish",
    "row-template",
    "row-validate",
    "run",
    "scale-polish",
    "scale-replay",
    "sfac-check",
    "thickness-scout",
    "validate",
    "validate-ls1",
}


def cli_command_names(parser):
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if choices:
            return set(choices)
    raise AssertionError("No CLI subcommands found")


def test_all_registered_subcommands_have_help(capsys):
    parser = build_parser()
    assert cli_command_names(parser) == EXPECTED_CLI_COMMANDS

    for command in sorted(EXPECTED_CLI_COMMANDS):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([command, "--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out
        assert command in captured.out


def test_run_parser_exposes_staged_feature_grids():
    args = build_parser().parse_args(["run", "--out-dir", "out", "--hkl", "INTEGRATE.HKL", "--structure", "model.res"])
    assert args.sigma_b_corr_grid == "0.0"
    assert args.sg_jitter_sigma_frame_grid == "0.0"
    assert args.absorption_mode_grid == "none,midgley_like"
    assert args.absorption_mu_theta2_grid == "0.0"
    assert args.absorption_tds_b_grid == "0.0"
    assert args.dose_b_mode_grid == "none"


def test_thickness_scout_parser_exposes_contract_diagnostic_defaults():
    args = build_parser().parse_args(
        [
            "thickness-scout",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
        ]
    )
    assert args.command == "thickness-scout"
    assert args.n_basis_grid == "24,48"
    assert args.n_points == 10
    assert args.thickness_grid_nm == "50,75,110,150,180,220,260,300"
    assert args.scale_modes == "production,no-detector"
    assert args.feature_modes == "bare,final-diagnostic"
    assert args.local_confirmation is True
    assert args.thickness_scout_final_absorption_mode == "midgley_like"
    assert args.prediction_cache_scope == "shared"
    assert args.prediction_cache_compression == "none"
    assert args.compute_backend == "scipy"
    assert args.row_plan_cache == "off"
    assert args.jax_batch_size == 0
    assert args.jax_matrix_bucket_size == 0
    assert args.matrix_convention == "j_minus_i"
    assert args.profile_quadrature == "uniform"
    assert args.profile_accumulation == "sum"
    assert args.detector_scale_bins == "8"
    assert args.lambda_grid == "1.0"
    assert args.scale_smoothness == 0.0
    assert args.endpoint_b_grid == "0.0"
    assert args.publication_endpoint_chooser is True
    assert args.publication_endpoint_min_hkl_safe_factor == 0.25
    assert args.publication_endpoint_max_modulation_p95 == 20.0
    assert args.endpoint_calibration == "auto"
    assert "adaptive:120:det8" in args.endpoint_calibration_specs
    assert args.endpoint_calibration_lambda_q == 1.0
    assert args.endpoint_calibration_wght_modes == "freeze-zero,promote"
    assert args.endpoint_calibration_prefer_wght_mode == "freeze-zero"
    assert args.endpoint_calibration_goof_min == 0.85
    assert args.endpoint_calibration_goof_max == 1.15
    assert args.endpoint_calibration_max_r1_cost == 0.010
    assert args.endpoint_calibration_free_wr2_tolerance == 0.040
    assert args.endpoint_calibration_selection_policy == "bic"
    assert args.endpoint_calibration_information_criterion == "bic_fitted_variance"
    assert args.endpoint_calibration_sigma_min_scale == 0.05
    assert args.endpoint_calibration_shelxl_iterations == 3
    assert args.fast is False
    assert args.full_fast is False
    assert args.full_fast_n_basis == 24
    assert args.full_fast_n_points == 10
    assert args.full_fast_structure_rows == 0
    assert args.full_fast_structure_max_nfev == 0
    assert args.scout_n_basis == 8
    assert args.scout_n_points == 3
    assert args.transform_rerank_top == 3
    assert args.transform_rerank_n_basis == 24
    assert args.transform_rerank_n_points == 10
    assert args.transform_rerank_max_rows == 0
    assert args.transform_scout_jobs == 0
    assert args.transform_scout_workers_per_job == 1
    assert args.transform_metadata_prefilter == "auto"
    assert args.transform_metadata_max_candidates == 3
    assert args.transform_metadata_min_correlation == 0.70


def test_finish_parser_defaults_to_existing_run_dir():
    args = build_parser().parse_args(["finish", "--run-dir", "grond_out", "--no-make-maps"])
    assert args.command == "finish"
    assert args.run_dir == "grond_out"
    assert args.checkpoint == ""
    assert args.make_maps is False
    assert args.map_mode == "native"


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
    endpoint_lst.write_text("R1 = 0.05 for 1 Fo > 4sig(Fo) and 0.11 for all 2 data\n")
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
                            "shelxl_R1all": 0.1105,
                            "shelxl_wR2all": 0.1115,
                            "shelxl_GooF": 0.906,
                            "native_free_wR2_all": 0.1508,
                            "endpoint_res": str(endpoint_res),
                            "endpoint_hkl": str(endpoint_hkl),
                            "endpoint_lst": str(endpoint_lst),
                            "endpoint_manifest": str(endpoint / "manifest.json"),
                            "endpoint_row_table": str(endpoint.parent / "native_dyn_calc.csv"),
                        }
                    },
                },
            }
        )
    )

    args = build_parser().parse_args(["finish", "--run-dir", str(run), "--no-make-maps", "--json"])
    cli.cmd_finish(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
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
    for lst in (calibration_lst, polished_lst):
        lst.write_text("R1 = 0.05 for 1 Fo > 4sig(Fo) and 0.11 for all 2 data\n")
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
                        "hkl": str(polished_hkl),
                        "lst": str(polished_lst),
                        "res": str(polished_res),
                        "shelxl": {
                            "active_wght_a": 0.0,
                            "active_wght_b": 0.0,
                            "R1_gt": 0.0952,
                            "R1_all": 0.1537,
                            "wR2_all": 0.1922,
                            "GooF": 0.935,
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
    assert payload["files"]["publication_res"] == str(polished_res)
    final_text = (run / "FINAL_SUMMARY.md").read_text()
    quick_text = (run / "QUICK_STATS.md").read_text()
    assert "endpoint_chooser" in final_text
    assert "observation_polish" in final_text
    assert "endpoint_chooser | observation_polish" in quick_text


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
    assert "publication corrected-HKL endpoint | `n/a`" in text


def test_terminal_finish_summary_is_human_readable():
    payload = {
        "status": "ok",
        "out_dir": "/tmp/grond",
        "final_summary": "/tmp/grond/FINAL_SUMMARY.md",
        "quick_stats": "/tmp/grond/QUICK_STATS.md",
        "full_json": "/tmp/grond/grond_summary.json",
        "finished": {"completed_stages": 25, "estimated_stages": 30, "stage_elapsed_seconds": 822},
        "final_hand": "alternate",
        "final_hand_z_all": 1.505,
        "final_hand_z_free": 0.876,
        "selected_transform": "identity",
        "input_shelxl": {
            "R1gt": 0.1382,
            "R1all": 0.1714,
            "wR2all": 0.3845,
            "GooF": 1.406,
            "parameters": 123,
            "restraints": 75,
            "npd_count": 0,
            "split_count": 1,
        },
        "input_model": {"row_R1all": 0.22128, "row_wR2all": 0.43413, "born_R1all": 0.28, "born_wR2all": 0.42},
        "final_row_model": {
            "row_R1all": 0.21300,
            "row_wR2all": 0.33100,
            "row_free_wR2all": 0.33448,
            "born_R1all": 0.27,
            "born_wR2all": 0.41,
            "born_free_wR2all": 0.40,
            "row_table": "/tmp/rows.csv",
        },
        "final_feedback_endpoint": {
            "R1gt": 0.1100,
            "R1all": 0.1600,
            "wR2all": 0.2200,
            "GooF": 0.950,
            "npd_count": 0,
            "split_count": 0,
        },
        "publication_endpoint": {
            "R1gt": 0.1446,
            "R1all": 0.1785,
            "wR2all": 0.2572,
            "GooF": 1.036,
            "scale_spec": "adaptive_120_det_8",
            "scale_params": 184,
            "wght_mode": "freeze-zero",
            "npd_count": 0,
            "split_count": 0,
        },
        "files": {"publication_res": "/tmp/endpoint.res", "publication_hkl": "/tmp/endpoint.hkl"},
    }

    text = cli._format_terminal_finish_summary(payload)

    assert "GROND finished successfully in 13m42s." in text
    assert "Progress: 25 stages completed; initial estimate was 30." in text
    assert "Hand: alternate (Z all/free = 1.505 / 0.876" in text
    assert "Starting SHELXL reference" in text
    assert "R1gt 13.82%, R1all 17.14%, wR2all 38.45%, GooF 1.406" in text
    assert "Input dynamical rows: R1all 22.13%, wR2all 43.41%" in text
    assert "Input matched Born rows: R1all 28.00%, wR2all 42.00%" in text
    assert "Final dynamical rows: R1all 21.30%, wR2all 33.10%, free wR2all 33.45%" in text
    assert "Final matched Born rows: R1all 27.00%, wR2all 41.00%, free wR2all 40.00%" in text
    assert "Change: R1all -0.83 pp, wR2all -10.31 pp" in text
    assert "Final LS1 corrected-HKL endpoint: R1gt 11.00%, R1all 16.00%, wR2all 22.00%, GooF 0.950" in text
    assert "Publication endpoint: R1gt 14.46%, R1all 17.85%, wR2all 25.72%, GooF 1.036" in text


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


def test_auto_parser_hides_lambda_controls(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["auto", "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--lambda-grid" not in captured.out
    assert "--endpoint-calibration-lambda-q" not in captured.out
    assert "--observation-polish-lambda-grid" not in captured.out
    assert "--hand-control" in captured.out
    assert "--hand-n-basis" in captured.out
    assert "--hand-accept-delta" in captured.out

    args = parser.parse_args(["auto", "--out-dir", "out", "--hkl", "INTEGRATE.HKL", "--structure", "model.res"])
    assert args.lambda_grid == "1.0"
    assert args.endpoint_calibration_lambda_q == 1.0
    assert args.observation_polish_lambda_grid == ""
    assert args.final_basis_scan_limit == 500
    assert args.final_basis_margin == 10
    assert args.final_basis_max == 500
    assert args.sg_window_scout == "off"
    assert args.sg_window_scout_grid == "0.003,0.004,0.006,0.008"
    assert args.sg_window_scout_n_basis == 24
    assert args.sg_window_scout_n_points == 10
    assert args.sg_window_scout_max_rows == 0
    assert args.sg_window_scout_min_wR2_gain == 0.002
    assert args.sg_window_scout_free_wR2_tolerance == 0.003
    assert args.n_points == 20
    assert args.screen_n_basis == 24
    assert args.hand_n_basis == 24
    hand_args = parser.parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--hkl",
            "INTEGRATE.HKL",
            "--structure",
            "model.res",
            "--no-hand-control",
            "--hand-n-basis",
            "30",
            "--hand-n-points",
            "12",
            "--hand-accept-delta",
            "0.02",
        ]
    )
    assert hand_args.hand_control is False
    assert hand_args.hand_n_basis == 30
    assert hand_args.hand_n_points == 12
    assert hand_args.hand_accept_delta == 0.02
    assert args.structure_n_basis == 24
    assert args.scale_polish_frame_models == "adaptive,fixed"
    assert args.scale_polish_adaptive_counts == "auto"
    assert args.scale_polish_detector_bins == "exact"
    assert args.scale_polish_selection_metrics == "Rall,wRall,native_R1_all,native_wR2_all"
    assert args.scale_polish_scan_offsets is False
    assert args.resolution_limit_gate is True
    assert args.resolution_shells == 10
    assert args.resolution_strict_split_cc == 0.30
    assert args.resolution_max_shell_mwrall == 0.60
    assert args.dataset_policy == "auto"
    assert args.dataset_policy_min_weight == 0.25
    assert args.dataset_policy_unique_gain_fraction == 0.01
    assert args.structure_update == "auto"
    assert args.shelxl == "shelxl"
    assert args.config == ""
    assert args.allow_missing_tools is False
    assert args.shelxl_feedback_cycles == 5
    assert args.shelxl_feedback_shelxl_iterations == 1
    assert args.endpoint_wght_mode == "freeze-zero"
    assert args.dataset_sigma_policy == "auto"
    assert args.dataset_sigma_holdout_fraction == 0.20
    assert args.dataset_sigma_min_scale == 0.25
    assert args.dataset_sigma_max_scale == 4.0
    assert args.dataset_sigma_holdout_floor is True
    assert args.merge_sigma_policy == "auto"
    assert args.shelxl_iterations == 5
    assert args.post_bloch_sigma_policy == "auto"
    assert args.post_bloch_sigma_holdout_fraction == 0.20
    assert args.post_bloch_sigma_min_scale == 0.25
    assert args.post_bloch_sigma_max_scale == 4.0


def test_utility_commands_parse():
    doc = build_parser().parse_args(["doctor", "--probe-bloch"])
    assert doc.command == "doctor"
    assert doc.probe_bloch is True
    sfac = build_parser().parse_args(["sfac-check", "--structure", "model.res"])
    assert sfac.command == "sfac-check"
    assert build_parser().parse_args(["compile", "--force"]).force is True
    cfg = build_parser().parse_args(["init-config", "--out", "grond.config.json"])
    assert cfg.command == "init-config"
    assert cfg.format == "json"
    row_template = build_parser().parse_args(["row-template", "--rows-out", "rows.csv", "--metadata-out", "rows_metadata.json"])
    assert row_template.command == "row-template"
    row_validate = build_parser().parse_args(["row-validate", "--rows", "rows.csv", "--metadata", "rows_metadata.json"])
    assert row_validate.command == "row-validate"
    free_flags = build_parser().parse_args(
        [
            "make-free-flags",
            "--rows",
            "rows.csv",
            "--out",
            "free_flags.csv",
            "--cell",
            "10 11 12 90 90 90",
            "--space-group",
            "P 1",
        ]
    )
    assert free_flags.command == "make-free-flags"
    assert free_flags.merge_anomalous is True
    mtz = build_parser().parse_args(
        [
            "export-mtz",
            "--row-table",
            "rows.csv",
            "--free-flags",
            "free_flags.csv",
            "--out-mtz",
            "corrected.mtz",
            "--cell",
            "10 11 12 90 90 90",
            "--space-group",
            "P 1",
            "--q-policy",
            "low_resolution_damped",
        ]
    )
    assert mtz.command == "export-mtz"
    assert mtz.q_policy == "low_resolution_damped"
    protein = build_parser().parse_args(
        [
            "protein-ls1",
            "--model",
            "model.pdb",
            "--mtz",
            "corrected.mtz",
            "--out-dir",
            "protein_endpoint",
            "--command",
            "echo {model} {mtz}",
            "--output-model",
            "{out_dir}/refined.pdb",
            "--dry-run",
        ]
    )
    assert protein.command == "protein-ls1"
    assert protein.endpoint == "custom"
    assert protein.free_label == "FreeR_flag"
    endpoints = build_parser().parse_args(["endpoints", "--json"])
    assert endpoints.command == "endpoints"
    audit = build_parser().parse_args(["forensics", "--row-table", "native_dyn_calc.csv", "--hkl", "INTEGRATE.HKL", "--res", "model.res", "--out-dir", "audit", "--forensic-depth", "full"])
    assert audit.command == "forensics"
    assert audit.forensic_depth == "full"
    assert audit.forensic_gate is True
    assert audit.observation_polish == "auto"
    assert audit.observation_polish_combine == "ensemble"
    replay = build_parser().parse_args(["scale-replay", "--row-table", "native_dyn_calc.csv", "--out-dir", "replay"])
    assert replay.command == "scale-replay"
    assert replay.post_bloch_sigma_policy == "uniform_z_rms"
    assert replay.publication_prefer_wght_mode == "freeze-zero"
    calibrate = build_parser().parse_args(
        ["calibrate-endpoint", "--row-table", "native_dyn_calc.csv", "--out-dir", "calibrated"]
    )
    assert calibrate.command == "calibrate-endpoint"
    assert calibrate.wght_modes == "freeze-zero,promote"
    assert calibrate.shelxl_iterations == 3
    assert calibrate.post_bloch_sigma_min_scale == 0.05
    assert calibrate.publication_goof_min == 0.85
    assert calibrate.publication_free_wr2_tolerance == 0.040
    assert calibrate.publication_selection_policy == "bic"
    assert calibrate.publication_information_criterion == "bic_fitted_variance"
    ls1 = build_parser().parse_args(
        [
            "validate-ls1",
            "--source-summary",
            "summary.json",
            "--start-res",
            "model.res",
            "--out-dir",
            "validate_ls1",
            "--native-polish",
        ]
    )
    assert ls1.command == "validate-ls1"
    assert ls1.cycles == 5
    assert ls1.n_basis == 30
    assert ls1.n_points == 20
    assert ls1.compute_backend == "scipy_bethe"
    assert ls1.row_plan_cache == "off"
    assert ls1.q_control_seed == 12345
    assert ls1.q_control_shells == 10
    assert ls1.wrong_hand_control is True
    assert ls1.native_polish is True
    assert ls1.native_polish_blocks == "adp_ueq,xyz"
    ls1_no_hand = build_parser().parse_args(
        [
            "validate-ls1",
            "--source-summary",
            "summary.json",
            "--start-res",
            "model.res",
            "--out-dir",
            "validate_ls1",
            "--no-wrong-hand-control",
        ]
    )
    assert ls1_no_hand.wrong_hand_control is False
    push = build_parser().parse_args(
        [
            "push-endpoint",
            "--source-summary",
            "summary.json",
            "--start-res",
            "model.res",
            "--out-dir",
            "push",
            "--model-prep",
            "free-h-anis-no-restraints",
        ]
    )
    assert push.command == "push-endpoint"
    assert push.cycles == 3
    assert push.n_basis == 108
    assert push.n_points == 20
    assert push.compute_backend == "scipy_bethe"
    assert push.row_plan_cache == "readwrite"
    assert push.model_prep == "free-h-anis-no-restraints"
    assert push.set_ls_cycles == 1
    push_checkpoint = build_parser().parse_args(
        [
            "push-endpoint",
            "--checkpoint",
            "grond_continue_checkpoint.json",
            "--out-dir",
            "push",
        ]
    )
    assert push_checkpoint.command == "push-endpoint"
    assert push_checkpoint.checkpoint == "grond_continue_checkpoint.json"
    assert push_checkpoint.source_summary == ""
    assert push_checkpoint.start_res == ""


def test_sfac_check_rejects_element_only_shelx_factors(tmp_path: Path, capsys):
    model = tmp_path / "bad.res"
    model.write_text("TITL bad\nCELL 1 5 6 7 90 90 90\nSFAC C H N O\nHKLF 4\nEND\n")

    args = build_parser().parse_args(["sfac-check", "--structure", str(model)])
    assert args.func(args) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["ok"] is False
    assert payload["sfac_rows"][0]["kind"] == "element_only"
    assert "explicit electron scattering coefficient rows" in payload["messages"][1]["message"]


def test_sfac_check_accepts_numeric_rows_and_warns_on_placeholder_wavelength(tmp_path: Path):
    model = tmp_path / "ok.res"
    model.write_text("TITL ok\nCELL 1 5 6 7 90 90 90\nSFAC C 1 0 1 0 1 0 1 0\nHKLF 4\nEND\n")

    payload = check_sfac_contract(model)

    assert payload["status"] == "warning"
    assert payload["ok"] is True
    assert payload["sfac_rows"][0]["kind"] == "numeric_coefficients"
    assert payload["cell"]["endpoint_wavelength_A"] == pytest.approx(0.025079)


def test_sfac_check_warns_acta_and_hklf3_endpoint_conversion(tmp_path: Path):
    model = tmp_path / "hklf3.res"
    model.write_text("TITL ok\nCELL 0.02508 5 6 7 90 90 90\nSFAC C 1 0 1 0 1 0 1 0\nACTA\nHKLF 3\nEND\n")

    payload = check_sfac_contract(model)

    assert payload["ok"] is True
    assert payload["status"] == "warning"
    assert payload["endpoint_contract"]["hklf"] == "3"
    assert payload["endpoint_contract"]["hklf_status"] == "source_amplitudes_endpoint_forced_hklf4"
    messages = " ".join(item["message"] for item in payload["messages"])
    assert "ACTA" in messages
    assert "HKLF 3" in messages


def test_sfac_check_rejects_unsupported_shelx_endpoint_contracts(tmp_path: Path):
    cases = [
        ("HKLF 5", "HKLF 5"),
        ("TWIN 0 1 0 1 0 0 0 0 -1\nHKLF 4", "TWIN"),
        ("BASF 0.2\nHKLF 4", "BASF"),
        ("REM PLATON SQUEEZE applied\nHKLF 4", "SQUEEZE"),
    ]
    for body, expected in cases:
        model = tmp_path / f"{expected.lower()}.res"
        model.write_text(f"TITL bad\nCELL 0.02508 5 6 7 90 90 90\nSFAC C 1 0 1 0 1 0 1 0\n{body}\nEND\n")

        payload = check_sfac_contract(model)

        assert payload["ok"] is False
        assert payload["status"] == "error"
        assert any(expected in item["message"] for item in payload["messages"])


def test_doctor_reports_current_install_feature_flags(capsys):
    args = build_parser().parse_args(["doctor"])

    assert args.func(args) == 0

    payload = json.loads(capsys.readouterr().out)
    grond = payload["grond"]
    assert grond["install_status"] == "current_required_features_present"
    assert grond["required_features"] == {
        "shelxl_feedback_res_guard": True,
        "usable_feedback_model_selection": True,
        "final_feedback_row_recompute": True,
        "corrected_endpoint_hklf4_contract": True,
        "absolute_structure_no_pandas": True,
    }


def test_auto_parser_is_simple_run_alias():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"])
    assert args.command == "auto"
    assert args.mode == "auto"
    assert args.resume is True
    assert args.detector_scale_bins == "8"
    assert args.final_basis_max == 500
    assert args.scale_polish_detector_bins == "exact"


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


def test_experimental_row_plan_cache_flags_parse():
    auto = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--row-plan-cache",
            "readwrite",
        ]
    )
    assert auto.row_plan_cache == "readwrite"

    raw = build_parser().parse_args(
        [
            "raw-polish",
            "--out-dir",
            "out",
            "--hkl",
            "INTEGRATE.HKL",
            "--structure",
            "model.res",
            "--row-plan-cache",
            "rebuild",
        ]
    )
    assert raw.row_plan_cache == "rebuild"

    ls1 = build_parser().parse_args(
        [
            "validate-ls1",
            "--source-summary",
            "summary.json",
            "--start-res",
            "model.res",
            "--out-dir",
            "validate_ls1",
            "--row-plan-cache",
            "readwrite",
        ]
    )
    assert ls1.row_plan_cache == "readwrite"


def test_auto_parser_accepts_dials_input_pair():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--dials-refl",
            "integrated.refl",
            "--dials-expt",
            "integrated.expt",
            "--structure",
            "model.res",
            "--dials-python",
            "/path/to/dials.python",
        ]
    )
    assert args.command == "auto"
    assert args.dials_refl == "integrated.refl"
    assert args.dials_expt == "integrated.expt"
    assert args.dials_intensity_mode == "sum_complete"
    assert args.dials_sigma_mode == "auto"
    assert args.dials_python == "/path/to/dials.python"
    assert args.dials_drop_dont_integrate is False
    assert args.dials_keep_dont_integrate is False
    assert args.sg_window_mode == "auto"


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


def test_write_quick_stats_summarizes_row_and_endpoint_tables(tmp_path):
    out_dir = tmp_path / "grond_out"
    tables = out_dir / "tables"
    tables.mkdir(parents=True)
    (tables / "row_mr_metrics.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "row_count": 10,
                    "n_basis": "24",
                    "n_points": "20",
                    "thickness_A_mean": 1000.0,
                    "sg_window": "0.00400",
                    "absorption_mode": "midgley_like",
                    "sg_jitter_mode": "gaussian",
                },
                "resolution_scale_endpoints": {
                    "endpoints": [
                        {
                            "label": "requested_resolution_full_scales",
                            "dmin_A": 0.7,
                            "scale_params": "100",
                            "row_mr": {
                                "dynamic": {
                                    "Nobs": 4,
                                    "Nall": 10,
                                    "Rall": 0.10,
                                    "wRall": 0.11,
                                    "MRall": 0.08,
                                    "MwRall": 0.09,
                                },
                                "kinematic": {"Rall": 0.20, "wRall": 0.21},
                            },
                        }
                    ]
                },
            }
        )
        + "\n"
    )
    prep = PrepConfig(structure="model.res", datasets=[DatasetSpec(name="dataset", hkl="INTEGRATE.HKL")])
    final_run = out_dir / "runs" / "dataset" / "adaptive_promoted_hand_final"
    final_run.mkdir(parents=True)
    (final_run / "native_dyn_calc.csv").write_text(
        "\n".join(
            [
                "row_index,h,k,l,split,dyn_residual_over_sigma",
                "0,1,0,0,work,0.5",
                "1,0,1,0,work,-1.5",
                "2,0,0,1,free,2.0",
                "3,1,1,0,free,-1.0",
            ]
        )
        + "\n"
    )
    (final_run / "summary.json").write_text(
        json.dumps(
            {
                "best_native": {
                    "nuisance": {
                        "thickness_A": 1100.0,
                        "thickness_model": "uniform_gradient",
                        "thickness_grad_frac": 1.2,
                        "frame_offset": 0.2,
                        "absorption_mode": "midgley_like",
                        "absorption_frac": 0.0,
                        "sg_jitter_mode": "gaussian",
                        "matrix_convention": "j_minus_i",
                        "native_wght_a": 0.12,
                        "native_wght_b": 0.56,
                    },
                    "metrics": {
                        "all": {
                            "dyn": {"R1_all": 0.11, "wR2_all": 0.21},
                            "born": {"R1_all": 0.17, "wR2_all": 0.31},
                        },
                        "free": {
                            "dyn": {"wR2_all": 0.22},
                            "born": {"wR2_all": 0.32},
                        },
                    },
                },
                "best_endpoint": {"q_p05": 0.2, "q_median": 0.4, "q_p95": 1.9},
            }
        )
        + "\n"
    )
    decisions = out_dir / "decisions" / "dataset"
    decisions.mkdir(parents=True)
    diagnostic = final_run.parent / "adaptive_absolute_structure_diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "all": {"n": 4, "delta_chi2_other_minus_best": -25.0, "paired_row_t_delta_chi2": -5.0},
                "work": {"n": 2, "delta_chi2_other_minus_best": -16.0, "paired_row_t_delta_chi2": -4.0},
                "free": {"n": 2, "delta_chi2_other_minus_best": -9.0, "paired_row_t_delta_chi2": -3.0},
            }
        )
        + "\n"
    )
    decision_tree = decisions / "decision_tree.json"
    decision_tree.write_text(
        json.dumps(
            {
                "selected_transform": "identity",
                "selected_hand": "alternate",
                "transform_decision": {
                    "selected": {"transform": "identity", "status": "ok"},
                    "metadata_transform_decision": {
                        "status": "promoted",
                        "correlation_gap": 0.0876,
                        "records": [{"log_intensity_corr": 0.9658, "paired_rows": 3931, "unique_overlap": 4039}],
                    },
                },
                "absolute_structure_decision": {
                    "selected_hand": "original",
                    "score_delta_original_minus_alternate": -0.01,
                },
                "post_feedback_hand_promotion": {
                    "status": "promoted",
                    "active_hand": "original",
                    "opposite_hand": "alternate",
                    "diagnostic": str(diagnostic),
                    "score_delta_final_minus_opposite": 0.08,
                },
            }
        )
        + "\n"
    )
    (out_dir / "grond_progress.json").write_text(
        json.dumps(
            {
                "completed": {
                    "old": {
                        "returncode": 0,
                        "elapsed_seconds": 1.0,
                        "log": str(out_dir / "logs" / "old.log"),
                        "command": [
                            "python",
                            "/old/GROND/src/grond/legacy/engine/native_row_regularize.py",
                        ],
                    },
                    "new": {
                        "returncode": 0,
                        "elapsed_seconds": 2.0,
                        "log": str(out_dir / "logs" / "new.log"),
                        "command": [
                            "python",
                            "/new/GROND/src/grond/legacy/engine/native_row_regularize.py",
                        ],
                    },
                }
            }
        )
        + "\n"
    )
    (out_dir / "sfac_check.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "cell": {"endpoint_wavelength_A": 0.025079},
                "sfac_rows": [{"kind": "numeric_coefficients", "element": "C"}],
            }
        )
        + "\n"
    )
    result = {
        "model": "final.res",
        "primary_run": str(final_run),
        "decision_tree": str(decision_tree),
        "endpoint_calibration": {
            "report": str(out_dir / "endpoint_calibration" / "dataset" / "REPORT.md"),
            "publication_selection": {
                "publication_endpoint": {
                    "spec": "adaptive_80_det_8",
                    "wght_mode": "freeze-zero",
                    "scale_params": 143,
                    "shelxl_R1all": 0.0996,
                    "shelxl_wR2all": 0.1539,
                    "shelxl_GooF": 1.037,
                    "native_free_wR2_all": 0.2318,
                    "sigma_scale": 0.584,
                }
            },
        },
    }
    args = argparse.Namespace(out_dir=str(out_dir))

    info = cli.write_quick_stats(prep, result, args)
    text = (out_dir / "QUICK_STATS.md").read_text()

    assert info["status"] == "ok"
    assert "requested_resolution_full_scales" in text
    assert "10.00%" in text
    assert "selected_scale_replay" in text
    assert "15.39%" in text
    assert "Resume Warning" in text
    assert "mixed-code resume" in text
    assert "Stage Timings" in text
    assert "Total completed stage wall time: `0m03s`" in text
    assert "Input Contract, Hand, And Z Diagnostics" in text
    assert "SFAC / endpoint wavelength check | status `ok`; endpoint wavelength `0.025079` A" in text
    assert "selected HKL transform | `identity`" in text
    assert "final hand | `alternate`; post-feedback status `promoted`" in text
    assert "final hand Z, all/work/free | `5.000` / `4.000` / `3.000`" in text
    assert "free-row I residual Z | n `2`; RMSZ `1.581`" in text
    assert "thickness | `1100.0 A`; model `uniform_gradient`; gradient `1.200`" in text


def test_row_validate_reports_bad_portable_rows_without_traceback(tmp_path: Path, capsys):
    from grond.io.row_contract import write_template

    rows_path, metadata_path = write_template(tmp_path / "rows.csv", tmp_path / "rows_metadata.json")
    rows_path.write_text(rows_path.read_text().replace("42.1", "-42.1", 1))

    args = build_parser().parse_args(
        [
            "row-validate",
            "--rows",
            str(rows_path),
            "--metadata",
            str(metadata_path),
            "--write-xds-dir",
            str(tmp_path / "bridge"),
        ]
    )
    assert args.func(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "positive sigma_Iobs" in payload["error"]
    assert not (tmp_path / "bridge" / "INTEGRATE.HKL").exists()


def test_auto_sg_window_mode_keeps_fixed_width_unless_explicit():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"])
    ds = DatasetSpec(name="d", hkl="missing.INTEGRATE.HKL")

    assert resolved_sg_window_mode(args, ds) == "fixed"

    args = build_parser().parse_args(
        ["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res", "--sg-window-mode", "dials_bbox"]
    )
    assert resolved_sg_window_mode(args, ds) == "dials_bbox"


def test_parse_numeric_grid_accepts_commas_semicolons_and_spaces():
    assert parse_numeric_grid("0.003, 0.004;0.006 0.008") == [0.003, 0.004, 0.006, 0.008]


def test_import_dials_parser_exposes_bridge_options():
    args = build_parser().parse_args(
        [
            "import-dials",
            "--refl",
            "integrated.refl",
            "--expt",
            "integrated.expt",
            "--out-dir",
            "bridge",
            "--intensity-mode",
            "sum_raw",
            "--dataset-name",
            "370_dials",
        ]
    )
    assert args.command == "import-dials"
    assert args.intensity_mode == "sum_raw"
    assert args.sigma_mode == "auto"
    assert args.dataset_name == "370_dials"
    assert args.drop_dont_integrate is False
    assert args.keep_dont_integrate is False


def test_structure_update_strategy_can_force_shelxl_feedback():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--structure-update",
            "shelxl_feedback",
        ]
    )
    assert resolve_structure_update_strategy(args) == "shelxl_feedback"
    assert resolve_structure_update_strategy(args, multi_dataset=True) == "shelxl_feedback"


def test_shelxl_feedback_current_model_skips_empty_res(tmp_path: Path):
    run_root = tmp_path / "feedback"
    good_run = run_root / "cycle_04_N132Q20"
    bad_run = run_root / "cycle_05_N132Q20"
    good_res = good_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    bad_res = bad_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    good_res.parent.mkdir(parents=True)
    bad_res.parent.mkdir(parents=True)
    good_res.write_text("TITL ok\nCELL 0.02508 5.0 6.0 7.0 90 90 90\nSFAC C\nEND\n")
    bad_res.write_text("")
    (run_root / "shelxl_feedback_loop_summary.json").write_text(
        json.dumps(
            {
                "current_model": str(bad_res),
                "runs": [
                    {"run_dir": str(good_run), "returncode": 0, "output_model": str(good_res)},
                    {
                        "run_dir": str(bad_run),
                        "returncode": 3,
                        "output_model": str(bad_res),
                        "endpoint_res_health": {"ok": False, "reason": "empty_res"},
                    },
                ],
            }
        )
        + "\n"
    )

    assert cli.shelxl_feedback_current_model(run_root) == good_res
    assert cli.shelxl_feedback_current_run(run_root) == good_run


def test_shelxl_feedback_current_model_skips_npd_res(tmp_path: Path):
    run_root = tmp_path / "feedback"
    good_run = run_root / "cycle_04_N132Q20"
    npd_run = run_root / "cycle_05_N132Q20"
    good_res = good_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    npd_res = npd_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    good_res.parent.mkdir(parents=True)
    npd_res.parent.mkdir(parents=True)
    good_res.write_text("TITL ok\nCELL 0.02508 5.0 6.0 7.0 90 90 90\nSFAC C\nEND\n")
    npd_res.write_text("TITL npd\nCELL 0.02508 5.0 6.0 7.0 90 90 90\nSFAC C\nEND\n")
    (run_root / "shelxl_feedback_loop_summary.json").write_text(
        json.dumps(
            {
                "current_model": str(npd_res),
                "runs": [
                    {"run_dir": str(good_run), "returncode": 0, "output_model": str(good_res), "endpoint_res_health": {"ok": True, "safe_for_feedback": True}},
                    {
                        "run_dir": str(npd_run),
                        "returncode": 0,
                        "output_model": str(npd_res),
                        "endpoint_res_health": {
                            "ok": True,
                            "reason": "ok",
                            "safe_for_feedback": False,
                            "npd_count": 1,
                            "split_count": 0,
                            "npd_atoms": ["O001"],
                        },
                    },
                ],
            }
        )
        + "\n"
    )

    assert cli.shelxl_feedback_current_model(run_root) == good_res
    assert cli.shelxl_feedback_current_run(run_root) == good_run


def test_shelxl_feedback_current_run_prefers_final_row_recompute(tmp_path: Path):
    run_root = tmp_path / "feedback"
    cycle_run = run_root / "cycle_05_N108Q20"
    final_run = run_root / "final_row_recompute_N108Q20"
    final_res = cycle_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    final_res.parent.mkdir(parents=True)
    final_run.mkdir(parents=True)
    final_res.write_text("TITL ok\nCELL 0.02508 5.0 6.0 7.0 90 90 90\nSFAC C\nEND\n")
    (run_root / "shelxl_feedback_loop_summary.json").write_text(
        json.dumps(
            {
                "current_model": str(final_res),
                "final_row_stats_run": str(final_run),
                "final_row_stats_model": str(final_res),
                "runs": [
                    {"run_dir": str(cycle_run), "returncode": 0, "output_model": str(final_res)},
                    {
                        "label": "final_row_recompute",
                        "run_dir": str(final_run),
                        "returncode": 0,
                        "output_model": str(final_res),
                        "row_stats_model": str(final_res),
                        "skip_endpoint_refinement": True,
                    },
                ],
            }
        )
        + "\n"
    )

    assert cli.shelxl_feedback_current_model(run_root) == final_res
    assert cli.shelxl_feedback_current_run(run_root) == final_run


def test_best_endpoint_model_rejects_empty_res(tmp_path: Path):
    run = tmp_path / "native"
    endpoint_dir = run / "endpoint_lam_1p0_B_0p0_noexti"
    endpoint_dir.mkdir(parents=True)
    lst = endpoint_dir / "dyn_nuisance_corrected.lst"
    res = endpoint_dir / "dyn_nuisance_corrected.res"
    lst.write_text("dummy\n")
    res.write_text("")
    (run / "summary.json").write_text(json.dumps({"best_endpoint": {"shelxl": {"lst": str(lst)}}}) + "\n")

    assert cli.best_endpoint_model(run) is None


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
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"])

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
    assert "--dataset-sigma-holdout-floor" in cmd


def test_auto_parser_accepts_cuda_jax_backend_explicitly():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--compute-backend",
            "jax_cuda_padded",
            "--jax-batch-size",
            "192",
            "--jax-matrix-bucket-size",
            "8",
        ]
    )
    assert args.compute_backend == "jax_cuda_padded"
    assert args.jax_batch_size == 192
    assert args.jax_matrix_bucket_size == 8


def test_auto_parser_accepts_mps_real_expm_backend_explicitly():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--compute-backend",
            "jax_mps_real_expm_padded",
            "--jax-batch-size",
            "32",
            "--jax-matrix-bucket-size",
            "8",
        ]
    )
    assert args.compute_backend == "jax_mps_real_expm_padded"
    assert args.jax_batch_size == 32
    assert args.jax_matrix_bucket_size == 8


def test_auto_parser_accepts_bethe_backend_controls():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--compute-backend",
            "scipy_bethe",
            "--bethe-strong-count",
            "64",
            "--bethe-variant",
            "schur_target",
            "--bethe-promote-denom",
            "0.004",
            "--bethe-denom-eps",
            "1e-6",
        ]
    )
    assert args.compute_backend == "scipy_bethe"
    assert args.bethe_strong_count == 64
    assert args.bethe_variant == "schur_target"
    assert args.bethe_promote_denom == 0.004
    assert args.bethe_denom_eps == 1e-6


def test_raw_polish_parser_accepts_bethe_backend_controls():
    args = build_parser().parse_args(
        [
            "raw-polish",
            "--hkl",
            "INTEGRATE.HKL",
            "--structure",
            "model.res",
            "--out-dir",
            "raw",
            "--compute-backend",
            "scipy_bethe",
            "--bethe-strong-count",
            "64",
            "--bethe-variant",
            "schur_target",
        ]
    )
    assert args.command == "raw-polish"
    assert args.compute_backend == "scipy_bethe"
    assert args.bethe_strong_count == 64
    assert args.bethe_variant == "schur_target"


def test_full_fast_parser_exposes_middle_profile():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res", "--full-fast"])
    assert args.command == "auto"
    assert args.mode == "auto"
    assert args.fast is False
    assert args.full_fast is True
    assert args.full_fast_n_basis == 24
    assert args.full_fast_n_points == 10
    assert args.full_fast_structure_rows == 0
    assert args.full_fast_structure_max_nfev == 0


def test_auto_parser_exposes_endpoint_instability_guard():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--instability-u",
            "0.02",
        ]
    )
    assert args.command == "auto"
    assert args.instability_u == 0.02


def test_full_fast_profile_pins_cheap_automatic_work():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res", "--full-fast"])
    profile = apply_speed_profile(args)
    assert profile["profile"] == "full_fast"
    assert args.screen_n_basis == 24
    assert args.screen_n_points == 10
    assert args.transform_rerank_n_basis == 24
    assert args.transform_rerank_n_points == 10
    assert args.n_points == 10
    assert args.structure_objective_rows == 250
    assert args.structure_max_nfev == 2
    assert args.shelxl_iterations == 1
    assert args.endpoint_feedback == "off"
    assert args.observation_polish == "off"
    assert args.forensic_depth == "quick"


def test_full_fast_profile_can_lower_transform_rerank_cost():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--full-fast",
            "--full-fast-n-basis",
            "10",
            "--full-fast-n-points",
            "5",
        ]
    )
    apply_speed_profile(args)
    assert args.transform_rerank_n_basis == 10
    assert args.transform_rerank_n_points == 5


def test_publication_report_includes_scale_polish_and_resolution_summaries(tmp_path: Path):
    prep = PrepConfig(structure=str(tmp_path / "model.res"), datasets=[DatasetSpec(name="111", hkl="INTEGRATE.HKL")])
    result = {
        "primary_run": str(tmp_path / "runs" / "111" / "final"),
        "model": prep.structure,
        "scale_polish": {"status": "accepted", "out_dir": str(tmp_path / "scale_polish" / "111")},
        "resolution_limits": {"111": {"artifacts": {"summary": str(tmp_path / "resolution_limits" / "111" / "summary.json")}}},
    }

    cmd = publication_report_cmd("python", prep, result, tmp_path / "tables", "table")

    scale_index = cmd.index("--scale-polish")
    assert cmd[scale_index + 1].endswith("/scale_polish/111/scale_compression_summary.json")
    assert cmd[scale_index + 1].startswith("111=")
    resolution_index = cmd.index("--resolution-limit")
    assert cmd[resolution_index + 1].endswith("/resolution_limits/111/summary.json")
    assert cmd[resolution_index + 1].startswith("111=")


def test_transform_scout_outer_jobs_account_for_workers_per_job():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--workers",
            "12",
            "--transform-scout-workers-per-job",
            "3",
        ]
    )
    assert transform_scout_job_count(args, task_count=20) == 4


def test_transform_scout_outer_jobs_can_be_capped_explicitly():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--workers",
            "64",
            "--transform-scout-jobs",
            "8",
        ]
    )
    assert transform_scout_job_count(args, task_count=20) == 8
    assert transform_scout_job_count(args, task_count=3) == 3


def test_full_fast_profile_uses_lighter_structure_budget_for_larger_models(tmp_path: Path):
    model = tmp_path / "model.res"
    atoms = "\n".join(f"C{i:03d} 1 0.{i % 9} 0.{(i + 1) % 9} 0.{(i + 2) % 9} 11.0 0.05" for i in range(52))
    model.write_text(f"TITL test\nCELL 0.025 10 10 10 90 90 90\nSFAC C H\n{atoms}\nEND\n")
    assert estimate_model_nonh_atoms(str(model)) == 52
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", str(model), "--full-fast"])
    profile = apply_speed_profile(args, str(model))
    assert profile["structure_budget"]["non_h_atoms"] == 52
    assert args.structure_objective_rows == 160
    assert args.structure_max_nfev == 1


def test_raw_polish_parser_exposes_forward_model_knobs():
    args = build_parser().parse_args(
        [
            "raw-polish",
            "--out-dir",
            "out",
            "--hkl",
            "INTEGRATE.HKL",
            "--structure",
            "model.res",
            "--cbf-dir",
            "cbf",
            "--no-maps",
        ]
    )
    assert args.raw_integration_modes == "sum,bloch_profile_fit"
    assert args.matrix_conventions == "j_minus_i,i_minus_j"
    assert args.prediction_cache_compression == "none"
    assert args.make_maps is False


def test_compile_command_uses_temporary_bytecode_cache(monkeypatch, capsys):
    args = build_parser().parse_args(["compile"])
    seen = {}

    def fake_compile_dir(root, quiet, force, workers):
        seen["compile_prefix"] = cli.sys.pycache_prefix
        return True

    def fake_run(cmd, check, env):
        seen["run_prefix"] = env.get("PYTHONPYCACHEPREFIX")
        seen["cmd"] = cmd

    monkeypatch.setattr(cli.compileall, "compile_dir", fake_compile_dir)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "choose_python", lambda requested: "python-x")
    monkeypatch.setattr(cli, "external_tool_statuses", lambda args: {})

    assert cli.cmd_compile(args) == 0

    output = capsys.readouterr().out
    assert "compile_check_passed" in output
    assert "temporary" in output
    assert seen["compile_prefix"]
    assert seen["compile_prefix"] == seen["run_prefix"]
    assert seen["cmd"][0] == "python-x"


def test_tool_config_applies_shelxl_paths(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SHELXL", raising=False)
    monkeypatch.delenv("SHELX2MAP", raising=False)
    fake_shelxl = tmp_path / "shelxl"
    fake_shelx2map = tmp_path / "shelx2map"
    fake_shelxl.write_text("#!/bin/sh\n")
    fake_shelx2map.write_text("#!/bin/sh\n")
    cfg = tmp_path / "grond.config.json"
    cfg.write_text(
        '{\n'
        '  "tools": {\n'
        f'    "shelxl": "{fake_shelxl}",\n'
        f'    "shelx2map": "{fake_shelx2map}"\n'
        "  }\n"
        "}\n"
    )
    args = build_parser().parse_args(
        ["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res", "--config", str(cfg)]
    )
    apply_tool_config(args)
    assert args.shelxl == str(fake_shelxl)
    assert os.environ["SHELXL"] == str(fake_shelxl)
    assert os.environ["SHELX2MAP"] == str(fake_shelx2map)


def test_missing_map_tool_errors_before_full_auto(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("SHELXL", raising=False)
    monkeypatch.delenv("SHELX2MAP", raising=False)
    fake_shelxl = tmp_path / "shelxl"
    fake_shelxl.write_text("#!/bin/sh\n")
    cfg = tmp_path / "grond.config.json"
    cfg.write_text('{"tools": {"shelxl": "' + str(fake_shelxl) + '"}}\n')
    model = tmp_path / "model.res"
    model.write_text("TITL test\nEND\n")
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            str(model),
            "--config",
            str(cfg),
        ]
    )
    apply_tool_config(args)
    errors = required_external_tool_errors(args, structure=str(model))
    assert any("shelx2map" in item for item in errors)
    args.make_maps = False
    assert required_external_tool_errors(args, structure=str(model)) == []
