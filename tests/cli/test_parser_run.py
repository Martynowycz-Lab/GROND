"""CLI parser run regression tests."""

import argparse

import pytest

import grond.cli as cli
from grond.cli import (
    build_parser,
    parse_numeric_grid,
)
from tests.cli._support import cli_command_names

EXPECTED_CLI_COMMANDS = {
    "apply-diffbloch-model",
    "apply-parity-calibration",
    "audit-hydrogen-omit",
    "auto",
    "calibrate-endpoint",
    "canonicalize-hand",
    "compile",
    "create-parity-calibration",
    "doctor",
    "endpoints",
    "export-pets",
    "export-mtz",
    "finish",
    "forensics",
    "hand-check",
    "import-dials",
    "import-pets",
    "init-config",
    "make-free-flags",
    "manual",
    "never-h",
    "pets-evaluate",
    "prep",
    "protein-ls1",
    "publish",
    "push-endpoint",
    "raw-polish",
    "row-template",
    "row-validate",
    "run",
    "scout",
    "scale-polish",
    "scale-replay",
    "sfac-check",
    "thickness-scout",
    "thickness",
    "refine",
    "validate",
    "validate-ls1",
}


def test_all_registered_subcommands_have_help(capsys):
    parser = build_parser()
    assert cli_command_names(parser) == EXPECTED_CLI_COMMANDS

    for command in sorted(EXPECTED_CLI_COMMANDS):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([command, "--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out
        canonical = {"refine": "push-endpoint", "thickness": "thickness-scout"}.get(
            command, command
        )
        assert canonical in captured.out


def test_run_parser_exposes_staged_feature_grids():
    args = build_parser().parse_args(["run", "--out-dir", "out", "--hkl", "INTEGRATE.HKL", "--structure", "model.res"])
    assert args.sigma_b_corr_grid == "0.0"
    assert args.sg_jitter_sigma_frame_grid == "0.0"
    assert args.absorption_mode_grid == "none,midgley_like"
    assert args.absorption_mu_theta2_grid == "0.0"
    assert args.absorption_tds_b_grid == "0.0"
    assert args.dose_b_mode_grid == "none"
    assert args.production_scales is False


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
    assert args.gradient_grid == "0.0,0.4,0.8,1.0"
    assert args.gradient_max == 1.0
    assert args.scale_modes == "production,no-detector"
    assert args.feature_modes == "bare,final-diagnostic"
    assert args.local_confirmation is True
    assert args.thickness_scout_final_absorption_mode == "midgley_like"
    assert args.prediction_cache_scope == "shared"
    assert args.prediction_cache_compression == "none"
    assert args.compute_backend == "scipy"
    assert args.row_plan_cache == "auto"
    assert args.jax_batch_size == 0
    assert args.jax_matrix_bucket_size == 0
    assert args.matrix_convention == "j_minus_i"
    assert args.profile_quadrature == "uniform"
    assert args.profile_accumulation == "sum"
    assert args.detector_scale_bins == "0"
    assert args.detector_scale_ridge == 0.05
    assert args.detector_scale_smoothness == 0.10
    assert args.detector_scale_min_rows == 20
    assert args.lambda_grid == "1.0"
    assert args.scale_smoothness == 0.0
    assert args.endpoint_b_grid == "0.0"
    assert args.publication_endpoint_chooser is True
    assert args.publication_endpoint_min_hkl_safe_factor == 0.25
    assert args.publication_endpoint_max_modulation_p95 == 20.0
    assert args.publication_allow_npd is False
    assert args.endpoint_calibration == "auto"
    assert "adaptive:120:det8" in args.endpoint_calibration_specs
    assert args.endpoint_calibration_lambda_q == 1.0
    assert args.endpoint_calibration_wght_modes == "freeze-zero,promote"
    assert args.endpoint_calibration_prefer_wght_mode == "freeze-zero"
    assert args.endpoint_calibration_goof_min == 0.85
    assert args.endpoint_calibration_goof_max == 1.15
    assert args.endpoint_calibration_max_r1_cost == 0.010
    assert args.endpoint_calibration_free_wr2_tolerance == 0.040
    assert args.endpoint_calibration_selection_policy == "gated-smallest"
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
    assert args.transform_parity_policy == "proper-only"
    assert args.transform_metadata_prefilter == "auto"
    assert args.transform_metadata_max_candidates == 3
    assert args.transform_metadata_min_correlation == 0.70


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
    assert args.selection_metric == "work_wR2"
    assert args.continuous_thickness_metric == "work_wR2"
    assert args.n_points == 20
    assert args.max_support_sg == pytest.approx(0.01)
    assert args.max_support_sg == pytest.approx(cli.DEFAULT_MAX_SUPPORT_SG)
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
    gated_hand_args = parser.parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--hkl",
            "INTEGRATE.HKL",
            "--structure",
            "model.res",
            "--hand-max-wR2-all",
            "0.4",
            "--hand-max-free-wR2-all",
            "0.45",
            "--hand-min-free-wR2-gain",
            "0.01",
            "--hand-min-z-all",
            "2.5",
            "--hand-min-z-free",
            "1.0",
            "--hand-min-free-rows",
            "12",
        ]
    )
    assert gated_hand_args.hand_max_wR2_all == 0.4
    assert gated_hand_args.hand_max_free_wR2_all == 0.45
    assert gated_hand_args.hand_min_free_wR2_gain == 0.01
    assert gated_hand_args.hand_min_z_all == 2.5
    assert gated_hand_args.hand_min_z_free == 1.0
    assert gated_hand_args.hand_min_free_rows == 12
    assert args.structure_n_basis == 24
    assert args.scale_polish_frame_models == "adaptive,fixed"
    assert args.scale_polish_adaptive_counts == "auto"
    assert args.scale_polish_detector_bins == "0,2,4,8"
    assert args.scale_polish_objective == "row_mr"
    assert args.scale_polish_selection_metrics == "free_Rall,work_Rall"
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
    assert args.frame_offset_grid == "auto"


def test_auto_parser_is_simple_run_alias():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"])
    assert args.command == "auto"
    assert args.mode == "auto"
    assert args.resume is True
    assert args.detector_scale_bins == "0"
    assert args.final_basis_max == 500
    assert args.scale_polish_detector_bins == "0,2,4,8"
    assert args.make_maps is False


def test_parse_numeric_grid_accepts_commas_semicolons_and_spaces():
    assert parse_numeric_grid("0.003, 0.004;0.006 0.008") == [0.003, 0.004, 0.006, 0.008]


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


def test_auto_parser_exposes_jana_row_weight_instability_factor():
    parser = build_parser()
    args = parser.parse_args(
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
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    auto_parser = subparsers.choices["auto"]
    instability_action = next(action for action in auto_parser._actions if action.dest == "instability_u")
    assert "JANA/PETS-style row-weight instability factor" in instability_action.help
