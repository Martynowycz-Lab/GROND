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
from grond.io.config import DatasetSpec, PrepConfig

EXPECTED_CLI_COMMANDS = {
    "auto",
    "calibrate-endpoint",
    "compile",
    "doctor",
    "endpoints",
    "export-mtz",
    "forensics",
    "import-dials",
    "init-config",
    "make-free-flags",
    "prep",
    "protein-ls1",
    "push-endpoint",
    "raw-polish",
    "row-template",
    "row-validate",
    "run",
    "scale-polish",
    "scale-replay",
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
    assert args.endpoint_calibration_sigma_min_scale == 0.05
    assert args.endpoint_calibration_shelxl_iterations == 1
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
    assert calibrate.shelxl_iterations == 1
    assert calibrate.post_bloch_sigma_min_scale == 0.05
    assert calibrate.publication_goof_min == 0.85
    assert calibrate.publication_free_wr2_tolerance == 0.040
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


def test_auto_parser_is_simple_run_alias():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"])
    assert args.command == "auto"
    assert args.mode == "auto"
    assert args.resume is True
    assert args.detector_scale_bins == "8"
    assert args.final_basis_max == 500
    assert args.scale_polish_detector_bins == "exact"


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
