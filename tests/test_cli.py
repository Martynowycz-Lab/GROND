import argparse
import json
import os
import sys
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
from grond.progress import ProgressState

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


def valid_shelx_res_text(title: str = "ok") -> str:
    """Return a minimal model that is structurally usable by the Bloch parser."""

    return (
        f"TITL {title}\n"
        "CELL 0.02508 5.0 6.0 7.0 90 90 90\n"
        "SFAC C 1 0 1 0 1 0 1 0\n"
        "FVAR 1\n"
        "C1 1 0.10 0.20 0.30 11.0 0.05\n"
        "HKLF 4\n"
        "END\n"
    )


def bound_shelx_health(path: Path, **updates) -> dict:
    """Return a path/hash-bound health record for feedback-resume tests."""

    from grond.legacy.engine.native_row_regularize import shelx_res_health

    lst = path.with_suffix(".lst")
    if not lst.exists():
        lst.write_text("0 atoms may be split and 0 atoms NPD\n")
    health = shelx_res_health(path, lst)
    health.update(updates)
    return health


def combined_endpoint_summary(path: Path, *, status: str = "ok", **health_updates) -> dict:
    """Return a current-format combined endpoint summary bound to ``path``."""

    health = bound_shelx_health(path, **health_updates)
    return {
        "combined_endpoint": {
            "lambda_q": "1.0",
            "mod_p95": 1.0,
            "modulation_metric": "scale_invariant_symmetric_fold_change_p95",
            "artifacts": {
                "res": str(path),
                "res_sha256": health.get("sha256", ""),
            },
            "shelxl": {
                "status": status,
                "res_health": health,
            },
        }
    }


def test_run_logged_writes_concise_failure_artifact(tmp_path: Path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    monkeypatch.setattr(cli, "_PROGRESS", None)
    command = [cli.sys.executable, "-c", "print('decisive failure'); raise SystemExit(7)"]

    with pytest.raises(cli.subprocess.CalledProcessError):
        cli.run_logged(command, logs, "broken_stage", cwd=tmp_path)

    failure = json.loads((logs / "failure.json").read_text())
    assert failure["label"] == "broken_stage"
    assert failure["returncode"] == 7
    assert any("decisive failure" in line for line in failure["log_tail"])
    output = capsys.readouterr().out
    assert "FAILED broken_stage" in output
    assert str(logs / "broken_stage.log") in output
    assert "rerun the same GROND command" in output


def test_run_logged_preserves_subprocess_error_when_failure_artifact_write_fails(
    tmp_path: Path, monkeypatch, capsys
):
    logs = tmp_path / "logs"
    monkeypatch.setattr(cli, "_PROGRESS", None)
    command = [cli.sys.executable, "-c", "raise SystemExit(9)"]
    original_write_text = cli.Path.write_text

    def write_text(self, *args, **kwargs):
        if self.name.endswith("failure.json"):
            raise OSError("simulated full disk")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(cli.Path, "write_text", write_text)

    with pytest.raises(cli.subprocess.CalledProcessError) as caught:
        cli.run_logged(command, logs, "broken_artifact_stage", cwd=tmp_path)

    assert caught.value.returncode == 9
    output = capsys.readouterr().out
    assert "could not write failure artifact" in output
    assert "FAILED broken_artifact_stage" in output


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


def test_expert_stage_parsers_expose_safe_handoffs_and_aliases():
    parser = build_parser()
    scout = parser.parse_args(
        ["scout", "--prep", "prep.json", "--out-dir", "01_scout"]
    )
    assert scout.transform_parity_policy == "proper-only"
    assert scout.trust_hkl_transform is False

    hand = parser.parse_args(
        [
            "hand-check",
            "--source-summary",
            "summary.json",
            "--out-dir",
            "03_hand",
        ]
    )
    assert hand.hand_check_method == "negative-thickness"
    assert hand.hand_check_absorption == "elastic"
    assert hand.hand_check_n_basis == 0
    assert hand.hand_check_n_points == 0

    thickness = parser.parse_args(
        [
            "thickness",
            "--prep",
            "prep.json",
            "--out-dir",
            "02_thickness",
        ]
    )
    assert thickness.command == "thickness"
    assert thickness.func is cli.cmd_thickness_scout

    refine = parser.parse_args(
        [
            "refine",
            "--source-summary",
            "summary.json",
            "--start-res",
            "model.res",
            "--out-dir",
            "04_refine",
        ]
    )
    assert refine.command == "refine"
    assert refine.func is cli.cmd_push_endpoint


def test_target_ablation_is_hidden_from_public_parser():
    assert "target-ablation" not in cli_command_names(build_parser())


def test_target_ablation_parser_defaults_to_four_xds_contracts(monkeypatch):
    monkeypatch.setenv("GROND_ENABLE_INTERNAL_CLI", "1")
    args = build_parser().parse_args(
        [
            "target-ablation",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--baseline-hkl",
            "model.hkl",
            "--out-dir",
            "ablation",
        ]
    )
    assert args.command == "target-ablation"
    assert args.variants == "iobs,iobs_over_rlp,iobs_times_peak,recorded_proxy"
    assert args.shelxl_feedback_cycles == 5
    assert args.shelxl_feedback_shelxl_iterations == 1
    assert args.endpoint_ls_cycles == 1
    assert args.hand_control is True
    assert args.make_maps is False
    assert args.resume is False
    assert args.min_peak == 90.0
    assert args.row_plan_cache == "auto"


def test_target_ablation_passes_negative_signed_transform_as_one_argument(monkeypatch, tmp_path):
    monkeypatch.setenv("GROND_ENABLE_INTERNAL_CLI", "1")
    parser = build_parser()
    args = parser.parse_args(
        [
            "target-ablation",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--baseline-hkl",
            "model.hkl",
            "--out-dir",
            "ablation",
            "--hkl-transform=-h,k,-l",
        ]
    )
    command = cli._target_ablation_auto_command(args, "recorded_proxy", tmp_path)
    assert "--hkl-transform=-h,k,-l" in command
    inner = parser.parse_args(command[3:])
    assert inner.command == "auto"
    assert inner.hkl_transform == "-h,k,-l"


def test_finish_parser_defaults_to_existing_run_dir():
    args = build_parser().parse_args(["finish", "--run-dir", "grond_out", "--no-make-maps"])
    assert args.command == "finish"
    assert args.run_dir == "grond_out"
    assert args.checkpoint == ""
    assert args.make_maps is False
    assert args.map_mode == "native"


def test_publication_endpoint_chooser_includes_calibrated_replay(tmp_path: Path):
    run = tmp_path / "grond"
    native_run = run / "runs" / "dataset" / "final"
    endpoint = run / "endpoint_calibration" / "dataset" / "adaptive_80_det_8" / "endpoint_freeze-zero"
    native_run.mkdir(parents=True)
    endpoint.mkdir(parents=True)
    endpoint_res = endpoint / "dyn_scale_replay.res"
    endpoint_hkl = endpoint / "dyn_scale_replay.hkl"
    endpoint_lst = endpoint / "dyn_scale_replay.lst"
    endpoint_res.write_text("TITL calibrated\nEND\n")
    endpoint_hkl.write_text("   0   0   0    0.00    0.00\n")
    endpoint_lst.write_text(
        "wR2 = 0.1115, GooF = S = 0.906 for all data\n"
        "R1 = 0.05 for 1 Fo > 4sig(Fo) and 0.1105 for all 2 data\n"
        "0 atoms may be split and 0 atoms NPD\n"
    )
    endpoint_manifest = endpoint / "manifest.json"
    endpoint_manifest.write_text(
        json.dumps(
            {
                "mod_p95": 1.2,
                "modulation_metric": "scale_invariant_symmetric_fold_change_p95",
                "endpoint_b": 0.0,
                "keep_exti": False,
                "artifacts": {
                    "res_sha256": cli.sha256_file(endpoint_res),
                    "hkl_sha256": cli.sha256_file(endpoint_hkl),
                    "lst_sha256": cli.sha256_file(endpoint_lst),
                },
            }
        )
        + "\n"
    )
    result = {
        "primary_run": str(native_run),
        "endpoint_calibration": {
            "lambda_q": 1.0,
            "publication_selection": {
                "status": "passed",
                "publication_endpoint": {
                    "spec": "adaptive_80_det_8",
                    "wght_mode": "freeze-zero",
                    "scale_params": 144,
                    "shelxl_R1gt": 0.05,
                    "shelxl_R1all": 0.1105,
                    "shelxl_wR2all": 0.1115,
                    "shelxl_GooF": 0.906,
                    "shelxl_Ngt": 1,
                    "shelxl_Nall": 2,
                    "shelxl_npd_count": 0,
                    "shelxl_split_count": 0,
                    "hkl_safe_factor": 1.0,
                    "mod_p95": 1.2,
                    "endpoint_b": 0.0,
                    "keep_exti": False,
                    "endpoint_dir": str(endpoint),
                    "endpoint_res": str(endpoint_res),
                    "endpoint_hkl": str(endpoint_hkl),
                    "endpoint_lst": str(endpoint_lst),
                    "endpoint_manifest": str(endpoint_manifest),
                }
            },
        },
    }
    args = argparse.Namespace(
        out_dir=str(run),
        dry_run=False,
        publication_endpoint_chooser=True,
        publication_endpoint_min_hkl_safe_factor=0.25,
        publication_endpoint_max_modulation_p95=20.0,
        endpoint_calibration_goof_min=0.85,
        endpoint_calibration_goof_max=1.15,
        publication_allow_npd=False,
    )
    prep = PrepConfig(structure=str(tmp_path / "model.res"), datasets=[DatasetSpec(name="dataset", hkl="input.hkl")])

    choice = cli.run_publication_endpoint_chooser(prep, result, args)

    assert choice["status"] == "ok"
    assert choice["best_physical_endpoint"]["source_kind"] == "endpoint_calibration"
    assert choice["best_physical_endpoint"]["R1_all"] == pytest.approx(0.1105)
    assert choice["best_physical_endpoint"]["res"] == str(endpoint_res)
    assert choice["best_physical_endpoint"]["modulation_p95"] == pytest.approx(1.2)
    assert choice["best_physical_endpoint"]["res_sha256"] == cli.sha256_file(endpoint_res)


def test_publication_endpoint_chooser_treats_combined_calibration_as_combined(tmp_path: Path):
    run = tmp_path / "grond"
    combined_run = run / "runs" / "combined"
    endpoint = run / "endpoint_calibration" / "combined" / "adaptive_80_det_8" / "endpoint_freeze-zero"
    combined_run.mkdir(parents=True)
    endpoint.mkdir(parents=True)
    result = {
        "combined": str(combined_run),
        "publication_runs": {},
        "endpoint_calibration": {
            "label": "combined",
            "lambda_q": 1.0,
            "publication_selection": {
                "status": "passed",
                "publication_endpoint": {
                    "spec": "adaptive_80_det_8",
                    "shelxl_R1gt": 0.06,
                    "shelxl_R1all": 0.11,
                    "shelxl_wR2all": 0.12,
                    "shelxl_GooF": 0.95,
                    "hkl_safe_factor": 1.0,
                    "endpoint_dir": str(endpoint),
                },
            },
        },
    }
    args = argparse.Namespace(
        out_dir=str(run),
        dry_run=False,
        publication_endpoint_chooser=True,
        publication_endpoint_min_hkl_safe_factor=0.25,
        publication_endpoint_max_modulation_p95=20.0,
        endpoint_calibration_goof_min=0.85,
        endpoint_calibration_goof_max=1.15,
        publication_allow_npd=False,
    )
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="a", hkl="a.hkl"), DatasetSpec(name="b", hkl="b.hkl")],
    )

    choice = cli.run_publication_endpoint_chooser(prep, result, args)

    assert choice["status"] == "diagnostic_only"
    assert choice["best_physical_endpoint"] == {}
    diagnostic = choice["best_diagnostic_endpoint"]
    assert diagnostic["source_kind"] == "combined_endpoint_calibration"
    assert diagnostic["combined_baseline_gate"]["status"] == "warning_baseline_unavailable"
    assert diagnostic["combined_baseline_gate"]["passed"] is True
    assert choice["best_single_dataset_physical_endpoint"] == {}


def test_publication_endpoint_chooser_rejects_unpassed_or_unsafe_calibration(tmp_path: Path):
    endpoint = {
        "spec": "adaptive_80_det_8",
        "shelxl_R1gt": 0.06,
        "shelxl_R1all": 0.11,
        "shelxl_wR2all": 0.12,
        "shelxl_GooF": 0.95,
    }
    failed = {
        "endpoint_calibration": {
            "publication_selection": {"status": "no_candidate_passed_all_gates", "publication_endpoint": endpoint}
        }
    }
    passed_without_safety = {
        "endpoint_calibration": {"publication_selection": {"status": "passed", "publication_endpoint": endpoint}}
    }

    assert cli._endpoint_calibration_chooser_candidate(failed) == {}
    assert cli._endpoint_calibration_chooser_candidate(passed_without_safety)["hkl_safe_factor"] == 0.0


def test_publication_model_candidate_preserves_hash_bound_hkl_safety(tmp_path: Path):
    import hashlib

    combined = tmp_path / "cycle" / "combined"
    endpoint_dir = combined / "combined_endpoint_lam_1p0"
    endpoint_dir.mkdir(parents=True)
    res = endpoint_dir / "multi_dyn_corrected.res"
    lst = endpoint_dir / "multi_dyn_corrected.lst"
    hkl = endpoint_dir / "multi_dyn_corrected.hkl"
    res.write_text("TITL endpoint\nCELL 0.025079 5 6 7 90 90 90\nSFAC C\nC1 1 0 0 0 1 0.05\nEND\n")
    lst.write_text(
        "WGHT 0 0\n"
        "wR2 = 0.1000, GooF = S = 1.000 for all data\n"
        "R1 = 0.0500 for 1 Fo > 4sig(Fo) and 0.0800 for all 2 data\n"
        "0 atoms may be split and 0 atoms NPD\n"
    )
    hkl.write_text("0 0 0 0 0\n")
    digest = hashlib.sha256(res.read_bytes()).hexdigest()
    hkl_digest = hashlib.sha256(hkl.read_bytes()).hexdigest()
    lst_digest = hashlib.sha256(lst.read_bytes()).hexdigest()
    (combined / "combined_summary.json").write_text(
        json.dumps(
            {
                "combined_endpoint": {
                    "hkl_safe_factor": 0.10,
                    "endpoint_b": 0.0,
                    "keep_exti": False,
                    "artifacts": {
                        "res": str(res),
                        "res_sha256": digest,
                        "hkl": str(hkl),
                        "hkl_sha256": hkl_digest,
                        "lst": str(lst),
                        "lst_sha256": lst_digest,
                    },
                    "shelxl": {
                        "R1_gt": 0.05,
                        "R1_all": 0.08,
                        "wR2_all": 0.10,
                        "GooF": 1.0,
                        "lst": str(lst),
                        "res_health": {"path": str(res), "sha256": digest, "lst": str(lst)},
                    },
                }
            }
        )
    )

    candidate = cli._publication_model_endpoint_candidate(
        {"combined": str(combined), "model": str(res)}
    )

    assert candidate["res_provenance_ok"] is True
    assert candidate["hkl_safe_factor"] == pytest.approx(0.10)
    assert candidate["wght_mode"] is None
    assert candidate["pathology_status"] == "complete"
    assert candidate["shelxl"]["pathology_status"] == "complete"
    assert candidate["shelxl"]["active_wght_a"] == 0.0
    assert candidate["shelxl"]["active_wght_b"] == 0.0
    normalized = cli._normalize_publication_endpoint_block(
        candidate,
        selection_source="endpoint_chooser",
    )
    assert normalized["wght_mode"] == "freeze-zero"
    assert normalized["pathology_status"] == "complete"
    assert normalized["shelxl_npd_count"] == 0
    assert normalized["shelxl_split_count"] == 0

    res.write_text(res.read_text() + "REM mutated\n")
    stale = cli._publication_model_endpoint_candidate(
        {"combined": str(combined), "model": str(res)}
    )
    assert stale["res_provenance_ok"] is False
    assert stale["hkl_safe_factor"] == 0.0


def test_multi_feedback_endpoint_candidates_include_adopted_cycles(tmp_path: Path, monkeypatch):
    run = tmp_path / "grond"
    summary = run / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "cycles": [
                    {
                        "cycle": 1,
                        "output_model": str(tmp_path / "cycle_01.res"),
                        "combined": str(tmp_path / "cycle_01_combined"),
                        "adopted_for_feedback": True,
                        "endpoint_modulation_p95": 1.25,
                        "endpoint_modulation_metric": "fold_change_p95",
                    },
                    {
                        "cycle": 2,
                        "output_model": str(tmp_path / "cycle_02.res"),
                        "combined": str(tmp_path / "cycle_02_combined"),
                        "adopted_for_feedback": False,
                    },
                ]
            }
        )
    )

    monkeypatch.setattr(
        cli,
        "_publication_model_endpoint_candidate",
        lambda result: {
            "id": "final_ls1_corrected_hkl",
            "model": result["model"],
            "combined": result["combined"],
        },
    )

    candidates = cli._multi_feedback_endpoint_candidates(
        {"combined": str(tmp_path / "final_combined")},
        run,
    )

    assert len(candidates) == 1
    assert candidates[0]["id"] == "shared_ls1_cycle_01"
    assert candidates[0]["source_kind"] == "combined_feedback_cycle"
    assert candidates[0]["modulation_p95"] == pytest.approx(1.25)
    assert candidates[0]["model"] == str(tmp_path / "cycle_01.res")


def test_multi_feedback_endpoint_candidates_ignore_single_dataset_runs(tmp_path: Path):
    assert cli._multi_feedback_endpoint_candidates({"primary_run": "run"}, tmp_path) == []


def test_finish_native_maps_use_publication_runs_for_combined_endpoint(tmp_path: Path):
    run_a = tmp_path / "weighted_runs" / "a"
    run_b = tmp_path / "weighted_runs" / "b"
    run_c = tmp_path / "weighted_runs" / "c_endpoint_only"
    result = {
        "combined": str(tmp_path / "runs" / "combined"),
        "publication_runs": {"b": str(run_b), "a": str(run_a)},
        "combined_input_runs": {"b": str(run_b), "a": str(run_a), "c": str(run_c)},
        "publication_endpoint_chooser": {
            "status": "ok",
            "best_physical_endpoint": {
                "label": "combined",
                "source_kind": "combined_corrected_hkl",
                "GooF": 1.0,
                "R1_all": 0.10,
                "wR2_all": 0.20,
                "lst": str(tmp_path / "runs" / "combined" / "combined_endpoint_lam_1p0" / "multi_dyn_corrected.lst"),
            },
        },
    }

    row_tables = cli._finish_native_row_tables(result)

    assert row_tables == [
        ("a", run_a / "native_dyn_calc.csv"),
        ("b", run_b / "native_dyn_calc.csv"),
    ]


def test_finish_native_model_is_hash_bound_to_frozen_rows_not_endpoint(tmp_path: Path):
    import hashlib

    row_model = tmp_path / "row_model.res"
    endpoint_model = tmp_path / "endpoint_model.res"
    row_model.write_text("TITL row model\n")
    endpoint_model.write_text("TITL endpoint model\n")
    run_a = tmp_path / "weighted" / "a"
    run_b = tmp_path / "weighted" / "b"
    for run in (run_a, run_b):
        run.mkdir(parents=True)
        (run / "native_dyn_calc.csv").write_text("h,k,l\n")
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "inputs": {
                        "res": str(row_model),
                        "res_sha256": hashlib.sha256(row_model.read_bytes()).hexdigest(),
                    }
                }
            )
        )
    result = {
        "combined": str(tmp_path / "combined"),
        "model": str(row_model),
        "publication_runs": {"a": str(run_a), "b": str(run_b)},
        "publication_endpoint_chooser": {
            "status": "ok",
            "best_physical_endpoint": {
                "source_kind": "combined_corrected_hkl",
                "res": str(endpoint_model),
            },
        },
    }
    row_tables = cli._finish_native_row_tables(result)

    selected, contract = cli._finish_native_model_contract(
        result,
        argparse.Namespace(model=""),
        row_tables,
    )

    assert selected == row_model.resolve()
    assert selected != endpoint_model
    assert contract["status"] == "hash_bound_row_model"
    assert len(contract["row_models"]) == 2


def test_finish_native_model_rejects_mutated_row_model(tmp_path: Path):
    row_model = tmp_path / "row_model.res"
    row_model.write_text("TITL changed row model\n")
    run = tmp_path / "weighted" / "a"
    run.mkdir(parents=True)
    row_table = run / "native_dyn_calc.csv"
    row_table.write_text("h,k,l\n")
    (run / "summary.json").write_text(
        json.dumps({"inputs": {"res": str(row_model), "res_sha256": "0" * 64}})
    )

    with pytest.raises(RuntimeError, match="row model hash changed"):
        cli._finish_native_model_contract(
            {"model": str(row_model)},
            argparse.Namespace(model=""),
            [("a", row_table)],
        )


def test_finish_native_model_rejects_mixed_provenance(tmp_path: Path):
    import hashlib

    row_model = tmp_path / "row_model.res"
    row_model.write_text("TITL row model\n")
    digest = hashlib.sha256(row_model.read_bytes()).hexdigest()
    row_tables = []
    for label in ("verified", "legacy"):
        run = tmp_path / label
        run.mkdir()
        table = run / "native_dyn_calc.csv"
        table.write_text("h,k,l\n")
        payload = {"inputs": {"res": str(row_model), "res_sha256": digest}} if label == "verified" else {}
        (run / "summary.json").write_text(json.dumps(payload))
        row_tables.append((label, table))

    with pytest.raises(RuntimeError, match="mixed verified/unverified"):
        cli._finish_native_model_contract(
            {"model": str(row_model)},
            argparse.Namespace(model=""),
            row_tables,
        )


def test_finish_endpoint_maps_replays_list3_as_zero_cycle_list6(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    endpoint = tmp_path / "endpoint"
    endpoint.mkdir()
    res = endpoint / "model.res"
    hkl = endpoint / "model.hkl"
    fcf = endpoint / "model.fcf"
    lst = endpoint / "model.lst"
    res.write_text(
        "TITL endpoint\n"
        "CELL 0.025079 5 6 7 90 90 90\n"
        "L.S. 1\n"
        "LIST 3\n"
        "WGHT 0 0\n"
        "HKLF 4\n"
        "END\n"
    )
    hkl.write_text("   0   0   0    0.00    0.00\n")
    fcf.write_text("data_model\n_shelx_refln_list_code 3\n")
    metrics = (
        "wR2 = 0.2923, GooF = S = 1.074 for all data\n"
        "R1 = 0.1281 for 441 Fo > 4sig(Fo) and 0.2004 for all 1894 data\n"
        "0 atoms may be split and 0 atoms NPD\n"
    )
    lst.write_text(metrics)
    fake_shelxl = tmp_path / "shelxl"
    fake_shelx2map = tmp_path / "shelx2map"
    for executable in (fake_shelxl, fake_shelx2map):
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)

    calls: list[list[str]] = []

    def fake_run(command, *, cwd, text, stdout, stderr):
        calls.append(list(command))
        work = Path(cwd)
        if Path(command[0]) == fake_shelxl:
            stem = command[1]
            assert "L.S. 0" in (work / f"{stem}.ins").read_text()
            assert "LIST 6" in (work / f"{stem}.ins").read_text()
            (work / f"{stem}.fcf").write_text("data_model\n_shelx_refln_list_code 6\n")
            (work / f"{stem}.lst").write_text(metrics)
            return SimpleNamespace(returncode=0, stdout="SHELXL replay ok\n")
        out_map = Path(command[command.index("-o") + 1])
        out_map.write_text("map\n")
        return SimpleNamespace(returncode=0, stdout=f'Map data written to "{out_map}"\n')

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(
        cli,
        "_publication_endpoint_from_result",
        lambda _result: {"endpoint_res": str(res), "endpoint_hkl": str(hkl)},
    )
    args = argparse.Namespace(
        shelxl=str(fake_shelxl),
        shelxl_requested=str(fake_shelxl),
        tool_config={"tools": {"shelx2map": str(fake_shelx2map)}},
    )

    result = cli.run_finish_endpoint_maps({}, args)

    assert result["status"] == "ok"
    assert result["list6_replay"]["status"] == "ok"
    assert result["list6_replay"]["source_list_code"] == 3
    assert result["list6_replay"]["replay_list_code"] == 6
    assert result["list6_replay"]["metrics_unchanged"] is True
    assert len(calls) == 5
    map_calls = calls[1:]
    assert all(call[1:3] == ["-w", "1"] for call in map_calls[:3])
    assert map_calls[3][1:3] == ["-w", "0"]
    assert result["maps"]["fofc"]["shelx2map_weight"] == 1.0
    assert result["maps"]["fofc"]["display_role"] == "primary_weighted"
    assert result["maps"]["fofc_unweighted_diagnostic"]["display_role"] == "diagnostic_unweighted"
    assert all(record["exists"] for record in result["maps"].values())


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


def test_terminal_finish_summary_is_human_readable():
    payload = {
        "status": "ok",
        "out_dir": "/tmp/grond",
        "final_summary": "/tmp/grond/FINAL_SUMMARY.md",
        "quick_stats": "/tmp/grond/QUICK_STATS.md",
        "full_json": "/tmp/grond/grond_summary.json",
        "finished": {
            "completed_stages": 25,
            "completed_this_invocation": 20,
            "reused_completed_stages": 5,
            "estimated_stages": 30,
            "invocation_elapsed_seconds": 300,
            "cumulative_stage_elapsed_seconds": 822,
            "stage_elapsed_seconds": 822,
        },
        "final_hand": "alternate",
        "final_hand_t_style_all": 1.505,
        "final_hand_t_style_all_n": 40,
        "final_hand_t_style_all_df": 39,
        "final_hand_t_style_free": 0.876,
        "final_hand_t_style_free_n": 10,
        "final_hand_t_style_free_df": 9,
        "final_hand_t_style_scope": "single_dataset",
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

    assert "GROND finished successfully in 5m00s wall time for this invocation." in text
    assert "successfully in 13m42s" not in text
    assert (
        "Progress: 25 stages completed; initial estimate was 30; 20 ran in this invocation; "
        "5 reused from prior invocations." in text
    )
    assert "Timing: cumulative completed-stage runtime 13m42s." in text
    assert "concurrent stages overlap" in text
    assert "not workflow wall time" in text
    assert "Hand: alternate (paired t-style all/free = t 1.505 (n 40, df 39) / t 0.876 (n 10, df 9)" in text
    assert "not a calibrated normal Z score" in text
    assert "normal-equivalent confidence" not in text
    assert "Starting SHELXL reference" in text
    assert "R1(F)gt 13.82%, R1(F)all 17.14%, wR2(F^2)all 38.45%, GooF 1.406" in text
    assert "Input dynamical rows: R(I)all 22.13%, wR(I)all 43.41%" in text
    assert "Input matched Born rows: R(I)all 28.00%, wR(I)all 42.00%" in text
    assert "Final dynamical rows: R(I)all 21.30%, wR(I)all 33.10%, free wR(I)all 33.45%" in text
    assert "Final matched Born rows: R(I)all 27.00%, wR(I)all 41.00%, free wR(I)all 40.00%" in text
    assert "Change: R(I)all -0.83 pp, wR(I)all -10.31 pp" in text
    assert "Direct LS1 corrected-HKL candidate: R1(F)gt 11.00%, R1(F)all 16.00%, wR2(F^2)all 22.00%, GooF 0.950" in text
    assert "Recommended model-handoff endpoint: R1(F)gt 14.46%, R1(F)all 17.85%, wR2(F^2)all 25.72%, GooF 1.036" in text


def test_terminal_finish_summary_prefers_jana_pets_rows_and_links_table1():
    payload = {
        "status": "ok",
        "out_dir": "/tmp/grond",
        "table1": "/tmp/grond/TABLE1_FINAL_STATISTICS.md",
        "table1_json": "/tmp/grond/TABLE1_FINAL_STATISTICS.json",
        "input_model": {"row_R1all": 0.99, "row_wR2all": 0.98},
        "final_row_model": {"row_R1all": 0.97, "row_wR2all": 0.96},
        "jana_pets_row_statistics": {
            "born": {
                "Nobs": 20,
                "Nall": 30,
                "Robs": 0.12,
                "Rall": 0.14,
                "wRall": 0.16,
                "MNobs": 18,
                "MNall": 25,
                "MRobs": 0.10,
                "MRall": 0.11,
                "MwRall": 0.13,
            },
            "dynamic": {
                "Nobs": 20,
                "Nall": 30,
                "Robs": 0.07,
                "Rall": 0.08,
                "wRall": 0.09,
                "MNobs": 18,
                "MNall": 25,
                "MRobs": 0.05,
                "MRall": 0.06,
                "MwRall": 0.07,
            },
        },
        "publication_endpoint": {},
        "input_shelxl": {},
        "finished": {"completed_stages": 0, "cumulative_stage_elapsed_seconds": 0},
        "files": {},
    }

    text = cli._format_terminal_finish_summary(payload)

    assert "Final JANA/PETS-style row statistics (computed by GROND)" in text
    assert "Matched Born: Nobs/Nall 20/30, Robs 12.00%, Rall 14.00%, wRall 16.00%" in text
    assert "Bloch dynamical: Nobs/Nall 20/30, Robs 7.00%, Rall 8.00%, wRall 9.00%" in text
    assert "Legacy native-objective row diagnostics" not in text
    assert "No corrected endpoint passed the hard artifact" in text
    assert "Canonical Table 1: /tmp/grond/TABLE1_FINAL_STATISTICS.md" in text


def test_jana_legacy_fallback_requires_complete_core_and_does_not_invent_counts():
    assert cli._quick_final_jana_row_summary({"row_mr": {"dynamic": {"Rall": 0.1}}}) == {}

    legacy = {
        "Nobs": 4,
        "Nall": 10,
        "MNobs": 3,
        "MNall": 7,
        "Robs": 0.09,
        "Rall": 0.10,
        "wRall": 0.11,
        "MRobs": 0.07,
        "MRall": 0.08,
        "MwRall": 0.09,
    }
    summary = cli._quick_final_jana_row_summary(
        {"row_mr": {"definition": {"instability_u": 0.01}, "dynamic": legacy}}
    )
    record = cli._table1_jana_record("legacy", "row", summary["dynamic"], "legacy row_mr")

    assert record["status"] == "available"
    assert record["Ninput"] is None
    assert record["Nexcluded"] is None
    assert record["Nmatched"] is None
    assert record["Nmerge_used"] is None
    assert record["Nmerge_excluded"] is None

    incomplete = cli._table1_jana_record("partial", "row", {"Rall": 0.1}, "legacy row_mr")
    assert incomplete["status"] == "not_available"
    assert "incomplete core JANA/PETS factor family" in incomplete["status_reason"]


def test_terminal_finish_summary_does_not_treat_legacy_stage_sum_as_wall_time():
    text = cli._format_terminal_finish_summary(
        {
            "status": "ok",
            "out_dir": "/tmp/legacy",
            "finished": {
                "completed_stages": 4,
                "estimated_stages": 4,
                "stage_elapsed_seconds": 600,
            },
        }
    )

    assert "GROND finished successfully." in text
    assert "GROND finished successfully in 10m00s" not in text
    assert "Timing: cumulative completed-stage runtime 10m00s." in text
    assert "not workflow wall time" in text


def test_terminal_finish_summary_labels_recorded_workflow_timing_for_finish():
    text = cli._format_terminal_finish_summary(
        {
            "status": "ok",
            "out_dir": "/tmp/grond",
            "finished": {
                "completed_stages": 25,
                "estimated_stages": 25,
                "invocation_elapsed_seconds": 300,
                "cumulative_stage_elapsed_seconds": 822,
            },
            "timing_context": {
                "kind": "finish_replay",
                "finish_invocation_elapsed_seconds": 2,
            },
        }
    )

    assert "GROND finish completed successfully in 0m02s wall time." in text
    assert "recorded workflow wall time 5m00s" in text
    assert "wall time for this invocation" not in text
    assert "cumulative completed-stage runtime 13m42s" in text


def test_quick_progress_info_keeps_saved_metadata_but_recomputes_counts_and_stage_sum(tmp_path: Path):
    (tmp_path / "grond_progress.json").write_text(
        json.dumps(
            {
                "completed": {
                    "old": {"returncode": 0, "elapsed_seconds": 40.0},
                    "new": {"returncode": 0, "elapsed_seconds": 30.0},
                    "failed": {"returncode": 1, "elapsed_seconds": 99.0},
                },
                "finished": {
                    "status": "ok",
                    "snapshot_kind": "completed_invocation",
                    "invocation_elapsed_seconds": 20.0,
                    "completed_stages": 1,
                    "completed_this_invocation": 1,
                    "reused_completed_stages": 0,
                    "failed_stages": 0,
                    "cumulative_stage_elapsed_seconds": 40.0,
                    "stage_elapsed_seconds": 40.0,
                },
            }
        )
        + "\n"
    )

    info = cli._quick_progress_finish_info(tmp_path)

    assert info["status"] == "ok"
    assert info["snapshot_kind"] == "completed_invocation"
    assert info["invocation_elapsed_seconds"] == 20.0
    assert info["completed_stages"] == 2
    assert info["completed_this_invocation"] == 1
    assert info["reused_completed_stages"] == 1
    assert info["failed_stages"] == 1
    assert info["cumulative_stage_elapsed_seconds"] == 70.0
    assert info["stage_elapsed_seconds"] == 70.0
    assert info["cumulative_stage_elapsed"] == "1m10s"
    assert info["stage_elapsed"] == "1m10s"


def test_dry_run_finalization_does_not_require_materialized_row_statistics(tmp_path: Path, monkeypatch):
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="dataset", hkl=str(tmp_path / "INTEGRATE.HKL"))],
    )
    args = argparse.Namespace(out_dir=str(tmp_path / "dry_run"), dry_run=True)

    def unexpected_quick_stats(*_args, **_kwargs):
        raise AssertionError("dry-run finalization must not inspect materialized row tables")

    monkeypatch.setattr(cli, "write_quick_stats", unexpected_quick_stats)

    quick = cli._finalize_quick_stats(prep, {}, args)
    text = cli._format_terminal_finish_summary(
        {"status": "dry_run", "out_dir": str(tmp_path / "dry_run"), "quick_stats": quick}
    )

    assert quick["status"] == "dry_run"
    assert "No endpoint or row-level statistics were materialized" in quick["note"]
    assert "GROND dry run completed successfully" in text
    assert "no scientific stages ran" in text


def test_run_completion_report_failure_does_not_persist_finished(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "run"
    state = ProgressState(root=out_dir, resume=True, total=0)
    state.begin_invocation()
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="dataset", hkl=str(tmp_path / "INTEGRATE.HKL"))],
    )
    args = argparse.Namespace(out_dir=str(out_dir), dry_run=False)

    def fail_quick_stats(*_args, **_kwargs):
        raise RuntimeError("quick stats failed")

    monkeypatch.setattr(cli, "_finalize_quick_stats", fail_quick_stats)

    with pytest.raises(RuntimeError, match="quick stats failed"):
        cli._write_run_completion_artifacts(prep, {}, args)

    assert "finished" not in state.load()
    assert not (out_dir / "grond_summary.json").exists()


def test_run_completion_summary_write_failure_does_not_persist_finished(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "grond_summary.json").mkdir()
    state = ProgressState(root=out_dir, resume=True, total=0)
    state.begin_invocation()
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="dataset", hkl=str(tmp_path / "INTEGRATE.HKL"))],
    )
    args = argparse.Namespace(out_dir=str(out_dir), dry_run=False)
    monkeypatch.setattr(cli, "_finalize_quick_stats", lambda *_args, **_kwargs: {"status": "ok"})

    with pytest.raises(IsADirectoryError):
        cli._write_run_completion_artifacts(prep, {}, args)

    assert "finished" not in state.load()
    assert state.load()["invocation"]["status"] == "running"


def test_publication_report_refreshes_fingerprint_after_parent_enrichment(
    tmp_path: Path,
    monkeypatch,
):
    out_dir = tmp_path / "workflow"
    logs = out_dir / "logs"
    state = ProgressState(root=out_dir, resume=True, total=1)
    structure = tmp_path / "model.res"
    structure.write_text(valid_shelx_res_text())
    primary_run = tmp_path / "primary_run"
    primary_run.mkdir()
    prep = PrepConfig(
        structure=str(structure),
        datasets=[DatasetSpec(name="dataset", hkl=str(tmp_path / "INTEGRATE.HKL"))],
    )
    result = {
        "model": str(structure),
        "primary_run": str(primary_run),
        "resolution_limits": {"dataset": {"status": "accepted", "min_d_spacing": 1.0}},
    }
    args = argparse.Namespace(out_dir=str(out_dir), dry_run=False)
    captured_command: list[str] = []

    def fake_run_logged(command, log_dir, label, *, dry_run=False, cwd=cli.LEGACY_ROOT):
        del dry_run, cwd
        captured_command[:] = command
        table_dir = Path(command[command.index("--out-dir") + 1])
        table_dir.mkdir(parents=True)
        (table_dir / "publication_artifacts.json").write_text(
            json.dumps({"summary": str(table_dir / "row_mr_metrics.json")}) + "\n"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{label}.log"
        log.write_text("completed\n")
        state.mark_completed(label, command, 1.0, 0, log)

    monkeypatch.setattr(cli, "_PROGRESS", state)
    monkeypatch.setattr(cli, "run_logged", fake_run_logged)

    report = cli.run_publication_report(cli.sys.executable, prep, result, args, logs)

    artifact = json.loads((out_dir / "tables" / "publication_artifacts.json").read_text())
    record = state.completed_record("publication_row_mr_tables")
    assert report["status"] == "ok"
    assert artifact["resolution_limits"] == result["resolution_limits"]
    assert record["output_fingerprints_refreshed_at"]
    assert state.should_skip("publication_row_mr_tables", captured_command)


def test_configured_run_invocation_clears_finished_before_fully_skipped_resume(
    tmp_path: Path,
    monkeypatch,
):
    out_dir = tmp_path / "run"
    prior = ProgressState(root=out_dir, resume=True, total=0)
    prior.mark_finished("ok")
    assert prior.load()["finished"]["status"] == "ok"
    monkeypatch.setattr(cli, "_PROGRESS", None)

    cli.configure_progress(
        out_dir / "logs",
        resume=True,
        total=0,
        begin_invocation=True,
    )

    assert cli._PROGRESS is not None
    payload = cli._PROGRESS.load()
    assert "finished" not in payload
    assert payload["invocation"]["status"] == "running"


def test_generic_progress_configuration_preserves_finished_without_run_lifecycle(
    tmp_path: Path,
    monkeypatch,
):
    out_dir = tmp_path / "standalone"
    prior = ProgressState(root=out_dir, resume=True, total=0)
    prior.mark_finished("ok")
    monkeypatch.setattr(cli, "_PROGRESS", None)

    cli.configure_progress(out_dir / "logs", resume=True, total=1)

    assert cli._PROGRESS is not None
    assert cli._PROGRESS.load()["finished"]["status"] == "ok"


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
    assert args.selection_metric == "work_wR2"
    assert args.continuous_thickness_metric == "work_wR2"
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
    assert args.frame_offset_grid == "auto"


def test_fresh_baseline_no_harm_gate_warns_but_keeps_worse_endpoint_available():
    summary = {
        "baseline_exti": {
            "status": "ok",
            "R1_all": 0.1377,
            "npd_count": 0,
            "split_count": 0,
            "lst": "fresh.lst",
            "res_health": {"safe_for_feedback": True, "path": "fresh.res"},
        }
    }

    rejected = cli._endpoint_baseline_no_harm_gate(
        summary,
        {"shelxl_R1all": 0.1404},
    )
    accepted = cli._endpoint_baseline_no_harm_gate(
        summary,
        {"shelxl_R1all": 0.1330},
    )

    assert rejected["status"] == "warning_worse_than_baseline"
    assert rejected["passed"] is True
    assert rejected["no_harm"] is False
    assert rejected["reasons"] == []
    assert any("warning only" in warning for warning in rejected["warnings"])
    assert rejected["delta_R1all"] == pytest.approx(0.0027)
    assert rejected["baseline_res"] == "fresh.res"
    assert accepted["status"] == "passed"
    assert accepted["no_harm"] is True


def test_h_omit_mode_is_explicit_about_single_dataset_only():
    assert cli._validated_h_omit_mode("off", 10) == "off"
    assert cli._validated_h_omit_mode("on", 1) == "on"
    assert cli._validated_h_omit_mode("both", 1) == "on"
    with pytest.raises(ValueError, match="not implemented for multi-dataset"):
        cli._validated_h_omit_mode("on", 2)


def test_quick_baseline_prefers_fresh_final_run_summary(tmp_path: Path):
    structure = tmp_path / "model.res"
    structure.write_text("TITL stale\n")
    structure.with_suffix(".log").write_text(
        "R1 = 0.1000 for 1 Fo > 4sig(Fo) and 0.1537 for all 2 data\n"
    )
    baseline_hkl = tmp_path / "baseline.hkl"
    baseline_hkl.write_text("")
    prep = PrepConfig(
        structure=str(structure),
        datasets=[
            DatasetSpec(
                name="d",
                hkl=str(tmp_path / "INTEGRATE.HKL"),
                baseline_hkl=str(baseline_hkl),
            )
        ],
    )
    summary = {
        "baseline_exti": {
            "status": "ok",
            "R1_gt": 0.1051,
            "R1_all": 0.1377,
            "wR2_all": 0.2684,
            "GooF": 1.166,
            "npd_count": 0,
            "split_count": 0,
            "lst": str(tmp_path / "fresh.lst"),
            "res_health": {
                "safe_for_feedback": True,
                "path": str(tmp_path / "fresh.res"),
            },
        }
    }

    details = cli._quick_baseline_shelxl(prep, summary)

    assert details["R1_all"] == pytest.approx(0.1377)
    assert details["fresh_rerun"] is True
    assert details["baseline_source"] == "final_run_summary.baseline_exti"
    assert details["res"] == str(tmp_path / "fresh.res")


def test_quick_source_shelx_controls_reads_fixed_width_exti_without_space(
    tmp_path: Path,
):
    model = tmp_path / "fixed_width_exti.res"
    model.write_text(
        "\n".join(
            [
                "TITL fixed-width EXTI",
                "WGHT 0.1622 1.034",
                "EXTI33651.195312",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )

    controls = cli._quick_source_shelx_controls(model)

    assert controls["exti"] == "EXTI 33651.195312"
    assert "EXTI 33651.195312" in cli._quick_source_shelx_control_text(controls)


def test_quick_source_shelx_controls_recovers_exti_from_sibling_ins(
    tmp_path: Path,
):
    model = tmp_path / "refined.res"
    model.write_text("WGHT 0.1622 1.034\nHKLF 4\nEND\n")
    model.with_suffix(".ins").write_text(
        "WGHT 0.1622 1.034\nEXTI33651.195312\nHKLF 4\nEND\n"
    )

    controls = cli._quick_source_shelx_controls(model)

    assert controls["exti"] == "EXTI 33651.195312"
    assert controls["wght"] == "WGHT 0.162200 1.034000"


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


def test_effective_thickness_text_reports_mean_and_support():
    text = cli._quick_thickness_text(
        {"thickness_A": 1200.0, "thickness_model": "uniform_gradient", "thickness_grad_frac": 1.0}
    )

    assert text == "120.00 nm mean (1200.0 A); uniform-gradient support 0.00-240.00 nm"


def test_multi_thickness_text_and_table_keep_dataset_contracts_separate():
    contracts = [
        {
            "dataset": "a",
            "scale_count": 17,
            "native_weight": {"a": 0.0, "b": 0.0},
            "nuisance": {
                "thickness_A": 1200.0,
                "thickness_model": "uniform_gradient",
                "thickness_grad_frac": 0.5,
                "frame_offset": 0.0,
                "group_width_deg": 1.0,
                "detector_scale_bins": 8,
                "absorption_mode": "none",
                "absorption_frac": 0.0,
                "sg_jitter_mode": "none",
                "row_resolution_b": 0.0,
                "native_wght_a": 0.0,
                "native_wght_b": 0.0,
            },
        },
        {
            "dataset": "b",
            "scale_count": 23,
            "native_weight": {"a": 0.1, "b": 0.2},
            "nuisance": {
                "thickness_A": 600.0,
                "thickness_model": "uniform_gradient",
                "thickness_grad_frac": 1.0,
                "frame_offset": 0.25,
                "group_width_deg": 0.5,
                "detector_scale_bins": 4,
                "absorption_mode": "midgley_like",
                "absorption_frac": 0.02,
                "sg_jitter_mode": "none",
                "row_resolution_b": 0.1,
                "native_wght_a": 0.1,
                "native_wght_b": 0.2,
            },
        },
    ]

    text = cli._quick_thickness_text({}, {"thickness_A_mean": 900.0}, contracts)
    table = "\n".join(cli._quick_dataset_nuisance_table_lines(contracts))

    assert "a 120.00 nm (uniform_gradient, g=0.500)" in text
    assert "b 60.00 nm (uniform_gradient, g=1.000)" in text
    assert "center range 60.00-120.00 nm" in text
    assert "90.00-90.00 nm" not in text
    assert "120.00 nm; support 60.00-180.00 nm" in table
    assert "60.00 nm; support 0.00-120.00 nm" in table


def test_multi_endpoint_calibration_auto_skips_unsafe_single_table_replay(tmp_path: Path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    for run in (run_a, run_b):
        run.mkdir()
        (run / "native_dyn_calc.csv").write_text("h,k,l\n1,0,0\n")
    combined = tmp_path / "combined"
    combined.mkdir()
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="a", hkl="a.hkl"), DatasetSpec(name="b", hkl="b.hkl")],
    )
    result = {
        "combined": str(combined),
        "combined_input_runs": {"a": str(run_a), "b": str(run_b)},
    }
    args = argparse.Namespace(endpoint_calibration="auto", fast=False, dry_run=False, out_dir=str(tmp_path))

    status = cli.run_endpoint_calibration(prep, result, args, "python", tmp_path / "logs")

    assert status["status"] == "not_applicable_multi_dataset"
    assert status["row_tables"] == {
        "a": str(run_a / "native_dyn_calc.csv"),
        "b": str(run_b / "native_dyn_calc.csv"),
    }
    assert status["row_tables_exist"] == {"a": True, "b": True}
    assert "single-table" in status["reason"]


def test_multi_endpoint_calibration_forced_on_fails_closed(tmp_path: Path):
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="a", hkl="a.hkl"), DatasetSpec(name="b", hkl="b.hkl")],
    )
    result = {"combined": str(tmp_path / "combined")}
    args = argparse.Namespace(endpoint_calibration="on", fast=False, dry_run=False, out_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="not supported for multi-dataset"):
        cli.run_endpoint_calibration(prep, result, args, "python", tmp_path / "logs")


def test_continuous_nuisance_is_materialized_on_full_row_contract(tmp_path: Path, monkeypatch):
    selected = {
        "thickness_A": 875.0,
        "thickness_grad_frac": 0.8,
        "thickness_model": "uniform_gradient",
        "frame_offset": 0.0,
        "group_width_deg": 1.0,
        "detector_scale_bins": 8,
        "row_resolution_b": 0.0,
        "sg_jitter_mode": "none",
        "sg_jitter_sigma0": 0.0,
        "sg_jitter_sigma_d_inv": 0.0,
        "sg_jitter_sigma_frame": 0.0,
        "absorption_mode": "none",
        "absorption_mu": 0.0,
        "absorption_frac": 0.0,
        "dose_b_mode": "none",
        "dose_b_per_frame": 0.0,
        "dose_b_ref_frame": float("nan"),
        "matrix_convention": "j_minus_i",
    }
    continuous = tmp_path / "continuous"
    continuous.mkdir()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    baseline_nuisance = {**selected, "thickness_A": 900.0}

    def summary(nuisance, *, all_wr2, free_wr2, all_r1):
        return {
            "best_native": {
                "nuisance": nuisance,
                "metrics": {
                    "all": {
                        "dyn": {"wR2_all": all_wr2, "R1_all": all_r1, "R1_gt": all_r1, "CC": 0.9},
                        "born": {"R1_gt": 0.2, "CC": 0.9},
                    },
                    "free": {"dyn": {"wR2_all": free_wr2, "n": 10}},
                },
            }
        }

    (baseline / "summary.json").write_text(
        json.dumps(summary(baseline_nuisance, all_wr2=0.12, free_wr2=0.13, all_r1=0.13)) + "\n"
    )
    captured = {}

    def fake_native_stage_cmd(_python, _ds, _structure, _baseline, out_dir, _args, **kwargs):
        captured["out_dir"] = out_dir
        captured["pinned"] = dict(kwargs["pinned"])
        out_dir.mkdir(parents=True)
        (out_dir / "native_dyn_calc.csv").write_text("h,k,l\n1,0,0\n")
        (out_dir / "summary.json").write_text(
            json.dumps(summary(selected, all_wr2=0.10, free_wr2=0.11, all_r1=0.12)) + "\n"
        )
        return ["fake-native"]

    monkeypatch.setattr(cli, "native_stage_cmd", fake_native_stage_cmd)
    monkeypatch.setattr(cli, "run_logged", lambda *_args, **_kwargs: None)
    prep = PrepConfig(structure=str(tmp_path / "model.res"), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    ds = prep.datasets[0]
    args = argparse.Namespace(
        dry_run=False,
        continuous_thickness_min_gain=5e-5,
        continuous_thickness_max_free_wr2_loss=0.002,
        continuous_thickness_max_all_r1_loss=0.001,
    )

    nuisance, run, record = cli.materialize_continuous_nuisance_run(
        prep,
        ds,
        args,
        "python",
        tmp_path / "logs",
        active_model=prep.structure,
        baseline_run=baseline,
        continuous_run=continuous,
        selected_nuisance=selected,
        n_basis=40,
        n_points=10,
        stage="continuous_materialized",
    )

    assert captured["pinned"] == selected
    assert run == continuous / "materialized_full_rows"
    assert (run / "native_dyn_calc.csv").exists()
    assert nuisance["thickness_A"] == 875.0
    assert record["nuisance_contract_verified"] is True
    assert record["status"] == "accepted_full_row"
    assert record["full_row_score_gain"] > 0.0


@pytest.mark.parametrize(
    ("expected", "actual", "matches"),
    [
        (float("nan"), float("nan"), True),
        ("nan", float("nan"), True),
        (float("nan"), 0.0, False),
        (0.0, float("nan"), False),
        (float("inf"), float("inf"), False),
        ("not-a-number", "not-a-number", False),
        (1204.5805095106, 1204.580510, True),
        (1204.5805095106, 1204.6, False),
    ],
)
def test_numeric_nuisance_contract_handles_nan_only_as_matching_na(
    expected: object,
    actual: object,
    matches: bool,
) -> None:
    assert cli.numeric_nuisance_contract_matches(expected, actual) is matches


def test_discrete_thickness_baseline_fails_closed_but_has_dry_run_plan_seed():
    with pytest.raises(RuntimeError, match="no starting nuisance"):
        cli.discrete_thickness_baseline_nuisance(argparse.Namespace(dry_run=False), {})

    assert cli.discrete_thickness_baseline_nuisance(argparse.Namespace(dry_run=True), {}) == {
        "thickness_A": 1200.0,
        "thickness_grad_frac": 0.0,
        "frame_offset": 0.0,
    }


def test_continuous_full_row_gate_restores_coarse_baseline_on_regression(tmp_path: Path, monkeypatch):
    baseline = tmp_path / "coarse"
    continuous = tmp_path / "continuous"
    baseline.mkdir()
    continuous.mkdir()
    baseline_nuisance = {
        "thickness_A": 1200.0,
        "thickness_grad_frac": 0.4,
        "thickness_model": "uniform_gradient",
        "frame_offset": 0.0,
        "group_width_deg": 1.0,
        "detector_scale_bins": 0,
        "matrix_convention": "j_minus_i",
    }
    selected = {**baseline_nuisance, "thickness_A": 700.0, "thickness_grad_frac": 1.0}

    def payload(nuisance, all_wr2, free_wr2, all_r1):
        return {
            "best_native": {
                "nuisance": nuisance,
                "metrics": {
                    "all": {
                        "dyn": {"wR2_all": all_wr2, "R1_all": all_r1, "R1_gt": all_r1, "CC": 0.9},
                        "born": {"R1_gt": 0.2, "CC": 0.9},
                    },
                    "free": {"dyn": {"wR2_all": free_wr2, "n": 10}},
                },
            }
        }

    (baseline / "summary.json").write_text(json.dumps(payload(baseline_nuisance, 0.10, 0.11, 0.12)) + "\n")

    def fake_native_stage_cmd(_python, _ds, _structure, _baseline, out_dir, _args, **_kwargs):
        out_dir.mkdir(parents=True)
        (out_dir / "native_dyn_calc.csv").write_text("h,k,l\n1,0,0\n")
        (out_dir / "summary.json").write_text(json.dumps(payload(selected, 0.11, 0.13, 0.13)) + "\n")
        return ["fake-native"]

    monkeypatch.setattr(cli, "native_stage_cmd", fake_native_stage_cmd)
    monkeypatch.setattr(cli, "run_logged", lambda *_args, **_kwargs: None)
    prep = PrepConfig(structure=str(tmp_path / "model.res"), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    args = argparse.Namespace(
        dry_run=False,
        continuous_thickness_min_gain=5e-5,
        continuous_thickness_max_free_wr2_loss=0.002,
        continuous_thickness_max_all_r1_loss=0.001,
    )

    nuisance, run, record = cli.materialize_continuous_nuisance_run(
        prep,
        prep.datasets[0],
        args,
        "python",
        tmp_path / "logs",
        active_model=prep.structure,
        baseline_run=baseline,
        continuous_run=continuous,
        selected_nuisance=selected,
        n_basis=40,
        n_points=10,
        stage="continuous_materialized",
    )

    assert run == baseline
    assert nuisance["thickness_A"] == 1200.0
    assert record["status"] == "rejected_full_row"
    assert "full_row_gain_below_minimum" in record["promotion_reasons"]
    assert "full_row_free_wR2_loss" in record["promotion_reasons"]


def test_continuous_full_row_gate_requires_heldout_rows(tmp_path: Path):
    baseline = tmp_path / "baseline"
    continuous = tmp_path / "continuous"
    baseline.mkdir()
    continuous.mkdir()
    nuisance = {
        "thickness_A": 900.0,
        "thickness_grad_frac": 0.5,
        "thickness_model": "uniform_gradient",
    }

    def payload(*, all_wr2: float, free_wr2: float, free_n: int) -> dict:
        return {
            "best_native": {
                "nuisance": nuisance,
                "metrics": {
                    "all": {
                        "dyn": {"wR2_all": all_wr2, "R1_all": 0.10, "R1_gt": 0.08, "CC": 0.9},
                        "born": {"R1_gt": 0.2, "CC": 0.9},
                    },
                    "free": {"dyn": {"wR2_all": free_wr2, "n": free_n}},
                },
            }
        }

    (baseline / "summary.json").write_text(json.dumps(payload(all_wr2=0.12, free_wr2=0.13, free_n=10)))
    (continuous / "summary.json").write_text(json.dumps(payload(all_wr2=0.08, free_wr2=0.09, free_n=0)))
    (continuous / "native_dyn_calc.csv").write_text("h,k,l\n1,0,0\n")
    prep = PrepConfig(structure=str(tmp_path / "model.res"), datasets=[DatasetSpec(name="a", hkl="a.hkl")])
    args = argparse.Namespace(
        dry_run=False,
        continuous_thickness_min_gain=5e-5,
        continuous_thickness_max_free_wr2_loss=0.002,
        continuous_thickness_max_all_r1_loss=0.001,
    )

    _selected, selected_run, record = cli.materialize_continuous_nuisance_run(
        prep,
        prep.datasets[0],
        args,
        "python",
        tmp_path / "logs",
        active_model=prep.structure,
        baseline_run=baseline,
        continuous_run=continuous,
        selected_nuisance=nuisance,
        n_basis=40,
        n_points=10,
        stage="continuous_materialized",
    )

    assert selected_run == baseline
    assert record["candidate_free_rows"] == 0
    assert "candidate_missing_positive_free_rows" in record["promotion_reasons"]


def test_missing_materialized_policy_stops_instead_of_falling_back():
    with pytest.raises(RuntimeError, match="refusing to silently fall back"):
        cli.require_materialized_dataset_policy(
            {"status": "missing_row_table", "missing": ["a/native_dyn_calc.csv"]},
            context="structure-update",
            dry_run=False,
        )

    cli.require_materialized_dataset_policy(
        {"status": "missing_row_table", "missing": ["planned/native_dyn_calc.csv"]},
        context="structure-update",
        dry_run=True,
    )


def test_evaluated_dataset_policy_never_resurrects_excluded_runs(tmp_path: Path):
    runs = {"a": tmp_path / "a", "b": tmp_path / "b"}
    policy = {
        "status": "ok",
        "datasets": [
            {"name": "a", "action": "exclude"},
            {"name": "b", "action": "completeness_only"},
        ],
    }

    active = cli.policy_active_runs(runs, policy)

    assert active == {}
    with pytest.raises(RuntimeError, match="no primary or downweighted dataset"):
        cli.require_policy_active_runs(active, context="test")


def test_dataset_policy_returns_only_raw_row_anchor_runs(tmp_path: Path):
    runs = {"a": tmp_path / "a", "b": tmp_path / "b", "c": tmp_path / "c"}
    policy = {
        "status": "ok",
        "datasets": [
            {"name": "a", "action": "primary"},
            {"name": "b", "action": "downweight"},
            {"name": "c", "action": "completeness_only"},
        ],
    }

    active = cli.policy_active_runs(runs, policy)

    assert active == {"a": runs["a"], "b": runs["b"]}
    cli.require_policy_active_runs(active, context="test")


def test_selected_feedback_endpoint_uses_one_hash_bound_cycle(tmp_path: Path):
    cycle = tmp_path / "cycle_02" / "combined"
    endpoint = cycle / "combined_endpoint_lam_1p0"
    endpoint.mkdir(parents=True)
    model = endpoint / "multi_dyn_corrected.res"
    model.write_text("TITL clean\nEND\n")
    lst = model.with_suffix(".lst")
    lst.write_text(
        "wR2 = 0.0874, GooF = S = 0.980 for all data\n"
        "R1 = 0.0731 for 813 Fo > 4sig(Fo) and 0.1752 for all 1917 data\n"
        "     0  atoms may be split and     0  atoms NPD\n"
    )
    digest = cli.sha256_file(model)
    (cycle / "combined_summary.json").write_text(
        json.dumps(
            {
                "combined_endpoint": {
                    "shelxl": {
                        "R1_gt": 0.0741,
                        "R1_all": 0.1773,
                        "wR2_all": 0.0738,
                        "GooF": 0.99,
                        "res_health": {"path": str(model), "sha256": digest},
                    },
                    "artifacts": {"res": str(model), "res_sha256": digest},
                }
            }
        )
        + "\n"
    )
    result = {
        "model": str(model),
        "structure_feedback": {
            "current_model": str(model),
            "cycles": [
                {
                    "cycle": 2,
                    "output_model": str(model),
                    "combined": str(cycle),
                    "adopted_for_feedback": True,
                }
            ],
        },
    }

    details = cli._quick_selected_feedback_shelxl(
        result,
        model,
        {"R1_gt": 0.01, "R1_all": 0.02, "wR2_all": 0.03, "npd_count": 2},
    )

    assert details["R1_gt"] == pytest.approx(0.0741)
    assert details["R1_all"] == pytest.approx(0.1773)
    assert details["wR2_all"] == pytest.approx(0.0738)
    assert details["GooF"] == pytest.approx(0.990)
    assert details["npd_count"] == 0
    assert details["split_count"] == 0


def test_partial_lst_cannot_erase_recorded_npd_pathology(tmp_path: Path):
    lst = tmp_path / "partial.lst"
    lst.write_text(
        "wR2 = 0.0874, GooF = S = 0.980 for all data\n"
        "R1 = 0.0731 for 813 Fo > 4sig(Fo) and 0.1752 for all 1917 data\n"
    )

    details = cli._quick_shelxl_details(
        {"npd_count": 2, "npd_atoms": ["C6", "C7"], "has_npd": True, "split_count": 0},
        lst,
    )

    assert details["R1_all"] == pytest.approx(0.1752)
    assert details["npd_count"] == 2
    assert details["npd_atoms"] == ["C6", "C7"]
    assert details["has_npd"] is True


def test_zero_pathology_summary_cannot_erase_recorded_npd_or_split_warnings(tmp_path: Path):
    lst = tmp_path / "replacement.lst"
    lst.write_text(
        "wR2 = 0.0500, GooF = S = 1.000 for all data\n"
        "R1 = 0.0400 for 100 Fo > 4sig(Fo) and 0.0600 for all 200 data\n"
        "     0  atoms may be split and     0  atoms NPD\n"
    )
    recorded = {
        "R1_all": 0.20,
        "wR2_all": 0.25,
        "npd_count": 2,
        "npd_atoms": ["C6", "C7"],
        "split_count": 1,
        "has_npd": True,
        "has_split_warnings": True,
    }

    details = cli._quick_shelxl_details(recorded, lst)

    assert details["R1_all"] == pytest.approx(0.20)
    assert details["wR2_all"] == pytest.approx(0.25)
    assert details["npd_count"] == 2
    assert details["npd_atoms"] == ["C6", "C7"]
    assert details["split_count"] == 1
    assert details["has_npd"] is True
    assert details["has_split_warnings"] is True


def test_post_feedback_hand_diagnostic_gate_requires_signed_support():
    args = argparse.Namespace(hand_min_z_all=2.0, hand_min_z_free=0.5, hand_min_free_rows=10)
    ok, reasons = cli.hand_diagnostic_supports_other(
        {
            "all": {
                "n": 100,
                "paired_row_t_delta_chi2": -2.4,
                "klar_palatinus_adjusted_hand_z": {"z": 999.0},
            },
            "free": {
                "n": 10,
                "paired_row_t_delta_chi2": -0.8,
                "klar_palatinus_adjusted_hand_z": {"z": 999.0},
            },
        },
        args,
    )
    assert ok is True
    assert reasons == []

    weak, weak_reasons = cli.hand_diagnostic_supports_other(
        {
            "all": {
                "n": 100,
                "paired_row_t_delta_chi2": -0.1,
                "klar_palatinus_adjusted_hand_z": {"z": -999.0},
            },
            "free": {
                "n": 10,
                "paired_row_t_delta_chi2": 0.2,
                "klar_palatinus_adjusted_hand_z": {"z": -999.0},
            },
        },
        args,
    )
    assert weak is False
    assert "all-row paired t-style statistic" in weak_reasons[0]

    too_few, too_few_reasons = cli.hand_diagnostic_supports_other(
        {
            "all": {"n": 100, "paired_row_t_delta_chi2": -4.0},
            "free": {"n": 9, "paired_row_t_delta_chi2": -2.0},
        },
        args,
    )
    assert too_few is False
    assert any("below the minimum 10" in reason for reason in too_few_reasons)


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
    assert calibrate.publication_selection_policy == "gated-smallest"
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
    assert push.compute_backend == "scipy"
    assert push.row_plan_cache == "off"
    assert push.chunksize == 4
    assert push.topup_d_min == 0.7
    assert push.post_bloch_sigma_policy == "none"
    assert push.model_prep == "free-h-anis-no-restraints"
    assert push.set_ls_cycles == 1
    push_adp = build_parser().parse_args(
        [
            "push-endpoint",
            "--source-summary",
            "summary.json",
            "--start-res",
            "model.res",
            "--out-dir",
            "push",
            "--model-prep",
            "adp-stabilized",
        ]
    )
    assert push_adp.model_prep == "adp-stabilized"
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
    assert push_checkpoint.compute_backend == "scipy"
    assert push_checkpoint.row_plan_cache == "off"
    assert push_checkpoint.chunksize == 4
    assert push_checkpoint.topup_d_min == 0.7
    assert push_checkpoint.post_bloch_sigma_policy == "none"


def test_adp_stabilized_model_prep_targets_npd_atoms(tmp_path: Path):
    res = tmp_path / "model.res"
    res.write_text(
        "\n".join(
            [
                "TITL test",
                "CELL 0.025079 7 8 9 90 90 90",
                "ZERR 1 0 0 0 0 0 0",
                "LATT -1",
                "SFAC C O H",
                "UNIT 1 1 1",
                "FVAR 1",
                "C005 1 0.100000 0.200000 0.300000 11.00000 0.00100 0.00200 =",
                " 0.00300 0.00000 0.00000 0.00000",
                "O001 2 0.400000 0.500000 0.600000 11.00000 0.02000",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    res.with_suffix(".lst").write_text(" 0.1 0.2 0.3 C005 ** NON POSITIVE DEFINITE **\n")
    args = argparse.Namespace(start_res=str(res), model_prep="adp-stabilized", set_ls_cycles=1)

    prepared, audit = cli.prepare_push_endpoint_model(args, tmp_path / "push")
    text = prepared.read_text()
    payload = json.loads(audit.read_text())

    assert "RIGU" in text
    assert "SIMU 0.04 0.08" in text
    assert "ISOR 0.02 0.04 C005" in text
    assert any(line.startswith("C005") and "0.00600" in line and "=" not in line for line in text.splitlines())
    assert payload["adp_stabilized_npd_atoms"] == ["C005"]
    assert payload["adp_stabilized_isotropic_fallback"][0]["label"] == "C005"


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


def test_sfac_check_rejects_cif_without_electron_factor_contract(tmp_path: Path):
    model = tmp_path / "ordinary.cif"
    model.write_text(
        "data_test\n"
        "_cell_length_a 10\n"
        "_cell_length_b 10\n"
        "_cell_length_c 10\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
    )

    payload = check_sfac_contract(model)

    assert payload["status"] == "error"
    assert payload["ok"] is False
    assert "electron-scattering model" in payload["messages"][0]["message"]


def test_sfac_check_audits_embedded_shelx_cif(tmp_path: Path):
    model = tmp_path / "embedded.cif"
    model.write_text(
        """data_test
_shelx_res_file
;
TITL embedded
CELL 0.025079 10 10 10 90 90 90
LATT -1
SFAC C 1 0 1 0 1 0 1 0
C1 1 0.1 0.2 0.3 11 0.05
HKLF 4
END
;
"""
    )

    payload = check_sfac_contract(model)

    assert payload["ok"] is True
    assert payload["cif_model_contract"] == "embedded_shelx"
    assert payload["sfac_rows"][0]["kind"] == "numeric_coefficients"


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
        ("SWAT 0.1 1.0\nHKLF 4", "SWAT"),
        ("REM PLATON SQUEEZE applied\nHKLF 4", "SQUEEZE"),
    ]
    for body, expected in cases:
        model = tmp_path / f"{expected.lower()}.res"
        model.write_text(f"TITL bad\nCELL 0.02508 5 6 7 90 90 90\nSFAC C 1 0 1 0 1 0 1 0\n{body}\nEND\n")

        payload = check_sfac_contract(model)

        assert payload["ok"] is False
        assert payload["status"] == "error"
        assert any(expected in item["message"] for item in payload["messages"])


def test_quick_lst_diagnostics_one_term_wght_before_exti(tmp_path: Path):
    lst = tmp_path / "trial.lst"
    lst.write_text("WGHT    0.100000\nEXTI 1320.486328\n")

    payload = cli._quick_lst_diagnostics(lst)

    assert payload["active_wght_a"] == pytest.approx(0.1)
    assert payload["active_wght_b"] == pytest.approx(0.0)
    assert payload["active_wght_terms"] == [pytest.approx(0.1)]


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
        "proper_hand_parity_contract": True,
        "canonical_hand_hklf4_gauge_repair": True,
        "separate_parity_calibration_commands": True,
    }
    assert payload["ready"] is True
    assert payload["ready_scope"] == "core"
    assert payload["failures"] == []
    assert payload["requirements"]["core_python"] == ["numpy", "scipy", "gemmi"]
    assert payload["requirements"]["default_small_molecule_endpoint"] == ["shelxl"]
    assert payload["requirements"]["native_dynamical_maps"] == []
    assert set(payload["tool_config"]["external_tools"]) == {"shelxl", "shelx2map"}
    assert "optional_packages" not in payload
    assert "jax" not in payload


def test_doctor_strict_checks_exact_requested_python(monkeypatch, capsys):
    requested = "/usr/bin/python-without-grond-deps"
    seen = []

    def fake_probe(python):
        seen.append(python)
        return {
            "python": python,
            "returncode": 0,
            "packages": {
                "numpy": {"available": True},
                "scipy": {"available": True},
                "gemmi": {"available": False, "error": "ModuleNotFoundError: gemmi"},
            },
        }

    monkeypatch.setattr(cli, "probe_accelerators", fake_probe)
    args = build_parser().parse_args(["doctor", "--python", requested, "--strict"])

    assert args.func(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert seen == [requested]
    assert payload["ready"] is False
    assert any("required package gemmi is unavailable" in failure for failure in payload["failures"])


def test_doctor_non_strict_remains_report_only(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "probe_accelerators",
        lambda python: {
            "python": python,
            "probe_error": "FileNotFoundError: requested interpreter",
        },
    )
    args = build_parser().parse_args(["doctor", "--python", "/missing/python"])

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert any("dependency probe failed" in failure for failure in payload["failures"])


def test_doctor_small_molecule_scope_requires_shelxl(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "probe_accelerators",
        lambda python: {
            "python": python,
            "returncode": 0,
            "packages": {name: {"available": True} for name in ("numpy", "scipy", "gemmi")},
        },
    )
    monkeypatch.setattr(
        cli,
        "external_tool_statuses",
        lambda args: {
            "shelxl": {"available": False, "error": "not found"},
            "shelx2map": {"available": False, "error": "not found"},
        },
    )
    args = build_parser().parse_args(["doctor", "--workflow", "small-molecule", "--strict"])

    assert args.func(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready_scope"] == "small-molecule"
    assert any("small-molecule workflow requires shelxl" in failure for failure in payload["failures"])


def test_doctor_strict_fails_requested_bloch_operation(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "probe_accelerators",
        lambda python: {
            "python": python,
            "returncode": 0,
            "packages": {name: {"available": True} for name in ("numpy", "scipy", "gemmi")},
        },
    )
    monkeypatch.setattr(
        cli,
        "probe_bloch_ops",
        lambda python: {
            "python": python,
            "returncode": 0,
            "scipy_cpu": {
                "eig": {"ok": False, "error": "LAPACK failure"},
                "eigh": {"ok": True},
            },
        },
    )
    args = build_parser().parse_args(["doctor", "--probe-bloch", "--strict"])

    assert args.func(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert any("SciPy eig Bloch operation failed" in failure for failure in payload["failures"])


def test_auto_parser_is_simple_run_alias():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"])
    assert args.command == "auto"
    assert args.mode == "auto"
    assert args.resume is True
    assert args.detector_scale_bins == "0"
    assert args.final_basis_max == 500
    assert args.scale_polish_detector_bins == "0,2,4,8"
    assert args.make_maps is False


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


def test_experimental_row_plan_cache_flags_parse():
    default_auto = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
        ]
    )
    assert default_auto.row_plan_cache == "auto"
    assert cli.resolved_row_plan_cache(default_auto, fixed_model=True) == "auto"
    assert cli.resolved_row_plan_cache(default_auto, fixed_model=False) == "off"

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
    assert cli.resolved_row_plan_cache(auto, fixed_model=True) == "readwrite"
    assert cli.resolved_row_plan_cache(auto, fixed_model=False) == "readwrite"

    exact = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--row-plan-cache",
            "exact",
        ]
    )
    assert exact.row_plan_cache == "exact"

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
    assert args.dials_row_selection == "integrated"
    assert args.dials_sigma_mode == "raw"
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
        (run / "summary.json").write_text(json.dumps({"best_native": {"nuisance": {"thickness_A": thickness}}}) + "\n")
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

    monkeypatch.setattr(cli, "run_logged", fake_run_logged)
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


def _write_hydrogen_audit_model(
    path: Path,
    *,
    cell_a: float = 10.0,
    include_h: bool,
    symm: str = "",
    coordinate_shift: float = 0.0,
) -> None:
    atoms = [
        f"C1 1 {0.10 + coordinate_shift:.4f} 0.20 0.30 11.0 0.05",
        f"C2 1 {0.25 + coordinate_shift:.4f} 0.35 0.40 11.0 0.05",
        f"C3 1 {0.45 + coordinate_shift:.4f} 0.55 0.60 11.0 0.05",
    ]
    if include_h:
        atoms.append("H1 2 0.15 0.20 0.30 11.0 0.05")
    path.write_text(
        "\n".join(
            [
                "TITL H audit",
                f"CELL 0.02508 {cell_a:g} 11 12 90 90 90",
                "LATT -1",
                *([f"SYMM {symm}"] if symm else []),
                "SFAC C 1 0 1 0 1 0 1 0 0",
                "SFAC H 1 0 1 0 1 0 1 0 0",
                "UNIT 1 1",
                *atoms,
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )


def _write_hydrogen_audit_map(path: Path, *, cell_a: float = 10.0) -> None:
    import gemmi
    import numpy as np

    grid = gemmi.FloatGrid(8, 8, 8)
    grid.unit_cell = gemmi.UnitCell(cell_a, 11.0, 12.0, 90.0, 90.0, 90.0)
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    np.asarray(grid)[:] = 0.0
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header(2, True)
    ccp4.write_ccp4_map(str(path))


def test_hydrogen_omit_cli_writes_hash_bound_prospective_contract(tmp_path, capsys):
    reference = tmp_path / "full_h.res"
    never_h = tmp_path / "never_h.res"
    fofc = tmp_path / "fofc.map"
    out = tmp_path / "audit"
    _write_hydrogen_audit_model(reference, include_h=True)
    _write_hydrogen_audit_model(never_h, include_h=False)
    _write_hydrogen_audit_map(fofc)
    args = build_parser().parse_args(
        [
            "audit-hydrogen-omit",
            "--reference-res",
            str(reference),
            "--never-h-res",
            str(never_h),
            "--map",
            f"native={fofc}",
            "--out-dir",
            str(out),
        ]
    )

    assert args.func(args) == 0
    contract = json.loads((out / "input_contract.json").read_text())
    assert contract["reference_hydrogen_count"] == 1
    assert contract["never_h_hydrogen_count"] == 0
    assert contract["model_cell_max_length_relative_mismatch"] == pytest.approx(0.0)
    assert contract["common_non_h_count"] == 3
    assert contract["common_non_h_rms_A"] == pytest.approx(0.0)
    assert contract["maps"][0]["sha256"] == cli.sha256_file(fofc)
    assert (out / "hydrogen_omit_peaks.csv").is_file()
    assert '"status": "ok"' in capsys.readouterr().out


def test_hydrogen_omit_cli_rejects_unaligned_common_nonh_origin(tmp_path, capsys):
    reference = tmp_path / "full_h.res"
    never_h = tmp_path / "never_h_shifted.res"
    fofc = tmp_path / "fofc.map"
    _write_hydrogen_audit_model(reference, include_h=True)
    _write_hydrogen_audit_model(never_h, include_h=False, coordinate_shift=0.25)
    _write_hydrogen_audit_map(fofc)
    args = build_parser().parse_args(
        [
            "audit-hydrogen-omit",
            "--reference-res",
            str(reference),
            "--never-h-res",
            str(never_h),
            "--map",
            str(fofc),
            "--out-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert args.func(args) == 2
    assert "not aligned" in capsys.readouterr().out


def test_hydrogen_omit_cli_rejects_h_contaminated_never_h_model(tmp_path, capsys):
    reference = tmp_path / "full_h.res"
    fofc = tmp_path / "fofc.map"
    _write_hydrogen_audit_model(reference, include_h=True)
    _write_hydrogen_audit_map(fofc)
    args = build_parser().parse_args(
        [
            "audit-hydrogen-omit",
            "--reference-res",
            str(reference),
            "--never-h-res",
            str(reference),
            "--map",
            str(fofc),
            "--out-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert args.func(args) == 2
    assert "contains hydrogen atoms" in capsys.readouterr().out


def test_hydrogen_omit_cli_rejects_reference_model_setting_mismatch(tmp_path, capsys):
    reference = tmp_path / "full_h.res"
    never_h = tmp_path / "never_h_wrong_cell.res"
    fofc = tmp_path / "fofc.map"
    _write_hydrogen_audit_model(reference, include_h=True)
    _write_hydrogen_audit_model(never_h, cell_a=12.0, include_h=False)
    _write_hydrogen_audit_map(fofc)
    args = build_parser().parse_args(
        [
            "audit-hydrogen-omit",
            "--reference-res",
            str(reference),
            "--never-h-res",
            str(never_h),
            "--map",
            str(fofc),
            "--out-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert args.func(args) == 2
    assert "cell is incompatible" in capsys.readouterr().out


def test_hydrogen_omit_cli_accepts_equivalent_half_cell_symmetry_syntax(tmp_path):
    reference = tmp_path / "full_h.res"
    never_h = tmp_path / "never_h.res"
    fofc = tmp_path / "fofc.map"
    _write_hydrogen_audit_model(
        reference,
        include_h=True,
        symm="-X-1/2,-Y,Z-1/2",
    )
    _write_hydrogen_audit_model(
        never_h,
        include_h=False,
        symm="-X+1/2,-Y,Z+1/2",
    )
    _write_hydrogen_audit_map(fofc)
    args = build_parser().parse_args(
        [
            "audit-hydrogen-omit",
            "--reference-res",
            str(reference),
            "--never-h-res",
            str(never_h),
            "--map",
            str(fofc),
            "--out-dir",
            str(tmp_path / "audit"),
        ]
    )

    assert args.func(args) == 0


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
                "row_mr": {
                    "definition": {"instability_u": 0.02},
                    "dynamic": {
                        "Ninput": 10,
                        "Nmatched": 10,
                        "Nexcluded": 0,
                        "Nobs": 4,
                        "Nall": 10,
                        "Nmerge_input": 10,
                        "Nmerge_used": 9,
                        "Nmerge_excluded": 1,
                        "MNobs": 3,
                        "MNall": 7,
                        "Robs": 0.09,
                        "Rall": 0.10,
                        "wRall": 0.11,
                        "MRobs": 0.07,
                        "MRall": 0.08,
                        "MwRall": 0.09,
                    },
                    "born": {
                        "Ninput": 10,
                        "Nmatched": 10,
                        "Nexcluded": 0,
                        "Nobs": 4,
                        "Nall": 10,
                        "Nmerge_input": 10,
                        "Nmerge_used": 9,
                        "Nmerge_excluded": 1,
                        "MNobs": 3,
                        "MNall": 7,
                        "Robs": 0.19,
                        "Rall": 0.20,
                        "wRall": 0.21,
                        "MRobs": 0.17,
                        "MRall": 0.18,
                        "MwRall": 0.19,
                    },
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
                ",".join(
                    [
                        "row_index",
                        "h",
                        "k",
                        "l",
                        "d_A",
                        "group",
                        "detector_bin",
                        "basis_size",
                        "split",
                        "q_dyn_over_born",
                        "center_sg_inv_A",
                        "sg_window_inv_A",
                        "sg_window_raw_inv_A",
                        "sg_window_half_frames",
                        "sg_window_clipped",
                        "sg_dSg_domega_inv_A_per_deg",
                        "sg_profile_points_total",
                        "sg_profile_weight_sum",
                        "max_sg_solve_residual",
                        "dyn_residual_over_sigma",
                        "born_residual_over_sigma",
                    ]
                ),
                "0,1,0,0,0.80,0,0,32,work,0.90,0.001,0.004,0.005,0.5,0,0.002,60,1.0,0.00001,0.5,1.5",
                "1,0,1,0,0.90,0,1,36,work,1.10,-0.002,0.004,0.005,0.5,0,0.002,60,1.0,0.00002,-1.5,-2.0",
                "2,0,0,1,1.00,1,0,40,free,1.25,0.003,0.004,0.006,0.5,1,0.002,60,1.0,0.00003,2.0,2.5",
                "3,1,1,0,1.10,1,1,44,free,0.75,-0.004,0.004,0.006,0.5,0,0.002,60,1.0,0.00004,-1.0,-1.2",
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
                        "thickness_grad_frac": 1.0,
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
                    "failed": {
                        "returncode": 1,
                        "elapsed_seconds": 99.0,
                        "log": str(out_dir / "logs" / "failed.log"),
                        "command": ["false"],
                    },
                },
                "finished": {
                    "status": "ok",
                    "invocation_elapsed_seconds": 1.25,
                    "cumulative_stage_elapsed_seconds": 3.0,
                    "stage_elapsed_seconds": 3.0,
                    "completed_stages": 2,
                    "failed_stages": 1,
                    "estimated_stages": 2,
                },
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
    assert "Recorded workflow wall time: `0m01s`" in text
    assert "Cumulative completed-stage runtime: `0m03s`" in text
    assert "Total completed stage wall time" not in text
    assert "not workflow wall time" in text
    assert "Input Contract, Hand, And Paired T-Style Diagnostics" in text
    assert "SFAC / endpoint wavelength check | status `ok`; endpoint wavelength `0.025079` A" in text
    assert "selected HKL transform | `identity`" in text
    assert "final hand | `alternate`; post-feedback status `promoted`" in text
    assert (
        "final hand paired t-style, all/work/free | `t 5.000 (n 4, df 3)` / "
        "`t 4.000 (n 2, df 1)` / `t 3.000 (n 2, df 1)`" in text
    )
    assert "not a calibrated normal Z score" in text
    assert "free-row I residual Z | n `2`; RMSZ `1.581`" in text
    assert "thickness | `110.00 nm mean (1100.0 A); uniform-gradient support 0.00-220.00 nm`" in text
    assert "Legacy Native-Objective Audit Matrix (Not Table 1)" in text
    assert "raw signed rows (I), Bloch dynamical" in text
    assert "merged SHELXL endpoint (F/F^2)" in text
    assert "Legacy Native-Objective Delta Matrix (Not Table 1)" in text
    assert "final dynamical rows minus final matched Born rows" in text
    assert "Row Intensity R-Factor Definitions" in text
    assert "Raw signed-row intensity `wR(I)all` is `sqrt(sum_r w_r (Iobs_r - Icalc_r)^2 / sum_r w_r Iobs_r^2)`" in text
    assert "JANA/PETS-Style Row And Post-Refinement Merged Statistics By Resolution/Scale Contract" in text
    assert "Canonical Final Statistics" in text
    assert (out_dir / "TABLE1_FINAL_STATISTICS.md").exists()
    table1 = json.loads((out_dir / "TABLE1_FINAL_STATISTICS.json").read_text())
    assert table1["schema"] == "grond.table1_final_statistics.v1"
    assert table1["reporting_contract"]["row_instability_u"] == pytest.approx(0.02)
    assert table1["jana_pets_rows"][0]["status"] == "not_available"
    assert table1["jana_pets_rows"][1]["model"].startswith("GROND Born")
    assert table1["jana_pets_rows"][1]["Ninput"] == 10
    assert table1["jana_pets_rows"][1]["Nmerge_used"] == 9
    assert table1["jana_pets_rows"][1]["Nmerge_excluded"] == 1
    assert table1["shelxl_endpoints"][1]["status"] == "not_available"
    assert table1["shelxl_endpoints"][1]["WGHT"] is None
    assert table1["shelxl_endpoints"][1]["EXTI"] is None
    table1_text = (out_dir / "TABLE1_FINAL_STATISTICS.md").read_text()
    assert "instability factor `u=0.02`" in table1_text
    assert "GROND Born corrected-HKL endpoint | not available | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a" in table1_text
    endpoint_header = (out_dir / "TABLE1_SHELXL_ENDPOINTS.csv").read_text().splitlines()[0]
    row_header = (out_dir / "TABLE1_JANA_ROWS.csv").read_text().splitlines()[0]
    assert "R1gt" in endpoint_header and "wR2all" in endpoint_header
    assert "Robs" not in endpoint_header and "MRall" not in endpoint_header
    assert "Robs" in row_header and "MRall" in row_header and "MwRall" in row_header
    assert "R1gt" not in row_header and "wR2all" not in row_header and "GooF" not in row_header
    final_text = (out_dir / "FINAL_SUMMARY.md").read_text()
    assert "### A. SHELXL Endpoint Statistics" in final_text
    assert "### B. JANA/PETS-Style Row And Post-Refinement Merged Statistics" in final_text
    assert "Nmerge used/excluded" in final_text
    assert "free/held-out weighted R" not in final_text
    assert "Bloch matrix and row-contract diagnostics" in text
    bloch_diagnostics = out_dir / "BLOCH_DIAGNOSTICS.md"
    assert bloch_diagnostics.exists()
    bloch_text = bloch_diagnostics.read_text()
    assert "GROND Bloch Diagnostics" in bloch_text
    assert "Bethe Approximation" in bloch_text
    assert "Sg, Rocking, And Quadrature" in bloch_text
    assert "center excitation error Sg" in bloch_text
    assert "q = Idyn/Iborn" in bloch_text
    assert "Bloch matrix and row-contract diagnostics" in (out_dir / "FINAL_SUMMARY.md").read_text()
    assert info["console"]["files"]["bloch_diagnostics"] == str(bloch_diagnostics)
    assert info["console"]["final_hand_t_style_all"] == pytest.approx(5.0)
    assert info["console"]["final_hand_t_style_all_n"] == 4
    assert info["console"]["final_hand_t_style_all_df"] == 3
    assert info["console"]["final_hand_t_style_free"] == pytest.approx(3.0)
    assert info["console"]["final_hand_t_style_free_n"] == 2
    assert info["console"]["final_hand_t_style_free_df"] == 1
    assert "legacy field names only" in info["console"]["final_hand_z_compatibility_note"]


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
            "example_dials",
        ]
    )
    assert args.command == "import-dials"
    assert args.intensity_mode == "sum_raw"
    assert args.row_selection == "integrated"
    assert args.sigma_mode == "raw"
    assert args.dataset_name == "example_dials"
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


def test_structure_update_strategy_keeps_native_gradient_as_explicit_multi_dataset_diagnostic():
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
            "native_gradient",
        ]
    )
    assert resolve_structure_update_strategy(args, multi_dataset=True) == "native_gradient"


def test_shelxl_feedback_current_model_skips_empty_res(tmp_path: Path):
    run_root = tmp_path / "feedback"
    good_run = run_root / "cycle_04_N132Q20"
    bad_run = run_root / "cycle_05_N132Q20"
    good_res = good_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    bad_res = bad_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    good_res.parent.mkdir(parents=True)
    bad_res.parent.mkdir(parents=True)
    good_res.write_text(valid_shelx_res_text())
    bad_res.write_text("")
    (run_root / "shelxl_feedback_loop_summary.json").write_text(
        json.dumps(
            {
                    "current_model": str(bad_res),
                    "runs": [
                        {
                            "run_dir": str(good_run),
                            "returncode": 0,
                            "output_model": str(good_res),
                            "endpoint_shelxl_status": "ok",
                            "endpoint_res_health": bound_shelx_health(good_res),
                        },
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
    good_res.write_text(valid_shelx_res_text())
    npd_res.write_text(valid_shelx_res_text("npd"))
    (run_root / "shelxl_feedback_loop_summary.json").write_text(
        json.dumps(
            {
                    "current_model": str(npd_res),
                    "runs": [
                        {
                            "run_dir": str(good_run),
                            "returncode": 0,
                            "output_model": str(good_res),
                            "endpoint_shelxl_status": "ok",
                            "endpoint_res_health": bound_shelx_health(good_res),
                        },
                        {
                            "run_dir": str(npd_run),
                            "returncode": 0,
                            "output_model": str(npd_res),
                            "endpoint_shelxl_status": "adp_pathology",
                            "endpoint_res_health": bound_shelx_health(
                                npd_res,
                                safe_for_feedback=False,
                                npd_count=1,
                                split_count=0,
                                npd_atoms=["O001"],
                                has_npd=True,
                            ),
                    },
                ],
            }
        )
        + "\n"
    )

    assert cli.shelxl_feedback_current_model(run_root) == good_res
    assert cli.shelxl_feedback_current_run(run_root) == good_run


def test_shelxl_feedback_record_rejects_model_replaced_after_summary(tmp_path: Path):
    model = tmp_path / "endpoint.res"
    model.write_text(valid_shelx_res_text("recorded"))
    record = {
        "output_model": str(model),
        "endpoint_shelxl_status": "ok",
        "endpoint_res_health": bound_shelx_health(model),
    }
    assert cli.shelxl_feedback_record_safe_for_reuse(record) is True

    model.write_text(valid_shelx_res_text("replaced"))

    assert cli.shelxl_feedback_record_safe_for_reuse(record) is False


def test_multi_combined_endpoint_model_binds_current_lambda_and_artifact(tmp_path: Path):
    combined = tmp_path / "combined"
    stale = combined / "combined_endpoint_lam_1p0" / "multi_dyn_corrected.res"
    selected = combined / "combined_endpoint_lam_0p8" / "multi_dyn_corrected.res"
    stale.parent.mkdir(parents=True)
    selected.parent.mkdir(parents=True)
    stale.write_text(valid_shelx_res_text("stale_lambda_one"))
    selected.write_text(valid_shelx_res_text("selected_lambda_point_eight"))
    selected_health = bound_shelx_health(selected)
    summary = {
        "combined_endpoint": {
            "lambda_q": "0.8",
            "artifacts": {
                "res": str(selected),
                "res_sha256": selected_health["sha256"],
            },
            "shelxl": {"status": "ok", "res_health": selected_health},
        }
    }
    (combined / "combined_summary.json").write_text(json.dumps(summary) + "\n")

    assert cli.multi_combined_endpoint_model(combined) == selected

    summary["combined_endpoint"]["artifacts"]["res"] = str(stale)
    summary["combined_endpoint"]["artifacts"]["res_sha256"] = bound_shelx_health(stale)["sha256"]
    (combined / "combined_summary.json").write_text(json.dumps(summary) + "\n")
    assert cli.multi_combined_endpoint_model(combined) is None


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
    monkeypatch.setattr(cli, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(cli, "native_stage_cmd", lambda *args, **kwargs: ["native"])
    monkeypatch.setattr(cli, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(cli, "run_logged", lambda cmd, logs, label, **kwargs: labels.append(label))
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
    monkeypatch.setattr(cli, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(cli, "native_stage_cmd", lambda *args, **kwargs: ["native"])
    monkeypatch.setattr(cli, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(cli, "run_logged", lambda *args, **kwargs: None)
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
    monkeypatch.setattr(cli, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(
        cli,
        "native_stage_cmd",
        lambda _python, _ds, structure, _baseline, out_dir, _args, **kwargs: captured.append(
            (
                str(structure),
                str(out_dir),
                kwargs["stage"],
                kwargs["endpoint"],
                kwargs.get("strict_pinned_contract", False),
            )
        )
        or ["native"],
    )
    monkeypatch.setattr(cli, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(cli, "run_logged", lambda _cmd, _logs, label, **_kwargs: labels.append(label))
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
    monkeypatch.setattr(cli, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(
        cli,
        "native_stage_cmd",
        lambda _python, _ds, structure, _baseline, out_dir, _args, **kwargs: captured.append(
            (
                str(structure),
                str(out_dir),
                kwargs["stage"],
                kwargs["endpoint"],
                kwargs.get("strict_pinned_contract", False),
            )
        )
        or ["native"],
    )
    monkeypatch.setattr(cli, "run_logged", lambda _cmd, _logs, label, **_kwargs: labels.append(label))
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
    monkeypatch.setattr(cli, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(cli, "native_stage_cmd", lambda *args, **kwargs: ["native"])
    monkeypatch.setattr(cli, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(cli, "run_logged", lambda *args, **kwargs: None)
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
    monkeypatch.setattr(cli, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    monkeypatch.setattr(cli, "native_stage_cmd", lambda *args, **kwargs: ["native"])
    monkeypatch.setattr(cli, "combine_cmd", lambda *args, **kwargs: ["combine"])
    monkeypatch.setattr(cli, "run_logged", lambda cmd, logs, label, **kwargs: labels.append(label))
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


def test_shelxl_feedback_current_run_prefers_final_row_recompute(tmp_path: Path):
    run_root = tmp_path / "feedback"
    cycle_run = run_root / "cycle_05_N108Q20"
    final_run = run_root / "final_row_recompute_N108Q20"
    final_res = cycle_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    final_res.parent.mkdir(parents=True)
    final_run.mkdir(parents=True)
    final_res.write_text(valid_shelx_res_text())
    (run_root / "shelxl_feedback_loop_summary.json").write_text(
        json.dumps(
            {
                "current_model": str(final_res),
                "final_row_stats_run": str(final_run),
                "final_row_stats_model": str(final_res),
                    "runs": [
                        {
                            "run_dir": str(cycle_run),
                            "returncode": 0,
                            "output_model": str(final_res),
                            "endpoint_shelxl_status": "ok",
                            "endpoint_res_health": bound_shelx_health(final_res),
                        },
                        {
                        "label": "final_row_recompute",
                        "run_dir": str(final_run),
                        "returncode": 0,
                        "output_model": str(final_res),
                            "row_stats_model": str(final_res),
                            "skip_endpoint_refinement": True,
                            "endpoint_shelxl_status": "reused_validated_input_model",
                            "endpoint_res_health": bound_shelx_health(final_res),
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

    monkeypatch.setattr(cli, "run_logged", fake_run_logged)
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


def test_pinned_sg_jitter_gets_explicit_no_jitter_fallback(tmp_path: Path):
    args = build_parser().parse_args(
        ["run", "--out-dir", "out", "--hkl", "rows.hkl", "--structure", "model.res"]
    )
    dataset = DatasetSpec(name="sample", hkl="rows.hkl")
    cmd = cli.native_stage_cmd(
        "python",
        dataset,
        "model.res",
        "baseline.hkl",
        tmp_path / "run",
        args,
        n_basis=20,
        n_points=5,
        stage="multi_ls1_cycle_01",
        pinned={"sg_jitter_mode": "gaussian", "sg_jitter_sigma0": 0.0015},
        endpoint=False,
    )

    assert cmd[cmd.index("--sg-jitter-mode-grid") + 1] == "gaussian,none"


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


def test_native_stage_cmd_forwards_discrete_thickness_promotion_contract(tmp_path: Path):
    model = tmp_path / "model.res"
    model.write_text(valid_shelx_res_text())
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            str(tmp_path / "out"),
            "--xds-dir",
            str(tmp_path / "xds"),
            "--structure",
            str(model),
        ]
    )
    dataset = DatasetSpec(name="sample", hkl=str(tmp_path / "rows.hkl"))
    args.selection_metric = "free_wR2"
    args.production_scales = True
    baseline = {
        "thickness_A": 1204.5805095106,
        "thickness_grad_frac": 0.8,
        "dose_b_ref_frame": float("nan"),
    }

    command = cli.native_stage_cmd(
        "python",
        dataset,
        str(model),
        "",
        tmp_path / "thickness",
        args,
        n_basis=24,
        n_points=10,
        stage="multi_thickness_polish",
        pinned=baseline,
        promotion_baseline=baseline,
        scale_fit_split_override="work",
        selection_metric_override="work_wR2",
        endpoint=False,
    )

    forwarded = json.loads(command[command.index("--promotion-baseline-json") + 1])
    assert forwarded["thickness_A"] == baseline["thickness_A"]
    assert forwarded["thickness_grad_frac"] == baseline["thickness_grad_frac"]
    assert cli.math.isnan(forwarded["dose_b_ref_frame"])
    assert float(command[command.index("--promotion-min-gain") + 1]) == args.continuous_thickness_min_gain
    assert float(command[command.index("--promotion-max-free-wr2-loss") + 1]) == args.continuous_thickness_max_free_wr2_loss
    assert float(command[command.index("--promotion-max-all-r1-loss") + 1]) == args.continuous_thickness_max_all_r1_loss
    assert command[command.index("--scale-fit-split") + 1] == "work"
    assert command[command.index("--selection-metric") + 1] == "work_wR2"


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

    cmd = publication_report_cmd(
        "python",
        prep,
        result,
        tmp_path / "tables",
        "table",
        instability_u=0.02,
    )

    scale_index = cmd.index("--scale-polish")
    assert cmd[scale_index + 1].endswith("/scale_polish/111/scale_compression_summary.json")
    assert cmd[scale_index + 1].startswith("111=")
    resolution_index = cmd.index("--resolution-limit")
    assert cmd[resolution_index + 1].endswith("/resolution_limits/111/summary.json")
    assert cmd[resolution_index + 1].startswith("111=")
    assert cmd[cmd.index("--instability-u") + 1] == "0.02"


def test_multi_publication_report_separates_selected_and_full_scale_rows(tmp_path: Path):
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="a", hkl="a.hkl"), DatasetSpec(name="b", hkl="b.hkl")],
    )
    combined = tmp_path / "runs" / "combined"
    combined.mkdir(parents=True)
    (combined / "combined_summary.json").write_text(
        json.dumps({"combined_endpoint": {"requested_min_d_spacing_A": 0.75}}) + "\n"
    )
    result = {
        "combined": str(combined),
        "model": prep.structure,
        "publication_runs": {"a": "/selected/a", "b": "/selected/b"},
        "publication_full_runs": {"a": "/full/a", "b": "/full/b"},
        "scale_contract_selection": {
            "selected_contract": "compact_min_strict",
            "publication_scale_polish_summaries": {},
        },
    }

    cmd = publication_report_cmd("python", prep, result, tmp_path / "tables", "table")

    selected_values = [cmd[index + 1] for index, value in enumerate(cmd) if value == "--run"]
    full_values = [cmd[index + 1] for index, value in enumerate(cmd) if value == "--full-run"]
    assert selected_values == ["a=/selected/a", "b=/selected/b"]
    assert full_values == ["a=/full/a", "b=/full/b"]
    assert cmd[cmd.index("--selected-scale-contract") + 1] == "compact_min_strict"
    assert cmd[cmd.index("--requested-min-d-spacing") + 1] == "0.75"


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


def test_missing_map_tool_only_errors_when_legacy_maps_are_requested(tmp_path: Path, monkeypatch):
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
            "--make-maps",
        ]
    )
    apply_tool_config(args)
    errors = required_external_tool_errors(args, structure=str(model))
    assert any("shelx2map" in item for item in errors)
    args.make_maps = False
    assert required_external_tool_errors(args, structure=str(model)) == []


def test_explicit_shelxl_argument_satisfies_preflight_without_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty_path"))
    monkeypatch.delenv("SHELXL", raising=False)
    monkeypatch.delenv("SHELX2MAP", raising=False)
    fake_shelxl = tmp_path / "ccp4" / "shelxl"
    fake_shelxl.parent.mkdir()
    fake_shelxl.write_text("#!/bin/sh\nexit 0\n")
    fake_shelxl.chmod(0o755)
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
            "--shelxl",
            str(fake_shelxl),
            "--no-maps",
        ]
    )
    apply_tool_config(args)

    status = cli.external_tool_statuses(args)["shelxl"]
    assert status["available"] is True
    assert status["resolved"] == str(fake_shelxl.resolve())
    assert status["source"] == "argument:--shelxl"
    assert required_external_tool_errors(args, structure=str(model)) == []


def test_explicit_relative_shelxl_is_frozen_to_launch_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    fake_shelxl = tmp_path / "private-ccp4" / "shelxl"
    fake_shelxl.parent.mkdir()
    fake_shelxl.write_text("#!/bin/sh\nexit 0\n")
    fake_shelxl.chmod(0o755)
    model = tmp_path / "model.res"
    model.write_text(valid_shelx_res_text())
    requested = "./private-ccp4/shelxl"
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
            requested,
        ]
    )

    apply_tool_config(args)

    assert args.shelxl_requested == requested
    assert args.shelxl == str(fake_shelxl.resolve())
    status = cli.external_tool_statuses(args)["shelxl"]
    assert status["requested"] == requested
    assert status["resolved"] == str(fake_shelxl.resolve())


def test_default_shelxl_resolution_falls_back_to_shelxl2019(monkeypatch):
    monkeypatch.delenv("SHELXL", raising=False)

    def fake_resolve(_name: str, *, configured: str = "", **_kwargs):
        if configured:
            return {"available": False, "resolved": ""}
        return {"available": True, "resolved": "/opt/shelx/shelxl2019"}

    monkeypatch.setattr(cli, "resolve_executable", fake_resolve)
    args = build_parser().parse_args(
        ["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"]
    )

    apply_tool_config(args)

    assert args.shelxl_requested == "shelxl"
    assert args.shelxl == "/opt/shelx/shelxl2019"


def test_combined_summary_exposes_joint_native_metrics(tmp_path: Path):
    run_dir = tmp_path / "combined"
    run_dir.mkdir()
    (run_dir / "combined_summary.json").write_text(
        json.dumps(
            {
                "wght": {"a": 0.12, "b": 0.0},
                "native_joint": {
                    "dyn": {
                        "all": {"R1_all": 0.13, "wR2_all": 0.24},
                        "work": {"R1_all": 0.131, "wR2_all": 0.241},
                        "free": {"R1_all": 0.132, "wR2_all": 0.242},
                    },
                    "born": {
                        "all": {"R1_all": 0.17, "wR2_all": 0.30},
                        "work": {"R1_all": 0.171, "wR2_all": 0.301},
                        "free": {"R1_all": 0.172, "wR2_all": 0.302},
                    },
                },
            }
        )
        + "\n"
    )

    summary = cli._quick_load_run_summary(run_dir)
    assert cli._quick_dyn_metric(summary, "all", "R1_all") == 0.13
    assert cli._quick_born_metric(summary, "free", "wR2_all") == 0.302
    native = cli._quick_best_native(summary)
    assert native["native_weight"]["a"] == 0.12


def test_publication_endpoint_recovers_hkl_from_res(tmp_path: Path):
    endpoint_res = tmp_path / "multi_dyn_corrected.res"
    endpoint_hkl = tmp_path / "multi_dyn_corrected.hkl"
    endpoint_res.write_text("TITL test\nEND\n")
    endpoint_hkl.write_text("   0   0   0    0.00    0.00\n")

    endpoint = cli._normalize_publication_endpoint_block(
        {"res": str(endpoint_res), "shelxl": {"R1_all": 0.1}},
        selection_source="endpoint_chooser",
    )

    assert endpoint["endpoint_hkl"] == str(endpoint_hkl)
