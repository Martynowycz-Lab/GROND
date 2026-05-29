import os
from pathlib import Path

from grond.cli import (
    apply_speed_profile,
    apply_tool_config,
    build_parser,
    estimate_model_nonh_atoms,
    publication_report_cmd,
    required_external_tool_errors,
    resolve_structure_update_strategy,
    transform_scout_job_count,
)
from grond.io.config import DatasetSpec, PrepConfig


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
    assert args.jax_batch_size == 0
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
    assert args.fast is False
    assert args.full_fast is False
    assert args.full_fast_n_basis == 20
    assert args.full_fast_n_points == 10
    assert args.full_fast_structure_rows == 0
    assert args.full_fast_structure_max_nfev == 0
    assert args.scout_n_basis == 8
    assert args.scout_n_points == 3
    assert args.transform_rerank_top == 3
    assert args.transform_rerank_n_basis == 20
    assert args.transform_rerank_n_points == 10
    assert args.transform_rerank_max_rows == 0
    assert args.transform_scout_jobs == 0
    assert args.transform_scout_workers_per_job == 1
    assert args.transform_metadata_prefilter == "auto"
    assert args.transform_metadata_max_candidates == 3
    assert args.transform_metadata_min_correlation == 0.70
    assert args.final_basis_scan_limit == 500
    assert args.final_basis_margin == 10
    assert args.final_basis_max == 500
    assert args.n_points == 20
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
    assert args.shelxl_iterations == 5


def test_utility_commands_parse():
    doc = build_parser().parse_args(["doctor", "--probe-bloch"])
    assert doc.command == "doctor"
    assert doc.probe_bloch is True
    assert build_parser().parse_args(["compile", "--force"]).force is True
    cfg = build_parser().parse_args(["init-config", "--out", "grond.config.json"])
    assert cfg.command == "init-config"
    assert cfg.format == "json"
    audit = build_parser().parse_args(["forensics", "--row-table", "native_dyn_calc.csv", "--hkl", "INTEGRATE.HKL", "--res", "model.res", "--out-dir", "audit", "--forensic-depth", "full"])
    assert audit.command == "forensics"
    assert audit.forensic_depth == "full"
    assert audit.forensic_gate is True
    assert audit.observation_polish == "auto"
    assert audit.observation_polish_combine == "ensemble"


def test_auto_parser_is_simple_run_alias():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res"])
    assert args.command == "auto"
    assert args.mode == "auto"
    assert args.resume is True
    assert args.detector_scale_bins == "8"
    assert args.final_basis_max == 500
    assert args.scale_polish_detector_bins == "exact"


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
            "jax_cuda",
            "--jax-batch-size",
            "192",
        ]
    )
    assert args.compute_backend == "jax_cuda"
    assert args.jax_batch_size == 192


def test_full_fast_parser_exposes_middle_profile():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res", "--full-fast"])
    assert args.command == "auto"
    assert args.mode == "auto"
    assert args.fast is False
    assert args.full_fast is True
    assert args.full_fast_n_basis == 20
    assert args.full_fast_n_points == 10
    assert args.full_fast_structure_rows == 0
    assert args.full_fast_structure_max_nfev == 0


def test_full_fast_profile_pins_cheap_automatic_work():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res", "--full-fast"])
    profile = apply_speed_profile(args)
    assert profile["profile"] == "full_fast"
    assert args.screen_n_basis == 20
    assert args.screen_n_points == 10
    assert args.transform_rerank_n_basis == 20
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
