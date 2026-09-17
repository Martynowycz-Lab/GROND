"""CLI parser commands regression tests."""







import grond.cli as cli
from grond.cli import (
    build_parser,
)
from tests.cli._support import cli_command_names


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


def test_finish_parser_defaults_to_existing_run_dir():
    args = build_parser().parse_args(["finish", "--run-dir", "grond_out", "--no-make-maps"])
    assert args.command == "finish"
    assert args.run_dir == "grond_out"
    assert args.checkpoint == ""
    assert args.make_maps is False
    assert args.map_mode == "native"


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
    assert push.n_basis is None
    assert push.n_points is None
    assert push.compute_backend == "scipy"
    assert push.row_plan_cache == "off"
    assert push.chunksize == 4
    assert push.topup_d_min is None
    assert push.post_bloch_sigma_policy == "none"
    assert push.model_prep == "free-h-anis-no-restraints"
    assert push.set_ls_cycles == 1
    push_free_h_iso = build_parser().parse_args(
        [
            "push-endpoint",
            "--source-summary",
            "summary.json",
            "--start-res",
            "model.res",
            "--out-dir",
            "push",
            "--model-prep",
            "free-h-isotropic-no-restraints",
        ]
    )
    assert push_free_h_iso.model_prep == "free-h-isotropic-no-restraints"
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
    assert push_checkpoint.topup_d_min is None
    assert push_checkpoint.post_bloch_sigma_policy == "none"


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
