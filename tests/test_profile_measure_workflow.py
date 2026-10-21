"""Explicit profile measure survives automatic, native, nested and continuation command boundaries."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from grond.bloch.config import Config
from grond.cli_options.parser import build_parser
from grond.engine import nested_nuisance_stage as nested_stage
from grond.engine.shelxl_feedback_loop import native_command
from grond.io.config import DatasetSpec
from grond.refine.arguments_feedback import build_parser as feedback_parser
from grond.regularize.arguments_native import build_parser as native_parser
from grond.regularize.contracts import TARGET_MODES, Nuisance
from grond.workflows.checkpoints import _checkpoint_dataset_contract
from grond.workflows.crystal_parameters import resolve_crystal_parameters
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nested_nuisance import nested_variants
from grond.workflows.nuisance_policy import auto_mosaicity_grid
from grond.workflows.push_contracts import hydrate_push_args_for_multi
from tests.test_profile_measure import port, samples

OPTIONS = ["--profile-coordinate", "direct_angle", "--profile-normalization", "target_sg_equivalent_v1",
           "--profile-angle-max-half-width-deg", "3.5", "--profile-quadrature", "gauss_legendre"]


def parser(tmp_path):
    return native_parser("test", default_hkl=tmp_path/"input.hkl", default_res=tmp_path/"model.res",
                         default_baseline_hkl=tmp_path/"base.hkl", target_modes=TARGET_MODES,
                         max_support_sg=0.01, shelxl="shelxl", runtime_cache_directory=lambda *_: tmp_path/"cache")


def source_and_args(tmp_path):
    args = build_parser().parse_args(["auto", "--hkl", str(tmp_path/"input.hkl"), "--structure", str(tmp_path/"model.res"),
                                      "--out-dir", str(tmp_path/"run"), *OPTIONS, "--mosaicity-mode", "gaussian_cone",
                                      "--mosaicity-support-sigma-deg", "0.3"])
    args.dose_b_mode_grid = "none"  # a stage may narrow the AUTO dose grid on its namespace (nuisance_grid)
    ds = DatasetSpec(name="one", hkl=str(tmp_path/"input.hkl"))
    nui = asdict(Nuisance(1200., 0., 0., 1., 0, 0., 0., 0., mosaicity_mode="gaussian_cone",
                         mosaicity_sigma_deg=0.1, mosaicity_points=3, mosaicity_support_sigma_deg=0.3))
    command = native_stage_cmd("python", ds, str(tmp_path/"model.res"), "", tmp_path/"native", args,
                               n_basis=20, n_points=60, stage="final", pinned=nui, endpoint=False, make_maps=False)
    parsed = parser(tmp_path).parse_args(command[2:])
    summary = {"config": vars(parsed), "best_native": {"nuisance": nui}}
    return args, ds, nui, summary, parsed


def test_auto_to_native_to_saved_spectral_config_keeps_measure(tmp_path):
    args, ds, _nui, _summary, parsed = source_and_args(tmp_path)
    assert parsed.profile_coordinate == "direct_angle"
    assert parsed.profile_normalization == "target_sg_equivalent_v1"
    assert parsed.profile_angle_max_half_width_deg == 3.5
    cfg = Config(profile_coordinate=parsed.profile_coordinate, profile_normalization=parsed.profile_normalization,
                 profile_angle_max_half_width_deg=parsed.profile_angle_max_half_width_deg,
                 mosaicity_support_sigma_deg=parsed.mosaicity_support_sigma_deg)
    assert cfg.mosaicity_support_sigma_deg == 0.3
    variants = nested_variants(args, ds)
    widths = [v["config"]["mosaicity_sigma_deg"] for v in variants]
    assert 0.0 in widths and 0.3 in widths
    for variant in variants:
        assert variant["config"]["mosaicity_mode"] == "gaussian_cone"
        trial = nested_stage.variant_config({"base_config": asdict(cfg)}, variant)
        assert trial.profile_normalization == "target_sg_equivalent_v1"
        assert trial.mosaicity_support_sigma_deg == 0.3


def test_continuation_checkpoint_and_final_recompute_keep_measure(tmp_path):
    _args, ds, nui, summary, _parsed = source_and_args(tmp_path)
    path = tmp_path/"source_summary.json"
    path.write_text(json.dumps(summary))
    record, final_config = _checkpoint_dataset_contract(ds, path)
    assert final_config["profile_normalization"] == "target_sg_equivalent_v1"
    record.update(source_summary=str(path), final_config=final_config, nuisance=nui)
    checkpoint = {"datasets": [record]}
    args = build_parser().parse_args(["refine", "--checkpoint", "checkpoint.json", "--out-dir", str(tmp_path/"continue")])
    settings, pinned = resolve_crystal_parameters(checkpoint, [ds], {ds.name:nui}, args, checkpoint_dir=tmp_path)
    args.crystal_parameters = settings
    args = hydrate_push_args_for_multi(args, {ds.name:summary})
    for stage in ["multi_ls1_cycle_01", "final_row_recompute"]:
        command = native_stage_cmd("python", ds, "model.res", "", tmp_path/stage, args, n_basis=20,
                                   n_points=60, stage=stage, pinned=pinned[ds.name], endpoint=False, make_maps=False)
        parsed = parser(tmp_path).parse_args(command[2:])
        assert parsed.profile_coordinate == "direct_angle"
        assert parsed.profile_normalization == "target_sg_equivalent_v1"
        assert parsed.profile_angle_max_half_width_deg == 3.5
        assert parsed.mosaicity_support_sigma_deg == 0.3


def test_single_dataset_feedback_replays_and_can_explicitly_override_measure(tmp_path):
    _args, _ds, _nui, summary, _parsed = source_and_args(tmp_path)
    args = feedback_parser("test").parse_args(["--source-summary", "summary.json", "--start-res", "model.res",
                                              "--out-dir", str(tmp_path/"feedback")])
    for overrides, expected in [([], "target_sg_equivalent_v1"), (["--profile-normalization", "legacy"], "legacy")]:
        if overrides:
            args = feedback_parser("test").parse_args(["--source-summary", "summary.json", "--start-res", "model.res",
                                                      "--out-dir", str(tmp_path/"feedback"), *overrides])
        command = native_command(args, summary, tmp_path/"model.res", tmp_path/"step", n_basis=20, n_points=60)
        parsed = parser(tmp_path).parse_args(command[2:])
        assert parsed.profile_normalization == expected
        assert parsed.profile_coordinate == "direct_angle"


def test_auto_refuses_unmatched_full_cone_measure_or_insufficient_support(tmp_path):
    args, _ds, _nui, _summary, _parsed = source_and_args(tmp_path)
    args.profile_normalization = "legacy"
    with pytest.raises(ValueError, match="comparable row intensities"):
        auto_mosaicity_grid(args)
    args.profile_normalization = "target_sg_equivalent_v1"
    args.mosaicity_support_sigma_deg = 0.2
    with pytest.raises(ValueError, match="entire two-axis cone"):
        auto_mosaicity_grid(args)
    args.mosaicity_support_sigma_deg = 0.3
    args.nuisance_selection = "sequential"
    with pytest.raises(ValueError, match="same common beam envelope"):
        auto_mosaicity_grid(args)


def test_nonlinear_common_support_uses_exact_interval_independent_of_candidate_width():
    p = port(curvature=0.002)
    _, _, _, profile_meta, _ = samples(p)
    envelopes = []
    for sigma in [0.0, 0.1, 0.3]:
        p.cfg = replace(p.cfg, mosaicity_mode="gaussian_cone", mosaicity_sigma_deg=sigma,
                        mosaicity_support_sigma_deg=0.3)
        omega, cone, meta = p.cone_support_orientations(np.array([1, 0, 0]), p.center, 0.006, row_idx=0)
        np.testing.assert_allclose([omega[0], omega[-1]], profile_meta["sg_profile_angle_interval_deg"])
        envelopes.append((omega, cone))
        assert meta["mosaicity_support_angle_interval_deg"] == pytest.approx(profile_meta["sg_profile_angle_interval_deg"])
    for omega, cone in envelopes[1:]:
        np.testing.assert_array_equal(omega, envelopes[0][0])
        np.testing.assert_array_equal(cone, envelopes[0][1])
