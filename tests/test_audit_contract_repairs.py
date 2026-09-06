"""Regressions for physical inputs crossing cache and refinement boundaries."""

import argparse
import csv
import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import gemmi
import numpy as np
import pytest

from grond.legacy.engine import combine_native_nuisance_runs as combine
from grond.legacy.engine import native_row_regularize as nr
from grond.legacy.engine import shelxl_feedback_loop as feedback
from grond.legacy.engine.bloch_engine import Config, validate_passive_absorption


@pytest.mark.parametrize("sidecar_kind", ["geometry_path", "pets"])
def test_changed_geometry_recomputes_outer_cached_prediction(tmp_path, monkeypatch, sidecar_kind):
    hkl, res, sidecar = [tmp_path / name for name in ("rows.hkl", "model.res", "geometry.dat")]
    hkl.write_text("unchanged rows")
    res.write_text("unchanged model")
    sidecar.write_text("1")
    data = SimpleNamespace(a_scan_points_path=None, hkl_transform_spec="identity")
    cfg = Config()
    if sidecar_kind == "geometry_path":
        cfg.geometry_path_file = str(sidecar)
    else:
        data.pets_exact_geometry_path = sidecar
    calls = []

    class GeometryDependentPort:
        def predict_rows(self, indices, **kwargs):
            value = float(sidecar.read_text())
            calls.append(value)
            return {
                "indices": indices,
                "dyn": np.array([value]), "born": np.array([1.0]),
                "basis_support_hash64": np.array([123], dtype=np.uint64),
                "sg_profile_same_branch_continuation_used": np.array([False]),
                "sg_profile_same_branch_continuation_segments": np.array([1]),
            }

    monkeypatch.setattr(nr, "XDSBlochPort", lambda *a, **kw: GeometryDependentPort())

    def predict():
        return nr.predict_with_cache(data, {}, "model", hkl, res, cfg, np.array([0]), tmp_path / "cache", 1, 1)

    assert predict()["dyn"][0] == 1.0
    assert predict()["cache_status"] == "hit"
    sidecar.write_text("2")
    assert predict()["dyn"][0] == 2.0
    assert calls == [1.0, 2.0]
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="required .* file is missing"):
        predict()


def test_feedback_preserves_recorded_physics_and_row_split_inputs(tmp_path):
    args = Namespace(
        python="python", hkl="rows.hkl", baseline_hkl="", hkl_transform="identity",
        min_d_spacing=0.7, min_peak=0.0, geometry_percentile=100.0, topup_d_min=0.7,
        lambda_grid="1", shelxl="shelxl", shelxl_iterations=1, endpoint_ls_cycles=1,
        post_bloch_sigma_policy="none", post_bloch_sigma_holdout_fraction=0.2,
        post_bloch_sigma_min_scale=0.25, post_bloch_sigma_max_scale=4.0,
        rescale="none", workers=1, chunksize=1, cache_compression="none", compute_backend="scipy",
        jax_batch_size=0, jax_matrix_bucket_size=0, bethe_strong_count=48,
        bethe_variant="schur_target", bethe_promote_denom=0.005, bethe_denom_eps=1e-5, max_rows=0,
    )
    config = {
        "row_contract": "exact_rows.csv", "split_contract": "split.json", "geometry_path": "tilt.json",
        "profile_basis_mode": "common_profile", "profile_coordinate": "direct_angle",
        "profile_angle_max_half_width_deg": 1.5, "sg_window_min": 0.002, "sg_window_max": 0.013,
        "sg_window_scale": 1.4, "sg_window_frame_floor": 0.6,
        "mosaicity_mode": "gaussian_cone", "mosaicity_sigma_deg": 0.04, "mosaicity_points": 5,
        "mosaicity_support_sigma_deg": 0.08, "mosaicity_support_points": 7,
        "mosaicity_support_width_sigma": 4.5, "mosaicity_max_sigma_deg": 0.3,
        "detector_scale_ridge": 0.23, "detector_scale_smoothness": 0.31, "detector_scale_min_rows": 17,
        "dose_b_ref_frame": 7.5, "max_rows": 123,
    }
    summary = {"config": config, "best_native": {"nuisance": {"thickness_A": 700.0}}}
    command = feedback.native_command(args, summary, tmp_path / "model.res", tmp_path / "cycle", n_basis=64, n_points=20)
    for key, expected in config.items():
        flag = "--" + key.replace("_", "-")
        assert command.count(flag) == 1
        assert command[command.index(flag) + 1] == str(expected)

    # Parse the generated subprocess arguments with the actual native parser;
    # token presence alone cannot catch an invalid flag or ignored argument.
    parser = None
    original_parse = argparse.ArgumentParser.parse_args

    class ParserCaptured(Exception):
        pass

    def capture(self, *_args, **_kwargs):
        nonlocal parser
        parser = self
        raise ParserCaptured()

    from unittest.mock import patch
    with patch.object(argparse.ArgumentParser, "parse_args", capture), pytest.raises(ParserCaptured):
        nr.main()
    parsed = original_parse(parser, command[2:])
    for key, expected in config.items():
        assert getattr(parsed, key) == expected


@pytest.mark.parametrize("factors", [[0.5, 1.0, 2.0], [1.0, 1.0, 1.0]])
def test_combined_export_restores_intensity_and_sigma_units(factors):
    rows = [
        {"h": i + 1, "k": 0, "l": 0, "target_I": obs * f, "sigma_eff": 2.0 * f,
         "target_mode": "iobs_over_rlp", "source_to_target_factor": f,
         "dyn_raw": 1.0, "dyn_scaled": 1.0, "q_dyn_over_born": 1.0}
        for i, (obs, f) in enumerate(zip([10.0, 20.0, 30.0], factors, strict=True))
    ]
    out, _ = combine.endpoint_rows_and_corrections_from_native(rows, 1.0)
    np.testing.assert_allclose([r[3] for r in out], [10.0, 20.0, 30.0])
    np.testing.assert_allclose([r[4] for r in out], [2.0, 2.0, 2.0])
    del rows[0]["source_to_target_factor"]
    with pytest.raises(ValueError, match="cannot restore"):
        combine.endpoint_rows_and_corrections_from_native(rows, 1.0)


def test_predictive_rms_does_not_depend_on_training_parameter_count():
    obs, pred, sig = np.full(10, 11.0), np.full(10, 10.0), np.ones(10)
    m = nr.native_metrics(obs, sig, pred, np.ones(10, dtype=bool), 0.0, 0.0, n_params=20)
    assert m["predictive_rms_z"] == 1.0
    assert m["weighted_rms_residual"] == 1.0


def test_paired_inverse_preserves_modulation_once_and_support_hashes():
    n = 6
    data = SimpleNamespace(
        hkl=np.array([[i + 1, 0, 0] for i in range(n)]),
        cell=gemmi.UnitCell(10, 10, 10, 90, 90, 90),
        zcal=np.zeros(n), corr=np.full(n, 100.0), peak=np.full(n, 100.0),
        xcal=np.ones(n), ycal=np.ones(n), iobs=np.arange(1, n + 1, dtype=float),
        omega_from_z=lambda z: z, pets_exact_a_center=None,
    )
    masks = {"all": np.ones(n, bool), "work": np.array([1, 1, 1, 1, 0, 0], bool),
             "free": np.array([0, 0, 0, 0, 1, 1], bool)}
    nuisance = nr.Nuisance(1200.0, 0.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0, row_resolution_b=10.0)
    support = np.arange(100, 100 + n, dtype=np.uint64)
    pred = {"indices": np.arange(n), "dyn": np.ones(n), "dyn_inverse": np.ones(n),
            "born": np.ones(n), "basis_support_hash64": support}
    first = nr.evaluate_trial(data, data.iobs, np.ones(n), pred, np.arange(n), masks, nuisance, 0.0, 0.0)
    inverse = nr.paired_inverse_prediction(first)
    second = nr.evaluate_trial(data, data.iobs, np.ones(n), inverse, inverse["indices"], masks, nuisance, 0.0, 0.0)
    np.testing.assert_allclose(first["arrays"]["dyn_raw"], second["arrays"]["dyn_raw"])
    np.testing.assert_allclose(first["arrays"]["dyn_scaled"], second["arrays"]["dyn_scaled"])
    np.testing.assert_array_equal(inverse["basis_support_hash64"], support)
    assert first["metrics"]["free"]["dyn"]["dof"] == 2


def test_outer_cache_distinguishes_stale_loaded_geometry_from_reloaded_object(tmp_path, monkeypatch):
    hkl, res, sidecar = [tmp_path / name for name in ("rows.hkl", "model.res", "geometry.npz")]
    hkl.write_text("rows")
    res.write_text("model")
    sidecar.write_text("new-sidecar-bytes")
    old = SimpleNamespace(a_scan_points_path=None, hkl_transform_spec="identity",
                          pets_exact_geometry_path=sidecar, pets_exact_a_center=np.ones((1, 3, 3)))
    new = SimpleNamespace(**{**vars(old), "pets_exact_a_center": np.full((1, 3, 3), 2.)})
    calls = []

    class Port:
        def __init__(self, data):
            self.data = data

        def predict_rows(self, indices, **kwargs):
            value = float(self.data.pets_exact_a_center[0, 0, 0])
            calls.append(value)
            return {"indices": indices, "dyn": np.array([value]), "born": np.ones(1),
                    "basis_support_hash64": np.array([123], dtype=np.uint64),
                    "sg_profile_same_branch_continuation_used": np.array([False]),
                    "sg_profile_same_branch_continuation_segments": np.array([1])}

    monkeypatch.setattr(nr, "XDSBlochPort", lambda data, *args, **kwargs: Port(data))
    def predict(data):
        return nr.predict_with_cache(data, {}, "m", hkl, res, Config(), np.array([0]), tmp_path / "cache", 1, 1)
    assert predict(old)["dyn"][0] == 1.
    assert predict(new)["dyn"][0] == 2.
    assert predict(new)["cache_status"] == "hit"
    assert calls == [1., 2.]


def _native_grid_inputs(tmp_path):
    hkl = tmp_path / "rows.hkl"
    header = {
        "X-RAY_WAVELENGTH": ".025079", "STARTING_FRAME": "1", "STARTING_ANGLE": "0",
        "OSCILLATION_RANGE": "1", "ROTATION_AXIS": "0 1 0", "INCIDENT_BEAM_DIRECTION": "0 0 1",
        "UNIT_CELL_CONSTANTS": "10 10 10 90 90 90",
        "UNIT_CELL_A-AXIS": "10 0 0", "UNIT_CELL_B-AXIS": "0 10 0", "UNIT_CELL_C-AXIS": "0 0 10",
    }
    rows = [f"{i+1} 0 0 {10*(i+1)} 2 100 100 1 {0.5*(i+1)} 100 100 0" for i in range(6)]
    hkl.write_text("\n".join([*(f"!{key}={value}" for key, value in header.items()), *rows, ""]))
    res = tmp_path / "model.res"
    res.write_text("TITL grid\nCELL .025079 10 10 10 90 90 90\nLATT -1\nSFAC C 1 1 1 1 1 1 1 1 0\nFVAR 1\nC1 1 .1 .2 .3 11 .05\nWGHT 0 0\nHKLF 4\nEND\n")
    return hkl, res


@pytest.mark.parametrize("include_valid", [True, False])
def test_native_grid_reports_nonpassive_candidate_and_preserves_valid_result(tmp_path, monkeypatch, include_valid):
    hkl, res = _native_grid_inputs(tmp_path)
    out = tmp_path / "run"
    modes = "none,midgley_like" if include_valid else "midgley_like"
    argv = ["native", "--hkl", str(hkl), "--res", str(res), "--out-dir", str(out),
            "--skip-endpoint-refinement", "--thickness-grid", "700", "--thickness-model-grid", "single",
            "--frame-offset-grid", "0", "--group-width-grid", "3", "--detector-scale-bins", "0",
            "--sigma-a-grid", "0", "--sigma-b-corr-grid", "0", "--sigma-c-peak-grid", "0",
            "--absorption-mode-grid", modes, "--absorption-mu-grid", "0", "--absorption-frac-grid", ".02",
            "--n-basis", "2", "--n-points", "3", "--row-plan-cache", "off",
            "--target-mode", "iobs_over_rlp", "--profile-basis-mode", "common_profile",
            "--mosaicity-max-sigma-deg", ".3"]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []

    def synthetic_prediction(data, model, res_hash, hkl_path, res_path, cfg, indices, *args, **kwargs):
        # Run the real passivity validator against a matrix with known gain.
        # Mock only expensive row prediction; main/grid/selection/export stay real.
        calls.append(cfg.absorption_mode)
        hamiltonian = (1 + cfg.absorption_frac * 1j) * np.array([[0, .001], [.001, 0]])
        validate_passive_absorption(hamiltonian, mode=cfg.absorption_mode, row_index=2)
        return {"indices": indices, "dyn": data.iobs[indices], "born": data.iobs[indices] ** .8,
                "basis_size": np.full(len(indices), 2), "basis_support_hash64": np.full(len(indices), 123, dtype=np.uint64),
                "sg_profile_same_branch_continuation_used": np.zeros(len(indices), bool),
                "sg_profile_same_branch_continuation_segments": np.ones(len(indices), int)}

    monkeypatch.setattr(nr, "predict_with_cache", synthetic_prediction)
    if include_valid:
        nr.main()
        summary = json.loads((out / "summary.json").read_text())
        assert summary["best_native"]["nuisance"]["absorption_mode"] == "none"
        assert summary["nuisance_trial_count"] == 1
        assert summary["bloch_kernel_contract"]["basis"]["profile_basis_mode"] == "common_profile"
        with (out / "native_dyn_calc.csv").open() as handle:
            written = list(csv.DictReader(handle))
        assert written
        for row in written:
            assert float(row["source_sigma"]) == pytest.approx(float(row["xds_sigma"]) * float(row["source_to_target_factor"]))
        assert (out / "native_dyn_calc.csv").exists()
    else:
        with pytest.raises(RuntimeError, match="Every requested nuisance prediction candidate failed a physical contract"):
            nr.main()
        failure = json.loads((out / "prediction_grid_failure.json").read_text())
        assert "not passive" in failure["message"].lower()
        assert not (out / "summary.json").exists()
    rejected = json.loads((out / "rejected_prediction_candidates.json").read_text())
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "nonpassive_absorption"
    assert rejected[0]["details"]["row_index"] == 2
    assert rejected[0]["details"]["minimum_loss_eigenvalue_inv_A"] < 0
    assert rejected[0]["config"]["absorption_mode"] == "midgley_like"
    assert calls == (["none", "midgley_like"] if include_valid else ["midgley_like"])
