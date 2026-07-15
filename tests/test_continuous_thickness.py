import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from grond.legacy.engine import continuous_thickness_polish as continuous
from grond.legacy.engine.bloch_engine import (
    Config,
    XDSBlochPort,
    XDSData,
    parse_model,
    sha256,
)


def _write_single_row_inputs(tmp_path: Path) -> tuple[Path, Path]:
    hkl = tmp_path / "INTEGRATE.HKL"
    hkl.write_text(
        "!X-RAY_WAVELENGTH= 0.025079\n"
        "!STARTING_FRAME= 1.0\n"
        "!STARTING_ANGLE= 0.0\n"
        "!OSCILLATION_RANGE= 1.0\n"
        "!ROTATION_AXIS= 0.0 1.0 0.0\n"
        "!INCIDENT_BEAM_DIRECTION= 0.0 0.0 1.0\n"
        "!UNIT_CELL_CONSTANTS= 10.0 10.0 10.0 90.0 90.0 90.0\n"
        "!UNIT_CELL_A-AXIS= 10.0 0.0 0.0\n"
        "!UNIT_CELL_B-AXIS= 0.0 10.0 0.0\n"
        "!UNIT_CELL_C-AXIS= 0.0 0.0 10.0\n"
        "!SPACE_GROUP_NUMBER= 1\n"
        " 1 0 0 100.0 10.0 0.0 0.0 1.0718 0.05 100.0 95.0"
        " 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
    )
    model = tmp_path / "model.res"
    model.write_text(
        "TITL response cache\n"
        "CELL 0.025079 10.0 10.0 10.0 90.0 90.0 90.0\n"
        "LATT -1\n"
        "SFAC C 1 1 1 1 1 1 1 1 0\n"
        "C1 1 0.13 0.27 0.39 11.0 0.04\n"
        "HKLF 4\n"
        "END\n"
    )
    return hkl, model


def test_response_cache_matches_core_forward_model_with_frame_offset(tmp_path: Path):
    hkl_path, model_path = _write_single_row_inputs(tmp_path)
    data = XDSData(hkl_path)
    model = parse_model(model_path)
    cfg = Config(
        n_basis=2,
        n_points=3,
        sg_window=0.0001,
        sg_window_min=0.0001,
        sg_window_max=0.0001,
        topup_d_min=2.0,
        ug_d_min=2.0,
        thickness_A=75.0,
        thickness_model="uniform_gradient",
        thickness_grad_frac=0.2,
        thickness_points=3,
        frame_offset=0.2,
        group_width_deg=1.0,
    )
    cache_dir = tmp_path / "cache"
    continuous.args_progress_every = 1000

    records = continuous.build_response_cache(
        data,
        model,
        cfg,
        np.asarray([0], dtype=int),
        sha256(model_path),
        cache_dir,
    )
    cached = continuous.predict_from_response(
        records,
        n_rows=1,
        t0=cfg.thickness_A,
        grad=cfg.thickness_grad_frac,
        points=cfg.thickness_points,
    )
    direct_port = XDSBlochPort(
        data,
        model,
        cfg,
        cache_dir=cache_dir,
        res_hash=sha256(model_path),
    )
    direct = direct_port.row_prediction(0)

    assert records[0]["center_sg_inv_A"] != pytest.approx(0.0, abs=1e-8)
    assert records[0]["sg_profile_angle_jacobian"] is True
    assert cached["dyn"][0] == pytest.approx(direct["dyn"], rel=1e-12, abs=1e-14)
    assert cached["born"][0] == pytest.approx(direct["born"], rel=1e-12, abs=1e-14)


def test_response_cache_uses_direct_exponential_for_defective_absorptive_matrix():
    hamiltonian = np.asarray(
        [[0.0, 0.0], [1.0 + 0.2j, 0.0]],
        dtype=np.complex128,
    )
    thickness = np.asarray([3.0])
    spec = continuous.propagation_spec(hamiltonian, absorptive=True)

    assert "matrix" in spec
    got = continuous.dyn_from_spec(spec, thickness, np.asarray([1.0]))
    psi0 = np.asarray([1.0 + 0.0j, 0.0 + 0.0j])
    expected = abs((expm(2j * np.pi * thickness[0] * hamiltonian) @ psi0)[1]) ** 2

    assert got == pytest.approx(expected, rel=1e-12)


def test_optimizer_proposal_above_one_is_rejected_before_promotion():
    reasons = continuous.thickness_candidate_domain_reasons(
        794.6548514,
        1.182558445,
        t_min=400.0,
        t_max=2600.0,
        grad_min=0.0,
        grad_max=1.0,
    )

    assert "thickness_gradient_outside_physical_domain" in reasons
    assert "thickness_gradient_outside_search_bounds" in reasons
    with pytest.raises(ValueError, match="thickness_gradient_outside_physical_domain"):
        continuous.thickness_grid(794.6548514, 1.182558445, 5)


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (200.0, (130.0, 270.0)),
        (3000.0, (1950.0, 4050.0)),
    ],
)
def test_local_thickness_bounds_follow_coarse_seed_without_legacy_caps(seed, expected):
    assert continuous.local_thickness_search_bounds(seed, 0.35) == pytest.approx(expected)


def test_local_thickness_bounds_apply_only_explicit_clamps():
    assert continuous.local_thickness_search_bounds(
        1000.0,
        0.35,
        t_min=800.0,
        t_max=1200.0,
    ) == pytest.approx((800.0, 1200.0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_thickness": 3000.0, "bracket_frac": 0.35, "t_max": 2600.0}, "must include"),
        ({"start_thickness": 1000.0, "bracket_frac": 0.0}, "0 < bracket_frac < 1"),
        ({"start_thickness": 1000.0, "bracket_frac": float("nan")}, "0 < bracket_frac < 1"),
    ],
)
def test_local_thickness_bounds_fail_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        continuous.local_thickness_search_bounds(**kwargs)


def test_cli_rejects_search_bound_above_physical_gradient_domain(monkeypatch, capsys, tmp_path: Path):
    out = tmp_path / "must_not_be_created"
    monkeypatch.setattr(
        sys,
        "argv",
        ["continuous_thickness_polish.py", "--grad-max", "1.2", "--out-dir", str(out)],
    )

    with pytest.raises(SystemExit) as exc:
        continuous.main()

    assert exc.value.code == 2
    assert "grad_max <= 1" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    ("thickness", "grad", "reason"),
    [
        (float("nan"), 0.5, "nonfinite_thickness"),
        (800.0, float("nan"), "nonfinite_thickness_gradient"),
        (0.0, 0.5, "nonpositive_thickness"),
    ],
)
def test_continuous_thickness_candidate_validation_fails_closed(thickness, grad, reason):
    assert reason in continuous.thickness_candidate_domain_reasons(thickness, grad)


def test_valid_promoted_nuisance_remains_in_downstream_engine_domain():
    args = SimpleNamespace(
        frame_offset=0.0,
        group_width_deg=0.888,
        detector_scale_bins=8,
        sigma_a=0.0,
        sigma_b_corr=0.0,
        sigma_c_peak=0.0,
        matrix_convention="j_minus_i",
        row_resolution_b=0.0,
        scale_smoothness=0.0,
    )

    nuisance = continuous.make_nuisance(args, 794.6548514, 1.0)
    downstream = Config(
        thickness_A=nuisance.thickness_A,
        thickness_model=nuisance.thickness_model,
        thickness_grad_frac=nuisance.thickness_grad_frac,
    )

    assert nuisance.thickness_grad_frac == 1.0
    assert downstream.thickness_grad_frac == 1.0
