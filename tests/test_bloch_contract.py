import json
from types import SimpleNamespace

import gemmi
import numpy as np
import pytest
from scipy.linalg import expm

from grond.legacy.engine.bloch_engine import (
    Config,
    XDSBlochPort,
    _jax_matrix_bucket_size,
    _real_embedding_generator_batch,
    _stack_square_matrices,
    alternate_hand_model,
    atom_dw,
    cif_symmetry_records,
    fcalc_res,
    fcalc_res_grid,
    frozen_kernel_contract,
    load_geometry_path_contract,
    parse_model,
    promoted_bethe_effective_hamiltonian,
    relativistic_gamma_from_wavelength,
    rotation_matrix,
    row_plan_cache_validation_errors,
    split_masks,
)
from grond.legacy.engine.build_dynrows_xds import gvec_for_hkl, sg_for_hkl
from grond.legacy.engine.native_row_regularize import Nuisance, evaluate_trial

ELECTRON_SFAC_C = "SFAC C 1 1 1 1 1 1 1 1 0"


def test_raw_row_geometry_uses_full_s_r_f_a_chain():
    axis = np.array([1.0, 0.0, 0.0])
    setting = rotation_matrix(np.array([0.0, 1.0, 0.0]), -17.0)
    fixed = rotation_matrix(np.array([0.0, 0.0, 1.0]), 23.0)
    scan_a = np.diag([0.10, 0.125, 0.15])
    omega = 11.0
    hkl = np.array([2, -1, 1])
    s0 = np.array([0.0, 0.0, -40.0])
    data = SimpleNamespace(
        axis=axis,
        setting_rotation=setting,
        fixed_rotation=fixed,
        A=np.eye(3) * 99.0,
        A_for_omega=lambda _omega: scan_a,
        s0=s0,
        k=float(np.linalg.norm(s0)),
    )

    expected_g = setting @ rotation_matrix(axis, omega) @ fixed @ scan_a @ hkl
    expected_sg = (
        data.k**2 - float(np.dot(s0 + expected_g, s0 + expected_g))
    ) / (2.0 * data.k)

    np.testing.assert_allclose(gvec_for_hkl(data, hkl, omega), expected_g)
    assert sg_for_hkl(data, hkl, omega) == pytest.approx(expected_sg, abs=1e-12)


def write_shelx_model(path, body: list[str]):
    path.write_text(
        "\n".join(
            [
                "TITL parser contract",
                "CELL 0.025079 10 11 12 90 90 90",
                "LATT -1",
                ELECTRON_SFAC_C,
                *body,
                "HKLF 4",
                "END",
                "",
            ]
        )
    )
    return path


def test_shelx_parser_resolves_fvar_and_ignores_q_peaks_and_restraints(tmp_path):
    path = write_shelx_model(
        tmp_path / "model.res",
        [
            "FVAR 1.0 0.6",
            "C1 1 10.1 10.2 10.3 21.0 10.05",
            "C2 1 0.4 0.5 0.6 -21.0 0.06",
            "MPLA 6 C1 C2 C3 C4 C5 C6",
            "SUMP 1 0.001 1 2 1 3",
            "Q1 1 0.7 0.8 0.9 11.0 0.05",
        ],
    )

    model = parse_model(path)

    assert [atom["name"] for atom in model["atoms"]] == ["C1", "C2"]
    assert model["atoms"][0]["x"] == pytest.approx(0.1)
    assert model["atoms"][0]["y"] == pytest.approx(0.2)
    assert model["atoms"][0]["z"] == pytest.approx(0.3)
    assert model["atoms"][0]["occ"] == pytest.approx(0.6)
    assert model["atoms"][0]["uiso"] == pytest.approx(0.05)
    assert model["atoms"][1]["occ"] == pytest.approx(0.4)


def test_shelx_parser_rejects_missing_free_variable(tmp_path):
    path = write_shelx_model(
        tmp_path / "missing_fvar.res",
        [
            "FVAR 1.0",
            "C1 1 0.1 0.2 0.3 21.0 0.05",
        ],
    )

    with pytest.raises(ValueError, match="references FVAR 2"):
        parse_model(path)


def test_shelx_parser_rejects_missing_atoms_and_short_sfac(tmp_path):
    no_atoms = write_shelx_model(tmp_path / "no_atoms.res", ["FVAR 1.0"])
    with pytest.raises(ValueError, match="No atoms"):
        parse_model(no_atoms)

    short_sfac = tmp_path / "short_sfac.res"
    short_sfac.write_text(
        "CELL 0.025079 10 11 12 90 90 90\n"
        "SFAC C H O\n"
        "C1 1 0.1 0.2 0.3 11 0.05\n"
        "HKLF 4\n"
    )
    with pytest.raises(ValueError, match="four explicit"):
        parse_model(short_sfac)


def test_cif_parser_rejects_missing_scattering_coefficients(tmp_path):
    path = tmp_path / "ordinary.cif"
    path.write_text(
        "data_test\n"
        "_cell_length_a 10\n"
        "_cell_length_b 11\n"
        "_cell_length_c 12\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "loop_\n"
        "_atom_type_symbol\n"
        "C\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "C1 C 0.1 0.2 0.3\n"
    )

    with pytest.raises(ValueError, match="does not identify its scattering coefficients as electron"):
        parse_model(path)


def test_cif_parser_accepts_case_insensitive_electron_coefficient_tags(tmp_path):
    path = tmp_path / "electron.cif"
    path.write_text(
        """data_electron
_grond_scattering_factor_kind electron
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 90
_cell_angle_beta 92
_cell_angle_gamma 87
_space_group_IT_number 1

loop_
_atom_type_symbol
_atom_type_scat_cromer_mann_a1
_atom_type_scat_cromer_mann_b1
_atom_type_scat_cromer_mann_a2
_atom_type_scat_cromer_mann_b2
_atom_type_scat_cromer_mann_a3
_atom_type_scat_cromer_mann_b3
_atom_type_scat_cromer_mann_a4
_atom_type_scat_cromer_mann_b4
_atom_type_scat_cromer_mann_c
C 1 1 1 1 1 1 1 1 0

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_adp_type
_atom_site_u_iso_or_equiv
C1 C 0.1 0.2 0.3 Uiso 0.04
"""
    )

    model = parse_model(path)

    assert model["sfac"][0]["pairs"] == [(1.0, 1.0)] * 4
    assert model["atoms"][0]["uiso"] == pytest.approx(0.04)
    assert model["cell"] == pytest.approx((10.0, 11.0, 12.0, 90.0, 92.0, 87.0))


def test_cif_symmetry_accepts_case_insensitive_operation_tags():
    block = gemmi.cif.read_string(
        """data_symmetry
loop_
_SPACE_GROUP_SYMOP_OPERATION_XYZ
'x,y,z'
'-x,-y,z'
"""
    ).sole_block()

    operations = cif_symmetry_records(block)

    assert len(operations) == 2
    assert any(np.array_equal(np.asarray(op["rot"]), np.diag([-1, -1, 1])) for op in operations)


def test_cif_symmetry_resolves_hm_name_without_operation_loop_or_it_number():
    block = gemmi.cif.read_string(
        """data_symmetry
_space_group_name_H-M_alt 'P 21 21 21'
"""
    ).sole_block()

    operations = cif_symmetry_records(block)

    assert len(operations) == 4


def test_cif_symmetry_rejects_unrecognized_declared_name():
    block = gemmi.cif.read_string(
        """data_symmetry
_space_group_name_H-M_alt 'not a space group'
"""
    ).sole_block()

    with pytest.raises(ValueError, match="unrecognized space-group name"):
        cif_symmetry_records(block)


def test_structure_factor_deduplicates_special_position_mates():
    model = {
        "cell": (10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        "sfac": [{"element": "C", "pairs": [(1.0, 0.0)] * 4, "c": 0.0}],
        "atoms": [
            {
                "name": "C1",
                "sfac_index": 0,
                "element": "C",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "occ": 1.0,
                "adp_type": "iso",
                "uiso": 0.0,
            }
        ],
        "symmetry_ops": [
            {"rot": np.eye(3).tolist(), "trans": [0.0, 0.0, 0.0]},
            {"rot": (-np.eye(3)).tolist(), "trans": [0.0, 0.0, 0.0]},
        ],
    }

    scalar = fcalc_res(model, (1, 0, 0))
    grid = fcalc_res_grid(model, np.asarray([1]), np.asarray([0]), np.asarray([0]))[0]

    assert scalar == pytest.approx(4.0 + 0.0j)
    assert grid == pytest.approx(scalar)


def test_relativistic_gamma_tracks_electron_wavelength():
    assert relativistic_gamma_from_wavelength(0.025079) == pytest.approx(1.39139, rel=2e-5)
    assert relativistic_gamma_from_wavelength(0.019687) == pytest.approx(1.58711, rel=2e-5)
    with pytest.raises(ValueError, match="wavelength"):
        relativistic_gamma_from_wavelength(0.0)


def test_anisotropic_dw_is_rotated_for_each_symmetry_mate():
    cell = (7.0, 8.0, 9.0, 90.0, 90.0, 90.0)
    atom = {
        "name": "C1",
        "sfac_index": 0,
        "element": "C",
        "x": 0.123,
        "y": 0.234,
        "z": 0.345,
        "occ": 1.0,
        "adp_type": "anis",
        "u11": 0.04,
        "u22": 0.05,
        "u33": 0.06,
        "u23": 0.015,
        "u13": 0.012,
        "u12": 0.011,
    }
    operations = [
        {"rot": np.diag([1.0, 1.0, 1.0]).tolist(), "trans": [0.0, 0.0, 0.0]},
        {"rot": np.diag([-1.0, -1.0, 1.0]).tolist(), "trans": [0.5, 0.0, 0.5]},
        {"rot": np.diag([-1.0, 1.0, -1.0]).tolist(), "trans": [0.0, 0.5, 0.5]},
        {"rot": np.diag([1.0, -1.0, -1.0]).tolist(), "trans": [0.5, 0.5, 0.0]},
    ]
    model = {
        "cell": cell,
        "sfac": [{"element": "C", "pairs": [(0.0, 0.0)] * 4, "c": 1.0}],
        "atoms": [atom],
        "symmetry_ops": operations,
    }
    hkl = np.asarray([2, 1, 3], dtype=int)
    xyz = np.asarray([atom["x"], atom["y"], atom["z"]], dtype=float)
    expected = 0.0 + 0.0j
    for operation in operations:
        rotation = np.asarray(operation["rot"], dtype=float)
        translation = np.asarray(operation["trans"], dtype=float)
        mate_hkl = rotation.T @ hkl
        mate_xyz = rotation @ xyz + translation
        expected += atom_dw(atom, mate_hkl, cell) * np.exp(2j * np.pi * np.dot(hkl, mate_xyz))

    scalar = fcalc_res(model, hkl)
    grid = fcalc_res_grid(model, hkl[0:1], hkl[1:2], hkl[2:3])[0]

    assert expected == pytest.approx(0.00437572583254 + 0.105949076608j, rel=1e-10)
    assert scalar == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert grid == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert fcalc_res(model, -hkl) == pytest.approx(np.conj(expected), rel=1e-12, abs=1e-14)


def test_matrix_convention_is_recorded_in_kernel_contract():
    default = frozen_kernel_contract(Config())
    alternate = frozen_kernel_contract(Config(matrix_convention="i_minus_j"))
    assert default["hamiltonian"]["matrix_convention"] == "j_minus_i"
    assert default["hamiltonian"]["offdiagonal_sign"] == "basis[j] - basis[i]"
    assert alternate["hamiltonian"]["matrix_convention"] == "i_minus_j"
    assert alternate["hamiltonian"]["offdiagonal_sign"] == "basis[i] - basis[j]"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_basis": 1}, "n_basis"),
        ({"n_points": 0}, "n_points"),
        ({"thickness_A": -10.0}, "thickness_A"),
        ({"thickness_grad_frac": 1.01}, "thickness_grad_frac"),
        ({"group_width_deg": 0.0}, "group_width_deg"),
        ({"free_fraction": 1.0}, "free_fraction"),
        ({"absorption_mu": -0.1}, "absorption_mu"),
        ({"sg_window": 0.02}, "silent clipping"),
        ({"profile_quadrature": "unknown"}, "profile_quadrature"),
        ({"compute_backend": "unknown"}, "compute_backend"),
    ],
)
def test_config_rejects_invalid_numerical_contracts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Config(**kwargs)


def test_small_held_out_split_keeps_work_and_free_sets_nonempty():
    masks = split_masks(np.asarray([[1, 0, 0], [0, 1, 0]]), free_fraction=0.9, seed=7)
    assert np.count_nonzero(masks["work"]) == 1
    assert np.count_nonzero(masks["free"]) == 1


def test_held_out_split_keeps_symmetry_and_friedel_mates_together():
    symmetry_ops = [
        {"rot": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], "trans": [0.5, 0.0, 0.5]},
    ]
    hkls = np.asarray(
        [
            [1, 2, 3],
            [-1, -2, 3],
            [-1, -2, -3],
            [4, 1, 2],
        ]
    )

    masks = split_masks(hkls, free_fraction=0.5, seed=3, symmetry_ops=symmetry_ops)

    assert len(set(masks["free"][:3])) == 1
    assert masks["free"][3] != masks["free"][0]


def test_alternate_hand_inverts_non_half_symmetry_translations():
    model = {
        "cell": (10.0, 11.0, 12.0, 90.0, 90.0, 90.0),
        "sfac": [{"element": "C", "pairs": [(1.0, 1.0)] * 4, "c": 0.0}],
        "atoms": [
            {
                "name": "C1",
                "sfac_index": 0,
                "element": "C",
                "x": 0.13,
                "y": 0.27,
                "z": 0.39,
                "occ": 1.0,
                "adp_type": "iso",
                "uiso": 0.04,
            }
        ],
        "symmetry_ops": [
            {"rot": np.eye(3).tolist(), "trans": [0.25, 0.0, 0.0]},
        ],
    }

    inverted = alternate_hand_model(model)

    assert inverted["symmetry_ops"][0]["trans"] == [0.75, 0.0, 0.0]
    assert np.allclose(fcalc_res(inverted, (1, 2, 3)), np.conj(fcalc_res(model, (1, 2, 3))))


def test_local_sg_solver_does_not_jump_to_distant_ewald_crossing():
    port = object.__new__(XDSBlochPort)
    port.data = SimpleNamespace(oscillation_range=1.0)
    port.sg_for_hkls = lambda _hkls, omega: np.asarray([omega * (omega - 40.0)])

    assert port.solve_omega_for_sg((1, 0, 0), 39.0, 0.0) == pytest.approx(40.0)
    with pytest.raises(ValueError, match="no local"):
        port.solve_omega_for_sg((1, 0, 0), 10.0, 0.0)


def test_fixed_xds_sg_solver_matches_numerical_local_root():
    axis = np.asarray([0.31, -0.82, 0.48], dtype=float)
    axis /= np.linalg.norm(axis)
    axis_cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    a_matrix = np.asarray(
        [[0.111, 0.003, -0.002], [0.0, 0.093, 0.004], [0.001, 0.0, 0.077]]
    )
    data = SimpleNamespace(
        header={"GROND_SOURCE": "XDS"},
        A=a_matrix,
        A_at_scan_points=None,
        A_for_omega=lambda _omega: a_matrix,
        axis=axis,
        axis_eye=np.eye(3),
        axis_cross=axis_cross,
        axis_cross2=axis_cross @ axis_cross,
        setting_rotation=rotation_matrix(np.asarray([0.0, 1.0, 0.0]), -11.0),
        fixed_rotation=rotation_matrix(np.asarray([0.0, 0.0, 1.0]), 17.0),
        s0=np.asarray([0.0, 0.0, -39.873]),
        k=39.873,
        oscillation_range=0.5,
    )
    port = object.__new__(XDSBlochPort)
    port.data = data
    port.geometry_path = None
    hkl = np.asarray([2, -3, 1])
    omega0 = 7.0
    expected = 7.43
    target = float(port.sg_for_hkls(np.asarray([hkl]), expected)[0])

    analytic = port.solve_omega_for_sg(hkl, omega0, target, max_delta_deg=1.0)
    numerical = port._numerical_omega_for_sg(
        hkl, omega0, target, max_delta_deg=1.0
    )

    assert analytic == pytest.approx(expected, abs=1e-10)
    assert analytic == pytest.approx(numerical, abs=1e-10)
    assert float(port.sg_for_hkls(np.asarray([hkl]), analytic)[0]) == pytest.approx(
        target, abs=1e-12
    )
    step = 1e-5
    numerical_derivative = (
        float(port.sg_for_hkls(np.asarray([hkl]), expected + step)[0])
        - float(port.sg_for_hkls(np.asarray([hkl]), expected - step)[0])
    ) / (2.0 * step)
    assert port.sg_derivative_per_deg(hkl, expected) == pytest.approx(
        numerical_derivative, abs=1e-9
    )


def test_scan_varying_dials_sg_solver_keeps_numerical_fallback(monkeypatch):
    port = object.__new__(XDSBlochPort)
    port.data = SimpleNamespace(
        header={"DIALS_EXPERIMENT_UUID": "test-experiment"},
        oscillation_range=0.5,
    )
    port.geometry_path = None
    port.sg_for_hkls = lambda _hkls, omega: np.asarray([float(omega) - 2.0])
    calls = []
    original = port._numerical_omega_for_sg

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(port, "_numerical_omega_for_sg", counted)

    assert port.solve_omega_for_sg((1, 0, 0), 1.5, 0.0) == pytest.approx(2.0)
    assert len(calls) == 1
    assert port.sg_derivative_per_deg((1, 0, 0), 1.5) == pytest.approx(1.0)


def test_row_profile_stays_centered_on_frame_offset_geometry():
    def make_plan(frame_offset):
        port = object.__new__(XDSBlochPort)
        port.data = SimpleNamespace(hkl=np.asarray([[1, 0, 0]]), zcal=np.asarray([0.0]))
        port.cfg = SimpleNamespace(
            frame_offset=frame_offset,
            thickness_model="single",
            thickness_sigma_frac=0.0,
        )
        port.thick = np.asarray([1.0])
        port.thick_w = np.asarray([1.0])
        port.omega_from_z = lambda z: float(z) + float(frame_offset)
        port.sg_for_hkls = lambda hkls, omega: np.asarray(
            [0.0 if np.array_equal(hkl, [0, 0, 0]) else float(omega) for hkl in hkls]
        )
        port.solve_omega_for_sg = lambda _hkl, _omega0, target: float(target)
        port.sg_derivative_per_deg = lambda _hkl, _omega: 1.0
        port.select_basis = lambda _hkl, _omega, row_idx=None: np.asarray([[0, 0, 0], [1, 0, 0]])
        port.row_sg_window = lambda *_args: (0.2, {})
        port.row_profile_sg_points = lambda *_args: (
            np.asarray([-0.1, 0.1]),
            np.asarray([0.5, 0.5]),
            {},
        )
        port.hamiltonian_static_part = lambda *_args, **_kwargs: np.zeros((2, 2), dtype=np.complex128)
        port.absorption_enabled = lambda: False
        port.row_delta_b = lambda _row_idx: 0.0
        return port.row_prediction_plan(0)

    unshifted = make_plan(0.0)
    shifted = make_plan(0.25)

    np.testing.assert_allclose([matrix[1, 1].real for matrix in unshifted["_dyn_matrices"]], [-0.1, 0.1])
    np.testing.assert_allclose([matrix[1, 1].real for matrix in shifted["_dyn_matrices"]], [0.15, 0.35])


def test_defective_nonhermitian_propagation_matches_direct_exponential():
    port = object.__new__(XDSBlochPort)
    port.cfg = SimpleNamespace(compute_backend="scipy")
    port.absorption_enabled = lambda: True
    hamiltonian = np.asarray([[0.0, 0.0], [1.0 + 0.2j, 0.0]], dtype=np.complex128)
    thickness = 3.0

    got = port.propagate_direct_to_target(hamiltonian, thickness)
    psi0 = np.asarray([1.0 + 0.0j, 0.0 + 0.0j])
    expected = abs((expm(2j * np.pi * thickness * hamiltonian) @ psi0)[1]) ** 2

    assert got == pytest.approx(expected, rel=1e-12)


def test_row_plan_cache_validation_rejects_stale_metadata():
    indices = np.asarray([3, 7, 11], dtype=np.int64)
    meta = {
        "cache_schema": 1,
        "key": "expected",
        "hkl_sha256": "abc",
        "hkl_transform": "as_input",
        "indices_sha256": "idx",
        "n_basis": 24,
        "n_points": 10,
        "profile_quadrature": "uniform",
        "profile_accumulation": "sum",
    }
    valid_cache = {
        "meta": dict(meta),
        "indices": indices.copy(),
        "basis_offsets": np.asarray([0, 2, 4, 6], dtype=np.int64),
        "basis_flat": np.zeros((6, 3), dtype=np.int32),
        "profile_offsets": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "profile_weights_flat": np.ones(3, dtype=float),
        "sg_offsets": np.asarray([0, 2, 4, 6], dtype=np.int64),
        "sg_basis_flat": np.zeros(6, dtype=float),
        "diff_offsets": np.asarray([0, 4, 8, 12], dtype=np.int64),
        "diff_idx_flat": np.zeros(12, dtype=np.int64),
        "unique_diffs": np.zeros((1, 3), dtype=np.int32),
        "unique_inv_d2": np.zeros(1, dtype=float),
        "center_sg": np.zeros(3, dtype=float),
        "omega_center": np.zeros(3, dtype=float),
        "omega_bragg": np.zeros(3, dtype=float),
        "omega_delta": np.zeros(3, dtype=float),
        "sg_window": np.ones(3, dtype=float),
        "sg_domega": np.ones(3, dtype=float),
        "sg_dframe": np.ones(3, dtype=float),
        "sg_window_raw": np.ones(3, dtype=float),
        "sg_window_half_frames": np.ones(3, dtype=float),
        "sg_window_clipped": np.zeros(3, dtype=bool),
        "sg_jitter_sigma": np.zeros(3, dtype=float),
        "sg_jitter_d_inv": np.zeros(3, dtype=float),
        "sg_jitter_frame_fraction": np.zeros(3, dtype=float),
        "sg_jitter_kernel_points": np.ones(3, dtype=np.int32),
        "sg_profile_points_total": np.ones(3, dtype=np.int32),
        "sg_profile_weight_sum": np.ones(3, dtype=float),
        "sg_profile_same_branch_continuation_used": np.asarray([False, True, False]),
        "sg_profile_same_branch_continuation_segments": np.asarray([1, 2, 1], dtype=np.int32),
        "sg_residual_max": np.zeros(3, dtype=float),
        "dose_delta_b": np.zeros(3, dtype=float),
    }
    assert row_plan_cache_validation_errors(valid_cache, indices, "expected", meta) == []

    sentinel_meta = dict(meta, sg_window_mode="fixed", sg_jitter_mode="none")
    sentinel_cache = dict(
        valid_cache,
        meta=dict(sentinel_meta),
        sg_window_half_frames=np.full(3, np.nan),
        sg_jitter_d_inv=np.full(3, np.nan),
        sg_jitter_frame_fraction=np.full(3, np.nan),
    )
    assert row_plan_cache_validation_errors(
        sentinel_cache,
        indices,
        "expected",
        sentinel_meta,
    ) == []

    missing_contract = dict(sentinel_cache, meta=dict(meta))
    errors = row_plan_cache_validation_errors(missing_contract, indices, "expected", meta)
    assert "sg_window_half_frames is incomplete or non-finite" in errors
    assert "sg_jitter_d_inv is incomplete or non-finite" in errors
    assert "sg_jitter_frame_fraction is incomplete or non-finite" in errors

    applicable_window_meta = dict(sentinel_meta, sg_window_mode="dials_bbox")
    applicable_window = dict(sentinel_cache, meta=dict(applicable_window_meta))
    errors = row_plan_cache_validation_errors(
        applicable_window,
        indices,
        "expected",
        applicable_window_meta,
    )
    assert "sg_window_half_frames is incomplete or non-finite" in errors

    applicable_jitter_meta = dict(sentinel_meta, sg_jitter_mode="gaussian")
    applicable_jitter = dict(sentinel_cache, meta=dict(applicable_jitter_meta))
    errors = row_plan_cache_validation_errors(
        applicable_jitter,
        indices,
        "expected",
        applicable_jitter_meta,
    )
    assert "sg_jitter_d_inv is incomplete or non-finite" in errors
    assert "sg_jitter_frame_fraction is incomplete or non-finite" in errors

    invalid_geometry = dict(sentinel_cache, center_sg=np.asarray([0.0, np.nan, 0.0]))
    errors = row_plan_cache_validation_errors(
        invalid_geometry,
        indices,
        "expected",
        sentinel_meta,
    )
    assert "center_sg is incomplete or non-finite" in errors

    invalid_optional = dict(
        sentinel_cache,
        sg_jitter_d_inv=np.asarray([np.nan, np.inf, np.nan]),
    )
    errors = row_plan_cache_validation_errors(
        invalid_optional,
        indices,
        "expected",
        sentinel_meta,
    )
    assert "sg_jitter_d_inv is incomplete or non-finite" in errors

    stale_cache = dict(valid_cache)
    stale_cache["meta"] = dict(meta, n_basis=30)
    errors = row_plan_cache_validation_errors(stale_cache, indices, "expected", meta)
    assert "metadata mismatch: n_basis" in errors

    broken_cache = dict(valid_cache)
    broken_cache.pop("sg_basis_flat")
    broken_cache["basis_offsets"] = np.asarray([0, 1, 3, 5], dtype=np.int64)
    errors = row_plan_cache_validation_errors(broken_cache, indices, "expected", meta)
    assert any("missing arrays" in err for err in errors)
    assert "basis offsets do not contain direct and target beams for every row" in errors

    zero_weight_cache = dict(valid_cache)
    zero_weight_cache["profile_weights_flat"] = np.zeros(3, dtype=float)
    errors = row_plan_cache_validation_errors(zero_weight_cache, indices, "expected", meta)
    assert "profile weights contain an all-zero row profile" in errors
    assert "profile weight sums do not match cached row metadata" in errors

    wrong_sg_shape = dict(valid_cache)
    wrong_sg_shape["sg_offsets"] = np.asarray([0, 1, 3, 6], dtype=np.int64)
    errors = row_plan_cache_validation_errors(wrong_sg_shape, indices, "expected", meta)
    assert "Sg offsets do not contain one basis vector per profile point" in errors

    wrong_diff_shape = dict(valid_cache)
    wrong_diff_shape["diff_offsets"] = np.asarray([0, 5, 9, 12], dtype=np.int64)
    errors = row_plan_cache_validation_errors(wrong_diff_shape, indices, "expected", meta)
    assert "difference offsets do not contain one index per basis-matrix entry" in errors

    invalid_continuation = dict(valid_cache)
    invalid_continuation["sg_profile_same_branch_continuation_segments"] = np.ones(3, dtype=np.int32)
    errors = row_plan_cache_validation_errors(invalid_continuation, indices, "expected", meta)
    assert "profile continuation rows must record at least two strict local segments" in errors


def test_cached_static_hamiltonian_uses_frozen_dose_delta_b():
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(dose_b_mode="linear_frame", dose_b_per_frame=99.0)
    port.data = type("DummyData", (), {"k": 2.0})()

    basis = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=int)
    diff_idx = np.asarray([0, 1, 2, 3], dtype=np.int64)
    unique_diffs = np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 0]], dtype=np.int32)
    unique_inv_d2 = np.asarray([0.0, 4.0, 4.0, 0.0], dtype=float)
    unique_ug = np.asarray([0.0, 1.0 + 0.0j, 1.0 + 0.0j, 0.0], dtype=np.complex128)

    H = port.cached_static_hamiltonian(
        basis,
        diff_idx,
        unique_diffs,
        unique_inv_d2,
        unique_ug,
        row_idx=0,
        absorb=False,
        dose_delta_b=2.0,
    )
    expected = np.exp(-2.0) / (2.0 * port.data.k)
    assert np.isclose(H[0, 1], expected)
    assert np.isclose(H[1, 0], expected)


def profile_points(cfg: Config):
    port = object.__new__(XDSBlochPort)
    port.cfg = cfg
    return port.row_profile_sg_points(0, np.asarray([1, 0, 0]), 0.004)


def test_profile_quadrature_defaults_to_summed_subexposures():
    points, weights, meta = profile_points(Config(n_points=5))
    assert len(points) == 5
    assert np.isclose(float(np.sum(weights)), 5.0)
    assert meta["sg_profile_accumulation"] == "sum"
    assert meta["sg_profile_quadrature"] == "uniform"
    assert meta["sg_profile_weight_sum"] == 5.0


def test_profile_quadrature_mean_is_legacy_normalized_average():
    _, weights, meta = profile_points(Config(n_points=5, profile_accumulation="mean"))
    assert np.isclose(float(np.sum(weights)), 1.0)
    assert meta["sg_profile_accumulation"] == "mean"


def test_profile_quadrature_rules_have_stable_total_weight():
    for rule in ["uniform", "trapezoid", "gauss_legendre"]:
        points, weights, meta = profile_points(Config(n_points=7, profile_quadrature=rule))
        assert len(points) == 7
        assert np.isclose(float(np.sum(weights)), 7.0)
        assert meta["sg_profile_quadrature"] == rule


def test_single_profile_point_is_centered():
    points, weights, _ = profile_points(Config(n_points=1))
    assert np.allclose(points, [0.0])
    assert np.allclose(weights, [1.0])


def test_direct_angular_mosaicity_is_normalized_and_bounded():
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(
        mosaicity_mode="gaussian_omega",
        mosaicity_sigma_deg=0.02,
        mosaicity_max_sigma_deg=0.05,
        mosaicity_points=3,
    )

    offsets, weights, meta = port.row_mosaicity_angle_points()

    assert len(offsets) == 3
    assert np.allclose(offsets, -offsets[::-1])
    assert np.isclose(np.sum(weights), 1.0)
    assert meta["mosaicity_sigma_deg"] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="exceeds the explicit"):
        Config(
            mosaicity_mode="gaussian_omega",
            mosaicity_sigma_deg=0.051,
            mosaicity_max_sigma_deg=0.05,
        )


def test_gaussian_cone_quadrature_is_normalized_and_enclosed():
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(
        profile_coordinate="direct_angle",
        mosaicity_mode="gaussian_cone",
        mosaicity_sigma_deg=0.02,
        mosaicity_support_sigma_deg=0.04,
        mosaicity_points=5,
        mosaicity_support_width_sigma=4.0,
    )

    points, weights, meta = port.row_mosaicity_cone_points()

    assert points.shape == (25, 2)
    assert np.isclose(float(np.sum(weights)), 1.0)
    assert np.allclose(np.sum(points * weights[:, None], axis=0), 0.0, atol=1e-15)
    assert meta["mosaicity_cone_quadrature"] == "tensor_gauss_hermite_5x5"
    with pytest.raises(ValueError, match="support_sigma_deg"):
        Config(
            profile_coordinate="direct_angle",
            mosaicity_mode="gaussian_cone",
            mosaicity_sigma_deg=0.03,
            mosaicity_support_sigma_deg=0.02,
        )
    with pytest.raises(ValueError, match="requires profile_coordinate=direct_angle"):
        Config(
            mosaicity_mode="gaussian_cone",
            mosaicity_sigma_deg=0.01,
            mosaicity_support_sigma_deg=0.01,
        )


def test_gaussian_cone_zero_width_preserves_profile_exactly():
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(
        profile_coordinate="direct_angle",
        mosaicity_mode="gaussian_cone",
        mosaicity_sigma_deg=0.0,
        mosaicity_support_sigma_deg=0.04,
    )
    port.row_profile_omega_points = lambda *_args: (
        np.asarray([9.5, 10.0, 10.5], dtype=float),
        np.asarray([0.5, 1.0, 0.5], dtype=float),
        np.asarray([], dtype=float),
        {"sg_profile_points_total": 3, "sg_profile_weight_sum": 2.0},
        {"mosaicity_mode": "gaussian_cone"},
    )

    omegas, tilts, weights, residuals, profile_meta, mosaic_meta = (
        port.row_profile_orientation_points(
            0, np.asarray([1, 0, 0]), 10.0, 0.0, 0.004
        )
    )

    assert np.array_equal(omegas, np.asarray([9.5, 10.0, 10.5]))
    assert np.array_equal(tilts, np.zeros((3, 2)))
    assert np.array_equal(weights, np.asarray([0.5, 1.0, 0.5]))
    assert residuals.size == 0
    assert profile_meta["sg_profile_points_total"] == 3
    assert mosaic_meta["mosaicity_cone_points"] == 1


def test_common_cone_basis_is_independent_of_candidate_sigma():
    def make_port(candidate_sigma):
        port = object.__new__(XDSBlochPort)
        port.cfg = Config(
            n_basis=4,
            profile_coordinate="direct_angle",
            mosaicity_mode="gaussian_cone",
            mosaicity_sigma_deg=candidate_sigma,
            mosaicity_support_sigma_deg=0.04,
            mosaicity_support_points=3,
            max_support_sg=0.02,
        )
        port.pool_hkl = np.asarray(
            [[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [0, 2, 0]],
            dtype=int,
        )
        port.pool_ug = np.asarray([1.0, 0.8, 0.7, 0.6, 0.5], dtype=complex)
        port.sg_derivative_per_deg = lambda *_args: 0.02

        def sg_at_orientation(hkls, omega, along, transverse):
            hkls = np.asarray(hkls, dtype=float)
            return (
                0.01 * hkls[:, 0]
                - 0.006 * hkls[:, 1]
                + 0.001 * float(omega)
                + 0.1 * float(along)
                - 0.05 * float(transverse)
            )

        port.sg_for_hkls_at_orientation = sg_at_orientation
        return port

    zero_basis, zero_meta = make_port(0.0).select_basis_common_cone(
        np.asarray([1, 0, 0]), 0.0, 0.004
    )
    finite_basis, finite_meta = make_port(0.02).select_basis_common_cone(
        np.asarray([1, 0, 0]), 0.0, 0.004
    )

    assert np.array_equal(zero_basis, finite_basis)
    assert zero_meta == finite_meta
    assert zero_meta["mosaicity_basis_mode"] == "common_maximum_cone"


def test_common_profile_basis_includes_edge_beam_and_is_q_independent():
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(
        n_basis=4,
        profile_basis_mode="common_profile",
        mosaicity_support_points=3,
        max_support_sg=0.005,
    )
    port.pool_hkl = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=int,
    )
    port.pool_ug = np.asarray([1.0, 0.8, 0.2], dtype=complex)

    def sg_at_orientation(hkls, omega, _along, _transverse):
        hkls = np.asarray(hkls, dtype=int)
        out = np.full(len(hkls), 0.1, dtype=float)
        out[np.all(hkls == np.asarray([1, 0, 0]), axis=1)] = 0.0
        out[np.all(hkls == np.asarray([0, 1, 0]), axis=1)] = (
            0.02 * (1.0 - float(omega))
        )
        return out

    port.sg_for_hkls_at_orientation = sg_at_orientation
    port.sg_for_hkls = lambda hkls, omega: sg_at_orientation(hkls, omega, 0.0, 0.0)
    target = np.asarray([1, 0, 0])
    tilts20 = np.zeros((20, 2), dtype=float)
    tilts41 = np.zeros((41, 2), dtype=float)
    basis20, meta20 = port.select_basis_for_profile(
        target,
        0.0,
        0.004,
        np.linspace(-1.0, 1.0, 20),
        tilts20,
    )
    basis41, meta41 = port.select_basis_for_profile(
        target,
        0.0,
        0.004,
        np.linspace(-1.0, 1.0, 41),
        tilts41,
    )

    assert np.array_equal(basis20, basis41)
    assert [0, 1, 0] in basis20.tolist()
    assert meta20["profile_basis_mode"] == "common_profile"
    assert meta20["profile_basis_orientation_count"] == 3
    assert meta20 == meta41


def test_direct_angular_mosaicity_does_not_request_extra_sg_roots():
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(
        mosaicity_mode="gaussian_omega",
        mosaicity_sigma_deg=0.02,
        mosaicity_points=3,
    )
    port.row_profile_sg_points = lambda *_args: (
        np.asarray([-0.001, 0.001]),
        np.asarray([0.5, 0.5]),
        {},
    )
    solved_targets = []

    def solve_target(_row_idx, _hkl, _omega_center, target_sg, _meta):
        solved_targets.append(float(target_sg))
        return float(target_sg)

    port.solve_sg_profile_target = solve_target
    port.sg_for_hkls = lambda _hkls, omega: np.asarray([omega], dtype=float)
    port.sg_derivative_per_deg = lambda _hkl, _omega: 2.0

    omegas, weights, residuals, _sg_meta, mosaicity_meta = port.row_profile_omega_points(
        0,
        np.asarray([1, 0, 0]),
        0.0,
        0.0,
        0.004,
    )

    assert solved_targets == pytest.approx([-0.001, 0.001])
    assert len(omegas) == 6
    assert len(weights) == 6
    assert np.isclose(np.sum(weights), 1.0)
    assert np.allclose(residuals, 0.0)
    assert mosaicity_meta["mosaicity_kernel_points"] == 3


def test_direct_angle_profile_requires_no_sg_root():
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(
        n_points=5,
        profile_coordinate="direct_angle",
        profile_quadrature="gauss_legendre",
        mosaicity_mode="gaussian_omega",
        mosaicity_sigma_deg=0.01,
        mosaicity_points=3,
    )
    port.row_sg_jitter_sigma = lambda *_args: (
        0.0,
        {
            "sg_jitter_mode": "none",
            "sg_jitter_sigma_inv_A": 0.0,
            "sg_jitter_d_inv_A": float("nan"),
            "sg_jitter_frame_fraction": float("nan"),
        },
    )
    port.sg_derivative_per_deg = lambda *_args: 0.02
    port.solve_sg_profile_target = lambda *_args: pytest.fail(
        "direct-angle integration must not request an Sg root"
    )

    omegas, weights, residuals, meta, mosaicity_meta = port.row_profile_omega_points(
        0,
        np.asarray([1, 0, 0]),
        12.0,
        0.001,
        0.004,
    )

    assert len(omegas) == 15
    assert np.isclose(np.sum(weights), 5.0)
    assert residuals.size == 0
    assert meta["sg_profile_coordinate"] == "direct_angle"
    assert meta["sg_profile_angle_half_width_deg"] == pytest.approx(0.1)
    assert mosaicity_meta["mosaicity_kernel_points"] == 3


def test_geometry_path_contract_rejects_zero_axes_and_bound_violations(tmp_path):
    payload = {
        "schema_version": 1,
        "kind": "grond_geometry_path",
        "basis": "chebyshev",
        "frame_center": 10.0,
        "frame_half_range": 5.0,
        "coefficients": {
            "slew_frames": [0.1, 0.2],
            "tilt_x_deg": [0.0, 0.01],
            "tilt_y_deg": [0.0, -0.01],
        },
        "tilt_axes_lab": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "bounds": {"max_abs_slew_frames": 1.0, "max_abs_tilt_deg": 0.1},
        "source_hkl_sha256": "test-hash",
    }
    path = tmp_path / "path.json"
    path.write_text(json.dumps(payload))
    contract = load_geometry_path_contract(path)
    assert contract["source_hkl_sha256"] == "test-hash"
    assert contract["bounds"]["max_abs_slew_frames"] == pytest.approx(1.0)

    payload["tilt_axes_lab"][0] = [0.0, 0.0, 0.0]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="nonzero"):
        load_geometry_path_contract(path)

    payload["tilt_axes_lab"][0] = [1.0, 0.0, 0.0]
    payload["coefficients"]["slew_frames"] = [2.0]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="slew exceeds"):
        load_geometry_path_contract(path)


def test_expm_multiply_backend_matches_eigensolver_on_hermitian_matrix():
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    H = (raw + raw.conj().T) * 0.5
    H[np.diag_indices_from(H)] = np.real(np.diag(H))

    port = object.__new__(XDSBlochPort)
    port.cfg = Config(compute_backend="scipy")
    port.thick = np.asarray([300.0, 600.0, 900.0], dtype=float)
    port.thick_w = np.asarray([0.2, 0.5, 0.3], dtype=float)

    reference_single = port.propagate_direct_to_target(H, 700.0)
    reference_average = port.propagate_direct_to_target_thickness_average(H)

    port.cfg = Config(compute_backend="scipy_expm_multiply")
    trial_single = port.propagate_direct_to_target(H, 700.0)
    trial_average = port.propagate_direct_to_target_thickness_average(H)

    assert np.isclose(trial_single, reference_single, rtol=1e-9, atol=1e-11)
    assert np.isclose(trial_average, reference_average, rtol=1e-9, atol=1e-11)


def test_paired_inverse_hand_matches_explicit_conjugate_hamiltonian():
    """The elastic negative-thickness shortcut must equal coordinate inversion."""

    rng = np.random.default_rng(20260815)
    raw = 0.001 * (rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9)))
    H = (raw + raw.conj().T) * 0.5
    H[np.diag_indices_from(H)] = np.real(np.diag(H))

    port = object.__new__(XDSBlochPort)
    port.cfg = Config(compute_backend="scipy", paired_inverse_hand=True)
    port.thick = np.asarray([250.0, 600.0, 950.0], dtype=float)
    port.thick_w = np.asarray([0.2, 0.5, 0.3], dtype=float)
    port.absorption_enabled = lambda: False

    selected, inverse_from_negative_thickness = (
        port.propagate_direct_to_target_thickness_average_pair(H)
    )

    port.cfg = Config(compute_backend="scipy")
    explicit_selected = port.propagate_direct_to_target_thickness_average(H)
    explicit_inverse = port.propagate_direct_to_target_thickness_average(H.conj())

    assert selected == pytest.approx(explicit_selected, rel=1e-13, abs=1e-15)
    assert inverse_from_negative_thickness == pytest.approx(
        explicit_inverse, rel=1e-12, abs=1e-14
    )


def test_paired_inverse_hand_rejects_absorption_and_unsupported_backend():
    with pytest.raises(ValueError, match="requires absorption_mode=none"):
        Config(
            paired_inverse_hand=True,
            absorption_mode="midgley_like",
            absorption_mu=5e-5,
        )
    with pytest.raises(ValueError, match="requires the full SciPy Hermitian"):
        Config(paired_inverse_hand=True, compute_backend="scipy_expm")


def test_paired_inverse_hand_receives_same_row_resolution_envelope():
    data = SimpleNamespace(
        hkl=np.asarray([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]),
        zcal=np.asarray([0.0, 1.0, 2.0, 3.0]),
        omega_from_z=lambda z: float(z),
        corr=np.full(4, 100.0),
        peak=np.full(4, 100.0),
        xcal=np.arange(4, dtype=float),
        ycal=np.arange(4, dtype=float),
        header={},
        cell=gemmi.UnitCell(10.0, 11.0, 12.0, 90.0, 90.0, 90.0),
    )
    target = np.asarray([10.0, 20.0, 30.0, 40.0])
    sigma = np.ones(4)
    forward = np.asarray([1.0, 2.0, 3.0, 4.0])
    inverse = np.asarray([2.0, 4.0, 6.0, 8.0])
    pred = {
        "indices": np.arange(4),
        "dyn": forward,
        "dyn_inverse": inverse,
        "born": forward.copy(),
    }
    masks = {
        "all": np.ones(4, dtype=bool),
        "work": np.ones(4, dtype=bool),
        "free": np.zeros(4, dtype=bool),
    }
    nuisance = Nuisance(
        thickness_A=1000.0,
        thickness_grad_frac=0.0,
        frame_offset=0.0,
        group_width_deg=10.0,
        detector_scale_bins=1,
        sigma_a=0.0,
        sigma_b_corr=0.0,
        sigma_c_peak=0.0,
        row_resolution_b=2.0,
    )

    result = evaluate_trial(
        data,
        target,
        sigma,
        pred,
        np.arange(4),
        masks,
        nuisance,
        0.0,
        0.0,
    )
    factor = result["arrays"]["row_resolution_factor"]

    np.testing.assert_allclose(result["arrays"]["dyn_raw"], forward * factor)
    np.testing.assert_allclose(result["arrays"]["dyn_inverse_raw"], inverse * factor)
    np.testing.assert_allclose(
        result["arrays"]["dyn_inverse_raw_unmodulated"], inverse
    )


def test_real_embedding_generator_matches_complex_expm_column():
    rng = np.random.default_rng(321)
    raw = 0.001 * (rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    H = (raw + raw.conj().T) * 0.5
    H[np.diag_indices_from(H)] = np.real(np.diag(H))
    thickness = 350.0

    embedded = _real_embedding_generator_batch(H[None, :, :], thickness, "float64")[0]
    real_col = expm(embedded)[:, 0]
    trial = real_col[1] + 1j * real_col[H.shape[0] + 1]

    complex_col = expm((2j * np.pi * thickness) * H)[:, 0]
    assert np.isclose(abs(trial) ** 2, abs(complex_col[1]) ** 2, rtol=1e-11, atol=1e-13)


def test_bethe_backend_is_exact_when_strong_block_contains_all_beams():
    rng = np.random.default_rng(1234)
    raw = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    H = 0.001 * (raw + raw.conj().T) * 0.5
    H[np.diag_indices_from(H)] = np.real(np.diag(H))

    port = object.__new__(XDSBlochPort)
    port.thick = np.asarray([300.0, 600.0, 900.0], dtype=float)
    port.thick_w = np.asarray([0.2, 0.5, 0.3], dtype=float)
    port.cfg = Config(compute_backend="scipy")
    reference = port.propagate_direct_to_target_thickness_average(H)

    port.cfg = Config(compute_backend="scipy_bethe", bethe_strong_count=H.shape[0])
    trial = port.propagate_direct_to_target_thickness_average(H)

    assert np.isclose(trial, reference, rtol=1e-12, atol=1e-14)


def test_bethe_variants_return_finite_effective_hamiltonians():
    rng = np.random.default_rng(4321)
    raw = rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))
    H = 0.001 * (raw + raw.conj().T) * 0.5
    H[np.diag_indices_from(H)] = np.real(np.diag(H))

    for variant in ["truncate", "schur0", "schur_target", "schur_mean"]:
        cfg = Config(bethe_strong_count=4, bethe_variant=variant, bethe_promote_denom=0.0)
        effective = promoted_bethe_effective_hamiltonian(H, cfg)
        assert effective.shape == (4, 4)
        assert np.all(np.isfinite(effective))


def test_padded_jax_backend_defaults_to_eight_beam_buckets():
    assert _jax_matrix_bucket_size(Config(compute_backend="jax_cpu"), "jax_cpu") == 0
    assert _jax_matrix_bucket_size(Config(compute_backend="jax_cpu_padded"), "jax_cpu_padded") == 8
    assert _jax_matrix_bucket_size(Config(compute_backend="jax_mps_real_expm_padded"), "jax_mps_real_expm_padded") == 8
    assert (
        _jax_matrix_bucket_size(
            Config(compute_backend="jax_cpu_padded", jax_matrix_bucket_size=16),
            "jax_cpu_padded",
        )
        == 16
    )


def test_stack_square_matrices_pads_with_uncoupled_zero_beams():
    a = np.eye(2, dtype=np.complex128)
    b = np.ones((3, 3), dtype=np.complex128)
    stack = _stack_square_matrices([a, b], pad_to=4)
    assert stack.shape == (2, 4, 4)
    assert np.allclose(stack[0, :2, :2], a)
    assert np.allclose(stack[0, 2:, :], 0.0)
    assert np.allclose(stack[0, :, 2:], 0.0)
    assert np.allclose(stack[1, :3, :3], b)
