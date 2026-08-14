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
    parse_model,
    promoted_bethe_effective_hamiltonian,
    relativistic_gamma_from_wavelength,
    rotation_matrix,
    row_plan_cache_validation_errors,
    split_masks,
)
from grond.legacy.engine.build_dynrows_xds import gvec_for_hkl, sg_for_hkl

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
