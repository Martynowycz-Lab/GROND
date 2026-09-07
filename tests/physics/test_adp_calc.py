import math
from types import SimpleNamespace

import gemmi
import numpy as np

from grond.legacy.engine.bloch_engine import (
    Config,
    XDSBlochPort,
    XDSData,
    atom_dw,
    atom_dw_array,
    cell_orthogonalization_matrix,
    parse_model,
    shelx_u_cart_matrix,
)

CELL = (10.0, 15.0, 20.0, 90.0, 110.0, 90.0)
ANISOTROPIC_ATOM = {
    "adp_type": "anis",
    "u11": 0.05,
    "u22": 0.04,
    "u33": 0.03,
    "u12": 0.01,
    "u13": 0.02,
    "u23": 0.015,
}


def crystallographic_dw(atom, hkl, cell_tuple):
    """Independent SHELX/coreCIF anisotropic displacement-factor formula."""

    reciprocal = gemmi.UnitCell(*cell_tuple).reciprocal()
    scaled = np.asarray(
        [hkl[0] * reciprocal.a, hkl[1] * reciprocal.b, hkl[2] * reciprocal.c],
        dtype=float,
    )
    u = np.asarray(
        [
            [atom["u11"], atom["u12"], atom["u13"]],
            [atom["u12"], atom["u22"], atom["u23"]],
            [atom["u13"], atom["u23"], atom["u33"]],
        ],
        dtype=float,
    )
    return math.exp(-2.0 * math.pi**2 * float(scaled @ u @ scaled))


def test_anisotropic_dw_uses_crystallographic_axes_in_nonorthogonal_cell():
    hkl = (5, 2, 7)
    expected = crystallographic_dw(ANISOTROPIC_ATOM, hkl, CELL)

    assert math.isclose(expected, 0.5545986340935334, rel_tol=1e-12)
    assert math.isclose(atom_dw(ANISOTROPIC_ATOM, hkl, CELL), expected, rel_tol=1e-12)


def test_crystallographic_to_cartesian_u_preserves_the_exponent():
    hkl = np.asarray([5.0, 2.0, 7.0])
    orth = cell_orthogonalization_matrix(CELL)
    reciprocal_cart = np.linalg.inv(orth).T @ hkl
    u_cart = shelx_u_cart_matrix(ANISOTROPIC_ATOM, CELL)
    exponent_cart = float(reciprocal_cart @ u_cart @ reciprocal_cart)

    reciprocal = gemmi.UnitCell(*CELL).reciprocal()
    scaled = hkl * np.asarray([reciprocal.a, reciprocal.b, reciprocal.c])
    u_raw = np.asarray(
        [
            [0.05, 0.01, 0.02],
            [0.01, 0.04, 0.015],
            [0.02, 0.015, 0.03],
        ]
    )
    exponent_crystallographic = float(scaled @ u_raw @ scaled)

    assert np.allclose(u_cart, u_cart.T, rtol=0.0, atol=1e-15)
    assert math.isclose(exponent_cart, exponent_crystallographic, rel_tol=1e-12)


def test_anisotropic_dw_array_accepts_meshgrid_hkls():
    h, k, ell = np.meshgrid(
        np.asarray([-1.0, 0.0, 1.0]),
        np.asarray([0.0, 2.0]),
        np.asarray([-2.0, 3.0]),
        indexing="ij",
    )
    got = atom_dw_array(ANISOTROPIC_ATOM, h, k, ell, np.zeros_like(h), CELL)

    assert got.shape == h.shape
    for index in np.ndindex(h.shape):
        expected = crystallographic_dw(
            ANISOTROPIC_ATOM,
            (h[index], k[index], ell[index]),
            CELL,
        )
        assert math.isclose(float(got[index]), expected, rel_tol=1e-12)


def test_support_scan_is_multiprocessing_safe(tmp_path):

    hkl_file = tmp_path / "INTEGRATE.HKL"
    res_file = tmp_path / "model.res"
    hkl_file.write_text(
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
        " 1 0 0 100.0 10.0 0.0 0.0 0.0 0.05 100.0 95.0"
        " 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
        " 1 0 0 200.0 20.0 0.0 0.0 0.0 0.05 100.0 95.0"
        " 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
    )
    res_file.write_text(
        "TITL Mock\n"
        "CELL 1.0 10.0 10.0 10.0 90.0 90.0 90.0\n"
        "LATT -1\n"
        "SFAC C 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n"
        "UNIT 1\n"
        "C1 1 0.0 0.0 0.0 11.0 0.05\n"
        "END\n"
    )

    from grond.legacy.engine.support_basis_scan import scan_basis_sizes

    data = XDSData(hkl_file)
    cfg = Config(n_basis=10, n_points=1, topup_d_min=1.0, max_support_sg=0.005)
    port = XDSBlochPort(data, parse_model(res_file), cfg)

    sizes = scan_basis_sizes(port, np.asarray([0, 1]), workers=2, chunksize=1)
    assert len(sizes) == 2


def test_support_scan_sizes_basis_at_recorded_frame_geometry():
    from grond.legacy.engine.support_basis_scan import scan_basis_sizes

    selected_at = []
    port = SimpleNamespace(
        data=SimpleNamespace(
            hkl=np.asarray([[1, 0, 0]], dtype=int),
            zcal=np.asarray([3.0]),
        ),
        omega_from_z=lambda z: float(z) + 0.25,
        select_basis=lambda _hkl, omega, row_idx=None: (
            selected_at.append((float(omega), int(row_idx)))
            or np.asarray([[0, 0, 0], [1, 0, 0]], dtype=int)
        ),
    )

    sizes = scan_basis_sizes(port, np.asarray([0]), workers=1, chunksize=1)

    assert sizes.tolist() == [2]
    assert selected_at == [(3.25, 0)]
