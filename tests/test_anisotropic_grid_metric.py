"""Vector structure factors retain rotated ADPs without unused anisotropic d-spacing work."""
import gemmi
import numpy as np
import pytest

from grond.bloch import model as physics


@pytest.fixture(params=['centro_special', 'screw_mixed'])
def crystal(request, tmp_path):
    if request.param == 'centro_special':
        symmetry = ['LATT 1']
        first = 'C1 1 0 0 0 10.25 .025 .036 .044 .006 -.003 .004'
        second = 'C2 1 .13 .27 .41 21 .031 .023 .049 -.005 .004 .003'
    else:
        symmetry = ['LATT -1', 'SYMM -X,Y+1/2,-Z']
        first = 'C1 1 .17 .22 .31 10.8 .025 .036 .044 .006 -.003 .004'
        second = 'C2 1 .31 .18 .39 21 .037'
    path = tmp_path / 'metric.res'
    path.write_text('\n'.join([
        'TITL anisotropic grid metric test', 'CELL .025079 10 11 12 90 100 90', *symmetry,
        'SFAC C 1 .7 0 0 0 0 0 0 .2', 'FVAR 1 .6', first, second, 'HKLF 4', '',
    ]))
    return physics.parse_model(path)


def test_scalar_grid_parity_and_only_isotropic_mate_metrics(crystal, monkeypatch):
    hkls = np.array([[[2, -1, 3], [0, 0, 0]], [[-2, 1, -3], [1, 4, -2]]])
    expected = np.array([physics.fcalc_res(crystal, row) for row in hkls.reshape(-1, 3)]).reshape(2, 2)
    isotropic_mates = sum(len(list(physics.unique_symmetry_mates(atom, crystal)))
                        for atom in crystal['atoms'] if atom['adp_type'] != 'anis')
    original = physics.inv_d2_from_hkl_array
    calls = []

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(physics, 'inv_d2_from_hkl_array', counted)
    actual = physics.fcalc_res_grid(crystal, hkls[..., 0], hkls[..., 1], hkls[..., 2])
    # One metric evaluation for SFAC; only isotropic mates require additional s^2 values.
    assert len(calls) == 1 + isotropic_mates
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
    if crystal['atoms'][0]['site_symmetry_order'] == 2:
        assert crystal['atoms'][0]['occ'] == pytest.approx(.5)
        assert actual[0, 1] == pytest.approx(1.2 * (.5 + 2 * .6))


def test_anisotropic_dw_uses_rotated_indices_and_accepts_no_s2(crystal):
    atom = crystal['atoms'][0]
    hkl = np.array([2., -1., 3.])
    reciprocal = gemmi.UnitCell(*crystal['cell']).reciprocal()
    lengths = np.array([reciprocal.a, reciprocal.b, reciprocal.c])
    u = np.array([[atom['u11'], atom['u12'], atom['u13']],
                  [atom['u12'], atom['u22'], atom['u23']],
                  [atom['u13'], atom['u23'], atom['u33']]])
    for _, rotation in physics.unique_symmetry_mates(atom, crystal):
        rotated = np.asarray(rotation).T @ hkl
        q = lengths * rotated
        expected = np.exp(-2 * np.pi**2 * (q @ u @ q))
        got = physics.atom_dw_array(atom, *rotated, None, crystal['cell'])
        assert got == pytest.approx(expected, rel=2e-13, abs=1e-15)
