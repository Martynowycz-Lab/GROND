"""Repeated Miller indices share work, not physical symmetry or cached model state."""
from types import SimpleNamespace

import gemmi
import numpy as np
import pytest

from grond.bloch import potential
from grond.bloch.model import fcalc_res, parse_model


@pytest.fixture(params=['complex_p1', 'centro_special', 'monoclinic'])
def model(request, tmp_path):
    kind = request.param
    latt = -1 if kind == 'complex_p1' else 1
    symmetry = ['SYMM -X,Y+1/2,-Z'] if kind == 'monoclinic' else []
    cell = '10 11 12 90 100 90' if kind == 'monoclinic' else '10 11 12 90 90 90'
    first = 'C1 1 .13 .27 .31 10.63 .032' if kind == 'complex_p1' else 'C1 1 0 0 0 10.25 .032'
    second_sof = '10.81' if kind == 'complex_p1' else '21'
    path = tmp_path / 'model.res'
    path.write_text('\n'.join([
        'TITL sparse potential physical lookup test', f'CELL .025079 {cell}', f'LATT {latt}',
        *symmetry, 'SFAC C 1 0 0 0 0 0 0 0 0', 'FVAR 1 .6', first,
        f'C2 1 .33 .14 .41 {second_sof} .025 .036 .044 .006 -.003 .004', 'HKLF 4', '',
    ]))
    value = parse_model(path)
    value['test_kind'] = kind
    return value


def port_for(model):
    port = potential.PotentialMixin()
    port.model = model
    port.gamma = 1.39
    cell = gemmi.UnitCell(*model['cell'])
    port.data = SimpleNamespace(cell=cell, A=np.asarray(cell.frac.mat).T, zcal=np.array([5., 15.]))
    port.ug_cube = None
    port.cfg = SimpleNamespace(dose_b_mode='none', dose_b_per_frame=0.)
    port.dose_ref_frame = 10.
    return port


def scalar_reference(port, hkls):
    hkls = np.asarray(hkls, dtype=int)
    values = [port.gamma / (np.pi * port.data.cell.volume) * fcalc_res(port.model, row)
              for row in hkls.reshape(-1, 3)]
    return np.asarray(values, dtype=complex).reshape(hkls.shape[:-1])


@pytest.mark.parametrize('hkls', [
    np.array([2, -1, 3]),
    np.array([[2, -1, 3], [0, 0, 0], [-2, 1, -3]]),
    np.array([[[2, -1, 3], [0, 0, 0]], [[2, -1, 3], [-2, 1, -3]]]),
    np.empty((0, 3), dtype=int),
    np.empty((2, 0, 3), dtype=int),
])
def test_values_shape_order_and_unique_work(model, hkls, monkeypatch):
    port = port_for(model)
    calls = []
    original = potential.fcalc_res_grid

    def counted(current_model, h, k, miller_l):
        calls.append(np.column_stack((h, k, miller_l)))
        return original(current_model, h, k, miller_l)

    monkeypatch.setattr(potential, 'fcalc_res_grid', counted)
    actual = port.calculate_ug_array(hkls)
    assert actual.shape == hkls.shape[:-1]
    assert actual.dtype == np.complex128
    np.testing.assert_allclose(actual, scalar_reference(port, hkls), rtol=2e-13, atol=2e-16)
    if hkls.size:
        assert len(calls) == 1
        assert len({tuple(hkl) for hkl in calls[0]}) == len(calls[0])
        assert set(map(tuple, calls[0])) == set(map(tuple, hkls.reshape(-1, 3)))
    else:
        assert calls == []


def test_friedel_pair_and_special_site_occupancy_are_preserved(model):
    port = port_for(model)
    values = port.calculate_ug_array([[2, -1, 3], [-2, 1, -3], [0, 0, 0], [2, -1, 3]])
    assert values[0] == values[3]
    np.testing.assert_allclose(values[1], values[0].conjugate(), rtol=2e-13, atol=2e-16)
    if model['test_kind'] == 'complex_p1':
        assert abs(values[0].imag) > 1e-6  # Opposite indices must not be merged as one symmetry orbit.
        expected_zero = 1.44
    elif model['test_kind'] == 'centro_special':
        assert model['atoms'][0]['site_symmetry_order'] == 2
        assert model['atoms'][0]['occ'] == pytest.approx(.5)
        expected_zero = .5 + 2 * .6
    else:
        expected_zero = None
    if expected_zero is not None:
        assert values[2] == pytest.approx(port.gamma / (np.pi * port.data.cell.volume) * expected_zero)


def test_changed_coordinates_occupancy_adp_and_gamma_are_not_reused(model):
    port = port_for(model)
    hkls = np.array([[2, -1, 3], [0, 0, 0], [2, -1, 3], [-1, 2, 4]])
    before = port.calculate_ug_array(hkls)
    atom = model['atoms'][1]
    atom['x'] += .027
    atom['occ'] *= .7
    atom['u11'] *= 1.2
    port.gamma *= 1.1
    after = port.calculate_ug_array(hkls)
    np.testing.assert_allclose(after, scalar_reference(port, hkls), rtol=2e-13, atol=2e-16)
    assert not np.allclose(after, before, rtol=1e-3, atol=0.)


def test_row_dose_is_applied_after_the_base_potential(model):
    port = port_for(model)
    port.cfg = SimpleNamespace(dose_b_mode='linear_frame', dose_b_per_frame=.2)
    hkls = np.array([[[2, -1, 3], [0, 0, 0]], [[2, -1, 3], [-1, 2, 4]]])
    baseline = scalar_reference(port, hkls)
    inv_d2 = np.array([port.data.cell.calculate_1_d2(row.tolist()) for row in hkls.reshape(-1, 3)]).reshape(2, 2)
    np.testing.assert_allclose(port.ug_for_row_array(hkls, 0), baseline, rtol=2e-13, atol=2e-16)
    np.testing.assert_allclose(port.ug_for_row_array(hkls, 1), baseline * np.exp(-.25 * inv_d2),
                               rtol=2e-13, atol=2e-16)


def test_dense_cube_boundary_keeps_inside_and_repeated_outside_values(model):
    port = port_for(model)
    port.h_lim = port.k_lim = port.l_lim = 1
    hh, kk, ll = np.meshgrid(*([np.arange(-1, 2)] * 3), indexing='ij')
    port.ug_cube = scalar_reference(port, np.stack((hh, kk, ll), axis=-1))
    hkls = np.array([[0, 0, 0], [2, -1, 3], [-1, 0, 1], [2, -1, 3], [-2, 1, -3]])
    np.testing.assert_allclose(port.ug_array(hkls), scalar_reference(port, hkls), rtol=2e-13, atol=2e-16)
