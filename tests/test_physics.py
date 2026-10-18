"""Independent physical identities and parser regression tests."""
import copy
import math
from types import SimpleNamespace

import gemmi
import numpy as np
import pytest
from scipy.linalg import expm

from grond.engine.bloch_engine import (
    Config,
    XDSBlochPort,
    alternate_hand_model,
    atom_dw,
    fcalc_res,
    fcalc_res_grid,
    parse_model,
    relativistic_gamma_from_wavelength,
    shelx_u_cart_matrix,
    unique_symmetry_mates,
)


def res_file(tmp_path, atoms, *, latt=1, symmetry=(), cell='10 11 12 90 90 90', fvar='1 0.6'):
    path = tmp_path / 'model.res'
    path.write_text('\n'.join([
        'TITL independent physics check', f'CELL 0.025079 {cell}', f'LATT {latt}',
        *[f'SYMM {sym}' for sym in symmetry], 'SFAC C 1 0 0 0 0 0 0 0 0',
        f'FVAR {fvar}', *atoms, 'HKLF 4', '',
    ]))
    return path


@pytest.mark.parametrize('encoded,expected', [(10.5, 1), (20.5, .6), (-20.5, .4), (.25, .5)])
def test_res_special_site_occupancy_uses_physical_convention(tmp_path, encoded, expected):
    model = parse_model(res_file(tmp_path, [f'C1 1 0 0 0 {encoded} 0']))
    atom = model['atoms'][0]
    assert atom['shelx_sof_encoded'] == encoded
    assert atom['shelx_sof'] == pytest.approx(expected / 2)
    assert atom['site_symmetry_order'] == 2
    assert atom['occ'] == pytest.approx(expected)
    # A single physical atom with f=1 at the origin has F=occupancy for all HKLs.
    assert fcalc_res(model, (1, 2, 3)) == pytest.approx(expected)


def test_res_mixed_general_special_sites_match_explicit_unit_cell(tmp_path):
    model = parse_model(res_file(tmp_path, [
        'C1 1 0 0 0 10.5 0', 'C2 1 .13 .27 .31 21 0',
    ], cell='10 11 12 82 97 106'))
    hkls = np.array([[1, 0, 0], [2, -1, 3], [-3, 2, 1], [0, 0, 0]])
    expected = 1 + 1.2 * np.cos(2 * np.pi * (hkls @ np.array([.13, .27, .31])))
    actual = fcalc_res_grid(model, *hkls.T)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert model['atoms'][1]['site_symmetry_order'] == 1


@pytest.mark.parametrize('coordinate,expected_order', [('0.333333 0.666667 0.25', 3), ('0.3334 0.6667 0.25', 1)])
def test_rounded_third_special_site_is_distinct_from_nearby_general_site(tmp_path, coordinate, expected_order):
    model = parse_model(res_file(tmp_path, [f'C1 1 {coordinate} 10.3333333333333 0'],
        latt=-1, symmetry=('-y,x-y,z', '-x+y,-x,z'), cell='10 10 12 90 90 120'))
    assert model['atoms'][0]['site_symmetry_order'] == expected_order
    assert len(list(unique_symmetry_mates(model['atoms'][0], model))) == 3 // expected_order


def test_cif_physical_special_site_occupancy_is_not_multiplied(tmp_path):
    # Dictionary-level model is also the invariant used by CIF ingestion.
    model = parse_model(res_file(tmp_path, ['C1 1 0 0 0 10.5 0']))
    assert fcalc_res(model, (1, 1, 1)) == pytest.approx(1)
    model['model_source'] = 'cif'
    model['atoms'][0]['occ'] = .7
    assert fcalc_res(model, (1, 1, 1)) == pytest.approx(.7)


def electron_cif(tmp_path, *, adp_header='_atom_site_adp_type\n_atom_site_U_iso_or_equiv', adp_value='Uiso 0.04', anisotropic='', position='0 0 0', occupancy='1', symmetry='1'):
    path = tmp_path / 'model.cif'
    path.write_text(f'''data_electron
_grond_scattering_factor_kind electron
_cell_length_a 10
_cell_length_b 11
_cell_length_c 12
_cell_angle_alpha 82
_cell_angle_beta 97
_cell_angle_gamma 106
_space_group_IT_number {symmetry}
loop_
_atom_type_symbol
_atom_type_scat_Cromer_Mann_a1
_atom_type_scat_Cromer_Mann_b1
_atom_type_scat_Cromer_Mann_a2
_atom_type_scat_Cromer_Mann_b2
_atom_type_scat_Cromer_Mann_a3
_atom_type_scat_Cromer_Mann_b3
_atom_type_scat_Cromer_Mann_a4
_atom_type_scat_Cromer_Mann_b4
_atom_type_scat_Cromer_Mann_c
C 1 0 0 0 0 0 0 0 0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
{adp_header}
C1 C {position} {occupancy} {adp_value}
{anisotropic}
''')
    return path


def anisotropic_loop(kind='U', values=(.02, .03, .04, .005, -.002, .006)):
    tags = '\n'.join(f'_atom_site_aniso_{kind}_{ij}' for ij in ('11', '22', '33', '12', '13', '23'))
    return f"loop_\n_atom_site_aniso_label\n{tags}\nC1 " + ' '.join(map(str, values))


@pytest.mark.parametrize('tag', ['_atom_site_adp_type', '_atom_site_thermal_displace_type'])
def test_cif_biso_converts_with_both_type_tags(tmp_path, tag):
    model = parse_model(electron_cif(tmp_path, adp_header=tag+'\n_atom_site_B_iso_or_equiv', adp_value='Biso 1'))
    assert model['atoms'][0]['uiso'] == pytest.approx(1 / (8 * np.pi**2))
    hkl = [10, 0, 0]
    s2 = gemmi.UnitCell(*model['cell']).calculate_1_d2(hkl) / 4
    assert fcalc_res(model, hkl) == pytest.approx(np.exp(-s2))


@pytest.mark.parametrize('declared', [True, False])
def test_cif_bani_and_unlabelled_anisotropic_values_are_preserved(tmp_path, declared):
    u = np.array([.02, .03, .04, .005, -.002, .006])
    model = parse_model(electron_cif(tmp_path,
        adp_header='_atom_site_adp_type' if declared else '_atom_site_U_iso_or_equiv',
        adp_value='Bani' if declared else '.04',
        anisotropic=anisotropic_loop('B', u * 8 * np.pi**2)))
    atom = model['atoms'][0]
    np.testing.assert_allclose([atom['u'+ij] for ij in ('11','22','33','12','13','23')], u)
    assert atom['adp_type'] == 'anis'
    assert atom['adp_source'] == 'cif_bani'


def test_cif_uani_inferred_without_type_and_special_occupancy_unchanged(tmp_path):
    model = parse_model(electron_cif(tmp_path, adp_header='_atom_site_U_iso_or_equiv', adp_value='.04', anisotropic=anisotropic_loop(), symmetry='2', occupancy='.7'))
    assert model['atoms'][0]['adp_type'] == 'anis'
    assert model['atoms'][0]['occ'] == .7
    assert fcalc_res(model, (0, 0, 0)) == pytest.approx(.7)


@pytest.mark.parametrize('label,value', [('Uani','.04'), ('Biso','.04'), ('Umpe','.04')])
def test_cif_explicit_missing_or_unsupported_adp_does_not_default(tmp_path, label, value):
    with pytest.raises(ValueError, match='lacks|unsupported'):
        parse_model(electron_cif(tmp_path, adp_value=f'{label} {value}'))


def test_cif_incomplete_anisotropic_tensor_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='incomplete'):
        parse_model(electron_cif(tmp_path, adp_header='_atom_site_adp_type', adp_value='Uani',
            anisotropic='loop_\n_atom_site_aniso_label\n_atom_site_aniso_U_11\nC1 .04'))


@pytest.mark.parametrize('kind,value', [('U','malformed'), ('B','NaN'), ('U','?')])
def test_cif_unlabelled_invalid_displacement_never_silently_defaults(tmp_path, kind, value):
    with pytest.raises(ValueError, match='non-finite or malformed'):
        parse_model(electron_cif(tmp_path, adp_header=f'_atom_site_{kind}_iso_or_equiv', adp_value=value))


def test_cif_absent_displacement_records_explicit_default_provenance(tmp_path):
    model = parse_model(electron_cif(tmp_path, adp_header='', adp_value=''))
    assert model['atoms'][0]['uiso'] == .05
    assert model['atoms'][0]['adp_source'] == 'default_uiso'


def test_structure_factors_match_independent_gemmi_electron_calculator(tmp_path):
    model = parse_model(res_file(tmp_path, ['C1 1 0 0 0 10.5 .02', 'C2 1 .13 .27 .31 21 .04'],
        cell='10 11 12 82 97 106'))
    coef = gemmi.Element('C').c4322
    model['sfac'] = [{'element': 'C', 'pairs': list(zip(coef.a, coef.b, strict=True)), 'c': 0}]
    structure = gemmi.SmallStructure()
    structure.cell = gemmi.UnitCell(*model['cell'])
    structure.spacegroup_hm = 'P -1'
    determine = getattr(structure, 'determine_and_set_spacegroup', None) or structure.set_spacegroup
    determine('2')
    assert structure.spacegroup.number == 2
    for label, xyz, occ, u in [('C1', (0, 0, 0), 1, .02), ('C2', (.13, .27, .31), .6, .04)]:
        site = gemmi.SmallStructure.Site()
        site.label = label
        site.element = gemmi.Element('C')
        site.fract = gemmi.Fractional(*xyz)
        site.occ = occ
        site.u_iso = u
        structure.add_site(site)
    # Gemmi's calculator sums every operation, so its documented preparatory
    # conversion differs from GROND's unique-site internal representation.
    structure.change_occupancies_to_crystallographic(max_dist=1e-5)
    assert structure.sites[0].occ == .5
    assert structure.sites[1].occ == .6
    calculator = gemmi.StructureFactorCalculatorE(structure.cell)
    hkls = [[1, 0, 0], [2, -1, 3], [-3, 2, 1], [5, -6, 7], [0, 0, 0]]
    for hkl in hkls:
        assert fcalc_res(model, hkl) == pytest.approx(calculator.calculate_sf_from_small_structure(structure, hkl), rel=3e-7, abs=1e-7)


def minimal_port(**kwargs):
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(thickness_model='single', **kwargs)
    port.data = SimpleNamespace(k=40.)
    port.thick, port.thick_w = port.thickness_grid()
    return port


@pytest.mark.parametrize('coupling,delta,thickness', [(0, .002, 1000), (1e-3, 0, 321), (1e-3+.7e-3j, .003, 812), (1e-6, -.002, 111), (1e-3, .004, 0)])
def test_two_beam_exact_analytic_solution(coupling, delta, thickness):
    port = minimal_port()
    H = np.array([[0, np.conj(coupling)], [coupling, delta]], dtype=complex)
    frequency = math.sqrt(delta**2 + 4 * abs(coupling)**2)
    expected = 0 if not frequency else 4 * abs(coupling)**2 / frequency**2 * math.sin(np.pi * thickness * frequency)**2
    assert port.propagate_direct_to_target(H, thickness) == pytest.approx(expected, abs=3e-15)


def test_one_beam_phase_and_uniform_absorption_are_analytic():
    from grond.engine.bloch_engine import validate_passive_absorption
    port = minimal_port(absorption_mode='mean', absorption_mu=5e-5)
    port.ug_array = lambda hkls: np.ones(hkls.shape[:-1], dtype=complex) * 123
    H = port.hamiltonian(np.array([[0, 0, 0]]), np.array([.0123]))
    validate_passive_absorption(H, mode='mean')
    intensity = abs(expm(2j * np.pi * H * 1000)[0, 0])**2
    assert intensity == pytest.approx(np.exp(-4 * np.pi * 5e-5 * 1000), rel=1e-14)


def test_origin_shift_is_unitary_gauge_and_does_not_change_intensities(tmp_path):
    model = parse_model(res_file(tmp_path, ['C1 1 .13 .27 .31 11 .02', 'C2 1 .23 .17 .49 11 .04'], latt=-1))
    shifted = copy.deepcopy(model)
    origin = np.array([.173, -.229, .413])
    for atom in shifted['atoms']:
        for k, v in zip(('x', 'y', 'z'), origin, strict=True):
            atom[k] = (atom[k] + v) % 1
    basis = np.array([[0,0,0],[1,2,3],[2,-1,0],[-1,1,2],[3,2,-1]])
    sg = np.array([0,.001,-.002,.0003,-.0013])
    volume = gemmi.UnitCell(*model['cell']).volume
    port = minimal_port()
    def matrix(m):
        port.ug_array = lambda h: relativistic_gamma_from_wavelength(.025079) / (np.pi * volume) * fcalc_res_grid(m, *np.moveaxis(h, -1, 0))
        return port.hamiltonian(basis, sg)
    H = matrix(model)
    shifted_H = matrix(shifted)
    D = np.diag(np.exp(2j*np.pi*(basis @ origin)))
    np.testing.assert_allclose(shifted_H, D.conj().T @ H @ D, atol=1e-17)
    assert port.propagate_direct_to_target(H, 1357) == pytest.approx(port.propagate_direct_to_target(shifted_H, 1357), rel=2e-13)
    U = expm(2j*np.pi*H*1357)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(len(H)), atol=3e-15)


def test_physical_electron_units_give_expected_extinction_distance():
    wavelength = .025079
    gamma = relativistic_gamma_from_wavelength(wavelength)
    # Electron amplitude F is in angstroms; volume is A^3. No hidden 2*pi.
    F, volume = 12., 1000.
    ug = gamma * F / (np.pi * volume)
    coupling = ug / (2 / wavelength)
    extinction_distance = np.pi * volume / (gamma * wavelength * F)
    H = np.array([[0, coupling], [coupling, 0]])
    port = minimal_port()
    assert port.propagate_direct_to_target(H, extinction_distance / 2) == pytest.approx(1, abs=1e-14)
    assert port.propagate_direct_to_target(H, extinction_distance) == pytest.approx(0, abs=1e-25)


@pytest.mark.parametrize('model,grad,sigma,variance_factor', [('uniform_gradient',.4,0,.4**2/3), ('gaussian',0,.1,.1**2), ('gaussian_gradient',.3,.1,.3**2/3+.1**2)])
def test_thickness_quadrature_matches_distribution_moments(model, grad, sigma, variance_factor):
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(thickness_A=1000, thickness_model=model, thickness_grad_frac=grad, thickness_sigma_frac=sigma, thickness_points=9)
    nodes, weights = port.thickness_grid()
    assert np.sum(weights) == pytest.approx(1)
    assert np.sum(weights * nodes) == pytest.approx(1000)
    assert np.sum(weights * (nodes-1000)**2) == pytest.approx(1000**2 * variance_factor)
    assert np.all(nodes > 0)


def test_nonhermitian_passivity_check_rejects_gain_and_exception_is_picklable():
    import pickle

    from grond.engine.bloch_engine import NonPassiveAbsorptionError, validate_passive_absorption
    H = (1 + .02j) * np.array([[0, .001], [.001, 0]])
    # Independent demonstration: the proposed generator really creates intensity.
    assert np.sum(abs(expm(2j * np.pi * H * 1000) @ [1,0])**2) > 1.03
    with pytest.raises(NonPassiveAbsorptionError) as caught:
        validate_passive_absorption(H, mode='midgley_like', row_index=17)
    restored = pickle.loads(pickle.dumps(caught.value))
    assert restored.details['row_index'] == 17
    assert restored.details['minimum_loss_eigenvalue_inv_A'] == pytest.approx(-2e-5)
    passive = H + 3e-5j * np.eye(2)
    validate_passive_absorption(passive)
    for thickness in [1,100,1000,5000]:
        U = expm(2j*np.pi*passive*thickness)
        assert np.linalg.svd(U, compute_uv=False)[0] <= 1 + 1e-13
        port = minimal_port(absorption_mode='midgley_like', absorption_mu=3e-5, absorption_frac=.02)
        assert port.propagate_direct_to_target(passive, thickness) == pytest.approx(abs(U[1,0])**2, rel=1e-12, abs=1e-14)


def test_passivity_rejection_applies_to_uncached_and_cached_row_matrices():
    from grond.engine.bloch_engine import NonPassiveAbsorptionError
    port = minimal_port(absorption_mode='midgley_like', absorption_mu=0, absorption_frac=.02)
    port.ug_array = lambda hkls: np.ones(hkls.shape[:-1], dtype=complex) * .08
    basis = np.array([[0,0,0], [1,0,0]])
    with pytest.raises(NonPassiveAbsorptionError):
        port.hamiltonian_static_part(basis)
    diffs = (basis[None,:,:]-basis[:,None,:]).reshape(-1,3)
    with pytest.raises(NonPassiveAbsorptionError):
        port.cached_static_hamiltonian(basis, np.arange(4), diffs, np.zeros(4), np.full(4,.08+0j), 0, True, dose_delta_b=0)


def test_thickness_average_matches_analytic_uniform_pendellosung():
    port = object.__new__(XDSBlochPort)
    port.cfg = Config(thickness_A=1100, thickness_model='uniform_gradient', thickness_grad_frac=.4, thickness_points=12)
    port.thick, port.thick_w = port.thickness_grid()
    coupling = .0004
    H = np.array([[0, coupling], [coupling, 0]])
    t0, halfwidth = 1100., 440.
    a = 2 * np.pi * coupling
    expected = .5 - .5 * np.cos(2*a*t0) * np.sinc(2*a*halfwidth/np.pi)
    assert port.propagate_direct_to_target_thickness_average(H) == pytest.approx(expected, abs=2e-14)


def test_born_limit_at_finite_detuning_converges_quadratically():
    port = minimal_port()
    detuning, t = .002, 111.
    errors = []
    for coupling in [1e-5, 1e-6, 1e-7]:
        H = np.array([[0, coupling], [coupling, detuning]])
        born = (2*np.pi*coupling*t)**2 * np.sinc(detuning*t)**2
        dyn = port.propagate_direct_to_target(H,t)
        errors.append(abs(dyn-born)/born)
    assert 90 < errors[0] / errors[1] < 110
    assert 90 < errors[1] / errors[2] < 110
    assert errors[-1] < 2e-9


def test_proper_reciprocal_setting_change_preserves_skew_cell_factors(tmp_path):
    from grond.engine.bloch_engine import cell_from_real_matrix, cell_orthogonalization_matrix
    model = parse_model(res_file(tmp_path, ['C1 1 .13 .27 .31 11 .02', 'C2 1 .23 .17 .49 11 .04'],
        latt=-1, cell='10 11 12 82 97 106'))
    T = np.array([[0,1,0],[0,0,-1],[-1,0,0]])
    assert round(np.linalg.det(T)) == 1
    transformed = copy.deepcopy(model)
    real = cell_orthogonalization_matrix(model['cell'])
    cell = cell_from_real_matrix(real @ T.T)
    transformed['cell'] = (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
    for atom in transformed['atoms']:
        xyz = T @ np.array([atom[k] for k in ('x','y','z')])
        atom.update(zip(('x','y','z'), xyz % 1, strict=True))
    for hkl in np.array([[1,2,3],[-2,1,4],[5,-3,1]]):
        assert fcalc_res(transformed,T@hkl) == pytest.approx(fcalc_res(model,hkl), rel=1e-13, abs=1e-14)


def test_elastic_reciprocity_and_coordinate_inversion_use_same_hamiltonian(tmp_path):
    model = parse_model(res_file(tmp_path, ['C1 1 .13 .27 .31 11 .02', 'C2 1 .23 .17 .49 11 .04'], latt=-1))
    inverse = alternate_hand_model(model)
    basis = np.array([[0,0,0],[1,2,3],[2,-1,0],[-1,1,2]])
    diffs = basis[None,:,:]-basis[:,None,:]
    def H(m):
        matrix = fcalc_res_grid(m,*np.moveaxis(diffs,-1,0)) * .0004
        np.fill_diagonal(matrix,[0,.001,-.002,.0003])
        return matrix
    matrix = H(model)
    invmatrix = H(inverse)
    np.testing.assert_allclose(invmatrix,matrix.conj(),atol=2e-18)
    port = minimal_port(thickness_A=1321)
    original,inverted = port.propagate_direct_to_target_thickness_average_pair(matrix)
    U = expm(2j*np.pi*matrix*1321)
    assert original == pytest.approx(abs(U[1,0])**2,abs=2e-14)
    assert inverted == pytest.approx(abs(U[0,1])**2,abs=2e-14)
    assert inverted == pytest.approx(port.propagate_direct_to_target(invmatrix,1321),abs=2e-14)
    assert abs(original-inverted) > 1e-3


def test_anisotropic_dw_matches_cartesian_displacement_covariance(tmp_path):
    from grond.engine.bloch_engine import cell_orthogonalization_matrix
    model = parse_model(res_file(tmp_path, ['C1 1 .13 .27 .31 11 .02 .03 .04 .006 -.002 .005'],
        latt=-1,cell='10 11 12 82 97 106'))
    atom=model['atoms'][0]
    orth=cell_orthogonalization_matrix(model['cell'])
    rec=np.linalg.inv(orth).T
    D=np.diag([np.linalg.norm(rec[:,i]) for i in range(3)])
    u=np.array([[.02,.005,-.002],[.005,.03,.006],[-.002,.006,.04]])
    # Direct independent Cartesian covariance from normalized reciprocal axes.
    reciprocal_unit_axes=rec @ np.linalg.inv(D)
    u_cart=np.linalg.inv(reciprocal_unit_axes.T) @ u @ np.linalg.inv(reciprocal_unit_axes)
    np.testing.assert_allclose(shelx_u_cart_matrix(atom,model['cell']),u_cart,atol=3e-17)
    for hkl in np.array([[1,2,3],[-2,1,4],[5,-3,1]]):
        g=rec @ hkl
        expected=np.exp(-2*np.pi**2*(g@u_cart@g))
        assert atom_dw(atom,hkl,model['cell']) == pytest.approx(expected,rel=1e-14)
