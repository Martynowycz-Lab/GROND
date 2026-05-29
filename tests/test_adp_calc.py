import math

import gemmi
import numpy as np

from grond.legacy.engine.bloch_engine import atom_dw, atom_dw_array


def test_adp_math():
    # Create a non-orthogonal cell (monoclinic, beta = 110 degrees)
    a, b, c = 10.0, 15.0, 20.0
    alpha, beta, gamma = 90.0, 110.0, 90.0
    cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)
    
    # Reciprocal cell parameters
    rec = cell.reciprocal()
    ar = rec.a
    br = rec.b
    cr = rec.c
    betar = math.radians(rec.beta)
    
    # Let's check the reciprocal angles/cosines
    print("Reciprocal cell a*, b*, c*:", ar, br, cr)
    print("Reciprocal angles (deg) alpha*, beta*, gamma*:", rec.alpha, rec.beta, rec.gamma)
    
    # Create an atom and set anisotropic U
    atom = gemmi.Atom()
    atom.name = "C"
    atom.element = gemmi.Element("C")
    atom.pos = gemmi.Position(1.0, 2.0, 3.0)
    atom.occ = 1.0
    atom.b_iso = 0.0
    u11, u22, u33 = 0.05, 0.04, 0.03
    u12, u13, u23 = 0.01, 0.02, 0.015
    atom.aniso.u11 = u11
    atom.aniso.u22 = u22
    atom.aniso.u33 = u33
    atom.aniso.u12 = u12
    atom.aniso.u13 = u13
    atom.aniso.u23 = u23
    
    # Add to structure
    struct = gemmi.Structure()
    struct.cell = cell
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    res = gemmi.Residue()
    res.name = "ALA"
    res.seqid = gemmi.SeqId("1")
    res.add_atom(atom)
    chain.add_residue(res)
    model.add_chain(chain)
    struct.add_model(model)
    
    gemmi_beq = atom.b_eq()
    print("Gemmi calculated b_eq (in structure):", gemmi_beq)
    
    # Let's calculate Beq using the standard formula:
    # Beq = (8*pi^2/3) * sum_{i,j} U_ij a*_i a*_j a_i . a_j
    # where a_i . a_j are the dot products of real space cell vectors.
    # For monoclinic beta:
    # a . a = a^2
    # b . b = b^2
    # c . c = c^2
    # a . b = 0, b . c = 0
    # a . c = a * c * cos(beta)
    # So:
    # Beq = (8*pi^2 / 3) * [ U11 a*^2 a^2 + U22 b*^2 b^2 + U33 c*^2 c^2 + 2 * U13 * a* * c* * a * c * cos(beta) ]
    # Let's test different candidates for Beq:
    # Candidate A: standard crystallographic formula:
    # Beq = 8/3 * pi^2 * [ U11 a*^2 a^2 + U22 b*^2 b^2 + U33 c*^2 c^2 + 2*U12 a* b* a b cos(gamma) + 2*U13 a* c* a c cos(beta) + 2*U23 b* c* b c cos(alpha) ]
    # Note that for our cell:
    # alpha = 90, gamma = 90, beta = 110. cos(alpha) = 0, cos(gamma) = 0, cos(beta) = cos(110 deg)
    # Also, ar = a* (reciprocal length), a = real length.
    # Wait, does the formula use the reciprocal angles instead? i.e. cos(beta*)?
    # Let's try both!
    
    # Let's list some candidate terms for the cross terms:
    # 1. 2 * U13 * ar * cr * a * c * cos(beta)
    # 2. 2 * U13 * ar * cr * a * c * cos(beta*)
    # 3. 2 * U13 * ar * cr * a * c  (no cosine at all)
    # 4. 2 * U13 * cos(beta)
    # 5. 2 * U13 * cos(beta*)
    # 6. What if U_ij already has some factors?
    
    # Wait, let's compute:
    diag_sum = u11 * ar * ar * a * a + u22 * br * br * b * b + u33 * cr * cr * c * c
    
    # Try with cos(beta) in real space:
    beq_real_cos = (8.0 * math.pi * math.pi / 3.0) * (diag_sum + 2.0 * u13 * ar * cr * a * c * math.cos(math.radians(beta)))
    print("Candidate 1 (Real cos beta):", beq_real_cos)
    
    # Try with cos(beta*) in reciprocal space:
    beq_recip_cos = (8.0 * math.pi * math.pi / 3.0) * (diag_sum + 2.0 * u13 * ar * cr * a * c * math.cos(betar))
    print("Candidate 2 (Recip cos beta*):", beq_recip_cos)
    
    # Try with cos(beta) and cos(beta*)?
    # Let's print other variants:
    print("Gemmi:", gemmi_beq)
    
    # Let's also check if we can compute trace of U in cartesian coordinates.
    # How does Gemmi define U? Gemmi has aniso as an anisotropic displacement tensor.
    # In Gemmi, aniso is stored.
    # Let's check if the unit cell can transform the tensor to orthogonal/cartesian representation!
    # gemmi.UnitCell has methods to orthogonalize/fractionalize.
    # Let's see if we can find any methods on UnitCell or Atom:
    print("atom.aniso.as_mat33():", atom.aniso.as_mat33())
    print("atom.aniso.elements_pdb():", atom.aniso.elements_pdb())
    print("atom.aniso.elements_voigt():", atom.aniso.elements_voigt())
    print("atom.aniso.trace():", atom.aniso.trace())
    
    # Calculator for structure factor
    calc = gemmi.StructureFactorCalculatorX(cell)
    
    # We will test multiple HKLs
    hkls = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 0, -1),
        (1, 1, 1),
        (2, 1, 3)
    ]
    
    atom_in_struct = struct[0][0][0][0]
    for h, k, ell in hkls:
        # 1. Structure factor with B != 0 (aniso)
        atom_in_struct.aniso.u11 = u11
        atom_in_struct.aniso.u22 = u22
        atom_in_struct.aniso.u33 = u33
        atom_in_struct.aniso.u12 = u12
        atom_in_struct.aniso.u13 = u13
        atom_in_struct.aniso.u23 = u23
        f_aniso = calc.calculate_sf_from_model(struct[0], [h, k, ell])
        
        # 2. Structure factor with B = 0 (aniso = 0)
        atom_in_struct.aniso.u11 = 0.0
        atom_in_struct.aniso.u22 = 0.0
        atom_in_struct.aniso.u33 = 0.0
        atom_in_struct.aniso.u12 = 0.0
        atom_in_struct.aniso.u13 = 0.0
        atom_in_struct.aniso.u23 = 0.0
        f_zero = calc.calculate_sf_from_model(struct[0], [h, k, ell])
        
        # Gemmi DW factor is the ratio of their absolute values
        dw_gemmi = abs(f_aniso) / abs(f_zero)
        
        # Test our implemented functions
        atom_dict = {
            "adp_type": "anis",
            "u11": u11,
            "u22": u22,
            "u33": u33,
            "u12": u12,
            "u13": u13,
            "u23": u23
        }
        cell_tuple = (a, b, c, alpha, beta, gamma)
        
        dw_calc = atom_dw(atom_dict, [h, k, ell], cell_tuple)
        dw_array_calc = atom_dw_array(
            atom_dict,
            np.array([float(h)]),
            np.array([float(k)]),
            np.array([float(ell)]),
            np.array([0.0]),
            cell_tuple
        )[0]
        
        print(f"HKL ({h},{k},{ell}):")
        print(f"  Gemmi DW ratio: {dw_gemmi}")
        print(f"  Calculated dw:  {dw_calc} (diff: {abs(dw_gemmi - dw_calc):.2e})")
        print(f"  Array calc dw:  {dw_array_calc} (diff: {abs(dw_gemmi - dw_array_calc):.2e})")
        
        assert abs(dw_gemmi - dw_calc) < 1e-7
        assert abs(dw_gemmi - dw_array_calc) < 1e-7


def test_anisotropic_dw_array_accepts_meshgrid_hkls():
    cell_tuple = (10.0, 15.0, 20.0, 90.0, 110.0, 90.0)
    atom_dict = {
        "adp_type": "anis",
        "u11": 0.05,
        "u22": 0.04,
        "u33": 0.03,
        "u12": 0.01,
        "u13": 0.02,
        "u23": 0.015,
    }
    h, k, ell = np.meshgrid(
        np.array([-1.0, 0.0, 1.0]),
        np.array([0.0, 2.0]),
        np.array([-2.0, 3.0]),
        indexing="ij",
    )
    got = atom_dw_array(atom_dict, h, k, ell, np.zeros_like(h), cell_tuple)
    assert got.shape == h.shape
    for index in np.ndindex(h.shape):
        expected = atom_dw(atom_dict, [h[index], k[index], ell[index]], cell_tuple)
        assert abs(float(got[index]) - expected) < 1e-12


def test_multiprocessing_safe(tmp_path):
    import sys
    from pathlib import Path
    legacy_engine_dir = str(Path(__file__).resolve().parents[1] / "src" / "grond" / "legacy" / "engine")
    if legacy_engine_dir not in sys.path:
        sys.path.insert(0, legacy_engine_dir)
        
    hkl_file = tmp_path / "INTEGRATE.HKL"
    res_file = tmp_path / "model.res"
    
    # Write mock INTEGRATE.HKL
    hkl_file.write_text(
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
        " 1 0 0 100.0 10.0 0.0 0.0 0.0 0.05 100.0 95.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
        " 0 1 0 200.0 20.0 0.0 0.0 0.0 0.05 100.0 95.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n"
    )
    
    # Write mock model.res
    res_file.write_text(
        "TITL Mock\n"
        "CELL 1.0 10.0 10.0 10.0 90.0 90.0 90.0\n"
        "LATT -1\n"
        "SFAC C 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n"
        "UNIT 1\n"
        "C1 1 0.0 0.0 0.0 11.0 0.05\n"
        "END\n"
    )
    
    from grond.legacy.engine.bloch_engine import Config, XDSBlochPort, XDSData, parse_model
    from grond.legacy.engine.support_basis_scan import scan_basis_sizes
    
    data = XDSData(hkl_file)
    cfg = Config(n_basis=10, n_points=1, topup_d_min=1.0, max_support_sg=0.005)
    model = parse_model(res_file)
    
    port = XDSBlochPort(data, model, cfg)
    
    # Test support scanning with 2 workers (multiprocessing pool)
    sizes = scan_basis_sizes(port, np.array([0, 1]), workers=2, chunksize=1)
    assert len(sizes) == 2
