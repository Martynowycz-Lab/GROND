"""Independent reciprocal metrics and nonorthogonal PETS geometry roundtrips."""

from pathlib import Path

import gemmi
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from grond.io.pets_import import _busing_levy, import_pets_to_grond, read_pets_contract
from grond.io.xds import XDSData


@pytest.mark.parametrize(
    "parameters",
    [
        (5, 6, 7, 90, 90, 90),
        (5, 6, 7, 90, 100, 90),
        (5, 6, 7, 80, 90, 90),
        (5, 6, 7, 80, 100, 110),
        (7, 7, 7, 75, 75, 75),
        (5, 5, 7, 90, 90, 120),
    ],
)
def test_pets_reciprocal_metric_matches_gemmi(parameters: tuple[float, ...]) -> None:
    cell = gemmi.UnitCell(*parameters)
    reciprocal_basis = _busing_levy(np.asarray(parameters, dtype=float))
    # Gemmi supplies an independently implemented conventional fractionalization
    # matrix. Its transpose is the column reciprocal basis for this cell frame.
    reference = np.asarray(cell.frac.mat, dtype=float).T
    np.testing.assert_allclose(reciprocal_basis, reference, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(reciprocal_basis.T @ reciprocal_basis, reference.T @ reference, rtol=1e-13, atol=1e-15)
    for hkl in ((1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (-3, 2, -4)):
        assert np.linalg.norm(reciprocal_basis @ hkl) == pytest.approx(
            1.0 / cell.calculate_d(hkl), rel=1e-13, abs=1e-15
        )


def _write_triclinic_contract(tmp_path: Path) -> tuple[Path, list[np.ndarray], np.ndarray]:
    parameters = (5, 6, 7, 80, 100, 110)
    cell = gemmi.UnitCell(*parameters)
    reference_basis = np.asarray(cell.frac.mat, dtype=float).T
    orientation = Rotation.from_rotvec(np.asarray([0.12, -0.31, 0.27])).as_matrix()
    ub = orientation @ reference_basis
    axis_position = 17.0
    axis_correction = Rotation.from_euler("z", -axis_position, degrees=True).as_matrix()
    lines = ["data_independent_triclinic_pets"]
    for tag, value in zip(
        ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"),
        parameters,
        strict=True,
    ):
        lines.append(f"_cell_{tag} {value}")
    lines += [
        "_diffrn_radiation_wavelength 0.025",
        "_diffrn_measurement_details",
        ";",
        f"rotation axis position: {axis_position}",
        ";",
        "_diffrn_measurement_method_precession no",
        "_diffrn_measurement_rotation_mode rotation",
    ]
    for i in range(3):
        for j in range(3):
            lines.append(f"_diffrn_orient_matrix_UB_{i + 1}{j + 1} {ub[i, j]:.16g}")
    lines.append("loop_")
    lines += [
        "_diffrn_zone_axis_" + tag
        for tag in ("id", "u", "v", "w", "precession_angle", "alpha", "beta", "omega", "scale")
    ]
    expected_frames = []
    for ordinal, (alpha, beta, omega) in enumerate(((-20, 3, 7), (-18, 4, 8), (-16, 5, 9)), start=1):
        rx = Rotation.from_euler("x", alpha, degrees=True).as_matrix()
        ry = Rotation.from_euler("y", beta, degrees=True).as_matrix()
        rz = Rotation.from_euler("z", omega, degrees=True).as_matrix()
        frame = axis_correction @ rz @ rx @ ry @ ub
        expected_frames.append(frame)
        zone = frame.T @ np.asarray([0.0, 0.0, 1.0])
        zone /= np.max(np.abs(zone))
        lines.append(f"{ordinal} {zone[0]:.16g} {zone[1]:.16g} {zone[2]:.16g} 0.2 {alpha} {beta} {omega} 1")
    lines.append("loop_")
    lines += [
        "_refln_" + tag
        for tag in ("index_h", "index_k", "index_l", "intensity_meas", "intensity_sigma", "zone_axis_id")
    ]
    lines += ["1 0 1 10 1 1", "0 1 1 -20 2 2", "-1 2 1 30 3 3"]
    source = tmp_path / "triclinic.cif_pets"
    source.write_text("\n".join(lines) + "\n")
    source.with_suffix(".hkl").write_text(
        "1 0 1 10 1 1 0 0 0 -1 0 0 -1\n0 1 1 -20 2 2 0 0 0 -1 0 0 -1\n-1 2 1 30 3 3 0 0 0 -1 0 0 -1\n"
    )
    return source, expected_frames, reference_basis


def test_triclinic_pets_import_preserves_independent_frame_geometry(tmp_path: Path) -> None:
    source, expected_frames, reference_basis = _write_triclinic_contract(tmp_path)
    contract = read_pets_contract(source)
    np.testing.assert_allclose(contract["a_matrices"], expected_frames, rtol=1e-13, atol=1e-15)
    output = import_pets_to_grond(source, tmp_path / "bridge", space_group_number=1)
    imported = XDSData(output.hkl)
    np.testing.assert_allclose(imported.pets_exact_a_center, expected_frames, rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(imported.hkl, [[1, 0, 1], [0, 1, 1], [-1, 2, 1]])
    np.testing.assert_array_equal(imported.iobs, [10, -20, 30])
    np.testing.assert_array_equal(imported.sigma, [1, 2, 3])
    for frame in imported.pets_exact_a_center:
        np.testing.assert_allclose(frame.T @ frame, reference_basis.T @ reference_basis, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"length_a": "-5"}, "lengths must be positive"),
        ({"length_c": "0"}, "lengths must be positive"),
        ({"angle_alpha": "0"}, "angles must lie strictly"),
        ({"angle_gamma": "180"}, "angles must lie strictly"),
        ({"angle_alpha": "20", "angle_beta": "20", "angle_gamma": "170"}, "positive cell volume"),
        ({"length_b": "nan"}, "must be finite"),
    ],
)
def test_pets_import_rejects_nonphysical_cell(tmp_path: Path, replacement: dict[str, str], message: str) -> None:
    source, _, _ = _write_triclinic_contract(tmp_path)
    document = gemmi.cif.read_file(str(source))
    block = document.sole_block()
    for tag, value in replacement.items():
        block.set_pair("_cell_" + tag, value)
    document.write_file(str(source))
    with pytest.raises(ValueError, match=message):
        read_pets_contract(source)


def test_pets_import_rejects_ub_scale_mismatch(tmp_path: Path) -> None:
    source, _, _ = _write_triclinic_contract(tmp_path)
    document = gemmi.cif.read_file(str(source))
    block = document.sole_block()
    for i in range(1, 4):
        for j in range(1, 4):
            tag = f"_diffrn_orient_matrix_UB_{i}{j}"
            block.set_pair(tag, str(2.0 * float(block.find_value(tag))))
    document.write_file(str(source))
    with pytest.raises(ValueError, match="UB reciprocal metric is inconsistent"):
        read_pets_contract(source)


@pytest.mark.parametrize("perturbation", [0.0, 4.9e-6, -4.9e-6, 1e-3])
def test_pets_ub_precision_allowance_is_bounded(tmp_path: Path, perturbation: float) -> None:
    source, _, _ = _write_triclinic_contract(tmp_path)
    document = gemmi.cif.read_file(str(source))
    block = document.sole_block()
    for i in range(1, 4):
        for j in range(1, 4):
            tag = f"_diffrn_orient_matrix_UB_{i}{j}"
            value = float(block.find_value(tag))
            # The zero case represents actual five-decimal serialization; the
            # two half-digit cases bound entrywise quantization in either sign.
            replacement = f"{value:.5f}" if perturbation == 0.0 else f"{value + perturbation:.16g}"
            block.set_pair(tag, replacement)
    document.write_file(str(source))
    if abs(perturbation) > 5e-6:
        with pytest.raises(ValueError, match="UB reciprocal metric is inconsistent"):
            read_pets_contract(source)
    else:
        assert len(read_pets_contract(source)["observations"]) == 3
