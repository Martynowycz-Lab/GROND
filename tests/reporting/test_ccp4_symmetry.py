"""A nonstandard space-group setting must survive CCP4 map expansion."""
import struct

import gemmi
import numpy as np
import pytest

from grond.maps.ccp4 import normalize_ccp4_map, read_ccp4_map


def partial_n_glide_map(path):
    shape = (12, 16, 20)
    x, y, z = np.meshgrid(*(np.arange(n) / n for n in shape), indexing="ij")
    expected = (np.cos(2 * np.pi * (x + z)) * np.cos(4 * np.pi * y)).astype(np.float32)
    group = gemmi.find_spacegroup_by_name("P 1 21/n 1")
    partial = expected[:, : shape[1] // 4 + 1, :]
    grid = gemmi.FloatGrid(*partial.shape)
    grid.unit_cell = gemmi.UnitCell(6, 8, 10, 90, 100, 90)
    grid.spacegroup = group
    grid.array[:] = partial
    result = gemmi.Ccp4Map()
    result.grid = grid
    result.update_ccp4_header()
    for word, value in zip((8, 9, 10), shape, strict=True):
        result.set_header_i32(word, value)
    result.write_ccp4_map(str(path))
    data = path.read_bytes()
    header = bytearray(data[:1024])
    original_count = struct.unpack_from("<i", header, 23 * 4)[0]
    operators = "".join(op.triplet().ljust(80) for op in group.operations()).encode("ascii")
    struct.pack_into("<i", header, 23 * 4, len(operators))
    struct.pack_into("<i", header, 22 * 4, 14)
    path.write_bytes(header + operators + data[1024 + original_count :])
    return expected


def test_partial_map_uses_n_glide_not_default_c_glide(tmp_path):
    path = tmp_path / "partial.map"
    expected = partial_n_glide_map(path)
    result = read_ccp4_map(path)
    assert result.grid.spacegroup.xhm() == "P 1 21/n 1"
    result.setup(float("nan"))
    np.testing.assert_allclose(np.array(result.grid), expected, atol=1e-6)


def test_normalized_map_is_portable_full_cell_p1(tmp_path):
    path = tmp_path / "partial.map"
    expected = partial_n_glide_map(path)
    metadata = normalize_ccp4_map(path)
    assert metadata["source_spacegroup"] == "P 1 21/n 1"
    result = gemmi.read_ccp4_map(str(path))
    assert result.grid.spacegroup.number == 1
    result.setup(float("nan"))
    np.testing.assert_allclose(np.array(result.grid), expected, atol=1e-6)


def test_conflicting_symmetry_number_is_an_error(tmp_path):
    path = tmp_path / "conflicting.map"
    partial_n_glide_map(path)
    data = bytearray(path.read_bytes())
    struct.pack_into("<i", data, 22 * 4, 19)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="conflict"):
        read_ccp4_map(path)


def test_hydrogen_audit_uses_the_explicit_map_setting(tmp_path):
    from grond.legacy.engine.map_hydrogen_omit_audit import read_map

    path = tmp_path / "partial.map"
    expected = partial_n_glide_map(path)
    _, actual = read_map(path)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_normalization_does_not_discard_cartesian_origin(tmp_path):
    path = tmp_path / "offset.map"
    partial_n_glide_map(path)
    data = bytearray(path.read_bytes())
    struct.pack_into("<f", data, 49 * 4, 1.0)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="Cartesian map origin"):
        normalize_ccp4_map(path)
    assert path.read_bytes() == bytes(data)


def test_extended_ccp4_spacegroup_number_is_accepted(tmp_path):
    path = tmp_path / "extended.map"
    expected = partial_n_glide_map(path)
    data = bytearray(path.read_bytes())
    struct.pack_into("<i", data, 22 * 4, gemmi.find_spacegroup_by_name("P 1 21/n 1").ccp4)
    path.write_bytes(data)
    result = read_ccp4_map(path)
    result.setup(float("nan"))
    np.testing.assert_allclose(np.array(result.grid), expected, atol=1e-6)


def test_matched_map_resampling_preserves_n_glide(tmp_path):
    from grond.workflows.matched_maps import _dense_map

    source, output = tmp_path / "partial.map", tmp_path / "dense.map"
    partial_n_glide_map(source)
    model = {"cell": (6, 8, 10, 90, 100, 90)}
    _dense_map(source, output, model, 0.25)
    result = read_ccp4_map(output)
    result.setup(float("nan"))
    values = np.array(result.grid)
    x, y, z = np.meshgrid(*(np.arange(n) / n for n in values.shape), indexing="ij")
    expected = np.cos(2 * np.pi * (x + z)) * np.cos(4 * np.pi * y)
    np.testing.assert_allclose(values, expected, atol=1e-6)
