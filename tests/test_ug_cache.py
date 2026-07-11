import json

import numpy as np
import pytest

from grond.legacy.engine.bloch_engine import (
    DENSE_UG_BUILD_BYTES_PER_VOXEL,
    DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL,
    UG_CACHE_SCHEMA,
    XDSBlochPort,
    dense_ug_memory_plan,
    enforce_dense_ug_memory_limit,
)


def _fake_port(cache_dir, cube_value: complex) -> XDSBlochPort:
    port = object.__new__(XDSBlochPort)
    port.cache_dir = cache_dir
    port.h_lim = 0
    port.k_lim = 0
    port.l_lim = 0
    port.ug_cache_key = lambda: (
        "test-key",
        {"cache_schema": UG_CACHE_SCHEMA, "model": "synthetic"},
    )
    port.build_ug_cube = lambda: np.asarray([[[cube_value]]], dtype=np.complex128)
    return port


def test_dense_ug_guard_fails_before_oversized_allocation(monkeypatch):
    monkeypatch.setenv("GROND_MAX_DENSE_UG_GIB", "0.001")
    plan = dense_ug_memory_plan(50, 50, 50)

    with pytest.raises(MemoryError, match=r"grid 101 x 101 x 101.*GROND_MAX_DENSE_UG_GIB"):
        enforce_dense_ug_memory_limit(plan, operation="build")


def test_dense_ug_plan_keeps_measured_working_memory_headroom():
    plan = dense_ug_memory_plan(3, 4, 5)
    voxels = 7 * 9 * 11

    assert DENSE_UG_BUILD_BYTES_PER_VOXEL >= 256
    assert DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL >= 64
    assert plan["estimated_build_peak_bytes"] == voxels * DENSE_UG_BUILD_BYTES_PER_VOXEL
    assert plan["estimated_cache_load_peak_bytes"] == voxels * DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL


@pytest.mark.parametrize("value", ["0", "nan", "not-a-number"])
def test_dense_ug_guard_rejects_invalid_memory_limit(monkeypatch, value):
    monkeypatch.setenv("GROND_MAX_DENSE_UG_GIB", value)

    with pytest.raises(ValueError, match="positive finite"):
        enforce_dense_ug_memory_limit(dense_ug_memory_plan(0, 0, 0), operation="build")


def test_ug_cache_hit_validates_schema_and_metadata(tmp_path):
    first = _fake_port(tmp_path, 1.0 + 2.0j)
    built = first.load_or_build_ug_cube()

    assert built[0, 0, 0] == 1.0 + 2.0j
    assert first.ug_cache_status["status"] == "built"
    path = tmp_path / "ug_cube_test-key.npz"
    with np.load(path, allow_pickle=False) as loaded:
        meta = json.loads(str(np.asarray(loaded["meta_json"]).item()))
    assert meta == {
        "cache_schema": UG_CACHE_SCHEMA,
        "key": "test-key",
        "model": "synthetic",
    }

    second = _fake_port(tmp_path, 9.0 + 9.0j)
    hit = second.load_or_build_ug_cube()

    assert hit[0, 0, 0] == 1.0 + 2.0j
    assert second.ug_cache_status["status"] == "hit"


def test_ug_cache_corruption_is_rebuilt_atomically(tmp_path):
    path = tmp_path / "ug_cube_test-key.npz"
    path.write_bytes(b"not an npz archive")
    port = _fake_port(tmp_path, 3.0 + 4.0j)

    cube = port.load_or_build_ug_cube()

    assert cube[0, 0, 0] == 3.0 + 4.0j
    assert port.ug_cache_status["status"] == "rebuilt"
    assert "BadZipFile" in port.ug_cache_status["rebuild_reason"]
    assert not list(tmp_path.glob("*.tmp.npz"))
    reloaded = _fake_port(tmp_path, 0.0j)
    assert reloaded.load_or_build_ug_cube()[0, 0, 0] == 3.0 + 4.0j
    assert reloaded.ug_cache_status["status"] == "hit"


def test_ug_cache_nonfinite_values_are_rebuilt(tmp_path):
    path = tmp_path / "ug_cube_test-key.npz"
    np.savez_compressed(
        path,
        ug_cube=np.asarray([[[np.nan + 0.0j]]], dtype=np.complex128),
        lims=np.asarray([0, 0, 0], dtype=np.int64),
        meta_json=json.dumps(
            {
                "cache_schema": UG_CACHE_SCHEMA,
                "key": "test-key",
                "model": "synthetic",
            },
            sort_keys=True,
        ),
    )
    second = _fake_port(tmp_path, 5.0 + 0.0j)

    cube = second.load_or_build_ug_cube()

    assert cube[0, 0, 0] == 5.0 + 0.0j
    assert second.ug_cache_status["status"] == "rebuilt"
    assert "non-finite" in second.ug_cache_status["rebuild_reason"]


def test_nonfinite_ug_build_is_not_cached(tmp_path):
    port = _fake_port(tmp_path, np.nan + 0.0j)

    with pytest.raises(ValueError, match="non-finite"):
        port.load_or_build_ug_cube()

    assert not (tmp_path / "ug_cube_test-key.npz").exists()
    assert not list(tmp_path.glob(".*.tmp.npz"))
