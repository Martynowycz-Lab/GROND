"""Full row-spectrum adapter checks; only the device eigensolver is replaced by NumPy.

Independent matrix exponentials test the actual profile and uniform-loss semantics. These tests do not constitute
CUDA execution or speed evidence.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from grond.bloch import backends, spectral_gpu
from grond.bloch.config import Config
from grond.bloch.hamiltonian import ProfileHamiltonians
from grond.bloch.source import PHYSICS_SOURCE_PATHS
from grond.legacy.engine import spectra_stage as S


@pytest.fixture
def device_stub(monkeypatch):
    calls = {"dtypes": [], "matrices": 0, "compiled": 0}

    def eigh(matrices):
        calls["dtypes"].append(matrices.dtype)
        calls["matrices"] += len(matrices)
        return np.linalg.eigh(matrices)

    def pmap(function, *, devices):
        calls["compiled"] += 1

        def mapped(shards):
            assert len(shards) == len(devices)
            parts = [function(shard) for shard in shards]
            return tuple(np.stack([part[i] for part in parts]) for i in range(2))

        return mapped

    jax = SimpleNamespace(pmap=pmap)
    jnp = SimpleNamespace(linalg=SimpleNamespace(eigh=eigh), conj=np.conj)
    devices = tuple(SimpleNamespace(platform="gpu", id=i) for i in range(2))
    monkeypatch.setattr(spectral_gpu, "_jax_modules", lambda backend: (jax, jnp))
    monkeypatch.setattr(spectral_gpu, "_jax_devices", lambda backend: devices)
    monkeypatch.setattr(spectral_gpu, "_SOLVERS", {})
    return calls


def profile(mu=0.0, phase_shift=False, points=5):
    rng = np.random.default_rng(27)
    a = rng.normal(size=(6, 6)) * 0.00015
    static = (a + a.T).astype(np.complex128)
    if phase_shift:
        phase = np.exp(1j * np.linspace(0., 1.9, 6))
        static = phase[:, None] * static * phase.conj()[None, :]
    static += 1j * mu * np.eye(6)
    sg = rng.normal(size=(points, 6)) * 0.0003
    return ProfileHamiltonians(static, sg), np.linspace(0.3, 1.1, points)


def port(matrices, weights, *, backend="jax_cuda", mu=0.0):
    cfg = Config(compute_backend=backend, jax_batch_size=2)
    return SimpleNamespace(cfg=cfg, absorption_enabled=lambda: mu != 0.0,
                           row_prediction_plan=lambda row: {"_dyn_matrices": matrices,
                                                            "_profile_weights": weights})


def integrated_from_record(record, thicknesses, weights):
    _, profile_weights, lam, coupling, _, _, mu, _, _ = record
    phase = np.exp(2j * np.pi * lam[:, :, None] * thicknesses[None, None, :])
    amplitudes = np.sum(coupling[:, :, None] * phase, axis=1)
    per_point = (abs(amplitudes)**2 * np.exp(-4 * np.pi * mu * thicknesses)) @ weights
    return profile_weights @ per_point


def independent_intensity(matrices, profile_weights, thicknesses, weights):
    return sum(pw * sum(tw * abs(expm(2j * np.pi * t * matrix)[1, 0])**2
                        for t, tw in zip(thicknesses, weights, strict=True))
               for matrix, pw in zip(matrices, profile_weights, strict=True))


@pytest.mark.parametrize("mu", [0.0, 2e-5])
@pytest.mark.parametrize("phase_shift", [False, True])
def test_device_row_spectra_preserve_profile_uniform_loss_and_origin_phase(device_stub, mu, phase_shift):
    matrices, profile_weights = profile(mu, phase_shift)
    result = S.row_spectrum(port(matrices, profile_weights, mu=mu), 7)
    thicknesses, weights = np.array([0., 100., 850., 3000.]), np.array([0.1, 0.2, 0.4, 0.3])
    expected = independent_intensity(matrices, profile_weights, thicknesses, weights)
    assert integrated_from_record(result, thicknesses, weights) == pytest.approx(expected, rel=2e-12, abs=1e-14)
    assert result[0] == 7 and result[6] == pytest.approx(mu, abs=1e-18)
    np.testing.assert_array_equal(result[1], profile_weights)  # Keep the engine's sum convention; do not normalize.
    np.testing.assert_allclose(result[4], [abs(matrix[1, 0])**2 for matrix in matrices], rtol=1e-14)
    expected_dtype = np.dtype("complex128" if phase_shift else "float64")
    assert device_stub["dtypes"] and set(device_stub["dtypes"]) == {expected_dtype}


def test_device_spectra_reuse_exact_donor_points_and_reject_changed_matrix_keys(device_stub, tmp_path):
    matrices, weights = profile(mu=2e-5)
    narrow = ProfileHamiltonians(matrices.static, matrices.excitation[:2])
    donor_record = S.row_spectrum(port(narrow, weights[:2], backend="scipy", mu=2e-5), 7)
    donor_path = tmp_path / "donor.npz"
    S.write_spectra(donor_path, S.pack_spectra([donor_record]),
                    {"schema": S.SPECTRA_SCHEMA, "engine_sha256": S.engine_source_sha256()})
    donor = S.load_donor([donor_path])
    result = S.row_spectrum(port(matrices, weights, mu=2e-5), 7, donor)
    assert result[8] == 2
    np.testing.assert_array_equal(result[2][:2], donor_record[2])
    np.testing.assert_array_equal(result[3][:2], donor_record[3])
    assert device_stub["matrices"] == 4  # Three missing matrices, plus one padded slot across two devices.
    thicknesses, thickness_weights = np.array([50., 900.]), np.array([0.3, 0.7])
    assert integrated_from_record(result, thicknesses, thickness_weights) == pytest.approx(
        independent_intensity(matrices, weights, thicknesses, thickness_weights), rel=2e-12, abs=1e-14)
    moved = matrices.static.copy()
    moved[0, 1] += 1e-7
    moved[1, 0] += 1e-7
    changed = ProfileHamiltonians(moved, matrices.excitation)
    assert set(S.matrix_digests(changed, True)).isdisjoint(donor)
    assert set(S.matrix_digests(matrices, False)).isdisjoint(donor)  # Loss semantics are part of each digest.


def test_nonuniform_absorption_refused_before_device_eigh(device_stub):
    matrices, weights = profile(mu=1e-5)
    static = matrices.static.copy()
    static[0, 0] += 2e-5j
    matrices = ProfileHamiltonians(static, matrices.excitation)
    with pytest.raises(ValueError, match="non-uniform or off-diagonal"):
        S.row_spectrum(port(matrices, weights, mu=1e-5), 2)
    assert not device_stub["dtypes"]
    with pytest.raises(ValueError, match="non-Hermitian"):
        spectral_gpu.hermitian_spectra(np.asarray(matrices), "jax_cuda")
    assert not device_stub["dtypes"]


def test_device_compute_spectra_does_not_fork_and_cache_binds_source_inputs(device_stub, monkeypatch, tmp_path):
    hkl, res = tmp_path / "input.hkl", tmp_path / "model.res"
    hkl.write_text("input snapshot\n")
    res.write_text("model snapshot\n")
    matrices, weights = profile(points=2)
    fake_port = port(matrices, weights)
    process = {"inside_child": False, "starts": 0, "closes": 0, "chunks": []}

    def build_in_child(*args, **kwargs):
        assert process["inside_child"], "the parent must not build or initialize the device port"
        return fake_port

    class SpawnedPool:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            assert max_workers == 1 and mp_context.get_start_method() == "spawn"
            self.initializer, self.initargs = initializer, initargs

        def __enter__(self):
            process["starts"] += 1
            process["inside_child"] = True
            self.initializer(*self.initargs)
            return self

        def map(self, function, chunks):
            assert process["inside_child"]
            process["chunks"].extend(len(chunk) for chunk in chunks)
            return map(function, chunks)

        def __exit__(self, *args):
            process["closes"] += 1
            process["inside_child"] = False

    def forbidden_fork_context(*args, **kwargs):
        raise AssertionError("device spectra must not use the ordinary Linux fork context")

    monkeypatch.setattr(S, "build_port", build_in_child)
    monkeypatch.setattr(S, "ProcessPoolExecutor", SpawnedPool)
    monkeypatch.setattr(S, "_pool_context", forbidden_fork_context)
    rows = np.array([3, 1])
    spec, info = S.compute_spectra(hkl, res, fake_port.cfg, rows, workers=8, cache_dir=tmp_path / "cache")
    assert info["workers"] == 1 and info["spectral_backend"] == "jax_cuda"
    assert info["device_process"] == "spawn"
    assert not process["inside_child"] and process["starts"] == process["closes"] == 1
    np.testing.assert_array_equal(spec["row_ids"], rows)
    count = device_stub["matrices"]
    same, cached = S.compute_spectra(hkl, res, fake_port.cfg, rows, workers=8, cache_dir=tmp_path / "cache")
    assert cached["cache_status"] == "hit" and device_stub["matrices"] == count
    assert process["starts"] == 1  # Reusing cached spectra must not initialize a device process.
    np.testing.assert_array_equal(same["lam"], spec["lam"])
    res.write_text("changed model snapshot\n")
    _, changed = S.compute_spectra(hkl, res, fake_port.cfg, rows, workers=8, cache_dir=tmp_path / "cache")
    assert changed["key"] != info["key"] and device_stub["matrices"] > count
    assert process["starts"] == process["closes"] == 2 and not process["inside_child"]
    # A single requested row with workers=1 still needs the spawned owner.
    _, single = S.compute_spectra(hkl, res, fake_port.cfg, [7], workers=1)
    assert single["workers"] == 1 and process["starts"] == process["closes"] == 3
    assert all(size <= 32 for size in process["chunks"])


def test_new_solver_module_is_in_physics_manifest():
    assert "bloch/spectral_gpu.py" in PHYSICS_SOURCE_PATHS


def test_solver_source_change_invalidates_cache_identity(monkeypatch, tmp_path):
    helper = tmp_path / "spectral_gpu.py"
    helper.write_text("first implementation\n")
    monkeypatch.setattr(S, "physics_source_files", lambda: (helper,))
    first = S.engine_source_sha256()
    helper.write_text("changed implementation\n")
    assert S.engine_source_sha256() != first


@pytest.mark.parametrize("identity", [{}, {"schema": S.SPECTRA_SCHEMA},
                                       {"schema": S.SPECTRA_SCHEMA, "engine_sha256": "stale-source"},
                                       {"schema": "old-spectra", "engine_sha256": "stale-source"}])
def test_unbound_or_stale_donor_is_rejected(tmp_path, identity):
    matrices, weights = profile(points=1)
    record = S.row_spectrum(port(matrices, weights, backend="scipy"), 1)
    path = tmp_path / "donor.npz"
    S.write_spectra(path, S.pack_spectra([record]), identity)
    with pytest.raises(ValueError, match="identity|source|engine|schema"):
        S.load_donor([path])


def test_one_matrix_larger_than_host_budget_is_refused(device_stub, monkeypatch):
    matrices, weights = profile(points=1)
    monkeypatch.setattr(backends, "_jax_host_buffer_bytes", lambda: 1)
    with pytest.raises(MemoryError, match="matrix|buffer|budget"):
        S.row_spectrum(port(matrices, weights), 0)
    assert not device_stub["dtypes"]
