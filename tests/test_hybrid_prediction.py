"""Opt-in GPU row workers next to the CPU pool give the CPU rows, and the environment defaults are explicit."""

import multiprocessing

import numpy as np
import pytest

from grond.bloch import config as config_module, prediction, spectral_gpu
from grond.bloch.config import Config
from grond.legacy.engine import spectra_stage as S
from tests._nested_support import write_dataset
from tests.test_device_spectra import device_stub  # noqa: F401  (fixture)

COMMON = dict(n_basis=12, n_points=5, topup_d_min=1.2, ug_d_min=1.0, max_support_sg=0.05, thickness_model="gamma",
              thickness_A=600.0, thickness_gamma_shape=2.0, thickness_evaluator="table")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("hybrid"), max_rows=10)


def _port(dataset, **fields):
    hkl, res = dataset
    return S.build_port(str(hkl), "identity", str(res), Config(**{**COMMON, **fields}))


def test_device_list_and_eligibility(dataset, monkeypatch):
    monkeypatch.setenv(prediction.HYBRID_GPU_ENV, " 0, 1,1 ,")
    assert prediction.hybrid_gpu_devices() == ["0", "1", "1"]
    assert _port(dataset)._hybrid_gpu_devices() == ["0", "1", "1"]
    assert _port(dataset, thickness_evaluator="cf")._hybrid_gpu_devices() == []
    assert _port(dataset, paired_inverse_hand=True)._hybrid_gpu_devices() == []
    assert _port(dataset, compute_backend="jax_cuda")._hybrid_gpu_devices() == []
    monkeypatch.delenv(prediction.HYBRID_GPU_ENV)
    assert _port(dataset)._hybrid_gpu_devices() == []


@pytest.mark.parametrize("double", [False, True])
def test_device_row_equals_cpu_row(dataset, device_stub, double):  # noqa: F811
    port = _port(dataset, double_precision=double)
    for row in range(3):
        cpu = port.row_prediction(row)
        device = S.device_row_prediction(port, row)
        assert device.keys() == cpu.keys()
        np.testing.assert_allclose(device["dyn"], cpu["dyn"], rtol=2e-5 if not double else 1e-10)
        assert device["born"] == cpu["born"]
    assert set(device_stub["dtypes"]) <= {np.dtype(np.float32 if not double else np.float64),
                                          np.dtype(np.complex64 if not double else np.complex128)}


def test_single_precision_compact_solve_uses_float32(device_stub):  # noqa: F811
    rng = np.random.default_rng(3)
    a = rng.normal(size=(5, 5)) * 1e-3
    static = (a + a.T).astype(np.float32).astype(complex)
    np.fill_diagonal(static, 0)
    diagonal = (rng.normal(size=(4, 5)) * 1e-2).astype(np.float32).astype(float)
    values, coupling = spectral_gpu.hermitian_spectra_from_static(static, diagonal, "jax_cuda", 2, single_precision=True)
    assert set(device_stub["dtypes"]) == {np.dtype(np.float32)}
    dense = np.broadcast_to(static.real.astype(np.float32), (4, 5, 5)).copy()
    dense[:, np.arange(5), np.arange(5)] = diagonal.astype(np.float32)
    expected, vectors = np.linalg.eigh(dense)
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(coupling, (vectors[:, 1, :] * vectors[:, 0, :]).astype(complex))


def _fork_gpu_workers(monkeypatch, device_row):
    # Fork instead of spawn so that the GPU workers see the test's stand-in for the device calculation.
    monkeypatch.setattr(prediction, "_gpu_pool_context", lambda: multiprocessing.get_context("fork"))
    monkeypatch.setattr(prediction, "_init_gpu_prediction_worker", lambda port, device: prediction._init_prediction_worker(port))
    monkeypatch.setattr(S, "device_row_prediction", device_row)
    monkeypatch.setattr(prediction, "_pool_context_names", lambda: ["fork"])


def _tagged_device_row(port, row, **_options):
    return {**port.row_prediction(row), "computed_on": "gpu"}


def _failing_device_row(port, row, **_options):
    raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="needs the fork start method")
def test_hybrid_pool_returns_every_row_once_in_order(dataset, monkeypatch, capsys):
    port = _port(dataset)
    rows = np.arange(len(port.data.iobs))
    reference = [port.row_prediction(int(r)) for r in rows]
    _fork_gpu_workers(monkeypatch, _tagged_device_row)
    results = prediction._ordered_prediction_attempt(rows, port._predict_rows_hybrid(rows, 2, ["0", "1"]))
    assert [r["dyn"] for r in results] == [r["dyn"] for r in reference]
    assert any(r.get("computed_on") == "gpu" for r in results)
    assert "rows on GPU devices 0,1" in capsys.readouterr().out


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="needs the fork start method")
def test_failed_gpu_workers_hand_their_rows_to_the_cpu_pool(dataset, monkeypatch, capsys):
    port = _port(dataset)
    rows = np.arange(len(port.data.iobs))
    _fork_gpu_workers(monkeypatch, _failing_device_row)
    results = prediction._ordered_prediction_attempt(rows, port._predict_rows_hybrid(rows, 2, ["0"]))
    assert [r["dyn"] for r in results] == [port.row_prediction(int(r))["dyn"] for r in rows]
    out = capsys.readouterr().out
    assert "CUDA_ERROR_OUT_OF_MEMORY" in out and "1 GPU workers retired" in out


def test_centring_coset_default_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("GROND_BASIS_CENTRING_COSET", raising=False)
    assert Config().basis_centring_coset is False
    monkeypatch.setenv("GROND_BASIS_CENTRING_COSET", "1")
    assert Config().basis_centring_coset is True
    assert Config(basis_centring_coset=False).basis_centring_coset is False
    monkeypatch.setenv("GROND_BASIS_CENTRING_COSET", "maybe")
    with pytest.raises(ValueError, match="GROND_BASIS_CENTRING_COSET"):
        config_module._default_centring_coset()
