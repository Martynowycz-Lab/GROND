"""complex64 (default production precision) must stay within 1e-5 of complex128.

The rest of the physics suite runs in complex128 (see tests/conftest.py) and
checks analytic identities at ~1e-12. These tests pin the same identities for
the complex64 path at a tolerance appropriate to single precision.
"""

import numpy as np
import pytest

from grond.bloch.config import Config, _default_double_precision


def test_precision_env_default(monkeypatch):
    monkeypatch.setenv("GROND_PRECISION", "float32")
    assert _default_double_precision() is False
    assert Config().double_precision is False
    monkeypatch.setenv("GROND_PRECISION", "float64")
    assert _default_double_precision() is True
    assert Config().double_precision is True
    assert Config(double_precision=False).double_precision is False
    monkeypatch.setenv("GROND_PRECISION", "bogus")
    with pytest.raises(ValueError):
        _default_double_precision()


def _two_beam(dtype):
    u = 2.0e-3 + 1.0e-3j
    sg = 3.0e-3
    return np.array([[0.0, np.conj(u)], [u, sg]], dtype=dtype)


@pytest.mark.parametrize("thickness", [200.0, 1000.0, 2500.0])
def test_complex64_two_beam_matches_complex128(thickness):
    from scipy.linalg import expm

    ref = abs(expm(2j * np.pi * thickness * _two_beam(np.complex128))[1, 0]) ** 2
    h32 = _two_beam(np.complex64)
    lam, vec = np.linalg.eigh(h32)
    amp = vec[1, :] @ (np.exp(2j * np.pi * lam * thickness).astype(np.complex64) * np.conj(vec[0, :]))
    assert abs(amp) ** 2 == pytest.approx(ref, rel=1e-5, abs=1e-9)


def test_complex64_gauge_shift_invariance():
    rng = np.random.default_rng(7)
    n = 64
    a = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * 1e-4
    h = (a + a.conj().T) / 2.0
    h[np.diag_indices(n)] = rng.uniform(-0.01, 0.01, size=n)
    t = 1357.0

    def target(hm):
        lam, vec = np.linalg.eigh(hm)
        amp = vec[1, :] @ (np.exp(2j * np.pi * lam * t) * np.conj(vec[0, :]))
        return abs(amp) ** 2

    ref = target(h.astype(np.complex128))
    shifted = h.astype(np.complex64) + np.float32(0.004) * np.eye(n, dtype=np.complex64)
    assert target(shifted) == pytest.approx(ref, rel=2e-4, abs=1e-10)
