"""Nested-search spectra: a matrix seen before is reused exactly, never re-solved, and only on identical bytes."""

from types import SimpleNamespace

import numpy as np

from grond.bloch.hamiltonian import ProfileHamiltonians
from grond.legacy.engine import spectra_stage as S


def _port(static, excitation):
    profile = ProfileHamiltonians(static, excitation)
    weights = np.full(len(excitation), 1.0 / len(excitation))
    return SimpleNamespace(absorption_enabled=lambda: False,
                           row_prediction_plan=lambda row: {"_dyn_matrices": profile, "_profile_weights": weights})


def test_reused_spectra_equal_solved_spectra(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    a = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    static = ((a + a.conj().T) / 2e3).astype(np.complex64)
    untilted = rng.normal(size=(2, 6)) * 1e-3
    narrow = _port(static, untilted)  # zero-width cone: the untilted points only
    wide = _port(static, np.vstack([untilted, rng.normal(size=(2, 6)) * 1e-3]))  # adds two tilted points
    donor = tmp_path / "donor.npz"
    S.write_spectra(donor, S.pack_spectra([S.row_spectrum(narrow, 0)]), {"schema": S.SPECTRA_SCHEMA})

    solved = S.row_spectrum(wide, 0)
    real_solver, calls = S.matrix_spectrum, []
    monkeypatch.setattr(S, "matrix_spectrum", lambda *args: calls.append(1) or real_solver(*args))
    reused = S.row_spectrum(wide, 0, S.load_donor([donor]))

    assert reused[8] == 2 and len(calls) == 2  # the two untilted points come from the donor
    for got, ref in zip(reused[:7], solved[:7], strict=True):
        np.testing.assert_array_equal(got, ref)
    assert reused[7] == solved[7]


def test_digests_change_with_any_matrix_byte():
    static = np.eye(3, dtype=np.complex64)
    excitation = np.zeros((2, 3))
    excitation[1, 2] = 1e-3
    keys = S.matrix_digests(ProfileHamiltonians(static, excitation), False)
    assert keys[0] != keys[1]
    assert keys[0] != S.matrix_digests(ProfileHamiltonians(static, excitation), True)[0]
    assert keys[0] != S.matrix_digests(ProfileHamiltonians(static.astype(np.complex128), excitation), False)[0]
