"""Beams outside the direct beam's lattice-centring coset do not couple to it, so dropping them changes nothing."""

import numpy as np
import pytest

from grond.bloch import config as config_module
from grond.bloch.config import Config
from grond.bloch.potential import centring_coset_mask
from grond.engine import spectra_stage as S
from grond.regularize.nested_selection import ThicknessSupport, thickness_table
from tests._nested_support import write_dataset

I_CENTRED_MODEL = (
    "TITL body-centred fixture\n"
    "CELL 0.025079 8.0 9.0 10.0 90.0 90.0 90.0\n"
    "LATT -2\n"
    "SFAC C 1 1 1 1 1 1 1 1 0\n"
    "SFAC O 1 1 1 1 1 1 1 1 0\n"
    "C1 1 0.13 0.27 0.39 11.0 0.02\n"
    "C2 1 0.31 0.12 0.21 11.0 0.03\n"
    "O1 2 0.42 0.37 0.08 11.0 0.025\n"
    "HKLF 4\n"
    "END\n"
)


def test_coset_mask_follows_the_centring_translations():
    hkl = np.asarray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (-1, 1, 1), (2, 1, 0), (1, 2, 0), (3, 0, 0)])
    assert centring_coset_mask(hkl, 1).all()
    assert centring_coset_mask(hkl, -2).tolist() == [int(g.sum()) % 2 == 0 for g in hkl]
    assert centring_coset_mask(hkl, 3).tolist() == [(-g[0] + g[1] + g[2]) % 3 == 0 for g in hkl]


def test_coset_basis_reproduces_the_full_basis_intensities(tmp_path):
    hkl_path, res = write_dataset(tmp_path, max_rows=24)
    res.write_text(I_CENTRED_MODEL)
    support = ThicknessSupport(100.0, 3000.0, 2.0)
    common = dict(n_basis=1000, n_points=3, topup_d_min=1.2, ug_d_min=1.0, max_support_sg=0.02)
    full_port = S.build_port(str(hkl_path), "identity", str(res), Config(**common, basis_centring_coset=False))
    coset_port = S.build_port(str(hkl_path), "identity", str(res), Config(**common, basis_centring_coset=True))
    rows = [i for i, h in enumerate(full_port.data.hkl) if centring_coset_mask([h], -2)[0]][:6]
    assert len(rows) >= 3
    full_records, coset_records = [], []
    for row in rows:
        full_plan = full_port.row_prediction_plan(row)
        coset_plan = coset_port.row_prediction_plan(row)
        full_basis = np.asarray(full_plan["support_hkls"])
        coset_basis = np.asarray(coset_plan["support_hkls"])
        assert centring_coset_mask(coset_basis, -2).all()
        assert not centring_coset_mask(full_basis, -2).all()  # without a binding cap the legacy basis holds both cosets
        assert {tuple(h) for h in full_basis[centring_coset_mask(full_basis, -2)]} == {tuple(h) for h in coset_basis}
        full_records.append(S.row_spectrum(full_port, row))
        coset_records.append(S.row_spectrum(coset_port, row))
    full = thickness_table(S.pack_spectra(full_records), support, hands=(1, -1))
    coset = thickness_table(S.pack_spectra(coset_records), support, hands=(1, -1))
    for key in (1, -1, "born"):
        np.testing.assert_allclose(coset[key], full[key], rtol=1e-10, atol=1e-14 * np.max(full[key]))


def test_centring_coset_default_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("GROND_BASIS_CENTRING_COSET", raising=False)
    assert Config().basis_centring_coset is True
    monkeypatch.setenv("GROND_BASIS_CENTRING_COSET", "0")
    assert Config().basis_centring_coset is False
    assert Config(basis_centring_coset=False).basis_centring_coset is False
    monkeypatch.setenv("GROND_BASIS_CENTRING_COSET", "maybe")
    with pytest.raises(ValueError, match="GROND_BASIS_CENTRING_COSET"):
        config_module._default_centring_coset()
