"""Shared held-out primitives must reproduce the reference definitions (tests/reference/reference_lib.py)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from grond.validation import heldout as H

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
import reference_lib as G  # noqa: E402

MMM = [{"rot": np.diag(d)} for d in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1])]


def test_mmm_orbit_key_and_folds_match_the_reference():
    rng = np.random.default_rng(3)
    hkls = rng.integers(-9, 10, size=(300, 3))
    keys = H.orbit_keys(hkls, MMM)
    assert list(keys) == [G.orbit_key(*h) for h in hkls]
    folds = H.hash_fold(keys, G.DEV_SALT, 3)
    assert list(folds) == [G.hash_mod(G.DEV_SALT, k, 3) for k in keys]


def test_orbit_key_is_invariant_over_the_orbit_and_friedel():
    base = H.orbit_key((1, -2, 3), MMM)
    for s in ([1, 1, 1], [-1, 1, 1], [1, -1, -1], [-1, -1, -1]):
        assert H.orbit_key(np.array([1, -2, 3]) * s, MMM) == base
    assert H.orbit_key((1, 2, 3)) == H.orbit_key((-1, -2, -3)) != H.orbit_key((1, -2, 3))


def test_free_mask_is_crystal_independent_and_near_the_requested_fraction():
    keys = [f"{h}|{k}|{ell}" for h in range(12) for k in range(12) for ell in range(12)]
    free = H.free_mask(keys, "salt", 0.10)
    assert 0.07 < free.mean() < 0.13
    again = H.free_mask(keys[::-1], "salt", 0.10)[::-1]
    assert np.array_equal(free, again)


def _write_lockbox(path: Path, orbits: list[str], shell_dmin: list[float]) -> str:
    sha = hashlib.sha256("\n".join(orbits).encode()).hexdigest()
    shells = [{"shell": i, "d_min": d} for i, d in enumerate(shell_dmin)]
    path.write_text(json.dumps({"fraction": 0.2, "lockbox_orbits": orbits, "lockbox_sha256": sha, "shells": shells}))
    return sha


def test_lockbox_loader_checks_the_hash(tmp_path):
    orbits = [f"{h}|{k}|{ell}" for h in range(1, 4) for k in range(3) for ell in range(1, 4)]
    shell_dmin = [1.4, 1.1, 0.95, 0.87, 0.8]
    lock = tmp_path / "lockbox.json"
    sha = _write_lockbox(lock, orbits, shell_dmin)
    doc = H.load_orbit_exclusion(lock)
    assert doc["orbits"] == set(orbits) and len(doc["orbits"]) == 27
    assert doc["shell_dmin"] == shell_dmin and doc["sha256"] == sha
    bad = tmp_path / "bad.json"
    bad.write_text(lock.read_text().replace(sha, "0" * 64))
    with pytest.raises(ValueError, match="hash mismatch"):
        H.load_orbit_exclusion(bad)


def synthetic(n=900, seed=5):
    rng = np.random.default_rng(seed)
    hkls = rng.integers(-8, 9, size=(n, 3))
    keys = H.orbit_keys(hkls, MMM)
    raw = rng.gamma(2.0, 50.0, size=n)
    ang = np.sort(rng.uniform(-30, 30, size=n))
    group = np.digitize(ang, np.linspace(-30, 30, 31))
    true = raw * (1.0 + 0.3 * np.sin(ang / 10.0))
    sig = 0.05 * true + 2.0
    y = true + rng.normal(size=n) * sig
    # A reflection and its symmetry equivalents have the same d-spacing.
    d = 12.0 / np.maximum(np.linalg.norm(hkls, axis=1), 1.0)
    return y, sig, raw, ang, group, keys, d


def test_metrics_bootstrap_and_sign_test_match_the_reference():
    y, sig, raw, ang, group, keys, d = synthetic()
    shell = H.shell_index(d)
    pa = raw * 1.02
    pb = raw * (1.0 + 0.3 * np.sin(ang / 10.0))
    sel = np.ones(len(y), dtype=bool)
    got = H.paired_orbit_bootstrap(y, sig, pa, pb, keys, shell, sel, n_boot=200)
    ref = G.paired_orbit_bootstrap(y, sig, pa, pb, keys, shell, sel, n_boot=200)
    for name in ("r_all", "r_obs"):
        assert got[name]["delta"] == pytest.approx(ref[name]["delta"])
        assert got[name]["ci95"] == pytest.approx(ref[name]["ci95"])
    same = H.paired_orbit_bootstrap(y, sig, pb, pb, keys, shell, sel, n_boot=50)
    assert same["r_obs"]["delta"] == 0.0 and same["r_obs"]["ci95"] == [0.0, 0.0]
    m = H.r_metrics(y, sig, pb)
    g = G.metrics(y, sig, d, pb)
    assert m["r_all"] == pytest.approx(g["r_all"]) and m["r_obs"] == pytest.approx(g["r_obs"])
    st = H.orbit_sign_test(y, pb, pa, keys, sel)
    assert st["orbit_sign_z"] == pytest.approx(G.orbit_sign_test(y, pb, pa, keys, sel)["orbit_sign_z"])
    assert st["orbit_sign_z"] > 3.0


def test_pspline_refit_matches_the_reference_fit_scales_cv():
    y, sig, raw, ang, group, keys, _d = synthetic()
    folds = H.hash_fold(keys, G.DEV_SALT, 3)
    cfg = {"n_basis": 12, "lambda": 1.0, "objective": "row_mr"}
    ref, _ins, _info = G.fit_scales_cv(y, sig, raw, ang, group, folds, 3, cfg, clamp=True)
    got = H.cv_predictions(lambda tr: H.pspline_fit_predict(y, sig, raw, ang, group, tr, cfg)["prediction"], len(y), folds)
    assert np.allclose(got, ref, rtol=1e-10, equal_nan=True)


def test_group_scales_recover_truth_and_clamp_missing_groups():
    rng = np.random.default_rng(1)
    raw = rng.uniform(10, 100, size=200)
    group = np.repeat(np.arange(10), 20)
    s = np.linspace(0.5, 1.5, 10)
    y = s[group] * raw
    train = group != 9
    out = H.group_fit_predict(y, np.ones_like(y), raw, group, train)
    assert np.allclose(out["prediction"][train], y[train])
    assert out["n_groups_clamped"] == 1 and out["n_scales"] == 9
    assert np.allclose(out["prediction"][~train], np.median(s[:9]) * raw[~train])


def test_inner_cv_selection_returns_a_grid_member():
    y, sig, raw, ang, group, keys, _d = synthetic(n=600)
    train = H.hash_fold(keys, G.DEV_SALT, 3) != 0
    out = H.select_pspline_by_inner_cv(y, sig, raw, ang, group, train, keys,
                                       grid={"n_basis": (8, 12), "lambda": (1.0, 3.0)})
    assert out["scale"]["n_basis"] in (8, 12) and out["scale"]["lambda"] in (1.0, 3.0)
    assert len(out["grid"]) == 4 and out["inner_cv_r_all"] == min(g["inner_cv_r_all"] for g in out["grid"])


def test_bootstrap_shell_assignment_does_not_depend_on_row_order():
    # Three groups cross a shell boundary, as slightly different crystal cells can.
    keys = np.repeat(np.array(["a", "b", "c", "d", "e", "f"]), 2)
    observed = np.array([10, 12, 20, 23, 40, 44, 80, 90, 160, 170, 300, 330.])
    sigma = np.ones(12)
    prediction_a = observed * np.array([.5, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4, 1.8, .8, .6])
    prediction_b = observed * .98
    shells = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    selected = np.ones(12, dtype=bool)
    reference = H.paired_orbit_bootstrap(
        observed, sigma, prediction_a, prediction_b, keys, shells, selected, n_boot=500, seed=5,
    )
    for order in [np.arange(11, -1, -1), np.random.default_rng(7).permutation(12)]:
        reordered = H.paired_orbit_bootstrap(
            observed[order], sigma[order], prediction_a[order], prediction_b[order],
            keys[order], shells[order], selected[order], n_boot=500, seed=5,
        )
        for metric in ("r_all", "r_obs"):
            assert reordered[metric]["delta"] == pytest.approx(reference[metric]["delta"])
            assert reordered[metric]["ci95"] == pytest.approx(reference[metric]["ci95"])
