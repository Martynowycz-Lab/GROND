"""Nested nuisance selection (grond.regularize.nested_selection) against the reference implementation on synthetic spectra."""

from __future__ import annotations

import numpy as np
import pytest

from grond.regularize import nested_selection as NS
from grond.regularize.spline_scales import PSplineScaleFitter
from grond.validation.heldout import INNER_CV_SALT, hash_fold
from tests._nested_support import G, random_spectra, synthetic_rows

N_ROWS = 150
REFERENCE_SUPPORT = NS.ThicknessSupport(0.0, 12000.0, 2.0)
SMALL = NS.SelectionGrids(gamma_mean=(400.0, 800.0, 1200.0), gamma_shape=(1.0, 2.0, 4.0), n_basis=(8,), lam=(1.0, 10.0))


def _problem(true_variant=0, true_thickness=None, noise=0.01, seed=5):
    specs = [random_spectra(N_ROWS, seed=1), random_spectra(N_ROWS, seed=1, shift=2e-4)]
    keys, ang = synthetic_rows(N_ROWS, seed=seed)
    true_thickness = true_thickness or {"family": "gamma", "mean": 800.0, "shape": 2.0}
    dyn = NS.row_intensities(specs[true_variant], true_thickness, evaluator="table", support=REFERENCE_SUPPORT)["dyn"]
    rng = np.random.default_rng(seed)
    y = dyn * 1000.0 * (1.0 + 0.2 * np.sin(ang / 20.0)) * rng.normal(1.0, noise, N_ROWS)
    sig = 0.05 * y + 1.0
    return specs, keys, ang, y, sig


def _reference_scores(specs, y, sig, ang, keys, work, grids, hands=(1, -1)):
    """Reference inner-CV R_all (reference_lib._inner_cv_r_all) of every candidate, in the reference grid order."""
    G._SEL.clear()
    G._SEL.update(y=y, sig=sig, ang=ang, group=np.round(ang).astype(int), clamp=True,
                  inner=np.asarray([G.hash_mod(G.INNER_SALT, k, 3) for k in keys]), scale={})
    out = {h: [] for h in hands}
    t = REFERENCE_SUPPORT.grid()
    for sp in specs:
        tab = G.thickness_table(sp, dt=2.0, t_max=12000.0, hands=hands)
        for fam, prm in grids.thickness_candidates():
            for mu in grids.mu:
                w = G.distribution_weights(fam, prm, t, mu=mu)
                for h in hands:
                    dyn = G.table_intensities(tab, h, w)
                    for sopt in grids.scale_candidates("row_mr"):
                        out[h].append(G._inner_cv_r_all(dyn, work, sopt))
    return {h: np.asarray(v) for h, v in out.items()}


def test_scores_and_picks_equal_the_reference_table_selection():
    specs, keys, ang, y, sig = _problem()
    work = np.ones(N_ROWS, bool)
    variants = [NS.Variant("a", specs[0]), NS.Variant("b", specs[1])]
    mine = NS.select_nested(variants, y, sig, ang, keys, work, grids=SMALL, support=REFERENCE_SUPPORT, evaluator="table",
                            hands=(1, -1))
    reference_scores = _reference_scores(specs, y, sig, ang, keys, work, SMALL)
    folds = np.asarray([G.hash_mod(G.DEV_SALT, k, 3) for k in keys])
    R = {"y": y, "sig": sig, "ang": ang, "group": np.round(ang).astype(int), "keys": keys}
    sel_cfg = {"mode": "table", "family": "gamma", "dt": 2.0, "t_max": 12000.0, "mean_grid": list(SMALL.gamma_mean),
               "shape_grid": list(SMALL.gamma_shape), "mu_grid": [0.0],
               "scale_grid": [{"n_basis": 8, "lambda": 1.0}, {"n_basis": 8, "lambda": 10.0}]}
    reference = G.select_thickness_table(specs, R, folds, 3, sel_cfg,
                                         {"n_basis": 8, "lambda": 1.0, "objective": "row_mr"},
                                         True, 1, variant_labels=["a", "b"], hands=(1, -1))
    for h in (1, -1):
        # renormalisation on the support rescales raw intensities, which the P-spline scale absorbs exactly
        assert np.allclose(mine[h].scores, reference_scores[h], rtol=1e-9, atol=0)
        g = reference[h][2]["all_dev"]
        p = mine[h].pick
        assert p["variant"] == g["variant"]
        assert {k: p["thickness"][k] for k in ("mean", "shape")} == {k: g["thickness"][k] for k in ("mean", "shape")}
        assert p["scale"]["lambda"] == g["scale"]["lambda"] and p["mu"] == g["mu"]
        assert p["inner_r_all"] == pytest.approx(g["inner_r_all"], rel=1e-9)
        assert p["inner_r_all"] == pytest.approx(p["inner_r_all_selection"], rel=1e-12)
        assert sorted(p["edge"]) == sorted(g["edge"])


def test_selection_recovers_the_generating_nuisance():
    truth = {"family": "gamma", "mean": 800.0, "shape": 2.0}
    specs, keys, ang, y, sig = _problem(true_variant=1, true_thickness=truth, noise=0.002)
    variants = [NS.Variant("a", specs[0], params={"dose_b_total_A2": 0.0}),
                NS.Variant("b", specs[1], params={"dose_b_total_A2": 1.0})]
    out = NS.select_nested(variants, y, sig, ang, keys, np.ones(N_ROWS, bool), grids=SMALL, support=REFERENCE_SUPPORT,
                           evaluator="table")
    pick = out[1].pick
    assert pick["variant"] == "b" and pick["thickness"] == truth
    assert pick["inner_r_all"] < 1.0
    assert "dose_b_total_A2" in pick["edge"]              # the top of a two-point dose grid is an edge
    assert "mean" not in pick["edge"] and "shape" not in pick["edge"]   # 800 A and 2.0 are interior
    assert pick["hand"] == 1 and pick["variant_params"] == {"dose_b_total_A2": 1.0}


def test_free_rows_never_change_the_selection():
    specs, keys, ang, y, sig = _problem()
    free = hash_fold(list(keys), "GROND-FREE-20260925", 10) == 0
    assert 0 < free.sum() < N_ROWS
    variants = [NS.Variant("a", specs[0]), NS.Variant("b", specs[1])]
    base = NS.select_nested(variants, y, sig, ang, keys, ~free, grids=SMALL, support=REFERENCE_SUPPORT, evaluator="table")[1]
    y2, sig2 = y.copy(), sig.copy()
    y2[free] *= 50.0
    sig2[free] = 1e-3
    moved = NS.select_nested(variants, y2, sig2, ang, keys, ~free, grids=SMALL, support=REFERENCE_SUPPORT, evaluator="table")[1]
    assert np.array_equal(base.scores, moved.scores)
    assert base.pick == moved.pick
    assert np.array_equal(base.oof, moved.oof, equal_nan=True) and np.all(np.isnan(base.oof[free]))
    assert np.array_equal(base.prediction, moved.prediction)
    assert base.pick["inner_cv"]["n_work"] == int((~free).sum())


def test_out_of_fold_predictions_are_inner_cv_fits_on_other_orbits():
    specs, keys, ang, y, sig = _problem()
    work = np.ones(N_ROWS, bool)
    work[::7] = False
    sel = NS.select_nested([NS.Variant("a", specs[0])], y, sig, ang, keys, work, grids=SMALL, support=REFERENCE_SUPPORT,
                           evaluator="table")[1]
    inner = hash_fold(list(keys), INNER_CV_SALT, 3)
    scale = sel.pick["scale"]
    fitter = PSplineScaleFitter(n_basis=scale["n_basis"])
    for f in range(3):
        tr, va = work & (inner != f), work & (inner == f)
        a = np.clip(ang, ang[tr].min(), ang[tr].max())
        ref = fitter.fit(y, sig, sel.dyn, a, fit_mask=tr, lambda_smooth=scale["lambda"])["prediction"]
        assert np.allclose(sel.oof[va], ref[va], rtol=1e-12)
    assert np.all(np.isnan(sel.oof[~work]))
    fo = np.sqrt(y[work])
    r = 100 * np.sum(np.abs(fo - np.sqrt(sel.oof[work]))) / fo.sum()
    assert sel.pick["inner_r_all"] == pytest.approx(r, rel=1e-12)
    a = np.clip(ang, ang[work].min(), ang[work].max())
    full = fitter.fit(y, sig, sel.dyn, a, fit_mask=work, lambda_smooth=scale["lambda"])
    assert np.allclose(sel.prediction, full["prediction"], rtol=1e-12) and sel.scale_edf == pytest.approx(full["edf"])


def test_edges_and_physical_bounds():
    truth = {"family": "wedge", "t0": 1200.0, "grad": 1.0}
    specs, keys, ang, y, sig = _problem(true_thickness=truth, noise=0.002)
    variants = [NS.Variant("d0", specs[0], params={"dose_b_total_A2": 0.0, "mosaicity_sigma_deg": 0.0}),
                NS.Variant("d1", specs[1], params={"dose_b_total_A2": 1.0, "mosaicity_sigma_deg": 0.0})]
    grids = NS.SelectionGrids(families=("wedge",), wedge_t0=(800.0, 1200.0), wedge_grad=(0.5, 1.0), n_basis=(8,),
                              lam=(1.0, 10.0))
    pick = NS.select_nested(variants, y, sig, ang, keys, np.ones(N_ROWS, bool), grids=grids, support=REFERENCE_SUPPORT,
                            evaluator="table")[1].pick
    assert pick["thickness"] == truth and pick["variant"] == "d0"
    assert "t0" in pick["edge"]                                  # top of the t0 grid
    assert "grad" not in pick["edge"] and "grad" in pick["at_physical_bound"]          # gradient 1 is a physical bound
    assert "mu" in pick["at_physical_bound"] and "dose_b_total_A2" in pick["at_physical_bound"]
    assert "mosaicity_sigma_deg" in pick["at_physical_bound"]
    assert pick["any_edge"] is True
    # a mean on the support bound is a physical bound, not a grid edge
    specs2, keys2, ang2, y2, sig2 = _problem(true_thickness={"family": "gamma", "mean": 12000.0, "shape": 4.0}, noise=0.002)
    g2 = NS.SelectionGrids(gamma_mean=(6000.0, 12000.0), gamma_shape=(2.0, 4.0, 8.0), n_basis=(8,), lam=(1.0,))
    p2 = NS.select_nested([NS.Variant("a", specs2[0])], y2, sig2, ang2, keys2, np.ones(N_ROWS, bool), grids=g2,
                          support=REFERENCE_SUPPORT, evaluator="table")[1].pick
    assert p2["thickness"]["mean"] == 12000.0 and "mean" in p2["at_physical_bound"] and "mean" not in p2["edge"]


def test_auto_scores_on_the_table_and_evaluates_the_pick_exactly():
    specs, keys, ang, y, sig = _problem(true_thickness={"family": "gamma", "mean": 3000.0, "shape": 10.0}, noise=0.002)
    support = NS.ThicknessSupport(0.0, 20000.0, 2.0)
    grids = NS.SelectionGrids(gamma_mean=(2000.0, 3000.0), gamma_shape=(0.5, 10.0), n_basis=(8,), lam=(1.0,))
    sel = NS.select_nested([NS.Variant("a", specs[0])], y, sig, ang, keys, np.ones(N_ROWS, bool), grids=grids,
                           support=support, evaluator="auto", hands=(1, -1))
    for h in (1, -1):
        pick = sel[h].pick
        if pick["thickness"] == {"family": "gamma", "mean": 3000.0, "shape": 10.0}:
            assert pick["evaluator"] == "cf" and pick["score_evaluator"] == "table"
            assert pick["mass_outside_support"] < 1e-6
            cf = NS.cf_row_intensities(specs[0], pick["thickness"], hands=(h,))
            assert np.array_equal(sel[h].dyn, cf[h]) and np.array_equal(sel[h].born, cf["born"])
            assert pick["inner_r_all"] == pytest.approx(pick["inner_r_all_selection"], rel=1e-2)
    assert sel[1].pick["thickness"] == {"family": "gamma", "mean": 3000.0, "shape": 10.0}
    # scoring with the resolved evaluator uses cf for exactly the candidates auto resolves to cf
    resolved = NS.select_nested([NS.Variant("a", specs[0])], y, sig, ang, keys, np.ones(N_ROWS, bool), grids=grids,
                                support=support, evaluator="auto", score_with="resolved")[1]
    assert resolved.pick["score_evaluator"] == "cf" and resolved.pick["evaluator"] == "cf"
    assert resolved.pick["inner_r_all"] == pytest.approx(resolved.pick["inner_r_all_selection"], rel=1e-12)
    table_only = NS.select_nested([NS.Variant("a", specs[0])], y, sig, ang, keys, np.ones(N_ROWS, bool), grids=grids,
                                  support=support, evaluator="table")[1]
    assert table_only.pick["evaluator"] == "table"
    assert np.allclose(resolved.scores, table_only.scores, rtol=2e-2)


def test_parallel_selection_equals_serial():
    specs, keys, ang, y, sig = _problem()
    variants = [NS.Variant("a", specs[0]), NS.Variant("b", specs[1])]
    kw = dict(grids=SMALL, support=REFERENCE_SUPPORT, evaluator="table", hands=(1, -1))
    serial = NS.select_nested(variants, y, sig, ang, keys, np.ones(N_ROWS, bool), workers=1, **kw)
    parallel = NS.select_nested(variants, y, sig, ang, keys, np.ones(N_ROWS, bool), workers=3, **kw)
    for h in (1, -1):
        assert np.array_equal(serial[h].scores, parallel[h].scores) and serial[h].pick == parallel[h].pick
    table1 = NS.thickness_table(specs[0], REFERENCE_SUPPORT, hands=(1, -1), workers=1)
    table3 = NS.thickness_table(specs[0], REFERENCE_SUPPORT, hands=(1, -1), workers=3)
    assert all(np.array_equal(table1[k], table3[k]) for k in (1, -1, "born"))


def test_default_grids_follow_the_protocol():
    grids = NS.SelectionGrids()
    assert grids.families == ("gamma",)
    assert grids.gamma_mean[0] == 100.0 and grids.gamma_mean[-1] == 3400.0 and len(grids.gamma_mean) == 34
    assert grids.gamma_shape[0] == 0.25 and grids.gamma_shape[-1] == 40.0
    assert grids.n_basis == (32,) and grids.lam == (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    assert grids.mu == (0.0,)
    support = NS.ThicknessSupport()
    assert (support.t_min_A, support.t_max_A, support.dt_A) == (100.0, 10000.0, 2.0)
    assert NS.ThicknessSupport.from_nm(10, 1000) == support


@pytest.mark.parametrize("make,match", [
    (lambda: {"grids": NS.SelectionGrids(mu=(0.0, 1e-4))}, "degenerate"),
    (lambda: {"grids": NS.SelectionGrids(families=("wedge",), wedge_t0=(500.0,), wedge_grad=(0.5,)), "evaluator": "cf"},
     "gamma family only"),
    (lambda: {"evaluator": "quadrature"}, "unknown thickness evaluator"),
    (lambda: {"score_with": "cf"}, "unknown score_with"),
    (lambda: {"hands": (2,)}, "hands"),
    (lambda: {"grids": NS.SelectionGrids(families=("wedge",), wedge_t0=(20000.0,), wedge_grad=(0.1,))}, "without mass"),
    (lambda: {"grids": NS.SelectionGrids(wedge_grad=(0.5, 1.5))}, r"\[0, 1\]"),
    (lambda: {"grids": NS.SelectionGrids(gamma_shape=(1.0, 1.0))}, "repeated"),
])
def test_selection_fails_closed(make, match):
    specs, keys, ang, y, sig = _problem()
    with pytest.raises(ValueError, match=match):
        kw = {"grids": SMALL, "support": NS.ThicknessSupport(), **make()}
        NS.select_nested([NS.Variant("a", specs[0])], y, sig, ang, keys, np.ones(N_ROWS, bool), **kw)


def test_inconsistent_inputs_fail_closed():
    specs, keys, ang, y, sig = _problem()
    work = np.ones(N_ROWS, bool)
    other = random_spectra(N_ROWS - 1, seed=3)
    with pytest.raises(ValueError, match="different number of rows"):
        NS.select_nested([NS.Variant("a", specs[0]), NS.Variant("b", other)], y, sig, ang, keys, work, grids=SMALL)
    shuffled = dict(specs[1], row_ids=specs[1]["row_ids"][::-1].copy())
    with pytest.raises(ValueError, match="different rows"):
        NS.select_nested([NS.Variant("a", specs[0]), NS.Variant("b", shuffled)], y, sig, ang, keys, work, grids=SMALL)
    with pytest.raises(ValueError, match="unique"):
        NS.select_nested([NS.Variant("a", specs[0]), NS.Variant("a", specs[1])], y, sig, ang, keys, work, grids=SMALL)
    inner = hash_fold(list(keys), INNER_CV_SALT, 3)
    with pytest.raises(ValueError, match="every one of the 3 folds"):
        NS.select_nested([NS.Variant("a", specs[0])], y, sig, ang, keys, inner != 2, grids=SMALL)
    with pytest.raises(ValueError, match="needs work rows"):
        NS.select_nested([NS.Variant("a", specs[0])], y, sig, ang, keys, np.zeros(N_ROWS, bool), grids=SMALL)
    broken = dict(specs[0], lam=specs[0]["lam"][:-1])
    with pytest.raises(ValueError, match="lam has"):
        NS.validate_spectra(broken)


def test_engine_config_fields_map_the_pick():
    pick = {"thickness": {"family": "gamma", "mean": 900.0, "shape": 1.5}, "mu": 0.0, "evaluator": "table",
            "support": NS.ThicknessSupport().to_dict(), "variant_config": {"dose_b_mode": "linear_frame", "dose_b_per_frame": 0.01}}
    fields = NS.engine_config_fields(pick)
    assert fields["thickness_model"] == "gamma" and fields["thickness_A"] == 900.0 and fields["thickness_gamma_shape"] == 1.5
    assert fields["thickness_evaluator"] == "table" and fields["absorption_mode"] == "none"
    assert fields["dose_b_per_frame"] == 0.01 and fields["thickness_min_A"] == 100.0
    wedge = NS.engine_config_fields({**pick, "thickness": {"family": "wedge", "t0": 1200.0, "grad": 0.6}, "mu": 1e-4})
    assert wedge["thickness_model"] == "uniform_gradient" and wedge["thickness_grad_frac"] == 0.6
    assert wedge["absorption_mode"] == "mean" and wedge["absorption_mu"] == 1e-4
