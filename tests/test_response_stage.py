"""Streamed thickness response tables equal the tables built from packed spectra, and select like them."""

import json

import numpy as np
import pytest

from grond.bloch.config import Config
from grond.legacy.engine import response_stage as RS, spectra_stage as S
from grond.regularize import nested_selection as NS
from tests._nested_support import random_spectra, synthetic_rows, write_dataset

SUPPORT = NS.ThicknessSupport(100.0, 3000.0, 2.0)
CONFIG = Config(n_basis=8, n_points=3, topup_d_min=2.0, ug_d_min=2.0)
N_ROWS = 150
GRIDS = NS.SelectionGrids(gamma_mean=(400.0, 800.0, 1200.0), gamma_shape=(1.0, 2.0, 4.0), n_basis=(8,), lam=(1.0, 10.0))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("response"), max_rows=12)


def _eager(hkl, res, rows, hands):
    spec, _ = S.compute_spectra(hkl, res, CONFIG, rows)
    return spec, NS.thickness_table(spec, SUPPORT, hands=hands)


@pytest.mark.parametrize("workers", [1, 2])
def test_streamed_table_equals_the_table_of_packed_spectra(dataset, tmp_path, workers):
    hkl, res = dataset
    rows = np.asarray([5, 0, 9, 3, 11])
    spec, expected = _eager(hkl, res, rows, (1, -1))
    table, info = RS.compute_response_table(hkl, res, CONFIG, rows, SUPPORT, hands=(1, -1), workers=workers,
                                            cache_dir=tmp_path)
    assert info["cache_status"] == "built" and info["n_rows"] == len(rows)
    for key in (1, -1, "born"):
        assert np.array_equal(table[key], expected[key])
    assert np.array_equal(table["t"], expected["t"]) and table["support"] == expected["support"]
    assert np.array_equal(table["row_ids"], rows) and np.array_equal(table["mu"], spec["mu"])
    assert np.array_equal(table["n_p"], spec["n_p"]) and np.array_equal(table["n_b"], spec["n_b"])
    assert not isinstance(table[1], np.memmap) and not table[1].flags.writeable


def test_cache_hit_and_rebuild_of_a_foreign_table(dataset, tmp_path):
    hkl, res = dataset
    rows = np.asarray([2, 7])
    first, info = RS.compute_response_table(hkl, res, CONFIG, rows, SUPPORT, cache_dir=tmp_path)
    again, hit = RS.compute_response_table(hkl, res, CONFIG, rows, SUPPORT, cache_dir=tmp_path)
    assert hit["cache_status"] == "hit" and np.array_equal(again[1], first[1])
    identity = tmp_path / f"response_{info['key']}" / "identity.json"
    stored = json.loads(identity.read_text())
    identity.write_text(json.dumps({**stored, "n_rows": 99}, sort_keys=True))
    rebuilt, status = RS.compute_response_table(hkl, res, CONFIG, rows, SUPPORT, cache_dir=tmp_path)
    assert status["cache_status"] == "rebuilt" and "identity differs" in status["cache_rebuild_reason"]
    assert np.array_equal(rebuilt[1], first[1])
    other_support = NS.ThicknessSupport(100.0, 2000.0, 2.0)
    _, other = RS.compute_response_table(hkl, res, CONFIG, rows, other_support, cache_dir=tmp_path)
    assert other["key"] != info["key"]


def test_table_without_cache_is_in_memory(dataset):
    hkl, res = dataset
    table, info = RS.compute_response_table(hkl, res, CONFIG, [4], SUPPORT)
    assert info["cache_status"] == "disabled" and info["cache"] is None
    assert table[1].shape == (1, len(SUPPORT.grid()))


def test_bad_requests_fail_closed(dataset):
    hkl, res = dataset
    with pytest.raises(ValueError, match="unique"):
        RS.compute_response_table(hkl, res, CONFIG, [1, 1], SUPPORT)
    with pytest.raises(ValueError, match="hands"):
        RS.compute_response_table(hkl, res, CONFIG, [1], SUPPORT, hands=(2,))


# ----------------------------------------------------------------------------- selection on tables


def _problem():
    specs = [random_spectra(N_ROWS, seed=1), random_spectra(N_ROWS, seed=1, shift=2e-4)]
    keys, ang = synthetic_rows(N_ROWS, seed=5)
    truth = {"family": "gamma", "mean": 800.0, "shape": 2.0}
    dyn = NS.row_intensities(specs[0], truth, evaluator="table", support=SUPPORT)["dyn"]
    rng = np.random.default_rng(5)
    y = dyn * 1000.0 * (1.0 + 0.2 * np.sin(ang / 20.0)) * rng.normal(1.0, 0.01, N_ROWS)
    return specs, keys, ang, y, 0.05 * y + 1.0


def _as_table(spec, hands=(1, -1)):
    return {**NS.thickness_table(spec, SUPPORT, hands=hands), "row_ids": spec["row_ids"], "mu": spec["mu"]}


def test_selection_on_tables_equals_selection_on_spectra():
    specs, keys, ang, y, sig = _problem()
    work = np.ones(N_ROWS, bool)
    kw = dict(grids=GRIDS, support=SUPPORT, evaluator="table", hands=(1, -1))
    on_spectra = NS.select_nested([NS.Variant("a", specs[0]), NS.Variant("b", specs[1])], y, sig, ang, keys, work, **kw)
    tables = [NS.Variant("a", None, table=_as_table(specs[0])), NS.Variant("b", None, table=lambda: _as_table(specs[1]))]
    on_tables = NS.select_nested(tables, y, sig, ang, keys, work, **kw)
    for h in (1, -1):
        assert np.array_equal(on_tables[h].scores, on_spectra[h].scores)
        assert on_tables[h].pick == on_spectra[h].pick
        assert np.array_equal(on_tables[h].dyn, on_spectra[h].dyn)
        assert np.array_equal(on_tables[h].prediction, on_spectra[h].prediction)


@pytest.mark.parametrize("change,match", [
    (lambda t: {"evaluator": "auto"}, "need evaluator='table'"),
    (lambda t: {"score_with": "resolved"}, "need evaluator='table'"),
    (lambda t: {"variants": [NS.Variant("a", t["spec"], table=t["table"])]}, "not both"),
    (lambda t: {"variants": [NS.Variant("a", None, table={**t["table"], "support": {"t_min_A": 0.0}})]}, "support"),
    (lambda t: {"variants": [NS.Variant("a", None, table={k: v for k, v in t["table"].items() if k != -1})]},
     "no hand"),
    (lambda t: {"variants": [NS.Variant("a", None, table={**t["table"], "born": -t["table"]["born"]})]}, "negative"),
    (lambda t: {"variants": [NS.Variant("a", None, table={**t["table"], "row_ids": t["table"]["row_ids"][:-1]})]},
     "mu has"),
])
def test_table_selection_fails_closed(change, match):
    specs, keys, ang, y, sig = _problem()
    t = {"spec": specs[0], "table": _as_table(specs[0])}
    kw = {"variants": [NS.Variant("a", None, table=t["table"])], "grids": GRIDS, "support": SUPPORT,
          "evaluator": "table", "hands": (1, -1), **change(t)}
    variants = kw.pop("variants")
    with pytest.raises(ValueError, match=match):
        NS.select_nested(variants, y, sig, ang, keys, np.ones(N_ROWS, bool), **kw)
    with pytest.raises(ValueError, match="no spectra"):
        NS.Variant("a", None, table=t["table"]).load()


# ----------------------------------------------------------------------------- nested stage request


def _stage_request(tmp_path, hkl, res, n_rows, **extra):
    rng = np.random.default_rng(11)
    rows_npz = tmp_path / "rows.npz"
    np.savez_compressed(rows_npz, row_index=np.arange(n_rows), orbit_key=np.asarray([f"o{i}" for i in range(n_rows)]),
                        target=rng.uniform(50.0, 500.0, n_rows), sigma=np.full(n_rows, 5.0),
                        angle_deg=np.linspace(-40.0, 40.0, n_rows), work=np.ones(n_rows, bool),
                        free=np.zeros(n_rows, bool), valid=np.ones(n_rows, bool),
                        nuisance_json=np.asarray(["{}"]))
    grids = NS.SelectionGrids(families=("gamma",), gamma_mean=(400.0, 800.0), gamma_shape=(1.0, 4.0), n_basis=(8,),
                              lam=(1.0,))
    return {"schema": "grond.nested-nuisance-request.v1", "mode": "select", "hkl": str(hkl), "res": str(res),
            "rows_npz": str(rows_npz), "base_config": {"n_basis": 8, "n_points": 3, "topup_d_min": 2.0, "ug_d_min": 2.0},
            "variants": [{"label": "p3", "config": {}}, {"label": "p5", "config": {"n_points": 5}}],
            "grids": grids.to_dict(), "support": SUPPORT.to_dict(), "evaluator": "table", "score_with": "table",
            "workers": 1, **extra}


def test_stage_on_response_tables_equals_stage_on_spectra(tmp_path):
    from grond.legacy.engine import nested_nuisance_stage as stage

    hkl, res = write_dataset(tmp_path / "data", max_rows=60)
    n_rows = len([line for line in hkl.read_text().splitlines() if not line.startswith("!")])
    assert n_rows >= stage.MIN_SELECTION_ROWS
    on_spectra = stage.run_selection(_stage_request(tmp_path, hkl, res, n_rows), scratch_dir=tmp_path)
    on_tables = stage.run_selection(_stage_request(tmp_path, hkl, res, n_rows, response_tables=True), scratch_dir=tmp_path)
    assert on_tables["pick"] == on_spectra["pick"]
    assert on_tables["cv_r_all"] == on_spectra["cv_r_all"]
    assert [c["kind"] for c in on_tables["cost"]["spectra"]] == ["response_table", "response_table"]
    with pytest.raises(ValueError, match="needs evaluator 'table'"):
        stage.run_selection(_stage_request(tmp_path, hkl, res, n_rows, response_tables=True, evaluator="auto"))
