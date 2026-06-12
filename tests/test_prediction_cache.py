from __future__ import annotations

import numpy as np

from grond.legacy.engine.native_row_regularize import (
    cacheable_prediction_arrays,
    read_prediction_cache,
    write_prediction_cache,
)


def test_prediction_cache_skips_live_object_diagnostics(tmp_path):
    pred = {
        "indices": np.array([1, 2, 3], dtype=np.int64),
        "dyn": np.array([1.0, 2.0, 3.0], dtype=float),
        "born": np.array([0.9, 1.8, 2.7], dtype=float),
        "seconds": 1.25,
        "row_plan_cache_unique_diffs": 17,
        "row_plan_cache_status": {"status": "hit"},
        "support_hkls": [[0, 0, 0], [1, 0, 0]],
    }

    save = cacheable_prediction_arrays(pred)

    assert "row_plan_cache_status" not in save
    assert "support_hkls" not in save
    assert set(save) == {"indices", "dyn", "born", "seconds", "row_plan_cache_unique_diffs"}

    path = tmp_path / "pred.npz"
    write_prediction_cache(path, save, {"cache_schema": 2}, "none")
    loaded = read_prediction_cache(path)

    np.testing.assert_array_equal(loaded["indices"], pred["indices"])
    np.testing.assert_allclose(loaded["dyn"], pred["dyn"])
    np.testing.assert_allclose(loaded["born"], pred["born"])
    assert loaded["seconds"] == 1.25
    assert loaded["row_plan_cache_unique_diffs"] == 17


def test_prediction_cache_reader_tolerates_legacy_object_arrays(tmp_path):
    path = tmp_path / "legacy_pred.npz"
    np.savez(
        path,
        indices=np.array([4, 5], dtype=np.int64),
        dyn=np.array([1.0, 1.2], dtype=float),
        row_plan_cache_status=np.array({"status": "hit"}, dtype=object),
        meta_json='{"cache_schema": 2}',
    )

    loaded = read_prediction_cache(path)

    np.testing.assert_array_equal(loaded["indices"], np.array([4, 5], dtype=np.int64))
    np.testing.assert_allclose(loaded["dyn"], np.array([1.0, 1.2], dtype=float))
    assert loaded["prediction_cache_skipped_keys"] == ["row_plan_cache_status"]
