from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from grond.legacy.engine.bloch_engine import Config
from grond.legacy.engine.native_row_regularize import (
    cacheable_prediction_arrays,
    prediction_cache_key,
    prediction_cache_validation_errors,
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


def test_prediction_cache_validation_rejects_stale_or_nonfinite_content():
    payload = {"cache_schema": 3, "hkl_sha256": "expected"}
    pred = {
        "prediction_cache_meta": {"cache_schema": 2, "hkl_sha256": "old"},
        "indices": np.asarray([1, 2]),
        "dyn": np.asarray([1.0, np.nan]),
        "born": np.asarray([0.9, 1.1]),
    }

    errors = prediction_cache_validation_errors(pred, payload, np.asarray([1, 2]))

    assert any("metadata" in error for error in errors)
    assert any("dyn" in error for error in errors)


def test_prediction_cache_key_tracks_scan_matrix_sidecar_content(tmp_path):
    hkl = tmp_path / "INTEGRATE.HKL"
    model = tmp_path / "model.res"
    sidecar = tmp_path / "A_SCAN.txt"
    hkl.write_text("rows\n")
    model.write_text("model\n")
    sidecar.write_text("1 0 0 0 1 0 0 0 1\n")
    data = SimpleNamespace(hkl_transform_spec="identity", a_scan_points_path=sidecar)

    first, _ = prediction_cache_key(data, hkl, model, Config(), np.asarray([0, 1]))
    sidecar.write_text("2 0 0 0 2 0 0 0 2\n")
    second, _ = prediction_cache_key(data, hkl, model, Config(), np.asarray([0, 1]))

    assert first != second
