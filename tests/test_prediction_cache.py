from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import grond.legacy.engine.native_row_regularize as native_row_regularize
from grond.legacy.engine.bloch_engine import Config
from grond.legacy.engine.native_row_regularize import (
    PREDICTION_CACHE_SCHEMA,
    cacheable_prediction_arrays,
    predict_with_cache,
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
        "basis_support_hash64": np.array([11, 12, 13], dtype=np.uint64),
        "sg_profile_same_branch_continuation_used": np.array([False, True, False]),
        "sg_profile_same_branch_continuation_segments": np.array([1, 2, 1], dtype=np.int32),
    }

    save = cacheable_prediction_arrays(pred)

    assert "row_plan_cache_status" not in save
    assert "support_hkls" not in save
    assert set(save) == {
        "indices",
        "dyn",
        "born",
        "seconds",
        "row_plan_cache_unique_diffs",
        "sg_profile_same_branch_continuation_used",
        "sg_profile_same_branch_continuation_segments",
        "basis_support_hash64",
    }

    path = tmp_path / "pred.npz"
    write_prediction_cache(path, save, {"cache_schema": 2}, "none")
    loaded = read_prediction_cache(path)

    np.testing.assert_array_equal(loaded["indices"], pred["indices"])
    np.testing.assert_allclose(loaded["dyn"], pred["dyn"])
    np.testing.assert_allclose(loaded["born"], pred["born"])
    np.testing.assert_array_equal(
        loaded["sg_profile_same_branch_continuation_used"],
        pred["sg_profile_same_branch_continuation_used"],
    )
    np.testing.assert_array_equal(
        loaded["sg_profile_same_branch_continuation_segments"],
        pred["sg_profile_same_branch_continuation_segments"],
    )
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


def test_prediction_cache_round_trip_hits_with_default_nan_config(tmp_path, monkeypatch):
    hkl = tmp_path / "INTEGRATE.HKL"
    res = tmp_path / "model.res"
    hkl.write_text("rows\n")
    res.write_text("model\n")
    data = SimpleNamespace(hkl_transform_spec="identity", a_scan_points_path=None)
    indices = np.asarray([1, 2, 3], dtype=np.int64)
    calls = []

    class FakePort:
        def predict_rows(self, requested, workers, chunksize):
            calls.append((requested.copy(), workers, chunksize))
            return {
                "indices": requested.copy(),
                "dyn": np.asarray([1.0, 2.0, 3.0]),
                "born": np.asarray([0.9, 1.8, 2.7]),
                "basis_support_hash64": np.asarray([11, 12, 13], dtype=np.uint64),
                "sg_profile_same_branch_continuation_used": np.asarray([False, True, False]),
                "sg_profile_same_branch_continuation_segments": np.asarray([1, 2, 1]),
            }

    monkeypatch.setattr(
        native_row_regularize,
        "XDSBlochPort",
        lambda *args, **kwargs: FakePort(),
    )
    cache = tmp_path / "cache"

    first = predict_with_cache(
        data, object(), "model-hash", hkl, res, Config(), indices, cache, 2, 4, cache_compression="none"
    )
    second = predict_with_cache(
        data, object(), "model-hash", hkl, res, Config(), indices, cache, 2, 4, cache_compression="none"
    )

    assert first["cache_status"] == "built"
    assert second["cache_status"] == "hit"
    assert first["cache_path"] == second["cache_path"]
    assert len(calls) == 1
    np.testing.assert_array_equal(second["indices"], indices)
    np.testing.assert_array_equal(second["dyn"], first["dyn"])
    np.testing.assert_array_equal(second["born"], first["born"])


def test_prediction_cache_rejects_source_res_and_config_mismatches(tmp_path, monkeypatch):
    hkl = tmp_path / "INTEGRATE.HKL"
    res = tmp_path / "model.res"
    hkl.write_text("rows\n")
    res.write_text("model one\n")
    data = SimpleNamespace(hkl_transform_spec="identity", a_scan_points_path=None)
    indices = np.asarray([1, 2], dtype=np.int64)
    base_key, base_payload = prediction_cache_key(data, hkl, res, Config(), indices)
    assert base_payload["cache_schema"] == PREDICTION_CACHE_SCHEMA == 6
    cached = {
        "prediction_cache_meta": base_payload,
        "indices": indices,
        "dyn": np.asarray([1.0, 2.0]),
        "born": np.asarray([0.9, 1.8]),
        "basis_support_hash64": np.asarray([11, 12], dtype=np.uint64),
        "sg_profile_same_branch_continuation_used": np.asarray([False, True]),
        "sg_profile_same_branch_continuation_segments": np.asarray([1, 2]),
    }
    incomplete_provenance = dict(cached)
    incomplete_provenance.pop("sg_profile_same_branch_continuation_segments")
    assert (
        "profile continuation segment provenance is incomplete"
        in prediction_cache_validation_errors(incomplete_provenance, base_payload, indices)
    )
    incomplete_basis = dict(cached)
    incomplete_basis.pop("basis_support_hash64")
    assert (
        "basis-support hash provenance is incomplete"
        in prediction_cache_validation_errors(incomplete_basis, base_payload, indices)
    )

    res.write_text("model two\n")
    res_key, res_payload = prediction_cache_key(data, hkl, res, Config(), indices)
    assert res_key != base_key
    assert any("metadata" in error for error in prediction_cache_validation_errors(cached, res_payload, indices))

    config_key, config_payload = prediction_cache_key(data, hkl, res, Config(n_basis=21), indices)
    assert config_key != res_key
    assert any("metadata" in error for error in prediction_cache_validation_errors(cached, config_payload, indices))

    monkeypatch.setattr(native_row_regularize, "PREDICTION_ENGINE_SOURCE_SHA256", "changed-source")
    source_key, source_payload = prediction_cache_key(data, hkl, res, Config(), indices)
    assert source_key not in {base_key, res_key, config_key}
    assert any("metadata" in error for error in prediction_cache_validation_errors(cached, source_payload, indices))
