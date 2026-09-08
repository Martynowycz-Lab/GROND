"""Physical rejection is final; only execution-context failures may be retried."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from grond.bloch import prediction
from grond.bloch.errors import MissingObservedCentroidError, NonPassiveAbsorptionError
from grond.legacy.engine.bloch_engine import XDSBlochPort


class ControlledPort(prediction.PredictionMixin):
    def __init__(self, cached, reject=False):
        self.cfg = SimpleNamespace(compute_backend="scipy")
        self.cached = cached
        self.reject = reject
        self.calls = []

    def row_plan_cache_mode(self):
        return "on" if self.cached else "off"

    def load_or_build_row_plan_cache(self, indices):
        return {"indices": indices.copy(), "unique_diffs": np.array([[0, 0, 0]]), "status": {}}

    def current_unique_ug(self, differences):
        return np.ones(len(differences))

    def absorption_enabled(self):
        return False

    def row_prediction(self, index):
        self.calls.append(index)
        if self.reject:
            raise NonPassiveAbsorptionError("negative loss eigenvalue", details={"row": index, "minimum": -0.1})
        matrix = np.array([[0.0, 0.003 + 0.002j], [0.003 - 0.002j, 0.0002 * index]])
        value = XDSBlochPort.propagate_direct_to_target(self, matrix, 137.0)
        return {"dyn": value, "reference": abs(expm(2j * np.pi * 137.0 * matrix)[1, 0]) ** 2}

    def row_prediction_from_cache(self, cache, position, unique_ug):
        index = int(cache["indices"][position])
        return index, self.row_prediction(index)

    def row_prediction_from_active_cache(self, position):
        return self.row_prediction_from_cache(self._active_row_plan_cache, position, self._active_unique_ug)

    def pack_prediction_results(self, indices, results):
        return {"indices": indices.copy(), "dyn": np.array([r["dyn"] for r in results]),
                "reference": np.array([r["reference"] for r in results])}


def install_controlled_pool(monkeypatch):
    contexts = []

    class Pool:
        def __init__(self, *, processes, initializer, initargs):
            assert processes == 96
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def imap_unordered(self, function, indices, chunksize):
            # Out-of-order success also exercises the transaction's row identity.
            return (function(index) for index in reversed(indices))

    def context(name):
        contexts.append(name)
        return SimpleNamespace(Pool=Pool)

    monkeypatch.setattr(prediction.multiprocessing, "get_context", context)
    return contexts


@pytest.mark.parametrize("cached", [False, True])
def test_nonpassive_candidate_is_not_retried_or_run_serially(monkeypatch, cached):
    contexts = install_controlled_pool(monkeypatch)
    port = ControlledPort(cached, reject=True)
    with pytest.raises(NonPassiveAbsorptionError, match="negative loss eigenvalue") as caught:
        port.predict_rows(np.array([3, 1, 7]), workers=96, chunksize=1)
    assert len(contexts) == 1
    assert len(port.calls) == 1
    assert caught.value.details == {"row": 7, "minimum": -0.1}
    if cached:
        assert port._active_row_plan_cache is None
        assert port._active_unique_ug is None


@pytest.mark.parametrize("cached", [False, True])
def test_accepted_propagation_matches_serial_and_matrix_exponential(monkeypatch, cached):
    contexts = install_controlled_pool(monkeypatch)
    indices = np.array([3, 1, 7])
    serial = ControlledPort(cached).predict_rows(indices, workers=1)
    parallel = ControlledPort(cached).predict_rows(indices, workers=96, chunksize=1)
    assert len(contexts) == 1
    np.testing.assert_array_equal(parallel["indices"], indices)
    np.testing.assert_array_equal(parallel["dyn"], serial["dyn"])
    np.testing.assert_allclose(parallel["dyn"], parallel["reference"], atol=2e-15, rtol=2e-14)


@pytest.mark.parametrize("cached", [False, True])
def test_missing_centroid_error_is_not_retried(monkeypatch, cached):
    contexts = install_controlled_pool(monkeypatch)
    port = ControlledPort(cached)
    calls = []

    def reject(index):
        calls.append(index)
        raise MissingObservedCentroidError("row 7 has no observed centroid")

    port.row_prediction = reject
    with pytest.raises(MissingObservedCentroidError, match="row 7"):
        port.predict_rows(np.array([3, 1, 7]), workers=96, chunksize=1)
    assert len(contexts) == len(calls) == 1
