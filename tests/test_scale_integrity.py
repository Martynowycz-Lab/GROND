import numpy as np

from grond.legacy.engine.native_row_regularize import (
    fit_group_scales,
    native_metrics,
    rank_trials,
)


def test_work_fitted_scale_is_invariant_to_free_observations():
    prediction = np.full(4, 10.0)
    sigma = np.ones(4)
    groups = np.zeros(4, dtype=int)
    work = np.asarray([True, True, False, False])

    _, original = fit_group_scales(
        np.asarray([20.0, 20.0, 30.0, 30.0]),
        sigma,
        prediction,
        groups,
        work,
        0.0,
        0.0,
    )
    _, changed = fit_group_scales(
        np.asarray([20.0, 20.0, 3000.0, 0.01]),
        sigma,
        prediction,
        groups,
        work,
        0.0,
        0.0,
    )

    assert original == changed == {0: 2.0}


def test_group_without_work_rows_borrows_finite_global_scale():
    observations = np.asarray([20.0, 20.0, 50.0])
    prediction = np.asarray([10.0, 10.0, 10.0])
    groups = np.asarray([0, 0, 1])
    work = np.asarray([True, True, False])

    scaled, scales = fit_group_scales(
        observations,
        np.ones(3),
        prediction,
        groups,
        work,
        0.0,
        0.0,
    )

    assert scales == {0: 2.0, 1: 2.0}
    np.testing.assert_allclose(scaled, [20.0, 20.0, 20.0])


def test_native_metrics_excludes_nonpositive_sigmas():
    result = native_metrics(
        np.asarray([10.0, 10.0, 10.0]),
        np.asarray([1.0, 0.0, -1.0]),
        np.asarray([10.0, 10.0, 10.0]),
        np.ones(3, dtype=bool),
        0.0,
        0.0,
    )

    assert result["n"] == 1


def test_work_trial_ranking_does_not_use_free_rows_as_tie_breaker():
    def trial(label, work, free):
        return {
            "label": label,
            "metrics": {
                "work": {"dyn": {"wR2_all": work}},
                "free": {"dyn": {"wR2_all": free}},
            },
        }

    original = [trial("first", 0.2, 0.9), trial("second", 0.2, 0.1)]
    changed_free = [trial("first", 0.2, 0.01), trial("second", 0.2, 9.0)]

    assert rank_trials(original, "work_wR2")[0]["label"] == "first"
    assert rank_trials(changed_free, "work_wR2")[0]["label"] == "first"
