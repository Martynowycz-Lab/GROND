import numpy as np

from grond.legacy.engine.native_row_regularize import (
    fit_group_scales,
    solve_regularized_scales,
)


def test_solve_regularized_scales_basic():
    # Test case 1: n = 1
    res = solve_regularized_scales([2.0], [4.0], [0], 10.0)
    assert res == {0: 2.0}

    # Test case 2: n = 3, large lambda (should smooth/equalize all scales)
    # y/x ratios are 1.0, 2.0, 3.0
    diag = [1.0, 1.0, 1.0]
    x_y = [1.0, 2.0, 3.0]
    res_large = solve_regularized_scales(diag, x_y, [0, 1, 2], 1000.0)
    # The scales should be very close to the average ratio: 2.0
    assert abs(res_large[0] - 2.0) < 0.02
    assert abs(res_large[1] - 2.0) < 0.02
    assert abs(res_large[2] - 2.0) < 0.02


def test_unobserved_group_interpolation():
    # n = 3, group 1 is unobserved (diag = 0, x_y = 0)
    diag = [1.0, 0.0, 1.0]
    x_y = [1.0, 0.0, 3.0]
    
    # solve with lambda = 10.0
    res = solve_regularized_scales(diag, x_y, [0, 1, 2], 10.0)
    
    # Scale of group 1 should be exactly the average of group 0 and group 2 scales
    expected_g1 = 0.5 * (res[0] + res[2])
    assert abs(res[1] - expected_g1) < 1e-6


def test_fit_group_scales_smoothness():
    # Generate mock observations
    obs = np.array([1.0, 1.1, 2.0, 2.2, 3.0, 3.3], dtype=float)
    sig = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
    pred = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    groups = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    fit_mask = np.ones(6, dtype=bool)

    # With scale_smoothness = 0.0 (independent fitting)
    scaled_0, scales_0 = fit_group_scales(obs, sig, pred, groups, fit_mask, 0.0, 0.0, scale_smoothness=0.0)
    
    # With scale_smoothness = 100000.0 (heavily smoothed)
    scaled_100000, scales_100000 = fit_group_scales(obs, sig, pred, groups, fit_mask, 0.0, 0.0, scale_smoothness=100000.0)
    
    # Check that scales_100000 are much closer to each other than scales_0
    spread_0 = max(scales_0.values()) - min(scales_0.values())
    spread_100000 = max(scales_100000.values()) - min(scales_100000.values())
    assert spread_100000 < spread_0 * 0.01


def test_fit_group_scales_iterates_each_group_independently():
    pred = np.tile([1.0, 2.0], 3)
    obs = pred * np.repeat([1.0, 2.0, 4.0], 2)
    sig = np.ones_like(obs)
    groups = np.repeat([0, 1, 2], 2)
    fit_mask = np.ones_like(obs, dtype=bool)
    smoothness = 0.5

    scaled, scales = fit_group_scales(
        obs,
        sig,
        pred,
        groups,
        fit_mask,
        0.0,
        0.0,
        n_iter=3,
        scale_smoothness=smoothness,
    )
    expected = solve_regularized_scales(
        [5.0, 5.0, 5.0],
        [5.0, 10.0, 20.0],
        [0, 1, 2],
        smoothness,
    )

    np.testing.assert_allclose(
        [scales[group] for group in (0, 1, 2)],
        [expected[group] for group in (0, 1, 2)],
    )
    np.testing.assert_allclose(
        scaled,
        pred * np.asarray([scales[int(group)] for group in groups]),
    )
