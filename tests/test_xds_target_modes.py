from types import SimpleNamespace

import numpy as np

from grond.legacy.engine.native_row_regularize import TARGET_MODES, target_arrays


def test_xds_target_modes_scale_intensity_and_sigma_consistently():
    data = SimpleNamespace(
        iobs=np.array([100.0, 200.0]),
        sigma=np.array([10.0, 20.0]),
        rlp=np.array([2.0, 4.0]),
        peak=np.array([50.0, 100.0]),
    )

    assert TARGET_MODES == ("iobs", "iobs_over_rlp", "iobs_times_peak", "recorded_proxy")

    target, sigma = target_arrays(data, "iobs")
    np.testing.assert_allclose(target, [100.0, 200.0])
    np.testing.assert_allclose(sigma, [10.0, 20.0])

    target, sigma = target_arrays(data, "iobs_over_rlp")
    np.testing.assert_allclose(target, [50.0, 50.0])
    np.testing.assert_allclose(sigma, [5.0, 5.0])

    target, sigma = target_arrays(data, "iobs_times_peak")
    np.testing.assert_allclose(target, [50.0, 200.0])
    np.testing.assert_allclose(sigma, [5.0, 20.0])

    target, sigma = target_arrays(data, "recorded_proxy")
    np.testing.assert_allclose(target, [25.0, 50.0])
    np.testing.assert_allclose(sigma, [2.5, 5.0])
