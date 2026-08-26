import math

import numpy as np
import pytest

from grond.legacy.engine.native_structure_update import (
    NATIVE_UPDATE_TRACE_FIELDS,
    absolute_finite_difference_jacobian,
    atom_distance_A,
    frac_delta_min_image,
    parameter_absolute_steps,
    write_native_update_trace,
)


def test_closest_periodic_image_uses_skew_cell_metric():
    model = {
        "cell": (10.0, 10.0, 12.0, 90.0, 90.0, 20.0),
        "atoms": [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 0.49, "y": 0.49, "z": 0.0},
        ],
    }

    delta = frac_delta_min_image(model["atoms"][0], model["atoms"][1], model["cell"])

    assert {round(float(delta[0]), 2), round(float(delta[1]), 2)} == {-0.51, 0.49}
    assert delta[2] == 0.0
    expected = math.sqrt((-5.1) ** 2 + 4.9**2 + 2.0 * (-5.1) * 4.9 * math.cos(math.radians(20.0)))
    assert math.isclose(atom_distance_A(model, 0, 1), expected, rel_tol=1e-12)


def test_closest_periodic_image_is_exact_for_nonreduced_skew_cell():
    cell = (
        0.3975793,
        0.7004456,
        0.1139303,
        23.1224,
        120.3711,
        116.1997,
    )
    first = {"x": 0.0, "y": 0.0, "z": 0.0}
    second = {"x": -0.0141646, "y": 0.3894878, "z": 0.4340435}

    delta = frac_delta_min_image(first, second, cell)

    assert delta == pytest.approx((-0.0141646, 0.3894878, -2.5659565))
    model = {"cell": cell, "atoms": [first, second]}
    assert atom_distance_A(model, 0, 1) == pytest.approx(0.11365967283862, rel=1e-12)


def test_native_finite_difference_steps_are_group_aware_and_absolute():
    params = [
        {"kind": "xyz"},
        {"kind": "xyz"},
        {"kind": "adp_ueq"},
    ]

    steps = parameter_absolute_steps(
        params,
        xyz_step_frac=2e-5,
        adp_step_a2=7e-5,
    )
    np.testing.assert_array_equal(steps, [2e-5, 2e-5, 7e-5])
    np.testing.assert_array_equal(
        parameter_absolute_steps(params, legacy_step=3e-5),
        [3e-5, 3e-5, 3e-5],
    )

    with pytest.raises(ValueError, match="finite-difference step"):
        parameter_absolute_steps(params, xyz_step_frac=0.0)
    with pytest.raises(ValueError, match="unknown parameter kind"):
        parameter_absolute_steps([{"kind": "occupancy"}])


def test_absolute_finite_difference_jacobian_matches_analytic_and_converges():
    x = np.asarray([0.0, 0.02], dtype=float)
    lower = np.asarray([-1.0, 0.001], dtype=float)
    upper = np.asarray([1.0, 0.2], dtype=float)

    def residuals(values):
        return np.asarray(
            [
                np.sin(2.0 * values[0]) + values[1] ** 2,
                np.exp(values[0] - values[1]),
                values[0] * values[1],
            ]
        )

    exact = np.asarray(
        [
            [2.0, 0.04],
            [np.exp(-0.02), -np.exp(-0.02)],
            [0.02, 0.0],
        ]
    )
    coarse = absolute_finite_difference_jacobian(
        residuals,
        x,
        (lower, upper),
        np.asarray([1e-4, 1e-4]),
    )
    fine = absolute_finite_difference_jacobian(
        residuals,
        x,
        (lower, upper),
        np.asarray([5e-5, 5e-5]),
    )
    coarse_error = float(np.max(np.abs(coarse - exact)))
    fine_error = float(np.max(np.abs(fine - exact)))

    assert coarse_error < 1.1e-4
    assert fine_error < 0.55 * coarse_error


def test_absolute_finite_difference_jacobian_stays_within_bounds():
    x = np.asarray([0.99995, 0.00105], dtype=float)
    lower = np.asarray([-1.0, 0.001], dtype=float)
    upper = np.asarray([1.0, 0.2], dtype=float)
    calls = []

    def residuals(values):
        calls.append(np.asarray(values, dtype=float).copy())
        return np.asarray([values[0] ** 2 + 3.0 * values[1]])

    jacobian = absolute_finite_difference_jacobian(
        residuals,
        x,
        (lower, upper),
        np.asarray([1e-4, 1e-4]),
    )

    np.testing.assert_allclose(calls[1] - x, [-1e-4, 0.0], atol=1e-15)
    np.testing.assert_allclose(calls[2] - x, [0.0, 1e-4], atol=1e-15)
    np.testing.assert_allclose(jacobian, [[1.9998, 3.0]], atol=1e-12)


def test_native_update_trace_preserves_pets_cache_provenance(tmp_path):
    row = {field: 0 for field in NATIVE_UPDATE_TRACE_FIELDS}
    row["pets_objective_cache_only"] = True
    path = tmp_path / "trace.csv"

    write_native_update_trace(path, [row])

    header, values = path.read_text().splitlines()
    assert header.split(",")[-1] == "pets_objective_cache_only"
    assert values.split(",")[-1] == "True"


def test_native_update_trace_rejects_unregistered_fields(tmp_path):
    row = {field: 0 for field in NATIVE_UPDATE_TRACE_FIELDS}
    row["new_provenance"] = "must not be silently dropped"

    with pytest.raises(ValueError, match="new_provenance"):
        write_native_update_trace(tmp_path / "trace.csv", [row])
