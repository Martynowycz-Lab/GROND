import math

import pytest

from grond.legacy.engine.native_structure_update import atom_distance_A, frac_delta_min_image


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
