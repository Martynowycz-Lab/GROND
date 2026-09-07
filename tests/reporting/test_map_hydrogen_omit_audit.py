from __future__ import annotations

import gemmi
import numpy as np
import pytest

from grond.legacy.engine.map_hydrogen_omit_audit import grid_points_near, nearest_grid_value


def test_nearest_grid_value_rounds_to_the_nearest_ccp4_sample():
    arr = np.zeros((8, 8, 8), dtype=float)
    arr[2, 3, 4] = 7.0

    assert nearest_grid_value(arr, [0.22, 0.38, 0.49]) == 7.0


def test_grid_points_near_uses_ijk_over_shape_coordinates():
    arr = np.zeros((8, 8, 8), dtype=float)
    cell = gemmi.UnitCell(8.0, 8.0, 8.0, 90.0, 90.0, 90.0)
    target = np.asarray([2 / 8, 3 / 8, 4 / 8], dtype=float)

    points = list(grid_points_near(arr, cell, target, radius_A=0.01))

    assert len(points) == 1
    i, j, k, grid_frac, distance = points[0]
    assert (i, j, k) == (2, 3, 4)
    assert np.asarray(grid_frac) == pytest.approx(target)
    assert distance == pytest.approx(0.0, abs=1e-12)
