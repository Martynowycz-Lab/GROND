from types import SimpleNamespace

import numpy as np
import pytest

from grond.engine.bloch_engine import XDSBlochPort


@pytest.mark.parametrize("selector", ["center", "cone", "profile", "pets"])
def test_beam_cap_breaks_equal_scores_by_pool_order(selector):
    port = object.__new__(XDSBlochPort)
    port.cfg = SimpleNamespace(n_basis=5, max_support_sg=1.0, profile_basis_mode="common_profile",
                               mosaicity_mode="gaussian_cone" if selector == "cone" else "none")
    port.pool_hkl = np.column_stack([np.arange(1, 21), np.zeros((20, 2), dtype=int)])
    port.pool_ug = np.ones(20, dtype=complex)
    port.pool_inv_d2 = np.ones(20)
    port.sg_for_hkls = lambda *_args: np.zeros(20)
    port.sg_for_hkls_at_orientation = lambda *_args: np.zeros(20)
    port.pets_exact_sg_for_hkls = lambda *_args: np.zeros(20)
    port.dose_attenuation_for_inv_d2 = lambda *_args: np.ones(20)
    port.cone_support_orientations = lambda *_args: (np.array([0.0]), np.zeros((1, 2)), {})
    target = np.array([100, 0, 0])

    if selector == "center":
        basis = port.select_basis(target, 0.0)
    elif selector == "cone":
        basis, _ = port.select_basis_common_cone(target, 0.0, 0.01)
    elif selector == "profile":
        basis, _ = port.select_basis_for_profile(target, 0.0, 0.01, np.array([0.0]), np.zeros((1, 2)))
    else:
        basis, _ = port.pets_exact_select_common_basis(target, np.eye(3)[None, :, :], 0)

    np.testing.assert_array_equal(basis, [[0, 0, 0], [100, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
