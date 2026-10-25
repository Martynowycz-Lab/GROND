from types import SimpleNamespace

import numpy as np

from grond.engine.bloch_engine import XDSBlochPort


def test_beam_cap_breaks_equal_scores_by_pool_order():
    port = object.__new__(XDSBlochPort)
    port.cfg = SimpleNamespace(n_basis=5, max_support_sg=1.0, mosaicity_mode="none")
    port.pool_hkl = np.column_stack([np.arange(1, 21), np.zeros((20, 2), dtype=int)])
    port.pool_ug = np.ones(20, dtype=complex)
    port.pool_inv_d2 = np.ones(20)
    port.sg_for_hkls = lambda *_args: np.zeros(20)
    port.dose_attenuation_for_inv_d2 = lambda *_args: np.ones(20)
    target = np.array([100, 0, 0])

    basis = port.select_basis(target, 0.0)

    np.testing.assert_array_equal(basis, [[0, 0, 0], [100, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
