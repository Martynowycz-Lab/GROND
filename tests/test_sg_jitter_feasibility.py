from types import SimpleNamespace

import numpy as np
import pytest

from grond.legacy.engine.bloch_engine import (
    Config,
    InfeasibleSgJitterProfileError,
    LocalSgCrossingError,
    XDSBlochPort,
)
from grond.legacy.engine.native_row_regularize import sg_jitter_candidate_rejection


def test_local_sg_solver_keeps_strict_bound_with_typed_unreachable_error():
    port = object.__new__(XDSBlochPort)
    port.data = SimpleNamespace(oscillation_range=1.0)
    port.sg_for_hkls = lambda _hkls, omega: np.asarray([omega - 5.0])

    with pytest.raises(LocalSgCrossingError, match="within 2 degrees"):
        port.solve_omega_for_sg((1, 0, 0), 0.0, 0.0)


def test_active_jitter_unreachable_profile_point_rejects_whole_candidate():
    port = object.__new__(XDSBlochPort)

    def unreachable(_hkl, _omega_center, target_sg):
        raise LocalSgCrossingError(f"no local Sg={target_sg:g} crossing within 2 degrees")

    port.solve_omega_for_sg = unreachable
    meta = {
        "sg_jitter_mode": "gaussian",
        "sg_jitter_sigma_inv_A": 0.0015,
    }

    with pytest.raises(InfeasibleSgJitterProfileError, match="Reject this nuisance candidate") as caught:
        port.solve_sg_profile_target(500, (0, 2, -7), -49.8103, -0.00459108, meta)

    assert "within 2 degrees" in str(caught.value)
    record = sg_jitter_candidate_rejection(
        Config(sg_jitter_mode="gaussian", sg_jitter_sigma0=0.0015),
        caught.value,
    )
    assert record["status"] == "rejected"
    assert record["reason"] == "infeasible_local_sg_jitter_profile"
    assert record["config"]["sg_jitter_sigma0"] == pytest.approx(0.0015)


def test_zero_jitter_does_not_mask_an_unreachable_base_profile():
    port = object.__new__(XDSBlochPort)
    failure = LocalSgCrossingError("base profile is outside the strict local bound")
    port.solve_omega_for_sg = lambda *_args: (_ for _ in ()).throw(failure)

    with pytest.raises(LocalSgCrossingError) as caught:
        port.solve_sg_profile_target(
            3,
            (1, 0, 0),
            12.0,
            0.004,
            {"sg_jitter_mode": "gaussian", "sg_jitter_sigma_inv_A": 0.0},
        )

    assert caught.value is failure
