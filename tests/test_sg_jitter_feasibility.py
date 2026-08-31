from types import SimpleNamespace

import numpy as np
import pytest

import grond.legacy.engine.bloch_engine as bloch_engine
from grond.legacy.engine.bloch_engine import (
    Config,
    InfeasibleSgJitterProfileError,
    LocalSgCrossingError,
    XDSBlochPort,
    parse_hkl_transform,
)
from grond.legacy.engine.native_row_regularize import (
    filter_unreachable_dials_rows,
    sg_jitter_candidate_rejection,
)


def fake_prediction_result(idx, *, continuation=False):
    value = float(idx + 1)
    return {
        "dyn": value,
        "born": 0.9 * value,
        "basis_size": 20,
        "support_hkls": [(idx, 0, 0)],
        "max_sg_solve_residual": 1e-12,
        "center_sg_inv_A": 0.0,
        "omega_bragg_deg": value,
        "omega_bragg_minus_center_deg": 0.0,
        "sg_window_inv_A": 0.004,
        "sg_dSg_domega_inv_A_per_deg": 0.001,
        "sg_dSg_dframe_inv_A": 0.0005,
        "sg_window_raw_inv_A": 0.004,
        "sg_window_half_frames": 1.0,
        "sg_window_clipped": False,
        "sg_jitter_sigma_inv_A": 0.0,
        "sg_jitter_d_inv_A": float("nan"),
        "sg_jitter_frame_fraction": float("nan"),
        "sg_jitter_kernel_points": 1,
        "sg_profile_points_total": 5,
        "sg_profile_weight_sum": 1.0,
        "sg_profile_same_branch_continuation_used": continuation,
        "sg_profile_same_branch_continuation_segments": 2 if continuation else 1,
    }


def test_local_sg_solver_keeps_strict_bound_with_typed_unreachable_error():
    port = object.__new__(XDSBlochPort)
    port.data = SimpleNamespace(oscillation_range=1.0)
    port.sg_for_hkls = lambda _hkls, omega: np.asarray([omega - 5.0])

    with pytest.raises(LocalSgCrossingError, match="within 2 degrees"):
        port.solve_omega_for_sg((1, 0, 0), 0.0, 0.0)


def test_dials_unreachable_row_is_filtered_and_ledgered(monkeypatch):
    data = SimpleNamespace(
        header={"GROND_SOURCE": "DIALS"},
        hkl=np.asarray([[1, 0, 0], [2, 0, 0]], dtype=int),
        zcal=np.asarray([10.0, 20.0]),
        zobs=np.asarray([10.1, 24.5]),
        omega_from_z=lambda value: float(value),
    )

    def fake_root(_data, hkl, omega):
        if int(hkl[0]) == 2:
            raise ValueError("no local Sg=0 crossing near XDS ZCAL")
        return float(omega) + 0.25

    monkeypatch.setattr(
        "grond.legacy.engine.native_row_regularize._xds_header_bragg_omega",
        fake_root,
    )

    retained, audit = filter_unreachable_dials_rows(data, np.asarray([0, 1]))

    np.testing.assert_array_equal(retained, [0])
    assert audit["status"] == "filtered"
    assert audit["input_rows"] == 2
    assert audit["retained_rows"] == 1
    assert audit["excluded_rows"] == 1
    assert audit["excluded"][0]["row_index"] == 1
    assert audit["excluded"][0]["hkl"] == [2, 0, 0]


def test_local_sg_filter_does_not_change_xds_membership():
    data = SimpleNamespace(header={"GROND_SOURCE": "XDS"})

    retained, audit = filter_unreachable_dials_rows(data, np.asarray([3, 7]))

    np.testing.assert_array_equal(retained, [3, 7])
    assert audit["status"] == "not_applicable"
    assert audit["excluded_rows"] == 0


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
    port.data = SimpleNamespace(oscillation_range=1.0)
    port.sg_for_hkls = lambda _hkls, omega: np.asarray([float(omega) ** 2])

    with pytest.raises(LocalSgCrossingError, match="local Ewald branch"):
        port.solve_sg_profile_target(
            3,
            (1, 0, 0),
            0.0,
            -1.0,
            {"sg_jitter_mode": "gaussian", "sg_jitter_sigma_inv_A": 0.0},
        )


def test_base_profile_exact_target_continues_on_same_branch_for_small_cell_geometry():
    real_matrix = np.stack(
        (
            np.asarray([-3.968, -0.477, 3.367]),
            np.asarray([-4.940, 7.859, -4.667]),
            np.asarray([-9.948, -13.704, -13.384]),
        ),
        axis=1,
    )
    raw_a = np.linalg.inv(real_matrix).T
    raw_hkl = np.asarray([-1, -6, 5], dtype=int)
    axis = np.asarray([0.429142, -0.903207, 0.007286], dtype=float)
    axis /= np.linalg.norm(axis)
    axis_cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    beam = np.asarray([-0.000042, 0.000044, 1.0], dtype=float)
    k = 1.0 / 0.025079
    s0 = k * beam / np.linalg.norm(beam)
    target_sg = -0.0014769
    expected_root = 15.9482911053

    for transform_spec in ("identity", "-h,-k,l", "-h,k,-l", "h,-k,-l"):
        transform, _canonical = parse_hkl_transform(transform_spec)
        hkl = transform @ raw_hkl
        a_matrix = raw_a @ np.linalg.inv(transform)
        data = SimpleNamespace(
            oscillation_range=0.88,
            k=k,
            s0=s0,
            axis_eye=np.eye(3),
            axis_cross=axis_cross,
            axis_cross2=axis_cross @ axis_cross,
            setting_rotation=np.eye(3),
            fixed_rotation=np.eye(3),
            A_for_omega=lambda _omega, matrix=a_matrix: matrix,
        )
        port = object.__new__(XDSBlochPort)
        port.data = data

        with pytest.raises(LocalSgCrossingError):
            port.solve_omega_for_sg(hkl, 12.76, target_sg)
        jitter_meta = {"sg_jitter_mode": "none", "sg_jitter_sigma_inv_A": 0.0}
        solved = port.solve_sg_profile_target(2370, hkl, 12.76, target_sg, jitter_meta)

        assert solved == pytest.approx(expected_root, abs=1e-9)
        assert float(port.sg_for_hkls(np.asarray([hkl]), solved)[0]) == pytest.approx(target_sg, abs=1e-10)
        assert port.sg_derivative_per_deg(hkl, solved) < 0.0
        assert jitter_meta["sg_profile_same_branch_continuation_used"] is True
        assert jitter_meta["sg_profile_same_branch_continuation_segments"] == 2


def test_base_profile_continuation_rejects_an_interior_branch_reversal():
    port = object.__new__(XDSBlochPort)
    port.data = SimpleNamespace(oscillation_range=1.0)
    port.sg_for_hkls = lambda _hkls, omega: np.asarray(
        [float(omega) + 0.5025641025641026 * np.sin(2.0 * float(omega))]
    )

    with pytest.raises(LocalSgCrossingError, match="reverses or stalls"):
        port.solve_sg_profile_target(
            9,
            (1, 0, 0),
            0.0,
            2.1,
            {"sg_jitter_mode": "none", "sg_jitter_sigma_inv_A": 0.0},
        )


def test_parallel_local_sg_failure_is_not_retried_as_a_context_failure(monkeypatch):
    port = object.__new__(XDSBlochPort)
    port.cfg = SimpleNamespace(compute_backend="scipy")
    port.data = SimpleNamespace(iobs=np.ones(2))
    port.row_plan_cache_mode = lambda: "off"
    serial_calls = []
    port.row_prediction = lambda idx: serial_calls.append(idx)
    failure = LocalSgCrossingError("physical row geometry is unreachable")
    contexts = []

    class FailingPool:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def imap_unordered(self, *_args, **_kwargs):
            raise failure

    class FakeContext:
        def Pool(self, **_kwargs):
            return FailingPool()

    def fake_get_context(name):
        contexts.append(name)
        return FakeContext()

    monkeypatch.setattr(bloch_engine.multiprocessing, "get_context", fake_get_context)

    with pytest.raises(LocalSgCrossingError) as caught:
        port.predict_rows(np.asarray([0, 1]), workers=2, chunksize=1)

    assert caught.value is failure
    assert len(contexts) == 1
    assert serial_calls == []


def test_parallel_retry_discards_partial_attempt_transactionally(monkeypatch, capsys):
    port = object.__new__(XDSBlochPort)
    port.cfg = SimpleNamespace(compute_backend="scipy")
    port.data = SimpleNamespace(iobs=np.ones(2))
    port.row_plan_cache_mode = lambda: "off"
    serial_calls = []
    port.row_prediction = lambda idx: serial_calls.append(idx)
    contexts = []

    class FakePool:
        def __init__(self, attempt):
            self.attempt = attempt

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def imap_unordered(self, _worker, requested, **_kwargs):
            if self.attempt == 0:
                yield 0, fake_prediction_result(0, continuation=True)
                raise RuntimeError("backend failed after yielding one row")
            for idx in reversed(requested):
                yield idx, fake_prediction_result(idx, continuation=idx == 0)

    class FakeContext:
        def __init__(self, attempt):
            self.attempt = attempt

        def Pool(self, **_kwargs):
            return FakePool(self.attempt)

    def fake_get_context(name):
        contexts.append(name)
        return FakeContext(len(contexts) - 1)

    monkeypatch.setattr(bloch_engine.multiprocessing, "get_context", fake_get_context)

    pred = port.predict_rows(np.asarray([0, 1]), workers=2, chunksize=1)

    np.testing.assert_allclose(pred["dyn"], [1.0, 2.0])
    np.testing.assert_array_equal(
        pred["sg_profile_same_branch_continuation_used"],
        [True, False],
    )
    assert len(pred["support_hkls"]) == 2
    assert serial_calls == []
    assert len(contexts) == 2
    output = capsys.readouterr().out
    assert output.count("predicted 2/2 rows") == 1
    assert "predicted 3/2" not in output


def test_cached_parallel_retry_discards_partial_attempt_transactionally(monkeypatch, capsys):
    port = object.__new__(XDSBlochPort)
    indices = np.asarray([10, 11])
    cache = {
        "indices": indices.copy(),
        "unique_diffs": np.zeros((0, 3), dtype=int),
        "status": {"status": "hit"},
    }
    port.load_or_build_row_plan_cache = lambda requested: cache
    port.current_unique_ug = lambda _diffs: np.zeros(0, dtype=complex)
    serial_calls = []
    port.row_prediction_from_cache = lambda *_args: serial_calls.append(_args)
    contexts = []

    class FakePool:
        def __init__(self, attempt):
            self.attempt = attempt

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def imap_unordered(self, _worker, requested_positions, **_kwargs):
            if self.attempt == 0:
                yield 10, fake_prediction_result(10, continuation=True)
                raise RuntimeError("cached backend failed after yielding one row")
            for pos in reversed(requested_positions):
                idx = int(indices[pos])
                yield idx, fake_prediction_result(idx, continuation=idx == 10)

    class FakeContext:
        def __init__(self, attempt):
            self.attempt = attempt

        def Pool(self, **_kwargs):
            return FakePool(self.attempt)

    def fake_get_context(name):
        contexts.append(name)
        return FakeContext(len(contexts) - 1)

    monkeypatch.setattr(bloch_engine.multiprocessing, "get_context", fake_get_context)

    pred = port.predict_rows_with_row_cache(indices, workers=2, chunksize=1)

    np.testing.assert_allclose(pred["dyn"], [11.0, 12.0])
    np.testing.assert_array_equal(
        pred["sg_profile_same_branch_continuation_used"],
        [True, False],
    )
    assert len(pred["support_hkls"]) == 2
    assert serial_calls == []
    assert len(contexts) == 2
    output = capsys.readouterr().out
    assert output.count("cached predicted 2/2 rows") == 1
    assert "cached predicted 3/2" not in output
