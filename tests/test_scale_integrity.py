from types import SimpleNamespace

import numpy as np
import pytest

import grond.cli as cli
from grond.legacy.engine import native_row_regularize as native
from grond.legacy.engine.native_row_regularize import (
    automatic_frame_offset_grid,
    fit_group_scales,
    kernel_variance_terms,
    native_metrics,
    nuisance_reference_matches,
    rank_trials,
    resolve_frame_offset_grid,
    select_trial_with_promotion_gate,
    xds_frame_datum_summary,
)


def test_automatic_frame_grid_tracks_xds_datum_with_symmetric_residual_probes():
    assert automatic_frame_offset_grid(1.3447) == [1.0, 1.4, 1.8]
    assert automatic_frame_offset_grid(0.0) == [-0.4, 0.0, 0.4]
    assert automatic_frame_offset_grid(float("nan")) == [0.0, 0.2]


def test_xds_frame_datum_requires_many_consistent_geometry_roots(monkeypatch):
    data = SimpleNamespace(
        zcal=np.arange(20, dtype=float),
        hkl=np.zeros((20, 3), dtype=int),
        oscillation_range=-0.2,
        omega_from_z=lambda z: float(z),
    )
    monkeypatch.setattr(native, "_xds_header_bragg_omega", lambda data, hkl, omega: omega - 0.25)

    reliable = xds_frame_datum_summary(data, np.arange(20))
    underdetermined = xds_frame_datum_summary(data, np.arange(1))
    fallback_grid, fallback = resolve_frame_offset_grid("auto", data, np.arange(1))

    assert reliable["status"] == "ok"
    assert reliable["median_offset_frames"] == 1.25
    assert underdetermined["status"] == "unreliable"
    assert fallback["status"] == "unreliable"
    assert fallback_grid == [0.0, 0.2]


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


def test_native_sigma_terms_map_to_frozen_bloch_config_names():
    assert kernel_variance_terms(
        {"sigma_a": 0.1, "sigma_b_corr": 0.2, "sigma_c_peak": 0.3}
    ) == {
        "variance_a": 0.1,
        "variance_b_corr": 0.2,
        "variance_c_peak": 0.3,
    }


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


def _thickness_trial(
    label: str,
    thickness: float,
    *,
    work_wr2: float,
    free_wr2: float,
    all_wr2: float,
    all_r1: float,
    all_n: int = 100,
    work_n: int = 90,
    free_n: int = 10,
):
    return {
        "label": label,
        "nuisance": {
            "thickness_A": thickness,
            "thickness_grad_frac": 1.0,
            "thickness_model": "uniform_gradient",
            "dose_b_ref_frame": float("nan"),
        },
        "valid_rows": all_n,
        "scale_count": 8,
        "scale_fit_split": "work",
        "arrays": {"valid": np.ones(all_n, dtype=bool)},
        "metrics": {
            "all": {
                "dyn": {"n": all_n, "wR2_all": all_wr2, "R1_all": all_r1, "CC": 0.9},
                "born": {"n": all_n, "R1_gt": 0.2, "CC": 0.9},
            },
            "work": {"dyn": {"n": work_n, "wR2_all": work_wr2, "R1_all": all_r1}},
            "free": {"dyn": {"n": free_n, "wR2_all": free_wr2, "R1_all": all_r1}},
        },
    }


def test_nuisance_reference_matching_accepts_only_two_sided_nan_and_roundtrip_values():
    nuisance = {
        "thickness_A": 1204.5805095106,
        "thickness_model": "uniform_gradient",
        "mosaicity_mode": "none",
        "geometry_path_file": "",
        "dose_b_ref_frame": float("nan"),
    }

    assert nuisance_reference_matches(dict(nuisance), nuisance) is True
    assert nuisance_reference_matches({**nuisance, "dose_b_ref_frame": 0.0}, nuisance) is False
    assert nuisance_reference_matches({**nuisance, "thickness_A": 1204.6}, nuisance) is False
    assert nuisance_reference_matches(
        {**nuisance, "thickness_A": float("nan")},
        {**nuisance, "thickness_A": float("nan")},
    ) is False
    assert nuisance_reference_matches(
        {**nuisance, "mosaicity_mode": "gaussian_omega"}, nuisance
    ) is False
    assert nuisance_reference_matches(
        {**nuisance, "geometry_path_file": "geometry.json"}, nuisance
    ) is False


def test_pinned_grid_serialization_preserves_exact_baseline_matching():
    nuisance = {
        "thickness_A": 1204.5805095106,
        "group_width_deg": 0.1234567890123,
        "dose_b_ref_frame": float("nan"),
    }
    serialized = {
        key: float(cli.grid_from_nuisance(nuisance, key, "nan"))
        for key in ("thickness_A", "group_width_deg", "dose_b_ref_frame")
    }

    assert nuisance_reference_matches(nuisance, serialized) is True


def test_in_memory_promotion_score_matches_public_summary_score():
    trial = _thickness_trial(
        "trial",
        1200.0,
        work_wr2=0.20,
        free_wr2=0.21,
        all_wr2=0.202,
        all_r1=0.15,
    )

    assert native._trial_full_row_score(trial) == pytest.approx(
        cli.score_native_summary({"best_native": {"metrics": trial["metrics"]}})["score"]
    )


def test_discrete_thickness_gate_restores_802003_baseline_on_free_row_regression():
    baseline = _thickness_trial(
        "baseline",
        1000.0,
        work_wr2=0.3246139415,
        free_wr2=0.3490050525,
        all_wr2=0.327870,
        all_r1=0.1797603382,
        all_n=2529,
        work_n=2268,
        free_n=261,
    )
    candidate = _thickness_trial(
        "candidate",
        820.0,
        work_wr2=0.3241874073,
        free_wr2=0.3581405789,
        all_wr2=0.3287695,
        all_r1=0.1794613916,
        all_n=2529,
        work_n=2268,
        free_n=261,
    )

    selected, decision = select_trial_with_promotion_gate(
        rank_trials([baseline, candidate], "work_wR2"),
        baseline["nuisance"],
        selection_metric="work_wR2",
        min_gain=5e-5,
        max_free_wr2_loss=0.002,
        max_all_r1_loss=0.001,
    )

    assert selected is baseline
    assert decision["status"] == "rejected"
    assert decision["objective_gain"] > 5e-5
    assert decision["free_wR2_loss"] > 0.009
    assert "free_wR2_loss" in decision["promotion_reasons"]
    assert "full_row_score_gain_below_minimum" in decision["promotion_reasons"]


def test_discrete_thickness_gate_accepts_one_generalizing_work_winner():
    baseline = _thickness_trial(
        "baseline",
        1200.0,
        work_wr2=0.20,
        free_wr2=0.21,
        all_wr2=0.202,
        all_r1=0.15,
    )
    candidate = _thickness_trial(
        "candidate",
        1100.0,
        work_wr2=0.1998,
        free_wr2=0.211,
        all_wr2=0.201,
        all_r1=0.1505,
    )

    selected, decision = select_trial_with_promotion_gate(
        rank_trials([baseline, candidate], "work_wR2"),
        baseline["nuisance"],
        selection_metric="work_wR2",
        min_gain=5e-5,
        max_free_wr2_loss=0.002,
        max_all_r1_loss=0.001,
    )

    assert selected is candidate
    assert decision["status"] == "accepted"
    assert decision["promoted"] is True
    assert decision["promotion_reasons"] == []


def test_discrete_thickness_gate_rejects_noncomparable_or_missing_free_rows():
    baseline = _thickness_trial(
        "baseline",
        1200.0,
        work_wr2=0.20,
        free_wr2=0.21,
        all_wr2=0.202,
        all_r1=0.15,
        free_n=10,
    )
    candidate = _thickness_trial(
        "candidate",
        1100.0,
        work_wr2=0.19,
        free_wr2=float("nan"),
        all_wr2=0.19,
        all_r1=0.14,
        free_n=0,
    )

    selected, decision = select_trial_with_promotion_gate(
        rank_trials([baseline, candidate], "work_wR2"),
        baseline["nuisance"],
        selection_metric="work_wR2",
        min_gain=5e-5,
        max_free_wr2_loss=0.002,
        max_all_r1_loss=0.001,
    )

    assert selected is baseline
    assert "candidate_missing_positive_free_rows" in decision["promotion_reasons"]
    assert "free_row_count_mismatch" in decision["promotion_reasons"]
    assert "missing_finite_promotion_metrics" in decision["promotion_reasons"]


@pytest.mark.parametrize(
    ("candidate_work", "candidate_r1", "expected_reason"),
    [
        (0.19999, 0.1500, "gain_below_minimum"),
        (0.19000, 0.1511, "all_R1_loss"),
    ],
)
def test_discrete_thickness_gate_enforces_gain_and_all_r1_boundaries(
    candidate_work: float,
    candidate_r1: float,
    expected_reason: str,
):
    baseline = _thickness_trial(
        "baseline",
        1200.0,
        work_wr2=0.20,
        free_wr2=0.21,
        all_wr2=0.202,
        all_r1=0.15,
    )
    candidate = _thickness_trial(
        "candidate",
        1100.0,
        work_wr2=candidate_work,
        free_wr2=0.21,
        all_wr2=0.20,
        all_r1=candidate_r1,
    )

    selected, decision = select_trial_with_promotion_gate(
        rank_trials([baseline, candidate], "work_wR2"),
        baseline["nuisance"],
        selection_metric="work_wR2",
        min_gain=5e-5,
        max_free_wr2_loss=0.002,
        max_all_r1_loss=0.001,
    )

    assert selected is baseline
    assert expected_reason in decision["promotion_reasons"]


def test_discrete_thickness_gate_rejects_worse_full_row_composite_score():
    baseline = _thickness_trial(
        "baseline",
        1200.0,
        work_wr2=0.2000,
        free_wr2=0.2000,
        all_wr2=0.2000,
        all_r1=0.1500,
    )
    candidate = _thickness_trial(
        "candidate",
        1100.0,
        work_wr2=0.1999,
        free_wr2=0.2010,
        all_wr2=0.2002,
        all_r1=0.1500,
    )

    selected, decision = select_trial_with_promotion_gate(
        rank_trials([baseline, candidate], "work_wR2"),
        baseline["nuisance"],
        selection_metric="work_wR2",
        min_gain=5e-5,
        max_free_wr2_loss=0.002,
        max_all_r1_loss=0.001,
    )

    assert selected is baseline
    assert decision["full_row_score_gain"] < 0.0
    assert "full_row_score_gain_below_minimum" in decision["promotion_reasons"]


def test_discrete_thickness_gate_requires_identical_valid_row_identity():
    baseline = _thickness_trial(
        "baseline",
        1200.0,
        work_wr2=0.20,
        free_wr2=0.21,
        all_wr2=0.202,
        all_r1=0.15,
    )
    candidate = _thickness_trial(
        "candidate",
        1100.0,
        work_wr2=0.19,
        free_wr2=0.20,
        all_wr2=0.19,
        all_r1=0.14,
    )
    baseline["arrays"]["valid"] = np.asarray([False, *([True] * 100), False])
    candidate["arrays"]["valid"] = np.asarray([True, *([True] * 99), False, False])

    selected, decision = select_trial_with_promotion_gate(
        rank_trials([baseline, candidate], "work_wR2"),
        baseline["nuisance"],
        selection_metric="work_wR2",
        min_gain=5e-5,
        max_free_wr2_loss=0.002,
        max_all_r1_loss=0.001,
    )

    assert selected is baseline
    assert decision["valid_row_mismatch_count"] == 2
    assert "valid_row_identity_mismatch" in decision["promotion_reasons"]


@pytest.mark.parametrize(
    ("selection_metric", "scale_fit_split", "message"),
    [
        ("free_wR2", "work", "requires work_wR2 ranking"),
        ("work_wR2", "all", "requires work-only scale fitting"),
    ],
)
def test_discrete_thickness_gate_requires_independent_work_contract(
    selection_metric: str,
    scale_fit_split: str,
    message: str,
):
    baseline = _thickness_trial(
        "baseline",
        1200.0,
        work_wr2=0.20,
        free_wr2=0.21,
        all_wr2=0.202,
        all_r1=0.15,
    )
    candidate = _thickness_trial(
        "candidate",
        1100.0,
        work_wr2=0.19,
        free_wr2=0.20,
        all_wr2=0.19,
        all_r1=0.14,
    )
    baseline["scale_fit_split"] = scale_fit_split
    candidate["scale_fit_split"] = scale_fit_split

    with pytest.raises(RuntimeError, match=message):
        select_trial_with_promotion_gate(
            rank_trials([baseline, candidate], selection_metric),
            baseline["nuisance"],
            selection_metric=selection_metric,
            min_gain=5e-5,
            max_free_wr2_loss=0.002,
            max_all_r1_loss=0.001,
        )


def test_discrete_thickness_gate_stops_when_exact_baseline_trial_is_missing():
    candidate = _thickness_trial(
        "candidate",
        1100.0,
        work_wr2=0.19,
        free_wr2=0.20,
        all_wr2=0.20,
        all_r1=0.14,
    )

    with pytest.raises(RuntimeError, match="exactly one starting-nuisance trial"):
        select_trial_with_promotion_gate(
            [candidate],
            {"thickness_A": 1200.0},
            selection_metric="work_wR2",
            min_gain=5e-5,
            max_free_wr2_loss=0.002,
            max_all_r1_loss=0.001,
        )
